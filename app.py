from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import smtplib
import sys
import time
import xml.etree.ElementTree as ET
import uuid
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from email.message import EmailMessage
from html import escape
from typing import Callable, TypeVar
from zoneinfo import ZoneInfo

import requests

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_ANNOUNCEMENT_TZ = "America/New_York"
ARXIV_ANNOUNCEMENT_HOUR = 20
ARXIV_SUBMISSION_CUTOFF_HOUR = 14
ARXIV_PAGE_SIZE = 100
ARXIV_MIN_REQUEST_GAP_SECONDS = 3.2
ARXIV_MAX_RETRY_ATTEMPTS = 6
ARXIV_RETRY_BASE_DELAY_SECONDS = 3.0
ARXIV_RETRY_MAX_DELAY_SECONDS = 60.0
ARXIV_REQUEST_HEADERS = {
    "User-Agent": "arxiv-recommend/1.0",
    "Accept": "application/atom+xml",
}
_last_arxiv_request_monotonic = 0.0


@dataclass
class Paper:
    paper_id: str
    title: str
    summary: str
    authors: list[str]
    categories: list[str]
    published: dt.datetime
    updated: dt.datetime
    link: str


@dataclass
class Recommendation:
    paper: Paper
    title_zh: str
    abstract_zh: str
    score: float
    matched_interests: list[str]
    summary: str
    reason: str


def int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def str_env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def debug_log(enabled: bool, message: str) -> None:
    if not enabled:
        return
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[dbg {timestamp}] {message}", flush=True)


T = TypeVar("T")


def run_with_retries(
    fn: Callable[[], T],
    *,
    retries: int = 5,
    sleep_seconds: float = 10.0,
    dbg: bool,
    action_name: str,
) -> T:
    max_attempts = max(1, retries + 1)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_attempts:
                break
            debug_log(
                dbg,
                (
                    f"{action_name} failed on attempt {attempt}/{max_attempts} with {exc}. "
                    f"Retrying after {sleep_seconds:.1f}s."
                ),
            )
            time.sleep(max(0.0, sleep_seconds))

    if last_error is None:
        raise RuntimeError(f"{action_name} failed without an exception.")
    raise last_error


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    parsed: dict[str, str] = {}
    with open(path, encoding="utf-8") as env_file:
        for line in env_file:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                parsed[key] = value
    for key, value in parsed.items():
        if not str_env(key):
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch new arXiv papers, recommend by research profile, summarize, and email/save results."
    )
    parser.add_argument(
        "--research-profile",
        help="Paragraph describing your research background, topics, and goals.",
        default=str_env("RESEARCH_PROFILE", ""),
    )
    parser.add_argument(
        "--start",
        help="Flexible start time (e.g. 2026.03.04, March 4 2026, last Friday 9am). If both --start/--end are omitted, defaults to the latest fully announced arXiv submission window.",
    )
    parser.add_argument(
        "--end",
        help="Flexible end time (e.g. now, today 23:59, 2026/03/08 18:30). If both --start/--end are omitted, defaults to the latest fully announced arXiv submission window.",
    )
    parser.add_argument(
        "--timezone",
        help="Timezone name used for parsing explicit relative times (e.g. UTC, America/New_York).",
        default=str_env("APP_TIMEZONE", "UTC"),
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=2000,
        help="Maximum number of arXiv papers to fetch from the date window.",
    )
    parser.add_argument(
        "--to",
        help="Recipient email. Defaults to EMAIL_TO from environment. If empty, report is saved to file.",
        default=str_env("EMAIL_TO", ""),
    )
    parser.add_argument(
        "--output",
        help="Output file path for saved recommendations report.",
        default=str_env("OUTPUT_PATH", ""),
    )
    parser.add_argument(
        "--llm-model",
        help="LLM model used for recommendation and summarization.",
        default=str_env("OPENAI_MODEL", "gpt-4o-mini"),
    )
    parser.add_argument(
        "--time-parse-model",
        help="LLM model used to normalize flexible time expressions.",
        default=str_env("TIME_PARSE_MODEL", str_env("OPENAI_MODEL", "gpt-4o-mini")),
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=int_env("LLM_BATCH_SIZE", 40),
        help="Number of papers per LLM batch when evaluating all fetched papers.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=int_env("LLM_TIMEOUT", 60),
        help="Timeout (seconds) for LLM API requests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do everything except SMTP send. If --to is set, print email MIME; otherwise save report file.",
    )
    parser.add_argument(
        "--dbg",
        action="store_true",
        help="Print progress logs for debugging long-running stages.",
    )
    return parser.parse_args()


def normalize_research_profile(profile_raw: str) -> str:
    return " ".join(profile_raw.split())


def parse_user_datetime(
    raw: str | None,
    tz: ZoneInfo,
    now_local: dt.datetime,
    for_end: bool = False,
) -> dt.datetime | None:
    if raw is None:
        return None

    normalized = re.sub(r"\s+", " ", raw.strip())
    if not normalized:
        return None

    low = normalized.lower()
    if low in {"now", "right now"}:
        return now_local
    if low == "today":
        return now_local.replace(hour=23, minute=59, second=59, microsecond=0) if for_end else now_local.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    if low == "yesterday":
        base = now_local - dt.timedelta(days=1)
        return base.replace(hour=23, minute=59, second=59, microsecond=0) if for_end else base.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    if low == "tomorrow":
        base = now_local + dt.timedelta(days=1)
        return base.replace(hour=23, minute=59, second=59, microsecond=0) if for_end else base.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    cleaned = re.sub(
        r"^(from|since|start|starting|begin|beginning|to|until|till|end|ending)\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned.replace(",", " ").replace(" at ", " ")).strip()

    iso_candidate = cleaned.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(iso_candidate)
    except ValueError:
        parsed = None

    if parsed is not None:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz)
        return parsed.astimezone(tz)

    datetime_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y.%m.%d %H:%M:%S",
        "%Y.%m.%d %H:%M",
        "%Y%m%d%H%M",
        "%Y%m%d %H%M",
        "%d %b %Y %H:%M",
        "%d %B %Y %H:%M",
        "%b %d %Y %H:%M",
        "%B %d %Y %H:%M",
    ]
    for fmt in datetime_formats:
        try:
            parsed = dt.datetime.strptime(cleaned, fmt)
            return parsed.replace(tzinfo=tz)
        except ValueError:
            continue

    date_formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y%m%d",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
    ]
    for fmt in date_formats:
        try:
            parsed_date = dt.datetime.strptime(cleaned, fmt)
            parsed = parsed_date.replace(tzinfo=tz)
            if for_end:
                parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=0)
            return parsed
        except ValueError:
            continue

    return None


def normalize_times_with_llm(
    start_raw: str | None,
    end_raw: str | None,
    tz_name: str,
    reference_now: dt.datetime,
    llm_model: str,
    llm_timeout: int,
    dbg: bool = False,
) -> tuple[str | None, str | None]:
    if not (start_raw or end_raw):
        return None, None

    api_key = str_env("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set for time parsing.")

    base_url = str_env("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    endpoint = f"{base_url}/chat/completions"

    system_prompt = (
        "You normalize user-provided time expressions to strict ISO 8601 datetimes. "
        "Use the provided reference_now as the authoritative current time. "
        "Return JSON only with keys start and end."
    )
    user_payload = {
        "timezone": tz_name,
        "reference_now": reference_now.isoformat(),
        "reference_date": reference_now.strftime("%Y-%m-%d"),
        "reference_weekday": reference_now.strftime("%A"),
        "reference_unix": int(reference_now.timestamp()),
        "reference_utc_offset": reference_now.strftime("%z"),
        "start_input": start_raw,
        "end_input": end_raw,
        "rules": [
            "Output format: YYYY-MM-DDTHH:MM:SS+HH:MM.",
            "If input is date-only and refers to start, set time to 00:00:00.",
            "If input is date-only and refers to end, set time to 23:59:59.",
            "Resolve relative terms (today, yesterday, last Friday, now) using reference_now and timezone.",
            "If a side is missing, return null for that side.",
        ],
    }

    request_payload = {
        "model": llm_model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }

    def _request_and_parse() -> dict:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_payload,
            timeout=max(5, llm_timeout),
        )
        response.raise_for_status()

        payload = response.json()
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Time parser LLM response has no choices.")

        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Time parser LLM response has no text content.")
        return extract_json_object(content)

    debug_log(dbg, f"Normalizing time window via LLM model={llm_model}.")
    parsed = run_with_retries(
        _request_and_parse,
        dbg=dbg,
        action_name="Time parser LLM call",
    )

    start_value = parsed.get("start")
    end_value = parsed.get("end")

    normalized_start = str(start_value).strip() if start_value is not None else None
    normalized_end = str(end_value).strip() if end_value is not None else None

    if normalized_start == "":
        normalized_start = None
    if normalized_end == "":
        normalized_end = None

    debug_log(
        dbg,
        (
            "LLM time parsing completed with "
            f"start={normalized_start or 'null'} end={normalized_end or 'null'}."
        ),
    )
    return normalized_start, normalized_end


def compute_latest_announced_arxiv_window(
    now_utc: dt.datetime | None = None,
) -> tuple[dt.datetime, dt.datetime]:
    if now_utc is None:
        now_utc = dt.datetime.now(dt.timezone.utc)

    announcement_tz = ZoneInfo(ARXIV_ANNOUNCEMENT_TZ)
    now_local = now_utc.astimezone(announcement_tz)
    latest_announcement: dt.datetime | None = None

    for days_back in range(0, 8):
        candidate_date = now_local.date() - dt.timedelta(days=days_back)
        candidate = dt.datetime.combine(
            candidate_date,
            dt.time(hour=ARXIV_ANNOUNCEMENT_HOUR),
            tzinfo=announcement_tz,
        )
        if candidate > now_local:
            continue
        if candidate.weekday() not in {6, 0, 1, 2, 3}:
            continue
        latest_announcement = candidate
        break

    if latest_announcement is None:
        raise RuntimeError("Could not determine the latest arXiv announcement time.")

    if latest_announcement.weekday() == 6:
        start_date = latest_announcement.date() - dt.timedelta(days=3)
        end_date = latest_announcement.date() - dt.timedelta(days=2)
    elif latest_announcement.weekday() == 0:
        start_date = latest_announcement.date() - dt.timedelta(days=3)
        end_date = latest_announcement.date()
    else:
        start_date = latest_announcement.date() - dt.timedelta(days=1)
        end_date = latest_announcement.date()

    start_local = dt.datetime.combine(
        start_date,
        dt.time(hour=ARXIV_SUBMISSION_CUTOFF_HOUR),
        tzinfo=announcement_tz,
    )
    end_local = dt.datetime.combine(
        end_date,
        dt.time(hour=ARXIV_SUBMISSION_CUTOFF_HOUR),
        tzinfo=announcement_tz,
    )
    return start_local.astimezone(dt.timezone.utc), end_local.astimezone(dt.timezone.utc)


def compute_time_window(
    start_raw: str | None,
    end_raw: str | None,
    tz_name: str,
    time_parse_model: str,
    llm_timeout: int,
    dbg: bool = False,
) -> tuple[dt.datetime, dt.datetime]:
    tz = ZoneInfo(tz_name)
    now_local = dt.datetime.now(tz)

    llm_start_raw: str | None = None
    llm_end_raw: str | None = None
    if (start_raw and start_raw.strip()) or (end_raw and end_raw.strip()):
        try:
            llm_start_raw, llm_end_raw = normalize_times_with_llm(
                start_raw=start_raw,
                end_raw=end_raw,
                tz_name=tz_name,
                reference_now=now_local,
                llm_model=time_parse_model,
                llm_timeout=llm_timeout,
                dbg=dbg,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"LLM time parsing failed: {exc}. Falling back to local parsing.", file=sys.stderr)

    start_source = llm_start_raw if llm_start_raw is not None else start_raw
    end_source = llm_end_raw if llm_end_raw is not None else end_raw

    start_local = parse_user_datetime(start_source, tz, now_local, for_end=False)
    end_local = parse_user_datetime(end_source, tz, now_local, for_end=True)

    if start_local is None and end_local is None:
        start_utc, end_utc = compute_latest_announced_arxiv_window()
        debug_log(
            dbg,
            (
                "No explicit window provided; using the latest fully announced arXiv window "
                f"{start_utc.isoformat()} -> {end_utc.isoformat()}."
            ),
        )
        return start_utc, end_utc
    else:
        if end_local is None:
            end_local = now_local
        if start_local is None:
            start_local = end_local - dt.timedelta(hours=24)

    if end_local <= start_local:
        raise ValueError("End time must be after start time.")

    debug_log(
        dbg,
        (
            "Resolved query window to "
            f"{start_local.astimezone(dt.timezone.utc).isoformat()} -> "
            f"{end_local.astimezone(dt.timezone.utc).isoformat()}."
        ),
    )
    return start_local.astimezone(dt.timezone.utc), end_local.astimezone(dt.timezone.utc)

def datetime_to_arxiv(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).strftime("%Y%m%d%H%M")


def _sleep_to_respect_arxiv_pacing(dbg: bool, min_gap_seconds: float = ARXIV_MIN_REQUEST_GAP_SECONDS) -> None:
    global _last_arxiv_request_monotonic
    if _last_arxiv_request_monotonic <= 0:
        return

    elapsed = time.monotonic() - _last_arxiv_request_monotonic
    if elapsed >= min_gap_seconds:
        return

    delay = min_gap_seconds - elapsed
    debug_log(dbg, f"Sleeping {delay:.1f}s to respect arXiv API pacing.")
    time.sleep(delay)


def _mark_arxiv_request_time() -> None:
    global _last_arxiv_request_monotonic
    _last_arxiv_request_monotonic = time.monotonic()


def _parse_retry_after_seconds(retry_after: str | None) -> float | None:
    if retry_after is None:
        return None

    value = retry_after.strip()
    if not value:
        return None

    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    seconds = (parsed - dt.datetime.now(dt.timezone.utc)).total_seconds()
    return max(0.0, seconds)


def _compute_arxiv_backoff_delay(attempt: int) -> float:
    base_delay = min(
        ARXIV_RETRY_MAX_DELAY_SECONDS,
        ARXIV_RETRY_BASE_DELAY_SECONDS * (2 ** attempt),
    )
    return base_delay + random.uniform(0.0, 1.0)


def _compute_arxiv_retry_delay(response: requests.Response, attempt: int) -> float:
    retry_after_seconds = _parse_retry_after_seconds(response.headers.get("Retry-After"))
    if retry_after_seconds is not None:
        return max(ARXIV_RETRY_BASE_DELAY_SECONDS, retry_after_seconds)
    return _compute_arxiv_backoff_delay(attempt)


def fetch_arxiv_batch(
    params: dict[str, str | int],
    dbg: bool = False,
    max_attempts: int = ARXIV_MAX_RETRY_ATTEMPTS,
) -> str:
    last_error: Exception | None = None

    for attempt in range(max(1, max_attempts)):
        _sleep_to_respect_arxiv_pacing(dbg)
        try:
            response = requests.get(
                ARXIV_API_URL,
                params=params,
                headers=ARXIV_REQUEST_HEADERS,
                timeout=120,
            )
        except requests.RequestException as exc:
            _mark_arxiv_request_time()
            last_error = exc
            if attempt >= max_attempts - 1:
                break
            delay = _compute_arxiv_backoff_delay(attempt)
            debug_log(
                dbg,
                (
                    "arXiv request failed with "
                    f"{exc.__class__.__name__}: {exc}. Retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{max_attempts})."
                ),
            )
            time.sleep(delay)
            continue

        _mark_arxiv_request_time()

        if response.status_code == 429:
            last_error = requests.HTTPError(
                f"arXiv returned HTTP 429 for params={params}",
                response=response,
            )
            if attempt >= max_attempts - 1:
                break
            delay = _compute_arxiv_retry_delay(response, attempt)
            debug_log(
                dbg,
                (
                    "arXiv returned HTTP 429. "
                    f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_attempts})."
                ),
            )
            time.sleep(delay)
            continue

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            last_error = exc
            if response.status_code >= 500 and attempt < max_attempts - 1:
                delay = _compute_arxiv_backoff_delay(attempt)
                debug_log(
                    dbg,
                    (
                        f"arXiv returned HTTP {response.status_code}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_attempts})."
                    ),
                )
                time.sleep(delay)
                continue
            raise

        return response.text

    if last_error is None:
        raise RuntimeError("arXiv batch fetch failed without an exception.")
    raise last_error


def fetch_arxiv_papers(
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    max_results: int,
    dbg: bool = False,
) -> list[Paper]:
    if max_results <= 0:
        return []

    search_query = f"submittedDate:[{datetime_to_arxiv(start_utc)} TO {datetime_to_arxiv(end_utc)}]"
    collected: list[Paper] = []
    start_idx = 0
    page_size = ARXIV_PAGE_SIZE

    while len(collected) < max_results:
        batch_size = min(page_size, max_results - len(collected))
        debug_log(
            dbg,
            (
                "Fetching arXiv batch "
                f"start={start_idx} size={batch_size} for window "
                f"{start_utc.isoformat()} -> {end_utc.isoformat()}."
            ),
        )
        params = {
            "search_query": search_query,
            "start": start_idx,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        papers = parse_arxiv_feed(fetch_arxiv_batch(params, dbg=dbg))
        if not papers:
            break
        collected.extend(papers)
        if len(papers) < batch_size:
            break
        start_idx += batch_size

    debug_log(dbg, f"Fetched {len(collected)} arXiv paper(s).")
    return collected

def parse_arxiv_feed(feed_xml: str) -> list[Paper]:
    root = ET.fromstring(feed_xml)
    entries = root.findall("atom:entry", ATOM_NS)
    papers: list[Paper] = []
    for entry in entries:
        paper_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
        title = " ".join(entry.findtext("atom:title", default="", namespaces=ATOM_NS).split())
        summary = " ".join(entry.findtext("atom:summary", default="", namespaces=ATOM_NS).split())
        authors = [
            author.findtext("atom:name", default="", namespaces=ATOM_NS).strip()
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        categories = [
            category.attrib.get("term", "").strip()
            for category in entry.findall("atom:category", ATOM_NS)
            if category.attrib.get("term", "").strip()
        ]
        published_raw = entry.findtext("atom:published", default="", namespaces=ATOM_NS).strip()
        updated_raw = entry.findtext("atom:updated", default="", namespaces=ATOM_NS).strip()
        published = dt.datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
        updated = dt.datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
        link = extract_html_link(entry) or paper_id

        papers.append(
            Paper(
                paper_id=paper_id,
                title=title,
                summary=summary,
                authors=authors,
                categories=categories,
                published=published,
                updated=updated,
                link=link,
            )
        )
    return papers


def extract_html_link(entry: ET.Element) -> str:
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("type") == "text/html":
            return link.attrib.get("href", "").strip()
    return ""


def summarize_abstract(text: str, max_sentences: int = 2, max_chars: int = 380) -> str:
    clean = " ".join(text.split())
    if not clean:
        return "No abstract available."
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    candidate = " ".join(sentences[:max_sentences]).strip()
    if not candidate:
        candidate = clean
    if len(candidate) <= max_chars:
        return candidate
    return candidate[: max_chars - 3].rstrip() + "..."


def compact_text(text: str, max_chars: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def normalize_str_list(value: object) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                output.append(text)
        return output
    return []


def extract_json_object(raw: str) -> dict:
    stripped = raw.strip()
    stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise ValueError("LLM response does not contain a JSON object.")
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("LLM response root must be a JSON object.")
    return parsed

def recommend_with_llm_batch(
    papers: list[Paper],
    research_profile: str,
    llm_model: str,
    llm_timeout: int,
    dbg: bool = False,
    batch_label: str = "",
) -> list[Recommendation]:
    if not papers:
        return []

    api_key = str_env("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    base_url = str_env("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    endpoint = f"{base_url}/chat/completions"

    paper_map = {paper.paper_id: paper for paper in papers}
    serialized_candidates = [
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "categories": paper.categories,
            "published": paper.published.isoformat(),
            "link": paper.link,
            "abstract": compact_text(paper.summary, 1100),
        }
        for paper in papers
    ]

    system_prompt = (
        "You evaluate arXiv papers against a researcher's self-introduction and summarize relevant papers. "
        "Return strict JSON only. No markdown."
    )
    user_payload = {
        "task": "Return all relevant papers from this batch and summarize each in 2 concise sentences.",
        "research_profile": research_profile,
        "papers": serialized_candidates,
        "output_schema": {
            "recommended": [
                {
                    "paper_id": "string, must exactly match one input paper_id",
                    "score": "number between 0 and 100",
                    "matched_interests": ["matched concepts from research profile"],
                    "reason": "explicit connection between this paper and the research profile",
                    "summary": "2-sentence summary in English",
                    "title_zh": "Chinese title (Simplified Chinese)",
                    "abstract_zh": "Chinese abstract summary in 2-3 sentences (Simplified Chinese)",
                }
            ]
        },
        "constraints": [
            "Only use paper_ids from input.",
            "Return every relevant paper in this batch, not just top results.",
            "Exclude papers that are not relevant to the research profile.",
            "The reason must explicitly mention how the paper connects to the research profile.",
            "title_zh and abstract_zh must be written in Simplified Chinese.",
        ],
    }

    request_payload = {
        "model": llm_model,
        "temperature": 1.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }

    label = f" {batch_label}" if batch_label else ""
    debug_log(dbg, f"Sending recommendation request{label} with {len(papers)} paper(s) to model={llm_model}.")
    def _request_and_parse() -> list[object]:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_payload,
            timeout=max(5, llm_timeout),
        )
        response.raise_for_status()

        payload = response.json()
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM API response has no choices.")

        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM API response has no text content.")

        parsed = extract_json_object(content)
        raw_recommended = parsed.get("recommended", [])
        if not isinstance(raw_recommended, list):
            raise ValueError("LLM response JSON missing 'recommended' list.")
        return raw_recommended

    raw_recommended = run_with_retries(
        _request_and_parse,
        dbg=dbg,
        action_name=f"Recommendation LLM request{label}",
    )

    recommendations: list[Recommendation] = []
    seen_ids: set[str] = set()

    for item in raw_recommended:
        if not isinstance(item, dict):
            continue

        paper_id = str(item.get("paper_id", "")).strip()
        if not paper_id or paper_id in seen_ids or paper_id not in paper_map:
            continue

        paper = paper_map[paper_id]
        seen_ids.add(paper_id)

        raw_score = item.get("score", 0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0

        matched_interests = normalize_str_list(item.get("matched_interests", []))

        summary = compact_text(str(item.get("summary", "")).strip(), 420)
        if not summary:
            summary = summarize_abstract(paper.summary)

        reason = compact_text(str(item.get("reason", "")).strip(), 280)
        if not reason:
            hint = ", ".join(matched_interests[:3]) if matched_interests else "profile overlap"
            reason = f"Connected to your research profile through {hint}."
        title_zh = compact_text(str(item.get("title_zh", "")).strip(), 280)
        if not title_zh:
            title_zh = f"\uff08\u672a\u63d0\u4f9b\u4e2d\u6587\u6807\u9898\uff09{paper.title}"
        abstract_zh = compact_text(str(item.get("abstract_zh", "")).strip(), 520)
        if not abstract_zh:
            abstract_zh = "\uff08\u672a\u63d0\u4f9b\u4e2d\u6587\u6458\u8981\uff09"
        recommendations.append(
            Recommendation(
                paper=paper,
                title_zh=title_zh,
                abstract_zh=abstract_zh,
                score=score,
                matched_interests=matched_interests,
                summary=summary,
                reason=reason,
            )
        )

    debug_log(dbg, f"Recommendation request{label} returned {len(recommendations)} recommendation(s).")
    return recommendations

def recommend_with_llm(
    papers: list[Paper],
    research_profile: str,
    llm_model: str,
    llm_batch_size: int,
    llm_timeout: int,
    dbg: bool = False,
) -> list[Recommendation]:
    if not papers:
        return []

    ordered = sorted(papers, key=lambda item: item.published, reverse=True)
    batch_size = max(1, llm_batch_size)
    collected: list[Recommendation] = []
    total_batches = (len(ordered) + batch_size - 1) // batch_size

    for offset in range(0, len(ordered), batch_size):
        batch = ordered[offset : offset + batch_size]
        batch_number = (offset // batch_size) + 1
        collected.extend(
            recommend_with_llm_batch(
                papers=batch,
                research_profile=research_profile,
                llm_model=llm_model,
                llm_timeout=llm_timeout,
                dbg=dbg,
                batch_label=f"{batch_number}/{total_batches}",
            )
        )
        time.sleep(10.0)

    deduped: dict[str, Recommendation] = {}
    for recommendation in collected:
        existing = deduped.get(recommendation.paper.paper_id)
        if existing is None or recommendation.score > existing.score:
            deduped[recommendation.paper.paper_id] = recommendation

    output = list(deduped.values())
    output.sort(key=lambda item: (item.score, item.paper.published), reverse=True)
    debug_log(dbg, f"Deduped recommendations down to {len(output)} item(s).")
    return output

def recommend_and_summarize(
    papers: list[Paper],
    research_profile: str,
    llm_model: str,
    llm_batch_size: int,
    llm_timeout: int,
    dbg: bool = False,
) -> tuple[list[Recommendation], str, str | None]:
    if not papers:
        return [], "LLM skipped", "No arXiv papers found in this time window."

    try:
        debug_log(dbg, f"Running recommendation pipeline for {len(papers)} fetched paper(s).")
        recommendations = recommend_with_llm(
            papers=papers,
            research_profile=research_profile,
            llm_model=llm_model,
            llm_batch_size=llm_batch_size,
            llm_timeout=llm_timeout,
            dbg=dbg,
        )
        empty_message = None
        if not recommendations:
            empty_message = "No relevant papers were recommended in this time window."
        return recommendations, "LLM", empty_message
    except Exception as exc:  # noqa: BLE001
        print(f"LLM recommendation failed: {exc}.", file=sys.stderr)
        return [], "LLM unavailable", "Recommendations could not be generated because the LLM request failed."

def format_score(score: float) -> str:
    if score.is_integer():
        return str(int(score))
    return f"{score:.2f}"


def render_reports(
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    research_profile: str,
    recommendations: list[Recommendation],
    method_label: str,
    empty_message: str | None = None,
) -> tuple[str, str, str]:
    profile_text = compact_text(research_profile, 700)

    text_lines = [
        f"arXiv recommendations for {start_utc.isoformat()} to {end_utc.isoformat()}",
        f"Research profile: {profile_text}",
        f"Recommendation method: {method_label}",
        "",
    ]

    html_blocks = [
        "<h2>arXiv Recommendations</h2>",
        f"<p>Window: <code>{escape(start_utc.isoformat())}</code> to <code>{escape(end_utc.isoformat())}</code><br>",
        f"Research profile: {escape(profile_text)}<br>",
        f"Recommendation method: {escape(method_label)}</p>",
        "<ol>",
    ]

    markdown_lines = [
        "# arXiv Recommendations",
        "",
        f"- Window: `{start_utc.isoformat()}` to `{end_utc.isoformat()}`",
        f"- Research profile: {profile_text}",
        f"- Recommendation method: {method_label}",
        "",
    ]

    if not recommendations:
        message = empty_message or "No relevant papers found in this time window."
        text_lines.append(message)
        html_blocks.append(f"<li>{escape(message)}</li>")
        markdown_lines.append(message)

    for idx, recommendation in enumerate(recommendations, start=1):
        paper = recommendation.paper
        authors_text = ", ".join(paper.authors) if paper.authors else "Unknown authors"
        matched_text = ", ".join(recommendation.matched_interests) if recommendation.matched_interests else "None"
        score_text = format_score(recommendation.score)

        text_lines.extend(
            [
                f"{idx}. {paper.title}",
                f"   Chinese title: {recommendation.title_zh}",
                f"   Authors: {authors_text}",
                f"   Published: {paper.published.isoformat()}",
                f"   Relevance score: {score_text}",
                f"   Matched concepts: {matched_text}",
                f"   Why selected: {recommendation.reason}",
                f"   Summary: {recommendation.summary}",
                f"   Chinese abstract: {recommendation.abstract_zh}",
                f"   URL: {paper.link}",
                "",
            ]
        )

        html_blocks.append(
            "<li>"
            f"<p><strong>{escape(paper.title)}</strong><br>"
            f"Chinese title: {escape(recommendation.title_zh)}<br>"
            f"Authors: {escape(authors_text)}<br>"
            f"Published: {escape(paper.published.isoformat())}<br>"
            f"Relevance score: {escape(score_text)}<br>"
            f"Matched concepts: {escape(matched_text)}<br>"
            f"Why selected: {escape(recommendation.reason)}<br>"
            f"Summary: {escape(recommendation.summary)}<br>"
            f"Chinese abstract: {escape(recommendation.abstract_zh)}<br>"
            f"URL: <a href='{escape(paper.link)}'>{escape(paper.link)}</a></p>"
            "</li>"
        )

        markdown_lines.extend(
            [
                f"## {idx}. {paper.title}",
                f"- Chinese title: {recommendation.title_zh}",
                f"- Authors: {authors_text}",
                f"- Published: {paper.published.isoformat()}",
                f"- Relevance score: {score_text}",
                f"- Matched concepts: {matched_text}",
                f"- Why selected: {recommendation.reason}",
                f"- Summary: {recommendation.summary}",
                f"- Chinese abstract: {recommendation.abstract_zh}",
                f"- URL: {paper.link}",
                "",
            ]
        )

    html_blocks.append("</ol>")
    return "\n".join(text_lines), "\n".join(html_blocks), "\n".join(markdown_lines)

def build_email(
    recipient: str,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    text_report: str,
    html_report: str,
) -> EmailMessage:
    msg = EmailMessage()
    sender = str_env("EMAIL_FROM", str_env("SMTP_USER", "noreply@example.com"))
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = (
        f"arXiv recommendations ({start_utc.strftime('%Y-%m-%d')} to {end_utc.strftime('%Y-%m-%d')})"
    )
    msg.set_content(text_report)
    msg.add_alternative(html_report, subtype="html")
    return msg


def send_email(message: EmailMessage, dbg: bool = False) -> None:
    host = str_env("SMTP_HOST")
    if not host:
        raise ValueError("SMTP_HOST is not set.")
    port = int(str_env("SMTP_PORT", "587"))
    use_tls = str_env("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes"}
    username = str_env("SMTP_USER", "")
    password = str_env("SMTP_PASS", "")
    use_implicit_ssl = use_tls and port in {465, 994}
    smtp_cls = smtplib.SMTP_SSL if use_implicit_ssl else smtplib.SMTP

    debug_log(
        dbg,
        (
            f"Connecting to SMTP host={host} port={port} "
            f"tls={use_tls} implicit_ssl={use_implicit_ssl}."
        ),
    )
    with smtp_cls(host, port, timeout=30) as smtp:
        smtp.ehlo()
        if use_tls and not use_implicit_ssl:
            debug_log(dbg, "Starting TLS negotiation.")
            smtp.starttls()
            smtp.ehlo()
        if username:
            debug_log(dbg, "Authenticating with SMTP server.")
            smtp.login(username, password)
        debug_log(dbg, "Sending email message.")
        smtp.send_message(message)
    debug_log(dbg, "SMTP send completed.")


def build_report_filename(
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    generated_utc: dt.datetime,
    uid: str,
) -> str:
    return (
        f"arxiv_recommendations_q{start_utc.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{end_utc.strftime('%Y%m%dT%H%M%SZ')}_"
        f"gen{generated_utc.strftime('%Y%m%dT%H%M%SZ')}_{uid}.md"
    )


def default_output_path(start_utc: dt.datetime, end_utc: dt.datetime) -> str:
    generated_utc = dt.datetime.now(dt.timezone.utc)
    uid = uuid.uuid4().hex[:8]
    filename = build_report_filename(start_utc, end_utc, generated_utc, uid)
    return os.path.join("reports", filename)


def resolve_output_path(raw_path: str, start_utc: dt.datetime, end_utc: dt.datetime) -> str:
    generated_utc = dt.datetime.now(dt.timezone.utc)
    uid = uuid.uuid4().hex[:8]
    auto_filename = build_report_filename(start_utc, end_utc, generated_utc, uid)

    cleaned = raw_path.strip()
    if not cleaned:
        return os.path.join("reports", auto_filename)

    expanded = os.path.expanduser(cleaned)
    basename = os.path.basename(expanded.rstrip("/\\"))
    ext = os.path.splitext(basename)[1]
    is_dir_hint = (
        cleaned.endswith("/")
        or cleaned.endswith("\\")
        or os.path.isdir(expanded)
        or ext == ""
    )

    if is_dir_hint:
        return os.path.join(expanded, auto_filename)

    return expanded


def save_report(path: str, markdown_report: str, start_utc: dt.datetime, end_utc: dt.datetime) -> str:
    target = resolve_output_path(path, start_utc, end_utc)
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)

    base, ext = os.path.splitext(target)
    if not ext:
        ext = ".md"
        target = base + ext

    candidate = target
    counter = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{counter}{ext}"
        counter += 1

    with open(candidate, "w", encoding="utf-8") as output_file:
        output_file.write(markdown_report)

    return candidate

def main() -> int:
    load_dotenv()
    args = parse_args()
    debug_log(args.dbg, "Loaded configuration and parsed CLI arguments.")

    research_profile = normalize_research_profile(args.research_profile)
    if not research_profile:
        print("No research profile provided. Set --research-profile or RESEARCH_PROFILE.", file=sys.stderr)
        return 1

    llm_batch_size = max(1, args.llm_batch_size)
    llm_timeout = max(5, args.llm_timeout)
    debug_log(
        args.dbg,
        (
            "Starting run with "
            f"max_results={args.max_results}, email_mode={bool(args.to)}, dry_run={args.dry_run}."
        ),
    )
    try:
        start_utc, end_utc = compute_time_window(
            start_raw=args.start,
            end_raw=args.end,
            tz_name=args.timezone,
            time_parse_model=args.time_parse_model,
            llm_timeout=llm_timeout,
            dbg=args.dbg,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Invalid time settings: {exc}", file=sys.stderr)
        return 1

    try:
        papers = fetch_arxiv_papers(start_utc, end_utc, args.max_results, dbg=args.dbg)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to fetch arXiv papers: {exc}", file=sys.stderr)
        return 1

    recommendations, method_label, empty_message = recommend_and_summarize(
        papers=papers,
        research_profile=research_profile,
        llm_model=args.llm_model,
        llm_batch_size=llm_batch_size,
        llm_timeout=llm_timeout,
        dbg=args.dbg,
    )
    debug_log(
        args.dbg,
        f"Initial recommendation stage completed with {len(recommendations)} item(s) using method={method_label}.",
    )

    text_report, html_report, markdown_report = render_reports(
        start_utc=start_utc,
        end_utc=end_utc,
        research_profile=research_profile,
        recommendations=recommendations,
        method_label=method_label,
        empty_message=empty_message,
    )
    debug_log(args.dbg, "Rendered text, HTML, and markdown reports.")

    if not args.to:
        output_path = args.output or default_output_path(start_utc, end_utc)
        saved_path = save_report(output_path, markdown_report, start_utc, end_utc)
        debug_log(args.dbg, f"Saved report to {saved_path}.")
        print(f"No email recipient provided. Saved report to {saved_path}.")
        return 0

    debug_log(args.dbg, "Building email message.")
    message = build_email(args.to, start_utc, end_utc, text_report, html_report)

    if args.dry_run:
        debug_log(args.dbg, "Dry-run enabled; printing MIME message instead of sending.")
        print(message)
    else:
        try:
            send_email(message, dbg=args.dbg)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to send email: {exc}", file=sys.stderr)
            return 1
        debug_log(args.dbg, f"Email send completed with {len(recommendations)} recommendation(s).")
        print(f"Sent report to {args.to} with {len(recommendations)} recommendation(s).")

    if args.output:
        saved_path = save_report(args.output, markdown_report, start_utc, end_utc)
        debug_log(args.dbg, f"Saved report to {saved_path}.")
        print(f"Saved report to {saved_path}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
