from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import smtplib
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from email.message import EmailMessage
from html import escape
from typing import Iterable
from zoneinfo import ZoneInfo

import requests

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

COMMON_PROFILE_STOPWORDS = {
    "about",
    "across",
    "agent",
    "agents",
    "also",
    "and",
    "approach",
    "approaches",
    "applications",
    "based",
    "being",
    "between",
    "both",
    "focus",
    "focused",
    "focusing",
    "for",
    "from",
    "into",
    "language",
    "large",
    "model",
    "models",
    "paper",
    "papers",
    "research",
    "study",
    "system",
    "systems",
    "that",
    "their",
    "this",
    "using",
    "with",
    "work",
}


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


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as env_file:
        for line in env_file:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch new arXiv papers, recommend by research profile, summarize, and email/save results."
    )
    parser.add_argument(
        "--research-profile",
        help="Paragraph describing your research background, topics, and goals.",
        default=os.getenv("RESEARCH_PROFILE", ""),
    )
    parser.add_argument(
        "--start",
        help="Flexible start time (e.g. 2026.03.04, March 4 2026, last Friday 9am). Defaults to today's 00:00 in --timezone.",
    )
    parser.add_argument(
        "--end",
        help="Flexible end time (e.g. now, today 23:59, 2026/03/08 18:30). Defaults to now in --timezone.",
    )
    parser.add_argument(
        "--timezone",
        help="Timezone name used for date parsing and defaults (e.g. UTC, America/New_York).",
        default=os.getenv("APP_TIMEZONE", "UTC"),
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=200,
        help="Maximum number of arXiv papers to fetch from the date window.",
    )
    parser.add_argument(
        "--to",
        help="Recipient email. Defaults to EMAIL_TO from environment. If empty, report is saved to file.",
        default=os.getenv("EMAIL_TO", ""),
    )
    parser.add_argument(
        "--output",
        help="Output file path for saved recommendations report.",
        default=os.getenv("OUTPUT_PATH", ""),
    )
    parser.add_argument(
        "--llm-model",
        help="LLM model used for recommendation and summarization.",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
    parser.add_argument(
        "--time-parse-model",
        help="LLM model used to normalize flexible time expressions.",
        default=os.getenv("TIME_PARSE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
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
    return parser.parse_args()


def normalize_research_profile(profile_raw: str) -> str:
    return " ".join(profile_raw.split())


def extract_interest_terms(profile: str) -> list[str]:
    terms: list[str] = []

    for chunk in re.split(r"[.;\n]+", profile):
        phrase = " ".join(chunk.split())
        phrase_tokens = tokenize(phrase)
        if 2 <= len(phrase_tokens) <= 8:
            terms.append(phrase)

    token_counts = Counter(
        token
        for token in tokenize(profile)
        if len(token) >= 4 and token not in COMMON_PROFILE_STOPWORDS
    )
    for token, _ in token_counts.most_common(20):
        terms.append(token)

    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        clean = term.strip()
        key = clean.lower()
        if not clean or key in seen:
            continue
        seen.add(key)
        deduped.append(clean)
    return deduped

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
) -> tuple[str | None, str | None]:
    if not (start_raw or end_raw):
        return None, None

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set for time parsing.")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
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

    parsed = extract_json_object(content)

    start_value = parsed.get("start")
    end_value = parsed.get("end")

    normalized_start = str(start_value).strip() if start_value is not None else None
    normalized_end = str(end_value).strip() if end_value is not None else None

    if normalized_start == "":
        normalized_start = None
    if normalized_end == "":
        normalized_end = None

    return normalized_start, normalized_end


def compute_time_window(
    start_raw: str | None,
    end_raw: str | None,
    tz_name: str,
    time_parse_model: str,
    llm_timeout: int,
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
            )
        except Exception as exc:  # noqa: BLE001
            print(f"LLM time parsing failed: {exc}. Falling back to local parsing.", file=sys.stderr)

    start_source = llm_start_raw if llm_start_raw is not None else start_raw
    end_source = llm_end_raw if llm_end_raw is not None else end_raw

    start_local = parse_user_datetime(start_source, tz, now_local, for_end=False)
    end_local = parse_user_datetime(end_source, tz, now_local, for_end=True)

    if start_local is None:
        start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    if end_local is None:
        end_local = now_local

    if end_local <= start_local:
        raise ValueError("End time must be after start time.")

    return start_local.astimezone(dt.timezone.utc), end_local.astimezone(dt.timezone.utc)

def datetime_to_arxiv(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).strftime("%Y%m%d%H%M")


def fetch_arxiv_papers(
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    max_results: int,
) -> list[Paper]:
    if max_results <= 0:
        return []

    search_query = f"submittedDate:[{datetime_to_arxiv(start_utc)} TO {datetime_to_arxiv(end_utc)}]"
    collected: list[Paper] = []
    start_idx = 0
    page_size = 100

    while len(collected) < max_results:
        batch_size = min(page_size, max_results - len(collected))
        params = {
            "search_query": search_query,
            "start": start_idx,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        response = requests.get(ARXIV_API_URL, params=params, timeout=30)
        response.raise_for_status()
        papers = parse_arxiv_feed(response.text)
        if not papers:
            break
        collected.extend(papers)
        if len(papers) < batch_size:
            break
        start_idx += batch_size

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


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def score_paper(paper: Paper, interests: Iterable[str]) -> tuple[int, list[str]]:
    title = paper.title.lower()
    summary = paper.summary.lower()
    categories = " ".join(paper.categories).lower()
    combined = f"{title} {summary} {categories}"
    score = 0
    matched: list[str] = []

    for raw_interest in interests:
        interest = raw_interest.strip().lower()
        if not interest:
            continue

        unique_tokens = sorted({token for token in tokenize(interest) if len(token) >= 3})
        phrase_in_title = interest in title
        phrase_in_summary = interest in summary
        phrase_in_categories = interest in categories
        token_present_count = sum(1 for token in unique_tokens if token in combined)

        required_tokens = 0
        if unique_tokens:
            required_tokens = min(2, len(unique_tokens))

        if not (phrase_in_title or phrase_in_summary or phrase_in_categories):
            if required_tokens > 0 and token_present_count < required_tokens:
                continue

        local_score = 0
        if phrase_in_title:
            local_score += 12
        if phrase_in_categories:
            local_score += 7
        if phrase_in_summary:
            local_score += 6 + min(summary.count(interest), 3)

        for token in unique_tokens:
            if token in title:
                local_score += 2
            if token in summary:
                local_score += min(summary.count(token), 2)

        if local_score > 0:
            matched.append(raw_interest.strip())
            score += local_score

    return score, matched


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
    interest_terms: list[str],
    llm_model: str,
    llm_timeout: int,
) -> list[Recommendation]:
    if not papers:
        return []

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
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
        "interest_terms_hint": interest_terms,
        "papers": serialized_candidates,
        "output_schema": {
            "recommended": [
                {
                    "paper_id": "string, must exactly match one input paper_id",
                    "score": "number between 0 and 100",
                    "matched_interests": ["matched concepts from research profile"],
                    "reason": "explicit connection between this paper and the research profile",
                    "summary": "2-sentence summary",
                }
            ]
        },
        "constraints": [
            "Only use paper_ids from input.",
            "Return every relevant paper in this batch, not just top results.",
            "Exclude papers that are not relevant to the research profile.",
            "The reason must explicitly mention how the paper connects to the research profile.",
        ],
    }

    request_payload = {
        "model": llm_model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }

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
        if not matched_interests:
            _, matched_interests = score_paper(paper, interest_terms)

        summary = compact_text(str(item.get("summary", "")).strip(), 420)
        if not summary:
            summary = summarize_abstract(paper.summary)

        reason = compact_text(str(item.get("reason", "")).strip(), 280)
        if not reason:
            hint = ", ".join(matched_interests[:3]) if matched_interests else "profile overlap"
            reason = f"Connected to your research profile through {hint}."

        recommendations.append(
            Recommendation(
                paper=paper,
                score=score,
                matched_interests=matched_interests,
                summary=summary,
                reason=reason,
            )
        )

    return recommendations

def recommend_with_llm(
    papers: list[Paper],
    research_profile: str,
    interest_terms: list[str],
    llm_model: str,
    llm_batch_size: int,
    llm_timeout: int,
) -> list[Recommendation]:
    if not papers:
        return []

    ordered = sorted(papers, key=lambda item: item.published, reverse=True)
    batch_size = max(1, llm_batch_size)
    collected: list[Recommendation] = []

    for offset in range(0, len(ordered), batch_size):
        batch = ordered[offset : offset + batch_size]
        collected.extend(
            recommend_with_llm_batch(
                papers=batch,
                research_profile=research_profile,
                interest_terms=interest_terms,
                llm_model=llm_model,
                llm_timeout=llm_timeout,
            )
        )

    deduped: dict[str, Recommendation] = {}
    for recommendation in collected:
        existing = deduped.get(recommendation.paper.paper_id)
        if existing is None or recommendation.score > existing.score:
            deduped[recommendation.paper.paper_id] = recommendation

    output = list(deduped.values())
    output.sort(key=lambda item: (item.score, item.paper.published), reverse=True)
    return output

def recommend_with_local_fallback(
    papers: list[Paper],
    research_profile: str,
    interest_terms: list[str],
) -> list[Recommendation]:
    if not papers:
        return []

    profile_focus = compact_text(research_profile, 160)
    recommendations: list[Recommendation] = []
    for paper in papers:
        score, matched = score_paper(paper, interest_terms)
        if score <= 0:
            continue

        profile_hint = ", ".join(matched[:3]) if matched else "topic overlap"
        reason = f"Connected to your profile ({profile_focus}) via {profile_hint}."

        recommendations.append(
            Recommendation(
                paper=paper,
                score=float(score),
                matched_interests=matched,
                summary=summarize_abstract(paper.summary),
                reason=reason,
            )
        )

    recommendations.sort(key=lambda item: (item.score, item.paper.published), reverse=True)
    return recommendations

def recommend_and_summarize(
    papers: list[Paper],
    research_profile: str,
    interest_terms: list[str],
    llm_model: str,
    llm_batch_size: int,
    llm_timeout: int,
) -> tuple[list[Recommendation], str]:
    if not papers:
        return [], "No papers"

    try:
        return (
            recommend_with_llm(
                papers=papers,
                research_profile=research_profile,
                interest_terms=interest_terms,
                llm_model=llm_model,
                llm_batch_size=llm_batch_size,
                llm_timeout=llm_timeout,
            ),
            "LLM",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"LLM recommendation failed: {exc}. Falling back to local ranking.", file=sys.stderr)

    return recommend_with_local_fallback(papers, research_profile, interest_terms), "Local fallback"

def format_score(score: float) -> str:
    if score.is_integer():
        return str(int(score))
    return f"{score:.2f}"


def render_reports(
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    research_profile: str,
    interest_terms: list[str],
    recommendations: list[Recommendation],
    method_label: str,
) -> tuple[str, str, str]:
    profile_text = compact_text(research_profile, 700)
    terms_text = ", ".join(interest_terms[:25]) if interest_terms else "N/A"

    text_lines = [
        f"arXiv recommendations for {start_utc.isoformat()} to {end_utc.isoformat()}",
        f"Research profile: {profile_text}",
        f"Extracted concepts: {terms_text}",
        f"Recommendation method: {method_label}",
        "",
    ]

    html_blocks = [
        "<h2>arXiv Recommendations</h2>",
        f"<p>Window: <code>{escape(start_utc.isoformat())}</code> to <code>{escape(end_utc.isoformat())}</code><br>",
        f"Research profile: {escape(profile_text)}<br>",
        f"Extracted concepts: {escape(terms_text)}<br>",
        f"Recommendation method: {escape(method_label)}</p>",
        "<ol>",
    ]

    markdown_lines = [
        "# arXiv Recommendations",
        "",
        f"- Window: `{start_utc.isoformat()}` to `{end_utc.isoformat()}`",
        f"- Research profile: {profile_text}",
        f"- Extracted concepts: {terms_text}",
        f"- Recommendation method: {method_label}",
        "",
    ]

    if not recommendations:
        text_lines.append("No relevant papers found in this time window.")
        html_blocks.append("<li>No relevant papers found in this time window.</li>")
        markdown_lines.append("No relevant papers found in this time window.")

    for idx, recommendation in enumerate(recommendations, start=1):
        paper = recommendation.paper
        authors_text = ", ".join(paper.authors) if paper.authors else "Unknown authors"
        matched_text = ", ".join(recommendation.matched_interests) if recommendation.matched_interests else "None"
        score_text = format_score(recommendation.score)

        text_lines.extend(
            [
                f"{idx}. {paper.title}",
                f"   Authors: {authors_text}",
                f"   Published: {paper.published.isoformat()}",
                f"   Relevance score: {score_text}",
                f"   Matched concepts: {matched_text}",
                f"   Why selected: {recommendation.reason}",
                f"   Summary: {recommendation.summary}",
                f"   URL: {paper.link}",
                "",
            ]
        )

        html_blocks.append(
            "<li>"
            f"<p><strong>{escape(paper.title)}</strong><br>"
            f"Authors: {escape(authors_text)}<br>"
            f"Published: {escape(paper.published.isoformat())}<br>"
            f"Relevance score: {escape(score_text)}<br>"
            f"Matched concepts: {escape(matched_text)}<br>"
            f"Why selected: {escape(recommendation.reason)}<br>"
            f"Summary: {escape(recommendation.summary)}<br>"
            f"URL: <a href='{escape(paper.link)}'>{escape(paper.link)}</a></p>"
            "</li>"
        )

        markdown_lines.extend(
            [
                f"## {idx}. {paper.title}",
                f"- Authors: {authors_text}",
                f"- Published: {paper.published.isoformat()}",
                f"- Relevance score: {score_text}",
                f"- Matched concepts: {matched_text}",
                f"- Why selected: {recommendation.reason}",
                f"- Summary: {recommendation.summary}",
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
    sender = os.getenv("EMAIL_FROM", os.getenv("SMTP_USER", "noreply@example.com"))
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = (
        f"arXiv recommendations ({start_utc.strftime('%Y-%m-%d')} to {end_utc.strftime('%Y-%m-%d')})"
    )
    msg.set_content(text_report)
    msg.add_alternative(html_report, subtype="html")
    return msg


def send_email(message: EmailMessage) -> None:
    host = os.getenv("SMTP_HOST")
    if not host:
        raise ValueError("SMTP_HOST is not set.")
    port = int(os.getenv("SMTP_PORT", "587"))
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes"}
    username = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASS", "")

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        if use_tls:
            smtp.starttls()
            smtp.ehlo()
        if username:
            smtp.login(username, password)
        smtp.send_message(message)


def default_output_path(start_utc: dt.datetime, end_utc: dt.datetime) -> str:
    return (
        f"arxiv_recommendations_{start_utc.strftime('%Y%m%dT%H%M')}_"
        f"{end_utc.strftime('%Y%m%dT%H%M')}.md"
    )


def save_report(path: str, markdown_report: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write(markdown_report)


def main() -> int:
    load_dotenv()
    args = parse_args()

    research_profile = normalize_research_profile(args.research_profile)
    if not research_profile:
        print("No research profile provided. Set --research-profile or RESEARCH_PROFILE.", file=sys.stderr)
        return 1

    interest_terms = extract_interest_terms(research_profile)

    try:
        start_utc, end_utc = compute_time_window(
            start_raw=args.start,
            end_raw=args.end,
            tz_name=args.timezone,
            time_parse_model=args.time_parse_model,
            llm_timeout=max(5, args.llm_timeout),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Invalid time settings: {exc}", file=sys.stderr)
        return 1

    try:
        papers = fetch_arxiv_papers(start_utc, end_utc, args.max_results)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to fetch arXiv papers: {exc}", file=sys.stderr)
        return 1

    recommendations, method_label = recommend_and_summarize(
        papers=papers,
        research_profile=research_profile,
        interest_terms=interest_terms,
        llm_model=args.llm_model,
        llm_batch_size=max(1, args.llm_batch_size),
        llm_timeout=max(5, args.llm_timeout),
    )

    text_report, html_report, markdown_report = render_reports(
        start_utc=start_utc,
        end_utc=end_utc,
        research_profile=research_profile,
        interest_terms=interest_terms,
        recommendations=recommendations,
        method_label=method_label,
    )

    if not args.to:
        output_path = args.output or default_output_path(start_utc, end_utc)
        save_report(output_path, markdown_report)
        print(f"No email recipient provided. Saved recommendations to {output_path}.")
        return 0

    message = build_email(args.to, start_utc, end_utc, text_report, html_report)

    if args.dry_run:
        print(message)
    else:
        try:
            send_email(message)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to send email: {exc}", file=sys.stderr)
            return 1
        print(f"Sent {len(recommendations)} recommendations to {args.to}.")

    if args.output:
        save_report(args.output, markdown_report)
        print(f"Saved recommendations to {args.output}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


























