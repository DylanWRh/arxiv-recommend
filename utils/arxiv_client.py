from __future__ import annotations

import datetime as dt
import random
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import requests

from utils.models import Paper
from utils.runtime import debug_log

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_PAGE_SIZE = 2000
ARXIV_MIN_REQUEST_GAP_SECONDS = 5.0
ARXIV_MAX_RETRY_ATTEMPTS = 60
ARXIV_RETRY_BASE_DELAY_SECONDS = 5.0
ARXIV_RETRY_MAX_DELAY_SECONDS = 60.0
ARXIV_REQUEST_HEADERS = {
    "User-Agent": "arxiv-recommend/1.0",
    "Accept": "application/atom+xml",
}

_last_arxiv_request_monotonic = 0.0


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
    page_size = min(ARXIV_PAGE_SIZE, max_results)

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
        start_idx += len(papers)
        if len(papers) < batch_size:
            break

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
