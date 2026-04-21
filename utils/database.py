from __future__ import annotations

import datetime as dt
import json
import os
import re

from utils.models import Paper, Recommendation
from utils.runtime import debug_log

DEFAULT_RECOMMENDATIONS_STATE_DIR = "data"
RECOMMENDED_PAPER_RECORDS_DIRNAME = "recommended-papers"
UNRECOMMENDED_PAPER_RECORDS_DIRNAME = "not-recommended-papers"
DEFAULT_RECOMMENDED_PAPER_REVIEW_STATUS = "unchecked"


def resolve_state_dir(path: str) -> str:
    resolved = os.path.expanduser(path).strip()
    return resolved or DEFAULT_RECOMMENDATIONS_STATE_DIR


def _recommended_paper_records_dir(state_dir: str) -> str:
    return os.path.join(state_dir, RECOMMENDED_PAPER_RECORDS_DIRNAME)


def _unrecommended_paper_records_dir(state_dir: str) -> str:
    return os.path.join(state_dir, UNRECOMMENDED_PAPER_RECORDS_DIRNAME)


def ensure_recommendations_state_layout(state_dir: str) -> str:
    resolved_state_dir = resolve_state_dir(state_dir)
    os.makedirs(_recommended_paper_records_dir(resolved_state_dir), exist_ok=True)
    os.makedirs(_unrecommended_paper_records_dir(resolved_state_dir), exist_ok=True)
    return resolved_state_dir


def _load_json_file(path: str, default: object) -> object:
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json_file(path: str, payload: object) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")


def _canonical_paper_identifier(paper_id: str) -> str:
    normalized = paper_id.strip()
    if not normalized:
        return ""
    if "/abs/" in normalized:
        normalized = normalized.split("/abs/", 1)[1]
    return normalized.strip().strip("/")


def _paper_yymm(paper_id: str) -> str:
    identifier = _canonical_paper_identifier(paper_id)
    match = re.match(r"(?:[^/]+/)?(?P<yymm>\d{4})", identifier)
    if match:
        return match.group("yymm")
    return "unknown"


def _normalize_timestamp(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _paper_date_parts(paper_id: str, timestamp: dt.datetime) -> tuple[str, str]:
    normalized_timestamp = _normalize_timestamp(timestamp)
    yymm = normalized_timestamp.strftime("%y%m") or _paper_yymm(paper_id)
    day = normalized_timestamp.strftime("%d")
    return yymm, day


def _day_record_path(state_dir: str, dirname: str, paper: Paper) -> tuple[str, str]:
    normalized_timestamp = _normalize_timestamp(paper.published)
    yymm, day = _paper_date_parts(paper.paper_id, normalized_timestamp)
    return (
        os.path.join(state_dir, dirname, yymm, f"{day}.json"),
        normalized_timestamp.date().isoformat(),
    )


def _iter_json_record_paths(records_dir: str) -> list[str]:
    record_paths: list[str] = []
    if not os.path.isdir(records_dir):
        return record_paths

    for root, _, files in os.walk(records_dir):
        for name in files:
            if name.endswith(".json"):
                record_paths.append(os.path.join(root, name))
    record_paths.sort()
    return record_paths


def _load_day_record(path: str, fallback_date: str) -> tuple[str, dict[str, dict[str, object]]]:
    payload = _load_json_file(path, {})
    record_date = fallback_date
    papers_by_id: dict[str, dict[str, object]] = {}

    if not isinstance(payload, dict):
        return record_date, papers_by_id

    payload_date = str(payload.get("date", "")).strip()
    if payload_date:
        record_date = payload_date

    papers_payload = payload.get("papers")
    if not isinstance(papers_payload, list):
        return record_date, papers_by_id

    for entry in papers_payload:
        if not isinstance(entry, dict):
            continue
        paper_id = _canonical_paper_identifier(str(entry.get("paper_id", "")).strip())
        if not paper_id:
            continue
        normalized_entry = dict(entry)
        normalized_entry["paper_id"] = paper_id
        papers_by_id[paper_id] = normalized_entry
    return record_date, papers_by_id


def _write_day_record(path: str, record_date: str, papers_by_id: dict[str, dict[str, object]]) -> None:
    payload = {
        "date": record_date,
        "papers": sorted(papers_by_id.values(), key=lambda item: str(item.get("paper_id", ""))),
    }
    _write_json_file(path, payload)


def _load_saved_paper_ids_from_records_dir(records_dir: str) -> set[str]:
    saved_ids: set[str] = set()
    for path in _iter_json_record_paths(records_dir):
        _, papers_by_id = _load_day_record(path, "")
        saved_ids.update(papers_by_id)
    return saved_ids


def load_saved_paper_ids(state_dir: str, dbg: bool = False) -> set[str]:
    resolved_state_dir = ensure_recommendations_state_layout(state_dir)
    saved_ids = _load_saved_paper_ids_from_records_dir(_recommended_paper_records_dir(resolved_state_dir))
    saved_ids.update(_load_saved_paper_ids_from_records_dir(_unrecommended_paper_records_dir(resolved_state_dir)))
    debug_log(dbg, f"Loaded {len(saved_ids)} saved paper id(s) from {resolved_state_dir}.")
    return saved_ids


def exclude_saved_papers(papers: list[Paper], saved_paper_ids: set[str]) -> tuple[list[Paper], int]:
    if not papers or not saved_paper_ids:
        return papers, 0

    unsaved_papers = [
        paper for paper in papers if _canonical_paper_identifier(paper.paper_id) not in saved_paper_ids
    ]
    return unsaved_papers, len(papers) - len(unsaved_papers)


def _serialize_recommendation(
    recommendation: Recommendation,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    recommended_at: str,
) -> dict[str, object]:
    paper = recommendation.paper
    return {
        "abstract_zh": recommendation.abstract_zh,
        "authors": paper.authors,
        "categories": paper.categories,
        "matched_interests": recommendation.matched_interests,
        "paper_id": _canonical_paper_identifier(paper.paper_id),
        "published_at": paper.published.isoformat(),
        "query_window_end": end_utc.isoformat(),
        "query_window_start": start_utc.isoformat(),
        "reason": recommendation.reason,
        "recommended_at": recommended_at,
        "review_status": DEFAULT_RECOMMENDED_PAPER_REVIEW_STATUS,
        "score": recommendation.score,
        "summary": recommendation.summary,
        "title": paper.title,
        "title_zh": recommendation.title_zh,
        "updated_at": paper.updated.isoformat(),
        "url": paper.link,
    }


def _serialize_paper_summary(paper: Paper) -> dict[str, str]:
    return {
        "paper_id": _canonical_paper_identifier(paper.paper_id),
        "title": paper.title,
        "url": paper.link,
    }


def save_processed_papers_state(
    state_dir: str,
    processed_papers: list[Paper],
    recommendations: list[Recommendation],
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    dbg: bool = False,
) -> int:
    if not processed_papers:
        debug_log(dbg, "No processed papers to persist to the state store.")
        return 0

    resolved_state_dir = ensure_recommendations_state_layout(state_dir)
    existing_ids = load_saved_paper_ids(resolved_state_dir, dbg=False)

    new_papers_by_id: dict[str, Paper] = {}
    for paper in processed_papers:
        paper_id = _canonical_paper_identifier(paper.paper_id)
        if not paper_id or paper_id in existing_ids or paper_id in new_papers_by_id:
            continue
        new_papers_by_id[paper_id] = paper

    if not new_papers_by_id:
        debug_log(dbg, "All processed papers were already present in the state store.")
        return 0

    recommendation_by_id: dict[str, Recommendation] = {}
    for recommendation in recommendations:
        paper_id = _canonical_paper_identifier(recommendation.paper.paper_id)
        if paper_id and paper_id in new_papers_by_id and paper_id not in recommendation_by_id:
            recommendation_by_id[paper_id] = recommendation

    recommended_at = dt.datetime.now(dt.timezone.utc).isoformat()
    grouped_records: dict[str, tuple[str, dict[str, dict[str, object]]]] = {}

    for paper_id, paper in new_papers_by_id.items():
        if paper_id in recommendation_by_id:
            record_path, record_date = _day_record_path(
                resolved_state_dir,
                RECOMMENDED_PAPER_RECORDS_DIRNAME,
                paper,
            )
            payload = _serialize_recommendation(
                recommendation_by_id[paper_id],
                start_utc=start_utc,
                end_utc=end_utc,
                recommended_at=recommended_at,
            )
        else:
            record_path, record_date = _day_record_path(
                resolved_state_dir,
                UNRECOMMENDED_PAPER_RECORDS_DIRNAME,
                paper,
            )
            payload = _serialize_paper_summary(paper)

        existing_date, existing_papers = grouped_records.get(record_path, (record_date, {}))
        existing_papers[paper_id] = payload
        grouped_records[record_path] = (existing_date or record_date, existing_papers)

    for record_path, (record_date, new_papers) in grouped_records.items():
        existing_date, existing_papers = _load_day_record(record_path, record_date)
        existing_papers.update(new_papers)
        _write_day_record(record_path, existing_date or record_date, existing_papers)

    inserted = len(new_papers_by_id)
    debug_log(
        dbg,
        (
            f"Persisted {inserted} processed paper(s) to {resolved_state_dir}, "
            f"including {len(recommendation_by_id)} recommendation(s)."
        ),
    )
    return inserted
