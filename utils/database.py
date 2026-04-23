from __future__ import annotations

import datetime as dt
import json
import os

from utils.models import Paper, Recommendation
from utils.runtime import debug_log

DEFAULT_RECOMMENDATIONS_STATE_DIR = "data"
RECOMMENDED_PAPER_RECORDS_DIRNAME = "recommended-papers"
UNRECOMMENDED_PAPER_RECORDS_DIRNAME = "not-recommended-papers"
STATE_RECORD_DIRNAMES = (
    RECOMMENDED_PAPER_RECORDS_DIRNAME,
    UNRECOMMENDED_PAPER_RECORDS_DIRNAME,
)
DEFAULT_RECOMMENDED_PAPER_REVIEW_STATUS = "unchecked"
RECOMMENDED_RECORD_GENERATION_KEYS = (
    "score",
    "reason",
    "matched_interests",
    "recommended_at",
    "query_window_start",
    "query_window_end",
)


def resolve_state_dir(path: str) -> str:
    resolved = os.path.expanduser(path).strip()
    return resolved or DEFAULT_RECOMMENDATIONS_STATE_DIR


def _records_dir(state_dir: str, dirname: str) -> str:
    return os.path.join(state_dir, dirname)


def ensure_recommendations_state_layout(state_dir: str) -> str:
    resolved_state_dir = resolve_state_dir(state_dir)
    for dirname in STATE_RECORD_DIRNAMES:
        os.makedirs(_records_dir(resolved_state_dir, dirname), exist_ok=True)
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


def _normalize_text_field(value: object) -> str:
    return " ".join(str(value or "").split())


def _is_generated_recommendation_record(record: dict[str, object]) -> bool:
    return any(key in record for key in RECOMMENDED_RECORD_GENERATION_KEYS)


def _normalize_recommended_record_entry(entry: dict[str, object]) -> dict[str, object]:
    normalized_entry = dict(entry)
    legacy_summary = _normalize_text_field(normalized_entry.pop("summary", ""))
    legacy_summary_zh = _normalize_text_field(normalized_entry.pop("abstract_zh", ""))
    legacy_abstract = _normalize_text_field(normalized_entry.pop("abstract", ""))

    paper_abstract = _normalize_text_field(normalized_entry.get("paper_abstract", legacy_abstract))
    llm_summary = _normalize_text_field(normalized_entry.get("llm_summary", ""))
    summary_zh = _normalize_text_field(normalized_entry.get("summary_zh", ""))

    if not llm_summary and legacy_summary and _is_generated_recommendation_record(normalized_entry):
        llm_summary = legacy_summary
    if not summary_zh and legacy_summary_zh:
        summary_zh = legacy_summary_zh

    normalized_entry["paper_abstract"] = paper_abstract
    normalized_entry["llm_summary"] = llm_summary
    normalized_entry["summary_zh"] = summary_zh
    return normalized_entry


def _normalize_entry(dirname: str, entry: dict[str, object]) -> dict[str, object]:
    if dirname != RECOMMENDED_PAPER_RECORDS_DIRNAME:
        return dict(entry)
    return _normalize_recommended_record_entry(entry)


def _record_path(state_dir: str, dirname: str, paper: Paper) -> tuple[str, str]:
    published_at = paper.published
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=dt.timezone.utc)
    else:
        published_at = published_at.astimezone(dt.timezone.utc)
    yymm = published_at.strftime("%y%m")
    day = published_at.strftime("%d")
    return (
        os.path.join(_records_dir(state_dir, dirname), yymm, f"{day}.json"),
        published_at.date().isoformat(),
    )


def _iter_record_files(records_dir: str) -> list[str]:
    record_paths: list[str] = []
    if not os.path.isdir(records_dir):
        return record_paths

    for root, _, files in os.walk(records_dir):
        for name in files:
            if name.endswith(".json"):
                record_paths.append(os.path.join(root, name))
    record_paths.sort()
    return record_paths


def _read_day_file(
    path: str,
    fallback_date: str,
    dirname: str,
) -> tuple[str, dict[str, dict[str, object]]]:
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
        normalized_entry = _normalize_entry(dirname, entry)
        normalized_entry["paper_id"] = paper_id
        papers_by_id[paper_id] = normalized_entry
    return record_date, papers_by_id


def _day_payload(
    record_date: str,
    papers_by_id: dict[str, dict[str, object]],
    dirname: str,
) -> dict[str, object]:
    return {
        "date": record_date,
        "papers": sorted(
            (_normalize_entry(dirname, item) for item in papers_by_id.values()),
            key=lambda item: str(item.get("paper_id", "")),
        ),
    }


def _write_day_file(path: str, record_date: str, papers_by_id: dict[str, dict[str, object]], dirname: str) -> None:
    _write_json_file(path, _day_payload(record_date, papers_by_id, dirname))


def load_saved_paper_ids(state_dir: str, dbg: bool = False) -> set[str]:
    resolved_state_dir = resolve_state_dir(state_dir)
    saved_ids: set[str] = set()
    for dirname in STATE_RECORD_DIRNAMES:
        for path in _iter_record_files(_records_dir(resolved_state_dir, dirname)):
            _, papers_by_id = _read_day_file(path, "", dirname)
            saved_ids.update(papers_by_id)
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
        "authors": paper.authors,
        "categories": paper.categories,
        "llm_summary": recommendation.llm_summary,
        "matched_interests": recommendation.matched_interests,
        "paper_abstract": paper.abstract,
        "paper_id": _canonical_paper_identifier(paper.paper_id),
        "published_at": paper.published.isoformat(),
        "query_window_end": end_utc.isoformat(),
        "query_window_start": start_utc.isoformat(),
        "reason": recommendation.reason,
        "recommended_at": recommended_at,
        "review_status": DEFAULT_RECOMMENDED_PAPER_REVIEW_STATUS,
        "score": recommendation.score,
        "summary_zh": recommendation.summary_zh,
        "title": paper.title,
        "title_zh": recommendation.title_zh,
        "updated_at": paper.updated.isoformat(),
        "url": paper.link,
    }


def _serialize_paper_record(paper: Paper) -> dict[str, str]:
    return {
        "paper_id": _canonical_paper_identifier(paper.paper_id),
        "title": paper.title,
        "url": paper.link,
    }


def normalize_state_records(state_dir: str, dbg: bool = False) -> int:
    resolved_state_dir = ensure_recommendations_state_layout(state_dir)
    normalized_files = 0

    for dirname in STATE_RECORD_DIRNAMES:
        for path in _iter_record_files(_records_dir(resolved_state_dir, dirname)):
            record_date, papers_by_id = _read_day_file(path, "", dirname)
            expected_payload = _day_payload(record_date, papers_by_id, dirname)
            current_payload = _load_json_file(path, {})
            if current_payload == expected_payload:
                continue
            _write_json_file(path, expected_payload)
            normalized_files += 1

    debug_log(dbg, f"Normalized {normalized_files} recommendation state file(s) in {resolved_state_dir}.")
    return normalized_files


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
    normalize_state_records(resolved_state_dir, dbg=False)
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
    pending_records: dict[str, tuple[str, str, dict[str, dict[str, object]]]] = {}

    for paper_id, paper in new_papers_by_id.items():
        if paper_id in recommendation_by_id:
            dirname = RECOMMENDED_PAPER_RECORDS_DIRNAME
            record_path, record_date = _record_path(
                resolved_state_dir,
                dirname,
                paper,
            )
            payload = _serialize_recommendation(
                recommendation_by_id[paper_id],
                start_utc=start_utc,
                end_utc=end_utc,
                recommended_at=recommended_at,
            )
        else:
            dirname = UNRECOMMENDED_PAPER_RECORDS_DIRNAME
            record_path, record_date = _record_path(
                resolved_state_dir,
                dirname,
                paper,
            )
            payload = _serialize_paper_record(paper)

        existing_dirname, existing_date, existing_papers = pending_records.get(
            record_path,
            (dirname, record_date, {}),
        )
        existing_papers[paper_id] = payload
        pending_records[record_path] = (existing_dirname, existing_date or record_date, existing_papers)

    for record_path, (dirname, record_date, new_papers) in pending_records.items():
        existing_date, existing_papers = _read_day_file(record_path, record_date, dirname)
        existing_papers.update(new_papers)
        _write_day_file(record_path, existing_date or record_date, existing_papers, dirname)

    inserted = len(new_papers_by_id)
    debug_log(
        dbg,
        (
            f"Persisted {inserted} processed paper(s) to {resolved_state_dir}, "
            f"including {len(recommendation_by_id)} recommendation(s)."
        ),
    )
    return inserted
