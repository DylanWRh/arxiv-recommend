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
LEGACY_PAPER_RECORDS_DIRNAME = "papers"
LEGACY_PAPER_ID_SHARDS_DIRNAME = "paper-id-shards"
LEGACY_RECOMMENDATION_RUNS_DIRNAME = "recommendation-runs"


def resolve_state_dir(path: str) -> str:
    resolved = os.path.expanduser(path).strip()
    if not resolved:
        return DEFAULT_RECOMMENDATIONS_STATE_DIR
    if resolved.lower().endswith(".db"):
        parent = os.path.dirname(resolved)
        return parent or DEFAULT_RECOMMENDATIONS_STATE_DIR
    return resolved


def _recommended_paper_records_dir(state_dir: str) -> str:
    return os.path.join(state_dir, RECOMMENDED_PAPER_RECORDS_DIRNAME)


def _unrecommended_paper_records_dir(state_dir: str) -> str:
    return os.path.join(state_dir, UNRECOMMENDED_PAPER_RECORDS_DIRNAME)


def _legacy_paper_records_dir(state_dir: str) -> str:
    return os.path.join(state_dir, LEGACY_PAPER_RECORDS_DIRNAME)


def _legacy_paper_id_shards_dir(state_dir: str) -> str:
    return os.path.join(state_dir, LEGACY_PAPER_ID_SHARDS_DIRNAME)


def _legacy_recommendation_runs_dir(state_dir: str) -> str:
    return os.path.join(state_dir, LEGACY_RECOMMENDATION_RUNS_DIRNAME)


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


def _paper_record_filename(paper_id: str) -> str:
    identifier = _canonical_paper_identifier(paper_id)
    safe_identifier = identifier.replace("/", "__")
    return f"{safe_identifier or 'unknown'}.json"


def _paper_record_path(state_dir: str, paper_id: str, *, recommended: bool) -> str:
    root_dir = _recommended_paper_records_dir(state_dir) if recommended else _unrecommended_paper_records_dir(state_dir)
    return os.path.join(root_dir, _paper_yymm(paper_id), _paper_record_filename(paper_id))


def _paper_record_dirs(state_dir: str) -> tuple[str, ...]:
    return (
        _recommended_paper_records_dir(state_dir),
        _unrecommended_paper_records_dir(state_dir),
        _legacy_paper_records_dir(state_dir),
    )


def _load_saved_paper_ids_from_paper_files(state_dir: str) -> set[str]:
    saved_ids: set[str] = set()
    for records_dir in _paper_record_dirs(state_dir):
        if not os.path.isdir(records_dir):
            continue
        for root, _, files in os.walk(records_dir):
            for name in files:
                if not name.endswith(".json"):
                    continue
                payload = _load_json_file(os.path.join(root, name), {})
                if not isinstance(payload, dict):
                    continue
                paper_id = _canonical_paper_identifier(str(payload.get("paper_id", "")).strip())
                if paper_id:
                    saved_ids.add(paper_id)
    return saved_ids


def _load_saved_paper_ids_from_legacy_shards(state_dir: str) -> set[str]:
    shard_dir = _legacy_paper_id_shards_dir(state_dir)
    if not os.path.isdir(shard_dir):
        return set()

    saved_ids: set[str] = set()
    for name in sorted(os.listdir(shard_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(shard_dir, name)
        payload = _load_json_file(path, {})
        if not isinstance(payload, dict):
            continue

        papers_payload = payload.get("papers")
        if isinstance(papers_payload, dict):
            for raw_paper_id, value in papers_payload.items():
                paper_id = _canonical_paper_identifier(str(raw_paper_id).strip())
                if not paper_id and isinstance(value, dict):
                    paper_id = _canonical_paper_identifier(str(value.get("paper_id", "")).strip())
                if paper_id:
                    saved_ids.add(paper_id)

        paper_ids = payload.get("paper_ids")
        if isinstance(paper_ids, list):
            for raw_paper_id in paper_ids:
                paper_id = _canonical_paper_identifier(str(raw_paper_id).strip())
                if paper_id:
                    saved_ids.add(paper_id)

    return saved_ids


def _load_saved_paper_ids_from_legacy_runs(state_dir: str) -> set[str]:
    runs_dir = _legacy_recommendation_runs_dir(state_dir)
    if not os.path.isdir(runs_dir):
        return set()

    saved_ids: set[str] = set()
    for root, _, files in os.walk(runs_dir):
        for name in files:
            if not name.endswith(".json"):
                continue
            payload = _load_json_file(os.path.join(root, name), {})
            if not isinstance(payload, dict):
                continue

            for key in ("recommended_papers", "not_recommended_papers", "recommendations"):
                entries = payload.get(key, [])
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    paper_id = _canonical_paper_identifier(str(entry.get("paper_id", "")).strip())
                    if paper_id:
                        saved_ids.add(paper_id)

    return saved_ids


def load_saved_paper_ids(state_dir: str, dbg: bool = False) -> set[str]:
    resolved_state_dir = ensure_recommendations_state_layout(state_dir)

    saved_ids = _load_saved_paper_ids_from_paper_files(resolved_state_dir)
    saved_ids.update(_load_saved_paper_ids_from_legacy_shards(resolved_state_dir))
    saved_ids.update(_load_saved_paper_ids_from_legacy_runs(resolved_state_dir))

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

    for paper_id, paper in new_papers_by_id.items():
        is_recommended = paper_id in recommendation_by_id
        if is_recommended:
            payload = _serialize_recommendation(
                recommendation_by_id[paper_id],
                start_utc=start_utc,
                end_utc=end_utc,
                recommended_at=recommended_at,
            )
        else:
            payload = _serialize_paper_summary(paper)
        _write_json_file(_paper_record_path(resolved_state_dir, paper_id, recommended=is_recommended), payload)

    inserted = len(new_papers_by_id)
    debug_log(
        dbg,
        (
            f"Persisted {inserted} processed paper(s) to {resolved_state_dir}, "
            f"including {len(recommendation_by_id)} recommendation(s)."
        ),
    )
    return inserted


def save_recommendations_history(
    state_dir: str,
    recommendations: list[Recommendation],
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    dbg: bool = False,
) -> int:
    papers = [recommendation.paper for recommendation in recommendations]
    return save_processed_papers_state(
        state_dir=state_dir,
        processed_papers=papers,
        recommendations=recommendations,
        start_utc=start_utc,
        end_utc=end_utc,
        dbg=dbg,
    )
