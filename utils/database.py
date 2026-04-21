from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from collections import defaultdict

from utils.models import Recommendation
from utils.runtime import debug_log

DEFAULT_RECOMMENDATIONS_STATE_DIR = "data"
PAPER_ID_SHARDS_DIRNAME = "paper-id-shards"
RECOMMENDATION_RUNS_DIRNAME = "recommendation-runs"


def resolve_state_dir(path: str) -> str:
    resolved = os.path.expanduser(path).strip()
    if not resolved:
        return DEFAULT_RECOMMENDATIONS_STATE_DIR
    if resolved.lower().endswith(".db"):
        parent = os.path.dirname(resolved)
        return parent or DEFAULT_RECOMMENDATIONS_STATE_DIR
    return resolved


def _paper_id_shards_dir(state_dir: str) -> str:
    return os.path.join(state_dir, PAPER_ID_SHARDS_DIRNAME)


def _recommendation_runs_dir(state_dir: str) -> str:
    return os.path.join(state_dir, RECOMMENDATION_RUNS_DIRNAME)


def ensure_recommendations_state_layout(state_dir: str) -> str:
    resolved_state_dir = resolve_state_dir(state_dir)
    os.makedirs(_paper_id_shards_dir(resolved_state_dir), exist_ok=True)
    os.makedirs(_recommendation_runs_dir(resolved_state_dir), exist_ok=True)
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


def _paper_id_shard_name(paper_id: str) -> str:
    return hashlib.sha1(paper_id.encode("utf-8")).hexdigest()[:2] + ".json"


def _paper_id_shard_path(state_dir: str, shard_name: str) -> str:
    return os.path.join(_paper_id_shards_dir(state_dir), shard_name)


def load_saved_paper_ids(state_dir: str, dbg: bool = False) -> set[str]:
    resolved_state_dir = ensure_recommendations_state_layout(state_dir)
    shard_dir = _paper_id_shards_dir(resolved_state_dir)

    saved_ids: set[str] = set()
    for name in sorted(os.listdir(shard_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(shard_dir, name)
        payload = _load_json_file(path, {"paper_ids": []})
        if not isinstance(payload, dict):
            continue
        paper_ids = payload.get("paper_ids", [])
        if not isinstance(paper_ids, list):
            continue
        for paper_id in paper_ids:
            normalized = str(paper_id).strip()
            if normalized:
                saved_ids.add(normalized)

    debug_log(dbg, f"Loaded {len(saved_ids)} saved paper id(s) from {resolved_state_dir}.")
    return saved_ids


def exclude_saved_papers(papers: list, saved_paper_ids: set[str]) -> tuple[list, int]:
    if not papers or not saved_paper_ids:
        return papers, 0

    unsaved_papers = [paper for paper in papers if paper.paper_id not in saved_paper_ids]
    return unsaved_papers, len(papers) - len(unsaved_papers)


def _serialize_recommendation(
    recommendation: Recommendation,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    recommended_at: str,
) -> dict[str, object]:
    paper = recommendation.paper
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "link": paper.link,
        "authors": paper.authors,
        "categories": paper.categories,
        "published_at": paper.published.isoformat(),
        "updated_at": paper.updated.isoformat(),
        "recommended_at": recommended_at,
        "score": recommendation.score,
        "matched_interests": recommendation.matched_interests,
        "reason": recommendation.reason,
        "summary": recommendation.summary,
        "title_zh": recommendation.title_zh,
        "abstract_zh": recommendation.abstract_zh,
        "query_window_start": start_utc.isoformat(),
        "query_window_end": end_utc.isoformat(),
    }


def _build_run_record_path(
    state_dir: str,
    recommended_at_utc: dt.datetime,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
) -> str:
    year_dir = recommended_at_utc.strftime("%Y")
    month_dir = recommended_at_utc.strftime("%Y-%m")
    filename = (
        f"run_{recommended_at_utc.strftime('%Y%m%dT%H%M%SZ')}"
        f"_q{start_utc.strftime('%Y%m%dT%H%M%SZ')}"
        f"_{end_utc.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    return os.path.join(_recommendation_runs_dir(state_dir), year_dir, month_dir, filename)


def save_recommendations_history(
    state_dir: str,
    recommendations: list[Recommendation],
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    dbg: bool = False,
) -> int:
    if not recommendations:
        debug_log(dbg, "No recommendations to persist to the state store.")
        return 0

    resolved_state_dir = ensure_recommendations_state_layout(state_dir)
    existing_ids = load_saved_paper_ids(resolved_state_dir, dbg=False)

    deduped_by_id: dict[str, Recommendation] = {}
    for recommendation in recommendations:
        paper_id = recommendation.paper.paper_id.strip()
        if not paper_id or paper_id in existing_ids or paper_id in deduped_by_id:
            continue
        deduped_by_id[paper_id] = recommendation

    if not deduped_by_id:
        debug_log(dbg, "All recommendations were already present in the state store.")
        return 0

    recommended_at_utc = dt.datetime.now(dt.timezone.utc)
    recommended_at = recommended_at_utc.isoformat()
    new_recommendations = list(deduped_by_id.values())
    serialized_recommendations = [
        _serialize_recommendation(recommendation, start_utc, end_utc, recommended_at)
        for recommendation in new_recommendations
    ]

    run_record = {
        "generated_at": recommended_at,
        "query_window_start": start_utc.isoformat(),
        "query_window_end": end_utc.isoformat(),
        "recommendation_count": len(serialized_recommendations),
        "recommendations": serialized_recommendations,
    }
    _write_json_file(
        _build_run_record_path(resolved_state_dir, recommended_at_utc, start_utc, end_utc),
        run_record,
    )

    shard_updates: dict[str, list[str]] = defaultdict(list)
    for paper_id in deduped_by_id:
        shard_updates[_paper_id_shard_name(paper_id)].append(paper_id)

    for shard_name, new_ids in shard_updates.items():
        shard_path = _paper_id_shard_path(resolved_state_dir, shard_name)
        payload = _load_json_file(shard_path, {"paper_ids": [], "updated_at": recommended_at})
        if not isinstance(payload, dict):
            payload = {"paper_ids": [], "updated_at": recommended_at}
        existing_shard_ids = payload.get("paper_ids", [])
        if not isinstance(existing_shard_ids, list):
            existing_shard_ids = []
        merged_ids = sorted({str(paper_id).strip() for paper_id in existing_shard_ids + new_ids if str(paper_id).strip()})
        _write_json_file(
            shard_path,
            {
                "paper_ids": merged_ids,
                "updated_at": recommended_at,
            },
        )

    inserted = len(new_recommendations)
    debug_log(dbg, f"Persisted {inserted} recommendation(s) to {resolved_state_dir}.")
    return inserted
