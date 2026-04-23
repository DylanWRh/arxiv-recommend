from __future__ import annotations

import json
import sys
import time

import requests

from utils.models import Paper, Recommendation
from utils.runtime import debug_log, run_with_retries, str_env
from utils.text import compact_text, extract_json_object, normalize_str_list, summarize_abstract


def _pick_text(item: dict[str, object], keys: tuple[str, ...], limit: int, fallback: str = "") -> str:
    for key in keys:
        value = compact_text(str(item.get(key, "")).strip(), limit)
        if value:
            return value
    return fallback


def _build_rec(item: dict[str, object], paper: Paper) -> Recommendation:
    raw_score = item.get("score", 0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0

    matched = normalize_str_list(item.get("matched_interests", []))
    llm_summary = _pick_text(item, ("llm_summary", "summary"), 420, summarize_abstract(paper.abstract))

    reason = _pick_text(item, ("reason",), 280)
    if not reason:
        hint = ", ".join(matched[:3]) if matched else "profile overlap"
        reason = f"Connected to your research profile through {hint}."

    title_zh = _pick_text(item, ("title_zh",), 280, f"\uff08\u672a\u63d0\u4f9b\u4e2d\u6587\u6807\u9898\uff09{paper.title}")
    summary_zh = _pick_text(item, ("summary_zh", "abstract_zh"), 520, "\uff08\u672a\u63d0\u4f9b\u4e2d\u6587\u603b\u7ed3\uff09")

    return Recommendation(
        paper=paper,
        title_zh=title_zh,
        summary_zh=summary_zh,
        score=score,
        matched_interests=matched,
        llm_summary=llm_summary,
        reason=reason,
    )


def _recommend_batch(
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
            "abstract": compact_text(paper.abstract, 1100),
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
                    "llm_summary": "2-sentence summary in English",
                    "title_zh": "Chinese title (Simplified Chinese)",
                    "summary_zh": "Chinese summary in 2-3 sentences (Simplified Chinese)",
                }
            ]
        },
        "constraints": [
            "Only use paper_ids from input.",
            "Return every relevant paper in this batch, not just top results.",
            "Exclude papers that are not relevant to the research profile.",
            "The reason must explicitly mention how the paper connects to the research profile.",
            "title_zh and summary_zh must be written in Simplified Chinese.",
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

    recs: list[Recommendation] = []
    seen_ids: set[str] = set()

    for item in raw_recommended:
        if not isinstance(item, dict):
            continue

        paper_id = str(item.get("paper_id", "")).strip()
        if not paper_id or paper_id in seen_ids or paper_id not in paper_map:
            continue

        paper = paper_map[paper_id]
        seen_ids.add(paper_id)
        recs.append(_build_rec(item, paper))

    debug_log(dbg, f"Recommendation request{label} returned {len(recs)} recommendation(s).")
    return recs


def _recommend_all(
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
            _recommend_batch(
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
        recommendations = _recommend_all(
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
