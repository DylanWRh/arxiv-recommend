from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3

from utils.models import Paper, Recommendation
from utils.runtime import debug_log

DEFAULT_RECOMMENDATIONS_DB_PATH = os.path.join("data", "recommendations.db")


def ensure_recommendations_db_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recommended_papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            link TEXT NOT NULL,
            authors_json TEXT NOT NULL,
            categories_json TEXT NOT NULL,
            published_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            recommended_at TEXT NOT NULL,
            score REAL NOT NULL,
            matched_interests_json TEXT NOT NULL,
            reason TEXT NOT NULL,
            summary TEXT NOT NULL,
            title_zh TEXT NOT NULL,
            abstract_zh TEXT NOT NULL,
            query_window_start TEXT NOT NULL,
            query_window_end TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_recommended_papers_recommended_at
        ON recommended_papers (recommended_at)
        """
    )


def connect_recommendations_db(path: str) -> sqlite3.Connection:
    resolved_path = os.path.expanduser(path)
    directory = os.path.dirname(resolved_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    conn = sqlite3.connect(resolved_path)
    ensure_recommendations_db_schema(conn)
    conn.commit()
    return conn


def load_saved_paper_ids(db_path: str, dbg: bool = False) -> set[str]:
    with connect_recommendations_db(db_path) as conn:
        rows = conn.execute("SELECT paper_id FROM recommended_papers")
        saved_ids = {str(row[0]).strip() for row in rows if str(row[0]).strip()}

    debug_log(dbg, f"Loaded {len(saved_ids)} saved paper id(s) from {os.path.expanduser(db_path)}.")
    return saved_ids


def exclude_saved_papers(papers: list[Paper], saved_paper_ids: set[str]) -> tuple[list[Paper], int]:
    if not papers or not saved_paper_ids:
        return papers, 0

    unsaved_papers = [paper for paper in papers if paper.paper_id not in saved_paper_ids]
    return unsaved_papers, len(papers) - len(unsaved_papers)


def save_recommendations_history(
    db_path: str,
    recommendations: list[Recommendation],
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    dbg: bool = False,
) -> int:
    if not recommendations:
        debug_log(dbg, "No recommendations to persist to the history database.")
        return 0

    recommended_at = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = [
        (
            recommendation.paper.paper_id,
            recommendation.paper.title,
            recommendation.paper.link,
            json.dumps(recommendation.paper.authors, ensure_ascii=False),
            json.dumps(recommendation.paper.categories, ensure_ascii=False),
            recommendation.paper.published.isoformat(),
            recommendation.paper.updated.isoformat(),
            recommended_at,
            recommendation.score,
            json.dumps(recommendation.matched_interests, ensure_ascii=False),
            recommendation.reason,
            recommendation.summary,
            recommendation.title_zh,
            recommendation.abstract_zh,
            start_utc.isoformat(),
            end_utc.isoformat(),
        )
        for recommendation in recommendations
    ]

    with connect_recommendations_db(db_path) as conn:
        before_changes = conn.total_changes
        conn.executemany(
            """
            INSERT OR IGNORE INTO recommended_papers (
                paper_id,
                title,
                link,
                authors_json,
                categories_json,
                published_at,
                updated_at,
                recommended_at,
                score,
                matched_interests_json,
                reason,
                summary,
                title_zh,
                abstract_zh,
                query_window_start,
                query_window_end
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        inserted = conn.total_changes - before_changes
        conn.commit()

    debug_log(
        dbg,
        f"Persisted {inserted} recommendation(s) to {os.path.expanduser(db_path)}.",
    )
    return inserted
