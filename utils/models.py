from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: dt.datetime
    updated: dt.datetime
    link: str


@dataclass
class Recommendation:
    paper: Paper
    title_zh: str
    summary_zh: str
    score: float
    matched_interests: list[str]
    llm_summary: str
    reason: str
