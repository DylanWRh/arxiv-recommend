from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


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
