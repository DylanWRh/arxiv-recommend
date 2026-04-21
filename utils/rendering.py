from __future__ import annotations

import datetime as dt
import os
import uuid
from html import escape

from utils.models import Recommendation
from utils.text import compact_text


def format_score(score: float) -> str:
    if score.is_integer():
        return str(int(score))
    return f"{score:.2f}"


def render_reports(
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    research_profile: str,
    recommendations: list[Recommendation],
    method_label: str,
    empty_message: str | None = None,
    history_note: str | None = None,
) -> tuple[str, str, str]:
    profile_text = compact_text(research_profile, 700)

    text_lines = [
        f"arXiv recommendations for {start_utc.isoformat()} to {end_utc.isoformat()}",
        f"Research profile: {profile_text}",
        f"Recommendation method: {method_label}",
    ]

    html_blocks = [
        "<h2>arXiv Recommendations</h2>",
        f"<p>Window: <code>{escape(start_utc.isoformat())}</code> to <code>{escape(end_utc.isoformat())}</code><br>",
        f"Research profile: {escape(profile_text)}<br>",
        f"Recommendation method: {escape(method_label)}</p>",
    ]

    markdown_lines = [
        "# arXiv Recommendations",
        "",
        f"- Window: `{start_utc.isoformat()}` to `{end_utc.isoformat()}`",
        f"- Research profile: {profile_text}",
        f"- Recommendation method: {method_label}",
    ]

    if history_note:
        text_lines.append(f"History: {history_note}")
        html_blocks.append(f"<p>History: {escape(history_note)}</p>")
        markdown_lines.append(f"- History: {history_note}")

    text_lines.append("")
    markdown_lines.append("")
    html_blocks.append("<ol>")

    if not recommendations:
        message = empty_message or "No relevant papers found in this time window."
        text_lines.append(message)
        html_blocks.append(f"<li>{escape(message)}</li>")
        markdown_lines.append(message)

    for idx, recommendation in enumerate(recommendations, start=1):
        paper = recommendation.paper
        authors_text = ", ".join(paper.authors) if paper.authors else "Unknown authors"
        matched_text = ", ".join(recommendation.matched_interests) if recommendation.matched_interests else "None"
        score_text = format_score(recommendation.score)

        text_lines.extend(
            [
                f"{idx}. {paper.title}",
                f"   Chinese title: {recommendation.title_zh}",
                f"   Authors: {authors_text}",
                f"   Published: {paper.published.isoformat()}",
                f"   Relevance score: {score_text}",
                f"   Matched concepts: {matched_text}",
                f"   Why selected: {recommendation.reason}",
                f"   Summary: {recommendation.summary}",
                f"   Chinese abstract: {recommendation.abstract_zh}",
                f"   URL: {paper.link}",
                "",
            ]
        )

        html_blocks.append(
            "<li>"
            f"<p><strong>{escape(paper.title)}</strong><br>"
            f"Chinese title: {escape(recommendation.title_zh)}<br>"
            f"Authors: {escape(authors_text)}<br>"
            f"Published: {escape(paper.published.isoformat())}<br>"
            f"Relevance score: {escape(score_text)}<br>"
            f"Matched concepts: {escape(matched_text)}<br>"
            f"Why selected: {escape(recommendation.reason)}<br>"
            f"Summary: {escape(recommendation.summary)}<br>"
            f"Chinese abstract: {escape(recommendation.abstract_zh)}<br>"
            f"URL: <a href='{escape(paper.link)}'>{escape(paper.link)}</a></p>"
            "</li>"
        )

        markdown_lines.extend(
            [
                f"## {idx}. {paper.title}",
                f"- Chinese title: {recommendation.title_zh}",
                f"- Authors: {authors_text}",
                f"- Published: {paper.published.isoformat()}",
                f"- Relevance score: {score_text}",
                f"- Matched concepts: {matched_text}",
                f"- Why selected: {recommendation.reason}",
                f"- Summary: {recommendation.summary}",
                f"- Chinese abstract: {recommendation.abstract_zh}",
                f"- URL: {paper.link}",
                "",
            ]
        )

    html_blocks.append("</ol>")
    return "\n".join(text_lines), "\n".join(html_blocks), "\n".join(markdown_lines)


def build_report_filename(
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    generated_utc: dt.datetime,
    uid: str,
) -> str:
    return (
        f"arxiv_recommendations_q{start_utc.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{end_utc.strftime('%Y%m%dT%H%M%SZ')}_"
        f"gen{generated_utc.strftime('%Y%m%dT%H%M%SZ')}_{uid}.md"
    )


def default_output_path(start_utc: dt.datetime, end_utc: dt.datetime) -> str:
    generated_utc = dt.datetime.now(dt.timezone.utc)
    uid = uuid.uuid4().hex[:8]
    filename = build_report_filename(start_utc, end_utc, generated_utc, uid)
    return os.path.join("reports", filename)


def resolve_output_path(raw_path: str, start_utc: dt.datetime, end_utc: dt.datetime) -> str:
    generated_utc = dt.datetime.now(dt.timezone.utc)
    uid = uuid.uuid4().hex[:8]
    auto_filename = build_report_filename(start_utc, end_utc, generated_utc, uid)

    cleaned = raw_path.strip()
    if not cleaned:
        return os.path.join("reports", auto_filename)

    expanded = os.path.expanduser(cleaned)
    basename = os.path.basename(expanded.rstrip("/\\"))
    ext = os.path.splitext(basename)[1]
    is_dir_hint = (
        cleaned.endswith("/")
        or cleaned.endswith("\\")
        or os.path.isdir(expanded)
        or ext == ""
    )

    if is_dir_hint:
        return os.path.join(expanded, auto_filename)

    return expanded


def save_report(path: str, markdown_report: str, start_utc: dt.datetime, end_utc: dt.datetime) -> str:
    target = resolve_output_path(path, start_utc, end_utc)
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)

    base, ext = os.path.splitext(target)
    if not ext:
        ext = ".md"
        target = base + ext

    candidate = target
    counter = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{counter}{ext}"
        counter += 1

    with open(candidate, "w", encoding="utf-8") as output_file:
        output_file.write(markdown_report)

    return candidate
