from __future__ import annotations

import argparse
import sys

from utils.arxiv_client import fetch_arxiv_papers
from utils.database import (
    exclude_saved_papers,
    load_saved_paper_ids,
    resolve_state_dir,
    save_processed_papers_state,
)
from utils.emailing import build_email, send_email
from utils.llm_recommender import recommend_and_summarize
from utils.rendering import render_reports, save_report
from utils.runtime import bool_env, debug_log, int_env, load_dotenv, str_env
from utils.time_window import compute_time_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch new arXiv papers, recommend by research profile, generate LLM summaries, and email/save results."
    )
    parser.add_argument(
        "--research-profile",
        help="Paragraph describing your research background, topics, and goals.",
        default=str_env("RESEARCH_PROFILE", ""),
    )
    parser.add_argument(
        "--start",
        help="Flexible start time (e.g. 2026.03.04, March 4 2026, last Friday 9am). If both --start/--end are omitted, defaults to the latest fully announced arXiv submission window.",
    )
    parser.add_argument(
        "--end",
        help="Flexible end time (e.g. now, today 23:59, 2026/03/08 18:30). If both --start/--end are omitted, defaults to the latest fully announced arXiv submission window.",
    )
    parser.add_argument(
        "--timezone",
        help="Timezone name used for parsing explicit relative times (e.g. UTC, America/New_York).",
        default=str_env("APP_TIMEZONE", "UTC"),
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=2000,
        help="Maximum number of arXiv papers to fetch from the date window.",
    )
    parser.add_argument(
        "--to",
        help="Recipient email. Used when email sending is enabled. Defaults to EMAIL_TO from environment.",
        default=str_env("EMAIL_TO", ""),
    )
    parser.add_argument(
        "--output",
        help="Output file path or directory for the saved Markdown report.",
        default=str_env("OUTPUT_PATH", ""),
    )
    parser.add_argument(
        "--save-report",
        action=argparse.BooleanOptionalAction,
        default=bool_env("SAVE_REPORT", True),
        help="Whether to save a Markdown report to disk.",
    )
    parser.add_argument(
        "--send-email",
        action=argparse.BooleanOptionalAction,
        default=bool_env("SEND_EMAIL", False),
        help="Whether to send the recommendation email.",
    )
    parser.add_argument(
        "--save-to-db",
        action=argparse.BooleanOptionalAction,
        default=bool_env("SAVE_TO_DB", False),
        help="Whether to persist processed papers to the JSON state store.",
    )
    parser.add_argument(
        "--state-dir",
        dest="state_dir",
        help="Directory used to store daily JSON state files under recommended/not-recommended YYMM folders.",
        default=str_env("RECOMMENDATIONS_STATE_DIR", ""),
    )
    parser.add_argument(
        "--llm-model",
        help="LLM model used for recommendation and generated summaries.",
        default=str_env("OPENAI_MODEL", "gpt-4o-mini"),
    )
    parser.add_argument(
        "--time-parse-model",
        help="LLM model used to normalize flexible time expressions.",
        default=str_env("TIME_PARSE_MODEL", str_env("OPENAI_MODEL", "gpt-4o-mini")),
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=int_env("LLM_BATCH_SIZE", 40),
        help="Number of papers per LLM batch when evaluating all fetched papers.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=int_env("LLM_TIMEOUT", 60),
        help="Timeout (seconds) for LLM API requests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do everything except SMTP send. If email sending is enabled, print the MIME email instead.",
    )
    parser.add_argument(
        "--dbg",
        action="store_true",
        help="Print progress logs for debugging long-running stages.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    debug_log(args.dbg, "Loaded configuration and parsed CLI arguments.")

    research_profile = " ".join(args.research_profile.split())
    if not research_profile:
        print("No research profile provided. Set --research-profile or RESEARCH_PROFILE.", file=sys.stderr)
        return 1
    if not args.save_report and not args.send_email and not args.save_to_db:
        print("No output actions are enabled. Enable at least one of --save-report, --send-email, or --save-to-db.")
        return 1

    raw_state_dir = args.state_dir.strip()
    if args.save_to_db and not raw_state_dir:
        print(
            "Database saving is enabled but no state directory was configured. "
            "Set --state-dir or RECOMMENDATIONS_STATE_DIR.",
            file=sys.stderr,
        )
        return 1

    state_dir: str | None = None
    saved_paper_ids: set[str] = set()
    if raw_state_dir:
        state_dir = resolve_state_dir(raw_state_dir)
        try:
            saved_paper_ids = load_saved_paper_ids(state_dir, dbg=args.dbg)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to open recommendations state store: {exc}", file=sys.stderr)
            return 1
    else:
        debug_log(args.dbg, "No recommendations state directory configured; skipping state-based filtering.")

    llm_batch_size = max(1, args.llm_batch_size)
    llm_timeout = max(5, args.llm_timeout)
    debug_log(
        args.dbg,
        (
            "Starting run with "
            f"max_results={args.max_results}, save_report={args.save_report}, "
            f"send_email={args.send_email}, save_to_db={args.save_to_db}, dry_run={args.dry_run}."
        ),
    )
    try:
        start_utc, end_utc = compute_time_window(
            start_raw=args.start,
            end_raw=args.end,
            tz_name=args.timezone,
            time_parse_model=args.time_parse_model,
            llm_timeout=llm_timeout,
            dbg=args.dbg,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Invalid time settings: {exc}", file=sys.stderr)
        return 1

    try:
        papers = fetch_arxiv_papers(start_utc, end_utc, args.max_results, dbg=args.dbg)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to fetch arXiv papers: {exc}", file=sys.stderr)
        return 1

    fetched_paper_count = len(papers)
    papers, skipped_saved_count = exclude_saved_papers(papers, saved_paper_ids)
    history_note: str | None = None
    if skipped_saved_count:
        history_note = (
            f"Skipped {skipped_saved_count} paper(s) that were already saved in the recommendations state store."
        )
        debug_log(
            args.dbg,
            (
                f"Filtered out {skipped_saved_count} previously saved paper(s); "
                f"{len(papers)} candidate paper(s) remain for recommendation."
            ),
        )

    recommendations, method_label, empty_message = recommend_and_summarize(
        papers=papers,
        research_profile=research_profile,
        llm_model=args.llm_model,
        llm_batch_size=llm_batch_size,
        llm_timeout=llm_timeout,
        dbg=args.dbg,
    )
    debug_log(
        args.dbg,
        f"Initial recommendation stage completed with {len(recommendations)} item(s) using method={method_label}.",
    )
    if fetched_paper_count > 0 and not papers and skipped_saved_count == fetched_paper_count:
        empty_message = (
            f"All {fetched_paper_count} paper(s) in this time window were already saved in the recommendations state store."
        )

    text_report, html_report, markdown_report = render_reports(
        start_utc=start_utc,
        end_utc=end_utc,
        research_profile=research_profile,
        recommendations=recommendations,
        method_label=method_label,
        empty_message=empty_message,
        history_note=history_note,
    )
    debug_log(args.dbg, "Rendered text, HTML, and markdown reports.")

    saved_path: str | None = None
    if args.save_report:
        try:
            saved_path = save_report(args.output, markdown_report, start_utc, end_utc)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save report: {exc}", file=sys.stderr)
            return 1
        debug_log(args.dbg, f"Saved report to {saved_path}.")
        print(f"Saved report to {saved_path}.")

    if args.send_email:
        if not args.to:
            print("Email sending is enabled but no recipient was provided. Set --to or EMAIL_TO.", file=sys.stderr)
            return 1

        debug_log(args.dbg, "Building email message.")
        message = build_email(args.to, start_utc, end_utc, text_report, html_report)

        if args.dry_run:
            debug_log(args.dbg, "Dry-run enabled; printing MIME message instead of sending.")
            print(message)
        else:
            try:
                send_email(message, dbg=args.dbg)
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to send email: {exc}", file=sys.stderr)
                return 1
            debug_log(args.dbg, f"Email send completed with {len(recommendations)} recommendation(s).")
            print(f"Sent report to {args.to} with {len(recommendations)} recommendation(s).")

    if args.save_to_db:
        assert state_dir is not None
        try:
            save_processed_papers_state(
                state_dir=state_dir,
                processed_papers=papers,
                recommendations=recommendations,
                start_utc=start_utc,
                end_utc=end_utc,
                dbg=args.dbg,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to update recommendations state store: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
