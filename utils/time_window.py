from __future__ import annotations

import datetime as dt
import json
import re
import sys
from zoneinfo import ZoneInfo

import requests

from utils.runtime import debug_log, run_with_retries, str_env
from utils.text import extract_json_object

ARXIV_ANNOUNCEMENT_TZ = "America/New_York"
ARXIV_ANNOUNCEMENT_HOUR = 20
ARXIV_SUBMISSION_CUTOFF_HOUR = 14


def _day_edge(base: dt.datetime, for_end: bool) -> dt.datetime:
    if for_end:
        return base.replace(hour=23, minute=59, second=59, microsecond=0)
    return base.replace(hour=0, minute=0, second=0, microsecond=0)


def _parse_formats(raw: str, formats: list[str], tz: ZoneInfo) -> dt.datetime | None:
    for fmt in formats:
        try:
            return dt.datetime.strptime(raw, fmt).replace(tzinfo=tz)
        except ValueError:
            continue
    return None


def _parse_time(
    raw: str | None,
    tz: ZoneInfo,
    now_local: dt.datetime,
    for_end: bool = False,
) -> dt.datetime | None:
    if raw is None:
        return None

    normalized = re.sub(r"\s+", " ", raw.strip())
    if not normalized:
        return None

    low = normalized.lower()
    if low in {"now", "right now"}:
        return now_local
    if low == "today":
        return _day_edge(now_local, for_end)
    if low == "yesterday":
        return _day_edge(now_local - dt.timedelta(days=1), for_end)
    if low == "tomorrow":
        return _day_edge(now_local + dt.timedelta(days=1), for_end)

    cleaned = re.sub(
        r"^(from|since|start|starting|begin|beginning|to|until|till|end|ending)\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned.replace(",", " ").replace(" at ", " ")).strip()

    iso_candidate = cleaned.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(iso_candidate)
    except ValueError:
        parsed = None

    if parsed is not None:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz)
        return parsed.astimezone(tz)

    datetime_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y.%m.%d %H:%M:%S",
        "%Y.%m.%d %H:%M",
        "%Y%m%d%H%M",
        "%Y%m%d %H%M",
        "%d %b %Y %H:%M",
        "%d %B %Y %H:%M",
        "%b %d %Y %H:%M",
        "%B %d %Y %H:%M",
    ]
    parsed = _parse_formats(cleaned, datetime_formats, tz)
    if parsed is not None:
        return parsed

    date_formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y%m%d",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
    ]
    parsed = _parse_formats(cleaned, date_formats, tz)
    if parsed is not None:
        return _day_edge(parsed, for_end)

    return None


def _llm_parse_window(
    start_raw: str | None,
    end_raw: str | None,
    tz_name: str,
    reference_now: dt.datetime,
    llm_model: str,
    llm_timeout: int,
    dbg: bool = False,
) -> tuple[str | None, str | None]:
    if not (start_raw or end_raw):
        return None, None

    api_key = str_env("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set for time parsing.")

    base_url = str_env("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    endpoint = f"{base_url}/chat/completions"

    system_prompt = (
        "You normalize user-provided time expressions to strict ISO 8601 datetimes. "
        "Use the provided reference_now as the authoritative current time. "
        "Return JSON only with keys start and end."
    )
    user_payload = {
        "timezone": tz_name,
        "reference_now": reference_now.isoformat(),
        "reference_date": reference_now.strftime("%Y-%m-%d"),
        "reference_weekday": reference_now.strftime("%A"),
        "reference_unix": int(reference_now.timestamp()),
        "reference_utc_offset": reference_now.strftime("%z"),
        "start_input": start_raw,
        "end_input": end_raw,
        "rules": [
            "Output format: YYYY-MM-DDTHH:MM:SS+HH:MM.",
            "If input is date-only and refers to start, set time to 00:00:00.",
            "If input is date-only and refers to end, set time to 23:59:59.",
            "Resolve relative terms (today, yesterday, last Friday, now) using reference_now and timezone.",
            "If a side is missing, return null for that side.",
        ],
    }

    request_payload = {
        "model": llm_model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }

    def _request_and_parse() -> dict:
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
            raise ValueError("Time parser LLM response has no choices.")

        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Time parser LLM response has no text content.")
        return extract_json_object(content)

    debug_log(dbg, f"Normalizing time window via LLM model={llm_model}.")
    parsed = run_with_retries(
        _request_and_parse,
        dbg=dbg,
        action_name="Time parser LLM call",
    )

    start_value = parsed.get("start")
    end_value = parsed.get("end")

    normalized_start = str(start_value).strip() if start_value is not None else None
    normalized_end = str(end_value).strip() if end_value is not None else None

    if normalized_start == "":
        normalized_start = None
    if normalized_end == "":
        normalized_end = None

    debug_log(
        dbg,
        (
            "LLM time parsing completed with "
            f"start={normalized_start or 'null'} end={normalized_end or 'null'}."
        ),
    )
    return normalized_start, normalized_end


def _latest_arxiv_window(
    now_utc: dt.datetime | None = None,
) -> tuple[dt.datetime, dt.datetime]:
    if now_utc is None:
        now_utc = dt.datetime.now(dt.timezone.utc)

    announcement_tz = ZoneInfo(ARXIV_ANNOUNCEMENT_TZ)
    now_local = now_utc.astimezone(announcement_tz)
    latest_announcement: dt.datetime | None = None

    for days_back in range(0, 8):
        candidate_date = now_local.date() - dt.timedelta(days=days_back)
        candidate = dt.datetime.combine(
            candidate_date,
            dt.time(hour=ARXIV_ANNOUNCEMENT_HOUR),
            tzinfo=announcement_tz,
        )
        if candidate > now_local:
            continue
        if candidate.weekday() not in {6, 0, 1, 2, 3}:
            continue
        latest_announcement = candidate
        break

    if latest_announcement is None:
        raise RuntimeError("Could not determine the latest arXiv announcement time.")

    if latest_announcement.weekday() == 6:
        start_date = latest_announcement.date() - dt.timedelta(days=3)
        end_date = latest_announcement.date() - dt.timedelta(days=2)
    elif latest_announcement.weekday() == 0:
        start_date = latest_announcement.date() - dt.timedelta(days=3)
        end_date = latest_announcement.date()
    else:
        start_date = latest_announcement.date() - dt.timedelta(days=1)
        end_date = latest_announcement.date()

    start_local = dt.datetime.combine(
        start_date,
        dt.time(hour=ARXIV_SUBMISSION_CUTOFF_HOUR),
        tzinfo=announcement_tz,
    )
    end_local = dt.datetime.combine(
        end_date,
        dt.time(hour=ARXIV_SUBMISSION_CUTOFF_HOUR),
        tzinfo=announcement_tz,
    )
    return start_local.astimezone(dt.timezone.utc), end_local.astimezone(dt.timezone.utc)


def compute_time_window(
    start_raw: str | None,
    end_raw: str | None,
    tz_name: str,
    time_parse_model: str,
    llm_timeout: int,
    dbg: bool = False,
) -> tuple[dt.datetime, dt.datetime]:
    tz = ZoneInfo(tz_name)
    now_local = dt.datetime.now(tz)

    llm_start_raw: str | None = None
    llm_end_raw: str | None = None
    if (start_raw and start_raw.strip()) or (end_raw and end_raw.strip()):
        try:
            llm_start_raw, llm_end_raw = _llm_parse_window(
                start_raw=start_raw,
                end_raw=end_raw,
                tz_name=tz_name,
                reference_now=now_local,
                llm_model=time_parse_model,
                llm_timeout=llm_timeout,
                dbg=dbg,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"LLM time parsing failed: {exc}. Falling back to local parsing.", file=sys.stderr)

    start_source = llm_start_raw if llm_start_raw is not None else start_raw
    end_source = llm_end_raw if llm_end_raw is not None else end_raw

    start_local = _parse_time(start_source, tz, now_local, for_end=False)
    end_local = _parse_time(end_source, tz, now_local, for_end=True)

    if start_local is None and end_local is None:
        start_utc, end_utc = _latest_arxiv_window()
        debug_log(
            dbg,
            (
                "No explicit window provided; using the latest fully announced arXiv window "
                f"{start_utc.isoformat()} -> {end_utc.isoformat()}."
            ),
        )
        return start_utc, end_utc

    if end_local is None:
        end_local = now_local
    if start_local is None:
        start_local = end_local - dt.timedelta(hours=24)

    if end_local <= start_local:
        raise ValueError("End time must be after start time.")

    debug_log(
        dbg,
        (
            "Resolved query window to "
            f"{start_local.astimezone(dt.timezone.utc).isoformat()} -> "
            f"{end_local.astimezone(dt.timezone.utc).isoformat()}."
        ),
    )
    return start_local.astimezone(dt.timezone.utc), end_local.astimezone(dt.timezone.utc)
