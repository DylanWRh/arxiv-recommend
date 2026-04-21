from __future__ import annotations

import datetime as dt
import os
import time
from typing import Callable, TypeVar

from dotenv import load_dotenv as _load_dotenv

T = TypeVar("T")


def int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def str_env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def debug_log(enabled: bool, message: str) -> None:
    if not enabled:
        return
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[dbg {timestamp}] {message}", flush=True)


def run_with_retries(
    fn: Callable[[], T],
    *,
    retries: int = 5,
    sleep_seconds: float = 10.0,
    dbg: bool,
    action_name: str,
) -> T:
    max_attempts = max(1, retries + 1)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_attempts:
                break
            debug_log(
                dbg,
                (
                    f"{action_name} failed on attempt {attempt}/{max_attempts} with {exc}. "
                    f"Retrying after {sleep_seconds:.1f}s."
                ),
            )
            time.sleep(max(0.0, sleep_seconds))

    if last_error is None:
        raise RuntimeError(f"{action_name} failed without an exception.")
    raise last_error


def load_dotenv(path: str = ".env") -> None:
    _load_dotenv(dotenv_path=path, override=False)
