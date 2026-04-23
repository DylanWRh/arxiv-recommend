from __future__ import annotations

import json
import re


def _clean(text: str) -> str:
    return " ".join(text.split())


def summarize_abstract(text: str, max_sentences: int = 2, max_chars: int = 380) -> str:
    clean = _clean(text)
    if not clean:
        return "No abstract available."
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    candidate = " ".join(sentences[:max_sentences]).strip()
    if not candidate:
        candidate = clean
    if len(candidate) <= max_chars:
        return candidate
    return candidate[: max_chars - 3].rstrip() + "..."


def compact_text(text: str, max_chars: int) -> str:
    clean = _clean(text)
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def normalize_str_list(value: object) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                output.append(text)
        return output
    return []


def extract_json_object(raw: str) -> dict:
    stripped = raw.strip()
    stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise ValueError("LLM response does not contain a JSON object.")
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("LLM response root must be a JSON object.")
    return parsed
