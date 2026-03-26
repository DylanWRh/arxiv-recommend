# arXiv Recommendation App

This app does three things:
1. Fetches new arXiv papers for a start/end time window.
2. Uses an LLM to recommend relevant papers from your research-profile paragraph, summarize them, and include Chinese title/abstract fields.
3. Includes explicit "why this matches your research" reasons, then emails results or saves a report file.

## Requirements
- Conda environment: `py310`
- Python 3.10+

## Setup
Install dependencies in `py310`:

```bash
conda run -n py310 python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` (or set environment variables directly).

## Key Environment Variables
- `RESEARCH_PROFILE`: paragraph self-introduction of your research interests
- `APP_TIMEZONE`: timezone for default date window and relative time resolution
- `EMAIL_TO`: recipient email (optional). For GitHub Actions, keep a personal address in a repository or environment secret instead of a variable.
- `OUTPUT_PATH`: report file path or directory (optional, default `./reports`)

### LLM settings
- `OPENAI_API_KEY`: API key (required for LLM mode)
- `OPENAI_MODEL`: model for recommendation/summarization
- `TIME_PARSE_MODEL`: model for flexible time parsing
- `OPENAI_BASE_URL`: API base URL (default `https://api.openai.com/v1`)
- `LLM_BATCH_SIZE`: number of papers per LLM batch (default `40`)
- `LLM_TIMEOUT`: LLM request timeout in seconds (default `60`)

### SMTP settings (only needed when sending email)
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USE_TLS`, `SMTP_USER`, `SMTP_PASS`, `EMAIL_FROM`
- For GitHub Actions, keep `SMTP_USER`, `SMTP_PASS`, and `EMAIL_FROM` in repository or environment secrets.
- `.env.example` defaults to Gmail SMTP: `smtp.gmail.com:587` with STARTTLS, and `SMTP_PASS` should be a Google App Password.

## Flexible Time Parsing
You can pass natural time expressions for `--start` and `--end`.
If both are omitted, the app defaults to yesterday in your timezone (`start=yesterday 00:00:00`, `end=yesterday 23:59:59`). If that raw run returns no recommendations, the app auto-backfills recent windows (2/3/5/7/14 days) to reduce empty daily output.
The app calls an LLM to normalize them into ISO datetimes and provides explicit context:
- current reference timestamp (`reference_now`)
- reference date and weekday
- timezone and UTC offset

Examples:
- `--start "2026.03.04" --end "today"`
- `--start "last Friday 9am" --end "now"`
- `--start "March 1, 2026" --end "March 7, 2026 23:59"`

## Usage
### 1) Save report file (no email)
By default, reports are saved under `./reports` using a robust filename with query window, generation time, and short UID.

```bash
conda run -n py310 python app.py \
  --research-profile "I work on 3D vision and graphics, especially reconstruction and generative modeling with language-conditioned agents." \
  --start "2026.03.04" \
  --end "today"
```

### 2) Email results

```bash
conda run -n py310 python app.py \
  --research-profile "I focus on retrieval-augmented generation and evaluation for LLM systems." \
  --start "yesterday" \
  --end "now" \
  --to "you@example.com"
```

## CLI Options
- `--research-profile`: paragraph self-introduction for recommendation grounding
- `--start`: flexible start time expression (default when both omitted: yesterday 00:00:00 in `--timezone`)
- `--end`: flexible end time expression (default when both omitted: yesterday 23:59:59 in `--timezone`)
- `--timezone`: IANA timezone, default `UTC`
- `--max-results`: max arXiv papers fetched from the window, default `200`
- `--to`: recipient email; if omitted, app saves report file
- `--output`: output report file path or directory
- `--llm-model`: LLM model for recommendations
- `--time-parse-model`: LLM model for time parsing
- `--llm-batch-size`: number of papers per LLM batch
- `--llm-timeout`: LLM timeout seconds
- `--dry-run`: no SMTP send; prints email MIME when `--to` is set
- `--dbg`: print stage-by-stage debug progress logs

## Notes
- arXiv query uses `submittedDate` and sorts by newest first.
- Recommendations include all papers judged relevant in the fetched window, and each recommendation includes `Chinese title` and `Chinese abstract` fields.
- Every recommendation includes a reason that explains the connection to your research profile.
- If LLM time parsing fails, the app falls back to local flexible parsing for common formats.
- If recommendation LLM calls fail (or return no matches) during raw `python app.py`, the app uses auto-backfill and then a recent-paper fallback mode.
- Default report filename format is `arxiv_recommendations_q<start>_<end>_gen<time>_<uid>.md`.





