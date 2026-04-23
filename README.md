# arXiv Recommendation App

Fetch newly announced arXiv papers for a time window, ask an LLM which ones match your research profile, then:

- save a Markdown report
- send an email digest
- optionally persist Git-friendly JSON state in a separate repo under `data/`

The app stores the original arXiv abstract in JSON state, but omits it from email so messages stay shorter.

## Overview

- If `--start` and `--end` are omitted, the app uses the latest fully announced arXiv submission window.
- Flexible times like `today`, `yesterday`, `now`, and `last Friday 9am` are supported.
- Papers already present in the state store are skipped automatically.
- Recommended papers are stored with full metadata; nonrecommended papers are stored as lightweight records.

## Requirements

- Python 3.10+
- Network access to arXiv
- `OPENAI_API_KEY`
- SMTP settings only if you want email delivery

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick Start

Copy the example config:

```bash
cp .env.example .env
```

Set at minimum:

- `RESEARCH_PROFILE`
- `OPENAI_API_KEY`

Optional: clone the separate state repo into `data/`:

```bash
git clone https://github.com/DylanWRh/arxiv-recommend-state.git data
```

Then enable JSON persistence with:

```dotenv
SAVE_TO_DB=true
RECOMMENDATIONS_STATE_DIR=./data
```

## Important Config

| Variable | Required | Purpose |
| --- | --- | --- |
| `RESEARCH_PROFILE` | Yes | Research topics and goals used for ranking |
| `OPENAI_API_KEY` | Yes | API key for recommendation and summarization |
| `APP_TIMEZONE` | No | Timezone for flexible date parsing |
| `SAVE_REPORT` | No | Save a Markdown report locally; default `true` |
| `SEND_EMAIL` | No | Send email output; default `false` |
| `SAVE_TO_DB` | No | Persist JSON recommendation state; default `false` |
| `OUTPUT_PATH` | No | File or directory for saved Markdown reports |
| `EMAIL_TO` | No | Default recipient when email is enabled |
| `RECOMMENDATIONS_STATE_DIR` | If `SAVE_TO_DB=true` | Root directory for JSON state files |
| `OPENAI_MODEL` | No | Main recommendation model |
| `TIME_PARSE_MODEL` | No | Model used to normalize flexible time input |
| `OPENAI_BASE_URL` | No | Base URL for an OpenAI-compatible provider |
| `LLM_BATCH_SIZE` | No | Papers evaluated per LLM batch |
| `LLM_TIMEOUT` | No | Request timeout in seconds |

SMTP values are only needed for email:

- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USE_TLS`
- `SMTP_USER`
- `SMTP_PASS`
- `EMAIL_FROM`

Notes:

- `.env.example` defaults to an OpenRouter-style base URL. For OpenAI directly, use `OPENAI_BASE_URL=https://api.openai.com/v1`.
- `.env.example` also defaults to Gmail SMTP with STARTTLS.
- For Gmail, `SMTP_PASS` should be a Google App Password.

## Usage

Save a local report:

```bash
python app.py \
  --research-profile "I work on 3D vision, graphics, and generative modeling." \
  --start "2026-04-01" \
  --end "2026-04-02"
```

Send an email digest:

```bash
python app.py \
  --research-profile "I focus on retrieval-augmented generation and evaluation for LLM systems." \
  --start "yesterday" \
  --end "now" \
  --send-email \
  --to "you@example.com"
```

Save to the JSON state repo:

```bash
python app.py --dbg --save-to-db --state-dir ./data
```

Use `.env` defaults only:

```bash
python app.py --dbg
```

CLI highlights:

- `--research-profile`
- `--start`, `--end`, `--timezone`
- `--max-results`
- `--save-report` / `--no-save-report`
- `--send-email` / `--no-send-email`
- `--save-to-db` / `--no-save-to-db`
- `--output`, `--to`, `--state-dir`, `--dry-run`, `--dbg`
- `--llm-model`, `--time-parse-model`, `--llm-batch-size`, `--llm-timeout`

Run `python app.py --help` for the full CLI reference.

## JSON State Format

When `SAVE_TO_DB=true` or `--save-to-db` is enabled, the app writes daily files under the configured state directory:

- `recommended-papers/<yymm>/<dd>.json`
  - canonical `paper_id`
  - `paper_abstract`, `llm_summary`, `summary_zh`
  - `review_status`, which starts as `unchecked`
- `not-recommended-papers/<yymm>/<dd>.json`
  - `paper_id`, `title`, `url`

Allowed `review_status` values are:

- `unchecked`
- `interested`
- `uninterested`
- `rejected`
- `readed`

## GitHub Actions

The daily workflow lives at `.github/workflows/daily-app-run.yml`. It:

- installs Python and dependencies
- checks out the separate state repo into `data/`
- builds `.env` from Actions variables and secrets
- forces `SAVE_TO_DB=true` and `RECOMMENDATIONS_STATE_DIR=./data`
- runs `python app.py --dbg --send-email`
- pushes updated state data back to the state repo if needed

Typical Actions setup:

- Variables: `RESEARCH_PROFILE`, optionally `APP_TIMEZONE`
- Secrets: `OPENAI_API_KEY`, `EMAIL_TO`, `SMTP_USER`, `SMTP_PASS`, `EMAIL_FROM`, `STATE_REPO_TOKEN`

If you want a different run time, edit the workflow `cron` schedule in `.github/workflows/daily-app-run.yml`.

## Behavior Notes

- Local defaults are `SAVE_REPORT=true`, `SEND_EMAIL=false`, and `SAVE_TO_DB=false`.
- If no papers are found, the report says so directly.
- If no papers are recommended, the report says so directly.
- If the LLM request fails, the report is still generated without recommendations.
