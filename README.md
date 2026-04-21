# arXiv Recommendation App

## Introduction

This repo fetches newly submitted arXiv papers, asks an LLM which ones match your research interests, and produces a report you can save or email. It also keeps a Git-friendly JSON history so the same paper is not recommended twice.

The app repo and the state repo are split:

- this repo stores the app code and workflow
- the separate `arxiv-recommend-state` repo stores JSON state files under `data/`

Each recommended paper includes:

- a relevance score
- matched concepts from your research profile
- a short English summary
- a Chinese title
- a Chinese abstract summary
- a short explanation of why it matches your interests
- a `review_status` field for manual tracking

If you do not pass `--start` and `--end`, the app automatically queries the latest fully announced arXiv submission window.

## Setup and Requirements

### Requirements

- Python 3.10+
- Network access to arXiv
- An OpenAI-compatible API key in `OPENAI_API_KEY`
- SMTP credentials only if you want email delivery

### Install

```bash
python -m pip install -r requirements.txt
```

### Configure environment variables

Copy the example file and edit it:

```bash
cp .env.example .env
```

If you want to use the private JSON state store, clone the state repo into `data/`:

```bash
git clone https://github.com/DylanWRh/arxiv-recommend-state.git data
```

Minimum required values for a normal local run:

- `RESEARCH_PROFILE`: your research interests, background, or topics you want to track
- `OPENAI_API_KEY`: API key for the LLM

Common optional values:

- `APP_TIMEZONE`: timezone used when parsing inputs like `today`, `yesterday`, or `last Friday 9am`
- `SAVE_REPORT`: whether to save a Markdown report locally. Default: `true`
- `SEND_EMAIL`: whether to send email when running locally. Default: `false`
- `SAVE_TO_DB`: whether to record processed papers in the JSON state store. Default: `false`
- `EMAIL_TO`: default email recipient
- `OUTPUT_PATH`: default output path when saving a report
- `RECOMMENDATIONS_STATE_DIR`: directory used to store recommendation state JSON files. Required when `SAVE_TO_DB=true`
  Recommended papers are stored under `recommended-papers/<yymm>/<dd>.json`, one JSON per day.
  Unrecommended papers are stored under `not-recommended-papers/<yymm>/<dd>.json`, one JSON per day.
- `OPENAI_MODEL`: recommendation model
- `TIME_PARSE_MODEL`: model used to normalize flexible time expressions
- `OPENAI_BASE_URL`: API base URL for an OpenAI-compatible provider
- `LLM_BATCH_SIZE`: number of papers evaluated per LLM batch
- `LLM_TIMEOUT`: request timeout in seconds

SMTP values are only needed when sending email:

- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USE_TLS`
- `SMTP_USER`
- `SMTP_PASS`
- `EMAIL_FROM`

Notes:

- `.env.example` currently uses an OpenRouter-style base URL. If you use OpenAI directly, set `OPENAI_BASE_URL=https://api.openai.com/v1`.
- `.env.example` also defaults to Gmail SMTP: `smtp.gmail.com:587` with STARTTLS.
- For Gmail, `SMTP_PASS` should be a Google App Password, not your normal account password.

### Private database directory

Database writes are disabled by default for local runs.

To enable the private state store locally:

1. Clone your private state repo somewhere local, for example `./data`.
2. Set `SAVE_TO_DB=true`.
3. Set `RECOMMENDATIONS_STATE_DIR` to that private repo path.

Example:

```bash
git clone https://github.com/DylanWRh/arxiv-recommend-state.git data
```

```dotenv
SAVE_TO_DB=true
RECOMMENDATIONS_STATE_DIR=./data
```

You can also enable it per run:

```bash
python app.py --save-to-db --state-dir ./data
```

## GitHub Actions Daily Recommendation Setup

The workflow file is `.github/workflows/daily-app-run.yml`.

What it does:

- runs every day on the GitHub Actions schedule
- can also be started manually with `workflow_dispatch`
- installs Python 3.11 and dependencies
- checks out the separate state repo into `data/`
- builds a `.env` file from GitHub Actions variables and secrets
- forces `SAVE_TO_DB=true` and `RECOMMENDATIONS_STATE_DIR=./data`
- runs `python app.py --dbg --send-email`
- commits updated `data/` contents back to the state repo if they changed

### Daily execution time

The current schedule is:

```yaml
cron: '5 12 * * *'
```

GitHub Actions cron uses UTC. This workflow currently runs every day at `12:05 UTC`.

If you want a different time, edit the cron line in `.github/workflows/daily-app-run.yml`.

### How to configure Actions variables and secrets

In your GitHub repository:

1. Open `Settings`.
2. Go to `Secrets and variables` -> `Actions`.
3. Add the following values.

Recommended repository variables:

| Name | Required | Purpose |
| --- | --- | --- |
| `RESEARCH_PROFILE` | Yes | The text used to decide which papers are relevant |
| `APP_TIMEZONE` | No | Timezone for flexible time parsing |
| `OPENAI_MODEL` | No | Recommendation model |
| `TIME_PARSE_MODEL` | No | Model used for parsing flexible time expressions |
| `OPENAI_BASE_URL` | No | Base URL for your OpenAI-compatible provider |
| `LLM_BATCH_SIZE` | No | Papers per LLM batch |
| `LLM_TIMEOUT` | No | LLM timeout in seconds |
| `SMTP_HOST` | No | SMTP server host |
| `SMTP_PORT` | No | SMTP server port |
| `SMTP_USE_TLS` | No | `true` or `false` |
| `OUTPUT_PATH` | No | Optional report output path |

Required repository secrets for the current workflow:

| Name | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | API key for the LLM provider |
| `STATE_REPO_TOKEN` | Yes | Token used to clone and push `DylanWRh/arxiv-recommend-state` |
| `EMAIL_TO` | Yes | Recipient email address for the daily report |
| `SMTP_USER` | Yes | SMTP login username |
| `SMTP_PASS` | Yes | SMTP login password or app password |
| `EMAIL_FROM` | Yes | Sender address shown in the email |

Practical notes:

- `EMAIL_TO` is the receiver of the report.
- `STATE_REPO_TOKEN` should have access to the separate state repo.
- `EMAIL_FROM` is the sender shown in the email.
- `EMAIL_FROM` is often the same as `SMTP_USER`, but that depends on your mail provider.
- If you use Gmail, you can usually keep `SMTP_HOST=smtp.gmail.com`, `SMTP_PORT=587`, and `SMTP_USE_TLS=true`.
- The workflow checks out `DylanWRh/arxiv-recommend-state` into `data/`, then copies `.env.example`, sets `SAVE_TO_DB=true`, and writes `RECOMMENDATIONS_STATE_DIR=./data`.

### Recommended Actions setup for Gmail

Variables:

- `RESEARCH_PROFILE`
- `APP_TIMEZONE=Asia/Shanghai` or your own timezone

Secrets:

- `OPENAI_API_KEY`
- `EMAIL_TO=you@example.com`
- `SMTP_USER=your-gmail-address@gmail.com`
- `SMTP_PASS=your-google-app-password`
- `EMAIL_FROM=your-gmail-address@gmail.com`

You can leave `SMTP_HOST`, `SMTP_PORT`, and `SMTP_USE_TLS` unset if you want to use the Gmail defaults from `.env.example`.

### Test the workflow

After saving the variables and secrets:

1. Open the `Actions` tab.
2. Open `Daily app run`.
3. Click `Run workflow`.
4. Check the logs to confirm that the app fetched papers and sent the email.

## Usage

CLI arguments override values from `.env`. If you run `python app.py` without `--start` or `--end`, the app uses the latest fully announced arXiv window.

### Save a report to a file

```bash
python app.py \
  --research-profile "I work on 3D vision, graphics, and generative modeling." \
  --start "2026-04-01" \
  --end "2026-04-02"
```

If `--to` is not set, the app saves a Markdown report under `reports/` by default.

### Send the report by email

```bash
python app.py \
  --research-profile "I focus on retrieval-augmented generation and evaluation for LLM systems." \
  --start "yesterday" \
  --end "now" \
  --send-email \
  --to "you@example.com"
```

### Use only `.env` values

```bash
python app.py --dbg
```

### CLI options

| Option | Description |
| --- | --- |
| `--research-profile` | Research profile text used for recommendation |
| `--start` | Flexible start time such as `2026-04-01`, `yesterday`, or `last Friday 9am` |
| `--end` | Flexible end time such as `now`, `today`, or `2026-04-02 23:59` |
| `--timezone` | Timezone used for parsing flexible times. Defaults to `APP_TIMEZONE` or `UTC` |
| `--max-results` | Maximum number of arXiv papers to fetch. Default: `2000` |
| `--to` | Email recipient used when email sending is enabled |
| `--output` | Output file path or directory for the Markdown report |
| `--save-report`, `--no-save-report` | Enable or disable saving the Markdown report. Default: `true` |
| `--send-email`, `--no-send-email` | Enable or disable sending email. Default: `false` |
| `--save-to-db`, `--no-save-to-db` | Enable or disable writing processed papers to the JSON state store. Default: `false` |
| `--state-dir` | Directory used to persist `recommended-papers/<yymm>/<dd>.json` and `not-recommended-papers/<yymm>/<dd>.json` |
| `--llm-model` | Model used for recommendation and summarization |
| `--time-parse-model` | Model used to normalize flexible time expressions |
| `--llm-batch-size` | Number of papers evaluated per LLM batch |
| `--llm-timeout` | Timeout for LLM requests in seconds |
| `--dry-run` | Builds the result but does not send email. If email sending is enabled, prints the MIME email instead |
| `--dbg` | Prints debug logs |

### Behavior summary

- arXiv papers are fetched with `submittedDate` and sorted newest first.
- Time expressions like `today`, `yesterday`, and `now` are supported.
- The app creates a JSON state store automatically and skips papers already saved there.
- Recommended papers are grouped into one daily JSON file under `recommended-papers/<yymm>/<dd>.json`.
- Unrecommended papers are grouped into one daily JSON file under `not-recommended-papers/<yymm>/<dd>.json`.
- Each stored JSON uses the canonical arXiv identifier in `paper_id`, not the full arXiv URL.
- Recommended papers are stored with full recommendation detail.
- Recommended papers also include `review_status`, which starts as `unchecked`.
- Allowed `review_status` values are `rejected`, `uninterested`, `unchecked`, `interested`, and `readed`.
- Nonrecommended paper entries store only `paper_id`, `title`, and `url`.
- Local default behavior is: save report `true`, send email `false`, save to DB `false`.
- If `SAVE_TO_DB=true`, you must also set `RECOMMENDATIONS_STATE_DIR` or pass `--state-dir`.
- The GitHub Actions workflow overrides that default, enables database saving, checks out the private state repo into `data/`, and pushes the updated `data/` contents back to that repo.
- If no papers are found, the report says so directly.
- If the LLM finds no relevant papers, the report says so directly.
- If the LLM request fails, the report is still generated, but without recommendations.
