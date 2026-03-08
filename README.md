# arXiv Recommendation App

This app does three things:
1. Fetches new arXiv papers for a start/end time window (defaults to today).
2. Uses an LLM to recommend relevant papers from your research-profile paragraph and summarize them.
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
- `USER_INTERESTS`: optional legacy keyword list (used as hint)
- `APP_TIMEZONE`: timezone for default date window and naive inputs
- `EMAIL_TO`: recipient email (optional)
- `OUTPUT_PATH`: report path (optional)

### LLM settings
- `OPENAI_API_KEY`: API key (required for LLM mode)
- `OPENAI_MODEL`: model name (default `gpt-4o-mini`)
- `OPENAI_BASE_URL`: API base URL (default `https://api.openai.com/v1`)
- `LLM_BATCH_SIZE`: number of papers per LLM batch (default `40`)
- `LLM_TIMEOUT`: LLM request timeout in seconds (default `60`)

### SMTP settings (only needed when sending email)
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USE_TLS`, `SMTP_USER`, `SMTP_PASS`, `EMAIL_FROM`

## Usage
### 1) Save report file (no email)

```bash
conda run -n py310 python app.py \
  --research-profile "I work on 3D vision and graphics, especially reconstruction and generative modeling with language-conditioned agents."
```

### 2) Add legacy keyword hints (optional)

```bash
conda run -n py310 python app.py \
  --research-profile "I study multimodal agents for scientific workflows." \
  --interests "agent systems, multimodal reasoning"
```

### 3) Email results

```bash
conda run -n py310 python app.py \
  --research-profile "I focus on retrieval-augmented generation and evaluation for LLM systems." \
  --to "you@example.com"
```

## CLI Options
- `--research-profile`: paragraph self-introduction for recommendation grounding
- `--interests`: legacy keyword hint list (optional)
- `--start`: ISO datetime (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM`)
- `--end`: ISO datetime (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM`)
- `--timezone`: IANA timezone, default `UTC`
- `--max-results`: max arXiv papers fetched from the window, default `200`
- `--to`: recipient email; if omitted, app saves report file
- `--output`: output report file path
- `--llm-model`: LLM model name
- `--llm-batch-size`: number of papers per LLM batch
- `--llm-timeout`: LLM timeout seconds
- `--dry-run`: no SMTP send; prints email MIME when `--to` is set

## Notes
- arXiv query uses `submittedDate` and sorts by newest first.
- Recommendations include all papers judged relevant in the fetched window.
- Every recommendation includes a reason that explains the connection to your research profile.
- If the LLM call fails, the app falls back to local keyword matching derived from your profile.
