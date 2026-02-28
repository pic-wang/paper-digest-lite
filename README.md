# Paper Digest Agent

A lightweight, single-file agent that delivers a daily arXiv paper digest to your inbox.

**arXiv fetch → Gemini scoring → deep summary → HTML email. One file. Zero infrastructure.**

```
main.py        ← entire agent (~200 lines)
config.yaml    ← your keywords
```

## Quick Start

### 1. Get Keys

| Secret | How |
|--------|-----|
| `GEMINI_API_KEY` | https://aistudio.google.com (free) |
| `EMAIL_PASSWORD` | Google Account → Security → 2-Step Verification → App Passwords |

### 2. Local Test

```bash
pip install -r requirements.txt

export GEMINI_API_KEY="xxx"
export EMAIL_SENDER="you@gmail.com"
export EMAIL_PASSWORD="xxxx xxxx xxxx xxxx"
export EMAIL_RECIPIENT="you@gmail.com"

python main.py --dry-run   # fetch + rank only, no email
python main.py             # full run
```

### 3. Deploy (GitHub Actions)

Add 4 secrets in **Settings → Secrets → Actions**:

`GEMINI_API_KEY` · `EMAIL_SENDER` · `EMAIL_PASSWORD` · `EMAIL_RECIPIENT`

Push and it runs daily at 9:00 AM Beijing time. Manual trigger available in Actions tab.

### 4. Customize

Edit `config.yaml` — change `keywords`, `categories`, or `language`.

## How It Works

```
arXiv API  ──→  Gemini relevance filter (1-5)  ──→  Gemini deep summary  ──→  HTML email
 50 papers         keep score ≥ 3                    top 5 papers              your inbox
```

## Stack

- **arXiv API** — paper fetching
- **Gemini 2.0 Flash** — scoring + summarization (free tier, ~10 calls/day)
- **Gmail SMTP** — email delivery
- **GitHub Actions** — daily cron
