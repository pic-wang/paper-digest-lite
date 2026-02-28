"""
Paper Digest Agent — single-file daily arXiv paper digest.

Usage:
    python main.py              # full run, sends email
    python main.py --dry-run    # fetch + rank only, no email
"""

import argparse
import json
import logging
import os
import smtplib
import time
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import arxiv
import yaml
from google import genai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("digest")

# ─── Gemini ─────────────────────────────────────────────────────────
MODEL = "gemini-2.0-flash"
GEMINI_CLIENT = None


def get_gemini():
    global GEMINI_CLIENT
    if GEMINI_CLIENT is None:
        GEMINI_CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return GEMINI_CLIENT


def ask_gemini(prompt: str, retries: int = 3) -> str:
    """Call Gemini with automatic retry."""
    for i in range(retries):
        try:
            resp = get_gemini().models.generate_content(model=MODEL, contents=prompt)
            return resp.text
        except Exception as e:
            log.warning("Gemini call failed (attempt %d): %s", i + 1, e)
            if i < retries - 1:
                time.sleep(2 ** (i + 1))
    return ""


# ─── Step 1: Fetch arXiv papers ─────────────────────────────────────
def fetch_papers(cfg: dict) -> list[dict]:
    cats = cfg["categories"]
    kws = cfg["keywords"]
    exclude = cfg.get("exclude_keywords", [])
    max_fetch = cfg.get("max_fetch", 50)

    # Fetch 3 days on Monday to cover weekends
    days_back = 3 if datetime.now(timezone.utc).weekday() == 0 else 1
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    # Build query: (cat:A OR cat:B) AND (ti/abs:"kw1" OR ...)
    cat_q = " OR ".join(f"cat:{c}" for c in cats)
    kw_q = " OR ".join(f'ti:"{k}" OR abs:"{k}"' for k in kws)
    query = f"({cat_q}) AND ({kw_q})"
    log.info("arXiv query: %s", query[:120] + "...")

    client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_fetch * 2,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for r in client.results(search):
        pub = r.published.replace(tzinfo=timezone.utc)
        if pub < cutoff:
            continue
        text = (r.title + " " + r.summary).lower()
        if any(ex.lower() in text for ex in exclude):
            continue
        papers.append({
            "id": r.entry_id.split("/abs/")[-1],
            "title": r.title.strip().replace("\n", " "),
            "authors": ", ".join(a.name for a in r.authors[:5]),
            "abstract": r.summary.strip().replace("\n", " "),
            "url": r.entry_id,
        })
        if len(papers) >= max_fetch:
            break

    log.info("Fetched %d papers", len(papers))
    return papers


# ─── Step 2: LLM relevance scoring ──────────────────────────────────
RANK_PROMPT = """You are a research assistant specializing in AI for PDE / Physics Simulation.
Rate each paper's relevance (1-5) to these topics:
- Neural PDE solvers (PINN, Neural Operator, FNO, DeepONet)
- ML-accelerated physics simulation (learned simulation, surrogate model)
- Scientific computing x deep learning (differentiable physics, hybrid solver)
- AI methods for CFD
- Domain decomposition x neural networks

Output strict JSON only:
```json
[{{"id": "...", "score": 5, "reason": "one-line reason"}}]
```

Papers:
{papers}"""


def rank_papers(papers: list[dict], max_digest: int = 5) -> list[dict]:
    text = "\n".join(
        f"---\nID: {p['id']}\nTitle: {p['title']}\nAbstract: {p['abstract'][:300]}"
        for p in papers
    )
    raw = ask_gemini(RANK_PROMPT.format(papers=text))
    if not raw:
        log.error("Scoring failed, keeping first %d papers", max_digest)
        for p in papers:
            p["score"], p["reason"] = 3, "scoring failed"
        return papers[:max_digest]

    # Parse JSON
    try:
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0]
        scores = {s["id"]: s for s in json.loads(cleaned)}
    except (json.JSONDecodeError, KeyError):
        log.error("JSON parse failed, keeping first %d papers", max_digest)
        for p in papers:
            p["score"], p["reason"] = 3, "parse failed"
        return papers[:max_digest]

    for p in papers:
        info = scores.get(p["id"], {})
        p["score"] = info.get("score", 1)
        p["reason"] = info.get("reason", "")

    result = sorted(
        [p for p in papers if p["score"] >= 3],
        key=lambda x: x["score"],
        reverse=True,
    )
    log.info("Scoring done: %d/%d papers passed threshold", len(result), len(papers))
    return result[:max_digest]


# ─── Step 3: Summarize each paper ───────────────────────────────────
SUMMARY_PROMPT = """You are a senior researcher in AI for PDE / Scientific Computing.
Provide a deep summary of this paper in {lang}, using this Markdown structure:

### One-line Summary
### Core Method (3-5 sentences: input/output, innovation, difference from PINN/FNO)
### Key Results (benchmarks, metrics, comparison with baselines)
### Highlights & Limitations (1-2 each)
### Relevance to Urban Wind / CFD (if none, say "weak relevance")

Paper:
- Title: {title}
- Authors: {authors}
- Abstract: {abstract}
- arXiv ID: {id}"""


def summarize_papers(papers: list[dict], lang: str = "Chinese") -> list[str]:
    summaries = []
    for i, p in enumerate(papers):
        if i > 0:
            time.sleep(4)  # respect 15 RPM limit
        log.info("Summarizing %d/%d: %s", i + 1, len(papers), p["title"][:50])
        s = ask_gemini(SUMMARY_PROMPT.format(lang=lang, **p))
        summaries.append(s or f"**{p['title']}**\n\n{p['abstract']}\n\n*(summary failed)*")
    return summaries


# ─── Step 4: Compose HTML digest ────────────────────────────────────
COMPOSE_PROMPT = """You are an AI for Science newsletter editor. Compose an HTML email digest from these paper summaries.

Requirements:
- Language: {lang}, Date: {date}
- Open with 2-3 sentences summarizing today's themes
- Rank by relevance score (high to low), include arXiv links
- End with a "Must-Read Pick" recommending 1 paper with reason
- Output pure HTML with inline CSS, email-client compatible
- Clean, professional layout

Paper summaries:
{summaries}"""


def compose_digest(papers, summaries, lang, date) -> str:
    combined = "\n\n---\n\n".join(
        f"## [{p['title']}]({p['url']})\nRelevance: {p['score']}/5 — {p['reason']}\n\n{s}"
        for p, s in zip(papers, summaries)
    )
    html = ask_gemini(COMPOSE_PROMPT.format(lang=lang, date=date, summaries=combined))

    if html:
        # Strip markdown code fences if present
        if "```html" in html:
            html = html.split("```html", 1)[1]
        if "```" in html:
            html = html.rsplit("```", 1)[0]
        return html.strip()

    # Fallback: simple HTML
    log.warning("HTML generation failed, using fallback")
    cards = "".join(
        f'<div style="margin:16px 0;padding:16px;border:1px solid #ddd;border-radius:8px;">'
        f'<h3><a href="{p["url"]}" style="color:#1a73e8">{p["title"]}</a></h3>'
        f'<p style="color:#666;font-size:13px">Score: {p["score"]}/5 — {p["reason"]}</p>'
        f'<div style="white-space:pre-wrap;font-size:14px">{s}</div></div>'
        for p, s in zip(papers, summaries)
    )
    return (
        f'<html><body style="font-family:sans-serif;max-width:700px;margin:auto;padding:20px">'
        f'<h1>AI4PDE Paper Digest — {date}</h1>{cards}'
        f'<hr><p style="color:#999;font-size:12px">Paper Digest Agent</p></body></html>'
    )


# ─── Step 5: Send email ─────────────────────────────────────────────
def send_email(subject: str, body: str, html: bool = False):
    sender = os.environ["EMAIL_SENDER"]
    password = os.environ["EMAIL_PASSWORD"]
    recipient = os.environ["EMAIL_RECIPIENT"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(body, "html" if html else "plain", "utf-8"))

    with smtplib.SMTP(os.environ.get("SMTP_SERVER", "smtp.gmail.com"), 587) as s:
        s.starttls()
        s.login(sender, password)
        s.sendmail(sender, [recipient], msg.as_string())
    log.info("Email sent to %s", recipient)


# ─── Main pipeline ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Paper Digest Agent")
    parser.add_argument("--dry-run", action="store_true", help="fetch + rank only, no email")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    search = cfg["search"]
    digest = cfg.get("digest", {})
    lang = digest.get("language", "Chinese")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Fetch
    papers = fetch_papers(search)
    if not papers:
        if not args.dry_run:
            send_email(f"AI4PDE Digest — {today}", "No relevant papers found today.")
        return log.info("No papers found")

    # 2. Rank
    top = rank_papers(papers, search.get("max_digest", 5))
    if not top:
        if not args.dry_run:
            send_email(f"AI4PDE Digest — {today}", "Papers found but none scored high enough.")
        return log.info("No high-relevance papers")

    if args.dry_run:
        for p in top:
            log.info("  [%d] %s — %s", p["score"], p["id"], p["title"][:60])
        return

    # 3. Summarize
    summaries = summarize_papers(top, lang)

    # 4. Compose HTML
    html = compose_digest(top, summaries, lang, today)

    # 5. Send
    send_email(f"AI4PDE Paper Digest — {today} ({len(top)} papers)", html, html=True)
    log.info("Done! Sent digest with %d papers", len(top))


if __name__ == "__main__":
    main()
