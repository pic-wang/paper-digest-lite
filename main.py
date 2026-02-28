"""
Paper Digest Agent — 每日 arXiv 论文简报，单文件实现。

Usage:
    python main.py              # 完整运行，发送邮件
    python main.py --dry-run    # 只抓取+评分，不发邮件
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

# ─── Gemini 配置 ────────────────────────────────────────────────────
MODEL = "gemini-2.0-flash"
GEMINI_CLIENT = None


def get_gemini():
    global GEMINI_CLIENT
    if GEMINI_CLIENT is None:
        GEMINI_CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return GEMINI_CLIENT


def ask_gemini(prompt: str, retries: int = 3) -> str:
    """调用 Gemini，自带重试。"""
    for i in range(retries):
        try:
            resp = get_gemini().models.generate_content(model=MODEL, contents=prompt)
            return resp.text
        except Exception as e:
            log.warning("Gemini 调用失败 (第%d次): %s", i + 1, e)
            if i < retries - 1:
                time.sleep(2 ** (i + 1))
    return ""


# ─── Step 1: 抓取 arXiv ────────────────────────────────────────────
def fetch_papers(cfg: dict) -> list[dict]:
    cats = cfg["categories"]
    kws = cfg["keywords"]
    exclude = cfg.get("exclude_keywords", [])
    max_fetch = cfg.get("max_fetch", 50)

    # 周一多抓3天覆盖周末
    days_back = 3 if datetime.now(timezone.utc).weekday() == 0 else 1
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    # 构建查询: (cat:A OR cat:B) AND (ti/abs:"kw1" OR ...)
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

    log.info("抓取到 %d 篇论文", len(papers))
    return papers


# ─── Step 2: LLM 相关性评分 ────────────────────────────────────────
RANK_PROMPT = """你是 AI for PDE / Physics Simulation 领域的研究助手。
为每篇论文评估与以下方向的相关性（1-5分）：
- 神经网络求解 PDE（PINN, Neural Operator, FNO, DeepONet）
- 物理仿真 ML 加速（learned simulation, surrogate model）
- 科学计算×深度学习（differentiable physics, hybrid solver）
- CFD 的 AI 方法
- 域分解×神经网络

严格输出 JSON，无其他内容：
```json
[{{"id": "...", "score": 5, "reason": "一句话"}}]
```

论文列表：
{papers}"""


def rank_papers(papers: list[dict], max_digest: int = 5) -> list[dict]:
    text = "\n".join(
        f"---\nID: {p['id']}\nTitle: {p['title']}\nAbstract: {p['abstract'][:300]}"
        for p in papers
    )
    raw = ask_gemini(RANK_PROMPT.format(papers=text))
    if not raw:
        log.error("评分失败，保留前 %d 篇", max_digest)
        for p in papers:
            p["score"], p["reason"] = 3, "评分失败"
        return papers[:max_digest]

    # 解析 JSON
    try:
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0]
        scores = {s["id"]: s for s in json.loads(cleaned)}
    except (json.JSONDecodeError, KeyError):
        log.error("JSON 解析失败，保留前 %d 篇", max_digest)
        for p in papers:
            p["score"], p["reason"] = 3, "解析失败"
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
    log.info("评分完成：%d/%d 篇通过阈值", len(result), len(papers))
    return result[:max_digest]


# ─── Step 3: 逐篇总结 ──────────────────────────────────────────────
SUMMARY_PROMPT = """你是 AI for PDE / Scientific Computing 高级研究员。
用{lang}深度总结这篇论文，按以下结构输出 Markdown：

🎯 一句话总结
🔬 核心方法（3-5句，说清输入/输出、创新点、与 PINN/FNO 等的区别）
📊 主要结果（benchmark、关键指标、与 baseline 对比）
💡 亮点与局限（各 1-2 点）
🔗 与 Urban Wind / CFD 研究的关联（无则说"关联较弱"）

论文：
- 标题：{title}
- 作者：{authors}
- 摘要：{abstract}
- arXiv ID：{id}"""


def summarize_papers(papers: list[dict], lang: str = "中文") -> list[str]:
    summaries = []
    for i, p in enumerate(papers):
        if i > 0:
            time.sleep(4)  # 15 RPM 限制
        log.info("总结 %d/%d: %s", i + 1, len(papers), p["title"][:50])
        s = ask_gemini(SUMMARY_PROMPT.format(lang=lang, **p))
        summaries.append(s or f"**{p['title']}**\n\n{p['abstract']}\n\n*（总结失败）*")
    return summaries


# ─── Step 4: 组装 HTML 简报 ────────────────────────────────────────
COMPOSE_PROMPT = """你是 AI for Science 日报编辑。将以下论文总结整合为 HTML 邮件简报。

要求：
- 语言：{lang}，日期：{date}
- 开头 2-3 句概述今日主题
- 按相关性从高到低排列，每篇附 arXiv 链接
- 末尾附 "📌 今日最值得精读" 推荐 1 篇
- 输出纯 HTML，用 inline CSS，适配邮件客户端
- 风格简洁专业，排版美观

论文总结：
{summaries}"""


def compose_digest(papers, summaries, lang, date) -> str:
    combined = "\n\n---\n\n".join(
        f"## [{p['title']}]({p['url']})\n相关性：{p['score']}分 — {p['reason']}\n\n{s}"
        for p, s in zip(papers, summaries)
    )
    html = ask_gemini(COMPOSE_PROMPT.format(lang=lang, date=date, summaries=combined))

    if html:
        # 去掉 Gemini 可能加的 markdown 代码围栏
        if "```html" in html:
            html = html.split("```html", 1)[1]
        if "```" in html:
            html = html.rsplit("```", 1)[0]
        return html.strip()

    # Fallback：简易 HTML
    log.warning("HTML 生成失败，使用 fallback")
    cards = "".join(
        f'<div style="margin:16px 0;padding:16px;border:1px solid #ddd;border-radius:8px;">'
        f'<h3><a href="{p["url"]}" style="color:#1a73e8">{p["title"]}</a></h3>'
        f'<p style="color:#666;font-size:13px">评分：{p["score"]} — {p["reason"]}</p>'
        f'<div style="white-space:pre-wrap;font-size:14px">{s}</div></div>'
        for p, s in zip(papers, summaries)
    )
    return (
        f'<html><body style="font-family:sans-serif;max-width:700px;margin:auto;padding:20px">'
        f'<h1>📑 AI4PDE 论文日报 — {date}</h1>{cards}'
        f'<hr><p style="color:#999;font-size:12px">Paper Digest Agent</p></body></html>'
    )


# ─── Step 5: 发送邮件 ──────────────────────────────────────────────
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
    log.info("邮件已发送 → %s", recipient)


# ─── 主流程 ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Paper Digest Agent")
    parser.add_argument("--dry-run", action="store_true", help="只抓取+评分，不发邮件")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    search = cfg["search"]
    digest = cfg.get("digest", {})
    lang = digest.get("language", "中文")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. 抓取
    papers = fetch_papers(search)
    if not papers:
        if not args.dry_run:
            send_email(f"📭 AI4PDE 日报 — {today}", "今天没有检索到相关论文。")
        return log.info("无论文，结束")

    # 2. 评分
    top = rank_papers(papers, search.get("max_digest", 5))
    if not top:
        if not args.dry_run:
            send_email(f"📭 AI4PDE 日报 — {today}", "检索到论文但相关性不高。")
        return log.info("无高相关论文，结束")

    if args.dry_run:
        for p in top:
            log.info("  [%d分] %s — %s", p["score"], p["id"], p["title"][:60])
        return

    # 3. 总结
    summaries = summarize_papers(top, lang)

    # 4. 组装 HTML
    html = compose_digest(top, summaries, lang, today)

    # 5. 发送
    send_email(f"📑 AI4PDE 论文日报 — {today} ({len(top)}篇)", html, html=True)
    log.info("完成！已发送 %d 篇论文简报", len(top))


if __name__ == "__main__":
    main()
