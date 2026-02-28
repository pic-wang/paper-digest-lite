# Paper Digest Agent

每日自动抓取 arXiv 论文 → Gemini 评分+总结 → 邮件发送 HTML 简报。

```
config.yaml   ← 改关键词
main.py       ← 全部逻辑（单文件）
```

## 快速开始

### 1. 获取密钥

| 密钥 | 获取方式 |
|------|----------|
| `GEMINI_API_KEY` | https://aistudio.google.com → 免费 |
| `EMAIL_PASSWORD` | Google 账号 → 安全 → 两步验证 → 应用密码 |

### 2. 本地测试

```bash
cd paper-digest-agent
pip install -r requirements.txt

export GEMINI_API_KEY="xxx"
export EMAIL_SENDER="you@gmail.com"
export EMAIL_PASSWORD="xxxx xxxx xxxx xxxx"
export EMAIL_RECIPIENT="you@gmail.com"

python main.py --dry-run   # 只抓取+评分，不发邮件
python main.py             # 完整运行
```

### 3. 部署到 GitHub Actions

在 repo **Settings → Secrets → Actions** 添加 4 个 secret：

- `GEMINI_API_KEY`
- `EMAIL_SENDER`
- `EMAIL_PASSWORD`
- `EMAIL_RECIPIENT`

Push 后每天北京时间 9:00 自动运行。也可在 Actions 页面手动触发。

### 4. 自定义关键词

编辑 `config.yaml` 中的 `keywords` 和 `categories`。
