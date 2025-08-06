
# AI Trends MCP Server

Stay on top of the *fast-moving* GenAI landscape without doom-scrolling hundreds of feeds.  
**AI Trends** is a public [Model Context Protocol](https://github.com/microsoft/mcp) (MCP) server that surfaces daily, weekly, and monthly trend summaries extracted **only from pre-vetted, high-authority sources**.


Endpoint (SSE)
[https://aitrends-remote-mcp-server-authless.hupa-albert.workers.dev/sse](https://aitrends-remote-mcp-server-authless.hupa-albert.workers.dev/sse)



| Function        | Description                                                                      |
|-----------------|----------------------------------------------------------------------------------|
| `get_latest_trends` | Returns JSON with daily / weekly / monthly trend snapshots (regenerated nightly). |
| `ask_trends`        | RAG-style Q&A over the curated corpus (citations included in the response).        |

---

## Why use AI Trends?

* **Fresh:** Trends are recomputed every 24 h.  
* **Trustworthy:** Source list is hand-curated (peer-reviewed journals, major labs, respected analysts).  
* **Portable:** Works in any MCP-capable client—VS Code, GitHub Copilot Chat, Claude Desktop/Web, Cursor, Continue, etc.  
* **Zero setup:** No keys; no auth.

---

## Quick Start (VS Code ≥ 1.99, Copilot Chat)

1. Open **Command Palette → “Preferences: Open User Settings (JSON)”**.  
2. Add:

```json
    "chat.mcp.enabled": true,
    "mcp": {
        "servers": {
            "ai-trends": {
                "url": "https://aitrends-remote-mcp-server-authless.hupa-albert.workers.dev/sse"
            }
        }
    },
````

3. Reload VS Code.
4. In Copilot Chat, type `/agent` and ask:

   ```
   latest weekly genai trends
   ```

---

## Quick Start (Claude Desktop)

1. **Claude → Settings → Developer → Edit Config**.
2. Insert under `mcpServers`:

   ```json
   {
     "name": "AI Trends",
     "url": "https://aitrends-remote-mcp-server-authless.hupa-albert.workers.dev/sse"
   }
   ```
3. Restart the app.
4. In any chat, click the ⚙️ Tools icon → enable **AI Trends**.

---

## Example Prompts

| Prompt                                                                    | Result                                                        |
| ------------------------------------------------------------------------- | ------------------------------------------------------------- |
| “latest daily trends”                                                     | Returns today’s top GenAI topics with one-sentence summaries. |
| “weekly trend: state of open-source LLMs?”                                | Focused extract on OSS model momentum over the past 7 days.   |
| “ask trends Which vendors invest most in retrieval-augmented generation?” | RAG answer + citations.                                       |

---

## Advanced: Re-use in other clients

| Client                       | How to add                                        |
| ---------------------------- | ------------------------------------------------- |
| **Cline / Roo (VS Code)**    | `MCP Servers` panel → **Add Server** → paste URL. |
| **Cursor**                   | *Settings → MCP* → **Add Remote Server**.         |
| **Continue**                 | `.continue/config.json` → add server entry.       |
| **Cloudflare AI Playground** | Paste endpoint URL and start chatting—no install. |
| **Custom Script**            | Any MCP SDK that supports `sse` transport.        |

---

## Output Schema

### `get_latest_trends`

```json
{
  "daily":    [{ "title": "Flash-Attention-3", "summary": "...", "rank": 1 }, ...],
  "weekly":   [{ "title": "MoE scaling laws",  "summary": "...", "rank": 1 }, ...],
  "monthly":  [{ "title": "AI safety funding", "summary": "...", "rank": 1 }, ...],
  "generated_at": "2025-06-30T02:10:00Z"
}
```

### `ask_trends`

```json
{
  "answer": "Anthropic and OpenAI are the two largest RAG investors...",
  "citations": [
    { "title": "Anthropic raises $X B", "source": "FT",  "url": "..." },
    { "title": "OpenAI introduces Retrieval-QA", "source": "OpenAI Blog", "url": "..." }
  ]
}
```

---

## Troubleshooting

| Symptom              | Fix                                                           |
| -------------------- | ------------------------------------------------------------- |
| *“Server not found”* | Check corporate firewall; ensure SSE (port 443) allowed.      |
| Empty trend lists    | Daily rebuild occurs \~02:00 UTC. Try again in a few minutes. |
| Timeouts in IDE      | Increase MCP client timeout to ≥ 30 s (large RAG responses).  |

---

## Contributing / Feedback

Open an issue or PR on this repo.
Source curation criteria live in `/sources/whitelist.yaml`.

---

## License

MIT (server), CC-BY-4.0 (trend datasets).

---

> *Built on Cloudflare Workers. Maintained by the AI Trends team—pull requests welcome!*

```
```
