# DeepSeek Cost Proxy

OpenAI-compatible local proxy for Hermes or other agents.

## Models

Supported models:

- `deepseek-v4-flash`
- `deepseek-v4-pro`

## Run

```bash
DEEPSEEK_API_KEY=sk-xxx PROXY_AUTH_KEY=local-proxy-key go run .
```

Hermes config:

```text
base_url = http://<unraid-ip>:18188/v1
api_key = local-proxy-key
model = deepseek-v4-flash
```

Use `deepseek-v4-flash` as the low-cost default. Use `deepseek-v4-pro` by
setting the request `model` to `deepseek-v4-pro`, or by creating a second Hermes
model entry with the same `base_url` and `api_key`.

Thinking mode is enabled by default in DeepSeek V4. To disable it, send:

```json
{"thinking":{"type":"disabled"}}
```

To keep thinking mode and control effort, send:

```json
{"thinking":{"type":"enabled"},"reasoning_effort":"high"}
```

`reasoning_effort` also accepts `max`. When thinking mode is used with tool
calls, callers must preserve and send back `reasoning_content` in later turns.

Hardcoded CNY pricing estimate per 1M tokens:

| Model | Cache hit input | Cache miss input | Output |
| --- | ---: | ---: | ---: |
| `deepseek-v4-flash` | 0.02 | 1 | 2 |
| `deepseek-v4-pro` | 0.025 | 3 | 6 |

## Docker / Unraid

Local build:

```bash
docker compose up -d --build
```

Pullable GHCR image after GitHub Actions publishes:

```bash
docker pull ghcr.io/<github-owner>/deepseek-cost-proxy:latest
IMAGE_NAME=ghcr.io/<github-owner>/deepseek-cost-proxy:latest docker compose up -d
```

### Unraid Docker Settings

Container settings:

| Field | Value |
| --- | --- |
| Repository | `ghcr.io/<github-owner>/deepseek-cost-proxy:latest` |
| Network Type | `bridge` |
| WebUI | `http://[IP]:[PORT:18188]/dashboard` |
| Container Port | `18188` |
| Host Port | `18188` |
| Path | `/data` -> `/mnt/user/appdata/deepseek-cost-proxy` |

Environment variables:

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `DEEPSEEK_API_KEY` | yes | none | Real DeepSeek key, stored only in Docker env. |
| `PROXY_AUTH_KEY` | no | `local-proxy-key` | LAN proxy key used by Hermes. |
| `DEFAULT_MODEL` | no | `deepseek-v4-flash` | Default model when Hermes omits `model`. |
| `PROXY_ADDR` | no | `18188` | Listen port inside the container. Host-style values like `:18188` still work. |
| `TRACE_DIR` | no | `/data/traces` in compose | JSONL metrics/debug trace directory. |
| `DEEPSEEK_CHAT_URL` | no | `https://api.deepseek.com/chat/completions` | DeepSeek upstream URL. |

Persistent data:

```text
/mnt/user/appdata/deepseek-cost-proxy:/data
```

The proxy writes request metrics and debug traces to:

```text
/data/traces/requests.jsonl
```

## Endpoints

- `GET /v1`
- `GET /healthz`
- `GET /dashboard`
- `GET /debug` redirects to `/dashboard#debug`
- `GET /metrics`
- `GET /debug/requests`
- `GET /debug/requests/<id>`
- `POST /v1/chat/completions`

Open the unified WebUI in a browser:

```text
http://<unraid-ip>:18188/dashboard
```

The WebUI has two tabs: 数据看板 and Prompt 调试. 数据看板 shows recent
requests, prompt tokens, cached tokens, new tokens, cache hit rate, estimated
cost, and estimated savings. Both buffered and streaming responses are tracked
when the upstream response includes usage data.
The cost fields are estimates based on the hardcoded CNY pricing in `main.go`.

Open the debug tab directly:

```text
http://<unraid-ip>:18188/dashboard#debug
```

Debug endpoints compare the raw Hermes request with the normalized request sent
to DeepSeek. They expose prefix hashes, system hash, tools hash, original tool
order, normalized tool order, and truncated request previews.

## Cache Test

短 prompt 看不出缓存优化。要测试优化效果，需要连续发送两次带有相同长
`system + tools` 前缀的请求，然后观察第二次的 `cached`、`hitRate` 和
`savedCNY` 是否明显上升。

Start the proxy, then run:

```powershell
.\scripts\cache-test.ps1 `
  -ProxyUrl http://192.168.1.50:18188 `
  -ProxyKey <your-proxy-key> `
  -Model deepseek-v4-flash
```

Expected shape:

```text
request prompt cached new  completion hitRate
1       ...    0      ...  ...        0%
2       ...    high   low  ...        high%
```

The second request should show a higher cache hit rate if DeepSeek reuses the
stable `system + tools` prefix.

## Project Checklist

Done:

- Go proxy for `/v1/chat/completions`.
- `deepseek-v4-flash` and `deepseek-v4-pro` model whitelist.
- Proxy auth key and container-side DeepSeek key.
- Buffered and streaming forwarding.
- Usage capture for buffered and SSE responses.
- `/metrics`, `/dashboard`, `/debug/requests`, `/debug/requests/<id>`.
- `/dashboard` unified Dashboard and Prompt Debug UI.
- Tool sorting by `function.name`.
- Prefix hash and debug trace.
- Trace and metrics persistence through JSONL.
- Token savings visualization and CNY estimate.
- GHCR Docker publish workflow.
- Unraid configuration reference.
- Unit tests.

Pending:

- Docker build verification.
- Hermes real connection verification.
