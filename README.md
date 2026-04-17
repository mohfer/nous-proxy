# NousProxy

OpenAI + Anthropic compatible REST API proxy backed by NousResearch Portal OAuth — run your own inference gateway with subscription-based rate limits. Works with agentic coding tools like **OpenCode**, **Claude Code**, and **OpenAI Codex CLI**.

## How It Works

```
App Client --[API Key]--> NousProxy --[OAuth Agent Key]--> NousResearch Inference API
```

1. Proxy handles OAuth device code flow (one-time setup).
2. Auto-refreshes access token and agent key in the background.
3. Clients use a static proxy API key for authentication.
4. Forwards requests to NousResearch inference API with product attribution (`tags: ["product=hermes-agent"]`).

## Quick Start (Docker — Recommended)

```bash
cd /opt/nous-proxy

# Build & start
docker compose up -d

# Get your proxy API key
cat data/api_keys.json

# Auth (one-time, auto-polls until you authorize)
curl -X POST http://localhost:8090/auth/device-code \
  -H "Authorization: Bearer np-xxx"

# → Open the URL, enter the code
# → Polling happens automatically in background

# Check auth status
curl http://localhost:8090/auth/status \
  -H "Authorization: Bearer np-xxx"

# Use it
curl http://localhost:8090/v1/chat/completions \
  -H "Authorization: Bearer np-xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "xiaomi/mimo-v2-pro",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Logs
docker logs -f nous-proxy

# Stop
docker compose down

# Rebuild after code changes
docker compose up -d --build
```

Data (tokens, API keys) persists in `./data/` via bind mount.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Landing page |
| GET | `/health` | Health check + token status |
| POST | `/auth/device-code` | Start OAuth + auto-poll |
| GET | `/auth/status` | Check auth/polling status |
| POST | `/auth/poll` | Fallback: wait for auth completion |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions |
| GET | `/v1/models` | List available models |
| POST | `/v1/messages` | Anthropic Messages API (Claude Code) |
| POST | `/v1/messages/count_tokens` | Anthropic token counting stub |
| POST | `/admin/generate-key` | Generate a new proxy API key |

## Agentic Coding Tools

### 1. OpenCode

OpenCode uses AI SDK with OpenAI-compatible API. Configure in `~/.opencode.json`:

```json
{
  "provider": {
    "nous": {
      "options": {
        "baseURL": "http://localhost:8090/v1",
        "apiKey": "np-YOUR_PROXY_KEY"
      }
    }
  }
}
```

Then run `/models` in OpenCode to select a model (e.g. `xiaomi/mimo-v2-pro`).

### 2. Claude Code

Claude Code uses Anthropic Messages API. Set environment variables:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8090
export ANTHROPIC_API_KEY=np-YOUR_PROXY_KEY
claude
```

The proxy translates Anthropic Messages API format to OpenAI chat completions
automatically, including tool use and streaming.

### 3. OpenAI Codex CLI

Codex CLI uses standard OpenAI API. Set environment variables:

```bash
export OPENAI_BASE_URL=http://localhost:8090/v1
export OPENAI_API_KEY=np-YOUR_PROXY_KEY
codex
```

Or sign in with ChatGPT (no proxy needed for that path).

## Models — Free Plan

NousResearch agent key from **free OAuth subscription** has model access restrictions:

| Category | Access | Notes |
|----------|--------|-------|
| Standard models | ✅ | 350+ models available |
| `:free` models (26) | ❌ | Blocked: "OpenRouter free models are not supported" |
| `openrouter/*` models (4) | ❌ | Blocked: "This model is not supported on Free Tier" |

### $0 Standard Models (Free Plan)

These models cost $0 on the standard (non-`:free`) path and work with free subscription:

| Model | Context | Max Output | Tools | Reasoning |
|-------|---------|------------|-------|-----------|
| `xiaomi/mimo-v2-pro` | 1M | 131K | ✅ | ✅ |
| `xiaomi/mimo-v2-omni` | 262K | 65K | ✅ | ✅ |

Other $0 models exist (image gen, video gen, reranking) but are not text chat LLMs:
- `black-forest-labs/flux.2-*` — image generation
- `alibaba/wan-2.6`, `alibaba/wan-2.7` — video generation
- `bytedance/seedance-*` — video generation
- `openai/sora-2-pro` — video generation
- `google/veo-3.1` — video generation
- `cohere/rerank-*` — reranking (not chat)

### Cheap Paid Models

For access beyond `$0` models, check available models via:
```bash
curl http://localhost:8090/v1/models \
  -H "Authorization: Bearer np-xxx" | python3 -m json.tool
```

## Configuration

Edit `/opt/nous-proxy/.env`:

```env
NOUS_PORTAL_URL=https://portal.nousresearch.com
NOUS_INFERENCE_URL=https://inference-api.nousresearch.com/v1
PROXY_PORT=8090
PROXY_API_KEYS=np-xxx   # Comma-separated, or auto-generated
```

## Project Structure

```
/opt/nous-proxy/
├── pyproject.toml         # Dependencies & build config
├── Dockerfile             # Multi-stage Docker build
├── docker-compose.yml     # Container orchestration
├── docker-entrypoint.sh   # Auto-fix data dir ownership
├── .dockerignore          # Docker build exclusions
├── .env                   # Environment config
├── .env.example           # Template
├── data/                  # Persisted tokens & API keys (bind mount)
│   ├── tokens.json
│   └── api_keys.json
└── nous_proxy/
    ├── __init__.py
    ├── config.py          # Settings (pydantic-settings)
    ├── auth.py            # OAuth device code flow
    ├── token_manager.py   # Token lifecycle + auto-refresh
    ├── api_keys.py        # API key validation
    ├── proxy.py           # Request forwarding + attribution
    ├── anthropic.py       # Anthropic Messages API translator (Claude Code)
    └── main.py            # FastAPI app + CLI
```

## Development

```bash
cd /opt/nous-proxy
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# Run with auto-reload
python -m nous_proxy.main --reload
```
