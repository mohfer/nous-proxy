# NousProxy

OpenAI-compatible REST API proxy backed by NousResearch Portal OAuth — run your own inference gateway with subscription-based rate limits.

## How It Works

```
App Client --[API Key]--> NousProxy --[OAuth Agent Key]--> NousResearch Inference API
```

1. Proxy handles OAuth device code flow (one-time setup).
2. Auto-refreshes access token and agent key in the background.
3. Clients use a static proxy API key for authentication.
4. Forwards requests to NousResearch inference API with product attribution (`tags: ["product=hermes-agent"]`).

## Quick Start

```bash
# Install
cd /opt/nous-proxy
bash scripts/install.sh
systemctl start nous-proxy

# Get your proxy API key
cat /opt/nous-proxy/data/api_keys.json

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
```

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
| POST | `/admin/generate-key` | Generate a new proxy API key |

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
├── .env                   # Environment config
├── .env.example           # Template
├── data/                  # Persisted tokens & API keys
│   ├── tokens.json
│   └── api_keys.json
├── nous_proxy/
│   ├── __init__.py
│   ├── config.py          # Settings (pydantic-settings)
│   ├── auth.py            # OAuth device code flow
│   ├── token_manager.py   # Token lifecycle + auto-refresh
│   ├── api_keys.py        # API key validation
│   ├── proxy.py           # Request forwarding + attribution
│   └── main.py            # FastAPI app + CLI
└── scripts/
    ├── install.sh         # Setup script
    └── nous-proxy.service # Systemd unit
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
