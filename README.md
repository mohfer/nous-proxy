# NousResearch OAuth Proxy

A proxy that converts the NousResearch Portal OAuth device code flow into an OpenAI-compatible REST API.

## How It Works

```
App Client --[API Key]--> NousProxy --[OAuth Agent Key]--> NousResearch Inference API
```

1. The proxy handles the OAuth device code flow (one-time setup).
2. It auto-refreshes the access token and agent key in the background.
3. Clients use a static proxy API key (generated) for authentication.
4. The proxy forwards requests to the NousResearch inference API.

## Quick Start

```bash
# Setup
cd /opt/nous-proxy
bash scripts/install.sh

# Start
systemctl start nous-proxy

# Check API key
cat /opt/nous-proxy/data/api_keys.json

# Auth (one-time)
curl -X POST http://localhost:8090/auth/device-code \
  -H "Authorization: Bearer <your-api-key>"

# Open the returned URL and enter user_code in your browser

curl -X POST http://localhost:8090/auth/poll \
  -H "Authorization: Bearer <your-api-key>"

# Use it like an OpenAI API endpoint
curl http://localhost:8090/v1/chat/completions \
  -H "Authorization: Bearer <your-api-key>" \
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
| POST | `/auth/device-code` | Start OAuth device code flow |
| POST | `/auth/poll` | Complete OAuth auth after user authorizes |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions |
| GET | `/v1/models` | List available models |
| POST | `/admin/generate-key` | Generate a new proxy API key |

## Configuration

Edit `/opt/nous-proxy/.env`:

```env
NOUS_PORTAL_URL=https://portal.nousresearch.com
NOUS_INFERENCE_URL=https://inference-api.nousresearch.com/v1
PROXY_PORT=8090
PROXY_API_KEYS=   # Comma-separated, or auto-generated
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
│   ├── proxy.py           # Request forwarding
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
