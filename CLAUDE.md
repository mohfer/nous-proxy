# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

### Environment setup
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

### Run the proxy
```bash
# Local development (auto-reload)
python -m nous_proxy.main --reload

# Installed console script entrypoint
nous-proxy
```

### Lint and test
```bash
ruff check .
pytest

# Run a single test
pytest tests/path/test_file.py::test_name
```

Note: this repository currently does not include a first-party `tests/` directory; use the single-test form above when tests are added.

### Build package artifacts
```bash
uv build
```

### Linux service/deployment (from repo scripts)
```bash
# Docker (recommended)
docker compose up -d
```

## Architecture overview

This project is a FastAPI proxy that exposes OpenAI-compatible endpoints while authenticating upstream calls through NousResearch OAuth device flow.

High-level request path:
1. Client calls proxy with static proxy API key (`Authorization: Bearer np-...`).
2. `verify_api_key` validates inbound key (`nous_proxy/api_keys.py`).
3. Proxy endpoint handler forwards request via `nous_proxy/proxy.py`.
4. `TokenManager.ensure_valid_agent_key()` ensures a valid OAuth access token and minted agent key (`nous_proxy/token_manager.py`).
5. Request is sent to Nous inference API with agent key bearer auth.

### Core modules and responsibilities
- `nous_proxy/main.py`: FastAPI app wiring, lifespan startup/shutdown, auth endpoints (`/auth/device-code`, `/auth/poll`), OpenAI-compatible routes (`/v1/chat/completions`, `/v1/models`, `/v1/embeddings`), and CLI entrypoint.
- `nous_proxy/auth.py`: OAuth device code flow primitives (request device code, poll token, refresh token, mint agent key).
- `nous_proxy/token_manager.py`: In-memory + persisted token state, validity checks, refresh/mint orchestration, background refresh loop.
- `nous_proxy/proxy.py`: Actual HTTP forwarding to inference API, including streaming passthrough for chat completions.
- `nous_proxy/api_keys.py`: Proxy-side API key loading (env + file), generation, persistence, and request validation dependency.
- `nous_proxy/config.py`: All environment-driven settings (portal/inference URLs, ports, API key list, data paths, log level).

### Stateful data
- `data/tokens.json`: persisted OAuth and agent-key lifecycle state.
- `data/api_keys.json`: persisted inbound proxy API keys.

### Operational assumptions from scripts/docs
- Docker is the primary deployment method (docker-compose.yml, Dockerfile).
- Data directory (`data/`) is bind-mounted to host `./data/`.
- Entrypoint auto-chowns `data/` to `app` user (UID 999) on startup.
- First-time OAuth is a two-step flow: `POST /auth/device-code` then `POST /auth/poll` after browser authorization.
