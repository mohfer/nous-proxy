"""NousResearch OAuth Proxy - FastAPI application entry point.

Exposes NousResearch inference API as standard REST API with static API key auth.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse, HTMLResponse

from nous_proxy.config import settings
from nous_proxy.api_keys import load_api_keys, verify_api_key, create_and_store_api_key
from nous_proxy.token_manager import token_manager
from nous_proxy.auth import request_device_code, poll_for_token, OAuthError, create_portal_client
from nous_proxy.proxy import (
    proxy_chat_completions,
    proxy_models,
    init_proxy_client,
    close_proxy_client,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nous_proxy")

# ---------------------------------------------------------------------------
# In-memory device code state (for current auth flow)
# ---------------------------------------------------------------------------
_device_code_state: dict = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_api_keys()
    token_manager.load()
    init_proxy_client()
    refresh_task = token_manager.start_background_refresh()
    logger.info("NousResearch Proxy starting on %s:%d", settings.proxy_host, settings.proxy_port)

    if not token_manager.state.has_credentials:
        logger.warning("No OAuth credentials found. Call POST /auth/device-code to authenticate.")

    yield

    # Shutdown
    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass

    await close_proxy_client()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NousResearch Proxy",
    description="OAuth proxy for NousResearch inference API (OpenAI-compatible)",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check endpoint."""
    s = token_manager.state
    now = time.time()
    return {
        "status": "ok",
        "authenticated": bool(s.access_token),
        "agent_key_valid": s.is_agent_key_valid,
        "token_expires_in": max(0, int(s.token_expires_at - now)) if s.access_token else 0,
        "agent_key_expires_in": max(0, int(s.agent_key_expires_at - now)) if s.agent_key else 0,
    }


# ---------------------------------------------------------------------------
# Auth Endpoints
# ---------------------------------------------------------------------------
@app.post("/auth/device-code")
async def start_device_auth(api_key: str = Depends(verify_api_key)):
    """Start OAuth device code flow. Returns URL + code for user to authorize.

    After calling this, user opens verification_uri and enters user_code,
    then call POST /auth/poll to complete the flow.
    """
    global _device_code_state

    async with create_portal_client() as client:
        try:
            data = await request_device_code(client)
        except httpx.HTTPStatusError as e:
            return JSONResponse(
                status_code=502,
                content={"error": f"Portal returned HTTP {e.response.status_code}"},
            )
        except httpx.HTTPError as e:
            return JSONResponse(
                status_code=502,
                content={"error": f"Connection error: {e}"},
            )

    # Store device_code for polling
    _device_code_state = {
        "device_code": data["device_code"],
        "interval": data.get("interval", 5),
        "obtained_at": time.time(),
    }

    return {
        "user_code": data["user_code"],
        "verification_uri": data.get("verification_uri_complete", data.get("verification_uri")),
        "expires_in": data.get("expires_in", 600),
        "message": f"Open the URL and enter code: {data['user_code']}",
    }


@app.post("/auth/poll")
async def poll_auth(api_key: str = Depends(verify_api_key)):
    """Poll for token after user has authorized via device code.

    Call this after POST /auth/device-code and user authorization.
    """
    if not _device_code_state.get("device_code"):
        return JSONResponse(
            status_code=400,
            content={"error": "No pending device code. Call /auth/device-code first."},
        )

    async with create_portal_client() as client:
        try:
            data = await poll_for_token(
                client,
                device_code=_device_code_state["device_code"],
                interval=_device_code_state["interval"],
                timeout=60,  # Short timeout for single poll attempt
            )
            token_manager.update_from_token_response(data)
            _device_code_state.clear()

            return {
                "status": "authenticated",
                "expires_in": data.get("expires_in", 3600),
                "message": "Successfully authenticated!",
            }

        except OAuthError as e:
            if e.error == "timeout":
                return JSONResponse(
                    status_code=408,
                    content={"error": "User has not authorized yet. Try again."},
                )
            return JSONResponse(
                status_code=400,
                content={"error": e.error, "description": e.description},
            )
        except httpx.HTTPError as e:
            return JSONResponse(
                status_code=502,
                content={"error": f"Connection error: {e}"},
            )


# ---------------------------------------------------------------------------
# OpenAI-Compatible Endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    """OpenAI-compatible chat completions endpoint.

    Usage:
        curl http://localhost:8090/v1/chat/completions \\
          -H "Authorization: Bearer <proxy-api-key>" \\
          -H "Content-Type: application/json" \\
          -d '{"model":"hermes-3-405b","messages":[{"role":"user","content":"Hi"}]}'
    """
    return await proxy_chat_completions(request)


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """OpenAI-compatible models list endpoint."""
    return await proxy_models()



# ---------------------------------------------------------------------------
# Convenience: non-versioned routes (same as /v1/*)
# ---------------------------------------------------------------------------
@app.post("/chat/completions")
async def chat_completions_short(request: Request, api_key: str = Depends(verify_api_key)):
    return await proxy_chat_completions(request)


@app.get("/models")
async def list_models_short(api_key: str = Depends(verify_api_key)):
    return await proxy_models()


# ---------------------------------------------------------------------------
# API Key Management
# ---------------------------------------------------------------------------
@app.post("/admin/generate-key")
async def admin_generate_key(api_key: str = Depends(verify_api_key)):
    """Generate a new API key (requires existing valid key)."""
    new_key = create_and_store_api_key()
    return {"api_key": new_key}


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Simple landing page."""
    return HTMLResponse("""
    <html>
    <head><title>NousResearch Proxy</title></head>
    <body style="font-family:system-ui;max-width:600px;margin:40px auto;padding:0 20px">
        <h1>NousResearch Proxy</h1>
        <p>OpenAI-compatible API proxy backed by NousResearch OAuth.</p>
        <h2>Endpoints</h2>
        <ul>
            <li><code>POST /v1/chat/completions</code></li>
            <li><code>GET  /v1/models</code></li>
            <li><code>GET  /health</code></li>
        </ul>
        <h2>Auth</h2>
        <ol>
            <li><code>POST /auth/device-code</code> — start OAuth flow</li>
            <li>Open URL + enter code in browser</li>
            <li><code>POST /auth/poll</code> — complete auth</li>
        </ol>
    </body>
    </html>
    """)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def cli():
    """CLI entry point: nous-proxy"""
    import uvicorn
    uvicorn.run(
        "nous_proxy.main:app",
        host=settings.proxy_host,
        port=settings.proxy_port,
        log_level=settings.log_level,
        reload="--reload" in sys.argv,
    )


if __name__ == "__main__":
    cli()
