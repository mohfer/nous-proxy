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
_auto_poll_task: asyncio.Task | None = None


async def _auto_poll_until_authorized(device_code: str, interval: int) -> None:
    """Background task: poll NousPortal until user authorizes, then update tokens."""
    global _device_code_state, _auto_poll_task
    async with create_portal_client() as client:
        try:
            data = await poll_for_token(
                client,
                device_code=device_code,
                interval=interval,
                timeout=600,  # 10 minutes max
            )
            token_manager.update_from_token_response(data)
            _device_code_state["status"] = "authenticated"
            _device_code_state["authenticated_at"] = time.time()
            logger.info("Auto-poll: authentication successful!")
        except OAuthError as e:
            _device_code_state["status"] = "error"
            _device_code_state["error"] = e.error
            _device_code_state["error_description"] = e.description
            logger.error("Auto-poll failed: %s", e)
        except Exception as e:
            _device_code_state["status"] = "error"
            _device_code_state["error"] = "unexpected"
            _device_code_state["error_description"] = str(e)
            logger.exception("Auto-poll unexpected error")
        finally:
            _auto_poll_task = None


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
    # Cancel auto-poll task if still running
    if _auto_poll_task and not _auto_poll_task.done():
        _auto_poll_task.cancel()
        try:
            await _auto_poll_task
        except asyncio.CancelledError:
            pass
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
    """Start OAuth device code flow. Returns URL for user to authorize.

    Automatically starts background polling — no need to call /auth/poll.
    Check /auth/status to see if auth completed.
    """
    global _device_code_state, _auto_poll_task

    # Cancel any existing poll task
    if _auto_poll_task and not _auto_poll_task.done():
        _auto_poll_task.cancel()
        try:
            await _auto_poll_task
        except asyncio.CancelledError:
            pass

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

    device_code = data["device_code"]
    interval = data.get("interval", 5)

    # Store state
    _device_code_state = {
        "device_code": device_code,
        "interval": interval,
        "status": "polling",
        "obtained_at": time.time(),
    }

    # Start auto-poll in background
    _auto_poll_task = asyncio.create_task(
        _auto_poll_until_authorized(device_code, interval)
    )

    verification_url = data.get("verification_uri_complete", data.get("verification_uri"))

    return {
        "status": "polling",
        "user_code": data["user_code"],
        "verification_uri": verification_url,
        "expires_in": data.get("expires_in", 600),
        "message": f"Open the URL and enter code: {data['user_code']}. Polling in background...",
    }


@app.get("/auth/status")
async def auth_status(api_key: str = Depends(verify_api_key)):
    """Check current authentication / device-code flow status."""
    state = _device_code_state
    if not state:
        s = token_manager.state
        return {
            "authenticated": bool(s.access_token and s.is_token_valid),
            "agent_key_valid": s.is_agent_key_valid,
            "flow_active": False,
        }

    status = state.get("status", "unknown")
    resp = {
        "flow_active": True,
        "status": status,
    }

    if status == "polling":
        resp["message"] = "Waiting for user to authorize in browser..."
    elif status == "authenticated":
        resp["message"] = "Successfully authenticated!"
    elif status == "error":
        resp["error"] = state.get("error", "")
        resp["error_description"] = state.get("error_description", "")

    return resp


@app.post("/auth/poll")
async def poll_auth(api_key: str = Depends(verify_api_key)):
    """Poll for token — fallback if auto-poll is still running.

    In most cases you don't need this. Just hit GET /auth/status instead.
    This will wait up to 60s for the auto-poll to complete.
    """
    if not _device_code_state.get("device_code"):
        return JSONResponse(
            status_code=400,
            content={"error": "No pending device code. Call /auth/device-code first."},
        )

    # If auto-poll already finished, return its result
    status = _device_code_state.get("status")
    if status == "authenticated":
        return {"status": "authenticated", "message": "Already authenticated!"}
    if status == "error":
        return JSONResponse(
            status_code=400,
            content={
                "error": _device_code_state.get("error", "unknown"),
                "description": _device_code_state.get("error_description", ""),
            },
        )

    # Still polling — wait for the background task to finish
    if _auto_poll_task:
        try:
            await asyncio.wait_for(_auto_poll_task, timeout=60)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=408,
                content={"error": "Still waiting for user to authorize. Try again or check /auth/status."},
            )

    # Re-check after waiting
    status = _device_code_state.get("status")
    if status == "authenticated":
        return {"status": "authenticated", "message": "Successfully authenticated!"}
    return JSONResponse(
        status_code=400,
        content={
            "error": _device_code_state.get("error", "unknown"),
            "description": _device_code_state.get("error_description", ""),
        },
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
        <h2>Auth (auto-poll)</h2>
        <ol>
            <li><code>POST /auth/device-code</code> — start OAuth + auto-poll</li>
            <li>Open URL + enter code in browser</li>
            <li><code>GET /auth/status</code> — check if auth completed</li>
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
