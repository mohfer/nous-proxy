"""Request proxy: forward OpenAI-compatible requests to NousResearch inference API."""

from __future__ import annotations

import json
import logging
import time

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from nous_proxy.token_manager import token_manager

logger = logging.getLogger(__name__)

# Headers that hermes-agent sends — NousPortal may use these for rate-limit tiering.
_HERMES_UPSTREAM_HEADERS = {
    "User-Agent": "HermesAgent/0.10.0 (hermes-agent; Python/3.11)",
}

_proxy_client: httpx.AsyncClient | None = None
_PROXY_TIMEOUT = httpx.Timeout(120, connect=10)


def init_proxy_client() -> None:
    """Initialize shared upstream HTTP client."""
    _get_proxy_client()


def _get_proxy_client() -> httpx.AsyncClient:
    global _proxy_client
    if _proxy_client is None:
        _proxy_client = httpx.AsyncClient(timeout=_PROXY_TIMEOUT)
    return _proxy_client


async def close_proxy_client() -> None:
    """Close shared upstream HTTP client."""
    global _proxy_client
    if _proxy_client is not None:
        client = _proxy_client
        _proxy_client = None
        await client.aclose()


# Model families that support reasoning extra_body (mirrors hermes-agent _supports_reasoning_extra_body)
_REASONING_MODEL_PREFIXES = (
    "anthropic/", "claude-",
    "deepseek/",
    "openai/", "o1-", "o3-", "o4-",
    "google/gemini-2",
    "qwen/qwen3",
    "x-ai/",
)


def _model_supports_reasoning(model: str) -> bool:
    """Check if a model family supports reasoning extra_body."""
    model_lower = (model or "").lower()
    return any(model_lower.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def _inject_hermes_attribution(req_data: dict) -> dict:
    """Inject hermes-agent attribution fields so NousPortal applies the same rate-limit tier.

    hermes-agent sends:
      - extra_body.tags = ["product=hermes-agent"]
      - extra_body.reasoning = {"enabled": True, "effort": "medium"}  (only for supported models)

    NousPortal likely uses tags to identify product-origin and assign rate limits.
    Reasoning is only injected for models that actually support it (Claude, DeepSeek, etc.)
    to avoid 400 errors on models like Gemma, Nemotron, MiMo.
    """
    model = req_data.get("model", "")

    # Merge client's extra_body or create one
    extra = dict(req_data.get("extra_body") or {})

    # Product attribution — hermes-agent always sends this for NousPortal
    if "tags" not in extra:
        extra["tags"] = ["product=hermes-agent"]

    # Reasoning — only for supported models (mirrors hermes-agent logic)
    if "reasoning" not in extra and _model_supports_reasoning(model):
        extra["reasoning"] = {"enabled": True, "effort": "medium"}

    # Write back
    req_data["extra_body"] = extra
    return req_data


async def proxy_chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    """Forward chat/completions request to inference API.

    Handles both streaming (SSE) and non-streaming responses.
    """
    t0 = time.monotonic()
    body = await request.body()
    req_data = json.loads(body) if body else {}
    is_stream = req_data.get("stream", False)

    # Inject hermes-agent attribution fields (tags, reasoning)
    req_data = _inject_hermes_attribution(req_data)
    body = json.dumps(req_data).encode()

    client = _get_proxy_client()
    t_key = time.monotonic()
    agent_key = await token_manager.ensure_valid_agent_key(client)
    key_ms = (time.monotonic() - t_key) * 1000
    inference_url = token_manager.state.effective_inference_url

    if key_ms > 100:
        logger.info("OpenAI handler: agent_key took %.0fms", key_ms)

    if is_stream:
        async def generate():
            try:
                async with client.stream(
                    "POST",
                    f"{inference_url}/chat/completions",
                    content=body,
                    headers={
                        **_HERMES_UPSTREAM_HEADERS,
                        "Authorization": f"Bearer {agent_key}",
                        "Content-Type": "application/json",
                    },
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            except httpx.HTTPError as e:
                logger.error("Stream error: %s", e)
                yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    resp = await client.post(
        f"{inference_url}/chat/completions",
        content=body,
        headers={
            **_HERMES_UPSTREAM_HEADERS,
            "Authorization": f"Bearer {agent_key}",
            "Content-Type": "application/json",
        },
    )
    return JSONResponse(
        content=resp.json(),
        status_code=resp.status_code,
    )


async def proxy_models() -> JSONResponse:
    """Forward models list request to inference API."""
    client = _get_proxy_client()
    agent_key = await token_manager.ensure_valid_agent_key(client)
    inference_url = token_manager.state.effective_inference_url

    resp = await client.get(
        f"{inference_url}/models",
        headers={
            **_HERMES_UPSTREAM_HEADERS,
            "Authorization": f"Bearer {agent_key}",
        },
    )
    return JSONResponse(
        content=resp.json(),
        status_code=resp.status_code,
    )

