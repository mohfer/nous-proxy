"""Request proxy: forward OpenAI-compatible requests to NousResearch inference API."""

from __future__ import annotations

import json
import logging

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from nous_proxy.token_manager import token_manager

logger = logging.getLogger(__name__)

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


async def proxy_chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    """Forward chat/completions request to inference API.

    Handles both streaming (SSE) and non-streaming responses.
    """
    body = await request.body()
    req_data = json.loads(body) if body else {}
    is_stream = req_data.get("stream", False)

    client = _get_proxy_client()
    agent_key = await token_manager.ensure_valid_agent_key(client)
    inference_url = token_manager.state.effective_inference_url

    if is_stream:
        async def generate():
            try:
                async with client.stream(
                    "POST",
                    f"{inference_url}/chat/completions",
                    content=body,
                    headers={
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
            },
        )

    resp = await client.post(
        f"{inference_url}/chat/completions",
        content=body,
        headers={
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
        headers={"Authorization": f"Bearer {agent_key}"},
    )
    return JSONResponse(
        content=resp.json(),
        status_code=resp.status_code,
    )

