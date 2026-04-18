"""Anthropic Messages API <-> OpenAI Chat Completions translator.

Allows Claude Code (and other Anthropic SDK clients) to use the proxy
by translating between the two API formats.

Design based on copilot-api patterns for production-grade compatibility.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator

from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from nous_proxy.proxy import _get_proxy_client, _HERMES_UPSTREAM_HEADERS, _inject_hermes_attribution
from nous_proxy.token_manager import token_manager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing (before translation)
# ---------------------------------------------------------------------------

def _strip_cache_control(payload: dict) -> None:
    """Remove cache_control from content blocks — OpenAI API rejects extra fields."""
    for msg in payload.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    del block["cache_control"]
    # System prompt
    system = payload.get("system", "")
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and "cache_control" in block:
                del block["cache_control"]


def _normalize_tool_schema(schema: dict) -> dict:
    """Ensure object schemas have properties field — OpenAI rejects without it."""
    if isinstance(schema, dict) and schema.get("type") == "object" and "properties" not in schema:
        return {**schema, "properties": {}}
    return schema


def _sanitize_tools(payload: dict) -> None:
    """Normalize tool schemas and remove unsupported fields."""
    tools = payload.get("tools", [])
    if not tools:
        return
    for tool in tools:
        if "input_schema" in tool:
            tool["input_schema"] = _normalize_tool_schema(tool.get("input_schema", {}))


def _merge_tool_result_text(payload: dict) -> None:
    """Merge adjacent tool_result + text blocks in user messages.

    Claude Code sometimes sends:
      [{"type":"tool_result",...}, {"type":"text","text":"..."}]
    Some backends handle this better as merged tool_result.
    """
    for msg in payload.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue

        tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
        text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
        other_blocks = [b for b in content if isinstance(b, dict) and b.get("type") not in ("tool_result", "text")]

        if not tool_results or not text_blocks:
            continue

        # Merge text into last tool_result
        merged = []
        text_consumed = False
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                if not text_consumed and block is tool_results[-1]:
                    # Append text to this tool_result
                    existing = block.get("content", "")
                    text_content = "\n\n".join(tb.get("text", "") for tb in text_blocks)
                    if isinstance(existing, str):
                        block["content"] = f"{existing}\n\n{text_content}" if existing else text_content
                    elif isinstance(existing, list):
                        existing.append({"type": "text", "text": text_content})
                    text_consumed = True
                merged.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                # Skip text blocks (consumed into tool_result)
                continue
            else:
                merged.append(block)

        if text_consumed:
            msg["content"] = merged


def preprocess_anthropic_payload(payload: dict) -> None:
    """Run all preprocessing steps on Anthropic payload."""
    _strip_cache_control(payload)
    _sanitize_tools(payload)
    _merge_tool_result_text(payload)


# ---------------------------------------------------------------------------
# Anthropic -> OpenAI translation
# ---------------------------------------------------------------------------

def _anthropic_to_openai(body: dict) -> tuple[dict, dict]:
    """Convert Anthropic Messages API request to OpenAI chat completions format.

    Returns (openai_request, meta) where meta carries fields needed for
    the reverse translation.
    """
    meta: dict[str, Any] = {}

    model = body.get("model", "")

    # --- messages ---
    messages: list[dict] = []

    # System prompt
    system = body.get("system", "")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            sys_text = "\n\n".join(
                b.get("text", "") for b in system
                if isinstance(b, dict) and b.get("type") == "text"
            )
            if sys_text:
                messages.append({"role": "system", "content": sys_text})

    # User/assistant messages
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            _translate_content_blocks(content, role, messages)

    # --- tools ---
    tools = body.get("tools", [])
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })

    # --- tool_choice ---
    tool_choice = body.get("tool_choice")
    openai_tool_choice = _translate_tool_choice(tool_choice)

    # --- max_tokens ---
    max_tokens = body.get("max_tokens", 4096)

    # --- temperature ---
    temperature = body.get("temperature")

    # --- stream ---
    stream = body.get("stream", False)

    # --- Build request ---
    openai_req: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if temperature is not None:
        openai_req["temperature"] = temperature
    if openai_tools:
        openai_req["tools"] = openai_tools
    if openai_tool_choice is not None:
        openai_req["tool_choice"] = openai_tool_choice

    return openai_req, meta


def _translate_content_blocks(content: list, role: str, messages: list[dict]) -> None:
    """Translate Anthropic content blocks to OpenAI messages."""
    tool_calls: list[dict] = []
    openai_content: list[dict] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")

        if btype == "text":
            text = block.get("text", "")
            if text:
                openai_content.append({"type": "text", "text": text})

        elif btype == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                openai_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"},
                })
            elif source.get("type") == "url":
                openai_content.append({
                    "type": "image_url",
                    "image_url": {"url": source.get("url", "")},
                })

        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

        elif btype == "tool_result":
            tool_result_content = block.get("content", "")
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
            }
            if isinstance(tool_result_content, str):
                tool_msg["content"] = tool_result_content
            elif isinstance(tool_result_content, list):
                parts = []
                for p in tool_result_content:
                    if isinstance(p, dict):
                        if p.get("type") == "text":
                            parts.append(p.get("text", ""))
                        elif p.get("type") == "image":
                            # Tool result with image — convert to text placeholder
                            parts.append("[image]")
                tool_msg["content"] = "\n".join(parts) if parts else ""
            else:
                tool_msg["content"] = ""
            messages.append(tool_msg)

        elif btype == "thinking":
            # Skip thinking blocks — not supported in OpenAI format
            pass

    # Build assistant message with tool_calls if present
    if tool_calls:
        msg: dict[str, Any] = {"role": "assistant"}
        msg["content"] = openai_content if openai_content else None
        msg["tool_calls"] = tool_calls
        messages.append(msg)
    elif openai_content:
        if len(openai_content) == 1 and openai_content[0].get("type") == "text":
            messages.append({"role": role, "content": openai_content[0]["text"]})
        else:
            messages.append({"role": role, "content": openai_content})


def _translate_tool_choice(tool_choice: Any) -> Any:
    """Translate Anthropic tool_choice to OpenAI format."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        mapping = {"auto": "auto", "any": "required", "none": "none"}
        return mapping.get(tool_choice, "auto")
    if isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type", "")
        if tc_type == "auto":
            return "auto"
        elif tc_type == "any":
            return "required"
        elif tc_type == "none":
            return "none"
        elif tc_type == "tool":
            name = tool_choice.get("name", "")
            if name:
                return {"type": "function", "function": {"name": name}}
    return None


# ---------------------------------------------------------------------------
# OpenAI -> Anthropic translation (non-streaming)
# ---------------------------------------------------------------------------

def _openai_to_anthropic(openai_resp: dict, meta: dict) -> dict:
    """Convert OpenAI chat completion response to Anthropic Messages format."""
    choice = (openai_resp.get("choices") or [{}])[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")

    stop_reason = _map_stop_reason(finish_reason)

    # Build content blocks
    content: list[dict] = []

    # Text content — handle reasoning models that put output in "reasoning" field
    text = message.get("content") or ""
    if not text and message.get("reasoning"):
        text = message["reasoning"]
    if text:
        content.append({"type": "text", "text": text})

    # Tool calls
    for tc in message.get("tool_calls", []):
        fn = tc.get("function", {})
        tool_input = _safe_json_parse(fn.get("arguments", "{}"))
        content.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "input": tool_input,
        })

    # Usage
    usage = openai_resp.get("usage", {})
    return {
        "id": openai_resp.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": openai_resp.get("model", ""),
        "content": content or [{"type": "text", "text": ""}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _map_stop_reason(finish_reason: str | None) -> str | None:
    if finish_reason is None:
        return None
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(finish_reason, "end_turn")


def _safe_json_parse(s: str) -> dict:
    """Parse JSON, returning raw string wrapped in dict on failure."""
    try:
        result = json.loads(s)
        return result if isinstance(result, dict) else {"value": result}
    except (json.JSONDecodeError, TypeError):
        return {"raw": s} if s else {}


# ---------------------------------------------------------------------------
# Streaming state machine
# ---------------------------------------------------------------------------

class _StreamState:
    """Track content block state for proper Anthropic SSE formatting."""

    def __init__(self):
        self.message_start_sent = False
        self.content_block_index = 0
        self.content_block_open = False
        self.thinking_block_open = False
        self.tool_calls: dict[int, dict] = {}  # openai_tool_index -> {id, name, anthropic_index}

    @property
    def is_tool_block_open(self) -> bool:
        if not self.content_block_open:
            return False
        return any(
            tc["anthropic_index"] == self.content_block_index
            for tc in self.tool_calls.values()
        )

    def close_block(self) -> list[dict]:
        """Close the current content block, return events."""
        events = []
        if self.content_block_open:
            events.append({
                "type": "content_block_stop",
                "index": self.content_block_index,
            })
            self.content_block_open = False
            self.content_block_index += 1
        elif self.thinking_block_open:
            events.append({
                "type": "content_block_stop",
                "index": self.content_block_index,
            })
            self.thinking_block_open = False
            self.content_block_index += 1
        return events


# ---------------------------------------------------------------------------
# Streaming: OpenAI SSE -> Anthropic SSE
# ---------------------------------------------------------------------------

async def _anthropic_stream_generator(
    async_chunks: AsyncGenerator[bytes, None],
    meta: dict,
) -> AsyncGenerator[bytes, None]:
    """Convert OpenAI SSE stream to Anthropic SSE format."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    model = meta.get("model", "")
    input_tokens = meta.get("input_tokens", 0)
    state = _StreamState()
    output_tokens = 0
    buffer = b""

    # message_start
    yield _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id, "type": "message", "role": "assistant",
            "model": model, "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })

    async for chunk in async_chunks:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            if not line.startswith(b"data: "):
                continue

            data_str = line[6:].decode("utf-8", errors="replace")
            if data_str.strip() == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Skip empty choices (some backends send ping-like chunks)
            choices = data.get("choices")
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            finish = choice.get("finish_reason")

            # Handle thinking/reasoning
            reasoning_text = delta.get("reasoning") or delta.get("reasoning_text") or ""
            if reasoning_text:
                if not state.thinking_block_open and not state.content_block_open:
                    # Open thinking block (first reasoning chunk)
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": state.content_block_index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    })
                    state.thinking_block_open = True
                if state.thinking_block_open:
                    # Accumulate reasoning into thinking block
                    yield _sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": state.content_block_index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning_text},
                    })

            # Handle reasoning_opaque (signature) — only emit when non-empty
            reasoning_opaque = delta.get("reasoning_opaque") or ""
            if reasoning_opaque and state.thinking_block_open:
                yield _sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": state.content_block_index,
                    "delta": {"type": "signature_delta", "signature": reasoning_opaque},
                })

            # Handle text content
            text_delta = delta.get("content") or ""
            # Do NOT fallback to reasoning_text — reasoning stays in thinking block

            if text_delta:
                # Close thinking block if open
                if state.thinking_block_open:
                    yield _sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": state.content_block_index,
                        "delta": {"type": "signature_delta", "signature": ""},
                    })
                    yield _sse_event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": state.content_block_index,
                    })
                    state.thinking_block_open = False
                    state.content_block_index += 1

                # Close tool block if open
                if state.is_tool_block_open:
                    yield _sse_event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": state.content_block_index,
                    })
                    state.content_block_open = False
                    state.content_block_index += 1

                # Open text block if needed
                if not state.content_block_open:
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": state.content_block_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                    state.content_block_open = True

                yield _sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": state.content_block_index,
                    "delta": {"type": "text_delta", "text": text_delta},
                })

            # Handle tool calls
            for tc in delta.get("tool_calls", []):
                tc_id = tc.get("id", "")
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                fn_args = fn.get("arguments", "")

                if tc_id and fn_name:
                    # New tool call — close previous block
                    prev_events = state.close_block()
                    for evt in prev_events:
                        yield _sse_event(evt["type"], evt)

                    ai_index = tc.get("index", len(state.tool_calls))
                    state.tool_calls[ai_index] = {
                        "id": tc_id, "name": fn_name,
                        "anthropic_index": state.content_block_index,
                    }

                    yield _sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": state.content_block_index,
                        "content_block": {
                            "type": "tool_use", "id": tc_id,
                            "name": fn_name, "input": {},
                        },
                    })
                    state.content_block_open = True

                # Arguments delta
                if fn_args:
                    ai_index = tc.get("index", 0)
                    tc_info = state.tool_calls.get(ai_index)
                    if tc_info:
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": tc_info["anthropic_index"],
                            "delta": {"type": "input_json_delta", "partial_json": fn_args},
                        })

            # Handle finish
            if finish:
                # Close any open block
                prev_events = state.close_block()
                for evt in prev_events:
                    yield _sse_event(evt["type"], evt)

                stop_reason = _map_stop_reason(finish) or "end_turn"
                yield _sse_event("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                })
                yield _sse_event("message_stop", {"type": "message_stop"})
                return

    # Stream ended without explicit finish — close and finalize
    prev_events = state.close_block()
    for evt in prev_events:
        yield _sse_event(evt["type"], evt)
    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield _sse_event("message_stop", {"type": "message_stop"})


def _sse_event(event_type: str, data: dict) -> bytes:
    """Format a single SSE event as bytes."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

async def proxy_anthropic_messages(request: Request) -> StreamingResponse | JSONResponse:
    """Handle Anthropic Messages API requests."""
    t0 = time.monotonic()
    body = await request.body()
    anthropic_req = json.loads(body) if body else {}
    stream = anthropic_req.get("stream", False)

    # Preprocess
    preprocess_anthropic_payload(anthropic_req)

    # Translate to OpenAI format
    openai_req, meta = _anthropic_to_openai(anthropic_req)
    meta["model"] = openai_req.get("model", "")

    # Inject hermes-agent attribution
    openai_req = _inject_hermes_attribution(openai_req)

    # EAGERLY get agent key BEFORE creating stream handler.
    # This ensures token refresh happens here (blocking) rather than
    # after the StreamingResponse is created, which would delay the
    # upstream HTTP request by several seconds.
    client = _get_proxy_client()
    t_key = time.monotonic()
    agent_key = await token_manager.ensure_valid_agent_key(client)
    key_ms = (time.monotonic() - t_key) * 1000
    inference_url = token_manager.state.effective_inference_url

    openai_body = json.dumps(openai_req).encode()
    prep_ms = (time.monotonic() - t0) * 1000

    if key_ms > 100:
        logger.info("Anthropic handler: agent_key took %.0fms (total prep %.0fms, %d bytes body)",
                     key_ms, prep_ms, len(openai_body))

    if stream:
        return StreamingResponse(
            _stream_handler(client, inference_url, agent_key, openai_body, meta),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "anthropic-version": "2023-06-01",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming
    resp = await client.post(
        f"{inference_url}/chat/completions",
        content=openai_body,
        headers={
            **_HERMES_UPSTREAM_HEADERS,
            "Authorization": f"Bearer {agent_key}",
            "Content-Type": "application/json",
        },
    )

    if resp.status_code != 200:
        try:
            error_data = resp.json()
        except Exception:
            error_data = {"error": {"message": resp.text}}
        error_msg = error_data.get("error", {}).get("message", str(error_data))
        return JSONResponse(
            status_code=resp.status_code,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": error_msg},
            },
        )

    openai_resp = resp.json()
    anthropic_resp = _openai_to_anthropic(openai_resp, meta)
    return JSONResponse(content=anthropic_resp, status_code=200)


async def _stream_handler(client, inference_url, agent_key, openai_body, meta):
    """Async generator for streaming responses."""
    t0 = time.monotonic()
    try:
        async with client.stream(
            "POST",
            f"{inference_url}/chat/completions",
            content=openai_body,
            headers={
                **_HERMES_UPSTREAM_HEADERS,
                "Authorization": f"Bearer {agent_key}",
                "Content-Type": "application/json",
            },
        ) as resp:
            connect_ms = (time.monotonic() - t0) * 1000
            if resp.status_code != 200:
                error_body = await resp.aread()
                try:
                    error_data = json.loads(error_body)
                except json.JSONDecodeError:
                    error_data = {"error": {"message": error_body.decode()}}
                err_msg = error_data.get("error", {}).get("message", str(error_data))
                yield _sse_event("error", {
                    "type": "error",
                    "error": {"type": "api_error", "message": err_msg},
                })
                return

            # Estimate input tokens
            meta["input_tokens"] = len(openai_body) // 4
            logger.info("Stream handler: upstream connect %.0fms, body %d bytes", connect_ms, len(openai_body))

            first_chunk = True
            async def chunk_iter():
                nonlocal first_chunk
                async for chunk in resp.aiter_bytes():
                    if first_chunk:
                        ttft = (time.monotonic() - t0) * 1000
                        logger.info("Stream handler: first chunk at %.0fms", ttft)
                        first_chunk = False
                    yield chunk

            async for event_bytes in _anthropic_stream_generator(chunk_iter(), meta):
                yield event_bytes

    except Exception as e:
        logger.error("Anthropic stream error: %s", e)
        yield _sse_event("error", {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        })
