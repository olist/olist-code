"""FastAPI application with Anthropic-to-OpenAI proxy routes."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, cast

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .adapter import (
    anthropic_to_openai,
    build_anthropic_content_block_start_text,
    build_anthropic_content_block_start_tool,
    build_anthropic_content_block_stop,
    build_anthropic_error,
    build_anthropic_message_delta,
    build_anthropic_stream_start,
    build_anthropic_stream_stop,
    build_anthropic_text_delta,
    build_anthropic_tool_delta,
    find_stop_sequence,
    openai_to_anthropic_response,
    parse_openai_finish_reason,
)
from .auth import AuthError
from .config import claude_family_models, load_config
from .logging_setup import request_id
from .models import (
    AdapterConfig,
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicRequest,
    AnthropicRole,
    AnthropicTool,
    JsonDict,
    OpenAIChatCompletionsRequest,
    OpenAIChoiceChunk,
    OpenAIStreamChunk,
    OpenAIToolDef,
)

_app_config: AdapterConfig | None = None


def set_app_config(config: AdapterConfig) -> None:
    global _app_config
    _app_config = config


# Gateway model ids, fetched on the first Claude-family request; None until a fetch succeeds.
_gateway_model_ids: set[str] | None = None
_logger = logging.getLogger("olist_code.server")


async def _gateway_models(config: AdapterConfig) -> set[str] | None:
    global _gateway_model_ids
    if _gateway_model_ids is None:
        try:
            _gateway_model_ids = {m["id"] for m in await fetch_gateway_models(config) if m.get("id")}
        except Exception as exc:
            _logger.warning("Could not fetch gateway models, skipping model reroute: %s", exc)
    return _gateway_model_ids


async def _reroute_unknown_claude_model(request: AnthropicRequest, config: AdapterConfig) -> None:
    """Swap a Claude id the gateway doesn't serve (e.g. `--model claude-opus-5-5`) for its family's model.

    Same mapping as ANTHROPIC_DEFAULT_*_MODEL, which only covers the `opus`/`sonnet`/`haiku` aliases.
    """
    lowered = request.model.lower()
    family = next((f for f in ("opus", "sonnet", "haiku") if f in lowered), None)
    if family is None:
        return
    known = await _gateway_models(config)
    if known is None or request.model in known:
        return
    target = claude_family_models(config.models)[family]
    _logger.info("Rerouting unknown model %s to %s", request.model, target)
    request.model = target


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _app_config
    if _app_config is None:
        _app_config = load_config()
    yield


app = FastAPI(
    title="Olist Code Client",
    description="Proxy that translates Anthropic Messages API to OpenAI Chat Completions",
    version="2.2.0",
    lifespan=lifespan,
)

_access_logger = logging.getLogger("olist_code.access")

_UPSTREAM_PATH = "/v1/chat/completions"
_ERROR_BODY_LIMIT = 2000
_INVALID_JSON_LIMIT = 500
_SHAPE_DETAILED_MESSAGES = 10


@dataclass
class _RequestLog:
    started: float = field(default_factory=time.monotonic)
    body_bytes: int | None = None
    model: str | None = None
    upstream_model: str | None = None
    stream: bool = False
    request: AnthropicRequest | None = None
    upstream_messages: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started) * 1000)

    def shape(self) -> str:
        if self.request is None:
            return f"body={self.body_bytes}B"
        return _request_shape(self.request, self.body_bytes, self.upstream_messages)


_request_log: ContextVar[_RequestLog | None] = ContextVar("olist_code_request_log", default=None)


def _current_log() -> _RequestLog:
    """The log entry of the request being served; a throwaway one outside a request."""
    return _request_log.get() or _RequestLog()


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}…(+{len(text) - limit} chars)"


def _format_counts(counts: Counter[str]) -> str:
    return ",".join(f"{block_type}×{n}" for block_type, n in counts.items())


def _block_counts(message: AnthropicMessage) -> Counter[str]:
    if isinstance(message.content, str):
        return Counter({"text": 1})
    return Counter(str(block.get("type", "?")) for block in message.content)


def _request_shape(
    request: AnthropicRequest, body_bytes: int | None = None, upstream_messages: int | None = None
) -> str:
    """Structure of a request for error logs: roles and block types only, never text, tool inputs or headers."""
    counts = [_block_counts(m) for m in request.messages]
    parts = [f"msgs={len(request.messages)}"]
    if len(request.messages) > _SHAPE_DETAILED_MESSAGES:
        parts.append(f"blocks[{_format_counts(sum(counts, Counter()))}] last{_SHAPE_DETAILED_MESSAGES}:")
    recent = zip(request.messages[-_SHAPE_DETAILED_MESSAGES:], counts[-_SHAPE_DETAILED_MESSAGES:])
    parts.extend(f"{m.role.value}[{_format_counts(c)}]" for m, c in recent)
    system = request.system
    parts.append(f"system={0 if not system else 1 if isinstance(system, str) else len(system)}")
    parts.append(f"tools={len(request.tools or [])}")
    parts.append(f"max_tokens={request.max_tokens}")
    parts.append(f"stream={'yes' if request.stream else 'no'}")
    if body_bytes is not None:
        parts.append(f"body={body_bytes}B")
    if upstream_messages is not None:
        parts.append(f"upstream_msgs={upstream_messages}")
    return " ".join(parts)


def _log_upstream_status(entry: _RequestLog, status: int, body: str) -> None:
    _logger.warning(
        "Upstream returned HTTP %s for upstream_model=%s: %s | %s",
        status,
        entry.upstream_model,
        _truncate(body, _ERROR_BODY_LIMIT) or "<empty body>",
        entry.shape(),
    )


def _log_invalid_json(status: int, body: str) -> None:
    _logger.warning(
        "Upstream returned HTTP %s with empty or invalid JSON: %r",
        status,
        _truncate(body, _INVALID_JSON_LIMIT),
    )


def _log_request_error(entry: _RequestLog, exc: httpx.RequestError) -> None:
    _logger.warning(
        "Upstream request failed: %s %r upstream=%s after %dms | %s",
        type(exc).__name__,
        exc,
        _UPSTREAM_PATH,
        entry.elapsed_ms(),
        entry.shape(),
    )


def _log_summary(method: str, path: str, status: int, entry: _RequestLog) -> None:
    parts = [f"{method} {path}"]
    if entry.model:
        model = entry.model
        if entry.upstream_model and entry.upstream_model != entry.model:
            model = f"{model}→{entry.upstream_model}"
        parts.append(f"model={model} stream={'yes' if entry.stream else 'no'}")
    parts.append(f"status={status}")
    if entry.error:
        parts.append(f"error={entry.error}")
    parts.append(f"{entry.elapsed_ms()}ms")
    if entry.input_tokens is not None or entry.output_tokens is not None:
        parts.append(f"in={entry.input_tokens or 0} out={entry.output_tokens or 0}")
    if path == "/health":
        level = logging.DEBUG
    elif status >= 400 or entry.error:
        level = logging.WARNING
    else:
        level = logging.INFO
    _access_logger.log(level, " ".join(parts))


class _RequestLogMiddleware:
    """Tags every log line with a request id and emits one summary line once the response is fully sent.

    A plain ASGI middleware, so for streams the summary waits until the stream ends.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        content_length = headers.get("content-length", "")
        entry = _RequestLog(body_bytes=int(content_length) if content_length.isdigit() else None)
        status = 500
        id_token = request_id.set(uuid.uuid4().hex[:6])
        log_token = _request_log.set(entry)
        _access_logger.debug("%s %s user-agent=%r", scope["method"], scope["path"], headers.get("user-agent", "-"))

        async def send_with_status(message: Any) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        except Exception:
            _logger.exception("Unhandled error | %s", entry.shape())
            raise
        finally:
            _log_summary(scope["method"], scope["path"], status, entry)
            _request_log.reset(log_token)
            request_id.reset(id_token)


app.add_middleware(_RequestLogMiddleware)


@app.exception_handler(AuthError)
async def auth_error_handler(_request: Request, exc: AuthError) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"error": {"type": "authentication_error", "message": str(exc)}},
    )


@app.get("/health")
async def health_check():
    from . import __version__

    return {"status": "ok", "version": __version__}


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: AnthropicRequest):
    config = _app_config
    if config is None:
        raise HTTPException(status_code=503, detail="Adapter not configured.")

    entry = _current_log()
    entry.model, entry.request = request.model, request
    await _reroute_unknown_claude_model(request, config)
    entry.upstream_model = request.model
    openai_request = anthropic_to_openai(request, config)
    entry.upstream_messages = len(openai_request.messages)
    request_data = {**openai_request.model_dump(exclude_none=True), "max_completion_tokens": 1, "stream": False}

    try:
        response = await forward_request(config, request_data)
    except httpx.RequestError as exc:
        _log_request_error(entry, exc)
        return JSONResponse(status_code=502, content=build_anthropic_error({"error": {"type": "upstream_error", "message": str(exc)}}))

    if response.status_code != 200:
        _log_upstream_status(entry, response.status_code, response.text)
        try:
            err: dict[str, object] = response.json()
        except Exception:
            err = {"error": {"type": "upstream_error", "message": response.text or f"HTTP {response.status_code}"}}
        return JSONResponse(status_code=response.status_code, content=build_anthropic_error(err))

    try:
        data = response.json()
    except Exception:
        _log_invalid_json(response.status_code, response.text)
        return JSONResponse(status_code=502, content=build_anthropic_error({"error": {"type": "upstream_error", "message": "Invalid JSON from upstream"}}))

    usage = data.get("usage") or {}
    entry.input_tokens = int(usage.get("prompt_tokens") or 0)
    return JSONResponse({"input_tokens": entry.input_tokens})


@app.post("/v1/messages")
async def proxy_messages(request: AnthropicRequest):
    config = _app_config
    if config is None:
        raise HTTPException(
            status_code=503,
            detail="Adapter not configured. Run --init first.",
        )

    entry = _current_log()
    entry.model, entry.request, entry.stream = request.model, request, request.stream
    await _reroute_unknown_claude_model(request, config)
    entry.upstream_model = request.model
    try:
        openai_request = anthropic_to_openai(request, config)
        entry.upstream_messages = len(openai_request.messages)
        request_data = openai_request.model_dump(exclude_none=True)

        if request.stream:
            upstream_client, upstream_response = await open_upstream_stream(config, request_data)
            if upstream_response.status_code != 200:
                error_text = await upstream_response.aread()
                await upstream_response.aclose()
                await upstream_client.aclose()
                _log_upstream_status(entry, upstream_response.status_code, error_text.decode(errors="replace"))
                try:
                    error_body_raw: dict[str, object] = json.loads(error_text)
                except Exception:
                    error_body_raw = {
                        "error": {
                            "type": "upstream_error",
                            "message": error_text.decode(errors="replace") or f"HTTP {upstream_response.status_code}",
                        }
                    }
                return JSONResponse(
                    status_code=upstream_response.status_code,
                    content=build_anthropic_error(error_body_raw),
                )
            return StreamingResponse(
                _stream_response(upstream_client, upstream_response),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        response = await forward_request(config, request_data)

        if response.status_code != 200:
            _log_upstream_status(entry, response.status_code, response.text)
            try:
                error_body_raw: dict[str, object] = response.json()
            except Exception:
                error_body_raw = {
                    "error": {"type": "upstream_error", "message": response.text or f"HTTP {response.status_code}"}
                }
            return JSONResponse(
                status_code=response.status_code,
                content=build_anthropic_error(error_body_raw),
            )

        try:
            openai_data = response.json()
        except Exception:
            _log_invalid_json(response.status_code, response.text)
            return JSONResponse(
                status_code=502,
                content=build_anthropic_error(
                    {
                        "error": {
                            "type": "upstream_error",
                            "message": (
                                f"Empty or invalid JSON response (HTTP {response.status_code}): {response.text[:200]}"
                            ),
                        }
                    }
                ),
            )
        anthropic_data = openai_to_anthropic_response(cast(OpenAIStreamChunk, openai_data), request.stop_sequences)
        entry.input_tokens = anthropic_data["usage"]["input_tokens"]
        entry.output_tokens = anthropic_data["usage"]["output_tokens"]

        return JSONResponse(
            status_code=200,
            content={
                "id": openai_data.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
                "type": "message",
                "role": "assistant",
                "model": config.models.opus,
                **anthropic_data,
            },
        )

    except httpx.HTTPStatusError as e:
        _log_upstream_status(entry, e.response.status_code, e.response.text)
        error_body: dict[str, object] = {"error": {"type": "upstream_error", "message": str(e)}}
        return JSONResponse(
            status_code=e.response.status_code,
            content=build_anthropic_error(error_body),
        )
    except httpx.RequestError as e:
        _log_request_error(entry, e)
        return JSONResponse(
            status_code=502,
            content=build_anthropic_error({"error": {"type": "upstream_error", "message": str(e)}}),
        )


def _sse(event_type: str, data: object) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _stream_response(upstream_client: httpx.AsyncClient, upstream_response: httpx.Response):
    next_block_index = 0
    open_block_index: int | None = None
    text_block_index: int | None = None
    tool_block_index: dict[int, int] = {}
    finish_reason: str | None = None
    prompt_tokens = 0
    completion_tokens = 0
    entry = _current_log()

    yield _sse("message_start", build_anthropic_stream_start())

    try:
        async for line in upstream_response.aiter_lines():
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                continue
            try:
                openai_chunk = cast(OpenAIStreamChunk, json.loads(data_str))
            except json.JSONDecodeError:
                _logger.debug("Skipping non-JSON stream line: %r", _truncate(data_str, 200))
                continue

            chunk_usage = openai_chunk.get("usage") or {}
            if chunk_usage:
                prompt_tokens = int(chunk_usage.get("prompt_tokens") or 0)
                completion_tokens = int(chunk_usage.get("completion_tokens") or 0)

            choices = openai_chunk.get("choices", [])
            if not choices:
                if "error" in openai_chunk:
                    entry.error = "upstream_error"
                    _logger.warning(
                        "Upstream sent an error inside the stream: %s",
                        _truncate(json.dumps(openai_chunk["error"]), _ERROR_BODY_LIMIT),
                    )
                continue

            choice: OpenAIChoiceChunk = choices[0]
            delta = choice.get("delta", {})
            fr = choice.get("finish_reason")
            if fr is not None:
                finish_reason = fr

            text = delta.get("content")
            if text is not None and text != "":
                if text_block_index is None:
                    if open_block_index is not None:
                        yield _sse("content_block_stop", build_anthropic_content_block_stop(open_block_index))
                    text_block_index = open_block_index = next_block_index
                    next_block_index += 1
                    yield _sse("content_block_start", build_anthropic_content_block_start_text(text_block_index))
                yield _sse("content_block_delta", build_anthropic_text_delta(text_block_index, text))

            tool_calls = delta.get("tool_calls", [])
            for tc in tool_calls:
                tc_index = tc.get("index", 0)
                func = tc.get("function", {})
                tc_id = tc.get("id")

                if tc_index not in tool_block_index:
                    if open_block_index is not None:
                        yield _sse("content_block_stop", build_anthropic_content_block_stop(open_block_index))
                    text_block_index = None
                    block_idx = open_block_index = next_block_index
                    next_block_index += 1
                    tool_block_index[tc_index] = block_idx
                    yield _sse(
                        "content_block_start",
                        build_anthropic_content_block_start_tool(
                            block_idx,
                            tc_id or f"toolu_{uuid.uuid4().hex[:24]}",
                            func.get("name", "unknown_tool"),
                        ),
                    )

                block_idx = tool_block_index[tc_index]
                partial_json = func.get("arguments", "")
                if partial_json:
                    yield _sse("content_block_delta", build_anthropic_tool_delta(block_idx, partial_json))

        if open_block_index is not None:
            yield _sse("content_block_stop", build_anthropic_content_block_stop(open_block_index))

        if finish_reason == "content_filter":
            _logger.warning("Upstream stopped the stream with finish_reason=content_filter")
        if next_block_index == 0 or finish_reason is None:
            _logger.warning(
                "Stream ended with %s and %s",
                "no content" if next_block_index == 0 else f"{next_block_index} blocks",
                f"finish_reason={finish_reason}" if finish_reason else "no finish_reason",
            )

        stop_reason = parse_openai_finish_reason(finish_reason)
        yield _sse("message_delta", build_anthropic_message_delta(stop_reason, prompt_tokens, completion_tokens))
        yield _sse("message_stop", build_anthropic_stream_stop())

    except Exception as exc:
        entry.error = type(exc).__name__
        _log_stream_exception(exc, entry)
        error_payload = build_anthropic_error({"error": {"type": "stream_error", "message": str(exc)}})
        yield _sse("error", error_payload)
    finally:
        entry.input_tokens, entry.output_tokens = prompt_tokens, completion_tokens
        await upstream_response.aclose()
        await upstream_client.aclose()


def _log_stream_exception(exc: Exception, entry: _RequestLog) -> None:
    if isinstance(exc, httpx.HTTPError):
        _logger.warning("Stream failed after %dms: %s %r", entry.elapsed_ms(), type(exc).__name__, exc)
    else:
        _logger.exception("Stream failed after %dms | %s", entry.elapsed_ms(), entry.shape())


async def _openai_stream_passthrough(upstream_client: httpx.AsyncClient, upstream_response: httpx.Response):
    try:
        async for line in upstream_response.aiter_lines():
            if not line or line.startswith(":"):
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    continue
                yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        entry = _current_log()
        entry.error = type(exc).__name__
        _log_stream_exception(exc, entry)
        yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"
    finally:
        await upstream_response.aclose()
        await upstream_client.aclose()


def _apply_stop_to_chat_completion(openai_data: dict[str, Any], stop_sequences: list[str] | None) -> None:
    for choice in openai_data.get("choices") or []:
        message = choice.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            continue
        index, matched = find_stop_sequence(content, stop_sequences)
        if matched is not None:
            message["content"] = content[:index]
            message.pop("tool_calls", None)
            choice["finish_reason"] = "stop"


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    config = _app_config
    if config is None:
        raise HTTPException(status_code=503, detail="Adapter not configured.")

    try:
        body = cast(OpenAIChatCompletionsRequest, await request.json())
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    messages_raw = body.get("messages", [])

    system: str | list[JsonDict] | None = None
    anthropic_messages: list[AnthropicMessage] = []
    for msg in messages_raw:
        role: str = str(msg.get("role", "user"))
        content: str | list[JsonDict] | None = msg.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system = content
            elif isinstance(content, list):
                system = content
            continue

        if isinstance(content, str):
            content_blocks: list[JsonDict] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            content_blocks = content
        else:
            content_blocks = [{"type": "text", "text": str(content)}]

        tool_calls_raw = msg.get("tool_calls")
        if tool_calls_raw and role == "assistant":
            for tc in tool_calls_raw:
                func = tc.get("function", {})
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", str(uuid.uuid4())[:8]),
                        "name": func.get("name", "unknown"),
                        "input": json.loads(func.get("arguments", "{}")),
                    }
                )
        elif role == "tool":
            tool_call_id = str(msg.get("tool_call_id", ""))
            content_blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                }
            ]

        anthropic_role: str = "user" if role in ("user", "tool") else role
        anthropic_messages.append(
            AnthropicMessage(
                role=AnthropicRole(anthropic_role),
                content=cast(list[AnthropicContentBlock], content_blocks),
            )
        )

    tools_raw: list[OpenAIToolDef] | None = body.get("tools")
    anthropic_req = AnthropicRequest(
        model=str(body.get("model", "claude-sonnet-4-20250514")),
        messages=anthropic_messages,
        max_tokens=int(body.get("max_tokens", 0)) or int(body.get("max_completion_tokens", 0)) or 4096,
        system=system,
        tools=(
            [
                AnthropicTool(
                    name=str(func["name"]),
                    description=str(func.get("description", "")) if func.get("description") else None,
                    input_schema=func["parameters"],
                )
                for t in tools_raw
                if isinstance(func := t.get("function"), dict)
            ]
            if isinstance(tools_raw, list)
            else None
        ),
        stream=bool(body.get("stream", False)),
        temperature=float(body["temperature"]) if "temperature" in body else None,
        top_p=float(body["top_p"]) if "top_p" in body else None,
        stop_sequences=body.get("stop") if isinstance(body.get("stop"), list) else None,
    )

    entry = _current_log()
    entry.model = entry.upstream_model = anthropic_req.model
    entry.request, entry.stream = anthropic_req, anthropic_req.stream
    try:
        openai_request = anthropic_to_openai(anthropic_req, config)
        entry.upstream_messages = len(openai_request.messages)
        request_data = openai_request.model_dump(exclude_none=True)

        if anthropic_req.stream:
            upstream_client, upstream_response = await open_upstream_stream(config, request_data)
            if upstream_response.status_code != 200:
                error_text = await upstream_response.aread()
                await upstream_response.aclose()
                await upstream_client.aclose()
                _log_upstream_status(entry, upstream_response.status_code, error_text.decode(errors="replace"))
                try:
                    resp_content: dict[str, object] = json.loads(error_text)
                except Exception:
                    resp_content = {
                        "error": {
                            "message": error_text.decode(errors="replace") or f"HTTP {upstream_response.status_code}"
                        }
                    }
                return JSONResponse(status_code=upstream_response.status_code, content=resp_content)
            return StreamingResponse(
                _openai_stream_passthrough(upstream_client, upstream_response),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        response = await forward_request(config, request_data)

        if response.status_code != 200:
            _log_upstream_status(entry, response.status_code, response.text)
            try:
                resp_content = response.json()
            except Exception:
                resp_content: dict[str, object] = {
                    "error": {"message": response.text or f"HTTP {response.status_code}"}
                }
            return JSONResponse(status_code=response.status_code, content=resp_content)

        try:
            openai_data = response.json()
        except Exception:
            _log_invalid_json(response.status_code, response.text)
            raise HTTPException(status_code=502, detail=f"Empty or invalid JSON response (HTTP {response.status_code})")
        _apply_stop_to_chat_completion(openai_data, anthropic_req.stop_sequences)
        usage = openai_data.get("usage") or {}
        entry.input_tokens, entry.output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        return JSONResponse(status_code=200, content=openai_data)

    except httpx.RequestError as exc:
        _log_request_error(entry, exc)
        raise HTTPException(status_code=502, detail=str(exc))


from .proxy import fetch_gateway_models, forward_request, open_upstream_stream  # noqa: E402
