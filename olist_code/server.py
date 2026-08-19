"""FastAPI application with Anthropic-to-OpenAI proxy routes."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
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
    openai_to_anthropic_response,
    parse_openai_finish_reason,
)
from .auth import AuthError
from .config import load_config
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
_access_logger = logging.getLogger("olist_code.access")


@app.middleware("http")
async def log_user_agent(request: Request, call_next):
    response = await call_next(request)
    user_agent = request.headers.get("user-agent", "-")
    _access_logger.info(
        '"%s %s HTTP/%s" %s user-agent=%r',
        request.method,
        request.url.path,
        request.scope.get("http_version", "1.1"),
        response.status_code,
        user_agent,
    )
    return response


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

    openai_request = anthropic_to_openai(request, config)
    request_data = {**openai_request.model_dump(exclude_none=True), "max_completion_tokens": 1, "stream": False}

    try:
        response = await forward_request(config, request_data)
    except httpx.RequestError as exc:
        return JSONResponse(status_code=502, content=build_anthropic_error({"error": {"type": "upstream_error", "message": str(exc)}}))

    if response.status_code != 200:
        try:
            err: dict[str, object] = response.json()
        except Exception:
            err = {"error": {"type": "upstream_error", "message": response.text or f"HTTP {response.status_code}"}}
        return JSONResponse(status_code=response.status_code, content=build_anthropic_error(err))

    try:
        data = response.json()
    except Exception:
        return JSONResponse(status_code=502, content=build_anthropic_error({"error": {"type": "upstream_error", "message": "Invalid JSON from upstream"}}))

    usage = data.get("usage") or {}
    return JSONResponse({"input_tokens": int(usage.get("prompt_tokens") or 0)})


@app.post("/v1/messages")
async def proxy_messages(request: AnthropicRequest):
    config = _app_config
    if config is None:
        raise HTTPException(
            status_code=503,
            detail="Adapter not configured. Run --init first.",
        )

    try:
        openai_request = anthropic_to_openai(request, config)
        request_data = openai_request.model_dump(exclude_none=True)

        if request.stream:
            upstream_client, upstream_response = await open_upstream_stream(config, request_data)
            if upstream_response.status_code != 200:
                error_text = await upstream_response.aread()
                await upstream_response.aclose()
                await upstream_client.aclose()
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
        anthropic_data = openai_to_anthropic_response(cast(OpenAIStreamChunk, openai_data))

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
        error_body: dict[str, object] = {"error": {"type": "upstream_error", "message": str(e)}}
        return JSONResponse(
            status_code=e.response.status_code,
            content=build_anthropic_error(error_body),
        )
    except httpx.RequestError as e:
        return JSONResponse(
            status_code=502,
            content=build_anthropic_error({"error": {"type": "upstream_error", "message": str(e)}}),
        )


def _sse(event_type: str, data: object) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _stream_response(upstream_client: httpx.AsyncClient, upstream_response: httpx.Response):
    content_block_index = 0
    text_block_started = False
    tool_block_index: dict[int, int] = {}
    finish_reason: str | None = None
    prompt_tokens = 0
    completion_tokens = 0

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
                continue

            chunk_usage = openai_chunk.get("usage") or {}
            if chunk_usage:
                prompt_tokens = int(chunk_usage.get("prompt_tokens") or 0)
                completion_tokens = int(chunk_usage.get("completion_tokens") or 0)

            choices = openai_chunk.get("choices", [])
            if not choices:
                continue

            choice: OpenAIChoiceChunk = choices[0]
            delta = choice.get("delta", {})
            fr = choice.get("finish_reason")
            if fr is not None:
                finish_reason = fr

            text = delta.get("content")
            if text is not None and text != "":
                if not text_block_started:
                    yield _sse("content_block_start", build_anthropic_content_block_start_text(content_block_index))
                    text_block_started = True
                yield _sse("content_block_delta", build_anthropic_text_delta(content_block_index, text))

            tool_calls = delta.get("tool_calls", [])
            for tc in tool_calls:
                tc_index = tc.get("index", 0)
                func = tc.get("function", {})
                tc_id = tc.get("id")

                if tc_index not in tool_block_index:
                    if text_block_started:
                        yield _sse("content_block_stop", build_anthropic_content_block_stop(content_block_index))
                        text_block_started = False
                        content_block_index += 1

                    block_idx = content_block_index
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

        if text_block_started:
            yield _sse("content_block_stop", build_anthropic_content_block_stop(content_block_index))

        for _, block_idx in sorted(tool_block_index.items()):
            yield _sse("content_block_stop", build_anthropic_content_block_stop(block_idx))

        stop_reason = parse_openai_finish_reason(finish_reason)
        yield _sse("message_delta", build_anthropic_message_delta(stop_reason, prompt_tokens, completion_tokens))
        yield _sse("message_stop", build_anthropic_stream_stop())

    except Exception as exc:
        error_payload = build_anthropic_error({"error": {"type": "stream_error", "message": str(exc)}})
        yield _sse("error", error_payload)
    finally:
        await upstream_response.aclose()
        await upstream_client.aclose()


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
        yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"
    finally:
        await upstream_response.aclose()
        await upstream_client.aclose()


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

    try:
        openai_request = anthropic_to_openai(anthropic_req, config)
        request_data = openai_request.model_dump(exclude_none=True)

        if anthropic_req.stream:
            upstream_client, upstream_response = await open_upstream_stream(config, request_data)
            if upstream_response.status_code != 200:
                error_text = await upstream_response.aread()
                await upstream_response.aclose()
                await upstream_client.aclose()
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
            raise HTTPException(status_code=502, detail=f"Empty or invalid JSON response (HTTP {response.status_code})")
        return JSONResponse(status_code=200, content=openai_data)

    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


from .proxy import forward_request, open_upstream_stream  # noqa: E402
