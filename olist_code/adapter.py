"""Translation layer between Anthropic Messages API and OpenAI Chat Completions formats."""

from __future__ import annotations

import json
import uuid
from typing import Any, cast

from .models import (
    AdapterConfig,
    AnthropicContentBlock,
    AnthropicContentBlockDeltaEvent,
    AnthropicContentBlockStartEvent,
    AnthropicContentBlockStopEvent,
    AnthropicErrorEvent,
    AnthropicMessageDeltaEvent,
    AnthropicMessageStartEvent,
    AnthropicMessageStopEvent,
    AnthropicRequest,
    AnthropicResponseDict,
    AnthropicResponseTextContent,
    AnthropicResponseToolContent,
    AnthropicTool,
    OpenAIRequest,
    OpenAIMessage,
    OpenAITool,
    OpenAIToolCall,
    OpenAIResponseChunk,
)


class _OpenAIMessageDict(dict[str, Any]):
    pass


def _block_to_openai(block: AnthropicContentBlock) -> dict[str, Any]:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {
            "type": "function",
            "id": block.id,
            "function": {
                "name": block.name,
                "arguments": json.dumps(block.input),
            },
        }
    text: str
    content = block.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = " ".join(str(c.get("text", "")) for c in content)
    else:
        text = ""
    return {"type": "text", "text": text}


def _tool_to_openai(tool: AnthropicTool) -> OpenAITool:
    return OpenAITool(
        function={
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        }
    )


def _parse_content_for_message(
    content: str | list[AnthropicContentBlock],
) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if len(content) == 1 and content[0].type == "text":
        return content[0].text
    return [_block_to_openai(b) for b in content]


def anthropic_to_openai(request: AnthropicRequest, _config: AdapterConfig) -> OpenAIRequest:
    system_parts: list[str] = []
    other_messages: list[_OpenAIMessageDict] = []

    if request.system:
        if isinstance(request.system, str):
            system_parts.append(request.system)
        else:
            for item in request.system:
                if item.get("type") == "text":
                    text_val = item.get("text", "")
                    system_parts.append(text_val if isinstance(text_val, str) else str(text_val))

    for msg in request.messages:
        role = msg.role
        content = msg.content

        if role == "system":
            parsed = _parse_content_for_message(content)
            system_parts.append(parsed if isinstance(parsed, str) else str(parsed))
        elif role == "user":
            if isinstance(content, str):
                other_messages.append(_OpenAIMessageDict(role="user", content=content))
            else:
                non_tool_blocks: list[AnthropicContentBlock] = []
                for block in content:
                    if block.type == "tool_result":
                        result_text: str
                        if isinstance(block.content, str):
                            result_text = block.content
                        elif isinstance(block.content, list):
                            result_text = " ".join(
                                str(c.get("text", "")) for c in block.content
                            )
                        else:
                            result_text = ""
                        other_messages.append(
                            _OpenAIMessageDict(
                                role="tool",
                                content=result_text,
                                tool_call_id=block.tool_use_id,
                            )
                        )
                    else:
                        non_tool_blocks.append(block)

                if non_tool_blocks:
                    other_messages.append(
                        _OpenAIMessageDict(
                            role="user",
                            content=_parse_content_for_message(non_tool_blocks),
                        )
                    )
        elif role == "assistant":
            if isinstance(content, str):
                other_messages.append(_OpenAIMessageDict(role="assistant", content=content))
            else:
                tool_use_blocks = [b for b in content if b.type == "tool_use"]
                text_blocks = [b for b in content if b.type == "text"]

                if tool_use_blocks:
                    tool_calls = [
                        OpenAIToolCall(
                            id=block.id,
                            function={
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            },
                        )
                        for block in tool_use_blocks
                    ]
                    other_messages.append(
                        _OpenAIMessageDict(
                            role="assistant",
                            content=text_blocks[0].text if text_blocks else None,
                            tool_calls=tool_calls,
                        )
                    )
                else:
                    text = " ".join(b.text for b in text_blocks if b.text)
                    other_messages.append(_OpenAIMessageDict(role="assistant", content=text))

    system_messages: list[_OpenAIMessageDict] = []
    if system_parts:
        system_messages.append(_OpenAIMessageDict(role="system", content="\n\n".join(system_parts)))

    messages = system_messages + other_messages

    tools: list[OpenAITool] | None = None
    if request.tools:
        tools = [_tool_to_openai(t) for t in request.tools]

    max_tokens = request.max_tokens

    return OpenAIRequest(
        model=request.model,
        messages=[OpenAIMessage(**msg) for msg in messages],
        # Only max_tokens, not both: some upstreams (Huawei ModelArts)
        # reject requests that set both max_tokens and
        # max_completion_tokens, and some self-hosted OpenAI-compatible
        # servers (older vLLM deployments) reject max_completion_tokens
        # outright, since they only implement the older field.
        max_tokens=max_tokens,
        stream=request.stream,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop_sequences,
        tools=tools,
    )


def openai_to_anthropic_response(data: OpenAIResponseChunk) -> AnthropicResponseDict:
    choices = data.get("choices", [])
    usage = data.get("usage", {})

    if not choices:
        return {"content": [], "stop_reason": "end_turn", "usage": {"input_tokens": 0, "output_tokens": 0}}

    choice = choices[0]
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")

    stop_reason = "end_turn"
    if finish_reason == "tool_calls":
        stop_reason = "tool_use"
    elif finish_reason == "stop":
        stop_reason = "end_turn"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    elif finish_reason == "content_filter":
        stop_reason = "content_filter"

    content_blocks: list[AnthropicResponseTextContent | AnthropicResponseToolContent] = []

    text = delta.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    tool_calls = delta.get("tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", {})
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", str(uuid.uuid4())[:8]),
                "name": func.get("name", "unknown_tool"),
                "input": _parse_function_args(func.get("arguments", "{}")),
            }
        )

    anthropic_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }

    return cast(
        AnthropicResponseDict,
        cast(
            object,
            {
                "content": content_blocks,
                "stop_reason": stop_reason,
                "usage": anthropic_usage,
            },
        ),
    )


def _parse_function_args(arguments: str) -> dict[str, Any]:
    if not arguments:
        return {}
    try:
        return cast(dict[str, Any], json.loads(arguments))
    except json.JSONDecodeError, TypeError:
        return {"raw": arguments}


def build_anthropic_text_delta(index: int, text: str) -> AnthropicContentBlockDeltaEvent:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }


def build_anthropic_tool_delta(index: int, partial_json: str) -> AnthropicContentBlockDeltaEvent:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    }


def build_anthropic_content_block_start_text(index: int) -> AnthropicContentBlockStartEvent:
    return {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "text", "text": ""},
    }


def build_anthropic_content_block_start_tool(
    index: int, tool_id: str, tool_name: str
) -> AnthropicContentBlockStartEvent:
    return {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}},
    }


def build_anthropic_content_block_stop(index: int) -> AnthropicContentBlockStopEvent:
    return {
        "type": "content_block_stop",
        "index": index,
    }


def build_anthropic_message_delta(stop_reason: str) -> AnthropicMessageDeltaEvent:
    return cast(
        AnthropicMessageDeltaEvent,
        cast(
            object,
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
        ),
    )


def parse_openai_finish_reason(finish_reason: str | None) -> str:
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def build_anthropic_stream_start() -> AnthropicMessageStartEvent:
    return {
        "type": "message_start",
        "message": {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "role": "assistant",
            "content": [],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }


def build_anthropic_stream_stop() -> AnthropicMessageStopEvent:
    return {
        "type": "message_stop",
        "delta": {"stop_reason": "end_turn"},
    }


def build_anthropic_error(error: dict[str, object]) -> AnthropicErrorEvent:
    error_inner = error.get("error", {})
    error_type: str = "invalid_request_error"
    error_msg: str = "Unknown error"
    if isinstance(error_inner, dict):
        raw_type = error_inner.get("type", "invalid_request_error")
        raw_msg = error_inner.get("message", "Unknown error")
        error_type = raw_type if isinstance(raw_type, str) else "invalid_request_error"
        error_msg = raw_msg if isinstance(raw_msg, str) else "Unknown error"

    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": error_msg,
        },
    }
