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
    if block.get("type") == "text":
        return {"type": "text", "text": block.get("text", "")}
    if block.get("type") == "tool_use":
        return {
            "type": "function",
            "id": str(block.get("id", "")),
            "function": {
                "name": str(block.get("name", "")),
                "arguments": json.dumps(block.get("input", {})),
            },
        }
    text: str
    content = block.get("content")
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
    if len(content) == 1 and content[0].get("type") == "text":
        return content[0].get("text", "")
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
                    if block.get("type") == "tool_result":
                        result_text: str
                        block_content = block.get("content")
                        if isinstance(block_content, str):
                            result_text = block_content
                        elif isinstance(block_content, list):
                            result_text = " ".join(
                                str(c.get("text", "")) for c in block_content
                            )
                        else:
                            result_text = ""
                        other_messages.append(
                            _OpenAIMessageDict(
                                role="tool",
                                content=result_text,
                                tool_call_id=block.get("tool_use_id"),
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
                tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]
                text_blocks = [b for b in content if b.get("type") == "text"]

                if tool_use_blocks:
                    tool_calls = [
                        OpenAIToolCall(
                            id=str(block.get("id", "")),
                            function={
                                "name": str(block.get("name", "")),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        )
                        for block in tool_use_blocks
                    ]
                    other_messages.append(
                        _OpenAIMessageDict(
                            role="assistant",
                            content=text_blocks[0].get("text") if text_blocks else None,
                            tool_calls=tool_calls,
                        )
                    )
                else:
                    text = " ".join(b.get("text", "") for b in text_blocks if b.get("text"))
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
        max_completion_tokens=max_tokens,
        max_tokens=max_tokens,
        stream=request.stream,
        temperature=request.temperature,
        top_p=request.top_p,
        tools=tools,
    )


def find_stop_sequence(text: str, stop_sequences: list[str] | None) -> tuple[int, str | None]:
    best_index = -1
    matched: str | None = None
    for seq in stop_sequences or []:
        if not seq:
            continue
        index = text.find(seq)
        if index != -1 and (best_index == -1 or index < best_index):
            best_index = index
            matched = seq
    return best_index, matched


class StopSequenceMatcher:
    """Incremental stop sequence detection for streamed text.

    Trailing text that could be the start of a stop sequence is held back until
    the next chunk disambiguates it, so a matched sequence is never emitted.
    """

    def __init__(self, stop_sequences: list[str] | None) -> None:
        self._stop_sequences = [s for s in stop_sequences or [] if s]
        self._pending = ""
        self.matched: str | None = None

    def feed(self, text: str) -> str:
        if self.matched is not None:
            return ""
        buffer = self._pending + text
        index, matched = find_stop_sequence(buffer, self._stop_sequences)
        if matched is not None:
            self.matched = matched
            self._pending = ""
            return buffer[:index]
        held = self._held_length(buffer)
        self._pending = buffer[len(buffer) - held :]
        return buffer[: len(buffer) - held]

    def flush(self) -> str:
        pending, self._pending = self._pending, ""
        return pending

    def _held_length(self, buffer: str) -> int:
        longest = 0
        for seq in self._stop_sequences:
            for length in range(min(len(seq) - 1, len(buffer)), longest, -1):
                if buffer.endswith(seq[:length]):
                    longest = length
                    break
        return longest


def openai_to_anthropic_response(
    data: OpenAIResponseChunk, stop_sequences: list[str] | None = None
) -> AnthropicResponseDict:
    choices = data.get("choices", [])
    usage = data.get("usage", {})

    if not choices:
        return {
            "content": [],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    choice = choices[0]
    # Non-streaming responses carry the message under "message"; only
    # streaming chunks use "delta". Prefer whichever is present.
    delta = choice.get("message") or choice.get("delta") or {}
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
    tool_calls = delta.get("tool_calls", [])
    stop_sequence: str | None = None
    if text:
        index, stop_sequence = find_stop_sequence(text, stop_sequences)
        if stop_sequence is not None:
            text = text[:index]
            stop_reason = "stop_sequence"
            tool_calls = []
    if text:
        content_blocks.append({"type": "text", "text": text})

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
                "stop_sequence": stop_sequence,
                "usage": anthropic_usage,
            },
        ),
    )


def _parse_function_args(arguments: str) -> dict[str, Any]:
    if not arguments:
        return {}
    try:
        return cast(dict[str, Any], json.loads(arguments))
    except (json.JSONDecodeError, TypeError):
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


def build_anthropic_message_delta(
    stop_reason: str, input_tokens: int = 0, output_tokens: int = 0, stop_sequence: str | None = None
) -> AnthropicMessageDeltaEvent:
    return cast(
        AnthropicMessageDeltaEvent,
        cast(
            object,
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
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
