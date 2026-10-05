"""Tests for Anthropic ↔ OpenAI translation layer."""

from __future__ import annotations

from typing import cast

import pytest

from olist_code.adapter import (
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
from olist_code.models import (
    AdapterConfig,
    AnthropicContentBlock,
    AnthropicInputJsonDeltaDict,
    AnthropicMessage,
    AnthropicRequest,
    AnthropicRole,
    AnthropicResponseToolContent,
    AnthropicTextDeltaDict,
    AnthropicTool,
    AnthropicToolContentBlock,
    ModelConfig,
    OpenAIResponseChunk,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def config() -> AdapterConfig:
    return AdapterConfig(
        base_url="https://api.openai.com",
        api_key="sk-test-123",
        models=ModelConfig(opus="claude-sonnet-4-20250514", sonnet="claude-sonnet-4-6"),
        tool_format="native",
        port=3080,
    )


# ── Anthropic → OpenAI: simple_text ──────────────────────────────────────────


class TestAnthropicToOpenaiSimpleText:
    def test_simple_text_request(self, config: AdapterConfig) -> None:
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[AnthropicMessage(role=AnthropicRole.user, content="Olá, como vai?")],
        )
        result = anthropic_to_openai(req, config)

        assert result.model == "claude-sonnet-4-6"
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Olá, como vai?"
        assert result.max_completion_tokens == 1024
        assert result.stream is False
        assert result.tools is None

    def test_simple_text_response(self) -> None:
        openai_resp: OpenAIResponseChunk = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1720000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Olá! Estou bem, obrigado. Como posso ajudar?"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 18, "total_tokens": 30},
        }
        result = openai_to_anthropic_response(openai_resp)

        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 1
        assert result["content"][0] == {"type": "text", "text": "Olá! Estou bem, obrigado. Como posso ajudar?"}
        assert result["usage"]["input_tokens"] == 12
        assert result["usage"]["output_tokens"] == 18

    def test_non_streaming_response_uses_message_key(self) -> None:
        """The real (non-streaming) OpenAI Chat Completions API puts the
        assistant reply under choices[0].message, not choices[0].delta —
        delta only appears in streaming chunks."""
        openai_resp: OpenAIResponseChunk = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1720000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Olá! Estou bem, obrigado."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 18, "total_tokens": 30},
        }
        result = openai_to_anthropic_response(openai_resp)

        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 1
        assert result["content"][0] == {"type": "text", "text": "Olá! Estou bem, obrigado."}


# ── Anthropic → OpenAI: with_system ──────────────────────────────────────────


class TestAnthropicToOpenaiWithSystem:
    def test_system_as_string(self, config: AdapterConfig) -> None:
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system="Você é um assistente útil da Olist.",
            messages=[AnthropicMessage(role=AnthropicRole.user, content="Qual é o meu saldo?")],
        )
        result = anthropic_to_openai(req, config)

        assert len(result.messages) == 2
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "Você é um assistente útil da Olist."
        assert result.messages[1].role == "user"
        assert result.messages[1].content == "Qual é o meu saldo?"

    def test_system_as_list_of_blocks(self, config: AdapterConfig) -> None:
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=[{"type": "text", "text": "Você é um assistente útil da Olist."}],
            messages=[AnthropicMessage(role=AnthropicRole.user, content="Qual é o meu saldo?")],
        )
        result = anthropic_to_openai(req, config)

        assert len(result.messages) == 2
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "Você é um assistente útil da Olist."


# ── Anthropic → OpenAI: content_blocks ───────────────────────────────────────


class TestAnthropicToOpenaiContentBlocks:
    def test_content_blocks_list(self, config: AdapterConfig) -> None:
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role=AnthropicRole.user,
                    content=[{"type": "text", "text": "Resuma esse documento."}],
                )
            ],
        )
        result = anthropic_to_openai(req, config)

        assert len(result.messages) == 1
        assert result.messages[0].content == "Resuma esse documento."

    def test_content_blocks_multiple(self, config: AdapterConfig) -> None:
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role=AnthropicRole.user,
                    content=[
                        {"type": "text", "text": "Resuma esse documento."},
                        {"type": "text", "text": "Foque nos pontos principais."},
                    ],
                )
            ],
        )
        result = anthropic_to_openai(req, config)

        assert len(result.messages) == 1
        content = result.messages[0].content
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Resuma esse documento."}
        assert content[1] == {"type": "text", "text": "Foque nos pontos principais."}

    def test_unknown_block_types_do_not_fail_validation(self, config: AdapterConfig) -> None:
        """Extended-thinking and image blocks must not 422 the whole request."""
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role=AnthropicRole.assistant,
                    content=[
                        {"type": "thinking", "thinking": "let me think...", "signature": "abc"},
                        {"type": "text", "text": "Aqui está a resposta."},
                    ],
                ),
                AnthropicMessage(
                    role=AnthropicRole.user,
                    content=[
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
                        {"type": "text", "text": "O que tem nessa imagem?"},
                    ],
                ),
            ],
        )
        result = anthropic_to_openai(req, config)

        assert len(result.messages) == 2
        assert result.messages[0].role == "assistant"
        assert result.messages[0].content == "Aqui está a resposta."
        assert result.messages[1].role == "user"
        user_content = result.messages[1].content
        assert isinstance(user_content, list)
        assert {"type": "text", "text": "O que tem nessa imagem?"} in user_content


# ── Anthropic → OpenAI: multi_turn ───────────────────────────────────────────


class TestAnthropicToOpenaiMultiTurn:
    def test_multi_turn_conversation(self, config: AdapterConfig) -> None:
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                AnthropicMessage(role=AnthropicRole.user, content="Qual é a capital do Brasil?"),
                AnthropicMessage(
                    role=AnthropicRole.assistant,
                    content=[{"type": "text", "text": "A capital do Brasil é Brasília."}],
                ),
                AnthropicMessage(role=AnthropicRole.user, content="E a do Chile?"),
            ],
        )
        result = anthropic_to_openai(req, config)

        assert len(result.messages) == 3
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Qual é a capital do Brasil?"
        assert result.messages[1].role == "assistant"
        assert result.messages[1].content == "A capital do Brasil é Brasília."
        assert result.messages[2].role == "user"
        assert result.messages[2].content == "E a do Chile?"


# ── Anthropic → OpenAI: with_tools ───────────────────────────────────────────


class TestAnthropicToOpenaiWithTools:
    def test_tools_translated(self, config: AdapterConfig) -> None:
        tool = AnthropicTool(
            name="get_order",
            description="Busca informações de um pedido pelo ID.",
            input_schema={
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "ID do pedido"}},
                "required": ["order_id"],
            },
        )
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=[tool],
            messages=[AnthropicMessage(role=AnthropicRole.user, content="Qual o status do pedido 12345?")],
        )
        result = anthropic_to_openai(req, config)

        assert result.tools is not None
        assert len(result.tools) == 1
        tool_def = result.tools[0]
        assert tool_def.function["name"] == "get_order"
        assert tool_def.function["description"] == "Busca informações de um pedido pelo ID."
        assert tool_def.function["parameters"]["type"] == "object"
        assert "order_id" in tool_def.function["parameters"]["properties"]

    def test_response_with_tool_use(self) -> None:
        openai_resp: OpenAIResponseChunk = {
            "id": "chatcmpl-def456",
            "object": "chat.completion",
            "created": 1720000001,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {"name": "get_order", "arguments": '{"order_id": "12345"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 85, "completion_tokens": 32, "total_tokens": 117},
        }
        result = openai_to_anthropic_response(openai_resp)

        assert result["stop_reason"] == "tool_use"
        assert len(result["content"]) == 1
        block = cast(AnthropicResponseToolContent, result["content"][0])
        assert block["type"] == "tool_use"
        assert block["id"] == "call_abc123"
        assert block["name"] == "get_order"
        assert block["input"] == {"order_id": "12345"}
        assert result["usage"]["input_tokens"] == 85
        assert result["usage"]["output_tokens"] == 32


# ── Anthropic → OpenAI: with_tool_result ─────────────────────────────────────


class TestAnthropicToOpenaiWithToolResult:
    def test_tool_result_converted_to_openai(self, config: AdapterConfig) -> None:
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                AnthropicMessage(role=AnthropicRole.user, content="Qual o status do pedido 12345?"),
                AnthropicMessage(
                    role=AnthropicRole.assistant,
                    content=[
                        {
                            "type": "tool_use",
                            "id": "toolu_01abc123",
                            "name": "get_order",
                            "input": {"order_id": "12345"},
                        }
                    ],
                ),
                AnthropicMessage(
                    role=AnthropicRole.user,
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01abc123",
                            "content": '{"status": "shipped", "tracking": "BR123456789"}',
                        }
                    ],
                ),
            ],
        )
        result = anthropic_to_openai(req, config)

        assert len(result.messages) == 3
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Qual o status do pedido 12345?"
        assert result.messages[1].role == "assistant"
        assert result.messages[1].tool_calls is not None
        assert len(result.messages[1].tool_calls) == 1
        assert result.messages[1].tool_calls[0].function["name"] == "get_order"
        assert result.messages[1].tool_calls[0].function["arguments"] == '{"order_id": "12345"}'
        assert result.messages[2].role == "tool"
        assert result.messages[2].tool_call_id == "toolu_01abc123"
        assert result.messages[2].content == '{"status": "shipped", "tracking": "BR123456789"}'


def _tool_round_request(
    call_ids: list[str], result_ids: list[str], trailing_text: str | None = None
) -> AnthropicRequest:
    results: list[AnthropicContentBlock] = [
        {"type": "tool_result", "tool_use_id": rid, "content": f"result {rid}"} for rid in result_ids
    ]
    if trailing_text is not None:
        results.append({"type": "text", "text": trailing_text})
    return AnthropicRequest(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            AnthropicMessage(role=AnthropicRole.user, content="go"),
            AnthropicMessage(
                role=AnthropicRole.assistant,
                content=[{"type": "tool_use", "id": cid, "name": "t", "input": {}} for cid in call_ids],
            ),
            AnthropicMessage(role=AnthropicRole.user, content=results),
        ],
    )


def _tool_ids(request: AnthropicRequest, config: AdapterConfig) -> list[str | None]:
    result = anthropic_to_openai(request, config)
    return [m.tool_call_id for m in result.messages if m.role == "tool"]


class TestAnthropicToOpenaiToolResultOrder:
    def test_reversed_results_follow_tool_calls_order(self, config: AdapterConfig) -> None:
        req = _tool_round_request(["a", "b"], ["b", "a"])
        assert _tool_ids(req, config) == ["a", "b"]

    def test_shuffled_results_follow_tool_calls_order(self, config: AdapterConfig) -> None:
        req = _tool_round_request(["a", "b", "c", "d", "e", "f"], ["d", "a", "f", "c", "e", "b"])
        assert _tool_ids(req, config) == ["a", "b", "c", "d", "e", "f"]

    def test_unknown_result_ids_kept_after_matched(self, config: AdapterConfig) -> None:
        req = _tool_round_request(["a", "b"], ["x", "b", "y", "a"])
        assert _tool_ids(req, config) == ["a", "b", "x", "y"]

    def test_ordered_results_unchanged(self, config: AdapterConfig) -> None:
        req = _tool_round_request(["a", "b", "c"], ["a", "b", "c"])
        result = anthropic_to_openai(req, config)
        tools = [m for m in result.messages if m.role == "tool"]
        assert [m.tool_call_id for m in tools] == ["a", "b", "c"]
        assert [m.content for m in tools] == ["result a", "result b", "result c"]

    def test_user_text_stays_after_reordered_results(self, config: AdapterConfig) -> None:
        req = _tool_round_request(["a", "b"], ["b", "a"], trailing_text="next")
        result = anthropic_to_openai(req, config)
        tail = result.messages[-3:]
        assert [m.role for m in tail] == ["tool", "tool", "user"]
        assert [m.tool_call_id for m in tail[:2]] == ["a", "b"]
        assert [m.content for m in tail[:2]] == ["result a", "result b"]
        assert tail[2].content == "next"


# ── Streaming: text ──────────────────────────────────────────────────────────


class TestStreamingText:
    def test_stream_start_event(self) -> None:
        event = build_anthropic_stream_start()
        assert event["type"] == "message_start"
        assert event["message"]["role"] == "assistant"
        assert event["message"]["content"] == []
        assert event["message"]["stop_reason"] is None
        assert "id" in event["message"]
        msg_id = event["message"]["id"]
        assert isinstance(msg_id, str) and msg_id.startswith("msg_")

    def test_stream_content_block_start_text(self) -> None:
        event = build_anthropic_content_block_start_text(0)
        assert event["type"] == "content_block_start"
        assert event["index"] == 0
        assert event["content_block"]["type"] == "text"

    def test_stream_text_delta(self) -> None:
        event = build_anthropic_text_delta(0, "RAG (Retrieval-Augmented Generation)")
        assert event["type"] == "content_block_delta"
        assert event["index"] == 0
        delta = cast(AnthropicTextDeltaDict, event["delta"])
        assert delta["type"] == "text_delta"
        assert delta["text"] == "RAG (Retrieval-Augmented Generation)"

    def test_stream_content_block_stop(self) -> None:
        event = build_anthropic_content_block_stop(0)
        assert event["type"] == "content_block_stop"
        assert event["index"] == 0

    def test_stream_message_delta(self) -> None:
        event = build_anthropic_message_delta("end_turn")
        assert event["type"] == "message_delta"
        assert event["delta"]["stop_reason"] == "end_turn"

    def test_stream_stop_event(self) -> None:
        event = build_anthropic_stream_stop()
        assert event["type"] == "message_stop"

    def test_parse_finish_reason(self) -> None:
        assert parse_openai_finish_reason("stop") == "end_turn"
        assert parse_openai_finish_reason("tool_calls") == "tool_use"
        assert parse_openai_finish_reason("length") == "max_tokens"
        assert parse_openai_finish_reason(None) == "end_turn"


# ── Streaming: tool calls ────────────────────────────────────────────────────


class TestStreamingToolCalls:
    def test_stream_tool_call_start(self) -> None:
        event = build_anthropic_content_block_start_tool(1, "toolu_abc123", "get_order")
        assert event["type"] == "content_block_start"
        assert event["index"] == 1
        cb = cast(AnthropicToolContentBlock, event["content_block"])
        assert cb["type"] == "tool_use"
        assert cb["id"] == "toolu_abc123"
        assert cb["name"] == "get_order"

    def test_stream_tool_call_delta(self) -> None:
        event = build_anthropic_tool_delta(1, '{"order_id":')
        assert event["type"] == "content_block_delta"
        assert event["index"] == 1
        delta = cast(AnthropicInputJsonDeltaDict, event["delta"])
        assert delta["type"] == "input_json_delta"
        assert delta["partial_json"] == '{"order_id":'

    def test_stream_tool_call_stop(self) -> None:
        event = build_anthropic_content_block_stop(1)
        assert event["type"] == "content_block_stop"
        assert event["index"] == 1

    def test_message_delta_tool_use(self) -> None:
        event = build_anthropic_message_delta("tool_use")
        assert event["type"] == "message_delta"
        assert event["delta"]["stop_reason"] == "tool_use"


# ── Error handling ───────────────────────────────────────────────────────────


class TestErrorTranslation:
    def test_invalid_request_error(self) -> None:
        error: dict[str, object] = {"error": {"type": "invalid_request_error", "message": "Unknown model"}}
        result = build_anthropic_error(error)
        assert result["type"] == "error"
        assert result["error"]["type"] == "invalid_request_error"
        assert result["error"]["message"] == "Unknown model"

    def test_upstream_error(self) -> None:
        error: dict[str, object] = {"error": {"type": "upstream_error", "message": "Connection refused"}}
        result = build_anthropic_error(error)
        assert result["error"]["type"] == "upstream_error"
        assert result["error"]["message"] == "Connection refused"

    def test_unknown_error_defaults(self) -> None:
        error: dict[str, object] = {}
        result = build_anthropic_error(error)
        assert result["error"]["type"] == "invalid_request_error"
        assert result["error"]["message"] == "Unknown error"


# ── OpenAI → Anthropic: finish_reason mapping ────────────────────────────────


class TestFinishReasonMapping:
    @pytest.mark.parametrize(
        ("finish_reason", "expected_stop"),
        [
            ("stop", "end_turn"),
            ("tool_calls", "tool_use"),
            ("length", "max_tokens"),
            ("content_filter", "content_filter"),
        ],
    )
    def test_finish_reason_map(self, finish_reason: str, expected_stop: str) -> None:
        openai_resp: OpenAIResponseChunk = {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": finish_reason}],
        }
        result = openai_to_anthropic_response(openai_resp)
        assert result["stop_reason"] == expected_stop

    def test_empty_choices(self) -> None:
        openai_resp: OpenAIResponseChunk = {"choices": []}
        result = openai_to_anthropic_response(openai_resp)
        assert result["stop_reason"] == "end_turn"
        assert result["content"] == []


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_tool_arguments(self) -> None:
        openai_resp: OpenAIResponseChunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call_001", "type": "function", "function": {"name": "noop", "arguments": ""}}
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        result = openai_to_anthropic_response(openai_resp)
        assert cast(AnthropicResponseToolContent, result["content"][0])["input"] == {}

    def test_malformed_tool_arguments(self) -> None:
        openai_resp: OpenAIResponseChunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call_001", "type": "function", "function": {"name": "bad", "arguments": "not json"}}
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        result = openai_to_anthropic_response(openai_resp)
        assert cast(AnthropicResponseToolContent, result["content"][0])["input"] == {"raw": "not json"}

    def test_no_usage_in_response(self) -> None:
        openai_resp: OpenAIResponseChunk = {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        }
        result = openai_to_anthropic_response(openai_resp)
        assert result["usage"]["input_tokens"] == 0
        assert result["usage"]["output_tokens"] == 0

    def test_streaming_request_flag(self, config: AdapterConfig) -> None:
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            stream=True,
            messages=[AnthropicMessage(role=AnthropicRole.user, content="Explique RAG.")],
        )
        result = anthropic_to_openai(req, config)
        assert result.stream is True

    def test_temperature_and_top_p(self, config: AdapterConfig) -> None:
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            messages=[AnthropicMessage(role=AnthropicRole.user, content="Hello")],
        )
        result = anthropic_to_openai(req, config)
        assert result.temperature == 0.7
        assert result.top_p == 0.9

    def test_stop_sequences_mapped_to_stop(self, config: AdapterConfig) -> None:
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            stream=True,
            stop_sequences=["\n\nHuman:", "\n\nAssistant:"],
            messages=[AnthropicMessage(role=AnthropicRole.user, content="Hello")],
        )
        result = anthropic_to_openai(req, config)
        assert result.stop == ["\n\nHuman:", "\n\nAssistant:"]

    def test_stop_sequences_not_forwarded_when_not_streaming(self, config: AdapterConfig) -> None:
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            stop_sequences=["\n\nHuman:", "\n\nAssistant:"],
            messages=[AnthropicMessage(role=AnthropicRole.user, content="Hello")],
        )
        result = anthropic_to_openai(req, config)
        assert "stop" not in result.model_dump(exclude_none=True)

    def test_assistant_text_only(self, config: AdapterConfig) -> None:
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                AnthropicMessage(role=AnthropicRole.user, content="Hi"),
                AnthropicMessage(role=AnthropicRole.assistant, content="Hello! How can I help?"),
            ],
        )
        result = anthropic_to_openai(req, config)
        assert result.messages[1].role == "assistant"
        assert result.messages[1].content == "Hello! How can I help?"
        assert result.messages[1].tool_calls is None

    def test_tool_format_native_preserved(self, config: AdapterConfig) -> None:
        config.tool_format = "native"
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[AnthropicMessage(role=AnthropicRole.user, content="Hi")],
        )
        result = anthropic_to_openai(req, config)
        assert result.tools is None

    def test_max_completion_tokens_mapped(self, config: AdapterConfig) -> None:
        req = AnthropicRequest(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[AnthropicMessage(role=AnthropicRole.user, content="Hi")],
        )
        result = anthropic_to_openai(req, config)
        assert result.max_completion_tokens == 2048
        assert result.max_tokens == 2048


# ── Stop sequence emulation ──────────────────────────────────────────────────


def _text_response(text: str) -> OpenAIResponseChunk:
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


class TestStopSequenceEmulationNonStreaming:
    def test_truncates_at_stop_sequence(self) -> None:
        result = openai_to_anthropic_response(_text_response("<block>no</block> trailing"), ["</block>"])
        assert result["content"] == [{"type": "text", "text": "<block>no"}]
        assert result["stop_reason"] == "stop_sequence"
        assert result["stop_sequence"] == "</block>"

    def test_earliest_stop_sequence_wins(self) -> None:
        result = openai_to_anthropic_response(_text_response("a END b STOP c"), ["STOP", "END"])
        assert result["content"] == [{"type": "text", "text": "a "}]
        assert result["stop_sequence"] == "END"

    def test_no_match_keeps_text_and_stop_reason(self) -> None:
        result = openai_to_anthropic_response(_text_response("<block>no"), ["</block>"])
        assert result["content"] == [{"type": "text", "text": "<block>no"}]
        assert result["stop_reason"] == "end_turn"
        assert result["stop_sequence"] is None

    def test_match_at_start_yields_no_text_block(self) -> None:
        result = openai_to_anthropic_response(_text_response("</block>rest"), ["</block>"])
        assert result["content"] == []
        assert result["stop_reason"] == "stop_sequence"

    def test_without_stop_sequences_stop_sequence_is_none(self) -> None:
        result = openai_to_anthropic_response(_text_response("hi"))
        assert result["stop_sequence"] is None
