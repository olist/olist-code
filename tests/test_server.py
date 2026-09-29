"""Tests for stop sequence emulation in the proxy routes."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from olist_code import server
from olist_code.models import AdapterConfig, ModelConfig


class _FakeUpstreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.consumed = 0
        self.closed = False

    async def aiter_lines(self):
        for line in self._lines:
            self.consumed += 1
            yield line

    async def aclose(self) -> None:
        self.closed = True


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _FakeJsonResponse:
    status_code = 200

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.text = json.dumps(data)

    def json(self) -> dict[str, Any]:
        return self._data


def _chunk(text: str | None = None, finish_reason: str | None = None) -> str:
    delta: dict[str, Any] = {} if text is None else {"content": text}
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]})


def _parse_sse(raw: list[str]) -> list[dict[str, Any]]:
    events = []
    for item in raw:
        data_line = item.split("\n")[1]
        events.append(json.loads(data_line[len("data: ") :]))
    return events


async def _collect(gen) -> list[str]:
    return [item async for item in gen]


@pytest.fixture
def config() -> AdapterConfig:
    return AdapterConfig(
        base_url="https://api.openai.com",
        api_key="sk-test-123",
        models=ModelConfig(opus="claude-sonnet-4-20250514", sonnet="claude-sonnet-4-6"),
        tool_format="native",
        port=3080,
    )


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch, config: AdapterConfig) -> dict[str, Any]:
    state: dict[str, Any] = {"response": {}}

    async def fake_forward(_config: AdapterConfig, request_data: dict[str, Any]) -> _FakeJsonResponse:
        state["request"] = request_data
        return _FakeJsonResponse(state["response"])

    monkeypatch.setattr(server, "forward_request", fake_forward)
    server.set_app_config(config)
    return state


def _openai_text(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }


class TestNonStreamingMessages:
    def test_stop_sequence_emulated(self, captured: dict[str, Any]) -> None:
        captured["response"] = _openai_text("<block>no</block>\nextra")
        client = TestClient(server.app)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "grok-4.6",
                "max_tokens": 64,
                "stop_sequences": ["</block>"],
                "messages": [{"role": "user", "content": "classify"}],
            },
        )
        body = resp.json()
        assert "stop" not in captured["request"]
        assert body["content"] == [{"type": "text", "text": "<block>no"}]
        assert body["stop_reason"] == "stop_sequence"
        assert body["stop_sequence"] == "</block>"


class TestNonStreamingChatCompletions:
    def test_stop_emulated_in_openai_shape(self, captured: dict[str, Any]) -> None:
        captured["response"] = _openai_text("answer END ignored")
        client = TestClient(server.app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "grok-4.6", "stop": ["END"], "messages": [{"role": "user", "content": "hi"}]},
        )
        body = resp.json()
        assert "stop" not in captured["request"]
        assert body["choices"][0]["message"]["content"] == "answer "
        assert body["choices"][0]["finish_reason"] == "stop"


class TestStreamingStopSequence:
    async def test_stops_at_sequence_split_across_chunks(self) -> None:
        upstream = _FakeUpstreamResponse(
            [_chunk("<block>no</bl"), _chunk("ock> tail"), _chunk(" more"), _chunk(finish_reason="stop")]
        )
        client = _FakeClient()
        events = _parse_sse(await _collect(server._stream_response(client, upstream, ["</block>"])))

        text = "".join(e["delta"]["text"] for e in events if e["type"] == "content_block_delta")
        assert text == "<block>no"
        message_delta = next(e for e in events if e["type"] == "message_delta")
        assert message_delta["delta"] == {"stop_reason": "stop_sequence", "stop_sequence": "</block>"}
        assert events[-1]["type"] == "message_stop"
        assert [e["type"] for e in events].count("content_block_stop") == 1
        assert upstream.consumed == 2
        assert upstream.closed and client.closed

    async def test_flushes_held_text_when_no_match(self) -> None:
        upstream = _FakeUpstreamResponse([_chunk("a</b"), _chunk(finish_reason="stop")])
        events = _parse_sse(await _collect(server._stream_response(_FakeClient(), upstream, ["</block>"])))

        text = "".join(e["delta"]["text"] for e in events if e["type"] == "content_block_delta")
        assert text == "a</b"
        message_delta = next(e for e in events if e["type"] == "message_delta")
        assert message_delta["delta"] == {"stop_reason": "end_turn", "stop_sequence": None}


def _parse_openai_sse(raw: list[str]) -> list[Any]:
    return [item[len("data: ") :].strip() for item in raw]


def _chunks_json(payloads: list[str]) -> list[dict[str, Any]]:
    return [json.loads(p) for p in payloads if p != "[DONE]"]


class TestChatCompletionsStreamingStop:
    async def test_stops_at_sequence_split_across_chunks(self) -> None:
        upstream = _FakeUpstreamResponse(
            [_chunk("answer EN"), _chunk("D ignored"), _chunk(" more"), _chunk(finish_reason="stop"), "data: [DONE]"]
        )
        client = _FakeClient()
        payloads = _parse_openai_sse(await _collect(server._openai_stream_passthrough(client, upstream, ["END"])))

        chunks = _chunks_json(payloads)
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        assert text == "answer "
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert payloads[-1] == "[DONE]"
        assert upstream.consumed == 2
        assert upstream.closed and client.closed

    async def test_drops_tool_calls_after_match(self) -> None:
        tool_chunk = "data: " + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "xEND", "tool_calls": [{"index": 0, "function": {"name": "t"}}]},
                        "finish_reason": None,
                    }
                ]
            }
        )
        upstream = _FakeUpstreamResponse([tool_chunk])
        stream = server._openai_stream_passthrough(_FakeClient(), upstream, ["END"])
        payloads = _parse_openai_sse(await _collect(stream))

        chunks = _chunks_json(payloads)
        assert all("tool_calls" not in c["choices"][0]["delta"] for c in chunks)
        assert "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks) == "x"

    async def test_flushes_held_text_when_no_match(self) -> None:
        upstream = _FakeUpstreamResponse([_chunk("a EN"), _chunk(finish_reason="stop"), "data: [DONE]"])
        stream = server._openai_stream_passthrough(_FakeClient(), upstream, ["END"])
        payloads = _parse_openai_sse(await _collect(stream))

        chunks = _chunks_json(payloads)
        assert "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks) == "a EN"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert payloads[-1] == "[DONE]"

    async def test_flushes_held_text_at_end_without_finish_chunk(self) -> None:
        upstream = _FakeUpstreamResponse([_chunk("a EN")])
        stream = server._openai_stream_passthrough(_FakeClient(), upstream, ["END"])
        payloads = _parse_openai_sse(await _collect(stream))

        chunks = _chunks_json(payloads)
        assert "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks) == "a EN"

    async def test_without_stop_passes_bytes_through(self) -> None:
        raw = _chunk("answer END")
        upstream = _FakeUpstreamResponse([raw, "data: [DONE]"])
        out = await _collect(server._openai_stream_passthrough(_FakeClient(), upstream))
        assert out == [raw + "\n\n", "data: [DONE]\n\n"]
