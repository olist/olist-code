"""Tests for the proxy routes: SSE stream building and stop sequence emulation."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient

from olist_code import server
from olist_code.models import AdapterConfig, ModelConfig


class _FakeUpstreamResponse:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
        self.closed = False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self) -> None:
        self.closed = True


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def _text(content: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": content}, "finish_reason": None}]}


def _tool(tc_index: int, tc_id: str | None, name: str | None, args: str) -> dict[str, Any]:
    func: dict[str, Any] = {"arguments": args}
    if name is not None:
        func["name"] = name
    tc: dict[str, Any] = {"index": tc_index, "function": func}
    if tc_id is not None:
        tc["id"] = tc_id
    return {"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": None}]}


def _finish(reason: str) -> dict[str, Any]:
    return {"choices": [{"delta": {}, "finish_reason": reason}]}


async def _collect(chunks: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    upstream = _FakeUpstreamResponse(chunks)
    client = _FakeClient()
    events: list[tuple[str, dict[str, Any]]] = []
    async for raw in server._stream_response(cast(httpx.AsyncClient, client), cast(httpx.Response, upstream)):
        event_line, data_line = raw.strip().split("\n")
        events.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    assert upstream.closed
    assert client.closed
    return events


def _block_events(events: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, int]]:
    return [(name, data["index"]) for name, data in events if name.startswith("content_block_")]


def _assert_well_formed(events: list[tuple[str, dict[str, Any]]]) -> None:
    open_index: int | None = None
    started: list[int] = []
    stopped: list[int] = []
    for name, data in events:
        if name == "content_block_start":
            assert open_index is None
            open_index = data["index"]
            started.append(data["index"])
        elif name == "content_block_delta":
            assert data["index"] == open_index
        elif name == "content_block_stop":
            assert data["index"] == open_index
            stopped.append(data["index"])
            open_index = None
    assert open_index is None
    assert started == list(range(len(started)))
    assert stopped == started


class TestStreaming:
    async def test_two_tool_calls_get_distinct_indexes(self) -> None:
        events = await _collect(
            [
                _tool(0, "call_a", "read", '{"p":'),
                _tool(0, None, None, '"a"}'),
                _tool(1, "call_b", "grep", '{"q":"b"}'),
                _finish("tool_calls"),
            ]
        )

        _assert_well_formed(events)
        starts = [data for name, data in events if name == "content_block_start"]
        assert [s["index"] for s in starts] == [0, 1]
        assert [s["content_block"]["id"] for s in starts] == ["call_a", "call_b"]
        assert _block_events(events) == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
            ("content_block_delta", 0),
            ("content_block_stop", 0),
            ("content_block_start", 1),
            ("content_block_delta", 1),
            ("content_block_stop", 1),
        ]

    async def test_text_then_tool(self) -> None:
        events = await _collect([_text("hi"), _tool(0, "call_a", "read", "{}"), _finish("tool_calls")])

        _assert_well_formed(events)
        starts = [data for name, data in events if name == "content_block_start"]
        assert [(s["index"], s["content_block"]["type"]) for s in starts] == [(0, "text"), (1, "tool_use")]

    async def test_tool_then_text_opens_new_text_block(self) -> None:
        events = await _collect([_tool(0, "call_a", "read", "{}"), _text("done"), _finish("stop")])

        _assert_well_formed(events)
        starts = [data for name, data in events if name == "content_block_start"]
        assert [(s["index"], s["content_block"]["type"]) for s in starts] == [(0, "tool_use"), (1, "text")]
        text_deltas = [data for name, data in events if name == "content_block_delta" and data["index"] == 1]
        assert text_deltas[0]["delta"]["text"] == "done"

    async def test_text_only_reply(self) -> None:
        events = await _collect([_text("hel"), _text("lo"), _finish("stop")])

        _assert_well_formed(events)
        assert [name for name, _ in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert _block_events(events) == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
            ("content_block_delta", 0),
            ("content_block_stop", 0),
        ]


class _FakeJsonResponse:
    status_code = 200

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.text = json.dumps(data)

    def json(self) -> dict[str, Any]:
        return self._data


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
    monkeypatch.setattr(server, "_app_config", config)
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


@pytest.fixture
def gateway_models(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"ids": ["claude-sonnet-4-6", "glm-4.6"], "calls": 0, "fail": False}

    async def fake_fetch(_config: AdapterConfig) -> list[dict[str, Any]]:
        state["calls"] += 1
        if state["fail"]:
            raise httpx.ConnectError("gateway down")
        return [{"id": model_id} for model_id in state["ids"]]

    monkeypatch.setattr(server, "fetch_gateway_models", fake_fetch)
    monkeypatch.setattr(server, "_gateway_model_ids", None)
    return state


def _send(model: str, path: str = "/v1/messages") -> None:
    TestClient(server.app).post(
        path, json={"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    )


class TestUnknownClaudeModelReroute:
    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            ("claude-opus-5-5", "claude-sonnet-4-20250514"),
            ("claude-sonnet-4-5", "claude-sonnet-4-6"),
            # haiku is unset in the config, so it falls back to sonnet like ANTHROPIC_DEFAULT_HAIKU_MODEL does.
            ("claude-haiku-4-5", "claude-sonnet-4-6"),
        ],
    )
    def test_unknown_claude_id_goes_to_the_family_model(
        self, captured: dict[str, Any], gateway_models: dict[str, Any], requested: str, expected: str
    ) -> None:
        captured["response"] = _openai_text("ok")
        _send(requested)
        assert captured["request"]["model"] == expected

    def test_known_claude_id_passes_through(self, captured: dict[str, Any], gateway_models: dict[str, Any]) -> None:
        captured["response"] = _openai_text("ok")
        _send("claude-sonnet-4-6")
        assert captured["request"]["model"] == "claude-sonnet-4-6"

    def test_unknown_non_claude_id_passes_through(
        self, captured: dict[str, Any], gateway_models: dict[str, Any]
    ) -> None:
        captured["response"] = _openai_text("ok")
        _send("gpt-9")
        assert captured["request"]["model"] == "gpt-9"
        assert gateway_models["calls"] == 0

    def test_passes_through_when_the_list_cannot_be_fetched(
        self, captured: dict[str, Any], gateway_models: dict[str, Any]
    ) -> None:
        gateway_models["fail"] = True
        captured["response"] = _openai_text("ok")
        _send("claude-opus-5-5")
        assert captured["request"]["model"] == "claude-opus-5-5"

    def test_fetches_the_gateway_list_once(self, captured: dict[str, Any], gateway_models: dict[str, Any]) -> None:
        captured["response"] = _openai_text("ok")
        _send("claude-opus-5-5")
        _send("claude-opus-5-5")
        assert gateway_models["calls"] == 1

    def test_count_tokens_is_rerouted_too(self, captured: dict[str, Any], gateway_models: dict[str, Any]) -> None:
        captured["response"] = {"usage": {"prompt_tokens": 5}}
        _send("claude-opus-5-5", "/v1/messages/count_tokens")
        assert captured["request"]["model"] == "claude-sonnet-4-20250514"
