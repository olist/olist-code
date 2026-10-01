"""Tests for the proxy routes: SSE stream building and stop sequence emulation."""

from __future__ import annotations

import json
import logging
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


class _FakeRawResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def json(self) -> Any:
        return json.loads(self.text)


class _FakeStreamResponse(_FakeUpstreamResponse):
    status_code = 200


_SECRET_PROMPT = "my confidential prompt text"


def _post(path: str = "/v1/messages", stream: bool = False, **extra: Any) -> httpx.Response:
    return TestClient(server.app, raise_server_exceptions=False).post(
        path,
        json={
            "model": "grok-4.6",
            "max_tokens": 64,
            "stream": stream,
            "messages": [{"role": "user", "content": _SECRET_PROMPT}],
            **extra,
        },
    )


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch, config: AdapterConfig) -> dict[str, Any]:
    state: dict[str, Any] = {"response": _FakeRawResponse(200, json.dumps(_openai_text("ok"))), "chunks": []}

    async def fake_forward(_config: AdapterConfig, _request_data: dict[str, Any]) -> Any:
        if isinstance(state["response"], Exception):
            raise state["response"]
        return state["response"]

    async def fake_open_stream(_config: AdapterConfig, _request_data: dict[str, Any]) -> Any:
        return _FakeClient(), _FakeStreamResponse(state["chunks"])

    monkeypatch.setattr(server, "forward_request", fake_forward)
    monkeypatch.setattr(server, "open_upstream_stream", fake_open_stream)
    monkeypatch.setattr(server, "_app_config", config)
    return state


def _messages(caplog: pytest.LogCaptureFixture, logger: str, level: int = logging.DEBUG) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == logger and r.levelno >= level]


class TestErrorLogging:
    def test_upstream_error_logs_status_body_and_shape(
        self, upstream: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        upstream["response"] = _FakeRawResponse(400, json.dumps({"detail": "model grok-4.6 not allowed"}))

        resp = _post()

        assert resp.status_code == 400
        [line] = _messages(caplog, "olist_code.server", logging.WARNING)
        assert "400" in line
        assert "model grok-4.6 not allowed" in line
        assert "upstream_model=grok-4.6" in line
        assert "user[text×1]" in line

    def test_upstream_error_body_is_truncated(self, upstream: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
        upstream["response"] = _FakeRawResponse(500, "x" * 10_000)

        _post()

        [line] = _messages(caplog, "olist_code.server", logging.WARNING)
        assert len(line) < 3_000

    def test_request_error_logs_exception_class(
        self, upstream: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        upstream["response"] = httpx.ReadTimeout("")

        resp = _post()

        assert resp.status_code == 502
        [line] = _messages(caplog, "olist_code.server", logging.WARNING)
        assert "ReadTimeout" in line
        assert "/v1/chat/completions" in line

    def test_invalid_json_logs_status_and_text(
        self, upstream: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        upstream["response"] = _FakeRawResponse(200, "<html>gateway hiccup</html>")

        resp = _post()

        assert resp.status_code == 502
        [line] = _messages(caplog, "olist_code.server", logging.WARNING)
        assert "gateway hiccup" in line

    def test_unexpected_exception_logs_traceback_and_shape(
        self, upstream: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def boom(*_args: Any) -> Any:
            raise ValueError("conversion failed")

        monkeypatch.setattr(server, "anthropic_to_openai", boom)

        resp = _post()

        assert resp.status_code == 500
        [record] = [r for r in caplog.records if r.name == "olist_code.server" and r.levelno >= logging.ERROR]
        assert record.exc_info is not None
        assert "user[text×1]" in record.getMessage()

    def test_count_tokens_upstream_error_is_logged(
        self, upstream: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        upstream["response"] = _FakeRawResponse(429, json.dumps({"detail": "slow down"}))

        _post("/v1/messages/count_tokens")

        [line] = _messages(caplog, "olist_code.server", logging.WARNING)
        assert "429" in line
        assert "slow down" in line

    def test_chat_completions_upstream_error_is_logged(
        self, upstream: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        upstream["response"] = _FakeRawResponse(400, json.dumps({"detail": "bad tools"}))

        _post("/v1/chat/completions")

        [line] = _messages(caplog, "olist_code.server", logging.WARNING)
        assert "bad tools" in line

    def test_logs_never_contain_prompt_or_token(
        self, upstream: dict[str, Any], config: AdapterConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        config.api_key = "sk-very-secret-token"
        upstream["response"] = _FakeRawResponse(400, json.dumps({"detail": "nope"}))

        _post()
        upstream["response"] = httpx.ConnectError("refused")
        _post()

        assert caplog.records
        assert _SECRET_PROMPT not in caplog.text
        assert "sk-very-secret-token" not in caplog.text


class TestStreamLogging:
    async def test_error_chunk_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        await _collect([_text("hi"), {"error": {"message": "rate limited upstream"}}])

        assert any("rate limited upstream" in m for m in _messages(caplog, "olist_code.server", logging.WARNING))

    async def test_content_filter_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        await _collect([_text("hi"), _finish("content_filter")])

        assert any("content_filter" in m for m in _messages(caplog, "olist_code.server", logging.WARNING))

    async def test_empty_stream_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        await _collect([])

        assert any("no content" in m for m in _messages(caplog, "olist_code.server", logging.WARNING))

    async def test_normal_stream_logs_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        await _collect([_text("hi"), _finish("stop")])

        assert _messages(caplog, "olist_code.server", logging.WARNING) == []

    async def test_exception_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        class _Broken(_FakeUpstreamResponse):
            async def aiter_lines(self):
                yield f"data: {json.dumps(_text('hi'))}"
                raise httpx.ReadError("")

        events = [
            raw
            async for raw in server._stream_response(
                cast(httpx.AsyncClient, _FakeClient()), cast(httpx.Response, _Broken([]))
            )
        ]

        assert "event: error" in events[-1]
        assert any("ReadError" in m for m in _messages(caplog, "olist_code.server", logging.WARNING))


class TestRequestSummary:
    def test_one_summary_line_per_request(self, upstream: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)

        _post()

        [line] = _messages(caplog, "olist_code.access", logging.INFO)
        assert "POST /v1/messages" in line
        assert "grok-4.6" in line
        assert "stream=no" in line
        assert "status=200" in line
        assert "ms" in line
        assert "in=3 out=4" in line

    def test_stream_summary_reflects_mid_stream_error(
        self, upstream: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        upstream["chunks"] = [_text("hi"), {"error": {"message": "overloaded"}}]

        _post(stream=True)

        [line] = _messages(caplog, "olist_code.access", logging.INFO)
        assert "stream=yes" in line
        assert "status=200" in line
        assert "error=" in line

    def test_stream_summary_has_tokens(self, upstream: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        upstream["chunks"] = [_text("hi"), {**_finish("stop"), "usage": {"prompt_tokens": 7, "completion_tokens": 2}}]

        _post(stream=True)

        [line] = _messages(caplog, "olist_code.access", logging.INFO)
        assert "in=7 out=2" in line
        assert "error=" not in line

    def test_health_is_not_logged_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)

        TestClient(server.app).get("/health")

        assert _messages(caplog, "olist_code.access", logging.INFO) == []


class TestRequestShape:
    def test_counts_blocks_per_message_without_content(self) -> None:
        request = server.AnthropicRequest.model_validate(
            {
                "model": "m",
                "max_tokens": 10,
                "system": [{"type": "text", "text": "sys"}],
                "tools": [{"name": "read", "input_schema": {}}],
                "messages": [
                    {"role": "user", "content": "hello there"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "x"},
                            {"type": "tool_use", "id": "1", "name": "read", "input": {"p": "secret"}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "1", "content": "data"}] * 3,
                    },
                ],
            }
        )

        shape = server._request_shape(request, body_bytes=1234, upstream_messages=5)

        assert "msgs=3" in shape
        assert "user[text×1] assistant[text×1,tool_use×1] user[tool_result×3]" in shape
        assert "system=1" in shape
        assert "tools=1" in shape
        assert "max_tokens=10" in shape
        assert "body=1234B" in shape
        assert "upstream_msgs=5" in shape
        assert "hello" not in shape
        assert "secret" not in shape

    def test_long_conversations_are_summarized(self) -> None:
        request = server.AnthropicRequest.model_validate(
            {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "x"}] * 50}
        )

        shape = server._request_shape(request)

        assert "msgs=50" in shape
        assert shape.count("user[text×1]") == 10
        assert "text×50" in shape
