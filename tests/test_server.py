"""Tests for Anthropic SSE stream built from OpenAI stream chunks."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx

from olist_code.server import _stream_response


class FakeUpstreamResponse:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
        self.closed = False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self) -> None:
        self.closed = True


class FakeClient:
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
    upstream = FakeUpstreamResponse(chunks)
    client = FakeClient()
    events: list[tuple[str, dict[str, Any]]] = []
    async for raw in _stream_response(cast(httpx.AsyncClient, client), cast(httpx.Response, upstream)):
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


async def test_two_tool_calls_get_distinct_indexes() -> None:
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


async def test_text_then_tool() -> None:
    events = await _collect([_text("hi"), _tool(0, "call_a", "read", "{}"), _finish("tool_calls")])

    _assert_well_formed(events)
    starts = [data for name, data in events if name == "content_block_start"]
    assert [(s["index"], s["content_block"]["type"]) for s in starts] == [(0, "text"), (1, "tool_use")]


async def test_tool_then_text_opens_new_text_block() -> None:
    events = await _collect([_tool(0, "call_a", "read", "{}"), _text("done"), _finish("stop")])

    _assert_well_formed(events)
    starts = [data for name, data in events if name == "content_block_start"]
    assert [(s["index"], s["content_block"]["type"]) for s in starts] == [(0, "tool_use"), (1, "text")]
    text_deltas = [data for name, data in events if name == "content_block_delta" and data["index"] == 1]
    assert text_deltas[0]["delta"]["text"] == "done"


async def test_text_only_reply() -> None:
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
