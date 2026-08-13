"""Type definitions for Anthropic and OpenAI API formats."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

type JsonDict = dict[str, Any]

# ── Anthropic TypedDicts for streaming events ────────────────────────────────


class AnthropicUsageDict(TypedDict):
    input_tokens: int
    output_tokens: int


class AnthropicStreamStartMessage(TypedDict):
    id: str
    role: str
    content: list[dict[str, object]]
    model: str
    stop_reason: None
    stop_sequence: None
    usage: AnthropicUsageDict


class AnthropicMessageStartEvent(TypedDict):
    type: str
    message: AnthropicStreamStartMessage


class AnthropicTextDeltaDict(TypedDict):
    type: str
    text: str


class AnthropicInputJsonDeltaDict(TypedDict):
    type: str
    partial_json: str


class AnthropicContentBlockDeltaEvent(TypedDict):
    type: str
    index: int
    delta: AnthropicTextDeltaDict | AnthropicInputJsonDeltaDict


class AnthropicTextContentBlock(TypedDict):
    type: str
    text: str


class AnthropicToolContentBlock(TypedDict):
    type: str
    id: str
    name: str
    input: dict[str, object]


class AnthropicContentBlockStartEvent(TypedDict):
    type: str
    index: int
    content_block: AnthropicTextContentBlock | AnthropicToolContentBlock


class AnthropicContentBlockStopEvent(TypedDict):
    type: str
    index: int


class AnthropicMessageDeltaStop(TypedDict):
    stop_reason: str
    stop_sequence: None


class AnthropicMessageDeltaEvent(TypedDict):
    type: str
    delta: AnthropicMessageDeltaStop
    usage: AnthropicUsageDict


class AnthropicMessageStopDelta(TypedDict):
    stop_reason: str


class AnthropicMessageStopEvent(TypedDict):
    type: str
    delta: AnthropicMessageStopDelta


class AnthropicErrorInner(TypedDict):
    type: str
    message: str


class AnthropicErrorEvent(TypedDict):
    type: str
    error: AnthropicErrorInner


class AnthropicResponseTextContent(TypedDict):
    type: str
    text: str


class AnthropicResponseToolContent(TypedDict):
    type: str
    id: str
    name: str
    input: JsonDict


class AnthropicResponseDict(TypedDict):
    content: list[AnthropicResponseTextContent | AnthropicResponseToolContent]
    stop_reason: str
    usage: AnthropicUsageDict


# ── OpenAI TypedDicts for streaming chunks ───────────────────────────────────


class OpenAIFunctionDelta(TypedDict, total=False):
    name: str
    arguments: str


class OpenAIToolCallDelta(TypedDict, total=False):
    index: int
    id: str
    type: str
    function: OpenAIFunctionDelta


class OpenAIDelta(TypedDict, total=False):
    role: str
    content: str | None
    tool_calls: list[OpenAIToolCallDelta]


class OpenAIChoiceChunk(TypedDict, total=False):
    index: int
    delta: OpenAIDelta
    finish_reason: str
    message: OpenAIDelta


class OpenAIUsageChunk(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIStreamChunk(TypedDict, total=False):
    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIChoiceChunk]
    usage: OpenAIUsageChunk


type OpenAIResponseChunk = OpenAIStreamChunk

# ── OpenAI Chat Completions Request TypedDict ────────────────────────────────


class OpenAIToolFunctionDef(TypedDict):
    name: str
    parameters: JsonDict


class OpenAIToolFunctionDefOptional(TypedDict, total=False):
    description: str


class OpenAIToolDef(TypedDict, total=False):
    type: str
    function: OpenAIToolFunctionDef


class OpenAIChatMessageToolCall(TypedDict, total=False):
    id: str
    type: str
    function: OpenAIFunctionDelta


class OpenAIChatMessage(TypedDict, total=False):
    role: str
    content: str | list[JsonDict] | None
    name: str
    tool_calls: list[OpenAIChatMessageToolCall]
    tool_call_id: str


class OpenAIChatCompletionsRequest(TypedDict, total=False):
    model: str
    messages: list[OpenAIChatMessage]
    max_tokens: int
    max_completion_tokens: int
    stream: bool
    temperature: float
    top_p: float
    stop: list[str]
    tools: list[OpenAIToolDef]

# ── Anthropic Pydantic Types ─────────────────────────────────────────────────


class AnthropicRole(StrEnum):
    user = "user"
    assistant = "assistant"
    system = "system"


class TextBlock(BaseModel):
    type: Literal["text"]
    text: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: JsonDict


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[JsonDict] | None = None


AnthropicContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class AnthropicMessage(BaseModel):
    role: AnthropicRole
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    name: str
    description: str | None = None
    input_schema: JsonDict


class AnthropicRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    system: str | list[JsonDict] | None = None
    tools: list[AnthropicTool] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None


class AnthropicDelta(BaseModel):
    type: Literal["text_delta", "input_json_delta"]
    text: str = ""
    partial_json: str = ""


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicStreamingEvent(BaseModel):
    type: Literal[
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    index: int | None = None
    delta: AnthropicDelta | None = None
    usage: AnthropicUsage | None = None
    message: JsonDict | None = None
    content_block: JsonDict | None = None


# ── OpenAI Pydantic Types ────────────────────────────────────────────────────


class OpenAIRole(StrEnum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class OpenAIContentBlock(BaseModel):
    type: Literal["text", "tool_use"] = "text"
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: JsonDict | None = None


class OpenAIToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: dict[str, str]


class OpenAIMessage(BaseModel):
    role: OpenAIRole
    content: str | list[JsonDict] | None = None
    name: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None
    tool_call_id: str | None = None


class OpenAITool(BaseModel):
    type: Literal["function"] = "function"
    function: JsonDict


class OpenAIRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    tools: list[OpenAITool] | None = None


class OpenAIChoiceDelta(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAIChoice(BaseModel):
    index: int
    delta: OpenAIChoiceDelta
    finish_reason: str | None = None


class OpenAIResponse(BaseModel):
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"] = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: dict[str, int] | None = None


# ── Config Types ─────────────────────────────────────────────────────────────


class ModelConfig(BaseModel):
    opus: str
    sonnet: str | None = None
    haiku: str | None = None


class SSOConfig(BaseModel):
    issuer: str = "https://auth-engine.olist.com/realms/backoffice"
    client_id: str = "olist-code-client"
    callback_port: int = 53682


class TokenSet(BaseModel):
    access_token: str
    refresh_token: str = ""
    expires_at: float
    refresh_expires_at: float | None = None


class AdapterConfig(BaseModel):
    base_url: str
    # Empty api_key means: authenticate with the backoffice SSO token instead.
    api_key: str = ""
    sso: SSOConfig | None = None
    models: ModelConfig
    tool_format: Literal["native", "xml"] = "native"
    port: int = 3080
    harness: Literal["claude", "opencode", "both"] = "both"


class ClaudeSettings(BaseModel):
    env: dict[str, str] = Field(default_factory=dict)
