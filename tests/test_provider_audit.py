"""DeepSeek 参数、工具轮 reasoning 回写与 provider-call 审计。"""

from __future__ import annotations

import pytest

from conftest import stream_text, stream_tool, text_turn, tool_turn
from yaoagent import (
    DynamicInstructionStream,
    DynamicInstructions,
    Instructions,
    LanguageModelSession,
    LLMConfig,
    Profile,
    Tool,
    Trace,
)


class Ping(Tool):
    name: str = "ping"
    description: str = "pong"

    def call(self) -> str:
        return "pong"


class WithTool(DynamicInstructions):
    def body(self, session) -> DynamicInstructionStream:
        yield Instructions("sys")
        yield Ping()


def _profile(*, reasoning="high", temperature=0.0):
    return (
        Profile(instructions=WithTool())
        .reasoning(reasoning)
        .temperature(temperature)
    )


def test_deepseek_thinking_config_validation():
    assert LLMConfig.deepseek(thinking="disabled").extra_body == {
        "thinking": {"type": "disabled"}
    }
    with pytest.raises(ValueError):
        LLMConfig.deepseek(thinking="sometimes")


def test_disabled_thinking_forwards_body_and_omits_reasoning(fake_model, run):
    fake = fake_model([text_turn("done")])
    session = LanguageModelSession(
        _profile(), llm_config=LLMConfig.deepseek(thinking="disabled")
    )
    run(session.respond("go"))
    request = fake.chat.completions.requests[0]
    assert request["temperature"] == 0.0
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in request


def test_deepseek_tool_round_replays_reasoning_and_audits_calls(fake_model, run):
    fake = fake_model([
        tool_turn("ping", "{}", reasoning="reason-before-tool"),
        text_turn("done", reasoning="reason-final"),
    ])
    events = []
    session = LanguageModelSession(
        _profile(),
        llm_config=LLMConfig.deepseek(thinking="enabled"),
        trace=Trace(events.append, level="debug"),
    )
    assert run(session.respond("go")) == "done"

    second_messages = fake.chat.completions.requests[1]["messages"]
    assistant = next(message for message in second_messages if message["role"] == "assistant")
    assert assistant["reasoning_content"] == "reason-before-tool"
    assert session.history[1]["reasoning_content"] == "reason-before-tool"

    provider_requests = [event for event in events if event["type"] == "provider_request"]
    provider_responses = [event for event in events if event["type"] == "provider_response"]
    assert [event["round"] for event in provider_requests] == [0, 1]
    assert [event["round"] for event in provider_responses] == [0, 1]
    assert {event["request_id"] for event in provider_requests + provider_responses} == {
        provider_requests[0]["request_id"]
    }
    assert [event["provider_call_id"] for event in provider_requests] == [
        event["provider_call_id"] for event in provider_responses
    ]
    assert provider_responses[0]["reasoning_present"] is True
    assert provider_responses[0]["reasoning_content"] == "reason-before-tool"
    assert provider_responses[0]["tool_call_count"] == 1
    assert provider_requests[0]["provider"] == {
        "api_base_url": "https://api.deepseek.com",
        "timeout": 60.0,
    }
    assert provider_responses[0]["message"]["tool_calls"][0]["id"] == "call_1"
    tool_events = [event for event in events if event["type"] in {"tool_call", "tool_output"}]
    assert {event["request_id"] for event in tool_events} == {
        provider_requests[0]["request_id"]
    }
    assert {event["provider_call_id"] for event in tool_events} == {
        provider_requests[0]["provider_call_id"]
    }
    assert {event["call_id"] for event in tool_events} == {"call_1"}


def test_proxy_model_name_enables_deepseek_reasoning_replay(fake_model, run):
    fake = fake_model([
        tool_turn("ping", "{}", reasoning="proxy-reason"),
        text_turn("done"),
    ])
    config = LLMConfig(
        api_base_url="https://proxy.example/v1",
        model_name="deepseek-v4-flash",
        api_key_env_name="DEEPSEEK_API_KEY",
    )
    run(LanguageModelSession(_profile(), llm_config=config).respond("go"))
    assistant = next(
        message
        for message in fake.chat.completions.requests[1]["messages"]
        if message["role"] == "assistant"
    )
    assert assistant["reasoning_content"] == "proxy-reason"


def test_stream_tool_round_replays_reasoning(fake_model, run):
    fake = fake_model([
        stream_tool("ping", "{}", reasoning_chunks=["reason-", "stream"]),
        stream_text(answer_chunks=["done"]),
    ])
    session = LanguageModelSession(
        _profile(), llm_config=LLMConfig.deepseek(thinking="enabled")
    )

    async def collect():
        return [chunk async for chunk in session.stream_response("go")]

    assert run(collect()) == ["done"]
    assistant = next(
        message
        for message in fake.chat.completions.requests[1]["messages"]
        if message["role"] == "assistant"
    )
    assert assistant["reasoning_content"] == "reason-stream"
