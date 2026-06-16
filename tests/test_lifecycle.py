"""生命周期钩子触发顺序 + 流式 answer/reasoning 钩子（用假模型）。"""

from __future__ import annotations

from conftest import stream_text, text_turn, tool_turn

from yaoagent import (
    DynamicInstructions,
    DynamicInstructionStream,
    Instructions,
    LanguageModelSession,
    LLMConfig,
    Profile,
    Tool,
)


class Ping(Tool):
    name: str = "ping"
    description: str = "返回 pong"

    def call(self) -> str:
        return "pong"


class WithTool(DynamicInstructions):
    def body(self, session) -> DynamicInstructionStream:
        yield Instructions("sys")
        yield Ping()


def _session(profile):
    return LanguageModelSession(profile, llm_config=LLMConfig.deepseek())


def test_firing_order(fake_model, run):
    fake_model([tool_turn("ping", "{}"), text_turn("final")])
    order = []
    profile = (
        Profile(instructions=WithTool())
        .on_prompt(lambda p: order.append("prompt"))
        .on_tool_call(lambda c: order.append("tool_call"))
        .on_tool_output(lambda c, o: order.append("tool_output"))
        .on_response(lambda t: order.append("response"))
    )
    answer = run(_session(profile).respond("hi"))
    assert answer == "final"
    assert order == ["prompt", "tool_call", "tool_output", "response"]


def test_usage_accumulates_across_rounds(fake_model, run):
    fake_model([tool_turn("ping", "{}", usage=(10, 5, 15)), text_turn("final", usage=(20, 8, 28))])
    answer = run(_session(Profile(instructions=WithTool())).respond("hi"))
    assert answer.usage.total_tokens == 43
    assert answer.usage.prompt_tokens == 30


def test_streaming_splits_answer_and_reasoning(fake_model, run):
    fake_model([stream_text(answer_chunks=["Hel", "lo"], reasoning_chunks=["think-a", "think-b"])])
    answers, reasonings = [], []
    profile = (
        Profile(instructions=WithTool())
        .on_response_stream(lambda c: answers.append(c))
        .on_reasoning_stream(lambda c: reasonings.append(c))
    )

    async def drive():
        out = []
        async for delta in _session(profile).stream_response("hi"):
            out.append(delta)
        return out

    yielded = run(drive())
    # 产出的流只含答案，不含 reasoning。
    assert yielded == ["Hel", "lo"]
    assert answers == ["Hel", "lo"]
    assert reasonings == ["think-a", "think-b"]
