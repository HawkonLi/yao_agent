"""工具：参数校验、可恢复自愈循环、max_round 优雅返回 / 默认报错。"""

from __future__ import annotations

import pytest
from conftest import text_turn, tool_turn

from yaoagent import (
    DynamicInstructions,
    DynamicInstructionStream,
    ErrorCode,
    Instructions,
    LanguageModelSession,
    LLMConfig,
    ModelError,
    Profile,
    Tool,
    ToolError,
    validate_arguments,
)


# ----------------------------------------------------------------- 纯函数：validate_arguments
def test_validate_arguments_subset():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
        "additionalProperties": False,
    }
    assert validate_arguments({"n": 3}, schema) is None
    assert "缺少必填" in validate_arguments({}, schema)
    assert "未知参数" in validate_arguments({"n": 1, "x": 2}, schema)
    assert "类型" in validate_arguments({"n": "no"}, schema)
    assert "对象" in validate_arguments([], schema)


# ----------------------------------------------------------------- 可恢复自愈
class Submit(Tool):
    name: str = "submit"
    description: str = "提交，至少 10 个"

    def call(self, n: int) -> str:
        if n < 10:
            raise ToolError("数量需≥10。", code=ErrorCode.INVALID_TOOL_ARGUMENTS, tool="submit")
        return "ok"


class Ping(Tool):
    name: str = "ping"
    description: str = "pong"

    def call(self) -> str:
        return "pong"


def _instr(tool):
    class I(DynamicInstructions):
        def body(self, session) -> DynamicInstructionStream:
            yield Instructions("sys")
            yield tool

    return I()


def _session(tool):
    return LanguageModelSession(Profile(instructions=_instr(tool)), llm_config=LLMConfig.deepseek())


def test_recoverable_self_heal(fake_model, run):
    fake = fake_model(
        [tool_turn("submit", '{"n": 3}'), tool_turn("submit", '{"n": 12}'), text_turn("done")]
    )
    answer = run(_session(Submit()).respond("go"))
    assert answer == "done"
    assert fake.chat.completions.calls == 3   # 坏→回灌→改正→答案


def test_max_round_graceful_return(fake_model, run):
    fake_model([tool_turn("ping", "{}")])   # 只给一轮工具调用
    # max_round(1)：发一次、执行工具就优雅返回（不报错）。
    answer = run(_session(Ping()).max_round(1).respond("go"))
    assert answer == ""                       # 最后一轮无文本


def test_default_respond_raises_on_runaway(fake_model, run):
    fake_model([tool_turn("ping", "{}")] * 8)  # 一直要工具，撞默认上限
    with pytest.raises(ModelError) as ei:
        run(_session(Ping()).respond("go"))
    assert ei.value.code is ErrorCode.MAX_TOOL_ROUNDS_EXCEEDED
