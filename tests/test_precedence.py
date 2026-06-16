"""三级优先级 + 穿透默认填充（纯求值，不触模型）。"""

from __future__ import annotations

from yaoagent import (
    DynamicInstructions,
    DynamicInstructionStream,
    DynamicProfile,
    Instructions,
    LanguageModelSession,
    Profile,
)


class Inner(DynamicInstructions):
    def body(self, session) -> DynamicInstructionStream:
        yield Instructions("hi")


def _session(profile):
    return LanguageModelSession(profile)


def test_callsite_beats_using_beats_profile():
    profile = Profile(instructions=Inner()).temperature(0.9)
    s = _session(profile)

    # 仅 profile
    assert s.resolve_request("p").temperature == 0.9
    # using 覆盖 profile
    with s.using(temperature=0.5):
        assert s.resolve_request("p").temperature == 0.5
        # 调用点显式参数 > using
        assert s.resolve_request("p", temperature=0.1).temperature == 0.1
    # 离开 using 恢复
    assert s.resolve_request("p").temperature == 0.9


def test_passthrough_default_fill_inner_wins():
    # 外层穿透 reasoning，内层已设 temperature；值类作默认值，内层已显式的不被覆盖。
    class Outer(DynamicProfile):
        def body(self, session) -> Profile:
            return Profile(instructions=Inner()).temperature(0.9)

    composed = Outer().temperature(0.1).reasoning("low")
    req = _session(composed).resolve_request("p")
    assert req.temperature == 0.9   # 内层 0.9 胜（默认填充只填 None）
    assert req.reasoning == "low"   # 外层穿透补上


def test_hooks_accumulate_outer_before_inner():
    order = []

    class Outer(DynamicProfile):
        def body(self, session) -> Profile:
            return Profile(instructions=Inner()).on_prompt(lambda p: order.append("inner"))

    composed = Outer().on_prompt(lambda p: order.append("outer"))
    req = _session(composed).resolve_request("p")
    # 外层钩子排在内层之前。
    assert [h.__name__ if hasattr(h, "__name__") else "" for h in req.prompt_hooks] or True
    assert len(req.prompt_hooks) == 2
    for h in req.prompt_hooks:
        h("p")
    assert order == ["outer", "inner"]
