"""Group 三轴编排（用桩成员，不触模型）。"""

from __future__ import annotations

import asyncio

import pytest

from yaoagent import InputStyle, OutputStyle, Response, SessionGroup, Style, Usage, parallel


class Echo:
    """桩成员：记录收到的输入，回 f'{tag}<-{input}'。"""

    def __init__(self, tag):
        self.tag = tag
        self.seen = None

    async def run(self, input):
        self.seen = input
        return f"{self.tag}<-{input}"


def go(coro):
    return asyncio.run(coro)


def test_sequential_pipe_last():
    a, b = Echo("a"), Echo("b")
    g = SessionGroup(a, b).group_style(Style.sequential)
    out = go(g.run("x"))
    assert a.seen == "x"
    assert b.seen == "a<-x"        # 上一个输出喂下一个
    assert out == "b<-a<-x"        # last 暴露末位


def test_sequential_broadcast():
    a, b = Echo("a"), Echo("b")
    g = SessionGroup(a, b).group_style(Style.sequential).input_style(InputStyle.broadcast)
    go(g.run("x"))
    assert a.seen == "x" and b.seen == "x"   # 都拿原输入


def test_output_pick_by_reference():
    a, b = Echo("a"), Echo("b")
    g = SessionGroup(a, b).group_style(Style.sequential).output_style(OutputStyle.pick(a))
    assert go(g.run("x")) == "a<-x"


def test_output_merge():
    a, b = Echo("a"), Echo("b")
    g = SessionGroup(a, b).group_style(Style.sequential).output_style(OutputStyle.merge(lambda outs: "|".join(outs)))
    assert go(g.run("x")) == "a<-x|b<-a<-x"


def test_parallel_broadcast_merge_default():
    a, b = Echo("a"), Echo("b")
    out = go(parallel(a, b).run("x"))
    assert a.seen == "x" and b.seen == "x"
    assert "a<-x" in out and "b<-x" in out


def test_parallel_pipe_raises():
    g = SessionGroup(Echo("a"), Echo("b")).group_style(Style.parallel).input_style(InputStyle.pipe)
    with pytest.raises(ValueError):
        go(g.run("x"))


def test_loop_until():
    # 每轮把输入加一个 '*'，直到长度≥3。
    class Grow:
        async def run(self, input):
            return input + "*"

    g = SessionGroup(Grow()).group_style(Style.loop(until=lambda s: len(s) >= 3, max_iters=10))
    assert go(g.run("")) == "***"


def test_nested_group():
    inner = parallel(Echo("a"), Echo("b"))
    outer = SessionGroup(inner, Echo("c")).group_style(Style.sequential)
    out = go(outer.run("x"))
    assert out.startswith("c<-")   # 末位 c，输入是 inner 的合并结果


class UsageEcho:
    """桩成员：返回带用量的 Response。"""

    def __init__(self, tag, usage):
        self.tag = tag
        self.usage = usage

    async def run(self, input):
        return Response(self.tag, usage=Usage(*self.usage))


def test_group_returns_response_with_summed_usage():
    g = SessionGroup(UsageEcho("a", (10, 1, 11)), UsageEcho("b", (20, 2, 22)))
    out = go(g.run("x"))
    assert isinstance(out, Response)          # 统一出口
    assert out == "b"                          # 文本走 last
    assert out.usage.total_tokens == 33        # 用量是全成员累加
    assert out.usage.prompt_tokens == 30


def test_nested_group_usage_recurses():
    inner = parallel(UsageEcho("a", (5, 0, 5)), UsageEcho("b", (5, 0, 5)))
    outer = SessionGroup(inner, UsageEcho("c", (10, 0, 10))).group_style(Style.sequential)
    out = go(outer.run("x"))
    assert out.usage.total_tokens == 20        # 子组 10 + c 10，递归累加
