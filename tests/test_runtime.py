"""Runtime / Handler：投递、默认值、ContextVar 隔离。"""

from __future__ import annotations

import asyncio

from yaoagent import Handler, Runtime, current_runtime, use_runtime
from yaoagent.tool import ToolCall


def test_handler_routes_to_sink():
    events = []
    h = Handler(events.append)
    h.prompt("hi")
    h.tool_call(ToolCall(name="t", arguments={"a": 1}))
    h.tool_output(ToolCall(name="t", arguments={}), "out")
    h.response_stream("de")
    h.reasoning_stream("think")
    h.response("done")
    kinds = [e["event"] for e in events]
    assert kinds == ["prompt", "tool_call", "tool_output", "response_stream", "reasoning_stream", "response"]
    assert events[1]["name"] == "t" and events[1]["arguments"] == {"a": 1}
    assert events[3]["chunk"] == "de"


def test_default_handler_is_noop_never_none():
    rt = Runtime()
    assert rt.log and rt.stream and rt.output
    # 不挂 sink 也不报错。
    rt.stream.response("x")


def test_current_runtime_default_and_override():
    default = current_runtime()
    assert isinstance(default, Runtime)

    seen = []
    custom = Runtime(stream=Handler(seen.append))
    with use_runtime(custom):
        assert current_runtime() is custom
        current_runtime().stream.response_stream("a")
    # 离开还原。
    assert current_runtime() is default
    assert seen and seen[0]["chunk"] == "a"


def test_runtime_isolated_across_tasks():
    """并发任务各自的 use_runtime 互不串（ContextVar 隔离）。"""
    results = {}

    async def worker(tag):
        seen = []
        with use_runtime(Runtime(stream=Handler(seen.append))):
            await asyncio.sleep(0)  # 让出，制造交错
            current_runtime().stream.response(tag)
            await asyncio.sleep(0)
        results[tag] = seen

    async def main():
        await asyncio.gather(worker("A"), worker("B"))

    asyncio.run(main())
    assert results["A"][0]["text"] == "A"
    assert results["B"][0]["text"] == "B"
