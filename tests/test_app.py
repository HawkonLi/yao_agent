"""App 信封、持久实例、对外通道与并发边界（用假模型）。"""

from __future__ import annotations

from conftest import stream_text, stream_tool, text_turn

from yaoagent import (
    App,
    DynamicInstructions,
    DynamicInstructionStream,
    DynamicProfile,
    Instructions,
    LanguageModelSession,
    LLMConfig,
    Profile,
    SessionGroup,
    Style,
    Tool,
)


class Sys(DynamicInstructions):
    def body(self, session) -> DynamicInstructionStream:
        yield Instructions("sys")


class PingTool(Tool):
    name: str = "ping"
    description: str = "pong"

    def call(self) -> str:
        return "pong"


class SysWithTool(DynamicInstructions):
    def body(self, session) -> DynamicInstructionStream:
        yield Instructions("sys")
        yield PingTool()


def _cfg():
    return LLMConfig.deepseek()


def test_envelope_basic(fake_model, run):
    fake_model([text_turn("hello", usage=(3, 4, 7))])

    class MyApp(App):
        def body(self):
            return LanguageModelSession(Profile(instructions=Sys()), llm_config=_cfg())

    env = run(MyApp().run("hi"))
    assert env["output"] == "hello"
    assert env["usage"]["total_tokens"] == 7
    assert env["finish_reason"] == "stop"
    assert len(env["run_id"]) == 12
    kinds = [e["event"] if "event" in e else e["type"] for e in env["events"]]
    assert "response" in kinds          # 日志事件被收进信封
    assert all("run_id" in e for e in env["events"])  # 事件都带关联 id


def test_custom_shape_via_super_run(fake_model, run):
    fake_model([text_turn("hello")])

    class MyApp(App):
        def body(self):
            return LanguageModelSession(Profile(instructions=Sys()), llm_config=_cfg())

        async def run(self, input):                 # 覆写 run + super() 拿信封再加工
            env = await super().run(input)
            return env["output"].upper()

    assert run(MyApp().run("hi")) == "HELLO"


def test_output_handler_chain(fake_model, run):
    """Profile 把 on_response 接到 io.output → App.on_output 收到（全链路）。"""
    fake_model([text_turn("hello")])
    captured = []

    class OutProfile(DynamicProfile):
        def body(self, session) -> Profile:
            io = session.runtime
            return Profile(instructions=Sys()).on_response(io.output.response)

    class MyApp(App):
        def body(self):
            return LanguageModelSession(OutProfile(), llm_config=_cfg())

        def on_output(self, event):
            captured.append(event)

    run(MyApp().run("hi"))
    assert captured and captured[0]["event"] == "response" and captured[0]["text"] == "hello"


def test_stream_text_reasoning_done(fake_model, run):
    fake_model([stream_text(answer_chunks=["Hel", "lo"], reasoning_chunks=["th-a", "th-b"])])

    class MyApp(App):
        def body(self):
            return LanguageModelSession(Profile(instructions=Sys()), llm_config=_cfg())

    async def collect():
        return [e async for e in MyApp().stream("hi")]

    events = run(collect())
    assert events[-1]["type"] == "done"
    assert "".join(e["chunk"] for e in events if e["type"] == "text") == "Hello"
    assert [e["chunk"] for e in events if e["type"] == "reasoning"] == ["th-a", "th-b"]
    assert events[-1]["output"] == "Hello"


def test_stream_tool_events(fake_model, run):
    fake_model([stream_tool("ping", "{}"), stream_text(answer_chunks=["ok"])])

    class MyApp(App):
        def body(self):
            return LanguageModelSession(Profile(instructions=SysWithTool()), llm_config=_cfg())

    async def collect():
        return [e async for e in MyApp().stream("hi")]

    events = run(collect())
    types = [e["type"] for e in events]
    assert "tool_call" in types and "tool_output" in types
    assert next(e for e in events if e["type"] == "tool_call")["name"] == "ping"
    assert next(e for e in events if e["type"] == "tool_output")["output"] == "pong"
    assert events[-1] == {"type": "done", "output": "ok", "run_id": events[-1]["run_id"]}


def test_stream_group_streams_last_member(fake_model, run):
    """串行 group：前置成员照常跑，末位（主答复者）逐 token 流式。"""
    fake_model([stream_text(answer_chunks=["re", "ply"])])

    class Mind:                       # 前置成员（桩，非会话）
        async def run(self, _input):
            return "mind-done"

    class MyApp(App):
        def body(self):
            rec = LanguageModelSession(Profile(instructions=Sys()), llm_config=_cfg())
            return SessionGroup(Mind(), rec).group_style(Style.sequential)

    async def collect():
        return [e async for e in MyApp().stream("hi")]

    events = run(collect())
    assert "".join(e["chunk"] for e in events if e["type"] == "text") == "reply"
    assert events[-1]["type"] == "done" and events[-1]["output"] == "reply"
    # 前置成员的进展事件可见。
    assert any(e["type"] == "progress" and e.get("phase") == "member_start" for e in events)


def test_same_app_reuses_session_and_history(fake_model, run):
    fake_model([text_turn("a"), text_turn("b")])

    class MyApp(App):
        def body(self):
            return LanguageModelSession(Profile(instructions=Sys()), llm_config=_cfg())

    app = MyApp()

    async def twice():
        return await app.run("x"), await app.run("y")

    e1, e2 = run(twice())
    assert e1["output"] == "a" and e2["output"] == "b"
    assert e1["run_id"] != e2["run_id"]
    assert app.runnable.history == [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "y"},
        {"role": "assistant", "content": "b"},
    ]
    session_ids = {
        event["session_id"]
        for envelope in (e1, e2)
        for event in envelope["events"]
        if event.get("session_id")
    }
    assert session_ids == {app.runnable.session_id}


def test_same_app_serializes_concurrent_runs(run):
    class Probe:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        async def run(self, value):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await __import__("asyncio").sleep(0.01)
            self.active -= 1
            return value

    class MyApp(App):
        def body(self):
            return Probe()

    app = MyApp()

    async def concurrent():
        return await __import__("asyncio").gather(app.run("a"), app.run("b"))

    outputs = run(concurrent())
    assert [item["output"] for item in outputs] == ["a", "b"]
    assert app.runnable.max_active == 1


def test_different_apps_can_run_concurrently(run):
    tracker = {"active": 0, "max_active": 0}

    class Probe:
        async def run(self, value):
            tracker["active"] += 1
            tracker["max_active"] = max(tracker["max_active"], tracker["active"])
            await __import__("asyncio").sleep(0.01)
            tracker["active"] -= 1
            return value

    class MyApp(App):
        def body(self):
            return Probe()

    async def concurrent():
        return await __import__("asyncio").gather(
            MyApp().run("a"), MyApp().run("b")
        )

    outputs = run(concurrent())
    assert [item["output"] for item in outputs] == ["a", "b"]
    assert tracker["max_active"] == 2


def test_prompt_updates_state_before_first_body(run):
    class Echo:
        def __init__(self, prefix):
            self.prefix = prefix

        async def run(self, value):
            return f"{self.prefix}:{value}"

    class MyApp(App):
        def __init__(self):
            super().__init__()
            self.prefix = None

        def prompt(self, request):
            self.prefix = request["prefix"]
            return request["text"]

        def body(self):
            assert self.prefix is not None
            return Echo(self.prefix)

    envelope = run(MyApp().run({"prefix": "ready", "text": "go"}))
    assert envelope["output"] == "ready:go"
