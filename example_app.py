"""
example_app.py —— 运行时输出封装（Runtime / Handler）与 App 级封装演示。

跑之前确保 `.env` 里有 DEEPSEEK_API_KEY（与其它 example 一致）。演示四件事：

1) `Runtime` 三个投递口（log/stream/output）+ `Handler`：把生命周期事件结构化投递出去。
2) 流式 answer / reasoning **分流**：`on_response_stream` 走答案、`on_reasoning_stream` 走思考。
3) `App`：把 session/group 包成部署外壳，统一 JSON 信封；覆写 on_stream/on_log/deliver 定制对外通道。
4) “research app”：App.body 直接返回一个 SessionGroup —— 把编排塞进去即可复用。
"""

from __future__ import annotations

import asyncio
import json

from yaoagent import (
    App,
    DynamicInstructions,
    DynamicInstructionStream,
    DynamicProfile,
    Handler,
    Instructions,
    LanguageModelSession,
    LLMConfig,
    Profile,
    Runtime,
    SessionGroup,
    Style,
    Tool,
    parallel,
    use_runtime,
)

CFG = LLMConfig.deepseek()


# ============================================================ 1+2) Runtime + 流式分流
class Assistant(DynamicInstructions):
    def body(self, session) -> DynamicInstructionStream:
        yield Instructions("你是简洁的中文助手，一句话作答。")


class StreamingProfile(DynamicProfile):
    """在 body 里从 session.runtime 取投递口，把答案/思考分别接到 stream 口。"""

    def body(self, session) -> Profile:
        io = session.runtime
        return (
            Profile(instructions=Assistant())
            .on_response_stream(io.stream.response_stream)   # 答案增量 → stream 口
            .on_reasoning_stream(io.stream.reasoning_stream)  # 思考增量 → stream 口（不同 channel）
        )


async def scene_streaming() -> None:
    print("\n=== 1+2) Runtime 投递口 + answer/reasoning 分流 ===")

    def sink(event: dict) -> None:
        if event["event"] == "response_stream":
            print(event["chunk"], end="", flush=True)
        elif event["event"] == "reasoning_stream":
            print(f"\033[90m{event['chunk']}\033[0m", end="", flush=True)  # 灰色显示思考

    rt = Runtime(stream=Handler(sink))
    session = LanguageModelSession(StreamingProfile(), llm_config=CFG)
    print("（灰色=思考，正常=答案）\n  ", end="")
    with use_runtime(rt):
        async for _ in session.stream_response("用一句话解释什么是张量。"):
            pass  # 增量已由 stream 投递口打印，这里无需再处理
    print()


# ============================================================ 3+4) App + research app
class Researcher(DynamicInstructions):
    def __init__(self, angle: str) -> None:
        self.angle = angle

    def body(self, session) -> DynamicInstructionStream:
        yield Instructions(f"你是研究员，只从「{self.angle}」角度给出 2 条要点，简洁中文。")


class Synthesizer(DynamicInstructions):
    def body(self, session) -> DynamicInstructionStream:
        yield Instructions("把收到的要点综述成一段 100 字以内的结论。")


def _researcher(angle: str) -> LanguageModelSession:
    return LanguageModelSession(Profile(instructions=Researcher(angle)).temperature(0.6))


class ResearchApp(App):
    """research app：body = 并行调研 → 综述；run() 返回标准 JSON 信封。"""

    def body(self, request) -> SessionGroup:
        pipeline = SessionGroup(
            parallel(_researcher("技术"), _researcher("商业")),
            LanguageModelSession(Profile(instructions=Synthesizer())),
        ).group_style(Style.sequential)
        return pipeline.llm_config(CFG)   # 连接向所有成员穿透

    def on_log(self, event: dict) -> None:
        # 把关键节点打到控制台（也会自动收进信封 events）。
        if event["type"] in ("group_start", "member_end", "response"):
            print(f"  · {event['type']}", {k: v for k, v in event.items() if k in ("member", "index")})


async def scene_app() -> None:
    print("\n=== 3+4) ResearchApp：编排进 App，输出标准 JSON 信封 ===")
    envelope = await ResearchApp().run("电动汽车的未来")
    print("\n信封：")
    print(json.dumps({k: v for k, v in envelope.items() if k != "events"}, ensure_ascii=False, indent=2))
    print(f"  events: {len(envelope['events'])} 条（run_id={envelope['run_id']}）")


# ============================================================ 5) 实时对话助手服务 App.stream()
class Weather(Tool):
    name: str = "weather"
    description: str = "查询某城市当前天气"

    def call(self, city: str) -> str:
        return f"{city}：晴，26°C"


class ChatAssistant(DynamicInstructions):
    def body(self, session) -> DynamicInstructionStream:
        yield Instructions("你是中文助手；需要天气时调用 weather 工具，再用一句话回答。")
        yield Weather()


class ChatApp(App):
    """实时对话助手服务：body 是会话，stream() 把工具/进展/流式文本吐成标准 UI 事件。"""

    def body(self, request) -> LanguageModelSession:
        return LanguageModelSession(Profile(instructions=ChatAssistant()), llm_config=CFG)


async def scene_serving() -> None:
    print("\n=== 5) RecSysApp 式实时服务：App.stream() 标准 UI 事件流 ===")
    async for event in ChatApp().stream("上海今天适合穿什么？先查天气。"):
        t = event["type"]
        if t == "text":
            print(event["chunk"], end="", flush=True)
        elif t == "tool_call":
            print(f"\n  🔧 调用 {event['name']}({event['arguments']})")
        elif t == "tool_output":
            print(f"  ↩ {event['output']}")
        elif t == "reasoning":
            print(f"\033[90m{event['chunk']}\033[0m", end="", flush=True)
        elif t == "done":
            print(f"\n  ✅ done (run_id={event['run_id']})")
        elif t == "error":
            print(f"\n  ❌ {event['error']}")


async def main() -> None:
    await scene_streaming()
    await scene_app()
    await scene_serving()


if __name__ == "__main__":
    asyncio.run(main())
