from __future__ import annotations

"""
App 级封装：部署 / 集成的最外层外壳（≈ SwiftUI 的 `App` 协议 / `@main`）。

定位：`Session` / `SessionGroup` 是「View」（可组合的智能体逻辑）；`App` 是把它们接到
外部世界（FastAPI、推荐系统、命令行…）的边界。框架内直接 `session.respond()` 即可，
**App 是可选的**——它的价值在于：

1. **统一出口**：`run()` 返回一个标准 JSON 信封 `{run_id, output, usage, finish_reason, elapsed_ms, events}`。
2. **实例生命周期**：`body()` 声明的 Runnable 树由 App 实例持有，多次 `run()` 复用其 history/state；
   同一 App 串行执行，不同 App 可安全并发。
3. **对外通道可覆写**：`on_stream` / `on_output` / `on_log` 三个钩子决定运行**过程中**三个投递口送去哪。

`body(self)` 声明 App 持有的 Runnable 树；请求通过 `prompt(request)` 增量送入该树。
需要独立对话时创建新的 App 实例。工具产出的结构化结果由工具
**直接转发**到 `runtime.output`（`self.session.runtime.output.send(...)`），框架收进信封的 `outputs`。
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import AsyncIterator

from .group import Runnable, SessionGroup
from .runtime import Handler, Runtime, use_runtime
from .trace import Trace, run_scope


class App(ABC):
    """
    可部署的智能体应用。子类实现 `body()`；按需覆写对外通道。
    """

    # 收进信封 events 的日志级别（"info" 精简；"debug" 含工具轮次与配置快照）。
    log_level: str = "info"

    def __init__(self) -> None:
        self.app_id = uuid.uuid4().hex[:12]
        self._root: Runnable | None = None
        self._run_lock = asyncio.Lock()

    @abstractmethod
    def body(self) -> Runnable:
        """
        声明本 App 持有的 Runnable（`LanguageModelSession` 或 `SessionGroup`）。

        每个 App 实例只求值一次；多次 `run()/stream()` 复用同一棵树。
        """
        raise NotImplementedError

    def prompt(self, request) -> str:
        """request → 模型输入（默认当字符串用；结构化请求覆写之）。"""
        return str(request)

    @property
    def runnable(self) -> Runnable:
        """惰性构造并返回本 App 持有的 Runnable 树。"""
        if self._root is None:
            self._root = self.body()
        return self._root

    # —— 对外通道（覆写定制；默认空操作）——
    def on_stream(self, event: dict) -> None:
        """运行中途的流式增量（answer/reasoning 文本）。覆写以推 SSE/websocket/终端。"""

    def on_output(self, event: dict) -> None:
        """工具直接转发出来的结构化结果。覆写以转发给上游（也会收进信封 outputs）。"""

    def on_log(self, event: dict) -> None:
        """日志事件（同时会收进信封的 events）。覆写以接 Trace/监控。"""

    async def run(self, request) -> dict:
        """
        执行一次：复用 `body()` 的 Runnable 树 → 跑 → 汇成并返回标准 JSON 信封
        `{run_id, output, outputs, usage, finish_reason, elapsed_ms, events}`。
        `outputs` = 运行中工具直接转发出来的结构化结果（如推荐列表）。
        """
        run_id = uuid.uuid4().hex[:12]
        events: list[dict] = []
        outputs: list[dict] = []

        def log_sink(event: dict) -> None:
            events.append(event)
            self.on_log(event)

        def output_sink(data: dict) -> None:
            outputs.append(data)
            self.on_output(data)

        runtime = Runtime(
            log=Handler(log_sink),
            stream=Handler(self.on_stream),
            output=Handler(output_sink),
        )
        trace = Trace(log_sink, level=self.log_level, context={"app_id": self.app_id})
        start = time.perf_counter()
        async with self._run_lock:
            with use_runtime(runtime), run_scope(run_id):
                prompt = self.prompt(request)
                runnable = self.runnable
                self._attach_trace(runnable, trace)
                trace.emit("app_start")
                try:
                    result = await runnable.run(prompt)
                except Exception as exc:
                    trace.emit("app_error", level="error", error=str(exc))
                    raise
                finally:
                    trace.emit("app_end")

        usage = getattr(result, "usage", None)
        return {
            "run_id": run_id,
            "output": str(result),
            "outputs": outputs,
            "usage": vars(usage) if usage is not None else None,
            "finish_reason": getattr(result, "finish_reason", None),
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
            "events": events,
        }

    async def stream(self, request) -> AsyncIterator[dict]:
        """
        实时服务：把编排全过程以**标准 UI 事件流**逐条产出，供对话助手前端渲染。

        事件统一形如 `{"type": ..., ...}`，类型涵盖：
        `text`（流式答案增量）/ `reasoning`（思考增量）/ `tool_call` / `tool_output` /
        `output`（工具直接转发的结构化结果）/ `progress`（group/member/iteration）/ `done` / `error`。

        `body()` 为单会话时逐 token 流式产出 `text`；为 group 时产出进展/工具事件、
        最终文本随 `done` 给出（group 暂不支持逐 token 流）。
        """
        queue: asyncio.Queue = asyncio.Queue()
        run_id = uuid.uuid4().hex[:12]

        def push(event: dict) -> None:
            ui = self._to_ui(event)
            if ui is not None:
                queue.put_nowait(ui)

        # log/stream 走 _to_ui；output 是工具转发的结构化结果，直接成 {type:"output"} 事件。
        runtime = Runtime(
            log=Handler(push),
            stream=Handler(push),
            output=Handler(lambda d: queue.put_nowait({"type": "output", "run_id": run_id, **d})),
        )
        trace = Trace(push, level="debug", context={"app_id": self.app_id})

        async def drive() -> None:
            try:
                async with self._run_lock:
                    with use_runtime(runtime), run_scope(run_id):
                        prompt = self.prompt(request)
                        runnable = self.runnable
                        self._attach_trace(runnable, trace)
                        trace.emit("app_start")
                        try:
                            # 会话可流式；group 仅在 streamable 时逐 token，否则回退 run()。
                            can_stream = hasattr(runnable, "stream_response") and getattr(runnable, "streamable", True)
                            if can_stream:
                                parts: list[str] = []
                                async for delta in runnable.stream_response(prompt):
                                    parts.append(delta)
                                    queue.put_nowait({"type": "text", "chunk": delta, "run_id": run_id})
                                output = "".join(parts)
                            else:
                                output = str(await runnable.run(prompt))
                        except Exception as exc:
                            trace.emit("app_error", level="error", error=str(exc))
                            raise
                        finally:
                            trace.emit("app_end")
                        queue.put_nowait({"type": "done", "output": output, "run_id": run_id})
            except Exception as exc:  # 转成 UI 错误事件，不静默吞
                queue.put_nowait({"type": "error", "error": str(exc), "run_id": run_id})
            finally:
                queue.put_nowait(None)  # 哨兵：结束

        task = asyncio.create_task(drive())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            await task

    @staticmethod
    def _to_ui(event: dict) -> dict | None:
        """把内部 trace/handler 事件翻译成标准 UI 事件；内部事件（request/response 等）返回 None 丢弃。"""
        kind = event.get("type")
        run_id = event.get("run_id")
        if kind == "tool_call":
            return {
                "type": "tool_call",
                "name": event.get("name"),
                "arguments": event.get("arguments"),
                "run_id": run_id,
            }
        if kind == "tool_output":
            return {"type": "tool_output", "name": event.get("name"), "output": event.get("output"), "run_id": run_id}
        if kind == "reasoning_stream":
            return {"type": "reasoning", "chunk": event.get("chunk"), "run_id": run_id}
        if kind in ("group_start", "group_end", "member_start", "member_end", "iteration", "activate", "deactivate"):
            extra = {k: event[k] for k in ("member", "index", "iteration", "style", "profile") if k in event}
            return {"type": "progress", "phase": kind, "run_id": run_id, **extra}
        if kind == "error":
            return {"type": "error", **{k: v for k, v in event.items() if k not in ("ts", "level")}}
        # request / response / response_stream（文本由生成器直供，避免重复）等内部事件不进 UI。
        return None

    @staticmethod
    def _attach_trace(runnable: Runnable, trace: Trace) -> None:
        """把当前 run 的日志主干接到持久 Runnable 树。"""
        if isinstance(runnable, SessionGroup):
            runnable._trace = trace
            for member in runnable.members:
                App._attach_trace(member, trace)
        elif hasattr(runnable, "trace"):
            runnable.trace = trace
