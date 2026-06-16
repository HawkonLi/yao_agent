from __future__ import annotations

"""
运行时 I/O 封装：把生命周期事件投递到三个“口”。

把“输出往哪送”从“生命周期里发生了什么”里拆出来，统一成一种东西——**Handler**：

- 一个 `Handler` 就是一组**形状 = 钩子**的方法（`tool_call` / `response` / `response_stream` …），
  每个方法把事件打成 dict，丢给同一个 `sink`（目的地）。所有方法默认 no-op，谁关心谁用。
- `Runtime` 只是装三个 Handler 的盒子：`log`（留档）/ `stream`（给用户实时）/ `output`（给上游系统）。
  三者的区别只在 `sink` 指向哪——这正是 `App` 子类要覆写的。
- `Runtime` **永远有默认值**：模块级默认实例 + `ContextVar`，于是任何地方（包括会话自行发起
  的请求）`current_runtime()` 都拿得到非空对象，绝不为 None。

把 handler 的某个方法塞进 Profile 的钩子即可路由：

    io = session.runtime
    Profile(instructions=...).on_response_stream(io.stream.response_stream).on_response(io.output.response)
"""

import contextvars
from contextlib import contextmanager
from typing import Any, Callable, Iterator

# 一个 sink 就是吃事件 dict 的可调用对象（与 Trace 的 sink 同形，可互通）。
Sink = Callable[[dict], None]


def _noop(event: dict) -> None:
    """默认目的地：什么都不做（保证引用 handler 永不报错）。"""


class Handler:
    """
    一个“投递口”：把生命周期事件打成 dict 送给 `sink`。

    每个方法的签名都对齐对应的钩子，所以可以直接挂：

        .on_tool_call(handler.tool_call)
        .on_response_stream(handler.response_stream)

    不关心的事件方法保持默认（仍会投递，目的地是同一个 sink；不想要就不挂那个钩子）。
    """

    def __init__(self, sink: Sink = _noop) -> None:
        self.sink = sink

    # —— 离散事件（流式 / 非流式都触发）——
    def prompt(self, prompt: str) -> None:
        self.sink({"event": "prompt", "prompt": prompt})

    def tool_call(self, call: Any) -> None:
        self.sink({"event": "tool_call", "name": call.name, "arguments": call.arguments})

    def tool_output(self, call: Any, output: str) -> None:
        self.sink({"event": "tool_output", "name": call.name, "output": output})

    def response(self, text: str) -> None:
        self.sink({"event": "response", "text": text})

    # —— 流式专有：答案 / 思考 的文本增量 ——
    def response_stream(self, chunk: str) -> None:
        self.sink({"event": "response_stream", "chunk": chunk})

    def reasoning_stream(self, chunk: str) -> None:
        self.sink({"event": "reasoning_stream", "chunk": chunk})


class Runtime:
    """
    三个投递口的容器：`log` / `stream` / `output`。

    任一未提供即用一个空操作 Handler 兜底，所以 `runtime.stream.response(...)` 永不为 None。
    通常由 `App` 在 `run()` 里按需把各口的 sink 指向自己的覆写方法。
    """

    def __init__(
        self,
        *,
        log: Handler | None = None,
        stream: Handler | None = None,
        output: Handler | None = None,
    ) -> None:
        self.log = log or Handler()        # 给开发者 / 留档（可接 Trace）
        self.stream = stream or Handler()  # 给最终用户（实时增量）
        self.output = output or Handler()  # 给上游系统（最终结构化）


# 模块级默认运行时（裸用会话时即此），保证 current_runtime() 始终有值。
_DEFAULT = Runtime()
_current: contextvars.ContextVar[Runtime] = contextvars.ContextVar(
    "yao_runtime", default=_DEFAULT
)


def current_runtime() -> Runtime:
    """取当前运行时（默认全局实例；`App.run()` 或 `use_runtime()` 期间为各自设置的实例）。"""
    return _current.get()


@contextmanager
def use_runtime(runtime: Runtime) -> Iterator[Runtime]:
    """
    在该上下文内把 `runtime` 设为当前运行时（基于 ContextVar，天然并发隔离）。

    `App.run()` 用它把本次请求的三个投递口设好；离开自动还原。
    """
    token = _current.set(runtime)
    try:
        yield runtime
    finally:
        _current.reset(token)
