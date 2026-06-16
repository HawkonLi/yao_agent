from __future__ import annotations

"""
结构化日志 / 可观测层。

设计极简:框架在关键节点发**结构化事件**(普通 dict),`Trace` 把事件分发给若干
**sink**(就是 `Callable[[dict], None]`,不搞类层级)。实验留档 = 用 `jsonl` sink;
接 SwanLab 等 = 自己写一个 `lambda e: ...` sink(适配器在你的代码里,不进框架)。

事件类型:request / tool_call / tool_output / response / activate / deactivate / error。
debug 级几乎全量(含解析后的完整配置快照),便于复现实验。
"""

import contextvars
import json
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterator

# 一个 sink 就是吃事件 dict 的可调用对象。
Sink = Callable[[dict], None]

_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}

# 当前逻辑请求的关联 ID：同一次 run（含其编排里的所有成员/工具轮次）共享一个 id，
# 于是并发/嵌套时事件可被归并到同一条时间线。基于 ContextVar，天然并发隔离。
_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "yao_run_id", default=None
)


def current_run_id() -> str | None:
    """取当前关联 ID（未在任何 run_scope 内时为 None）。"""
    return _run_id.get()


@contextmanager
def run_scope(run_id: str | None = None) -> Iterator[str]:
    """
    开一个关联作用域：块内所有 trace 事件都带上同一个 `run_id`。

    `App.run()` 与 `SessionGroup.run()` 用它把整条编排的事件串起来；不传则自动生成。
    已处于某个作用域内时，沿用外层 id（不嵌套覆盖）。
    """
    existing = _run_id.get()
    rid = run_id or existing or uuid.uuid4().hex[:12]
    token = _run_id.set(rid)
    try:
        yield rid
    finally:
        _run_id.reset(token)


def console(event: dict) -> None:
    """把事件打到标准输出(一行一条,适合开发时看)。"""
    extra = {k: v for k, v in event.items() if k not in ("ts", "type", "level")}
    print(f"[trace] {event['type']}: {extra}")


def jsonl(path: str) -> Sink:
    """返回一个把事件追加写入 JSONL 文件的 sink(适合实验留档、事后复现)。"""

    def sink(event: dict) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    return sink


class Trace:
    """
    把结构化事件分发给若干 sink,按 `level` 过滤。

        session = LanguageModelSession(profile, llm_config=cfg,
                                       trace=Trace(jsonl("runs/exp1.jsonl"), level="debug"))

    不传 sink 时默认打到控制台。
    """

    def __init__(self, *sinks: Sink, level: str = "info") -> None:
        self.sinks: list[Sink] = list(sinks) or [console]
        self.level = _LEVELS.get(level, 20)

    def emit(self, type: str, *, level: str = "info", **data: Any) -> None:
        if _LEVELS.get(level, 20) < self.level:
            return
        event = {"ts": time.time(), "type": type, "level": level}
        run_id = _run_id.get()
        if run_id is not None:
            event["run_id"] = run_id
        event.update(data)
        for sink in self.sinks:
            sink(event)
