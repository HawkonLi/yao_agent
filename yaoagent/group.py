from __future__ import annotations

"""
多智能体编排：用声明式、可嵌套的 group 拼装拓扑。

group 由三个**正交维度**描述（彼此原子，编排约束另外两个的合法取值）：
- **编排 `group_style`**（运行状态）：成员怎么跑——`Style.sequential` / `Style.parallel` / `Style.loop(...)`。
- **输入 `input_style`**（输入管理）：每个成员收什么——`InputStyle.pipe`（上一个输出喂下一个）/
  `InputStyle.broadcast`（都拿原输入，靠共享 environment 通信）。
- **输出 `output_style`**（输出管理）：谁的输出暴露给 group——`OutputStyle.last_session` /
  `OutputStyle.merge(fn)`。

（`input_style` / `output_style` 命名描述的是**智能体之间的内部接线**，与面向用户的运行时
输出层 `Runtime`（log/stream/output 投递口）是两回事，刻意区分。）

每种编排自带默认输入/输出，按需覆盖；非法组合（如 parallel + pipe）报错。
成员可以是 `LanguageModelSession`，也可以是另一个 `SessionGroup`（递归嵌套），
统一满足 `await x.run(input) -> str`。`environment(obj)` / `llm_config(cfg)` / `trace(t)` 向所有成员穿透。
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Protocol, runtime_checkable

from .session import Response, Usage

if TYPE_CHECKING:
    from .llm_config import LLMConfig
    from .trace import Trace


@runtime_checkable
class Runnable(Protocol):
    """可被编排的最小协议：会话与 group 都满足（统一返回 `Response`，它是 str 子类）。"""

    async def run(self, input: str) -> str: ...


def _sum_usage(results: "list[Any]") -> Usage:
    """累加各成员 `Response` 的用量（纯 str 成员贡献 0）；嵌套子组天然递归累加。"""
    total = Usage()
    for r in results:
        u = getattr(r, "usage", None)
        if u is not None:
            total.prompt_tokens += u.prompt_tokens
            total.completion_tokens += u.completion_tokens
            total.total_tokens += u.total_tokens
    return total


def _finalize(text: Any, results: "list[Any]") -> Response:
    """把编排得到的文本 + 全成员累加用量收成统一出口 `Response`。"""
    return Response(str(text), usage=_sum_usage(results))


# ============================================================ 输入管理
class InputStyle(Enum):
    """每个成员的输入从哪来。"""

    pipe = "pipe"            # 上一个成员的输出喂给下一个（需有序编排）
    broadcast = "broadcast"  # 每个成员都拿 group 的输入；成员间靠共享 environment 通信


# ============================================================ 输出管理
class OutputPolicy(ABC):
    """从各成员的输出里，决定 group 对外暴露什么。"""

    @abstractmethod
    def collect(self, outputs: list[str], members: tuple[Runnable, ...]) -> str:
        raise NotImplementedError


class _LastSession(OutputPolicy):
    """暴露成员列表中**最后一个 session** 的输出。"""

    def collect(self, outputs, members):
        return outputs[-1] if outputs else ""


class _Merge(OutputPolicy):
    def __init__(self, fn: Callable[[list[str]], str]) -> None:
        self.fn = fn

    def collect(self, outputs, members):
        return self.fn(list(outputs))


class OutputStyle:
    """输出策略的命名空间。"""

    last_session: OutputPolicy = _LastSession()

    @staticmethod
    def merge(fn: Callable[[list[str]], str] | None = None) -> OutputPolicy:
        """合并所有成员输出；默认用空行拼接。"""
        return _Merge(fn or (lambda outs: "\n\n".join(outs)))


# ============================================================ 编排（运行状态）
class _Orchestration(ABC):
    """决定成员的执行方式，并自带默认输入/输出。"""

    default_inputs: InputStyle = InputStyle.pipe
    default_outputs: OutputPolicy = OutputStyle.last_session

    def validate(self, inputs: InputStyle) -> None:
        """检查输入策略与本编排是否相容（默认都允许）。"""

    @abstractmethod
    async def run(
        self,
        members: tuple[Runnable, ...],
        group_input: str,
        inputs: InputStyle,
        outputs: OutputPolicy,
        emit: Callable[..., None] | None = None,
    ) -> str:
        raise NotImplementedError


async def _run_member(
    member: Runnable,
    index: int,
    member_input: str,
    emit: Callable[..., None] | None,
    iteration: int | None = None,
) -> Any:
    """跑一个成员并（若启用）记录 member_start/member_end 事件；返回原始输出（保留 Response 的用量）。"""
    if emit:
        emit(
            "member_start",
            index=index,
            member=type(member).__name__,
            member_id=getattr(member, "session_id", getattr(member, "group_id", None)),
            iteration=iteration,
        )
    output = await member.run(member_input)
    if emit:
        emit(
            "member_end",
            index=index,
            member=type(member).__name__,
            member_id=getattr(member, "session_id", getattr(member, "group_id", None)),
            iteration=iteration,
        )
    return output


class _Sequential(_Orchestration):
    default_inputs = InputStyle.pipe
    default_outputs = OutputStyle.last_session

    async def run(self, members, group_input, inputs, outputs, emit=None):
        results: list[Any] = []
        previous = group_input
        for index, member in enumerate(members):
            member_input = previous if inputs is InputStyle.pipe else group_input
            output = await _run_member(member, index, member_input, emit)
            results.append(output)
            previous = str(output)
        text = outputs.collect([str(r) for r in results], members)
        return _finalize(text, results)


class _Parallel(_Orchestration):
    default_inputs = InputStyle.broadcast
    default_outputs = _Merge(lambda outs: "\n\n".join(outs))

    def validate(self, inputs):
        if inputs is InputStyle.pipe:
            raise ValueError("parallel 不能用 InputStyle.pipe（并发无法链式传递）。")

    async def run(self, members, group_input, inputs, outputs, emit=None):
        results = list(
            await asyncio.gather(
                *(_run_member(m, i, group_input, emit) for i, m in enumerate(members))
            )
        )
        text = outputs.collect([str(r) for r in results], members)
        return _finalize(text, results)


class _Loop(_Orchestration):
    default_inputs = InputStyle.pipe
    default_outputs = OutputStyle.last_session

    def __init__(self, until: Callable[[str], bool], max_iters: int = 5) -> None:
        self.until = until
        self.max_iters = max_iters

    async def run(self, members, group_input, inputs, outputs, emit=None):
        current: Any = group_input
        all_results: list[Any] = []   # 累计所有迭代的成员输出，用量含全部轮次
        for iteration in range(self.max_iters):
            if emit:
                emit("iteration", index=iteration)
            results: list[Any] = []
            previous = str(current)
            for index, member in enumerate(members):
                member_input = previous if inputs is InputStyle.pipe else str(current)
                output = await _run_member(member, index, member_input, emit, iteration=iteration)
                results.append(output)
                all_results.append(output)
                previous = str(output)
            current = outputs.collect([str(r) for r in results], members)
            if self.until(str(current)):
                break
        return _finalize(current, all_results)


class Style:
    """编排样式命名空间（接近 SwiftUI 的点访问）。"""

    sequential: _Orchestration = _Sequential()
    parallel: _Orchestration = _Parallel()

    @staticmethod
    def loop(until: Callable[[str], bool], max_iters: int = 5) -> _Orchestration:
        return _Loop(until=until, max_iters=max_iters)


# ============================================================ 容器
class SessionGroup:
    """
    一组智能体的编排容器：构造器只收**成员**，其余（编排/输入/输出/环境/连接/日志）走链式修饰符。

        SessionGroup(a, b)
            .group_style(Style.sequential).input_style(InputStyle.broadcast)
            .environment(notebook, provider).llm_config(cfg)

    成员里的 `None` 会被自动滤除，于是条件包含可内联（`Agent() if cond else None`，
    类比 SwiftUI ViewBuilder 里的 `if`）。本身满足 `run`，可作为另一个 group 的成员，递归嵌套。
    """

    def __init__(
        self,
        *members: Runnable,
        style: _Orchestration | None = None,
        input_style: InputStyle | None = None,
        output_style: OutputPolicy | None = None,
    ) -> None:
        self.members = tuple(m for m in members if m is not None)
        self.group_id = uuid.uuid4().hex[:12]
        self._style = style or Style.sequential
        self._input_style = input_style    # None → 用编排的默认
        self._output_style = output_style  # None → 用编排的默认
        self._environment: dict[type, Any] = {}
        self._llm_config: "LLMConfig | None" = None
        self._trace: "Trace | None" = None

    def group_style(self, style: _Orchestration) -> "SessionGroup":
        """设置编排（运行状态）。"""
        self._style = style
        return self

    def input_style(self, policy: InputStyle) -> "SessionGroup":
        """设置输入管理（覆盖编排默认）。"""
        self._input_style = policy
        return self

    def output_style(self, policy: OutputPolicy) -> "SessionGroup":
        """设置输出管理（覆盖编排默认）。"""
        self._output_style = policy
        return self

    def environment(self, *objs: Any) -> "SessionGroup":
        """注入共享环境对象（可一次多个，按类型登记，向所有成员穿透，可链式）。"""
        for obj in objs:
            self._environment[type(obj)] = obj
        return self

    def llm_config(self, config: "LLMConfig") -> "SessionGroup":
        """为整组设置默认模型连接（向所有成员穿透；成员自带的优先）。"""
        self._llm_config = config
        return self

    def trace(self, trace: "Trace") -> "SessionGroup":
        """为整组设置日志（向所有成员穿透；成员自带的优先），并记录 group/member 事件。"""
        self._trace = trace
        return self

    def _emit(self, type: str, *, level: str = "info", **data: Any) -> None:
        if self._trace is not None:
            self._trace.emit(type, level=level, group_id=self.group_id, **data)

    def _inject(self) -> None:
        # 向成员下发本组的环境、默认 config 与日志（成员已有的优先，即“内层覆盖外层”）。
        for member in self.members:
            store = getattr(member, "_environment", None)
            if store is not None:
                for kind, obj in self._environment.items():
                    store.setdefault(kind, obj)
            if isinstance(member, SessionGroup):
                if member._llm_config is None:
                    member._llm_config = self._llm_config
                if member._trace is None:
                    member._trace = self._trace
            else:
                if self._llm_config is not None and getattr(member, "llm_config", None) is None:
                    member.llm_config = self._llm_config
                if self._trace is not None and getattr(member, "trace", None) is None:
                    member.trace = self._trace

    async def run(self, input: str) -> str:
        from .trace import run_scope

        self._inject()
        inputs = self._input_style if self._input_style is not None else self._style.default_inputs
        outputs = (
            self._output_style if self._output_style is not None else self._style.default_outputs
        )
        self._style.validate(inputs)
        # 整条编排共享一个关联 id（嵌套时沿用外层）。
        with run_scope():
            self._emit(
                "group_start",
                style=type(self._style).__name__,
                members=[
                    {
                        "type": type(member).__name__,
                        "member_id": getattr(
                            member, "session_id", getattr(member, "group_id", None)
                        ),
                    }
                    for member in self.members
                ],
            )
            result = await self._style.run(self.members, input, inputs, outputs, self._emit)
            self._emit("group_end", style=type(self._style).__name__)
            return result

    @property
    def streamable(self) -> bool:
        """
        本组能否逐 token 流式：串行 + 末位输出（常见流水线），或并行 + 末位输出（主答复 + 从并发）。

        并行时末位成员的流式增量逐 token 转发，其余成员并发跑并正常发射进度事件。
        loop / merge 输出不支持流式。
        """
        out_is_last = self._output_style is None or isinstance(self._output_style, _LastSession)
        if not out_is_last or not self.members:
            return False
        if isinstance(self._style, _Sequential):
            return True
        if isinstance(self._style, _Parallel):
            return True
        return False

    async def stream_response(self, input: str) -> AsyncIterator[str]:
        """
        流式版编排：**末位（输出）成员逐 token 流式产出**。

        - 串行：前置成员照常跑完，末位成员流式转发。
        - 并行：末位成员流式转发的同时，其余成员并发跑并正常发射进度事件；
          末位结束后等待其余成员收尾。

        仅在 `streamable` 为真时可用（否则请用 `run()`）。
        """
        if not self.streamable:
            raise NotImplementedError(
                "该编排不支持逐 token 流式（需 串行/并行 + last_session 输出）；请用 run()。"
            )
        from .trace import run_scope

        self._inject()
        inputs = self._input_style if self._input_style is not None else self._style.default_inputs
        with run_scope():
            self._emit(
                "group_start",
                style=type(self._style).__name__,
                members=[
                    {
                        "type": type(member).__name__,
                        "member_id": getattr(
                            member, "session_id", getattr(member, "group_id", None)
                        ),
                    }
                    for member in self.members
                ],
            )

            # 末位 = 输出成员
            *head, last = self.members
            last_index = len(self.members) - 1

            # 前置成员（或并行时的从成员）：后台跑，不流式
            slave_tasks: list[asyncio.Task] = []
            if isinstance(self._style, _Sequential):
                # 串行：前置成员逐个跑完
                previous = input
                for index, member in enumerate(head):
                    member_input = previous if inputs is InputStyle.pipe else input
                    output = await _run_member(member, index, member_input, self._emit)
                    previous = str(output)
                last_input = previous if inputs is InputStyle.pipe else input
            else:
                # 并行：从成员并发启动，主拿原始输入
                last_input = input
                for index, member in enumerate(head):
                    slave_tasks.append(
                        asyncio.create_task(
                            _run_member(member, index, input, self._emit)
                        )
                    )

            # 末位成员流式输出
            last_id = getattr(last, "session_id", getattr(last, "group_id", None))
            self._emit(
                "member_start", index=last_index, member=type(last).__name__, member_id=last_id
            )
            if hasattr(last, "stream_response"):
                async for delta in last.stream_response(last_input):
                    yield delta
            else:
                yield str(await last.run(last_input))
            self._emit(
                "member_end", index=last_index, member=type(last).__name__, member_id=last_id
            )

            # 并行时等待从成员收尾（它们的进度事件继续发射）
            if slave_tasks:
                await asyncio.gather(*slave_tasks, return_exceptions=True)

            self._emit("group_end", style=type(self._style).__name__)


# ============================================================ 便捷构造
def sequential(*members: Runnable) -> SessionGroup:
    """串行组（默认 pipe 输入、last 输出）。"""
    return SessionGroup(*members, style=Style.sequential)


def parallel(*members: Runnable, merge: Callable[[list[str]], str] | None = None) -> SessionGroup:
    """并行组（默认 broadcast 输入、merge 输出）；传 merge 即覆盖合并方式。"""
    group = SessionGroup(*members, style=Style.parallel)
    if merge is not None:
        group.output_style(OutputStyle.merge(merge))
    return group


def loop(*members: Runnable, until: Callable[[str], bool], max_iters: int = 5) -> SessionGroup:
    """迭代组（默认 pipe 输入、last 输出）。"""
    return SessionGroup(*members, style=Style.loop(until=until, max_iters=max_iters))
