from __future__ import annotations

"""
定义会话相关的核心类型。
"""

import inspect
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterable, Iterator, Mapping

from .errors import ConfigError, ErrorCode, YaoError

if TYPE_CHECKING:
    from .runtime import Runtime
from .instructions import DynamicInstructions, Instructions
from .profile import DynamicProfile, Profile
from .llm_config import LLMConfig
from .tool import Tool, bind_session
from .trace import Trace


@dataclass
class Usage:
    """一次请求消耗的 token 用量（含工具循环中各轮的累加）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Prompt(str):
    """
    一次请求的输入提示。

    与 `Response` 对称：本身就是 `str`（可直接当普通提示传给 `respond()`），额外可携带
    `attachments` 与 `metadata`。默认即纯文本，零破坏；钩子可在 `on_prompt(prompt)` 里读
    `prompt.metadata`。`attachments` 预留给将来的多模态输入。

        await session.respond(Prompt("分析这张图", attachments=[img], metadata={"lang": "zh"}))
    """

    attachments: tuple
    metadata: dict

    def __new__(
        cls, content: str, *, attachments: Iterable[Any] = (), metadata: Mapping[str, Any] | None = None
    ) -> "Prompt":
        obj = super().__new__(cls, content)
        obj.attachments = tuple(attachments)
        obj.metadata = dict(metadata or {})
        return obj


class Response(str):
    """
    `respond()` 的返回值。

    本身就是回复文本（`str` 子类，可直接打印/比较/拼接），额外携带 `usage` 与
    `finish_reason`，按需取用即可：

        answer = await session.respond("...")
        print(answer)          # 文本
        print(answer.usage)    # Usage(prompt_tokens=..., ...)
    """

    usage: Usage
    finish_reason: str | None

    def __new__(cls, content: str, *, usage: Usage | None = None, finish_reason: str | None = None) -> "Response":
        obj = super().__new__(cls, content)
        obj.usage = usage or Usage()
        obj.finish_reason = finish_reason
        return obj


@dataclass
class ResolvedRequest:
    """
    单次请求前求值得到的完整快照：指令、工具、模型参数与提示。

    这是 `respond()` 的产物——把声明式配置在“请求发生的此刻”冻结成的具体请求。
    """

    instructions: list[Instructions]
    tools: list[Tool]
    prompt: str
    model: str | None = None
    temperature: float | None = None
    reasoning: str | None = None
    # OpenAI 格式的历史消息列表（role/content）。
    history: list[dict[str, Any]] = field(default_factory=list)
    # 来自激活 Profile 的生命周期钩子与历史变换器。
    prompt_hooks: tuple[Callable[..., Any], ...] = ()
    response_hooks: tuple[Callable[..., Any], ...] = ()
    tool_call_hooks: tuple[Callable[..., Any], ...] = ()
    tool_output_hooks: tuple[Callable[..., Any], ...] = ()
    # 流式专有钩子：答案/思考的文本增量。
    response_stream_hooks: tuple[Callable[..., Any], ...] = ()
    reasoning_stream_hooks: tuple[Callable[..., Any], ...] = ()
    history_transform: Callable[[list], list] | None = None


class LanguageModelSession:
    """
    表示与语言模型交互的一个会话对象。

    功能范围：
    1. 持有激活配置（`Profile` 或 `DynamicProfile`）与基础连接配置。
    2. 持有通用会话状态 `state`，驱动动态指令/配置的响应式分支。
    3. `respond()` 在每次请求前重新求值配置与指令，得到请求快照。
    """

    def __init__(
        self,
        profile: DynamicProfile | Profile | DynamicInstructions,
        llm_config: LLMConfig | None = None,
        *,
        state: Any = None,
        history: list[dict[str, Any]] | None = None,
        trace: Trace | None = None,
        **state_kwargs: Any,
    ) -> None:
        # 允许直接传一组动态指令，等价于 Apple 的 init(dynamicInstructions:)，
        # 内部包裹成一个无额外参数的固定配置。
        if isinstance(profile, DynamicInstructions):
            profile = Profile(instructions=profile)
        # 当前会话关联的（动态）配置接口。
        self.profile = profile
        # 当前会话关联的 LLM 基础连接配置。
        self.llm_config = llm_config
        # 会话私有状态（≈ SwiftUI @State）：推荐传入显式类型化对象（dataclass）；
        # 不传时回退到由关键字参数构成的 SimpleNamespace（快速脚本用）。
        self.state = state if state is not None else SimpleNamespace(**state_kwargs)
        # 共享环境：按类型登记的可注入对象（≈ @EnvironmentObject）。
        self._environment: dict[type, Any] = {}
        # 结构化日志：在关键节点发事件给 sink（None 表示不记录）。
        self.trace = trace
        # 会话历史（OpenAI 格式的 transcript），可用既有历史种子初始化以续接对话。
        self.history: list[dict[str, Any]] = list(history) if history else []
        # `using()` 块内生效的临时覆写参数。
        self._overrides: dict[str, Any] = {}
        # 当前激活 profile 的身份与对象，用于驱动 onActivate/onDeactivate。
        self._active_key: tuple[Any, ...] | None = None
        self._active_profile: Profile | None = None

    @property
    def runtime(self) -> "Runtime":
        """
        当前运行时 I/O 封装（log/stream/output 三个投递口），用于在 `body` 里挂 handler：

            io = session.runtime
            Profile(...).on_response_stream(io.stream.response_stream).on_response(io.output.response)

        永远有值：未经 `App`/`use_runtime` 设置时返回模块级默认实例（空操作），绝不为 None。
        """
        from .runtime import current_runtime

        return current_runtime()

    def environment(self, obj: Any) -> "LanguageModelSession":
        """注入一个共享环境对象（按类型登记，可链式）。≈ SwiftUI .environmentObject。"""
        self._environment[type(obj)] = obj
        return self

    def resolve_environment(self, kind: type) -> Any:
        """
        取出已注入的环境对象（供 Environment(T) 描述符使用）。

        先按精确类型；找不到再按 isinstance 兜底——于是可以**按接口/基类注入并解析**
        （注入具体实现、用 `Environment(基类)` 取用）。多个匹配时取先注入的。
        """
        obj = self._environment.get(kind)
        if obj is None:
            obj = next((o for o in self._environment.values() if isinstance(o, kind)), None)
        if obj is None:
            raise ConfigError(code=ErrorCode.MISSING_ENVIRONMENT, kind=kind.__name__)
        return obj

    def _emit(self, type: str, *, level: str = "info", **data: Any) -> None:
        """若绑定了 trace，则发一条结构化事件。"""
        if self.trace is not None:
            self.trace.emit(type, level=level, **data)

    def describe(self, prompt: str = "") -> dict[str, Any]:
        """
        导出当前会话状态下解析出的完整配置快照（指令、工具、模型参数、状态）。

        用于把编排设置整体落盘，或作为 `request` 事件的内容。
        """
        request = self.resolve_request(prompt)
        return {
            "instructions": [item.text for item in request.instructions],
            "tools": [tool.to_openai()["function"] for tool in request.tools],
            "model": request.model,
            "temperature": request.temperature,
            "reasoning": request.reasoning,
            "state": dict(vars(self.state)) if hasattr(self.state, "__dict__") else repr(self.state),
        }

    @staticmethod
    async def _run_hooks(hooks: Iterable[Callable[..., Any]], *args: Any) -> None:
        """
        依次触发钩子（同步或异步）；钩子抛出的异常向上传播。
        """
        for hook in hooks:
            result = hook(*args)
            if inspect.isawaitable(result):
                await result

    async def _handle_activation(self, profile: Profile) -> None:
        """
        激活配置发生切换时，触发旧配置的 onDeactivate 与新配置的 onActivate。
        激活身份以 `profile.activation_key()` 判定。
        """
        key = profile.activation_key()
        if key == self._active_key:
            return
        if self._active_profile is not None:
            self._emit("deactivate", profile=type(self._active_profile.instructions).__name__)
            await self._run_hooks(self._active_profile.deactivate_hooks)
        self._emit("activate", profile=type(profile.instructions).__name__)
        await self._run_hooks(profile.activate_hooks)
        self._active_key = key
        self._active_profile = profile

    @contextmanager
    def using(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        reasoning: str | None = None,
    ) -> Iterator["LanguageModelSession"]:
        """
        在 `with` 块内临时重写本会话的模型参数（可嵌套）。

        块内的 `respond()` 会采用这些覆写值，离开块后自动还原：

            with session.using(temperature=0.0):
                await session.respond("...")   # 用 0.0
            await session.respond("...")       # 恢复配置默认

        优先级：`respond()` 显式实参 > `using()` 覆写 > 配置层取值。
        """
        new = {
            key: value
            for key, value in {
                "model": model,
                "temperature": temperature,
                "reasoning": reasoning,
            }.items()
            if value is not None
        }
        previous = self._overrides
        self._overrides = {**previous, **new}
        try:
            yield self
        finally:
            self._overrides = previous

    def _resolve_opt(self, explicit: Any, key: str, profile_value: Any) -> Any:
        """
        按三级优先级取值：显式实参 > using() 覆写 > 配置层取值。
        """
        if explicit is not None:
            return explicit
        if key in self._overrides:
            return self._overrides[key]
        return profile_value

    def resolve_profile(self) -> Profile:
        """
        解析当前会话状态下的激活配置。
        """
        return self.profile.resolve(self)

    def resolve_request(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        reasoning: str | None = None,
    ) -> ResolvedRequest:
        """
        在当前会话状态下，把配置与指令展开为一次具体请求快照。

        调用点显式传入的参数优先级最高，覆盖配置层的任何取值
        （对应文档三级优先级中最高的“调用点参数”）。
        """
        return self._build_request(
            self.resolve_profile(),
            prompt,
            model=model,
            temperature=temperature,
            reasoning=reasoning,
        )

    def _build_request(
        self,
        profile: Profile,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        reasoning: str | None = None,
    ) -> ResolvedRequest:
        """
        从一个已解析的 `Profile` 构造请求快照（应用三级优先级）。
        """
        resolved = profile.instructions.resolve(self)
        return ResolvedRequest(
            instructions=resolved.instructions,
            tools=resolved.tools,
            prompt=prompt,
            model=self._resolve_opt(model, "model", profile._model),
            temperature=self._resolve_opt(temperature, "temperature", profile._temperature),
            reasoning=self._resolve_opt(reasoning, "reasoning", profile._reasoning),
            history=list(self.history),
            prompt_hooks=profile.prompt_hooks,
            response_hooks=profile.response_hooks,
            tool_call_hooks=profile.tool_call_hooks,
            tool_output_hooks=profile.tool_output_hooks,
            response_stream_hooks=profile.response_stream_hooks,
            reasoning_stream_hooks=profile.reasoning_stream_hooks,
            history_transform=profile._history_transform,
        )

    async def _prepare(
        self,
        prompt: str,
        *,
        model: str | None,
        temperature: float | None,
        reasoning: str | None,
    ) -> ResolvedRequest:
        """每次请求前重新求值配置（响应式）、处理激活切换，并构造请求快照。"""
        if self.llm_config is None:
            raise ConfigError(code=ErrorCode.NO_LLM_CONFIG)
        profile = self.resolve_profile()
        await self._handle_activation(profile)
        return self._build_request(
            profile, prompt, model=model, temperature=temperature, reasoning=reasoning
        )

    async def _respond(
        self,
        prompt: str,
        *,
        model: str | None,
        temperature: float | None,
        reasoning: str | None,
        max_rounds: int | None,
    ) -> Response:
        # 延迟导入，避免未安装 openai 时影响纯求值用法。
        from .client import complete
        from .trace import run_scope

        # 关联作用域：独立调用生成新 id；在 group 内则沿用该次编排的 id。
        with run_scope():
            request = await self._prepare(
                prompt, model=model, temperature=temperature, reasoning=reasoning
            )
            # request 事件附带解析后的完整配置快照（debug 级可整体复现实验）。
            self._emit("request", level="debug", prompt=prompt, config=self.describe(prompt))
            start = time.perf_counter()
            try:
                # 在工具循环期间把本会话绑定为当前会话，供工具通过 self.session 访问。
                with bind_session(self):
                    text, turn, usage, finish_reason = await complete(
                        request, self.llm_config, emit=self._emit, max_rounds=max_rounds
                    )
            except YaoError as exc:
                self._emit("error", level="error", **exc.to_dict())
                raise
            # 持久化完整 transcript：user 提示、工具调用/输出、最终回复。
            self.history.extend(turn)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            self._emit(
                "response", text=text, usage=usage, finish_reason=finish_reason, elapsed_ms=elapsed_ms
            )
            return Response(text, usage=Usage(**usage), finish_reason=finish_reason)

    async def respond(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        reasoning: str | None = None,
    ) -> Response:
        """
        发起一次请求：每次都重新求值配置与指令（响应式语义），
        调用模型后端（含工具调用循环），返回 `Response`（`str` 子类）。

        工具循环跑到模型给出文本为止；要按阶段逐步控制，用 `session.max_round(n).respond(...)`。
        """
        return await self._respond(
            prompt, model=model, temperature=temperature, reasoning=reasoning, max_rounds=None
        )

    def max_round(self, n: int) -> "_RoundConfigured":
        """
        限制下一次 `respond()` 的工具循环最多发起 n 次模型调用，撞顶则**优雅返回**（不报错）。

        过程信息照常写入 `session.history` 与共享 environment，交还控制后由你的编排决定下一步：

            await session.max_round(1).respond(prompt)   # 只发一次,执行这轮工具就返回
        """
        return _RoundConfigured(self, n)

    async def run(self, input: str) -> Response:
        """统一编排入口：等价于 `respond(input)`，让会话能作为 SessionGroup 的成员。"""
        return await self.respond(input)

    async def stream_response(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        reasoning: str | None = None,
    ) -> AsyncIterator[str]:
        """
        流式版 `respond`：异步逐段产出最终回复的文本增量。

        工具调用循环在内部静默处理；流结束后把完整 transcript 写回历史：

            async for delta in session.stream_response("..."):
                print(delta, end="", flush=True)
        """
        from .client import stream

        request = await self._prepare(
            prompt, model=model, temperature=temperature, reasoning=reasoning
        )
        self._emit("request", level="debug", prompt=prompt, config=self.describe(prompt))
        turn: list[dict[str, Any]] = []
        with bind_session(self):
            async for delta in stream(request, self.llm_config, turn, emit=self._emit):
                yield delta
        # 流正常结束后持久化完整 transcript。
        self.history.extend(turn)

    def resolve_request_options(self, env: Mapping[str, str] | None = None) -> dict[str, Any]:
        """
        解析当前会话所需的模型连接参数。
        """
        if self.llm_config is None:
            raise ConfigError(code=ErrorCode.NO_LLM_CONFIG)
        return self.llm_config.to_request_options(env)


class _RoundConfigured:
    """`session.max_round(n)` 的返回值：把工具循环上限携带到这一次 `respond()`。"""

    def __init__(self, session: LanguageModelSession, max_rounds: int) -> None:
        self._session = session
        self._max_rounds = max_rounds

    async def respond(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        reasoning: str | None = None,
    ) -> Response:
        return await self._session._respond(
            prompt,
            model=model,
            temperature=temperature,
            reasoning=reasoning,
            max_rounds=self._max_rounds,
        )
