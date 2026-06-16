from __future__ import annotations

"""
定义动态配置相关的核心类型。

设计要点（对标 Apple Swift DSL，融合 Python 习惯）：
- `DynamicProfile.body()` 直接 `return` 一个 `Profile`，对应 Swift 的 `body: some Profile`。
- `Profile` 用不可变 dataclass 承载；链式修饰符用 Swift 同名命名：
  `.model()` / `.temperature()` / `.reasoning()` / `.history_transform()`，钩子用 `.on_*()`。
- `.modifier()` 可挂一个可复用的自定义修饰符，对应 Apple 的 `DynamicProfileModifier`。
- 所有修饰符既可挂在终态 `Profile`，也可挂在外层 `DynamicProfile` 并“穿透”到内层：
  值类作默认值填充（内层优先），钩子类跨层累加（外层先触发）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable

from .instructions import DynamicInstructions

if TYPE_CHECKING:
    from .session import LanguageModelSession

# 生命周期钩子签名（可同步或异步）。
Hook = Callable[..., Any]
# 历史变换器：在发送请求前对历史做局部裁剪/脱敏，只影响本次请求。
HistoryTransform = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
# 自定义修饰符：接收解析出的 Profile，返回新的 Profile。
ProfileModify = Callable[["Profile"], "Profile"]


class DynamicProfile(ABC):
    """
    根据当前会话状态选出唯一一个激活配置。

    `body()` 在每次请求前重新求值，`resolve()` 递归收敛到一个终态 `Profile`。
    在此层声明的修饰符都会穿透到内层解析出的 `Profile`。
    """

    @abstractmethod
    def body(self, session: "LanguageModelSession") -> "Profile":
        """动态配置的内容：根据会话状态返回一个 `Profile`。"""
        raise NotImplementedError

    def resolve(self, session: "LanguageModelSession") -> "Profile":
        """反复求值直到得到终态 `Profile`，支持动态配置之间的嵌套委托。"""
        result = self.body(session)
        return result if isinstance(result, Profile) else result.resolve(session)

    # —— 值类修饰符：作为默认值穿透，内层已显式设置的字段优先 ——
    def model(self, model: str | None) -> "DynamicProfile":
        return _ProfileModifiers(self, defaults={"_model": model})

    def temperature(self, temperature: float | None) -> "DynamicProfile":
        return _ProfileModifiers(self, defaults={"_temperature": temperature})

    def reasoning(self, reasoning: str | None) -> "DynamicProfile":
        return _ProfileModifiers(self, defaults={"_reasoning": reasoning})

    def history_transform(self, transform: HistoryTransform) -> "DynamicProfile":
        return _ProfileModifiers(self, defaults={"_history_transform": transform})

    # —— 钩子类修饰符：跨层累加 ——
    def on_prompt(self, hook: Hook) -> "DynamicProfile":
        return _ProfileModifiers(self, hooks={"prompt_hooks": (hook,)})

    def on_response(self, hook: Hook) -> "DynamicProfile":
        return _ProfileModifiers(self, hooks={"response_hooks": (hook,)})

    def on_tool_call(self, hook: Hook) -> "DynamicProfile":
        return _ProfileModifiers(self, hooks={"tool_call_hooks": (hook,)})

    def on_tool_output(self, hook: Hook) -> "DynamicProfile":
        return _ProfileModifiers(self, hooks={"tool_output_hooks": (hook,)})

    def on_response_stream(self, hook: Hook) -> "DynamicProfile":
        return _ProfileModifiers(self, hooks={"response_stream_hooks": (hook,)})

    def on_reasoning_stream(self, hook: Hook) -> "DynamicProfile":
        return _ProfileModifiers(self, hooks={"reasoning_stream_hooks": (hook,)})

    def on_activate(self, hook: Hook) -> "DynamicProfile":
        return _ProfileModifiers(self, hooks={"activate_hooks": (hook,)})

    def on_deactivate(self, hook: Hook) -> "DynamicProfile":
        return _ProfileModifiers(self, hooks={"deactivate_hooks": (hook,)})

    def modifier(self, modify: ProfileModify) -> "DynamicProfile":
        """
        挂一个可复用的自定义修饰符（接收解析出的 `Profile`，返回新 `Profile`）。

        对应 Apple 的 `DynamicProfileModifier` / `.modifier(_:)`：把一组成套的参数与
        钩子封装成一个可命名、可复用、可链式挂载的整体（如 `.modifier(debug())`）。
        """
        return _MappedProfile(self, modify)


class DynamicProfileModifier(ABC):
    """
    可复用的配置修饰符，对应 Apple 的 `LanguageModelSession.DynamicProfileModifier`。

    实现 `body(content)` 返回加工后的 `Profile`；实例本身可调用，可直接作为
    `.modifier()` 的参数。也可以直接用一个 `Callable[[Profile], Profile]` 函数。
    """

    @abstractmethod
    def body(self, content: "Profile") -> "Profile":
        raise NotImplementedError

    def __call__(self, content: "Profile") -> "Profile":
        return self.body(content)


class _ProfileModifiers(DynamicProfile):
    """
    给动态配置附加修饰符并在解析时穿透到内层 `Profile` 的轻量包装器。

    - `defaults`：值类字段，只填充内层仍为 `None` 的字段（内层优先）。
    - `hooks`：钩子类字段，累加到内层钩子前面（外层钩子先触发）。

    多个修饰符通过嵌套包装自然叠加。
    """

    def __init__(
        self,
        base: DynamicProfile,
        *,
        defaults: dict[str, Any] | None = None,
        hooks: dict[str, tuple[Hook, ...]] | None = None,
    ) -> None:
        self.base = base
        # 只保留非 None 的默认值。
        self.defaults = {k: v for k, v in (defaults or {}).items() if v is not None}
        self.hooks = hooks or {}

    def body(self, session: "LanguageModelSession") -> "Profile":
        return self.base.resolve(session)

    def resolve(self, session: "LanguageModelSession") -> "Profile":
        profile = self.base.resolve(session)
        updates: dict[str, Any] = {}
        # 值类：仅填充内层仍为 None 的字段。
        for key, value in self.defaults.items():
            if getattr(profile, key) is None:
                updates[key] = value
        # 钩子类：外层钩子累加到内层钩子前面。
        for field_name, fns in self.hooks.items():
            updates[field_name] = fns + getattr(profile, field_name)
        return replace(profile, **updates) if updates else profile


class _MappedProfile(DynamicProfile):
    """把一个自定义修饰符应用到内层解析结果的包装器（对应 `.modifier(_:)`）。"""

    def __init__(self, base: DynamicProfile, modify: ProfileModify) -> None:
        self.base = base
        self.modify = modify

    def body(self, session: "LanguageModelSession") -> "Profile":
        return self.resolve(session)

    def resolve(self, session: "LanguageModelSession") -> "Profile":
        return self.modify(self.base.resolve(session))


@dataclass(frozen=True)
class Profile(DynamicProfile):
    """
    一个固定配置：绑定动态指令，并携带会话级模型参数与生命周期钩子。

    既是终态配置，也是 `DynamicProfile` 的平凡实现（`body` 返回自身）。
    不可变，链式修饰符返回新副本：值类直接覆写，钩子类追加。
    （模型参数存于私有字段，公开 API 用 Swift 同名修饰符方法读写。）
    """

    # 当前配置关联的动态指令。
    instructions: DynamicInstructions
    # —— 模型参数（用 .model()/.temperature()/.reasoning() 设置）——
    _model: str | None = None
    _temperature: float | None = None
    _reasoning: str | None = None
    # —— 生命周期钩子（用 .on_*() 追加；可同步或异步，可闭包捕获 session）——
    prompt_hooks: tuple[Hook, ...] = ()
    response_hooks: tuple[Hook, ...] = ()
    tool_call_hooks: tuple[Hook, ...] = ()
    tool_output_hooks: tuple[Hook, ...] = ()
    # 流式专有钩子：答案/思考的文本增量（仅 stream_response 触发）。
    response_stream_hooks: tuple[Hook, ...] = ()
    reasoning_stream_hooks: tuple[Hook, ...] = ()
    activate_hooks: tuple[Hook, ...] = ()
    deactivate_hooks: tuple[Hook, ...] = ()
    # 请求前对历史做局部变换（用 .history_transform() 设置；不改全局历史）。
    _history_transform: HistoryTransform | None = None

    def body(self, session: "LanguageModelSession") -> "Profile":
        # 固定配置即终态，直接返回自身。
        return self

    def activation_key(self) -> tuple[Any, ...]:
        """
        激活身份：由指令类型 + 生成参数共同构成的“配置指纹”，
        用于驱动 onActivate/onDeactivate 的切换判断。

        相比只看指令类名，加入 model/temperature/reasoning 后，
        共用同一指令类、仅参数不同的分支也能被正确区分（钩子取闭包，每次新建，
        不能进指纹，否则会每次都误判为切换）。
        """
        return (type(self.instructions).__name__, self._model, self._temperature, self._reasoning)

    # —— 值类修饰符：直接覆写 ——
    def model(self, model: str | None) -> "Profile":
        return replace(self, _model=model)

    def temperature(self, temperature: float | None) -> "Profile":
        return replace(self, _temperature=temperature)

    def reasoning(self, reasoning: str | None) -> "Profile":
        return replace(self, _reasoning=reasoning)

    def history_transform(self, transform: HistoryTransform) -> "Profile":
        return replace(self, _history_transform=transform)

    # —— 钩子类修饰符：追加到本配置的钩子列表 ——
    def on_prompt(self, hook: Hook) -> "Profile":
        """提示进入历史后、发起模型请求前触发，入参为 prompt 文本。"""
        return replace(self, prompt_hooks=self.prompt_hooks + (hook,))

    def on_response(self, hook: Hook) -> "Profile":
        """模型给出最终回复后触发，入参为回复文本（可在此压缩历史）。"""
        return replace(self, response_hooks=self.response_hooks + (hook,))

    def on_tool_call(self, hook: Hook) -> "Profile":
        """模型请求调用工具、执行前触发，入参为 ToolCall，可抛异常以拒绝。"""
        return replace(self, tool_call_hooks=self.tool_call_hooks + (hook,))

    def on_tool_output(self, hook: Hook) -> "Profile":
        """工具产生输出后触发，入参为 (ToolCall, 输出文本)。"""
        return replace(self, tool_output_hooks=self.tool_output_hooks + (hook,))

    def on_response_stream(self, hook: Hook) -> "Profile":
        """流式答案到达时逐段触发，入参为文本增量（仅 stream_response 触发）。"""
        return replace(self, response_stream_hooks=self.response_stream_hooks + (hook,))

    def on_reasoning_stream(self, hook: Hook) -> "Profile":
        """流式思考(reasoning)到达时逐段触发，入参为文本增量（仅 stream_response 触发）。"""
        return replace(self, reasoning_stream_hooks=self.reasoning_stream_hooks + (hook,))

    def on_activate(self, hook: Hook) -> "Profile":
        """本配置成为激活配置时触发（无入参），适合做初始化。"""
        return replace(self, activate_hooks=self.activate_hooks + (hook,))

    def on_deactivate(self, hook: Hook) -> "Profile":
        """本配置被切换走时触发（无入参），适合做清理。"""
        return replace(self, deactivate_hooks=self.deactivate_hooks + (hook,))

    def modifier(self, modify: ProfileModify) -> "Profile":
        """应用一个可复用的自定义修饰符，立即返回加工后的 Profile。"""
        return modify(self)
