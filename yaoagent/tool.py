from __future__ import annotations

"""
定义工具相关的核心类型。
"""

import contextvars
import inspect
import typing
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict

from .environment import Environment

# 工具执行期间绑定的当前会话；由框架在工具循环外层设置，工具内用 self.session 读取。
_active_session: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "yao_active_session", default=None
)


@contextmanager
def bind_session(session: Any) -> Iterator[None]:
    """在该上下文内把 `session` 绑定为当前会话，供工具通过 `self.session` 访问。"""
    token = _active_session.set(session)
    try:
        yield
    finally:
        _active_session.reset(token)


@dataclass
class ToolCall:
    """
    一次工具调用的描述，作为生命周期钩子（onToolCall/onToolOutput）的载荷。
    """

    name: str
    arguments: dict[str, Any]


# Python 类型 → JSON Schema 类型。
_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _field_schema(annotation: Any) -> tuple[dict[str, Any], str | None]:
    """
    把单个参数的类型注解转成 JSON Schema 片段。

    支持 `Annotated[T, "描述"]`：第一个 str 元数据用作字段描述。
    """
    description: str | None = None
    if typing.get_origin(annotation) is typing.Annotated:
        args = typing.get_args(annotation)
        annotation = args[0]
        for meta in args[1:]:
            if isinstance(meta, str):
                description = meta
                break
    base = typing.get_origin(annotation) or annotation
    return {"type": _PY_TO_JSON.get(base, "string")}, description


def schema_from_callable(fn: Any) -> dict[str, Any]:
    """
    从一个可调用对象的签名 + 类型注解推导参数 JSON Schema。

    没有默认值的参数视为 required；`self`、`*args`、`**kwargs` 被忽略。
    """
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        if name == "self" or param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            continue
        schema, description = _field_schema(hints.get(name, str))
        if description:
            schema["description"] = description
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    return result


# JSON Schema 类型 → 用于运行时校验的 Python 类型。
_JSON_CHECK: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _type_matches(value: Any, want: str) -> bool:
    expected = _JSON_CHECK.get(want)
    if expected is None:
        return True
    # bool 是 int 的子类，但不应被当作 integer/number。
    if want in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def validate_arguments(arguments: Any, schema: dict[str, Any]) -> str | None:
    """
    用工具的 JSON Schema 校验模型给出的参数。

    校验通过返回 None；否则返回一句面向模型的中文错误说明（可直接回灌让其重试）。
    覆盖常见子集：对象类型、required、字段类型、additionalProperties=False 时的未知字段。
    """
    if not isinstance(arguments, dict):
        return "参数必须是一个对象。"

    properties: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    missing = [key for key in required if key not in arguments]
    if missing:
        return f"缺少必填参数：{', '.join(missing)}。"

    if schema.get("additionalProperties", True) is False:
        unknown = [key for key in arguments if key not in properties]
        if unknown:
            allowed = ", ".join(properties) or "（无）"
            return f"出现未知参数：{', '.join(unknown)}。允许的参数：{allowed}。"

    for key, value in arguments.items():
        want = properties.get(key, {}).get("type")
        if want and not _type_matches(value, want):
            return f"参数 {key} 的类型应为 {want}，但收到 {type(value).__name__}。"

    return None


class Tool(BaseModel, ABC):
    """
    一个可由模型在请求过程中调用的能力。

    保留 pydantic：工具参数可以借助字段声明自动获得校验与 schema，
    这是 DSL 节点（纯 dataclass）不需要、而工具需要的能力。
    """

    # 不开 strict（保持 Python 的宽松：允许常规类型强转）。
    # ignored_types：让 Environment(T) 描述符作为普通类属性存在，而非 pydantic 字段。
    model_config = ConfigDict(ignored_types=(Environment,))

    # 工具名称，用于标识当前可调用能力。
    name: str
    # 工具说明，用于告诉模型这个工具做什么。
    description: str

    @property
    def session(self) -> Any:
        """
        当前会话，工具执行期间由框架自动绑定（对标 Apple 的 @SessionProperty）。

        借此读写 `self.session.state` / `self.session.history`，无需污染 `call` 签名。
        仅在工具被调用期间可用。
        """
        session = _active_session.get()
        if session is None:
            raise RuntimeError("self.session 仅在工具执行期间可用。")
        return session

    @abstractmethod
    def call(self, **arguments: Any) -> str:
        """
        执行工具并返回工具输出。

        约定以关键字参数接收模型给出的调用参数，子类可声明更精确的签名。
        支持同步或异步（`async def call`）实现。
        """
        raise NotImplementedError

    def parameters(self) -> dict[str, Any]:
        """
        返回 JSON Schema 形式的参数定义。

        默认从 `call` 的签名与类型注解自动推导，例如：

            def call(self, city: Annotated[str, "城市名"]) -> str: ...

        会自动生成对应 schema。需要更精细控制时重写本方法即可。
        """
        return schema_from_callable(self.call)

    def to_openai(self) -> dict[str, Any]:
        """
        转换为 OpenAI Chat Completions 的 `tools` 项。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters(),
            },
        }
