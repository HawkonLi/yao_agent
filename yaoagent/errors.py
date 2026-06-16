from __future__ import annotations

"""
框架统一错误系统。

三件套：
1. `ErrorCode`：稳定错误码 + 自然语言解释（一处定义，全局复用）。
2. `YaoError`：错误基类，提供结构化暴露（`to_dict`）与人话解释（`explain`）。
3. 分类子类（Config/Resolve/Tool/Model）：便于调用方精确 `except`。

传递约定：所有框架内部错误都抛 `YaoError` 子类；包装底层异常时用
`raise ... from cause` 保留链路，并把原异常存入 `cause` 以便结构化暴露。
"""

from enum import Enum
from typing import Any


class ErrorCode(Enum):
    """
    错误码：枚举值为 (码, 自然语言解释, 是否可恢复)。

    可恢复（recoverable）= 这是“模型把工具调用错了”这类错误，框架不应直接中断，
    而应把解释回灌给模型让它在同一轮工具循环里改正重试；其余为致命错误，直接抛出。
    """

    # —— 配置类 1xxx ——
    NO_LLM_CONFIG = ("YAO-1001", "会话未绑定 llm_config，无法发起真实请求。", False)
    MISSING_API_KEY = ("YAO-1002", "未找到 API 密钥：对应的环境变量不存在或为空。", False)
    INVALID_CONFIG = ("YAO-1003", "连接配置不合法。", False)
    CONFIG_FILE_NOT_FOUND = ("YAO-1004", "找不到配置文件。", False)
    MISSING_ENVIRONMENT = ("YAO-1005", "环境中缺少所需类型的共享对象（用 .environment(...) 注入）。", False)

    # —— 解析类 2xxx ——
    UNSUPPORTED_INSTRUCTION = ("YAO-2001", "动态指令产出了不支持的内容项。", False)

    # —— 工具类 3xxx ——
    UNKNOWN_TOOL = ("YAO-3001", "模型请求调用了未注册的工具。", True)
    INVALID_TOOL_ARGUMENTS = ("YAO-3002", "工具参数不符合其 schema。", True)
    TOOL_EXECUTION_FAILED = ("YAO-3003", "工具执行时抛出异常。", False)
    TOOL_CALL_DENIED = ("YAO-3004", "工具调用被生命周期钩子拒绝。", False)

    # —— 模型类 4xxx ——
    MODEL_REQUEST_FAILED = ("YAO-4001", "调用模型后端失败。", False)
    MAX_TOOL_ROUNDS_EXCEEDED = ("YAO-4002", "超过最大工具调用轮数仍未得到最终回复。", False)

    @property
    def code(self) -> str:
        """稳定错误码，如 "YAO-3002"。"""
        return self.value[0]

    @property
    def explanation(self) -> str:
        """该错误码的自然语言解释。"""
        return self.value[1]

    @property
    def recoverable(self) -> bool:
        """是否应回灌给模型重试，而非直接抛出。"""
        return self.value[2]


class YaoError(Exception):
    """
    框架统一错误基类。

    参数：
    - message：具体场景消息（可选；缺省时用错误码解释）。
    - code：错误码（ErrorCode）。
    - cause：被包装的原始异常（同时建议 `raise ... from cause`）。
    - **context：结构化上下文，如 tool=、arg=、path=。
    """

    default_code: ErrorCode | None = None

    def __init__(
        self,
        message: str | None = None,
        *,
        code: ErrorCode | None = None,
        cause: BaseException | None = None,
        **context: Any,
    ) -> None:
        self.code = code or self.default_code
        self.message = message
        self.cause = cause
        self.context = context
        super().__init__(self.format())

    @property
    def recoverable(self) -> bool:
        """是否为可恢复错误（由错误码决定）。"""
        return bool(self.code and self.code.recoverable)

    def explain(self) -> str:
        """
        面向人/模型的自然语言解释：错误码解释 + 具体消息（若有）。
        """
        base = self.code.explanation if self.code else ""
        if self.message and self.message != base:
            return f"{base} {self.message}".strip()
        return base or (self.message or "")

    def format(self) -> str:
        """
        面向日志的完整字符串：[码] 解释（上下文）。
        """
        head = f"[{self.code.code}] " if self.code else ""
        body = self.explain()
        if self.context:
            ctx = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
            body = f"{body} ({ctx})" if body else ctx
        return f"{head}{body}"

    def to_dict(self) -> dict[str, Any]:
        """
        结构化暴露：便于日志、API 返回或回灌给模型。
        """
        return {
            "code": self.code.code if self.code else None,
            "name": self.code.name if self.code else None,
            "message": self.message,
            "explanation": self.explain(),
            "context": self.context,
            "cause": repr(self.cause) if self.cause is not None else None,
        }


class ConfigError(YaoError):
    """配置/连接相关错误（1xxx）。"""


class ResolveError(YaoError):
    """动态指令/配置解析相关错误（2xxx）。"""


class ToolError(YaoError):
    """工具相关错误（3xxx）。"""


class ModelError(YaoError):
    """模型后端相关错误（4xxx）。"""
