from __future__ import annotations

"""
定义动态指令相关的核心类型。

设计要点：
- `body()` 是生成器，对应 Swift 的 `@resultBuilder`：用 `yield` 顺序声明
  指令、工具、嵌套动态指令，是结果构造器最自然的 Python 写法。
- 轻量节点用标准库 dataclass 承载，不引入 pydantic 的样板。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Union

from .errors import ErrorCode, ResolveError
from .tool import Tool

if TYPE_CHECKING:
    from .session import LanguageModelSession


@dataclass
class Instructions:
    """
    定义模型预期行为的一段指令文本，会直接进入模型的可见上下文。
    """

    text: str


# `body()` 允许产出三类内容：指令、工具、嵌套动态指令。
DynamicInstructionItem = Union["Instructions", "Tool", "DynamicInstructions"]
DynamicInstructionStream = Iterator[DynamicInstructionItem]


@dataclass
class ResolvedInstructions:
    """
    单次请求展开后的指令与工具快照。
    """

    # 当前请求生效的指令列表。
    instructions: list[Instructions] = field(default_factory=list)
    # 当前请求可用的工具列表。
    tools: list[Tool] = field(default_factory=list)


class DynamicInstructions(ABC):
    """
    声明模型可见指令与工具；`body()` 在每次请求前被重新求值（响应式）。

    功能范围：
    1. 通过 `body()` 组合指令、工具和嵌套动态指令。
    2. 负责在请求前将声明式内容展开为确定顺序的可执行快照。
    """

    @abstractmethod
    def body(self, session: "LanguageModelSession") -> DynamicInstructionStream:
        """
        动态指令的内容，用 `yield` 顺序声明。
        """
        raise NotImplementedError

    def resolve(self, session: "LanguageModelSession") -> ResolvedInstructions:
        """
        将当前动态指令展开为单次请求使用的快照，保持声明顺序。
        """
        out = ResolvedInstructions()

        for item in self.body(session):
            match item:
                case Instructions():
                    out.instructions.append(item)
                case Tool():
                    out.tools.append(item)
                case DynamicInstructions():
                    # 遇到嵌套动态指令时，递归展开后就地合并，保持顺序与缓存友好性。
                    nested = item.resolve(session)
                    out.instructions.extend(nested.instructions)
                    out.tools.extend(nested.tools)
                case _:
                    raise ResolveError(
                        code=ErrorCode.UNSUPPORTED_INSTRUCTION, item=repr(item)
                    )
        return out
