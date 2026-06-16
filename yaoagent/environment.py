from __future__ import annotations

"""
全局依赖

- `EnvironmentObject`：可注入共享对象的标记基类（继承它只是表明“我可被注入”）。
- `Environment(T)`：声明式取用描述符，在工具/组件里按**类型**从当前会话环境解析。

注入由 `LanguageModelSession.environment(obj)` / `SessionGroup.environment(obj)` 完成，
按对象类型登记；一个类型一个实例（需要两份就包成不同类型）。

参考 SwiftUI 的 @EnvironmentObject 的全局依赖。
"""

from typing import Any, Generic, TypeVar

T = TypeVar("T")


class EnvironmentObject:
    """可注入共享对象的标记基类（可选继承，仅表意）。"""


class Environment(Generic[T]):
    """
    按类型从当前会话环境解析共享对象的描述符（≈ @EnvironmentObject）。

        class PlanTool(Tool):
            kitchen = Environment(KitchenEnv)     # 声明依赖
            def call(self, dish: str) -> str:
                return self.kitchen.snapshot()    # 自动按类型注入

    依赖宿主对象的 `self.session`（工具执行期间由框架绑定）。
    """

    def __init__(self, kind: type[T]) -> None:
        self.kind = kind

    def __get__(self, obj: Any, owner: type | None = None) -> T:
        if obj is None:
            return self  # type: ignore[return-value]
        return obj.session.resolve_environment(self.kind)
