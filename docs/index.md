# YaoAgent

> 声明你的智能体结构和编排，让智能体编排像声明界面一样清晰和简单

YaoAgent 是一个轻量级的声明式 Python 智能体框架，灵感来自 Apple Foundation Models 的
[Dynamic Sessions API](https://developer.apple.com/documentation/foundationmodels/composing-dynamic-sessions-with-instructions-and-profiles)
和 [SwiftUI](https://developer.apple.com/documentation/SwiftUI)。

框架将智能体结构的每一层一一对应：
- `DynamicProfile` → 按状态激活唯一配置
- `Profile` → 绑定指令、模型参数与生命周期钩子
- `DynamicInstructions` → `yield` 声明指令 / 工具 / 嵌套组合

支持多智能体编排、环境注入、工具自愈、流式输出与实验观测。

## 快速开始

```python
import asyncio
from typing import Annotated
from yaoagent import *

class GetWeather(Tool):
    name: str = "get_weather"
    description: str = "查询城市天气。"
    def call(self, city: Annotated[str, "城市名"]) -> str:
        return f'{{"city": "{city}", "temp": 22}}'

class Assistant(DynamicInstructions):
    def body(self, session) -> DynamicInstructionStream:
        yield Instructions("你是天气助手，需要时调用工具。")
        yield GetWeather()

async def main():
    session = LanguageModelSession(
        Assistant(),
        llm_config=LLMConfig.deepseek("deepseek-v4-flash"),
    )
    print(await session.respond("北京天气怎么样？"))

asyncio.run(main())
```

## 下一步

- [教学指南](GUIDE.md) — 从零开始（Python 开发者友好，无需 SwiftUI 背景）
- [API 参考](api/index.md) — 所有公开类型的完整文档
- [示例](example.md) — 功能速览 + 厨房助手 + 多智能体编排
