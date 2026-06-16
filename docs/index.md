# YaoAgent

> 爻者，言乎变者也。——《易经·系辞》

**爻（yáo）**是八卦的基本单元。阴爻与阳爻相叠，生出八卦；八卦相重，演为六十四卦。
YaoAgent 的哲学与此同源：**好的智能体不是写出来的，是组合出来的**。

把指令、工具、钩子做成小块，层层声明式地拼起来。组合变，行为就变。
框架替你跑编排、管状态、处理错误。你只决定零件怎么拼。

受 Apple [Foundation Models](https://developer.apple.com/documentation/foundationmodels/composing-dynamic-sessions-with-instructions-and-profiles) dynamic sessions API 启发。

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

- [教学指南](GUIDE.md) — 从零开始，面向 Python 开发者
- [API 参考](api/index.md) — 所有公开类型的完整文档
- [示例](example.md) — 功能速览 + 厨房助手 + 多智能体编排
