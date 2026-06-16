"""
完整 DSL 样例：多智能体研究助手。

一条声明式编排,串起几项新能力:
- 共享环境 `Notebook`(EnvironmentObject)——多个智能体通过 Environment(T) 读写同一块笔记本
- 工具自动 schema + self.session 访问注入的环境
- 可复用指令组件(每个角色一个 DynamicInstructions)
- SessionGroup 嵌套拓扑:sequential( parallel(两个调研员) → 综述 → loop(自我精修) )
  环境通过 .environment(notebook) 向所有成员(含嵌套子组)穿透

(类型化 state / 钩子 / 流式等见 example.py 与 README。)

运行前在项目目录放 .env:DEEPSEEK_API_KEY=sk-...
"""

import asyncio
import json
from typing import Annotated

from yaoagent import *


# ============ 共享环境:一块所有智能体共用的"研究笔记本" ============
class Notebook(EnvironmentObject):
    def __init__(self) -> None:
        self.findings: list[str] = []

    def add_finding(self, text: str) -> None:
        self.findings.append(text)

    def dump(self) -> str:
        return json.dumps({"findings": self.findings}, ensure_ascii=False)


# ============ 工具:通过 self.session 注入的 Notebook 读写 ============
class SaveFinding(Tool):
    name: str = "save_finding"
    description: str = "把一条调研发现写入共享笔记本。"
    notebook = Environment(Notebook)        # 按类型注入,不进 schema

    def call(self, finding: Annotated[str, "一条简短的发现"]) -> str:
        self.notebook.add_finding(finding)
        return "已记录"


class ReadNotebook(Tool):
    name: str = "read_notebook"
    description: str = "读取笔记本里目前的所有发现。"
    notebook = Environment(Notebook)

    def call(self) -> str:
        return self.notebook.dump()


# ============ 各角色的指令(可复用组件) ============
class ResearchInstructions(DynamicInstructions):
    def __init__(self, angle: str) -> None:
        self.angle = angle

    def body(self, session) -> DynamicInstructionStream:
        yield Instructions(
            f"你是研究员,只关注【{self.angle}】这个角度。就给定主题给出 1-2 条要点,"
            f"每条都用 save_finding 记入笔记本,然后简短复述。"
        )
        yield SaveFinding()


class SynthInstructions(DynamicInstructions):
    def body(self, session) -> DynamicInstructionStream:
        yield Instructions("你是综述员:用 read_notebook 读取全部发现,综合成一段连贯的草稿。")
        yield ReadNotebook()


class ReviseInstructions(DynamicInstructions):
    def body(self, session) -> DynamicInstructionStream:
        yield Instructions(
            "你是精修员:把收到的草稿改得更清晰流畅,直接输出改后的全文。"
            "若你认为已经足够好,就在结尾另起一行写 [OK]。"
        )


async def main() -> None:
    notebook = Notebook()                                  # 共享环境对象

    # 每个智能体就是一个会话；Profile 本身即 DynamicProfile，直接传即可。
    # 不在这里写 llm_config —— 统一在 pipeline 上注入并向下穿透。
    macro = LanguageModelSession(Profile(instructions=ResearchInstructions("宏观趋势")).temperature(0.7))
    micro = LanguageModelSession(Profile(instructions=ResearchInstructions("具体案例")).temperature(0.7))
    synth = LanguageModelSession(Profile(instructions=SynthInstructions()).temperature(0.3))
    reviser = LanguageModelSession(Profile(instructions=ReviseInstructions()).temperature(0.6))

    # 声明式编排:并行调研 → 综述 → 自我精修(迭代到出现 [OK])
    pipeline = (
        SessionGroup(
            parallel(macro, micro),                        # 两个角度并行,都写进 notebook
            synth,                                         # 读 notebook,综合成草稿
            loop(reviser, until=lambda out: "[OK]" in out, max_iters=3),
        )
        .group_style(Style.sequential)
        .environment(notebook)                             # 环境向所有成员穿透
        .llm_config(LLMConfig.deepseek("deepseek-v4-flash"))   # config 在此一次注入,穿透到每个 agent
    )

    topic = "大语言模型在科研中的应用"
    result = (await pipeline.run(topic)).replace("[OK]", "").strip()

    print("=== 最终稿 ===")
    print(result)
    print("\n=== 共享笔记本(各智能体协作留下的痕迹) ===")
    for i, f in enumerate(notebook.findings, 1):
        print(f"  {i}. {f}")


if __name__ == "__main__":
    asyncio.run(main())
