"""
YaoAgent 示例：

- feature_tour()：逐节速览各项能力。
- kitchen_demo()：一个完整的多阶段编排智能体，集中体现 DSL 编排能力。

标注 [实时] 的会真实请求 DeepSeek。运行前在项目目录放 .env：DEEPSEEK_API_KEY=sk-...
"""

import asyncio
import json
from typing import Annotated

from yaoagent import *


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ======================================================================
#  第一部分：能力速览
# ======================================================================

# —— 工具：参数 schema 由 call 的类型注解自动生成（Annotated 第二项作描述）——
class GetWeatherTool(Tool):
    name: str = "get_weather"
    description: str = "查询某个城市某天的天气。"

    def call(
        self,
        city: Annotated[str, "城市名，如 北京"],
        day: Annotated[str, "today 或 tomorrow"] = "today",
    ) -> str:
        base = {"北京": 22, "上海": 26, "广州": 30}.get(city, 20)
        if day == "tomorrow":
            base -= 4
        return json.dumps({"city": city, "day": day, "temp_c": base, "sky": "晴"}, ensure_ascii=False)


class WeatherInstructions(DynamicInstructions):
    def body(self, session: LanguageModelSession) -> DynamicInstructionStream:
        yield Instructions("你是简洁的天气助手，需要时调用工具。")
        yield GetWeatherTool()
        if getattr(session.state, "concise", False):
            yield Instructions("只用一句话回答。")


class WeatherProfile(DynamicProfile):
    def body(self, session: LanguageModelSession) -> Profile:
        return (
            Profile(instructions=WeatherInstructions())
            .temperature(0.7)
            .on_activate(lambda: print("  · [激活] 天气助手已就绪"))
            .on_tool_call(lambda c: print(f"  · [工具] {c.name}({c.arguments})"))
            .on_response(lambda _: _compact(session))
            .history_transform(lambda h: h[-20:])
        )


def _compact(session: LanguageModelSession) -> None:
    if len(session.history) > 100:
        session.history = session.history[-50:]


async def feature_tour() -> None:
    session = LanguageModelSession(
        WeatherProfile(),
        llm_config=LLMConfig.deepseek("deepseek-v4-flash"),
    )

    banner("1) [实时] 响应式指令 + 自动 schema 工具 + 生命周期钩子")
    print("  自动生成的工具 schema：", GetWeatherTool().parameters())
    answer = await session.respond("北京和上海今天哪个更暖和？")
    print("  回答：", answer)                       # Response 是 str，可直接打印
    print("  用量：", answer.usage, "| 结束原因：", answer.finish_reason)

    banner("2) 响应式：改变 session.state，指令树随之变化")
    print("  改前指令：", [i.text for i in session.resolve_request("_").instructions])
    session.state.concise = True
    print("  改后指令：", [i.text for i in session.resolve_request("_").instructions])

    banner("3) [实时] with 重写：块内临时换模型 + 开 reasoning，离开自动还原")
    with session.using(model="deepseek-v4-pro", reasoning="high"):
        snap = session.resolve_request("_")
        print(f"  块内 → model={snap.model}, reasoning={snap.reasoning}, temperature={snap.temperature}")
        print("  回答：", await session.respond("明天北京会比今天冷吗？给个理由。"))
    snap = session.resolve_request("_")
    print(f"  块外 → model={snap.model}, reasoning={snap.reasoning}（已还原）")

    banner("4) 穿透传值：外层 DynamicProfile 的修饰符累加到内层 Profile")

    class Inner(DynamicProfile):
        def body(self, s: LanguageModelSession) -> Profile:
            return Profile(instructions=WeatherInstructions()).on_tool_call(lambda c: None).temperature(0.9)

    # 外层 DynamicProfile 再叠加修饰符；已解析的值从 ResolvedRequest 快照读取。
    composed = Inner().on_tool_call(lambda c: None).temperature(0.1).reasoning("low")
    req = LanguageModelSession(composed).resolve_request("_")
    print(f"  钩子累加：tool_call 钩子 {len(req.tool_call_hooks)} 个（外 1 + 内 1）")
    print(f"  值类优先：temperature={req.temperature}（内层 0.9 胜），reasoning={req.reasoning}（外层穿透）")

    banner("5) 参数校验：错误参数会被拦截并转成自然语言（可回灌模型重试）")
    schema = GetWeatherTool().parameters()
    for bad in ({"day": "today"}, {"city": 123}, {"city": "北京", "x": 1}):
        print(f"  {bad} → {validate_arguments(bad, schema)}")

    banner("6) 错误系统：统一错误码 + 自然语言解释 + 结构化暴露")
    try:
        await LanguageModelSession(WeatherProfile()).respond("hi")
    except YaoError as e:
        print("  str    ：", e)
        print("  explain：", e.explain())
        print("  to_dict：", e.to_dict())

    banner("7) [实时] 流式输出：逐段产出，工具调用循环内部静默处理")
    print("  ", end="")
    async for delta in session.stream_response("上海现在大概多少度？一句话。"):
        print(delta, end="", flush=True)
    print()


# ======================================================================
#  第二部分：完整编排智能体 —— 多阶段厨房助手
#  顶层 DynamicProfile 按 session.state.stage 切换子配置，每个阶段拥有
#  各自的指令、工具、模型参数与激活钩子；阶段切换自动触发 onActivate/onDeactivate。
# ======================================================================

class SearchRecipesTool(Tool):
    name: str = "search_recipes"
    description: str = "按关键词搜索候选食谱。"

    def call(self, keyword: Annotated[str, "菜系或食材关键词"]) -> str:
        db = {"番茄": [{"name": "番茄炒蛋", "level": "易"}, {"name": "番茄牛腩", "level": "中"}]}
        hits = next((v for k, v in db.items() if k in keyword), [{"name": "家常小炒", "level": "易"}])
        return json.dumps(hits, ensure_ascii=False)


class CheckPantryTool(Tool):
    name: str = "check_pantry"
    description: str = "检查某项食材的库存数量。"

    def call(self, ingredient: Annotated[str, "食材名"]) -> str:
        stock = {"鸡蛋": 6, "番茄": 2, "盐": 1, "牛腩": 0}
        return json.dumps({"ingredient": ingredient, "qty": stock.get(ingredient, 0)}, ensure_ascii=False)


class SubstituteTool(Tool):
    name: str = "suggest_substitute"
    description: str = "为缺货食材给出替代品。"

    def call(self, ingredient: Annotated[str, "缺货食材"]) -> str:
        sub = {"牛腩": "鸡胸肉", "香菜": "葱花"}
        return json.dumps({"ingredient": ingredient, "substitute": sub.get(ingredient, "可省略")}, ensure_ascii=False)


class NextStepTool(Tool):
    name: str = "next_step"
    description: str = "获取某道菜的第 step 步操作。"

    def call(self, dish: Annotated[str, "菜名"], step: Annotated[int, "第几步，从 1 开始"] = 1) -> str:
        steps = ["切番茄、打散蛋液", "热油把蛋炒熟盛出", "下番茄炒出汁水", "倒入蛋翻匀，加盐出锅"]
        idx = step - 1
        text = steps[idx] if 0 <= idx < len(steps) else "已完成"
        return json.dumps({"dish": dish, "step": step, "do": text, "has_next": idx + 1 < len(steps)}, ensure_ascii=False)


# —— 各阶段指令（含一个嵌套 DynamicInstructions 的例子）——
class DiscoverInstructions(DynamicInstructions):
    def body(self, s: LanguageModelSession) -> DynamicInstructionStream:
        yield Instructions("你在【选菜】阶段：用 search_recipes 按用户喜好推荐 1-2 道菜，简短。")
        yield SearchRecipesTool()


class ShoppingInstructions(DynamicInstructions):
    def body(self, s: LanguageModelSession) -> DynamicInstructionStream:
        yield Instructions("你在【备料】阶段：用 check_pantry 核对食材，缺货就用 suggest_substitute 给替代。")
        yield CheckPantryTool()
        yield SubstituteTool()


class SafetyInstructions(DynamicInstructions):
    def body(self, s: LanguageModelSession) -> DynamicInstructionStream:
        yield Instructions("随时提醒用火安全与关火时机。")


class CookingInstructions(DynamicInstructions):
    def body(self, s: LanguageModelSession) -> DynamicInstructionStream:
        yield Instructions("你在【烹饪】阶段：用 next_step 一步步指导，每次只讲一步，简洁。")
        yield NextStepTool()
        yield SafetyInstructions()  # 嵌套 DynamicInstructions：自带指令、可跨场景复用


def staged(label: str) -> ProfileModify:
    """可复用修饰符：给阶段配置加上进入/离开日志（观察 onActivate/onDeactivate）。"""
    return lambda profile: (
        profile
        .on_activate(lambda: print(f"    >> 进入【{label}】"))
        .on_deactivate(lambda: print(f"    << 离开【{label}】"))
    )


class KitchenAssistant(DynamicProfile):
    """顶层编排：按阶段选出唯一激活的子配置，并赋予不同的模型参数。"""

    def body(self, session: LanguageModelSession) -> Profile:
        match getattr(session.state, "stage", "discover"):
            case "discover":
                return Profile(instructions=DiscoverInstructions()).temperature(0.8).modifier(staged("选菜"))
            case "shopping":
                return Profile(instructions=ShoppingInstructions()).temperature(0.3).modifier(staged("备料"))
            case _:
                return (
                    Profile(instructions=CookingInstructions())
                    .temperature(0.2)
                    .reasoning("high")
                    .modifier(staged("烹饪"))
                )


async def kitchen_demo() -> None:
    banner("完整 DSL 编排 [实时]：多阶段厨房助手（选菜 → 备料 → 烹饪）")

    # 全局钩子 + 历史变换在顶层声明，穿透到每个阶段。
    assistant = (
        KitchenAssistant()
        .on_tool_call(lambda c: print(f"    · {c.name}({c.arguments})"))
        .history_transform(lambda h: h[-12:])
    )
    session = LanguageModelSession(
        assistant, llm_config=LLMConfig.deepseek("deepseek-v4-flash"), stage="discover"
    )

    print("\n[阶段 1 · 选菜]  temperature=0.8")
    print("助手：", await session.respond("我想做点和番茄有关的简单家常菜。"))

    session.state.stage = "shopping"  # 切换 → 触发 选菜 离开 + 备料 进入
    print("\n[阶段 2 · 备料]  temperature=0.3")
    print("助手：", await session.respond("做番茄炒蛋，我家鸡蛋、番茄、牛腩够不够？不够给个替代。"))

    session.state.stage = "cooking"  # 切换 → 备料 离开 + 烹饪 进入（高推理）
    print("\n[阶段 3 · 烹饪]  temperature=0.2, reasoning=high")
    print("助手：", await session.respond("好，开始做番茄炒蛋，告诉我第一步。"))


async def main() -> None:
    await feature_tour()
    await kitchen_demo()


if __name__ == "__main__":
    asyncio.run(main())
