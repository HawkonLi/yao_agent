# 功能展示

源代码：[example.py](https://github.com/HawkonLi/yao_agent/blob/main/example.py)

9 个场景逐步展示 YaoAgent 核心能力：

1. **响应式指令** — `DynamicInstructions.body()` 每次请求前重新求值
2. **自动 schema** — `Tool.call` 类型注解自动生成 JSON schema
3. **生命周期钩子** — `on_prompt` / `on_response` / `on_tool_call` / `on_tool_output`
4. **with 重写** — `with session.using(temperature=...):` 临时覆盖参数
5. **参数校验** — 工具参数不合法自动回灌错误让模型自愈
6. **穿透传值** — `DynamicProfile` 上的温度/模型向子 Profile 穿透
7. **错误系统** — `YaoError` 的 code / explain / to_dict
8. **流式输出** — `stream_response()` 逐 token 产出
9. **厨房助手** — 多阶段编排（选菜 → 备料 → 烹饪），含嵌套 SafetyInstructions

```bash
python3 example.py
```
