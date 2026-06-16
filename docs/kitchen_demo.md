# 厨房助手

源代码：[example.py](https://github.com/HawkonLi/yao_agent/blob/main/example.py) 中的 `kitchen_demo()`

三层声明式编排：

- **顶层 `KitchenAssistant(DynamicProfile)`** — 按 `session.state.stage` 切换子 Profile
- **每阶段对应一个子 Profile** — 各绑定自己的温度、推理力度、工具
- **烹饪阶段嵌套 `SafetyInstructions`** — 展示指令组件嵌套

`on_tool_call` 和 `history_transform` 在顶层声明，穿透到所有阶段。

```bash
python3 example.py
```
