# Group（多智能体编排）

把多个会话按拓扑组合成可嵌套、可 `run` 的整体。

三正交维度：`group_style` + `input_style` + `output_style`。

::: yaoagent.group.SessionGroup
    options:
      members:
        - run
        - group_style
        - input_style
        - output_style
        - environment
        - llm_config
        - trace

::: yaoagent.group.Style
    options:
      members:
        - sequential
        - parallel
        - loop

::: yaoagent.group.InputStyle
    options:
      members:
        - pipe
        - broadcast

::: yaoagent.group.OutputStyle
    options:
      members:
        - last
        - pick
        - merge

::: yaoagent.group.Runnable

::: yaoagent.group.sequential
::: yaoagent.group.parallel
::: yaoagent.group.loop
