# Profile（配置）

绑定一组动态指令 + 模型参数 + 生命周期钩子的不可变单元。

::: yaoagent.profile.Profile
    options:
      members:
        - model
        - temperature
        - reasoning
        - history_transform
        - modifier
        - on_prompt
        - on_response
        - on_tool_call
        - on_tool_output
        - on_activate
        - on_deactivate
        - activation_key

::: yaoagent.profile.DynamicProfile
    options:
      members:
        - body
        - model
        - temperature
        - reasoning
        - history_transform
        - modifier
        - on_prompt
        - on_response
        - on_tool_call
        - on_tool_output
        - on_activate
        - on_deactivate

::: yaoagent.profile.DynamicProfileModifier
    options:
      members:
        - body

::: yaoagent.profile.ProfileModify
