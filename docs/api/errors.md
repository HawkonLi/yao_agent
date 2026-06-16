# Errors（错误系统）

所有框架错误都是 `YaoError` 子类，带稳定错误码与自然语言解释。

::: yaoagent.errors.YaoError
    options:
      members:
        - format
        - explain
        - to_dict
        - recoverable

::: yaoagent.errors.ErrorCode

::: yaoagent.errors.ConfigError

::: yaoagent.errors.ResolveError

::: yaoagent.errors.ToolError

::: yaoagent.errors.ModelError
