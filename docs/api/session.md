# Session（会话）

核心执行单元：一个会话绑定一组配置 + 历史 + 私有状态 + 共享环境。

::: yaoagent.session.LanguageModelSession
    options:
      members:
        - respond
        - stream_response
        - run
        - resolve_request
        - using
        - max_round
        - environment
        - resolve_environment
        - describe

::: yaoagent.session.Response

::: yaoagent.session.Usage

::: yaoagent.session.Prompt

::: yaoagent.session.ResolvedRequest
