# App（应用外壳）

把 `Session` / `SessionGroup` 接到外部世界的部署/集成边界。可选；框架内直接 `respond()` 即可。

::: yaoagent.app.App
    options:
      members:
        - body
        - run
        - stream
        - on_stream
        - on_output
        - on_log
