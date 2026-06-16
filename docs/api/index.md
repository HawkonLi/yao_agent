# API 概览

YaoAgent 的全部公开类型，按概念分层：

| 模块 | 核心类型 | 作用 |
|---|---|---|
| [session](session.md) | `LanguageModelSession`, `Response`, `Usage`, `ResolvedRequest`, `Prompt` | 会话、响应、输入 |
| [profile](profile.md) | `Profile`, `DynamicProfile`, `DynamicProfileModifier` | 配置层、修饰符 |
| [instructions](instructions.md) | `Instructions`, `DynamicInstructions`, `ResolvedInstructions` | 指令层 |
| [tool](tool.md) | `Tool`, `ToolCall`, `validate_arguments` | 工具、schema 自动生成、参数校验 |
| [group](group.md) | `SessionGroup`, `Style`, `InputStyle`, `OutputStyle` | 多智能体编排 |
| [app](app.md) | `App` | 部署 / 集成外壳 |
| [llm_config](llm_config.md) | `LLMConfig` | 模型连接配置 |
| [errors](errors.md) | `YaoError`, `ErrorCode`, `ConfigError`, `ToolError`, ... | 错误系统 |
| [environment](environment.md) | `EnvironmentObject`, `Environment` | 环境注入 |
| [trace](trace.md) | `Trace`, `console`, `jsonl` | 日志追踪 |
| [runtime](runtime.md) | `Runtime`, `Handler` | 运行时输出通道 |
