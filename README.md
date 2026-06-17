# YaoAgent

> 爻者，言乎变者也。——《易经·系辞》

爻，八卦之本。阴阳二爻不同排列即成不同卦象，六十四卦无非这爻的组合。

YaoAgent 取名于此：**好的智能体不是写出来的，是组合出来的。**

它是一个组合式编排框架。你声明零件怎么拼——指令、工具、钩子、环境，框架
负责生命周期管理、跨会话数据注入、工具自愈和日志追踪。你看到的是一个干净的编排文件，
状态和参数的细节都被收起。拼法不同，行为就不同。

Inspired by Apple [Foundation Models](https://developer.apple.com/documentation/foundationmodels/composing-dynamic-sessions-with-instructions-and-profiles) dynamic sessions API and [SwiftUI](https://developer.apple.com/documentation/SwiftUI).

[完整文档 →](https://hawkonli.github.io/yao_agent/)

## 安装

```bash
pip install yaoagent
```

在项目目录放一个 `.env`（框架会自动向上查找加载）：

```
DEEPSEEK_API_KEY=sk-...
```

## 概念速览

![架构图](https://raw.githubusercontent.com/HawkonLi/yao_agent/main/docs/assets/yaoagent-architecture.svg)

自顶向下：**App**（可选部署外壳）→ **SessionGroup**（多智能体编排）→ **LanguageModelSession**（智能体）→ **DynamicProfile**（智能体配置 Profile）→ **Profile**（静态特征）→ **DynamicInstructions**（`yield` 声明指令与工具）。`Environment` 横向跨会话共享。

## 快速上手

```python
import asyncio
from typing import Annotated
from yaoagent import *

# ═══════════════════════ 1. 工具层 ═══════════════════════
class SearchWeb(Tool):
    name: str = "search_web"
    description: str = "搜索互联网。"
    def call(self, keyword: Annotated[str, "搜索关键词"]) -> str:
        return f'关于 "{keyword}" 的搜索结果...'

class Summarize(Tool):
    name: str = "summarize"
    description: str = "总结文本。"
    def call(self, text: Annotated[str, "待总结文本"]) -> str:
        return f"摘要：{text[:100]}..."

# ═══════════════════════ 2. 指令层 ═══════════════════════
class ResearchInstructions(DynamicInstructions):
    def body(self, session) -> DynamicInstructionStream:
        yield Instructions("你是研究员。先搜索，再总结。用中文回答。")
        yield SearchWeb()
        yield Summarize()

# ═══════════════════════ 3. Profile 层 ═══════════════════════
class ResearchProfile(DynamicProfile):
    def body(self, session) -> Profile:
        return (Profile(instructions=ResearchInstructions())
                .temperature(0.7)                              # 链式修饰符：值类内层优先
                .model("deepseek-v4-flash")
                .on_prompt(lambda p: print(f"请求: {p}"))      # 生命周期钩子：on_prompt
                .on_tool_call(lambda c: print(f"工具: {c.name}"))  # on_tool_call
                .on_response(lambda r: print(f"回复: {r}"))    # on_response

# ═══════════════════════ 4. 运行 ═══════════════════════
async def single_agent():
    session = LanguageModelSession(ResearchProfile(), llm_config=LLMConfig.deepseek())
    answer = await session.respond("今天 AI 新闻？")
    # answer 是 Response（str 子类）→ 可直接打印
    print(answer, answer.usage)

# ═══════════════════════ 5. SessionGroup 多智能体 ═══════════════════════
# 共享黑板：按类型注入，跨智能体穿透
class Blackboard(EnvironmentObject):
    findings: list[str] = []

class SaveToBoard(Tool):
    name: str = "save"
    description: str = "记到共享黑板。"
    board = Environment(Blackboard)  # 声明依赖，不进模型 schema
    def call(self, note: str) -> str:
        self.board.findings.append(note); return "已记录"

class BoardInstructions(DynamicInstructions):
    def body(self, session) -> DynamicInstructionStream:
        yield Instructions("先搜索再记到黑板。")
        yield SearchWeb()
        yield SaveToBoard()

class BoardProfile(DynamicProfile):
    def body(self, session) -> Profile:
        return Profile(instructions=BoardInstructions()).temperature(0.5)

async def multi_agent():
    group = (
        SessionGroup(
            LanguageModelSession(BoardProfile()),          # 研究员 → 写黑板
            LanguageModelSession(ResearchProfile()),       # 写作者 → 读黑板
        )
        .group_style(Style.sequential)                    # 串行：一个跑完再跑下一个
        .input_style(InputStyle.broadcast)                 # 各拿原输入，靠黑板通信
        .environment(Blackboard())                         # 黑板穿透所有成员
        .llm_config(LLMConfig.deepseek())
    )
    print(await group.run("AI 最新进展"))

# ═══════════════════════ 6. App 部署外壳 ═══════════════════════
class ResearchApp(App):
    """同一个 body()，两种交付面：run() 批处理 / stream() 实时流。"""
    def body(self) -> SessionGroup:
        return (SessionGroup(
            LanguageModelSession(BoardProfile()),
            LanguageModelSession(ResearchProfile()),
        ).group_style(Style.sequential).environment(Blackboard()).llm_config(LLMConfig.deepseek()))

    def on_stream(self, event): print(event)  # 覆写：实时事件往哪送

async def app_run():
    env = await ResearchApp().run("AI 趋势")     # 批量：返回 JSON 信封
    print(env["output"], env["usage"])

    async for event in ResearchApp().stream("AI 趋势"):  # 实时：事件流
        print(event)

asyncio.run(app_run())
````

> 上面 6 步展示了 YaoAgent 的完整 DSL：
>
> - **`yield`** **= 声明式组合**：指令、工具、嵌套指令用生成器编排，每次请求前重新求值
> - **修饰符链式传递**：`.temperature(0.7).model("deepseek-v4-flash")`
> - **生命周期钩子 8 种**：`on_prompt` / `on_response` / `on_response_stream` / `on_reasoning_stream` / `on_tool_call` / `on_tool_output` / `on_activate` / `on_deactivate`
> - **`Environment`** **按类型注入**：声明 `Environment(Blackboard)`，框架自动注入，不进模型 schema
> - **`SessionGroup`** **拓扑编排**：三元组 `group_style` + `input_style` + `output_style`，成员递归嵌套
> - **`App`** **统一交付**：同一个 `body()` 出 `run()`（批量 JSON 信封）和 `stream()`（实时事件流）

## 核心概念

| 类型                                  | 作用                                                           |
| ----------------------------------- | ------------------------------------------------------------ |
| `Instructions`                      | 一段模型可见的指令文本。                                                 |
| `Tool`                              | 可被模型调用的能力；`call` 的类型注解自动生成参数 schema，支持同步/异步。                 |
| `DynamicInstructions`               | `body()` 是生成器，用 `yield` 声明指令、工具、嵌套指令；每次请求前重新求值。              |
| `Profile`                           | 绑定一组动态指令 + 模型参数（model/temperature/reasoning）+ 生命周期钩子，不可变。    |
| `DynamicProfile`                    | `body()` 按状态选出一个 `Profile`；外层修饰符穿透到内层。                       |
| `LanguageModelSession`              | 一个智能体：持有 Profile、状态、环境、历史；`respond()` / `stream_response()`。 |
| `EnvironmentObject` / `Environment` | 跨智能体共享的对象，按类型注入与取用（≈ SwiftUI `@EnvironmentObject`）。          |
| `SessionGroup`                      | 多智能体拓扑编排（串/并/循环），可嵌套可 `run`。                                 |
| `App`                               | 部署外壳（可选）：同一个 `body()` 出 `run()` 批量信封 + `stream()` 实时事件流。     |

三层结构：`DynamicProfile`（选哪个） → `Profile`（参数/钩子） → `DynamicInstructions`（指令/工具）。

## 链式修饰符（命名对齐 Swift DSL）

修饰符与 Swift 同名，链式书写；既能挂在终态 `Profile`，也能挂在外层 `DynamicProfile` 并“穿透”到内层：

```python
class MyProfile(DynamicProfile):
    def body(self, session) -> Profile:
        return (Profile(instructions=MyInstructions())
                .model("deepseek-v4-pro")
                .temperature(0.7)
                .reasoning("high")
                .on_tool_call(lambda c: log(c))      # 生命周期钩子
                .history_transform(lambda h: h[-20:]))
```

- **值类**（`.model/.temperature/.reasoning/.history_transform`）：外层作默认值，**内层优先**。
- **钩子类**（`.on_*`）：跨层**累加**，外层先触发。

### 可复用的自定义修饰符

把一组成套的参数与钩子封装成一个可命名、可复用、可链式挂载的修饰符
（对应 Apple 的 `DynamicProfileModifier` / `.modifier(_:)`）。可以是函数，也可以是类：

```python
# 函数形式：返回 Profile -> Profile
def staged(label: str) -> ProfileModify:
    return lambda p: (p.on_activate(lambda: print(f">> {label}"))
                       .on_deactivate(lambda: print(f"<< {label}")))

# 类形式：实现 body(content)
class Debug(DynamicProfileModifier):
    def body(self, content: Profile) -> Profile:
        return content.temperature(0.0).on_response(lambda r: print(r))

Profile(instructions=MyInstructions()).temperature(0.8).modifier(staged("写作"))
Profile(instructions=MyInstructions()).modifier(Debug())
```

## 参数优先级（从高到低）

1. 调用点：`respond(prompt, temperature=0.0)`
2. `with` 临时重写：`with session.using(temperature=0.0): ...`（块内生效，离开还原）
3. 配置层：`Profile` / `DynamicProfile` 上的取值

## 生命周期钩子

链式声明，可同步或异步；钩子可闭包捕获 `session` 以读写会话状态。

| 钩子                        | 触发时机                                     |
| ------------------------- | ---------------------------------------- |
| `on_prompt(fn)`           | 发起请求前（入参 prompt）                         |
| `on_response(fn)`         | 得到最终回复后（入参 text；可在此压缩历史）                 |
| `on_response_stream(fn)`  | 流式答案增量（仅 stream\_response）               |
| `on_reasoning_stream(fn)` | 流式思考增量（仅 stream\_response，DeepSeek 推理模型） |
| `on_tool_call(fn)`        | 执行工具前（入参 `ToolCall`；**抛异常即拒绝**）          |
| `on_tool_output(fn)`      | 工具产出后（入参 `ToolCall, output`）             |
| `on_activate(fn)`         | 配置成为激活态时（适合初始化）                          |
| `on_deactivate(fn)`       | 配置被切换走时（适合清理）                            |

`on_activate`/`on_deactivate` 由顶层 `DynamicProfile` 切换激活子配置时自动触发。

## 工具访问会话状态

工具默认是隔离的。需要会话时，用 `self.session` 访问即可（对标 Apple 的 `@SessionProperty`）——
`call` 签名保持纯净，只放模型参数；框架在工具执行期间自动绑定当前会话。借此读写
`session.state` / `session.history`，实现有状态工具（记忆、技能激活、给工具传上下文等）：

```python
class RememberTool(Tool):
    name: str = "remember"
    description: str = "记住一项用户偏好。"
    def call(self, key: str, value: str) -> str:   # 签名纯净，不掺框架参数
        self.session.state.prefs[key] = value      # self.session 自动可用
        return f"已记住 {key}={value}"

# 用 prefs={} 初始化会话状态，工具体里就无需处理默认值
session = LanguageModelSession(MyProfile(), llm_config=cfg, prefs={})
```

## 私有状态与共享环境

两层状态，边界清楚：

- **私有** **`state`（≈** **`@State`）**：会话自己拥有、跨请求持久。推荐传入**显式类型化对象**（dataclass），
  比无类型口袋安全：
  ```python
  @dataclass
  class KitchenState:
      stage: str = "discover"
      cart: list[str] = field(default_factory=list)

  session = LanguageModelSession(KitchenProfile(), llm_config=cfg, state=KitchenState())
  # body / 工具里 session.state.stage —— 类型已知、IDE/mypy 可查
  # （不传 state 时，关键字参数仍会汇成一个 SimpleNamespace，方便快速脚本）
  ```
- **共享** **`environment`（≈** **`@EnvironmentObject`）**：跨智能体共享的对象，**按类型**注入与取用：
  ```python
  class Notebook(EnvironmentObject):
      def __init__(self): self.findings = []

  class SaveFinding(Tool):
      name: str = "save_finding"; description: str = "记一条发现"
      notebook = Environment(Notebook)               # 按类型注入，不进 schema
      def call(self, text: str) -> str:
          self.notebook.findings.append(text); return "已记录"

  session.environment(Notebook())                    # 链式注入，可多个（一个类型一个实例）
  ```
  并发下保持「单一写者 + 同步读快照」即安全；需要跨 `await` 的多步更新就在你的环境对象里放一把
  `asyncio.Lock`。

## 多智能体编排

把多个会话（每个是一个智能体）按拓扑组合成一个可嵌套、可 `run` 的 `SessionGroup`：

```python
pipeline = (
    SessionGroup(
        parallel(researcher_a, researcher_b),          # 并行：同输入扇出
        synthesizer,                                   # 串行：上一步输出喂下一步
        loop(reviser, until=lambda o: "[OK]" in o, max_iters=3),  # 迭代到满足条件
    )
    .group_style(Style.sequential)                     # 顶层用串行把三段连起来
    .environment(Notebook())                           # 环境向所有成员（含嵌套子组）穿透
)
answer = await pipeline.run("研究主题")
```

group 由三个**正交维度**描述（编排约束另外两个的合法取值）：

- **编排** **`group_style`**（成员怎么跑）：`Style.sequential` / `Style.parallel` / `Style.loop(until=, max_iters=)`。
- **输入** **`input_style`**（成员收什么）：`InputStyle.pipe`（上一个输出喂下一个）/ `InputStyle.broadcast`（都拿原输入，靠共享 `environment` 通信）。
- **输出** **`output_style`**（谁的输出暴露给 group）：`OutputStyle.last_session` / `OutputStyle.merge(fn)`。

（`input_style` / `output_style` 命名描述的是**智能体之间的内部接线**，与面向用户的运行时输出层
`Runtime`（见下文）刻意区分。）每种编排自带默认输入/输出（如 sequential = pipe + last_session，
parallel = broadcast + merge），按需覆盖；非法组合（如 parallel + pipe）会报错。
便捷构造 `sequential() / parallel() / loop()` 即"编排 + 默认输入输出"。

```python
# 顺序跑、但成员各拿原输入、靠共享环境通信、返回末位成员（而非管道）：
SessionGroup(a, b).group_style(Style.sequential).input_style(InputStyle.broadcast).output_style(OutputStyle.last_session)
```

- 成员可以是会话，也可以是另一个 `SessionGroup`——递归嵌套。
- 并行就是 `asyncio.gather`；通信走共享 `environment`（黑板）或上下游的数据流。

## 会话历史与续接

`session.history` 是 OpenAI 格式的完整 transcript：包含 user 提示、工具调用、工具输出与
最终回复（不含指令）。可用既有历史种子初始化以续接对话：

```python
session = LanguageModelSession(MyProfile(), llm_config=cfg, history=prior_messages)
```

历史可在 `on_response` 钩子里压缩，或用 `.history_transform()` 在请求前做局部裁剪。

## 流式输出

`stream_response()` 是 `respond()` 的流式版：异步逐段产出最终回复的文本增量，
工具调用循环在内部静默处理，流结束后照常持久化完整 transcript。

```python
async for delta in session.stream_response("北京天气怎么样？"):
    print(delta, end="", flush=True)
```

底层模型把**答案**（content）和**思考**（reasoning，DeepSeek 推理模型）放在同一条流的不同字段里，
框架把它们 demux 成两个独立钩子，按需各接各的、零分支：

```python
Profile(instructions=...)
    .on_response_stream(handle_answer)     # 答案增量
    .on_reasoning_stream(handle_thinking)  # 思考增量（不想要就不写这行）
```

`stream_response()` 产出的流只含答案；思考只走 `on_reasoning_stream`，不混进答案、不进 transcript。

## 日志与可观测（实验复现）

绑一个 `Trace`，框架就在每个关键节点发**结构化事件**；`sink` 就是个 `Callable[[dict], None]`：

```python
session = LanguageModelSession(
    profile, llm_config=cfg,
    trace=Trace(jsonl("runs/exp1.jsonl"), console, level="debug"),
)
```

- 事件类型：`request`（含解析后的**完整配置快照**）/ `tool_call` / `tool_output` /
  `response`（含 token 用量与 `elapsed_ms`）/ `activate` / `deactivate` / `error`；
  多智能体编排另有 `group_start` / `group_end` / `member_start` / `member_end` / `iteration`。`debug` 级近乎全量。
- **关联 ID**：同一次 run（含其编排里所有成员/工具轮次）的事件共享一个 `run_id`，并发/嵌套时可归并到一条时间线。
- `SessionGroup.trace(t)` 把日志向所有成员穿透（成员自带的优先），整组事件自动带同一个 `run_id`。
- 内置 sink：`jsonl(path)`（一行一条，适合实验）、`console`；自定义就传任意 `lambda e: ...`。
- 接 SwanLab 等外部实验平台：`Trace(lambda e: swanlab.log(e))`——**适配器写在你的实验代码里，不进框架**。
- `session.describe()` 可随时导出当前解析出的配置快照（指令 / 工具 schema / 模型参数 / 状态）。

## 运行时输出封装（Runtime / Handler）

把“输出往哪送”从“生命周期里发生了什么”里拆出来，统一成 **Handler**——一组**形状 = 钩子**的方法
（`tool_call` / `response` / `response_stream` …），每个把事件打成 dict 丢给同一个 `sink`（目的地）。
`Runtime` 是装三个 Handler 的盒子，按受众分三个投递口：

| 投递口      | 给谁       | 典型去处                    |
| -------- | -------- | ----------------------- |
| `log`    | 开发者 / 留档 | Trace → jsonl / console |
| `stream` | 最终用户（实时） | SSE / websocket / 终端    |
| `output` | 上游系统     | 结构化 JSON                |

`Runtime` **永远有默认值**（模块级默认 + `ContextVar`），`session.runtime` 任何时候都拿得到非空对象。
在 `body` 里取投递口、把想要的事件接上去即可（不想要就不接，零分支）：

```python
class MyProfile(DynamicProfile):
    def body(self, session) -> Profile:
        io = session.runtime
        return (
            Profile(instructions=MyInstructions())
            .on_response_stream(io.stream.response_stream)  # 答案增量 → 实时给用户
            .on_tool_output(io.log.tool_output)             # 工具输出 → 留档
            .on_response(io.output.response)                # 最终回复 → 结构化给上游
        )
```

## App 级封装（部署 / 集成边界）

`Session` / `SessionGroup` 是"View"（可组合的智能体逻辑）；`App` 是把它们接到外部世界
（FastAPI、推荐系统、命令行）的最外层外壳——**可选**，框架内直接 `respond()` 即可。

参见上方快速上手第 6 步的完整示例，以及 [example\_app.py](example_app.py)。

## 输入（Prompt）

`respond()` 的入参就是 `str`，绝大多数场景直接传字符串即可。需要携带元数据/附件时用 `Prompt`
（`str` 子类，与 `Response` 对称，零破坏）：

```python
await session.respond(Prompt("分析这段", metadata={"lang": "zh"}))   # 钩子里可读 prompt.metadata
```

`attachments` 字段预留给将来的多模态输入。

## 返回值与用量

`respond()` 返回 `Response`——它是 `str` 子类（可直接打印/比较/拼接），额外携带
token 用量与结束原因：

```python
answer = await session.respond("北京天气怎么样？")
print(answer)              # 回复文本
print(answer.usage)        # Usage(prompt_tokens=..., completion_tokens=..., total_tokens=...)（跨工具轮次累加）
print(answer.finish_reason)
```

## 错误系统

所有框架错误都是 `YaoError` 子类，带稳定错误码与自然语言解释：

```python
try:
    await session.respond("...")
except ToolError as e:       # ConfigError / ResolveError / ToolError / ModelError
    e.code           # ErrorCode.INVALID_TOOL_ARGUMENTS
    str(e)           # "[YAO-3002] 工具参数不符合其 schema。 (tool='...')"
    e.explain()      # 面向人/模型的自然语言解释
    e.to_dict()      # 结构化暴露：code/name/explanation/context/cause
```

**参数校验自愈**：模型把工具叫错或参数不合法（可恢复错误）时，框架不会中断会话，
而是把自然语言解释作为工具结果回灌给模型，让它在下一轮自行改正；工具自身执行失败
（致命错误）才向上抛出。

## 在 Web 服务里用（FastAPI/Django）

框架是纯 asyncio，和 FastAPI 异步端点天然契合；`AsyncOpenAI` 客户端按事件循环 + 连接参数自动复用，
轻量并发没问题。几条纪律：

- **一对话一 session**：别把同一个可变 session 跨并发请求共享（`history`/`state` 会被写乱）。
- **共享 environment 按用户域**：全局可变就用单一写者 + 锁；注意多 worker 是多进程，
  内存对象不跨进程——要横向扩展就把共享状态外置到 Redis/DB。
- **工具别阻塞事件循环**：阻塞 IO 用 `async def call` 或 `asyncio.to_thread`。

## 运行示例

```bash
python3 example.py          # 能力速览 + 多阶段厨房编排智能体
python3 example_group.py    # 完整多智能体 DSL：并行调研 → 综述 → 自我精修，共享笔记本环境
```

- `example.py`：① 能力速览（响应式指令、自动 schema、生命周期钩子、`with` 重写 + reasoning、
  穿透传值、参数校验、错误系统、流式）；② 多阶段厨房助手（顶层 `DynamicProfile` 按阶段切换子配置）。
- `example_group.py`：用 `SessionGroup` 把多个智能体编排成 `sequential( parallel(...) → 综述 → loop(...) )`，
  并通过共享 `Notebook` 环境协作——集中体现多智能体 + 环境 + 嵌套拓扑。

## 许可证

[MIT](LICENSE)。
