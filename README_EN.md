# YaoAgent

> 爻者，言乎变者也。—— *I Ching, Book of Changes*

In the *I Ching*, a **yáo (爻)** is the basic unit of a trigram. Two kinds of line — unbroken
(⚊) and broken (⚋) — arranged in six positions produce 64 hexagrams. The entire cosmos,
captured in the arrangement of six lines.

YaoAgent is named after this idea: **a good agent isn't written — it's composed.**

It's a composable orchestration framework. You declare how pieces fit together — instructions,
tools, hooks, environment — and the framework handles lifecycle management, cross-session
data injection, tool self-healing, and tracing. What you see is a clean orchestration file.
State and parameter details are tucked away. Change the composition, change the behavior.

Inspired by Apple [Foundation Models](https://developer.apple.com/documentation/foundationmodels/composing-dynamic-sessions-with-instructions-and-profiles) dynamic sessions API and [SwiftUI](https://developer.apple.com/documentation/SwiftUI).

[Full Documentation →](https://hawkonli.github.io/yao_agent/)

## Installation

```bash
pip install yaoagent
```

Drop a `.env` in your project directory (the framework auto-discovers it by walking upward):

```
DEEPSEEK_API_KEY=sk-...
```

## Architecture

![Architecture](https://raw.githubusercontent.com/HawkonLi/yao_agent/main/docs/assets/yaoagent-architecture.svg)

Top-down: **App** (optional shell) → **SessionGroup** (multi-agent) → **LanguageModelSession** (agent) → **DynamicProfile** (select Profile) → **Profile** (params + hooks) → **DynamicInstructions** (`yield` instructions & tools). `Environment` flows laterally across sessions.

## Quick Start

```python
import asyncio
from typing import Annotated
from yaoagent import *

# ═══════════════════════ 1. Tools ═══════════════════════
class SearchWeb(Tool):
    name: str = "search_web"
    description: str = "Search the web."
    def call(self, keyword: Annotated[str, "search keyword"]) -> str:
        return f'Results for "{keyword}"...'

class Summarize(Tool):
    name: str = "summarize"
    description: str = "Summarize text."
    def call(self, text: Annotated[str, "text to summarize"]) -> str:
        return f"Summary: {text[:100]}..."

# ═══════════════════════ 2. Instructions ═══════════════════════
class ResearchInstructions(DynamicInstructions):
    def body(self, session) -> DynamicInstructionStream:
        yield Instructions("You are a researcher. Search first, then summarize.")
        yield SearchWeb()
        yield Summarize()

# ═══════════════════════ 3. Profile ═══════════════════════
class ResearchProfile(DynamicProfile):
    def body(self, session) -> Profile:
        return (Profile(instructions=ResearchInstructions())
                .temperature(0.7)
                .model("deepseek-v4-flash")
                .on_prompt(lambda p: print(f"Prompt: {p}"))
                .on_tool_call(lambda c: print(f"Tool: {c.name}"))
                .on_response(lambda r: print(f"Response: {r}"))

# ═══════════════════════ 4. Run ═══════════════════════
async def single_agent():
    session = LanguageModelSession(ResearchProfile(), llm_config=LLMConfig.deepseek())
    answer = await session.respond("Latest AI news?")
    print(answer, answer.usage)  # Response is a str subclass

# ═══════════════════════ 5. Multi-agent ═══════════════════════
class Blackboard(EnvironmentObject):
    findings: list[str] = []

class SaveToBoard(Tool):
    name: str = "save"
    description: str = "Write to the shared blackboard."
    board = Environment(Blackboard)  # declared dependency, excluded from model schema
    def call(self, note: str) -> str:
        self.board.findings.append(note); return "Saved"

class BoardInstructions(DynamicInstructions):
    def body(self, session) -> DynamicInstructionStream:
        yield Instructions("Search and save findings to the blackboard.")
        yield SearchWeb()
        yield SaveToBoard()

class BoardProfile(DynamicProfile):
    def body(self, session) -> Profile:
        return Profile(instructions=BoardInstructions()).temperature(0.5)

async def multi_agent():
    group = (
        SessionGroup(
            LanguageModelSession(BoardProfile()),       # researcher → writes to board
            LanguageModelSession(ResearchProfile()),    # writer → reads from board
        )
        .group_style(Style.sequential)
        .input_style(InputStyle.broadcast)
        .environment(Blackboard())
        .llm_config(LLMConfig.deepseek())
    )
    print(await group.run("Latest AI advances"))

# ═══════════════════════ 6. App ═══════════════════════
class ResearchApp(App):
    def body(self) -> SessionGroup:
        return (SessionGroup(
            LanguageModelSession(BoardProfile()),
            LanguageModelSession(ResearchProfile()),
        ).group_style(Style.sequential).environment(Blackboard()).llm_config(LLMConfig.deepseek()))

    def on_stream(self, event): print(event)

async def app_run():
    envelope = await ResearchApp().run("AI trends")
    print(envelope["output"], envelope["usage"])

    async for event in ResearchApp().stream("AI trends"):
        print(event)

asyncio.run(app_run())
```

> 6 steps covering YaoAgent's full DSL surface:
>
> - **`yield` = declarative composition**: compose instructions, tools, nesting with generators; re-evaluated per request
> - **Chained modifiers**: `.temperature(0.7).model("deepseek-v4-flash")`
> - **8 lifecycle hooks**: `on_prompt` / `on_response` / `on_response_stream` / `on_reasoning_stream` / `on_tool_call` / `on_tool_output` / `on_activate` / `on_deactivate`
> - **`Environment` type-injection**: declare `Environment(Blackboard)`, framework injects, excluded from model schema
> - **`SessionGroup` topology**: three axes — `group_style` + `input_style` + `output_style`
> - **`App` unified delivery**: one `body()` yields `run()` (batch JSON envelope) and `stream()` (real-time events)

## Core Concepts

| Type | Role |
|---|---|
| `Instructions` | A block of model-visible instruction text. |
| `Tool` | A model-callable capability; `call` signature auto-generates parameter schema. Supports sync and async. |
| `DynamicInstructions` | `body()` is a generator; `yield` instructions, tools, nested instructions. Re-evaluated before every request. |
| `Profile` | Binds instructions + model params (`model`/`temperature`/`reasoning`) + lifecycle hooks. Immutable. |
| `DynamicProfile` | `body()` selects one `Profile` by state; outer modifiers pass through to inner layers. |
| `LanguageModelSession` | An agent: holds Profile, state, environment, history. `respond()` / `stream_response()`. |
| `EnvironmentObject` / `Environment` | Cross-agent shared objects, injected and resolved by type (~ SwiftUI `@EnvironmentObject`). |
| `SessionGroup` | Multi-agent topology (sequential/parallel/loop). Nestable, `run()`-able. |
| `App` | Deployment shell (optional): one `body()` yields `run()` (batch) + `stream()` (real-time). |

Three-layer structure: `DynamicProfile` (which one) → `Profile` (params/hooks) → `DynamicInstructions` (instructions/tools).

## Chained Modifiers

Modifiers use chainable Swift-style names. Can be placed on `Profile` or on `DynamicProfile` (with passthrough to inner layers):

```python
class MyProfile(DynamicProfile):
    def body(self, session) -> Profile:
        return (Profile(instructions=MyInstructions())
                .model("deepseek-v4-pro")
                .temperature(0.7)
                .reasoning("high")
                .on_tool_call(lambda c: log(c))
                .history_transform(lambda h: h[-20:]))
```

- **Value modifiers** (`.model/.temperature/.reasoning/.history_transform`): outer = default, **inner wins**.
- **Hook modifiers** (`.on_*`): **accumulate** across layers, outer fires first.

### Reusable Custom Modifiers

Bundle params + hooks into a named, chainable unit:

```python
def staged(label: str) -> ProfileModify:
    return lambda p: (p.on_activate(lambda: print(f">> {label}"))
                       .on_deactivate(lambda: print(f"<< {label}")))

class Debug(DynamicProfileModifier):
    def body(self, content: Profile) -> Profile:
        return content.temperature(0.0).on_response(lambda r: print(r))

Profile(instructions=MyInstructions()).temperature(0.8).modifier(staged("Writing"))
Profile(instructions=MyInstructions()).modifier(Debug())
```

## Precedence (high to low)

1. Call-site: `respond(prompt, temperature=0.0)`
2. `with` override: `with session.using(temperature=0.0): ...`
3. Profile layer: values on `Profile` / `DynamicProfile`

## Lifecycle Hooks

Chain-declared, sync or async. Hooks can close over `session` to read/write session state.

| Hook | Fires when |
|---|---|
| `on_prompt(fn)` | Before request (arg: prompt) |
| `on_response(fn)` | After final reply (arg: text; compress history here) |
| `on_response_stream(fn)` | Answer delta (streaming only) |
| `on_reasoning_stream(fn)` | Reasoning delta (streaming, DeepSeek reasoning models) |
| `on_tool_call(fn)` | Before tool execution (arg: `ToolCall`; **raise to deny**) |
| `on_tool_output(fn)` | After tool output (args: `ToolCall, output`) |
| `on_activate(fn)` | Profile becomes active (init hook) |
| `on_deactivate(fn)` | Profile is switched away (cleanup hook) |

`on_activate`/`on_deactivate` fire when a top-level `DynamicProfile` switches active sub-profiles.

## Tool Access to Session State

Tools are isolated by default. Use `self.session` when access is needed — `call` signature stays pure
(model args only). Framework auto-binds the current session during tool execution:

```python
class RememberTool(Tool):
    name: str = "remember"
    description: str = "Remember a user preference."
    def call(self, key: str, value: str) -> str:
        self.session.state.prefs[key] = value
        return f"Remembered {key}={value}"

session = LanguageModelSession(MyProfile(), llm_config=cfg, prefs={})
```

## Private State & Shared Environment

Two layers of state, clearly separated:

- **Private `state`**: owned by the session, persistent across requests. Pass a typed dataclass for safety:
  ```python
  @dataclass
  class KitchenState:
      stage: str = "discover"
      cart: list[str] = field(default_factory=list)

  session = LanguageModelSession(KitchenProfile(), llm_config=cfg, state=KitchenState())
  ```
- **Shared `environment`**: cross-agent shared objects, injected and resolved **by type**:
  ```python
  class Notebook(EnvironmentObject):
      def __init__(self): self.findings = []

  class SaveFinding(Tool):
      name: str = "save_finding"; description: str = "Record a finding"
      notebook = Environment(Notebook)
      def call(self, text: str) -> str:
          self.notebook.findings.append(text); return "Recorded"

  session.environment(Notebook())
  ```

## Multi-Agent Orchestration

Combine sessions into a nestable, `run()`-able `SessionGroup`:

```python
pipeline = (
    SessionGroup(
        parallel(researcher_a, researcher_b),
        synthesizer,
        loop(reviser, until=lambda o: "[OK]" in o, max_iters=3),
    )
    .group_style(Style.sequential)
    .environment(Notebook())
)
answer = await pipeline.run("Research topic")
```

Three orthogonal axes:

- **Style** `group_style`: `Style.sequential` / `Style.parallel` / `Style.loop(until=, max_iters=)`
- **Input** `input_style`: `InputStyle.pipe` (previous output → next) / `InputStyle.broadcast` (all get original input)
- **Output** `output_style`: `OutputStyle.last_session` / `OutputStyle.merge(fn)`

Members can be sessions or nested `SessionGroup`s. Parallel = `asyncio.gather`.

## Session History & Continuation

`session.history` is a full OpenAI-format transcript. Seed with prior messages to continue:

```python
session = LanguageModelSession(MyProfile(), llm_config=cfg, history=prior_messages)
```

Compress in `on_response` hooks or trim with `.history_transform()`.

## Streaming

```python
async for delta in session.stream_response("Beijing weather?"):
    print(delta, end="", flush=True)
```

Reasoning and answer streams are demuxed into separate hooks:

```python
Profile(instructions=...)
    .on_response_stream(handle_answer)
    .on_reasoning_stream(handle_thinking)
```

## Tracing & Observability

Attach a `Trace` to get structured events at every key point:

```python
session = LanguageModelSession(
    profile, llm_config=cfg,
    trace=Trace(jsonl("runs/exp1.jsonl"), console, level="debug"),
)
```

Event types: `request` / `tool_call` / `tool_output` / `response` / `activate` / `deactivate` / `error`.
Group runs emit `group_start` / `group_end` / `member_start` / `member_end` / `iteration`.
All events within one run share a `run_id` for correlation.

## Runtime (Output Encapsulation)

Three delivery channels for a clean separation of concerns:

| Channel | Audience | Typical destination |
|---|---|---|
| `log` | Developer / archive | Trace → jsonl / console |
| `stream` | End user (real-time) | SSE / websocket / terminal |
| `output` | Upstream system | Structured JSON |

## Error System

All framework errors are `YaoError` subclasses with stable codes and natural-language explanations.

**Self-healing**: when the model calls a tool incorrectly (recoverable), the framework feeds the error
explanation back as tool output, letting the model self-correct in the next round.

## Running Examples

```bash
python3 example.py          # Feature tour + multi-stage kitchen orchestration
python3 example_group.py    # Full multi-agent DSL: parallel → synthesize → self-revise
```

## License

[MIT](LICENSE).
