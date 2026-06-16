from __future__ import annotations

"""
基于 OpenAI 标准格式的模型后端（默认对接 DeepSeek）。

只负责一件事：把一次 `ResolvedRequest` 快照发给模型，并在模型请求调用工具时
驱动工具调用循环，直到拿到最终文本回复。
"""

import asyncio
import inspect
import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterable

from openai import AsyncOpenAI, OpenAIError

from .errors import ErrorCode, ModelError, ToolError, YaoError
from .tool import ToolCall, validate_arguments

if TYPE_CHECKING:
    from .llm_config import LLMConfig
    from .session import ResolvedRequest

# 按 (事件循环, 连接参数) 复用 AsyncOpenAI 客户端，避免并发下每次请求都新建连接池。
# 按事件循环分键，避免把某个循环的客户端用到另一个（已关闭的）循环上。
_clients: dict[tuple[Any, ...], AsyncOpenAI] = {}


def _get_client(options: dict[str, Any]) -> AsyncOpenAI:
    key = (
        id(asyncio.get_running_loop()),
        options["api_base_url"],
        options["api_key"],
        options["timeout"],
    )
    client = _clients.get(key)
    if client is None:
        client = AsyncOpenAI(
            api_key=options["api_key"],
            base_url=options["api_base_url"],
            timeout=options["timeout"],
        )
        _clients[key] = client
    return client


async def _fire(hooks: Iterable[Any], *args: Any) -> None:
    """
    依次触发钩子（同步或异步）。钩子抛出的异常向上传播，可用作校验拦截。
    :rtype: None
    """
    for hook in hooks:
        result = hook(*args)
        if inspect.isawaitable(result):
            await result


def _accumulate_usage(total: dict[str, int], usage: Any) -> None:
    """把单次响应的 token 用量累加进 total（用量缺失时安全跳过）。"""
    if usage is None:
        return
    for key in total:
        total[key] += getattr(usage, key, 0) or 0


def build_messages(request: "ResolvedRequest", history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    把请求快照拼成 OpenAI 消息列表：指令 → system，历史，本轮提示 → user。
    """
    messages: list[dict[str, Any]] = []
    system = "\n".join(item.text for item in request.instructions).strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(history)
    messages.append({"role": "user", "content": request.prompt})
    return messages


async def complete(
    request: "ResolvedRequest",
    llm_config: "LLMConfig",
    *,
    emit: Callable[..., None] | None = None,
    max_rounds: int | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, int], str | None]:
    """
    发起一次（可能含多轮工具调用的）模型请求。

    返回 (最终文本, 本轮新增的 transcript 消息, 累计 token 用量, 结束原因)。
    transcript 含 user 提示、所有 assistant 工具调用消息、tool 结果与最终回复，供持久化。
    用量为工具循环中各轮的累加。

    `max_rounds=None`：循环到模型给出文本；内置安全上限（8），撞顶报错（防失控）。
    `max_rounds=N`（由 `session.max_round(n)` 显式传入）：最多 N 次模型调用，撞顶**优雅返回**
    （把最后一轮的文本当结果，过程已在 transcript/environment 里），交还控制给编排方。
    """
    rounds = max_rounds if max_rounds is not None else 8
    options = llm_config.to_request_options()
    client = _get_client(options)

    # 优先用配置层（Profile）指定的模型，否则回退到连接配置的默认模型。
    model = request.model or options["model_name"]
    tools_by_name = {tool.name: tool for tool in request.tools}
    tools_payload = [tool.to_openai() for tool in request.tools] or None

    # 请求前对历史做局部变换（只影响本次请求，不动全局历史）。
    history = request.history
    if request.history_transform is not None:
        history = list(request.history_transform(list(history)))

    messages = build_messages(request, history)
    # 本轮新增的 transcript（不含 system 与历史），用于回写 session.history。
    turn: list[dict[str, Any]] = [{"role": "user", "content": request.prompt}]
    # 跨工具轮次累加 token 用量。
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # 提示已进入请求、发起前触发 onPrompt。
    await _fire(request.prompt_hooks, request.prompt)

    for _ in range(rounds):
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if tools_payload:
            kwargs["tools"] = tools_payload
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.reasoning is not None:
            # OpenAI 标准的推理力度字段，DeepSeek v4 系列支持。
            kwargs["reasoning_effort"] = request.reasoning

        try:
            response = await client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise ModelError(
                code=ErrorCode.MODEL_REQUEST_FAILED, cause=exc, model=model
            ) from exc
        _accumulate_usage(usage, getattr(response, "usage", None))
        message = response.choices[0].message

        if not message.tool_calls:
            text = message.content or ""
            turn.append({"role": "assistant", "content": text})
            await _fire(request.response_hooks, text)
            return text, turn, usage, response.choices[0].finish_reason

        # 回放助手的工具调用消息（裁成可安全重放的最小形态），再逐个执行工具。
        assistant_msg = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        }
        messages.append(assistant_msg)
        turn.append(assistant_msg)
        for raw_call in message.tool_calls:
            name = raw_call.function.name
            if emit:
                emit("tool_call", level="debug", name=name, arguments=raw_call.function.arguments)
            try:
                output = await _execute_tool_call(
                    name, raw_call.function.arguments, tools_by_name, request
                )
            except ToolError as err:
                if not err.recoverable:
                    raise
                # 可恢复错误：把自然语言解释作为工具结果回灌，让模型在下一轮改正重试。
                output = f"工具调用失败，请修正后重试：{err.explain()}"
            if emit:
                emit("tool_output", level="debug", name=name, output=output)
            tool_msg = {"role": "tool", "tool_call_id": raw_call.id, "content": output}
            messages.append(tool_msg)
            turn.append(tool_msg)

    # 循环用尽（最后一轮仍在调工具）。
    if max_rounds is not None:
        # 显式上限 = 编排方主动分步 → 优雅返回；过程已在 transcript/environment 里。
        await _fire(request.response_hooks, message.content or "")
        return message.content or "", turn, usage, response.choices[0].finish_reason
    # 默认上限 = 失控保护 → 报错。
    raise ModelError(code=ErrorCode.MAX_TOOL_ROUNDS_EXCEEDED, rounds=rounds)


async def stream(
    request: "ResolvedRequest",
    llm_config: "LLMConfig",
    turn: list[dict[str, Any]],
    *,
    emit: Callable[..., None] | None = None,
    max_tool_rounds: int = 8,
) -> AsyncIterator[str]:
    """
    流式版 `complete`：边到达边 `yield` 最终回复的文本增量。

    工具调用轮次在内部静默处理（拼装分片 tool_calls → 执行 → 回灌），只把最终回复
    的文本增量产出给调用方。本轮完整 transcript 追加到传入的 `turn`，供会话持久化。
    除异常外，只有当LLM输出一段不包含toolcall的内容，才会终止。

    答案增量同时触发 `response_stream_hooks`，思考(reasoning)增量触发 `reasoning_stream_hooks`
    （思考只走钩子，不混进产出的答案流，也不进 transcript）。
    """
    options = llm_config.to_request_options()
    client = _get_client(options)

    model = request.model or options["model_name"]
    tools_by_name = {tool.name: tool for tool in request.tools}
    tools_payload = [tool.to_openai() for tool in request.tools] or None

    history = request.history
    if request.history_transform is not None:
        history = list(request.history_transform(list(history)))

    messages = build_messages(request, history)
    turn.append({"role": "user", "content": request.prompt})

    await _fire(request.prompt_hooks, request.prompt)

    for _ in range(max_tool_rounds):
        # 获取环境参数
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if tools_payload:
            kwargs["tools"] = tools_payload
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.reasoning is not None:
            kwargs["reasoning_effort"] = request.reasoning

        content = ""
        # 按 index 增量拼装分片到达的 tool_calls。
        tool_acc: dict[int, dict[str, Any]] = {}
        try:
            response = await client.chat.completions.create(**kwargs)
            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # 思考(reasoning)增量：只走钩子/日志，不产出、不进 transcript。
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    await _fire(request.reasoning_stream_hooks, reasoning)
                    if emit:
                        emit("reasoning_stream", level="debug", chunk=reasoning)
                if delta.content:
                    content += delta.content
                    await _fire(request.response_stream_hooks, delta.content)
                    yield delta.content
                for piece in delta.tool_calls or []:
                    slot = tool_acc.setdefault(
                        piece.index, {"id": None, "name": None, "arguments": ""}
                    )
                    if piece.id:
                        slot["id"] = piece.id
                    if piece.function and piece.function.name:
                        slot["name"] = piece.function.name
                    if piece.function and piece.function.arguments:
                        slot["arguments"] += piece.function.arguments
        except OpenAIError as exc:
            raise ModelError(
                code=ErrorCode.MODEL_REQUEST_FAILED, cause=exc, model=model
            ) from exc

        if not tool_acc:
            # 无工具调用：本轮内容即最终回复。
            turn.append({"role": "assistant", "content": content})
            await _fire(request.response_hooks, content)
            if emit:
                emit("response", text=content, usage=None, finish_reason="stop")
            return

        ordered = [tool_acc[i] for i in sorted(tool_acc)]
        assistant_msg = {
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in ordered
            ],
        }
        messages.append(assistant_msg)
        turn.append(assistant_msg)
        for tc in ordered:
            if emit:
                emit("tool_call", level="debug", name=tc["name"], arguments=tc["arguments"])
            try:
                output = await _execute_tool_call(
                    tc["name"], tc["arguments"], tools_by_name, request
                )
            except ToolError as err:
                if not err.recoverable:
                    raise
                output = f"工具调用失败，请修正后重试：{err.explain()}"
            if emit:
                emit("tool_output", level="debug", name=tc["name"], output=output)
            tool_msg = {"role": "tool", "tool_call_id": tc["id"], "content": output}
            messages.append(tool_msg)
            turn.append(tool_msg)

    raise ModelError(code=ErrorCode.MAX_TOOL_ROUNDS_EXCEEDED, rounds=max_tool_rounds)


async def _execute_tool_call(
    name: str,
    arguments_json: str,
    tools_by_name: dict[str, Any],
    request: "ResolvedRequest",
) -> str:
    """
    执行单次工具调用：解析参数 → 校验 schema → 触发钩子 → 调用工具。

    与具体后端对象解耦（只收工具名 + 参数 JSON 字符串），流式/非流式共用。
    “模型把工具调用错了”这类问题（坏 JSON / 未知工具 / 参数不合法）抛可恢复
    `ToolError`，由上层回灌重试；工具自身执行失败抛致命 `ToolError`。

    工具内可通过 `self.session` 访问当前会话（由 `respond` 在工具循环外层绑定）。
    """

    #解析参数
    try:
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as exc:
        raise ToolError(
            "参数不是合法的 JSON。", code=ErrorCode.INVALID_TOOL_ARGUMENTS, tool=name, cause=exc
        ) from exc

    #校验schema
    tool = tools_by_name.get(name)
    if tool is None:
        available = ", ".join(tools_by_name) or "（无）"
        raise ToolError(
            f"可用工具：{available}。", code=ErrorCode.UNKNOWN_TOOL, tool=name
        )

    problem = validate_arguments(arguments, tool.parameters())
    if problem:
        raise ToolError(problem, code=ErrorCode.INVALID_TOOL_ARGUMENTS, tool=name)

    
    call = ToolCall(name=name, arguments=arguments)
    # 执行前触发 onToolCall：钩子抛异常即拒绝本次调用（致命，原样向上传播）。
    await _fire(request.tool_call_hooks, call)

    # 执行toolcall
    try:
        result = tool.call(**arguments)
        output = await result if inspect.isawaitable(result) else result
    except YaoError:
        raise
    except Exception as exc:
        raise ToolError(
            code=ErrorCode.TOOL_EXECUTION_FAILED, tool=name, cause=exc
        ) from exc

    output = str(output)
    await _fire(request.tool_output_hooks, call, output)
    return output
