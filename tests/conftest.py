"""
测试公用件：一个**假模型后端**，让全部用例离线运行（不打真实 API）。

模型只在 `client._get_client(...).chat.completions.create(**kwargs)` 这一处被触达，
所以把 `_get_client` 换成返回脚本化的假客户端即可。流式/非流式共用一套脚本：
脚本是一串“轮次”，每轮要么吐文本、要么请求工具调用。
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

# 让 to_request_options() 能解析出 key（不依赖 .env / 真实密钥）。
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")


# ----------------------------------------------------------------- 假对象构造
def _usage(p=1, c=1, t=2):
    return SimpleNamespace(prompt_tokens=p, completion_tokens=c, total_tokens=t)


def _tool_call(call_id, name, arguments, index=0):
    return SimpleNamespace(
        id=call_id,
        type="function",
        index=index,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def text_turn(content, *, finish_reason="stop", usage=(1, 1, 2)):
    """一轮：模型直接给文本（非流式）。"""
    msg = SimpleNamespace(content=content, tool_calls=[], reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)], usage=_usage(*usage))


def tool_turn(name, arguments, *, call_id="call_1", usage=(1, 1, 2)):
    """一轮：模型请求调用工具（非流式）。"""
    tc = _tool_call(call_id, name, arguments)
    msg = SimpleNamespace(content=None, tool_calls=[tc], reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], usage=_usage(*usage))


def _chunk(*, content=None, reasoning=None, tool_pieces=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=tool_pieces or [])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def stream_text(*, answer_chunks=(), reasoning_chunks=()):
    """一轮（流式）：先吐 reasoning 增量，再吐 answer 增量，无工具。"""

    async def gen():
        for r in reasoning_chunks:
            yield _chunk(reasoning=r)
        for a in answer_chunks:
            yield _chunk(content=a)

    return gen()


def stream_tool(name, arguments, *, call_id="call_1"):
    """一轮（流式）：分片吐一个工具调用。"""

    async def gen():
        # name 在首片，arguments 拆两段。
        mid = len(arguments) // 2
        yield _chunk(tool_pieces=[_tool_call(call_id, name, arguments[:mid], index=0)])
        yield _chunk(tool_pieces=[_tool_call(None, None, arguments[mid:], index=0)])

    return gen()


# ----------------------------------------------------------------- 假客户端
class _FakeCompletions:
    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    async def create(self, **kwargs):
        item = self._script[self.calls]
        self.calls += 1
        return item  # 非流式=响应对象；流式=async generator（两者 create 都直接返回）


class _FakeClient:
    def __init__(self, script):
        self.chat = SimpleNamespace(completions=_FakeCompletions(script))


@pytest.fixture
def fake_model(monkeypatch):
    """
    返回一个 `install(script)` 函数：把 client._get_client 换成吐该脚本的假客户端，
    并返回假客户端（便于断言调用次数）。
    """
    import yaoagent.client as client

    holder = {}

    def install(script):
        fake = _FakeClient(script)
        holder["client"] = fake
        monkeypatch.setattr(client, "_get_client", lambda options: fake)
        return fake

    return install


@pytest.fixture
def run():
    """把异步协程跑到完成（替代 pytest-asyncio）。"""
    return lambda coro: asyncio.run(coro)
