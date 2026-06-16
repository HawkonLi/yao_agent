"""Prompt 输入类型、activation_key 指纹、run_scope 关联。"""

from __future__ import annotations

from yaoagent import Prompt, Profile, current_run_id, run_scope


class _Instr:
    """activation_key 只看 type(instructions).__name__ + 生成参数。"""


def test_prompt_is_str_with_extras():
    p = Prompt("分析", attachments=[1, 2], metadata={"lang": "zh"})
    assert isinstance(p, str)
    assert p == "分析"
    assert p.attachments == (1, 2)
    assert p.metadata == {"lang": "zh"}
    # 纯文本默认零负担。
    bare = Prompt("hi")
    assert bare == "hi" and bare.attachments == () and bare.metadata == {}


def test_activation_key_distinguishes_params():
    a = Profile(instructions=_Instr()).temperature(0.2)
    b = Profile(instructions=_Instr()).temperature(0.9)
    same = Profile(instructions=_Instr()).temperature(0.2)
    assert a.activation_key() != b.activation_key()   # 同指令类、不同温度 → 不同身份
    assert a.activation_key() == same.activation_key()


def test_activation_key_excludes_hooks():
    a = Profile(instructions=_Instr()).temperature(0.2)
    b = Profile(instructions=_Instr()).temperature(0.2).on_prompt(lambda p: None)
    assert a.activation_key() == b.activation_key()   # 钩子不进指纹（每次新闭包）


def test_run_scope_correlation():
    assert current_run_id() is None
    with run_scope() as rid:
        assert current_run_id() == rid
        with run_scope() as inner:      # 嵌套沿用外层 id
            assert inner == rid
            assert current_run_id() == rid
    assert current_run_id() is None
