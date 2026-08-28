"""上下文管理测试：token 预测、压缩、切割边界、会话存档。"""
from __future__ import annotations

from coding_agent.context import Conversation, adjust_cut, estimate_tokens

from fake_llm import FakeLLM, resp_text


def make_conv(**kw) -> Conversation:
    defaults = dict(system_prompt="sys", context_window=65536,
                    keep_last_messages=6, response_max_tokens=8192)
    defaults.update(kw)
    return Conversation(**defaults)


def _fill(conv: Conversation, n_pairs: int) -> None:
    for i in range(n_pairs):
        conv.add_user(f"任务{i}")
        conv.add_assistant(f"答复{i}", None)


# ---------------------------------------------------------------- 切割边界

def test_adjust_cut_skips_tool_messages():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1", "tool_calls": []},
        {"role": "tool", "tool_call_id": "1", "content": "r1"},
        {"role": "user", "content": "u2"},
    ]
    # 切在第 3 条（tool 消息）上：必须后移到第 4 条（user）
    assert adjust_cut(msgs, 3) == 4
    # 切在合法边界（user 消息）上：不动
    assert adjust_cut(msgs, 4) == 4
    assert adjust_cut(msgs, 1) == 1


# ---------------------------------------------------------------- 预测

def test_predict_from_usage():
    conv = make_conv()
    _fill(conv, 2)
    conv.update_usage(1000, 200)
    # 下次请求 ≈ 上次输入 + 上次输出 + 新增内容
    assert conv.predict_next_prompt(pending_chars=300) == 1000 + 200 + 100


def test_predict_fallback_when_no_usage():
    conv = make_conv()
    conv.add_user("hello world")
    assert conv.predict_next_prompt() > 0
    assert conv.predict_next_prompt() == sum(
        estimate_tokens(m) for m in conv.messages)


def test_needs_compression_trigger():
    # 窗口 10000，回复预留 100 → 安全水位 = 10000 - 100 - 1024 = 8876
    conv = make_conv(context_window=10000, response_max_tokens=100)
    conv.update_usage(1000, 200)  # 预测 1200 < 8876 → 不触发
    assert conv.needs_compression() is False
    conv.update_usage(8700, 200)  # 预测 8900 > 8876 → 触发
    assert conv.needs_compression() is True


# ---------------------------------------------------------------- 压缩

def test_compress_with_summary(tmp_path=None):
    conv = make_conv(keep_last_messages=2, context_window=1000,
                     response_max_tokens=100)
    _fill(conv, 5)
    conv.update_usage(900, 10)  # 触发压缩水位
    fake = FakeLLM([resp_text("摘要：完成了一部分任务。")])
    conv.compress(fake)
    # 压缩后：system + 摘要消息 + 最近 2 条
    assert len(conv.messages) == 4
    assert conv.messages[1]["role"] == "user"
    assert "摘要：完成了一部分任务" in conv.messages[1]["content"]
    # 摘要调用不应带工具，且是独立的一次性 system 提示
    assert fake.calls[0]["system"] is not None
    assert fake.calls[0]["tools"] is None
    # 压缩后旧 usage 失效，回归估算
    assert conv.last_prompt_tokens is None


def test_compress_fallback_when_summary_fails():
    from coding_agent.llm import LLMError
    conv = make_conv(keep_last_messages=2)
    _fill(conv, 5)

    class Boom:
        def chat(self, *a, **kw):
            raise LLMError("摘要服务不可用")
    conv.compress(Boom())
    assert len(conv.messages) == 4
    assert "已省略" in conv.messages[1]["content"]


def test_compress_keeps_tool_boundary():
    """切割点若落在 tool 消息上必须后移，保证历史格式合法。"""
    conv = make_conv(keep_last_messages=2)
    conv.add_user("任务")
    conv.messages += [
        # 一次带工具调用的回合：assistant(tool_calls) + tool
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "type": "function",
             "function": {"name": "list_dir", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "x", "content": "结果"},
        {"role": "assistant", "content": "最终答复"},
    ]
    # 5 条消息，keep_last=2 → 切点 3（tool 消息），必须后移到 4（assistant）
    cut = adjust_cut(conv.messages, len(conv.messages) - 2)
    assert cut == 4
    conv.compress(FakeLLM([resp_text("s")]))
    # 压缩后：system + 摘要 + assistant(最终答复)，没有孤立的 tool 消息
    assert [m["role"] for m in conv.messages] == ["system", "user", "assistant"]


def test_too_short_no_compress():
    conv = make_conv(keep_last_messages=2)
    _fill(conv, 1)
    before = list(conv.messages)
    conv.compress(FakeLLM([]))
    assert conv.messages == before


# ---------------------------------------------------------------- 紧急裁剪

def test_emergency_truncate():
    conv = make_conv(keep_last_messages=4)
    _fill(conv, 6)
    conv.update_usage(100, 10)
    assert conv.emergency_truncate() is True
    assert conv.messages[0]["role"] == "system"
    assert len(conv.messages) == 1 + 4
    assert conv.messages[-1]["content"] == "答复5"
    assert conv.last_prompt_tokens is None


# ---------------------------------------------------------------- 会话存档

def test_session_roundtrip():
    conv = make_conv()
    _fill(conv, 2)
    data = conv.to_session(extra={"model": "m"})
    conv2 = Conversation.from_session(data, "新系统提示", 65536, 6, 8192)
    assert conv2.messages[0] == {"role": "system", "content": "新系统提示"}
    assert len(conv2.messages) == len(conv.messages)
    assert conv2.messages[1]["content"] == "任务0"


def test_session_ignores_stale_system():
    conv = make_conv()
    _fill(conv, 1)
    data = conv.to_session()
    conv2 = Conversation.from_session(data, "更新的提示", 65536, 6, 8192)
    assert sum(1 for m in conv2.messages if m["role"] == "system") == 1
    assert conv2.messages[0]["content"] == "更新的提示"
