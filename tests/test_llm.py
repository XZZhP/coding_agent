"""LLM 层测试：流式解析（纯函数）与异常翻译。"""
from __future__ import annotations

import httpx2
import openai
import pytest

from coding_agent.config import Config
from coding_agent.llm import (LLMAuthError, LLMClient, LLMContextOverflow,
                              accumulate_stream)

from fake_llm import chunk, tool_delta

# 构造 openai 3.x 异常所需的 httpx 对象
_req = httpx2.Request("GET", "https://api.deepseek.com")


def _resp(status: int, message: str) -> httpx2.Response:
    return httpx2.Response(status, request=_req,
                           json={"error": {"message": message}})


def test_accumulate_content_and_finish():
    chunks = [
        chunk(content="你好"),
        chunk(content="，世界"),
        chunk(finish="stop"),
    ]
    r = accumulate_stream(chunks)
    assert r.content == "你好，世界"
    assert r.finish_reason == "stop"
    assert r.tool_calls == []
    assert r.prompt_tokens is None


def test_accumulate_stream_calls_callbacks():
    seen: list[str] = []
    chunks = [chunk(content="a"), chunk(content="b", reasoning="思考…")]
    r = accumulate_stream(chunks, on_text=seen.append,
                           on_reasoning=seen.append)
    assert seen == ["a", "b", "思考…"]


def test_accumulate_tool_call_fragments():
    """工具参数分片到达，须按 index 累积拼接。"""
    chunks = [
        chunk(tool_calls=[tool_delta(0, id="c1", name="read_file",
                                     args='{"path":')]),
        chunk(tool_calls=[tool_delta(0, args='"a.py"')]),
        chunk(tool_calls=[tool_delta(0, args="}")]),
        chunk(finish="tool_calls"),
    ]
    r = accumulate_stream(chunks)
    assert len(r.tool_calls) == 1
    call = r.tool_calls[0]
    assert call.id == "c1"
    assert call.name == "read_file"
    assert call.arguments == {"path": "a.py"}
    assert call.parse_error is None


def test_accumulate_multiple_tool_calls_interleaved():
    """多个工具调用的分片可能交错到达，按 index 分开累积。"""
    chunks = [
        chunk(tool_calls=[tool_delta(0, id="a", name="list_dir", args="{}"),
                          tool_delta(1, id="b", name="glob", args='{"p')]),
        chunk(tool_calls=[tool_delta(1, args='attern": "*.py"}')]),
    ]
    r = accumulate_stream(chunks)
    assert [c.name for c in r.tool_calls] == ["list_dir", "glob"]
    assert r.tool_calls[1].arguments == {"pattern": "*.py"}


def test_accumulate_usage_from_final_chunk():
    usage = type("U", (), {"prompt_tokens": 123, "completion_tokens": 45})()
    chunks = [chunk(content="x"), chunk(usage=usage)]
    r = accumulate_stream(chunks)
    assert r.prompt_tokens == 123
    assert r.completion_tokens == 45


def test_accumulate_bad_json_args():
    chunks = [chunk(tool_calls=[tool_delta(0, id="c", name="write_file",
                                           args="{不是JSON")])]
    call = accumulate_stream(chunks).tool_calls[0]
    assert call.arguments is None
    assert "不是合法 JSON" in (call.parse_error or "")


def test_accumulate_empty_args():
    chunks = [chunk(tool_calls=[tool_delta(0, id="c", name="run_command")])]
    call = accumulate_stream(chunks).tool_calls[0]
    assert "参数为空" in (call.parse_error or "")


# ---------------------------------------------------------------- 异常翻译

def _client(**kw) -> LLMClient:
    cfg = {k: v for k, v in {"api_key": "sk-test"}.items()}
    cfg.update(kw)
    return LLMClient(Config(**cfg))


def _mock_create(client, side_effect):
    from unittest.mock import patch
    return patch.object(client.client.chat.completions, "create",
                        side_effect=side_effect)


def test_auth_error_translation():
    """AuthenticationError 必须被翻译成带人工指引的 LLMAuthError。"""
    client = _client()
    with _mock_create(client, openai.AuthenticationError(
            "bad key", response=_resp(401, "invalid api key"), body=None)):
        with pytest.raises(LLMAuthError) as ei:
            client.chat(messages=[{"role": "user", "content": "hi"}])
    assert "DEEPSEEK_API_KEY" in str(ei.value)


def test_context_overflow_detection():
    client = _client()
    with _mock_create(client, openai.BadRequestError(
            "maximum context length exceeded",
            response=_resp(400, "maximum context length exceeded"), body=None)):
        with pytest.raises(LLMContextOverflow):
            client.chat(messages=[{"role": "user", "content": "hi"}])


def test_retry_on_transient_then_success():
    """瞬时错误应重试，且最终成功时不抛错。"""
    from unittest.mock import patch
    client = _client(max_retries=2)
    responses = [
        openai.APIConnectionError(message="conn reset", request=_req),
        openai.RateLimitError("limited", response=_resp(429, "slow down"),
                              body=None),
        [chunk(content="ok")],
    ]
    with _mock_create(client, responses) as create, \
            patch("coding_agent.llm.time.sleep"):  # 加速测试，不真睡
        r = client.chat(messages=[{"role": "user", "content": "hi"}])
    assert r.content == "ok"
    assert create.call_count == 3


def test_retries_exhausted_raises():
    client = _client(max_retries=1)
    from unittest.mock import patch
    from coding_agent.llm import LLMRequestError
    with _mock_create(client, openai.APIConnectionError(
            message="down", request=_req)), \
            patch("coding_agent.llm.time.sleep"):
        with pytest.raises(LLMRequestError):
            client.chat(messages=[{"role": "user", "content": "hi"}])
