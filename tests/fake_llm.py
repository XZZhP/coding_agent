"""测试替身：脚本化的 LLM 客户端。

FakeLLM 按脚本顺序吐出预设的 LLMResponse，并记录每次调用的
messages / tools / system 参数，供断言 agent 循环的真实行为。
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from coding_agent.llm import LLMError, LLMResponse, ToolCall


class FakeLLM:
    def __init__(self, script: list[LLMResponse] | None = None):
        self.script = list(script or [])
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, on_text=None, on_reasoning=None,
             temperature=None, max_tokens=None, system=None):
        self.calls.append({
            "messages": deepcopy(messages),
            "tools": tools,
            "system": system,
        })
        if not self.script:
            return LLMResponse(content="（测试脚本已耗尽）")
        resp = self.script.pop(0)
        if on_text and resp.content:
            on_text(resp.content)
        return resp


class RaisingLLM:
    """每次调用都抛出指定异常（测错误路径）。"""

    def __init__(self, exc: Exception):
        self.exc = exc
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        raise self.exc


def resp_text(text: str) -> LLMResponse:
    return LLMResponse(content=text)


def resp_calls(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(content="", tool_calls=list(calls))


def tc(name: str, args: dict | None = None, call_id: str | None = None,
       parse_error: str | None = None) -> ToolCall:
    import json
    return ToolCall(
        id=call_id or f"call_test_{name}",
        name=name,
        raw_arguments=json.dumps(args) if args is not None else "",
        arguments=args,
        parse_error=parse_error,
    )


def chunk(content=None, finish=None, usage=None, reasoning=None,
          tool_calls=None, delta_extra=None):
    """构造一个流式响应 chunk（SimpleNamespace，模拟 openai SDK 结构）。"""
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls or None,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage)


def tool_delta(index, id=None, name=None, args=None):
    fn = SimpleNamespace(name=name, arguments=args)
    return SimpleNamespace(index=index, id=id, function=fn)
