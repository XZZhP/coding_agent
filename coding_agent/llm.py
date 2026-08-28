"""LLM 客户端层：封装 DeepSeek（OpenAI 兼容）API 的流式调用。

职责边界（面试可展开）：本层只负责"发消息、收消息"——
构造请求、流式解析增量、累积工具调用参数、统计 token 用量、
对瞬时错误重试。不含任何 agent 决策逻辑（决策在 agent.py）。

依赖 openai 官方 SDK：规则允许的"模型厂商 API 客户端库"，
我们只用它的消息收发能力，循环、工具、终止条件全部自研。

解析细节：
- 流式响应中工具调用的参数是分片到达的（delta 增量），需要按 index
  累积拼接，这是"模型输出解析"的核心工作之一；
- 工具参数是 JSON 字符串，解析失败时不崩溃，而是把原始串与错误
  一起标记，由 agent 循环回传给模型让它重新生成；
- token 用量优先取 API 返回的真实 usage（stream_options.include_usage），
  为上下文压缩提供精确依据。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import openai
from openai import OpenAI

from .config import Config


# ---------------------------------------------------------------- 异常体系

class LLMError(Exception):
    """所有 LLM 层错误的基类。"""


class LLMAuthError(LLMError):
    """密钥无效/欠费。不可重试，直接给出人工指引。"""


class LLMContextOverflow(LLMError):
    """请求超出模型上下文窗口。由 agent 循环做紧急裁剪后重试。"""


class LLMRequestError(LLMError):
    """其它可诊断的请求错误（含重试耗尽）。"""


# 瞬时错误：网络抖动、限流、服务端 5xx —— 重试才有意义
_RETRYABLE = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


@dataclass
class ToolCall:
    """模型请求的一次工具调用。"""

    id: str
    name: str
    raw_arguments: str                  # 模型给出的原始 JSON 字符串
    arguments: dict | None = None       # 解析成功时为参数 dict
    parse_error: str | None = None      # 解析失败原因（回传给模型）


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int | None = None       # API 真实用量，可能为 None
    completion_tokens: int | None = None


# ---------------------------------------------------------------- 流解析

def accumulate_stream(
    chunks: Iterable,
    on_text: Callable[[str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
) -> LLMResponse:
    """把流式 chunk 序列累积为一次完整的 LLMResponse。

    独立成纯函数，便于用伪造 chunk 做单元测试（不依赖真实网络）。
    """
    content_parts: list[str] = []
    calls: dict[int, dict] = {}
    finish_reason = None
    usage_prompt = usage_completion = None

    for chunk in chunks:
        if chunk.usage is not None:
            usage_prompt = chunk.usage.prompt_tokens
            usage_completion = chunk.usage.completion_tokens
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        choice = choices[0]
        if getattr(choice, "finish_reason", None):
            finish_reason = choice.finish_reason
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue

        content = getattr(delta, "content", None)
        if content:
            content_parts.append(content)
            if on_text:
                on_text(content)

        # 推理模型会输出思维链增量（reasoning_content 非 OpenAI 标准字段）
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning and on_reasoning:
            on_reasoning(reasoning)

        for tc in getattr(delta, "tool_calls", None) or []:
            slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["name"] = fn.name
                if getattr(fn, "arguments", None):
                    slot["args"] += fn.arguments

    tool_calls: list[ToolCall] = []
    for index in sorted(calls):
        slot = calls[index]
        call = ToolCall(id=slot["id"] or f"call_{index}",
                        name=slot["name"], raw_arguments=slot["args"])
        if slot["args"]:
            try:
                call.arguments = json.loads(slot["args"])
            except json.JSONDecodeError as e:
                call.parse_error = f"工具参数不是合法 JSON：{e}"
        else:
            call.parse_error = "工具参数为空"
        tool_calls.append(call)

    return LLMResponse(
        content="".join(content_parts),
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        prompt_tokens=usage_prompt,
        completion_tokens=usage_completion,
    )


# ---------------------------------------------------------------- 客户端

class LLMClient:
    """DeepSeek API 客户端：流式聊天 + 重试。"""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout,
            max_retries=0,  # 重试策略自己实现：可控、可观测、可解释
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,   # 一次性系统提示（如摘要任务），不进入历史
    ) -> LLMResponse:
        cfg = self.config
        last_error: Exception | None = None
        for attempt in range(cfg.max_retries + 1):
            try:
                return self._request_once(
                    messages, tools, on_text, on_reasoning,
                    temperature, max_tokens, system,
                )
            except LLMContextOverflow:
                raise  # 不可重试，交给 agent 裁剪历史
            except LLMAuthError:
                raise  # 不可重试，交给 CLI 给出人工指引
            except _RETRYABLE as e:
                last_error = e
                if attempt < cfg.max_retries:
                    wait = 2 ** attempt
                    time.sleep(wait)
            except openai.BadRequestError as e:
                raise LLMRequestError(f"请求被拒绝：{_err_text(e)}") from e
            except openai.APIStatusError as e:
                raise LLMRequestError(
                    f"API 返回错误（状态码 {e.status_code}）：{_err_text(e)}"
                ) from e
        raise LLMRequestError(
            f"连续 {cfg.max_retries + 1} 次请求失败（网络/限流/服务端错误），"
            f"最后错误：{_err_text(last_error)}"
        )

    def _request_once(self, messages, tools, on_text, on_reasoning,
                      temperature, max_tokens, system) -> LLMResponse:
        cfg = self.config
        kwargs: dict = dict(
            model=cfg.model,
            messages=messages,
            stream=True,
            temperature=cfg.temperature if temperature is None else temperature,
            max_tokens=cfg.response_max_tokens if max_tokens is None else max_tokens,
            stream_options={"include_usage": True},  # 最后一帧携带真实 token 用量
        )
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["messages"] = [{"role": "system", "content": system}] + messages

        try:
            stream = self.client.chat.completions.create(**kwargs)
            return accumulate_stream(stream, on_text, on_reasoning)
        except openai.AuthenticationError as e:
            raise LLMAuthError(
                "API key 无效或账户不可用。请检查 DEEPSEEK_API_KEY / .env 中的密钥，"
                "并确认账户余额充足。"
            ) from e
        except openai.BadRequestError as e:
            text = _err_text(e)
            if any(k in text.lower() for k in
                   ("context", "token", "maximum context", "输入过长")):
                raise LLMContextOverflow(f"上下文超出窗口：{text}") from e
            raise  # 交给外层统一转 LLMRequestError


def _err_text(e: Exception) -> str:
    """提取 API 错误消息正文（响应体 JSON 或文本），截断到 500 字符。"""
    body = getattr(e, "response", None)
    if body is not None:
        try:
            data = body.json()
            msg = str(data.get("error", data)) if isinstance(data, dict) else str(data)
        except Exception:
            msg = str(getattr(body, "text", "") or "")
    else:
        msg = str(e)
    msg = msg.strip() or e.__class__.__name__
    return msg[:500]
