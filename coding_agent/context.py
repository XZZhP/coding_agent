"""上下文管理：对话历史、token 用量跟踪与自动压缩。

设计要点（面试可展开）：
1. 历史即 messages 列表，与 API 消息格式一一对应，不引入额外抽象层——
   tool 消息、assistant 消息的交替约束由压缩逻辑保证。
2. token 跟踪以 API 每轮返回的真实 usage 为基准（最精确）；
   字符估算只是 usage 缺失时的兜底。
3. 压缩时机：预测"下一次请求"的 token 数超过安全水位
   （窗口 - 回复预留 - 余量）时触发。
4. 压缩 = 调用模型把较早对话总结成一条摘要消息，保留最近若干条原文；
   摘要调用失败时降级为直接丢弃早期消息（可用性优先）。
5. 切割点永远落在"回合边界"上：tool 消息必须紧跟它的 assistant 消息，
   切割点会自动后移，保证发给 API 的消息序列永远合法。
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMClient, ToolCall

SUMMARY_PROMPT = (
    "你是会话历史压缩助手。请用简洁的中文总结下面这段 agent 与用户的对话，"
    "保留：任务目标、已完成的关键步骤、重要文件的路径与内容要点、"
    "已做出的技术决策、尚待完成的事项、以及任何重要的错误与修复。"
    "控制在 400 字以内，只输出总结本身。"
)


def _msg_text(msg: dict) -> str:
    """消息的文本化表示，用于估算 token 与压缩。"""
    try:
        return json.dumps(msg, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(msg)


def estimate_tokens(msg: dict) -> int:
    """兜底估算：中文约 1.5 字符/token、英文约 4 字符/token，取保守值 3。"""
    return max(1, len(_msg_text(msg)) // 3)


def adjust_cut(messages: list[dict], cut: int) -> int:
    """把切割点后移到合法边界：被切掉的第一条消息不能是 tool 消息。"""
    while cut < len(messages) and messages[cut].get("role") == "tool":
        cut += 1
    return cut


class Conversation:
    """一条对话的完整状态：历史消息 + token 计量。

    usage 追踪逻辑：API 每次返回的 prompt_tokens 是"该次请求的完整输入"，
    因此下一次请求的输入 ≈ 上次 prompt + 上次 completion + 新增消息。
    """

    def __init__(self, system_prompt: str, context_window: int,
                 keep_last_messages: int, response_max_tokens: int):
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.context_window = context_window
        self.keep_last = keep_last_messages
        self.response_max_tokens = response_max_tokens
        self.last_prompt_tokens: int | None = None
        self.last_completion_tokens: int | None = None

    # ------------------------------------------------ 基础操作

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, content: str, tool_calls: list["ToolCall"] | None) -> None:
        msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": c.raw_arguments}}
                for c in tool_calls
            ]
        self.messages.append(msg)

    def add_tool_result(self, call_id: str, result: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": call_id,
                              "content": result})

    def update_usage(self, prompt: int | None, completion: int | None) -> None:
        if prompt is not None:
            self.last_prompt_tokens = prompt
        if completion is not None:
            self.last_completion_tokens = completion

    # ------------------------------------------------ 预测与压缩

    def _reset_tracking(self) -> None:
        """历史被改写后，旧的 usage 数据不再成立，等待下一轮真实数据重建。"""
        self.last_prompt_tokens = None
        self.last_completion_tokens = None

    def predict_next_prompt(self, pending_chars: int = 0) -> int:
        """预测下一次请求的输入 token 数（含即将追加的工具结果等）。"""
        if self.last_prompt_tokens is not None and self.last_completion_tokens is not None:
            return (self.last_prompt_tokens + self.last_completion_tokens
                    + pending_chars // 3)
        return sum(estimate_tokens(m) for m in self.messages) + pending_chars // 3

    def needs_compression(self, pending_chars: int = 0) -> bool:
        """下一次请求是否会突破安全水位。"""
        reserve = self.response_max_tokens + 1024   # 回复预留 + 余量
        return self.predict_next_prompt(pending_chars) > self.context_window - reserve

    def compress(self, llm: "LLMClient") -> None:
        """压缩历史：旧消息 → 摘要消息（失败则直接丢弃），保留最近 keep_last 条。"""
        if len(self.messages) <= self.keep_last + 2:
            return  # 太短，没有压缩价值
        cut = adjust_cut(self.messages, len(self.messages) - self.keep_last)
        if cut <= 1:
            return
        old, recent = self.messages[1:cut], self.messages[cut:]
        summary = None
        try:
            resp = llm.chat(
                messages=[{"role": "user", "content": _msg_text(old)[:20000]}],
                system=SUMMARY_PROMPT, temperature=0.0, max_tokens=1024,
            )
            summary = resp.content.strip()
        except Exception:
            summary = None  # 摘要失败：降级为直接丢弃旧消息
        if summary:
            head = [self.messages[0],
                    {"role": "user",
                     "content": f"[会话早期内容摘要]\n{summary}\n\n——以下是摘要之后的对话——"}]
        else:
            head = [self.messages[0],
                    {"role": "user", "content": "[更早的对话因超出上下文窗口已省略]"}]
        self.messages = head + recent
        self._reset_tracking()

    def emergency_truncate(self) -> bool:
        """上下文溢出兜底：丢弃最早的消息，保留最近 keep_last 条。"""
        cut = adjust_cut(self.messages, max(1, len(self.messages) - self.keep_last))
        if cut <= 1:
            return False
        self.messages = [self.messages[0]] + self.messages[cut:]
        self._reset_tracking()
        return True

    # ------------------------------------------------ 会话存档

    def to_session(self, extra: dict | None = None) -> dict:
        return {
            "version": 1,
            "messages": self.messages,
            **(extra or {}),
        }

    @staticmethod
    def from_session(data: dict, system_prompt: str, context_window: int,
                     keep_last: int, response_max_tokens: int) -> "Conversation":
        conv = Conversation(system_prompt, context_window, keep_last,
                            response_max_tokens)
        msgs = data.get("messages") or []
        if msgs and msgs[0].get("role") == "system":
            msgs = msgs[1:]  # 用当前 system 提示替换存档中的旧版本
        conv.messages = [{"role": "system", "content": system_prompt}] + msgs
        return conv
