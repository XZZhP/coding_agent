"""Agent 核心循环 —— 项目的心脏。

每轮循环：
  1. 检查上下文水位，必要时先压缩历史；
  2. 把完整消息历史（system + 全部工具结果）发给模型，流式打印正文；
  3. 模型未请求工具 → 任务完成，循环自然终止；
  4. 模型请求工具 → 逐一经过权限确认后在本机执行，结果（或错误信息）
     作为 tool 消息追加进历史，进入下一轮，让模型基于真实反馈决策。

终止条件（多重保险，面试必考）：
  a. 模型不再请求工具（自然终止，最常见）；
  b. 达到 max_turns 上限（防止无限循环烧 token）；
  c. 连续 3 轮发出完全相同的工具调用（判定卡死，注入提示强制改变策略）；
  d. 用户 Ctrl+C 中断；
  e. 连续 API 错误超过重试上限（llm 层抛出）。

错误处理原则：错误是信息，不是终点。
  - 工具执行异常 → 错误文本回传模型，模型自行修正；
  - 上下文溢出 → 紧急裁剪早期消息后重试一次；
  - 用户拒绝调用 → "用户已拒绝"回传模型，要求其调整方案。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import console
from .config import Config
from .context import Conversation
from .llm import LLMAuthError, LLMClient, LLMContextOverflow, LLMError, ToolCall
from .tools import (ExecutionContext, Tool, ToolError, build_registry,
                    tool_schemas)

# 连续相同工具调用达到该次数时判定"卡死"，注入提示强制干预
REPEAT_LIMIT = 3


def build_system_prompt(config: Config) -> str:
    """系统提示：角色定位 + 行为准则 + 环境信息。

    环境信息动态注入（平台、shell、工作目录），让模型第一次尝试
    就写出可执行的命令，减少试错轮次。
    """
    import os
    shell = " ".join(config.default_shell())
    return (
        "你是 coding-agent，一个运行在用户本机的编程智能体。"
        "你通过调用本地工具完成用户交给你的编程任务：读写文件、搜索代码、执行命令。\n"
        f"当前工作目录：{config.workdir}\n"
        f"本机环境：{os.name}（{'Windows' if os.name == 'nt' else 'POSIX'}），"
        f"命令 shell：{shell}；执行命令时请使用该 shell 的语法。\n\n"
        "行为准则：\n"
        "1. 先理解再动手：修改前先 read_file / list_dir / search 了解现状；\n"
        "2. 小步推进：每次只做一件明确的事；小改动用 edit_file，整体重写才用 write_file；\n"
        "3. 命令失败时先阅读输出分析原因，再调整方案，不要盲目重复同一命令；\n"
        "4. 每个任务结束后，用简洁的总结告知用户：改了什么、为什么、如何验证；\n"
        "5. 与用户的交流使用中文。"
    )


class PermissionManager:
    """分级权限策略：read 自动放行；write/execute 默认确认；--yes 全自动。

    确认交互抽出为可注入的 confirm 回调，单元测试时可替换为脚本化应答。
    """

    def __init__(self, auto_yes: bool = False,
                 confirm: Callable[[str], str] | None = None):
        self.auto_yes = auto_yes
        self._confirm = confirm or console.confirm

    def ask(self, tool: Tool, args: dict) -> bool:
        if self.auto_yes or tool.permission == "read":
            return True
        answer = self._confirm(
            f"[确认] 允许执行 {tool.name}({tool.summarize(args)}) 吗？"
            "[y=允许 / n=拒绝 / a=之后全部允许] "
        )
        if answer == "a":
            self.auto_yes = True
            return True
        return answer == "y"


@dataclass
class RunResult:
    """一次任务运行的结局。"""

    status: str          # "finished" | "max_turns" | "interrupted" | "error"
    message: str = ""    # 补充说明（如中断原因）


class Agent:
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMClient(config)
        self.registry: dict[str, Tool] = build_registry()
        self.ctx = ExecutionContext(
            workdir=config.workdir,
            max_output=config.max_tool_output,
            command_timeout=config.command_timeout,
            shell=config.default_shell(),
        )
        self.conversation = Conversation(
            system_prompt=build_system_prompt(config),
            context_window=config.context_window,
            keep_last_messages=config.keep_last_messages,
            response_max_tokens=config.response_max_tokens,
        )
        self.permission = PermissionManager(auto_yes=config.auto_yes)
        self._last_call_signature: tuple | None = None
        self._repeat_count = 0

    # ------------------------------------------------ 主循环

    def run(self, task: str | None = None) -> RunResult:
        if task:
            self.conversation.add_user(task)
        turn = 0
        while turn < self.config.max_turns:
            turn += 1
            if turn > 1:
                console.print_system(f"—— 第 {turn}/{self.config.max_turns} 轮 ——")

            try:
                if self.conversation.needs_compression():
                    console.print_system("上下文接近窗口上限，正在压缩历史…")
                    self.conversation.compress(self.llm)
                    console.print_system("历史已压缩")

                resp = self._chat_with_overflow_retry()
            except KeyboardInterrupt:
                return RunResult("interrupted", "用户中断（Ctrl+C）")
            except LLMAuthError as e:
                raise  # 密钥问题：交由 CLI 顶层给出指引后退出
            except LLMError as e:
                console.print_error(str(e))
                return RunResult("error", str(e))

            self.conversation.update_usage(resp.prompt_tokens,
                                           resp.completion_tokens)
            if resp.content.strip():
                console.write_stream("\n")
            self.conversation.add_assistant(resp.content, resp.tool_calls or None)

            if not resp.tool_calls:
                return RunResult("finished")

            if not self._execute_tool_calls(resp.tool_calls):
                return RunResult("interrupted", "执行工具期间被用户中断")

        console.print_error(f"达到最大轮数上限（{self.config.max_turns}），已强制停止。")
        return RunResult("max_turns", "任务未在轮数上限内完成")

    def _chat_with_overflow_retry(self):
        """发一次消息；若上下文溢出，紧急裁剪早期历史后重试一次。"""
        try:
            return self.llm.chat(
                messages=self.conversation.messages,
                tools=tool_schemas(self.registry),
                on_text=console.write_stream,
                on_reasoning=lambda t: console.write_stream(t),
            )
        except LLMContextOverflow as e:
            console.print_system(f"上下文溢出（{e}），裁剪早期历史后重试…")
            if not self.conversation.emergency_truncate():
                raise
            return self.llm.chat(
                messages=self.conversation.messages,
                tools=tool_schemas(self.registry),
                on_text=console.write_stream,
            )

    # ------------------------------------------------ 工具执行

    def _execute_tool_calls(self, calls: list[ToolCall]) -> bool:
        """逐一执行本轮的工具调用；返回 False 表示被用户中断。"""
        signature = tuple((c.name, c.raw_arguments) for c in calls)
        if signature == self._last_call_signature:
            self._repeat_count += 1
        else:
            self._last_call_signature = signature
            self._repeat_count = 1

        if self._repeat_count == REPEAT_LIMIT:
            console.print_system(
                f"检测到连续 {REPEAT_LIMIT} 轮发出完全相同的工具调用，疑似卡死，已提示模型调整策略。"
            )
            self.conversation.add_user(
                "你连续多轮发出了完全相同的工具调用且没有取得任何进展。"
                "请停下来重新分析：工具参数是否正确？输出是否提示了失败原因？"
                "换一种思路或先查看相关文件，不要重复刚才的调用。"
            )
            self._repeat_count = 0

        for call in calls:
            try:
                self._execute_one(call)
            except KeyboardInterrupt:
                console.print_system("工具执行被用户中断")
                return False
        return True

    def _execute_one(self, call: ToolCall) -> None:
        tool = self.registry.get(call.name)
        args = call.arguments or {}

        if call.parse_error:
            self.conversation.add_tool_result(
                call.id, f"工具调用无效：{call.parse_error}。请重新生成符合 JSON Schema 的参数。"
            )
            console.print_tool_error(call.name, call.parse_error)
            return
        if tool is None:
            msg = (f"未知工具 {call.name}。当前可用工具："
                   + "、".join(sorted(self.registry)))
            self.conversation.add_tool_result(call.id, msg)
            console.print_tool_error(call.name, "未知工具")
            return

        console.print_tool_call(call.name, tool.summarize(args))

        if not self.permission.ask(tool, args):
            result = "用户拒绝了本次工具调用。请调整你的方案，换用其它方式完成任务。"
            console.print_denied(call.name)
        else:
            try:
                result = tool.func(args, self.ctx)
                console.print_tool_result(self._brief(result))
            except ToolError as e:
                result = f"工具执行失败：{e}"
                console.print_tool_error(call.name, str(e))

        self.conversation.add_tool_result(call.id, result)

    @staticmethod
    def _brief(result: str) -> str:
        """控制台只显示结果的第一行摘要，完整内容回传模型。"""
        first = result.strip().splitlines()[0] if result.strip() else "(空输出)"
        return first[:160] + ("…" if len(first) > 160 else "")
