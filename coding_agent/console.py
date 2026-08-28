"""终端输出：颜色分级与编码兜底，保证 Windows 终端可用。

视觉约定（与 Claude Code 类似的分层思路）：
- 模型正文：默认色，直接流式输出，是画面的主体；
- 工具调用：青色，展示 agent 正在做什么；
- 工具结果：暗灰色，只给摘要，完整内容回传模型；
- 系统消息：黄色（压缩历史、轮数警告等）；
- 错误：红色；权限确认：品红色。

编码兜底：强制 stdout/stderr 使用 UTF-8（errors=replace），
避免中文在 GBK 代码页的旧式终端上直接崩溃。
"""
from __future__ import annotations

import sys

from colorama import Fore, Style, init as _colorama_init

_setup_done = False


def setup() -> None:
    """进程启动时调用一次：初始化颜色并统一输出编码。"""
    global _setup_done
    if _setup_done:
        return
    _setup_done = True
    _colorama_init()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # 已关闭或不可重配置的流，跳过


def _emit(text: str, color: str, end: str = "\n") -> None:
    sys.stdout.write(f"{color}{text}{Style.RESET_ALL}{end}")
    sys.stdout.flush()


def write_stream(text: str) -> None:
    """流式输出模型正文增量（不换行、不加颜色）。"""
    sys.stdout.write(text)
    sys.stdout.flush()


def print_tool_call(name: str, args_brief: str) -> None:
    _emit(f"[工具] {name}({args_brief})", Fore.CYAN)


def print_tool_result(brief: str) -> None:
    _emit(f"[结果] {brief}", Style.DIM)


def print_tool_error(name: str, message: str) -> None:
    _emit(f"[工具错误] {name}: {message}", Fore.RED)


def print_system(message: str) -> None:
    _emit(f"[系统] {message}", Fore.YELLOW)


def print_error(message: str) -> None:
    _emit(f"[错误] {message}", Fore.RED)


def print_denied(name: str) -> None:
    _emit(f"[已拒绝] 用户拒绝了 {name} 的本次调用", Fore.MAGENTA)


def confirm(prompt: str) -> str:
    """权限确认：返回用户输入的小写串。Ctrl+C 会抛出 KeyboardInterrupt 交给上层。"""
    try:
        return input(f"{Fore.MAGENTA}{prompt}{Style.RESET_ALL}").strip().lower()
    except EOFError:
        return ""
