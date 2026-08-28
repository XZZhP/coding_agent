"""命令行入口：REPL 交互模式 + 单任务模式（--task）。

两种模式共用同一个 Agent 核心循环：
- REPL：多轮对话，历史自然累积，支持 /save 存档、/reset 重开等命令；
- --task：执行一次任务后退出，适合脚本化与视频演示。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from . import __version__, console
from .agent import Agent
from .config import Config
from .context import Conversation
from .llm import LLMAuthError

HELP_TEXT = """内置命令：
  /help        显示本帮助
  /save [路径] 保存当前会话（默认 sessions/session-<时间>.json，可用 --session 恢复）
  /reset       清空对话历史，重新开始
  /model 名    切换模型（如 /model deepseek-v4-pro）
  /yes         之后不再确认工具调用（等同启动参数 -y）
  /exit        退出
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coding-agent",
        description="自研编程智能体：基于 DeepSeek 大模型的本地 coding agent",
    )
    p.add_argument("-t", "--task", metavar="文本",
                   help="单任务模式：执行一次任务后退出")
    p.add_argument("-y", "--yes", action="store_true",
                   help="跳过所有工具调用确认（演示/脚本化场景）")
    p.add_argument("--model", metavar="名",
                   help="模型名（默认取环境变量 DEEPSEEK_MODEL，再默认 deepseek-v4-flash）")
    p.add_argument("-w", "--workdir", metavar="目录", default=None,
                   help="工作目录（agent 读写文件与执行命令的根目录，默认当前目录）")
    p.add_argument("--max-turns", type=int, default=None,
                   help="单次任务最大循环轮数（默认 30）")
    p.add_argument("--temperature", type=float, default=None,
                   help="采样温度（默认 0.3）")
    p.add_argument("--session", metavar="文件",
                   help="从 /save 保存的会话存档恢复继续")
    p.add_argument("--version", action="version",
                   version=f"coding-agent {__version__}")
    return p


def _load_session(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"[错误] 会话存档不存在：{path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[错误] 会话存档损坏（{e}）：{path}")


def _save_session(agent: Agent, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = agent.conversation.to_session(extra={
        "model": agent.config.model,
        "workdir": str(agent.config.workdir),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    })
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def _print_status(result) -> None:
    if result.status != "finished":
        console.print_system(
            f"本次任务结束状态：{result.status}"
            + (f"（{result.message}）" if result.message else "")
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cli_overrides = {k: v for k, v in {
        "model": args.model,
        "workdir": Path(args.workdir) if args.workdir else None,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "auto_yes": True if args.yes else None,
    }.items() if v is not None}

    config = Config.load(cli_overrides)
    config.validate()
    console.setup()

    console.print_system(f"coding-agent v{__version__}")
    console.print_system(f"模型：{config.model} | 工作目录：{config.workdir}"
                         f" | 权限：{'全自动（-y）' if config.auto_yes else '分级确认'}")

    agent = Agent(config)
    if args.session:
        data = _load_session(Path(args.session))
        agent.conversation = Conversation.from_session(
            data, agent.conversation.messages[0]["content"],
            config.context_window, config.keep_last_messages,
            config.response_max_tokens,
        )
        console.print_system(
            f"已恢复会话（{len(agent.conversation.messages) - 1} 条历史消息）"
        )

    # ---------------- 单任务模式 ----------------
    if args.task is not None:
        try:
            result = agent.run(args.task)
        except LLMAuthError as e:
            console.print_error(str(e))
            return 2
        _print_status(result)
        return 0 if result.status == "finished" else 1

    # ---------------- REPL 模式 ----------------
    console.print_system("输入任务开始。输入 /help 查看内置命令。")
    while True:
        try:
            line = input("你 › ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print_system("按 Ctrl+C 两次或输入 /exit 退出")
            continue

        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line[1:].strip().partition(" ")
            if cmd in ("exit", "quit"):
                break
            if cmd == "help":
                print(HELP_TEXT)
            elif cmd == "reset":
                agent.conversation.messages = agent.conversation.messages[:1]
                console.print_system("对话历史已清空")
            elif cmd == "model" and arg:
                agent.config.model = arg  # 模型名是请求级参数，下轮请求即生效
                console.print_system(f"已切换模型：{arg}")
            elif cmd == "yes":
                agent.permission.auto_yes = True
                console.print_system("已开启全自动模式，不再确认工具调用")
            elif cmd == "save":
                path = _save_session(
                    agent,
                    Path(arg) if arg else Path("sessions")
                    / f"session-{datetime.now():%Y%m%d-%H%M%S}.json",
                )
                console.print_system(f"会话已保存：{path}")
            else:
                console.print_error(f"未知命令：/{cmd}（输入 /help 查看帮助）")
            continue

        try:
            result = agent.run(line)
        except LLMAuthError as e:
            console.print_error(str(e))
            return 2
        _print_status(result)

    console.print_system("再见。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
