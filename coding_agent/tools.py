"""工具层：工具定义（JSON Schema）+ 本地执行器。

设计要点（面试可展开）：
1. 每个工具 = 名称/描述/参数 Schema + 本地 Python 执行函数。文件读写与命令执行
   全部在本机完成，不依赖任何服务端托管能力（满足题目对
   Code Interpreter / Files API 的禁令）。
2. 权限分级：read（只读，自动放行）/ write（修改文件）/ execute（执行命令）。
   后两类在 agent 循环执行前征求用户确认（--yes 可全自动）。
3. 失败即信息：工具执行失败抛出 ToolError，由 agent 循环捕获后把错误文本
   作为工具结果回传给模型——模型看到真实失败原因后自行修正，循环不崩溃。
4. 输出设上限：所有工具输出统一截断（中部截断，保头保尾），防止大输出
   撑爆模型上下文窗口。
5. 编码自适应：读文件与命令输出按 utf-8 → gbk 顺序解码，兼容中文 Windows
   下 GBK 编码的文件与控制台输出。
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

# search 工具跳过的大型/无关目录
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__",
              "dist", "build", ".idea", ".vscode", ".pytest_cache"}


class ToolError(Exception):
    """工具执行失败。消息会原样回传给模型，帮助其理解并修正。"""


def truncate_middle(text: str, limit: int) -> str:
    """中部截断：保留开头 70% 与结尾 30%，中间给出省略统计。

    选中部截断而非只留头部，是因为错误信息往往在尾部（traceback、
    命令输出的最后几行），只留头会丢掉关键信息。
    """
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    omitted = len(text) - limit
    return text[:head] + f"\n…[已截断，省略 {omitted} 字符]…\n" + text[-tail:]


def decode_bytes(data: bytes) -> str:
    """字节流 → 文本：依次尝试 utf-8、gbk，最后 latin-1 兜底（永不抛错）。"""
    for enc in ("utf-8", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _read_text(path: Path, max_bytes: int = 1_048_576) -> str:
    """读文本文件（有大小上限），返回解码后的文本。二进制文件抛出 ToolError。"""
    size = path.stat().st_size
    data = path.read_bytes() if size <= max_bytes else path.read_bytes()[:max_bytes]
    if b"\x00" in data[:8192]:
        raise ToolError(f"{path} 是二进制文件，read_file 仅支持文本文件")
    return decode_bytes(data)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n}B"


@dataclass
class ExecutionContext:
    """工具执行时的共享上下文。"""

    workdir: Path
    max_output: int = 12000
    command_timeout: float = 120.0
    shell: list[str] = field(default_factory=lambda: ["cmd", "/c"])

    def resolve(self, path: str) -> Path:
        """相对路径锚定到工作目录；绝对路径原样使用。"""
        p = Path(path)
        return p if p.is_absolute() else self.workdir / p

    def require_inside(self, path: str) -> Path:
        """写入类操作的安全边界：目标必须位于工作目录树内。

        用 normcase 做大小写不敏感比较（Windows 盘符与路径大小写无关）。
        """
        wd = os.path.normcase(str(self.workdir.resolve()))
        p = os.path.normcase(str(self.resolve(path).resolve()))
        if p != wd and not p.startswith(wd + os.sep):
            raise ToolError(
                f"出于安全考虑，写入/修改被限制在工作目录 {self.workdir} 内，"
                f"拒绝操作：{path}"
            )
        return Path(p)


@dataclass
class Tool:
    """一个工具：对模型的描述（Schema）+ 本地执行函数。"""

    name: str
    description: str
    parameters: dict          # JSON Schema（properties / required）
    permission: str           # "read" | "write" | "execute"
    func: Callable[[dict, ExecutionContext], str]

    def summarize(self, args: dict) -> str:
        """为权限确认生成一行可读摘要（截断超长参数，避免刷屏）。"""
        brief = json.dumps(args, ensure_ascii=False)
        if len(brief) > 120:
            brief = brief[:120] + f"…(+{len(brief) - 120}字符)"
        return brief

    def schema(self) -> dict:
        """转为 OpenAI tool calling 所需的结构。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------- 工具实现

def _tool_read_file(args: dict, ctx: ExecutionContext) -> str:
    p = ctx.resolve(args["path"])
    if not p.is_file():
        raise ToolError(f"文件不存在：{p}")
    text = _read_text(p)
    lines = text.splitlines()
    offset = max(1, int(args.get("offset", 1)))
    limit = min(int(args.get("limit", 200)), 2000)
    end = min(offset + limit - 1, len(lines))
    shown = lines[offset - 1 : end]
    if len(shown) > limit:
        raise ToolError(f"文件过大，请调小 limit（本次请求 {limit} 行）")
    out = [f"{p}（共 {len(lines)} 行，显示第 {offset}-{end} 行）"]
    out += [f"{i:>6}| {line}" for i, line in enumerate(shown, start=offset)]
    return truncate_middle("\n".join(out), ctx.max_output)


def _tool_write_file(args: dict, ctx: ExecutionContext) -> str:
    p = ctx.require_inside(args["path"])
    content = args["content"]
    p.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    p.write_bytes(data)
    return f"已写入 {p}：{len(content)} 个字符（{len(data)} 字节）"


def _tool_edit_file(args: dict, ctx: ExecutionContext) -> str:
    p = ctx.require_inside(args["path"])
    if not p.is_file():
        raise ToolError(f"文件不存在：{p}")
    old, new = args["old_string"], args["new_string"]
    text = _read_text(p)
    count = text.count(old)
    if count == 0:
        raise ToolError(
            "未找到要替换的内容（0 处匹配）。请用 read_file 重新查看文件当前内容，"
            "注意缩进、换行与空白字符是否一致。"
        )
    if count > 1:
        raise ToolError(
            f"old_string 不够唯一（找到 {count} 处匹配）。"
            "请提供更长的上下文使其唯一，或分多次修改。"
        )
    pos = text.index(old)
    text = text.replace(old, new, 1)
    p.write_bytes(text.encode("utf-8"))
    line_no = text[:pos].count("\n") + 1
    context_start = max(0, text.rfind("\n", 0, pos) + 1)
    context_end = text.find("\n", pos + len(new))
    if context_end < 0:
        context_end = len(text)
    snippet = text[context_start:context_end]
    return f"已替换 1 处（第 {line_no} 行附近）。修改后该行内容：\n{snippet[:300]}"


def _tool_list_dir(args: dict, ctx: ExecutionContext) -> str:
    p = ctx.resolve(args.get("path", "."))
    if not p.exists():
        raise ToolError(f"路径不存在：{p}")
    if p.is_file():
        return f"{p}（文件，{_human_size(p.stat().st_size)}）"
    entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    if len(entries) > 200:
        entries = entries[:200]
        note = f"\n…[目录项过多，仅显示前 200 项]"
    else:
        note = ""
    lines = [f"{p}（{len(list(p.iterdir()))} 项，显示 {len(entries)} 项）"]
    for e in entries:
        try:
            if e.is_dir():
                lines.append(f"[DIR]  {e.name}/")
            else:
                lines.append(f"       {e.name}  ({_human_size(e.stat().st_size)})")
        except OSError:
            lines.append(f"       {e.name}")
    return truncate_middle("\n".join(lines), ctx.max_output) + note


def _tool_search(args: dict, ctx: ExecutionContext) -> str:
    pattern = args["pattern"]
    base = ctx.resolve(args.get("path", "."))
    if not base.is_dir():
        raise ToolError(f"路径不存在或不是目录：{base}")
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"正则表达式无效：{e}") from e

    matches: list[str] = []
    per_file: dict[str, int] = {}
    truncated = False
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for name in sorted(files):
            fp = Path(root) / name
            try:
                if fp.stat().st_size > 1_048_576:  # 跳过 >1MB 的文件
                    continue
                data = fp.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:8192]:
                continue
            rel = fp.relative_to(ctx.workdir)
            for lineno, line in enumerate(decode_bytes(data).splitlines(), 1):
                if regex.search(line):
                    per_file[str(rel)] = per_file.get(str(rel), 0) + 1
                    if per_file[str(rel)] > 20:
                        truncated = True
                        continue
                    matches.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    if len(matches) >= 60:
                        out = [f"共 {len(matches)}+ 处匹配（已达上限 60，建议缩小范围）"] + matches
                        return truncate_middle("\n".join(out), ctx.max_output)
    head = f"共 {len(matches)} 处匹配"
    if truncated:
        head += "（部分文件匹配过多已截断）"
    return truncate_middle(head + "\n" + "\n".join(matches), ctx.max_output)


def _tool_glob(args: dict, ctx: ExecutionContext) -> str:
    pattern = args["pattern"]
    base = ctx.workdir
    found = sorted(
        p.relative_to(base) for p in base.glob(pattern) if p.is_file()
    )
    if not found:
        return f"没有文件匹配 {pattern}"
    note = ""
    if len(found) > 200:
        found = found[:200]
        note = f"\n…[匹配过多，仅显示前 200 个]"
    return "匹配文件：\n" + "\n".join(str(p) for p in found) + note


def _kill_tree(proc: subprocess.Popen) -> None:
    """超时后终止整个进程树（Windows 用 taskkill /T，POSIX 用进程组信号）。"""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True, timeout=15,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        try:
            proc.kill()
        except OSError:
            pass


def _tool_run_command(args: dict, ctx: ExecutionContext) -> str:
    command = args["command"]
    timeout = min(float(args.get("timeout", ctx.command_timeout)), 600.0)
    kwargs: dict = dict(
        cwd=str(ctx.workdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if os.name == "nt":
        kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    start = time.time()
    try:
        proc = subprocess.Popen(ctx.shell + [command], **kwargs)
    except FileNotFoundError:
        raise ToolError(f"命令解释器不可用：{ctx.shell}") from None
    try:
        out_bytes, _ = proc.communicate(timeout=timeout)
        elapsed = time.time() - start
        output = decode_bytes(out_bytes)
        head = f"[exit={proc.returncode}，耗时 {elapsed:.1f}s]"
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        out_bytes, _ = proc.communicate()
        output = decode_bytes(out_bytes)
        head = f"[exit=TIMEOUT，超过 {timeout:.0f}s 被强制终止]"
    return truncate_middle(f"{head}\n{output}".rstrip(), ctx.max_output)


# ---------------------------------------------------------------- 注册表

def build_registry() -> dict[str, Tool]:
    """工具注册表：名称 → Tool。agent 循环按模型请求的名字查表执行。"""
    tools = [
        Tool(
            name="read_file",
            description=(
                "读取文本文件内容（带行号）。首次查看文件时建议 limit 设小一点，"
                "文件很长时用 offset/limit 分页读取。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（相对工作目录或绝对路径）"},
                    "offset": {"type": "integer", "description": "起始行号（1 开始），默认 1"},
                    "limit": {"type": "integer", "description": "读取行数，默认 200"},
                },
                "required": ["path"],
            },
            permission="read",
            func=_tool_read_file,
        ),
        Tool(
            name="write_file",
            description=(
                "将 content 完整写入文件（覆盖原内容，自动创建父目录）。"
                "仅用于新建文件或整体重写；局部修改请用 edit_file。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（相对工作目录）"},
                    "content": {"type": "string", "description": "要写入的完整内容"},
                },
                "required": ["path", "content"],
            },
            permission="write",
            func=_tool_write_file,
        ),
        Tool(
            name="edit_file",
            description=(
                "对文件做精准字符串替换：将 old_string 替换为 new_string。"
                "old_string 必须在文件中恰好出现一次，否则会报错提示。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（相对工作目录）"},
                    "old_string": {"type": "string", "description": "要替换的原文（须唯一匹配）"},
                    "new_string": {"type": "string", "description": "替换后的新文本"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            permission="write",
            func=_tool_edit_file,
        ),
        Tool(
            name="list_dir",
            description="列出目录内容（目录在前，含文件大小）。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径，默认当前工作目录"},
                },
            },
            permission="read",
            func=_tool_list_dir,
        ),
        Tool(
            name="search",
            description=(
                "按正则表达式在工作目录内递归搜索文件内容，返回 文件:行号: 内容。"
                "自动跳过 .git、.venv 等无关目录与二进制文件。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "正则表达式，如 def\\s+main"},
                    "path": {"type": "string", "description": "起始目录，默认工作目录"},
                },
                "required": ["pattern"],
            },
            permission="read",
            func=_tool_search,
        ),
        Tool(
            name="glob",
            description="按通配符模式查找文件（如 **/*.py），返回匹配文件列表。",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "glob 模式，如 src/**/*.py"},
                },
                "required": ["pattern"],
            },
            permission="read",
            func=_tool_glob,
        ),
        Tool(
            name="run_command",
            description=(
                f"在当前工作目录执行一条 shell 命令（{('Windows cmd' if os.name == 'nt' else 'sh')}），"
                "返回退出码、耗时与合并后的输出。"
                "命令失败时先阅读输出分析原因，不要盲目重复执行。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的完整命令"},
                    "timeout": {"type": "number", "description": "超时秒数，默认 120"},
                },
                "required": ["command"],
            },
            permission="execute",
            func=_tool_run_command,
        ),
    ]
    return {t.name: t for t in tools}


def tool_schemas(registry: dict[str, Tool]) -> list[dict]:
    return [t.schema() for t in registry.values()]
