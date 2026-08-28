"""工具层测试：读写、搜索、命令执行、安全边界、截断。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from coding_agent.tools import (ExecutionContext, ToolError, build_registry,
                                decode_bytes, truncate_middle)


def make_ctx(tmp_path: Path) -> ExecutionContext:
    shell = ["cmd", "/c"] if os.name == "nt" else ["sh", "-c"]
    return ExecutionContext(workdir=tmp_path, shell=shell, command_timeout=30)


def run(tool_name: str, args: dict, ctx: ExecutionContext) -> str:
    return build_registry()[tool_name].func(args, ctx)


# ---------------------------------------------------------------- 基础读写

def test_write_and_read_roundtrip(tmp_path):
    ctx = make_ctx(tmp_path)
    out = run("write_file", {"path": "sub/dir/a.txt", "content": "第一行\n第二行"},
              ctx)
    assert "已写入" in out
    assert (tmp_path / "sub/dir/a.txt").is_file()  # 自动创建父目录

    shown = run("read_file", {"path": "sub/dir/a.txt"}, ctx)
    assert "共 2 行" in shown
    assert "     1| 第一行" in shown
    assert "     2| 第二行" in shown


def test_read_file_offset_limit(tmp_path):
    ctx = make_ctx(tmp_path)
    (tmp_path / "n.txt").write_text("\n".join(str(i) for i in range(1, 11)),
                                    encoding="utf-8")
    shown = run("read_file", {"path": "n.txt", "offset": 8, "limit": 2}, ctx)
    assert "第 8-9 行" in shown
    assert "     8| 8" in shown
    assert "    10| 10" not in shown


def test_read_file_missing(tmp_path):
    ctx = make_ctx(tmp_path)
    with pytest.raises(ToolError, match="文件不存在"):
        run("read_file", {"path": "nope.txt"}, ctx)


def test_read_file_binary_rejected(tmp_path):
    ctx = make_ctx(tmp_path)
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02" * 10)
    with pytest.raises(ToolError, match="二进制"):
        run("read_file", {"path": "bin.dat"}, ctx)


def test_read_gbk_encoded_file(tmp_path):
    """中文 Windows 常见的 GBK 文件应被正确解码。"""
    ctx = make_ctx(tmp_path)
    (tmp_path / "gbk.txt").write_bytes("中文内容测试".encode("gbk"))
    shown = run("read_file", {"path": "gbk.txt"}, ctx)
    assert "中文内容测试" in shown


def test_edit_file_success(tmp_path):
    ctx = make_ctx(tmp_path)
    (tmp_path / "c.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    out = run("edit_file", {"path": "c.py", "old_string": "x = 1",
                            "new_string": "x = 42"}, ctx)
    assert "已替换 1 处" in out
    assert (tmp_path / "c.py").read_text(encoding="utf-8") == "x = 42\ny = 2\n"


def test_edit_file_not_found(tmp_path):
    ctx = make_ctx(tmp_path)
    (tmp_path / "c.py").write_text("hello\n", encoding="utf-8")
    with pytest.raises(ToolError, match="0 处匹配"):
        run("edit_file", {"path": "c.py", "old_string": "nope",
                          "new_string": "x"}, ctx)


def test_edit_file_not_unique(tmp_path):
    ctx = make_ctx(tmp_path)
    (tmp_path / "c.py").write_text("a = 1\na = 2\n", encoding="utf-8")
    with pytest.raises(ToolError, match="不够唯一"):
        run("edit_file", {"path": "c.py", "old_string": "a = ",
                          "new_string": "b = "}, ctx)


# ---------------------------------------------------------------- 安全边界

def test_write_outside_workdir_rejected(tmp_path):
    ctx = make_ctx(tmp_path / "w")
    (tmp_path / "w").mkdir()
    outside = tmp_path / "outside.txt"
    with pytest.raises(ToolError, match="工作目录"):
        run("write_file", {"path": str(outside), "content": "x"}, ctx)
    assert not outside.exists()


def test_edit_outside_workdir_rejected(tmp_path):
    ctx = make_ctx(tmp_path / "w")
    (tmp_path / "w").mkdir()
    f = tmp_path / "secret.txt"
    f.write_text("keep me", encoding="utf-8")
    with pytest.raises(ToolError, match="工作目录"):
        run("edit_file", {"path": str(f), "old_string": "keep",
                          "new_string": "x"}, ctx)
    assert f.read_text(encoding="utf-8") == "keep me"


def test_write_inside_nested_path_allowed(tmp_path):
    ctx = make_ctx(tmp_path / "w")
    (tmp_path / "w").mkdir()
    run("write_file", {"path": "a/b/c.txt", "content": "ok"}, ctx)
    assert (tmp_path / "w/a/b/c.txt").read_text(encoding="utf-8") == "ok"


# ---------------------------------------------------------------- 搜索与列表

def test_search_finds_with_line_numbers(tmp_path):
    ctx = make_ctx(tmp_path)
    (tmp_path / "main.py").write_text("def main():\n    return 1\n",
                                      encoding="utf-8")
    (tmp_path / "other.py").write_text("x = main\n", encoding="utf-8")
    out = run("search", {"pattern": "main"}, ctx)
    assert "main.py:1:" in out
    assert "other.py:1:" in out
    assert "共 2 处匹配" in out


def test_search_skips_binary_and_git(tmp_path):
    ctx = make_ctx(tmp_path)
    (tmp_path / "a.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "bin.dat").write_bytes(b"needle\x00binary")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "b.py").write_text("needle\n", encoding="utf-8")
    out = run("search", {"pattern": "needle"}, ctx)
    assert "a.py" in out
    assert "bin.dat" not in out
    assert ".git" not in out


def test_search_invalid_regex(tmp_path):
    ctx = make_ctx(tmp_path)
    with pytest.raises(ToolError, match="正则"):
        run("search", {"pattern": "([unclosed"}, ctx)


def test_glob_matches(tmp_path):
    ctx = make_ctx(tmp_path)
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    (tmp_path / "c.txt").write_text("", encoding="utf-8")
    out = run("glob", {"pattern": "*.py"}, ctx)
    assert "a.py" in out and "b.py" in out and "c.txt" not in out


def test_list_dir(tmp_path):
    ctx = make_ctx(tmp_path)
    (tmp_path / "d").mkdir()
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    out = run("list_dir", {"path": "."}, ctx)
    assert "[DIR]  d/" in out
    assert "f.txt" in out


# ---------------------------------------------------------------- 命令执行

def test_run_command_echo(tmp_path):
    ctx = make_ctx(tmp_path)
    out = run("run_command", {"command": "echo hello-测试"}, ctx)
    assert "hello-测试" in out
    assert "[exit=0" in out


def test_run_command_failure_reports_exit_code(tmp_path):
    ctx = make_ctx(tmp_path)
    out = run("run_command", {"command": "exit 3"}, ctx)
    assert "[exit=3" in out


def test_run_command_timeout_kills(tmp_path):
    import sys
    import time
    ctx = make_ctx(tmp_path)
    # 用 Python 睡眠 30 秒（跨平台可靠），在 2 秒处强制终止整个进程树。
    # 命令中完全不用引号：cmd /c 对含多个引号段的命令有剥离内部引号的行为，
    # 改用脚本文件方式规避（这也是真实使用中的稳妥做法）。
    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    cmd = f"{sys.executable} {sleeper}"
    start = time.time()
    out = run("run_command", {"command": cmd, "timeout": 2}, ctx)
    assert "TIMEOUT" in out
    assert time.time() - start < 15


# ---------------------------------------------------------------- 工具函数

def test_truncate_middle_keeps_head_and_tail():
    text = "HEAD-" + "m" * 2000 + "-TAIL"
    out = truncate_middle(text, 100)
    assert out.startswith("HEAD-")
    assert out.endswith("-TAIL")
    assert "省略" in out


def test_truncate_middle_short_passthrough():
    assert truncate_middle("short", 100) == "short"


def test_decode_bytes_fallback_never_fails():
    assert decode_bytes("utf8文本".encode("utf-8")) == "utf8文本"
    assert decode_bytes("gbk文本".encode("gbk")) == "gbk文本"
    assert decode_bytes(b"\xff\xfe\x00")  # 非法字节流也不抛错
