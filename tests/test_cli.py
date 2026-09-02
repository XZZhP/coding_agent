"""CLI 层测试：/save 落盘位置决策（纯函数）。"""
from __future__ import annotations

import re
from pathlib import Path

from coding_agent.cli import choose_save_path


def test_save_path_explicit_arg_wins():
    """/save 带显式路径时优先使用显式路径。"""
    assert choose_save_path("my.json", Path("loaded.json")) == Path("my.json")
    assert choose_save_path("my.json", None) == Path("my.json")


def test_save_path_writes_back_to_loaded_session():
    """--session 载入后不带路径的 /save 应写回源文件。"""
    loaded = Path("sessions/session-20260902-132730.json")
    assert choose_save_path(None, loaded) == loaded


def test_save_path_default_timestamped_when_no_session():
    """全新会话（未用 --session）不带路径 /save → 新建时间戳存档。"""
    path = choose_save_path(None, None)
    assert path.parent == Path("sessions")
    assert re.fullmatch(r"session-\d{8}-\d{6}\.json", path.name)
