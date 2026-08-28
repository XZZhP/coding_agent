"""配置加载：环境变量 + 未入库的 .env 文件。

优先级：命令行参数 > 环境变量 > .env 文件 > 默认值。

安全约定（对应考核规则）：API key 只允许通过环境变量 DEEPSEEK_API_KEY
或未入库的 .env 文件提供；代码、README、视频中绝不出现真实密钥。
.env 已被 .gitignore 排除，.env.example 仅提供字段模板。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

# 环境变量名 → Config 字段名
_ENV_MAP = {
    "DEEPSEEK_API_KEY": "api_key",
    "DEEPSEEK_BASE_URL": "base_url",
    "DEEPSEEK_MODEL": "model",
}


def load_dotenv(path: Path) -> dict[str, str]:
    """解析 .env 文件为 dict。

    手写实现而非引入 python-dotenv：解析规则只有十几行（KEY=VALUE、
    忽略空行与 # 注释、去引号），核心逻辑自行编写是题目的明确要求。
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        value = value.strip().strip("\"'")
        values[key.strip()] = value
    return values


def find_dotenv() -> Path | None:
    """按 当前工作目录 → 项目根目录 的顺序寻找 .env。"""
    for base in (Path.cwd(), PROJECT_ROOT):
        candidate = base / ".env"
        if candidate.is_file():
            return candidate
    return None


@dataclass
class Config:
    """运行配置。所有字段均可用环境变量 / CLI 参数覆盖。"""

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    workdir: Path = field(default_factory=Path.cwd)

    # —— agent 循环 ——
    max_turns: int = 30              # 单次任务最多循环轮数
    temperature: float = 0.3         # 模型采样温度
    response_max_tokens: int = 8192  # 单次模型回复的长度上限

    # —— 上下文管理 ——
    context_window: int = 65536      # 模型上下文窗口（token）
    keep_last_messages: int = 6      # 压缩历史时保留的最近消息条数

    # —— 工具层 ——
    max_tool_output: int = 12000     # 单次工具输出上限（字符），超出截断
    command_timeout: float = 120.0   # 单条命令执行超时（秒）
    shell: list[str] | None = None   # 命令解释器，None = 按平台取默认

    # —— LLM 调用 ——
    request_timeout: float = 180.0
    max_retries: int = 3             # 瞬时错误（超时/限流/5xx）重试次数

    # —— 交互 ——
    auto_yes: bool = False           # True = 跳过所有工具调用确认（--yes）
    session_file: Path | None = None # 从会话存档恢复（--session）

    @staticmethod
    def load(cli: dict | None = None) -> "Config":
        """按优先级合并各来源配置。cli 为命令行参数映射（仅含显式传入项）。"""
        dotenv = load_dotenv(find_dotenv()) if find_dotenv() else {}
        merged = {
            "api_key": os.environ.get("DEEPSEEK_API_KEY") or dotenv.get("DEEPSEEK_API_KEY", ""),
            "base_url": os.environ.get("DEEPSEEK_BASE_URL") or dotenv.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
            "model": os.environ.get("DEEPSEEK_MODEL") or dotenv.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
        }
        cfg = Config(**merged)
        for key, value in (cli or {}).items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg

    def validate(self) -> None:
        """启动前校验。API key 缺失时给出明确指引，而不是含糊的 SDK 报错。"""
        if not self.api_key:
            raise SystemExit(
                "\n[错误] 未找到 DeepSeek API key。请二选一：\n"
                "  1. 设置环境变量：  set DEEPSEEK_API_KEY=sk-xxx\n"
                "  2. 复制 .env.example 为 .env 并填入密钥（.env 已被 git 忽略，不会入库）\n"
            )
        self.workdir = self.workdir.resolve()
        if not self.workdir.is_dir():
            raise SystemExit(f"[错误] 工作目录不存在：{self.workdir}")

    def default_shell(self) -> list[str]:
        """命令解释器默认值：Windows 用 cmd.exe，其它平台用 sh。

        可用环境变量 CODING_AGENT_SHELL 覆盖，例如（Git Bash）：
            set CODING_AGENT_SHELL=bash -c
        """
        if self.shell:
            return self.shell
        override = os.environ.get("CODING_AGENT_SHELL")
        if override:
            return override.split()
        return ["cmd", "/c"] if os.name == "nt" else ["sh", "-c"]
