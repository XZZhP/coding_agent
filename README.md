# coding-agent

一个从零实现的编程智能体（coding agent）：基于 DeepSeek 大模型的多轮交互，自主地**读写文件、搜索代码、执行命令**，完成交给它的编程任务 —— 定位为一个简化版的 Claude Code / Codex。

> **不使用任何 agent 框架/SDK**（无 LangChain、无 AutoGen…），不依赖服务端托管工具（无 Code Interpreter / Files API）。模型只通过 OpenAI 兼容接口收发消息，**核心逻辑全部自研**：对话历史与上下文管理、工具定义与本地执行、模型输出解析、循环终止条件、错误处理。

## 快速开始

```bash
# 1. 安装依赖（Python ≥ 3.10）
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows；Linux/macOS 用 .venv/bin/pip

# 2. 配置密钥：复制 .env.example 为 .env 并填入（.env 已被 git 忽略）
#    或设置环境变量 DEEPSEEK_API_KEY

# 3. 启动
.venv/Scripts/python -m coding_agent                 # 交互模式（REPL）
.venv/Scripts/python -m coding_agent -t "帮我写一个贪吃蛇小游戏"   # 单任务模式
.venv/Scripts/python -m coding_agent -y -t "..."     # -y：跳过工具确认（演示用）
```

运行测试：`.venv/Scripts/python -m pytest tests/`

## 它如何工作

```
你输入任务
   │
   ▼
┌────────────────────────────────────────────┐
│  Agent 循环（agent.py，最多 30 轮）          │
│                                            │
│  1. 历史（含全部工具结果）→ DeepSeek API     │◄── 上下文管理（context.py）
│     · 流式打印模型正文                       │      token 精确跟踪、自动压缩
│  2. 模型请求工具？                          │
│     ├─ 否 → 任务完成，循环终止               │
│     └─ 是 → 权限确认（分级）                 │
│             └→ 本地执行工具（tools.py）      │
│                └→ 结果回传模型，进入下一轮    │
└────────────────────────────────────────────┘
```

## 核心特性

| 特性 | 说明 |
|------|------|
| **7 个本地工具** | read_file / write_file / edit_file / list_dir / search / glob / run_command，全部在本机执行 |
| **流式输出** | 模型正文实时打印；工具调用过程以彩色标记展示 |
| **分级权限** | 只读工具自动放行；写文件/执行命令需确认（`-y` 全自动）；拒绝信息回传模型 |
| **错误即信息** | 工具失败、参数非法、未知工具 → 错误文本回传模型，模型自行修正，循环不崩溃 |
| **多重终止条件** | 模型停止调用工具（自然终止）/ 轮数上限 / 连续重复调用检测 / Ctrl+C |
| **上下文压缩** | 用 API 返回的真实 token 用量预测水位，超限时把旧对话压缩为摘要，保留最近消息 |
| **会话存档** | `/save` 保存、`--session` 恢复，跨进程继续对话 |
| **编码自适应** | 文件与命令输出按 utf-8 → gbk 解码，兼容中文 Windows |

## 目录结构

```
coding_agent/
├── cli.py        # 命令行入口：REPL + 单任务模式 + 会话保存/恢复
├── agent.py      # 核心循环：终止条件、权限分级、错误处理（项目心脏）
├── llm.py        # DeepSeek(OpenAI 兼容) 流式调用：分片累积、重试、异常翻译
├── tools.py      # 工具定义（JSON Schema）+ 本地执行器
├── context.py    # 对话历史、token 用量跟踪、摘要式压缩
├── config.py     # 环境变量 / .env 配置加载
└── console.py    # 终端彩色输出（Windows 兼容）
tests/            # 62 个单元测试（FakeLLM 脚本化，无真实网络请求）
```
