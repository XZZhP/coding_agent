"""Agent 核心循环测试：终止条件、工具执行、错误回传、权限分级。

全部使用 FakeLLM 脚本化响应，不产生真实网络请求。
"""
from __future__ import annotations

from pathlib import Path

from coding_agent.agent import Agent, PermissionManager
from coding_agent.config import Config
from coding_agent.llm import LLMContextOverflow

from fake_llm import FakeLLM, resp_calls, resp_text, tc


def make_agent(tmp_path: Path, fake, auto_yes=True, max_turns=5,
               confirm=None, **cfg_kw) -> Agent:
    cfg = Config(api_key="sk-test", workdir=tmp_path, max_turns=max_turns,
                 auto_yes=auto_yes, **cfg_kw)
    agent = Agent(cfg)
    agent.llm = fake
    if confirm is not None:
        agent.permission = PermissionManager(auto_yes=auto_yes, confirm=confirm)
    return agent


def _last(agent: Agent, role: str) -> dict:
    return [m for m in agent.conversation.messages if m["role"] == role][-1]


# ---------------------------------------------------------------- 终止条件

def test_natural_termination_when_no_tool_calls(tmp_path):
    """模型直接给出答复、不请求工具 → 循环自然终止。"""
    agent = make_agent(tmp_path, FakeLLM([resp_text("你好！有什么可以帮你？")]))
    result = agent.run("打个招呼")
    assert result.status == "finished"
    final = agent.conversation.messages[-1]
    assert final["role"] == "assistant"
    assert final["content"] == "你好！有什么可以帮你？"


def test_tool_loop_then_terminate(tmp_path):
    """先调用工具、看到结果后停止 → 完整的历史链路被正确记录。"""
    fake = FakeLLM([
        resp_calls(tc("list_dir", {"path": "."})),
        resp_text("目录里只有一个文件 a.txt"),
    ])
    agent = make_agent(tmp_path, fake)
    result = agent.run("看看目录里有什么")
    assert result.status == "finished"
    roles = [m["role"] for m in agent.conversation.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool_msg = _last(agent, "tool")
    assert tool_msg["tool_call_id"] == "call_test_list_dir"
    # 第二轮请求携带了完整历史（含工具结果）
    assert len(fake.calls) == 2
    sent = fake.calls[1]["messages"]
    assert sent[3]["role"] == "tool"


def test_max_turns_stops_loop(tmp_path):
    """模型每轮都要求工具 → 达到 max_turns 强制停止。"""
    fake = FakeLLM([resp_calls(tc("list_dir", {"path": "."}))] * 10)
    agent = make_agent(tmp_path, fake, max_turns=3)
    result = agent.run("干活")
    assert result.status == "max_turns"
    assert len(fake.calls) == 3


def test_repeat_detection_injects_intervention(tmp_path):
    """连续 3 轮完全相同的工具调用 → 判定卡死并注入干预提示。"""
    fake = FakeLLM([resp_calls(tc("list_dir", {"path": "."}))] * 4)
    agent = make_agent(tmp_path, fake, max_turns=10)
    agent.run("干活")
    contents = [m["content"] for m in agent.conversation.messages
                if m["role"] == "user"]
    assert any("完全相同的工具调用" in c for c in contents)


# ---------------------------------------------------------------- 工具执行

def test_write_file_via_agent(tmp_path):
    fake = FakeLLM([
        resp_calls(tc("write_file", {"path": "out.txt", "content": "由 agent 生成"})),
        resp_text("已完成写入"),
    ])
    agent = make_agent(tmp_path, fake)
    agent.run("创建 out.txt")
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "由 agent 生成"


def test_unknown_tool_reported_to_model(tmp_path):
    fake = FakeLLM([
        resp_calls(tc("hack_the_planet", {})),
        resp_text("收到，我换一个工具"),
    ])
    agent = make_agent(tmp_path, fake)
    agent.run("x")
    assert "未知工具" in _last(agent, "tool")["content"]


def test_tool_error_fed_back_to_model(tmp_path):
    """工具执行失败 → 错误文本回传模型（而非崩溃），模型自我修正。"""
    fake = FakeLLM([
        resp_calls(tc("read_file", {"path": "不存在.txt"})),
        resp_text("文件不存在，我先创建它"),
    ])
    agent = make_agent(tmp_path, fake)
    result = agent.run("读一个文件")
    assert result.status == "finished"
    assert "工具执行失败" in _last(agent, "tool")["content"]


def test_parse_error_fed_back_without_execution(tmp_path):
    fake = FakeLLM([
        resp_calls(tc("write_file", parse_error="参数不是 JSON")),
        resp_text("我重新生成参数"),
    ])
    agent = make_agent(tmp_path, fake)
    agent.run("写文件")
    assert "工具调用无效" in _last(agent, "tool")["content"]
    assert list(tmp_path.iterdir()) == []  # 未被实际执行


def test_multiple_tool_calls_executed_in_order(tmp_path):
    """一轮内多个工具调用按序执行。"""
    fake = FakeLLM([
        resp_calls(
            tc("write_file", {"path": "1.txt", "content": "一"}, call_id="c1"),
            tc("write_file", {"path": "2.txt", "content": "二"}, call_id="c2"),
        ),
        resp_text("两个文件都写好了"),
    ])
    agent = make_agent(tmp_path, fake)
    agent.run("写两个文件")
    assert (tmp_path / "1.txt").exists() and (tmp_path / "2.txt").exists()
    tool_ids = [m["tool_call_id"] for m in agent.conversation.messages
                if m["role"] == "tool"]
    assert tool_ids == ["c1", "c2"]


# ---------------------------------------------------------------- 权限分级

def test_read_tool_auto_allowed_without_confirm(tmp_path):
    asked: list[str] = []
    fake = FakeLLM([resp_calls(tc("list_dir", {"path": "."})),
                    resp_text("done")])
    agent = make_agent(tmp_path, fake, auto_yes=False,
                       confirm=lambda p: (asked.append(p), "y")[1])
    agent.run("列目录")
    assert asked == []  # read 级工具不询问


def test_write_tool_denied_by_user(tmp_path):
    fake = FakeLLM([
        resp_calls(tc("write_file", {"path": "x.txt", "content": "x"})),
        resp_text("好的，我不写了"),
    ])
    agent = make_agent(tmp_path, fake, auto_yes=False,
                       confirm=lambda p: "n")
    agent.run("写文件")
    assert not (tmp_path / "x.txt").exists()
    assert "用户拒绝了" in _last(agent, "tool")["content"]


def test_confirm_a_granted_switches_to_auto(tmp_path):
    """确认时选择 a（全部允许）→ 之后不再询问。"""
    asked: list[str] = []
    fake = FakeLLM([
        resp_calls(tc("write_file", {"path": "1.txt", "content": "1"}, call_id="c1")),
        resp_calls(tc("write_file", {"path": "2.txt", "content": "2"}, call_id="c2")),
        resp_text("done"),
    ])
    agent = make_agent(tmp_path, fake, auto_yes=False,
                       confirm=lambda p: (asked.append(p), "a")[1])
    agent.run("写两个文件")
    assert len(asked) == 1
    assert (tmp_path / "2.txt").exists()


def test_auto_yes_skips_all_confirmations(tmp_path):
    fake = FakeLLM([
        resp_calls(tc("run_command", {"command": "echo hi"})),
        resp_text("命令执行成功"),
    ])
    agent = make_agent(tmp_path, fake, auto_yes=True)
    agent.run("执行命令")
    assert "hi" in _last(agent, "tool")["content"]


# ---------------------------------------------------------------- 错误处理

def test_context_overflow_triggers_emergency_truncate(tmp_path):
    """API 报上下文超限 → 裁剪早期历史后重试一次并成功。"""

    class OverflowOnce:
        def __init__(self):
            self.calls = []
            self._boom = True

        def chat(self, messages, *args, **kwargs):
            self.calls.append(list(messages))
            if self._boom:
                self._boom = False
                raise LLMContextOverflow("窗口超限")
            return resp_text("最终答复")

    fake = OverflowOnce()
    agent = make_agent(tmp_path, fake)
    for i in range(8):  # 预填历史，使裁剪有内容可裁
        agent.conversation.add_user(f"任务{i}")
        agent.conversation.add_assistant(f"答复{i}", None)
    result = agent.run("继续")
    assert result.status == "finished"
    assert len(agent.conversation.messages) < 1 + 16  # 已被裁剪
    assert len(fake.calls) == 2  # 溢出 + 重试


def test_context_overflow_with_short_history_returns_error(tmp_path):
    """裁剪无可裁（历史太短）→ 溢出按普通错误处理，不崩溃。"""
    from coding_agent.llm import LLMContextOverflow as Overflow

    class AlwaysOverflow:
        def __init__(self):
            self.calls = 0

        def chat(self, *args, **kwargs):
            self.calls += 1
            raise Overflow("窗口超限")

    agent = make_agent(tmp_path, AlwaysOverflow())
    result = agent.run("任务")
    assert result.status == "error"
    assert "窗口超限" in result.message


def test_interrupt_during_run(tmp_path):
    """KeyboardInterrupt 应被捕获并返回 interrupted，而不是崩溃。"""

    class Interrupting:
        def chat(self, *args, **kwargs):
            raise KeyboardInterrupt

    agent = make_agent(tmp_path, Interrupting())
    result = agent.run("任务")
    assert result.status == "interrupted"


# ---------------------------------------------------------------- 权限管理器

def test_permission_manager_read_always_allowed():
    from coding_agent.tools import build_registry
    pm = PermissionManager(auto_yes=False, confirm=lambda p: "n")  # 全拒
    tool = build_registry()["read_file"]
    assert pm.ask(tool, {"path": "a"}) is True  # 只读不受拒


def test_permission_manager_write_requires_yes():
    from coding_agent.tools import build_registry
    reg = build_registry()
    pm = PermissionManager(auto_yes=False, confirm=lambda p: "y")
    assert pm.ask(reg["write_file"], {"path": "a", "content": "x"}) is True
    pm = PermissionManager(auto_yes=False, confirm=lambda p: "n")
    assert pm.ask(reg["run_command"], {"command": "del *.*"}) is False
