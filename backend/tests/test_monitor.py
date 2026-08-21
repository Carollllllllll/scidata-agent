from __future__ import annotations

from pathlib import Path

import pytest

from scidata_agent.agent.monitor import AgentMonitor


def test_monitor_rejects_path_traversal_task_id(tmp_path) -> None:
    with pytest.raises(ValueError, match="Invalid task ID"):
        AgentMonitor("../../escape", tmp_path)
    assert not (tmp_path.parent / "escape").exists()


def test_monitor_io_failure_does_not_replace_agent_result(tmp_path, monkeypatch) -> None:
    monitor = AgentMonitor("task_safe", tmp_path, console=False)

    def fail_open(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(Path, "open", fail_open)
    monitor.emit("step", "parse", "completed", "done")
