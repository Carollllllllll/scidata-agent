from __future__ import annotations

import sys

import pytest

from scidata_agent import cli


class _FakeResult:
    def model_dump(self, **kwargs):
        return {"status": "partial"}


class _FailedResult:
    status = "failed"

    def model_dump(self, **kwargs):
        return {"status": self.status}


class _FakeAgent:
    last_run: dict[str, object] = {}

    def __init__(self, **kwargs):
        self.init_options = kwargs

    def run(self, question, files, **kwargs):
        self.last_run = {"question": question, "files": files, **kwargs}
        _FakeAgent.last_run = self.last_run
        return _FakeResult()


def test_cli_forwards_task_id_and_resume(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "SciDataAgent", _FakeAgent)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scidata-agent",
            "--question",
            "Read the existing task.",
            "--task-id",
            "resume_task_1",
            "--resume",
        ],
    )

    cli.main()

    assert _FakeAgent.last_run["task_id"] == "resume_task_1"
    assert _FakeAgent.last_run["resume"] is True
    assert "partial" in capsys.readouterr().out


def test_cli_uses_dynamic_runtime_by_default(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "SciDataAgent", _FakeAgent)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scidata-agent", "--question", "Run the scientific Agent."],
    )

    cli.main()

    assert _FakeAgent.last_run["enable_dynamic_runtime"] is True
    capsys.readouterr()


def test_cli_can_explicitly_select_legacy_runtime(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "SciDataAgent", _FakeAgent)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scidata-agent", "--question", "Run the compatibility pipeline.", "--legacy-runtime"],
    )

    cli.main()

    assert _FakeAgent.last_run["enable_dynamic_runtime"] is False
    capsys.readouterr()


def test_cli_forwards_discovery_only_mode(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "SciDataAgent", _FakeAgent)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scidata-agent", "--question", "Discover sources.", "--discover-only"],
    )

    cli.main()

    assert _FakeAgent.last_run["files"] == []
    assert _FakeAgent.last_run["discovery_only"] is True
    capsys.readouterr()


def test_cli_returns_nonzero_for_failed_agent_result(monkeypatch, capsys) -> None:
    class _FailedAgent(_FakeAgent):
        def run(self, question, files, **kwargs):
            return _FailedResult()

    monkeypatch.setattr(cli, "SciDataAgent", _FailedAgent)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scidata-agent", "--question", "A failing task."],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1
    assert '"status": "failed"' in capsys.readouterr().out
