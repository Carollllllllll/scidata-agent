from __future__ import annotations

import sys

from scidata_agent import cli


class _FakeResult:
    def model_dump(self, **kwargs):
        return {"status": "partial"}


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
