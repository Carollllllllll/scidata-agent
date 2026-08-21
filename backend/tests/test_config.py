from __future__ import annotations

import os

from scidata_agent.config import load_dotenv


def test_dotenv_parses_quotes_and_inline_comments_without_stripping_data(tmp_path, monkeypatch) -> None:
    keys = ["SCIDATA_TEST_QUOTED", "SCIDATA_TEST_HASH", "SCIDATA_TEST_MISMATCH"]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\ufeffSCIDATA_TEST_QUOTED=\"value with spaces\" # comment\n"
        "SCIDATA_TEST_HASH=abc#part # comment\n"
        "SCIDATA_TEST_MISMATCH=\"value\"suffix\n",
        encoding="utf-8",
    )

    load_dotenv(env_file)

    assert os.environ["SCIDATA_TEST_QUOTED"] == "value with spaces"
    assert os.environ["SCIDATA_TEST_HASH"] == "abc#part"
    assert os.environ["SCIDATA_TEST_MISMATCH"] == '"value"suffix'


def test_non_utf8_dotenv_is_ignored_instead_of_crashing(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SCIDATA_TEST_INVALID", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"SCIDATA_TEST_INVALID=\xff\xfe")

    load_dotenv(env_file)

    assert "SCIDATA_TEST_INVALID" not in os.environ
