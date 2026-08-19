from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from scidata_agent.llm.client import LLMCallError, QwenBailianClient


class _Response:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self._payload


def _success_payload(text: str = "{\"ok\": true}") -> dict:
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


def _quota_error(url: str = "https://example.invalid") -> urllib.error.HTTPError:
    body = io.BytesIO(b'{"code":"QuotaExhausted","message":"quota exhausted"}')
    return urllib.error.HTTPError(url, 429, "Too Many Requests", {}, body)


def test_text_model_failover_on_quota(monkeypatch):
    calls: list[str] = []

    def fake_urlopen(request, timeout):
        model = json.loads(request.data.decode("utf-8"))["model"]
        calls.append(model)
        if model == "text-one":
            raise _quota_error()
        return _Response(_success_payload())

    client = QwenBailianClient(api_key="test", model="text-one", models=("text-one", "text-two"))
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = client.generate_text("test_node", "system", "user")

    assert json.loads(result)["ok"] is True
    assert calls == ["text-one", "text-two"]
    assert client.model == "text-two"
    assert client.model_events[0]["event"] == "model_switched"
    assert client.model_events[0]["next_model"] == "text-two"
    assert [trace.model for trace in client.traces] == ["text-one", "text-two"]


def test_vl_model_failover_on_quota(tmp_path):
    calls: list[str] = []
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"not-a-real-image-but-valid-for-request-mocking")

    def fake_urlopen(request, timeout):
        model = json.loads(request.data.decode("utf-8"))["model"]
        calls.append(model)
        if model == "vl-one":
            raise _quota_error()
        return _Response(_success_payload('{"figure_type":"chart"}'))

    client = QwenBailianClient(api_key="test", vl_models=("vl-one", "vl-two"))
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = client.generate_vision_text("chart_node", "system", "user", [image_path])

    assert json.loads(result)["figure_type"] == "chart"
    assert calls == ["vl-one", "vl-two"]
    assert client.vl_model == "vl-two"
    assert client.model_events[0]["kind"] == "vl"


def test_timeout_is_not_silently_converted_to_quota_failover():
    calls: list[str] = []

    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data.decode("utf-8"))["model"])
        raise TimeoutError("simulated network timeout")

    client = QwenBailianClient(api_key="test", models=("text-one", "text-two"))
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(LLMCallError, match="simulated network timeout"):
            client.generate_text("test_node", "system", "user")

    assert calls == ["text-one"]
    assert client.model == "text-one"
    assert not client.model_events
