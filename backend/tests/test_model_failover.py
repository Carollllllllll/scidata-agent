from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from scidata_agent.llm.client import LLMCallError, LLMProviderError, QwenBailianClient
from scidata_agent.llm.nodes import QwenAgentNodes


class _Response:
    def __init__(self, payload: dict | bytes, status_code: int = 200):
        self._payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.status_code = status_code
        self.reason_phrase = "error" if status_code >= 400 else "ok"

    @property
    def text(self):
        return self._payload.decode("utf-8")

    def read(self):
        return self._payload


def _success_payload(text: str = "{\"ok\": true}") -> dict:
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


class _HttpClient:
    def __init__(self, handler):
        self.handler = handler
        self.closed = False

    def post(self, url, **kwargs):
        request = type(
            "Request",
            (),
            {"full_url": url, "data": json.dumps(kwargs["json"]).encode("utf-8")},
        )()
        return self.handler(request, kwargs["timeout"])

    def close(self):
        self.closed = True


def _quota_error() -> _Response:
    return _Response(b'{"code":"QuotaExhausted","message":"quota exhausted"}', status_code=429)


def test_text_model_failover_on_quota(monkeypatch):
    calls: list[str] = []

    def fake_urlopen(request, timeout):
        model = json.loads(request.data.decode("utf-8"))["model"]
        calls.append(model)
        if model == "text-one":
            return _quota_error()
        return _Response(_success_payload())

    client = QwenBailianClient(
        api_key="test",
        model="text-one",
        models=("text-one", "text-two"),
        http_client=_HttpClient(fake_urlopen),
    )
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
            return _quota_error()
        return _Response(_success_payload('{"figure_type":"chart"}'))

    client = QwenBailianClient(
        api_key="test",
        vl_models=("vl-one", "vl-two"),
        http_client=_HttpClient(fake_urlopen),
    )
    result = client.generate_vision_text("chart_node", "system", "user", [image_path])

    assert json.loads(result)["figure_type"] == "chart"
    assert calls == ["vl-one", "vl-two"]
    assert client.vl_model == "vl-two"
    assert client.model_events[0]["kind"] == "vl"


def test_text_request_disables_thinking_for_non_streaming_call():
    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data.decode("utf-8")))
        return _Response(_success_payload())

    client = QwenBailianClient(
        api_key="test",
        models=("text-one",),
        http_client=_HttpClient(fake_urlopen),
    )
    client.generate_text("test_node", "system", "user")

    assert captured["enable_thinking"] is False
    assert "stream" not in captured


def test_vision_request_disables_thinking_for_non_streaming_call(tmp_path):
    captured: dict = {}
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"not-a-real-image-but-valid-for-request-mocking")

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data.decode("utf-8")))
        return _Response(_success_payload("vision response"))

    client = QwenBailianClient(
        api_key="test",
        vl_models=("vl-one",),
        http_client=_HttpClient(fake_urlopen),
    )
    client.generate_vision_text("chart_node", "system", "user", [image_path])

    assert captured["enable_thinking"] is False
    assert "stream" not in captured


def test_timeout_is_not_silently_converted_to_quota_failover():
    calls: list[str] = []

    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data.decode("utf-8"))["model"])
        raise TimeoutError("simulated network timeout")

    client = QwenBailianClient(
        api_key="test",
        models=("text-one", "text-two"),
        http_client=_HttpClient(fake_urlopen),
    )
    with pytest.raises(LLMCallError, match="simulated network timeout"):
        client.generate_text("test_node", "system", "user")

    assert calls == ["text-one"]
    assert client.model == "text-one"
    assert not client.model_events


def test_explicit_model_pool_and_empty_key_override_environment(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "environment-key")
    monkeypatch.setenv("QWEN_MODEL", "environment-primary")
    monkeypatch.setenv("QWEN_MODELS", "environment-primary,environment-fallback")

    client = QwenBailianClient(
        api_key="",
        model="explicit-primary",
        models=("explicit-fallback",),
        vl_models=("explicit-vl",),
    )

    assert client.configured is False
    assert client.text_models == ["explicit-primary", "explicit-fallback"]
    assert client.vl_models == ["explicit-vl"]


def test_text_request_limit_and_usage_trace(monkeypatch):
    captured: dict = {}
    payload = _success_payload()
    payload.update(
        {
            "id": "request-123",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }
    )

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data.decode("utf-8")))
        return _Response(payload)

    monkeypatch.setenv("QWEN_TEXT_MAX_TOKENS", "4096")
    client = QwenBailianClient(
        api_key="test",
        models=("text-one",),
        http_client=_HttpClient(fake_urlopen),
    )
    client.generate_text("trace_node", "system", "user")

    assert captured["max_tokens"] == 4096
    trace = client.traces[-1]
    assert (trace.prompt_tokens, trace.completion_tokens, trace.total_tokens) == (11, 7, 18)
    assert trace.request_id == "request-123"


def test_truncated_text_response_fails_explicitly():
    payload = {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}
    client = QwenBailianClient(
        api_key="test",
        models=("text-one",),
        http_client=_HttpClient(lambda *_args: _Response(payload)),
    )

    with pytest.raises(LLMCallError, match="truncated"):
        client.generate_text("test_node", "system", "user")


def test_prompt_limit_400_does_not_disable_or_switch_models():
    client = QwenBailianClient(api_key="test", models=("text-one", "text-two"))

    assert client._should_failover(LLMProviderError(400, "maximum token limit exceeded")) is False
    assert client._should_failover(LLMProviderError(503, "temporary unavailable")) is True


def test_account_level_auth_and_quota_errors_do_not_failover():
    client = QwenBailianClient(api_key="test", models=("text-one", "text-two"))

    assert client._should_failover(LLMProviderError(401, "invalid api key")) is False
    assert client._should_failover(
        LLMProviderError(403, "insufficient_quota: free quota exhausted")
    ) is False


def test_http_403_quota_error_does_not_rotate_the_model_pool():
    calls: list[str] = []

    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data.decode("utf-8"))["model"])
        return _Response(
            b'{"error":{"code":"insufficient_quota","message":"free quota exhausted"}}',
            status_code=403,
        )

    client = QwenBailianClient(
        api_key="test",
        models=("text-one", "text-two"),
        http_client=_HttpClient(fake_urlopen),
    )
    with pytest.raises(LLMProviderError, match="insufficient_quota"):
        client.generate_text("test_node", "system", "user")

    assert calls == ["text-one"]


def test_llm_node_does_not_retry_account_level_provider_errors():
    class _QuotaClient:
        configured = True

        def __init__(self):
            self.calls = 0

        def generate_json(self, *args, **kwargs):
            self.calls += 1
            raise LLMProviderError(403, "insufficient_quota: free quota exhausted")

    client = _QuotaClient()
    nodes = QwenAgentNodes(client)
    with pytest.raises(LLMCallError, match="insufficient_quota"):
        nodes._generate_json_with_retries("test_node", "system", "user")

    assert client.calls == 1


def test_transient_failure_does_not_permanently_exhaust_pool():
    calls: list[str] = []

    def fake_urlopen(request, timeout):
        model = json.loads(request.data.decode("utf-8"))["model"]
        calls.append(model)
        if len(calls) == 1:
            return _Response(b'{"message":"temporary unavailable"}', status_code=503)
        return _Response(_success_payload())

    client = QwenBailianClient(
        api_key="test",
        models=("text-one", "text-two"),
        http_client=_HttpClient(fake_urlopen),
    )
    client.generate_text("first", "system", "user")
    client._set_active_model("text", "text-one")
    client.generate_text("second", "system", "user")

    assert calls == ["text-one", "text-two", "text-one"]


def test_callbacks_and_traces_are_isolated_per_worker_thread():
    client = QwenBailianClient(api_key="test", models=("text-one",))
    barrier = Barrier(2)

    def worker(name: str) -> tuple[list[str], list[str]]:
        events: list[str] = []
        client.set_event_callback(lambda event: events.append(str(event["worker"])))
        barrier.wait()
        client.emit_runtime_event({"worker": name})
        client.traces.append(
            type("Trace", (), {"node": name})()  # only isolation is under test
        )
        return events, [trace.node for trace in client.traces]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, ["one", "two"]))

    assert sorted(results) == [(["one"], ["one"]), (["two"], ["two"])]


def test_invalid_numeric_environment_uses_defaults(monkeypatch):
    monkeypatch.setenv("QWEN_TIMEOUT_SECONDS", "invalid")
    monkeypatch.setenv("QWEN_TEXT_MAX_TOKENS", "invalid")
    monkeypatch.setenv("QWEN_VL_TIMEOUT_SECONDS", "invalid")
    monkeypatch.setenv("QWEN_VL_MAX_TOKENS", "invalid")

    client = QwenBailianClient(api_key="test", models=("text-one",), vl_models=("vl-one",))

    assert client.timeout == 60
    assert client.text_max_tokens == 8192
    assert client.vl_timeout == 180
    assert client.vl_max_tokens == 8192
