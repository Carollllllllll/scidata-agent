from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock, local
from typing import Any, Callable

import httpx


DEFAULT_TEXT_MODELS = (
    "qwen3.7-flash-2026-07-15",
    "qwen3.5-flash-2026-02-23",
    "qwen3-8b",
    "qwen-plus-0112",
    "qwen3-next-80b-a3b-thinking",
    "qwen3.5-27b",
    "qwen3-14b",
)

DEFAULT_VL_MODELS = (
    "qwen3-vl-235b-a22b-thinking",
    "qwen3-vl-32b-thinking",
    "qwen-vl-plus",
)


class LLMConfigurationError(RuntimeError):
    """Raised when the real Qwen/Bailian client is not configured."""


class LLMCallError(RuntimeError):
    """Raised when an LLM call fails or returns invalid content."""


class LLMProviderError(LLMCallError):
    """An HTTP/provider response error with status and response-body context."""

    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"HTTP {status}: {detail}")


@dataclass
class LLMCallTrace:
    node: str
    model: str
    prompt_chars: int
    response_chars: int
    elapsed_ms: int
    success: bool
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    request_id: str | None = None


class QwenBailianClient:
    """OpenAI-compatible Alibaba Cloud Bailian client with model failover.

    ``QWEN_MODELS`` and ``QWEN_VL_MODELS`` are comma-separated ordered pools.
    The singular ``QWEN_MODEL`` and ``QWEN_VL_MODEL`` values, when present,
    are placed first so existing deployments keep their configured primary.
    A model is removed from the current client pool only after a provider-level
    quota, rate-limit, availability, or server error. Timeouts and malformed
    responses are still reported honestly and are handled by the node retry
    policy instead of being silently treated as quota exhaustion.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        models: list[str] | tuple[str, ...] | None = None,
        vl_models: list[str] | tuple[str, ...] | None = None,
        http_client: httpx.Client | None = None,
    ):
        self._run_local = local()
        self._model_lock = RLock()
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_BAILIAN_API_KEY")
        )
        if models is not None:
            self.text_models = _model_pool(models, primary=model)
        else:
            self.text_models = _model_pool(
                _env_model_list("QWEN_MODELS") or DEFAULT_TEXT_MODELS,
                primary=model if model is not None else os.getenv("QWEN_MODEL"),
            )
        if vl_models is not None:
            self.vl_models = _model_pool(vl_models)
        else:
            self.vl_models = _model_pool(
                _env_model_list("QWEN_VL_MODELS") or DEFAULT_VL_MODELS,
                primary=os.getenv("QWEN_VL_MODEL"),
            )
        self._text_index = 0
        self._vl_index = 0
        self.model = self.text_models[0]
        self.base_url = (
            base_url
            if base_url is not None
            else os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            )
        )
        self.timeout = timeout if timeout is not None else _positive_env_int("QWEN_TIMEOUT_SECONDS", 60)
        self.text_max_tokens = _positive_env_int("QWEN_TEXT_MAX_TOKENS", 8192)
        self.vl_model = self.vl_models[0]
        self.vl_timeout = _positive_env_int("QWEN_VL_TIMEOUT_SECONDS", 180)
        self.vl_max_tokens = _positive_env_int("QWEN_VL_MAX_TOKENS", 8192)
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.Client(
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    def _unavailable_models(self, kind: str) -> set[str]:
        attribute = f"unavailable_{kind}_models"
        if not hasattr(self._run_local, attribute):
            setattr(self._run_local, attribute, set())
        return getattr(self._run_local, attribute)

    @property
    def traces(self) -> list[LLMCallTrace]:
        if not hasattr(self._run_local, "traces"):
            self._run_local.traces = []
        return self._run_local.traces

    @property
    def model_events(self) -> list[dict[str, Any]]:
        if not hasattr(self._run_local, "model_events"):
            self._run_local.model_events = []
        return self._run_local.model_events

    @property
    def event_callback(self) -> Callable[[dict[str, Any]], None] | None:
        return getattr(self._run_local, "event_callback", None)

    @event_callback.setter
    def event_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self._run_local.event_callback = callback

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def set_event_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        """Attach a per-run callback for model failover monitoring."""
        self.event_callback = callback

    def emit_runtime_event(self, event: dict[str, Any]) -> None:
        """Publish node-level retry telemetry through the existing monitor hook."""
        self._emit_event(event)

    def model_pool_status(self) -> dict[str, Any]:
        return {
            "text_models": list(self.text_models),
            "active_text_model": self.model,
            "unavailable_text_models": sorted(self._unavailable_models("text")),
            "vl_models": list(self.vl_models),
            "active_vl_model": self.vl_model,
            "unavailable_vl_models": sorted(self._unavailable_models("vl")),
        }

    def require_configured(self) -> None:
        if not self.configured:
            raise LLMConfigurationError(
                "Qwen/Bailian API key is not configured. Set DASHSCOPE_API_KEY "
                "or ALIBABA_BAILIAN_API_KEY."
            )

    def generate_text(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        self.require_configured()
        # Failover exclusions are scoped to one request. A transient provider
        # outage must not permanently drain a long-lived process model pool.
        self._unavailable_models("text").clear()
        prompt_chars = len(system_prompt) + len(user_prompt)
        last_exc: Exception | None = None
        for model in self._available_models("text"):
            started = time.time()
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": self.text_max_tokens,
                "enable_thinking": False,
            }
            try:
                data = self._post_json(payload, timeout=self.timeout)
                choice = data["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise LLMCallError(
                        f"Qwen output truncated at max_tokens={self.text_max_tokens}. "
                        "Increase QWEN_TEXT_MAX_TOKENS."
                    )
                text = choice["message"]["content"]
                if isinstance(text, list):
                    text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
                if not isinstance(text, str):
                    raise LLMCallError("Provider returned non-text message content.")
                self._set_active_model("text", model)
                self.traces.append(
                    LLMCallTrace(
                        node=node,
                        model=model,
                        prompt_chars=prompt_chars,
                        response_chars=len(text),
                        elapsed_ms=int((time.time() - started) * 1000),
                        success=True,
                        **_trace_usage(data),
                    )
                )
                return text
            except Exception as exc:
                last_exc = exc
                self._record_failure(node, model, prompt_chars, started, exc)
                if not self._should_failover(exc):
                    raise _as_llm_error("Qwen", exc) from exc
                self._disable_and_switch("text", model, exc)
        raise LLMCallError(f"Qwen text model pool exhausted; last_error={last_exc}") from last_exc

    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        text = self.generate_text(node, system_prompt, user_prompt, temperature=temperature)
        try:
            return parse_json_from_text(text)
        except ValueError as exc:
            raise LLMCallError(f"Qwen returned invalid JSON: {exc}") from exc

    def generate_vision_text(
        self,
        node: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str | Path],
        temperature: float = 0.1,
    ) -> str:
        """Call the Qwen-VL model with local images in OpenAI-compatible mode."""
        self.require_configured()
        self._unavailable_models("vl").clear()
        content: list[dict[str, Any]] = []
        for image_path in image_paths:
            path = Path(image_path)
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            suffix = path.suffix.lower().lstrip(".") or "png"
            mime = "jpeg" if suffix in {"jpg", "jpeg"} else "png"
            content.append({"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{encoded}"}})
        content.append({"type": "text", "text": user_prompt})
        prompt_chars = len(system_prompt) + len(user_prompt)
        last_exc: Exception | None = None
        for model in self._available_models("vl"):
            started = time.time()
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                "temperature": temperature,
                "max_tokens": self.vl_max_tokens,
                "enable_thinking": False,
            }
            try:
                data = self._post_json(payload, timeout=self.vl_timeout)
                choice = data["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise LLMCallError(
                        f"Qwen-VL output truncated at max_tokens={self.vl_max_tokens}. "
                        "Increase QWEN_VL_MAX_TOKENS."
                    )
                text = choice["message"]["content"]
                if isinstance(text, list):
                    text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
                if not isinstance(text, str):
                    raise LLMCallError("Provider returned non-text VL message content.")
                self._set_active_model("vl", model)
                self.traces.append(
                    LLMCallTrace(
                        node=node,
                        model=model,
                        prompt_chars=prompt_chars,
                        response_chars=len(text),
                        elapsed_ms=int((time.time() - started) * 1000),
                        success=True,
                        **_trace_usage(data),
                    )
                )
                return text
            except Exception as exc:
                last_exc = exc
                self._record_failure(node, model, prompt_chars, started, exc)
                if not self._should_failover(exc):
                    raise _as_llm_error("Qwen-VL", exc) from exc
                self._disable_and_switch("vl", model, exc)
        raise LLMCallError(f"Qwen-VL model pool exhausted; last_error={last_exc}") from last_exc

    def generate_vision_json(
        self,
        node: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str | Path],
        temperature: float = 0.1,
    ) -> Any:
        text = self.generate_vision_text(node, system_prompt, user_prompt, image_paths, temperature=temperature)
        try:
            return parse_json_from_text(text)
        except ValueError as exc:
            raise LLMCallError(f"Qwen-VL returned invalid JSON: {exc}") from exc

    def _post_json(self, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        try:
            response = self._http_client.post(
                self.base_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
        except (httpx.TimeoutException, httpx.TransportError, TimeoutError, OSError) as exc:
            raise LLMCallError(str(exc)) from exc
        raw = response.text
        if response.status_code >= 400:
            raise LLMProviderError(response.status_code, raw or response.reason_phrase)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMCallError(f"Provider returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMCallError("Provider returned a non-object JSON payload.")
        return data

    def close(self) -> None:
        """Release the owned keep-alive connection pool."""
        if self._owns_http_client:
            self._http_client.close()

    def _available_models(self, kind: str) -> list[str]:
        models = self.text_models if kind == "text" else self.vl_models
        unavailable = self._unavailable_models(kind)
        with self._model_lock:
            index = self._text_index if kind == "text" else self._vl_index
        ordered = models[index:] + models[:index]
        return [model for model in ordered if model not in unavailable]

    def _set_active_model(self, kind: str, model: str) -> None:
        models = self.text_models if kind == "text" else self.vl_models
        index = models.index(model)
        with self._model_lock:
            if kind == "text":
                self._text_index = index
                self.model = model
            else:
                self._vl_index = index
                self.vl_model = model

    def _disable_and_switch(self, kind: str, failed_model: str, exc: Exception) -> None:
        models = self.text_models if kind == "text" else self.vl_models
        unavailable = self._unavailable_models(kind)
        unavailable.add(failed_model)
        current_index = models.index(failed_model)
        next_model = next(
            (
                model
                for model in models[current_index + 1 :] + models[:current_index]
                if model not in unavailable
            ),
            None,
        )
        if next_model is None:
            self._emit_event(
                {
                    "event": "model_pool_exhausted",
                    "kind": kind,
                    "failed_model": failed_model,
                    "error": str(exc),
                }
            )
            return
        self._set_active_model(kind, next_model)
        self._emit_event(
            {
                "event": "model_switched",
                "kind": kind,
                "failed_model": failed_model,
                "next_model": next_model,
                "error": str(exc),
            }
        )

    def _record_failure(self, node: str, model: str, prompt_chars: int, started: float, exc: Exception) -> None:
        self.traces.append(
            LLMCallTrace(
                node=node,
                model=model,
                prompt_chars=prompt_chars,
                response_chars=0,
                elapsed_ms=int((time.time() - started) * 1000),
                success=False,
                error=str(exc),
            )
        )

    def _emit_event(self, event: dict[str, Any]) -> None:
        self.model_events.append(event)
        if self.event_callback is not None:
            self.event_callback(event)

    @staticmethod
    def _should_failover(exc: Exception) -> bool:
        if not isinstance(exc, LLMProviderError):
            return False
        detail = exc.detail.lower()
        # A 401/403 account-level failure is not specific to the selected
        # model. In particular, DashScope reports an exhausted free/paid
        # account quota as HTTP 403 (``insufficient_quota``). Rotating through
        # every model in that case only repeats the same doomed request and
        # makes a task look hung. Keep failover for 429 quota/rate-limit
        # responses, which can still be model- or window-specific.
        if _is_account_level_provider_error(exc):
            return False
        transient_or_quota_terms = (
            "quota",
            "rate limit",
            "ratelimit",
            "exhaust",
            "insufficient balance",
            "balance",
            "too many requests",
        )
        model_terms = ("model not found", "model_not_found", "invalid model", "model does not exist")
        if exc.status in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        if any(term in detail for term in transient_or_quota_terms):
            return exc.status not in {400, 413, 422}
        return exc.status in {400, 404} and any(term in detail for term in model_terms)


def _env_model_list(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _positive_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _trace_usage(data: dict[str, Any]) -> dict[str, Any]:
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return {
        "prompt_tokens": _optional_int(usage.get("prompt_tokens")),
        "completion_tokens": _optional_int(usage.get("completion_tokens")),
        "total_tokens": _optional_int(usage.get("total_tokens")),
        "request_id": str(data.get("id")) if data.get("id") else None,
    }


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _model_pool(models: list[str] | tuple[str, ...], primary: str | None = None) -> list[str]:
    ordered = ([primary] if primary else []) + list(models)
    result: list[str] = []
    for model in ordered:
        value = str(model).strip()
        if value and value not in result:
            result.append(value)
    if not result:
        raise LLMConfigurationError("No LLM model is configured.")
    return result


def _as_llm_error(prefix: str, exc: Exception) -> LLMCallError:
    if isinstance(exc, LLMCallError):
        return exc
    return LLMCallError(f"{prefix} call failed: {exc}")


def _is_account_level_provider_error(exc: Exception) -> bool:
    """Return whether a provider response cannot be fixed by changing models."""

    if not isinstance(exc, LLMProviderError):
        return False
    if exc.status == 401:
        return True
    if exc.status != 403:
        return False
    detail = exc.detail.casefold()
    return any(
        term in detail
        for term in (
            "insufficient_quota",
            "quota exhausted",
            "free quota",
            "add funds",
            "invalid api key",
            "api key is invalid",
            "authentication",
            "unauthorized",
        )
    )


def parse_json_from_text(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", stripped, re.DOTALL)
        if not match:
            raise ValueError("No JSON object or array found")
        return json.loads(match.group(1))
