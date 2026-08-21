from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def load_dotenv(path: str | Path | None = None) -> None:
    """Load simple KEY=VALUE lines from .env without an external dependency."""

    env_path = Path(path) if path else Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    try:
        content = env_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        LOGGER.warning("Unable to read dotenv file as UTF-8: %s", env_path, exc_info=True)
        return
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = _parse_dotenv_value(value)
        if key and key not in os.environ:
            os.environ[key] = value


def _parse_dotenv_value(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        quote = value[0]
        escaped = False
        for index in range(1, len(value)):
            character = value[index]
            if character == "\\" and not escaped:
                escaped = True
                continue
            if character == quote and not escaped:
                remainder = value[index + 1 :].strip()
                if not remainder or remainder.startswith("#"):
                    return value[1:index]
                return value
            escaped = False
        return value
    for index, character in enumerate(value):
        if character == "#" and index > 0 and value[index - 1].isspace():
            return value[:index].rstrip()
    return value


@dataclass(frozen=True)
class Settings:
    dashscope_api_key: str | None
    qwen_model: str
    qwen_vl_model: str
    qwen_models: tuple[str, ...]
    qwen_vl_models: tuple[str, ...]


def get_settings() -> Settings:
    """Read runtime settings from the environment (after load_dotenv)."""
    qwen_models = tuple(item.strip() for item in os.getenv("QWEN_MODELS", "").split(",") if item.strip())
    qwen_vl_models = tuple(item.strip() for item in os.getenv("QWEN_VL_MODELS", "").split(",") if item.strip())
    qwen_model = os.getenv("QWEN_MODEL", "qwen3.7-flash-2026-07-15")
    qwen_vl_model = os.getenv("QWEN_VL_MODEL", "qwen3-vl-235b-a22b-thinking")
    if qwen_model not in qwen_models:
        qwen_models = (qwen_model, *qwen_models)
    if qwen_vl_model not in qwen_vl_models:
        qwen_vl_models = (qwen_vl_model, *qwen_vl_models)
    return Settings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_BAILIAN_API_KEY"),
        qwen_model=qwen_model,
        qwen_vl_model=qwen_vl_model,
        qwen_models=qwen_models,
        qwen_vl_models=qwen_vl_models,
    )
