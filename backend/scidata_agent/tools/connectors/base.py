from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request

from scidata_agent.agent.schemas import DiscoveredSource, SourceSearchRequest
from scidata_agent.tools.url_safety import safe_urlopen


USER_AGENT = "SciDataAgent/0.1 (scientific multi-source discovery; contact=local)"


class ConnectorError(RuntimeError):
    """Raised when a public source connector cannot complete a request."""


class BaseConnector:
    name: str = "base"
    supported_source_types: tuple[str, ...] = ("unknown",)

    def search(self, request: SourceSearchRequest) -> list[DiscoveredSource]:
        raise NotImplementedError

    def download(self, source: DiscoveredSource, download_dir: Path) -> list[Path]:
        return []


def fetch_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = 20,
    headers: dict[str, str] | None = None,
    retries: int = 2,
    retry_sleep_seconds: float = 1.0,
) -> Any:
    query = urlencode({key: value for key, value in (params or {}).items() if value is not None}, doseq=True)
    full_url = f"{url}?{query}" if query else url
    request = Request(
        full_url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            **(headers or {}),
        },
    )
    attempts = max(1, retries + 1)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            host = urlsplit(full_url).hostname
            with safe_urlopen(request, timeout=timeout, allowed_hosts={host} if host else None) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:  # pragma: no cover - depends on public network state.
            last_exc = exc
            if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504} or attempt >= attempts:
                break
        except (TimeoutError, URLError) as exc:  # pragma: no cover - depends on public network state.
            last_exc = exc
            if attempt >= attempts:
                break
        except Exception as exc:  # pragma: no cover - depends on public network state.
            last_exc = exc
            break
        time.sleep(min(retry_sleep_seconds * (2 ** (attempt - 1)), 8.0))
    raise ConnectorError(f"JSON request failed: url={full_url}, attempts={attempts}, error={last_exc}") from last_exc


def compact_text(value: Any) -> str | None:
    if value in (None, "", []):
        return None
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if item not in (None, ""))
    return " ".join(str(value).split()) or None


def first_text(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            text = compact_text(item)
            if text:
                return text
        return None
    return compact_text(value)


def pick_date(value: Any) -> str | None:
    if isinstance(value, dict):
        date_parts = value.get("date-parts")
        if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list):
            return "-".join(f"{int(part):02d}" for part in date_parts[0])
    return compact_text(value)


def source_key(source: DiscoveredSource) -> str:
    doi = str(source.metadata.get("doi") or source.metadata.get("DOI") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    url = str(source.url or source.metadata.get("pdf_url") or source.metadata.get("open_access_url") or "").strip().lower()
    if url:
        return f"url:{url.rstrip('/')}"
    return f"title:{source.title.strip().lower()}"
