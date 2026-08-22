from __future__ import annotations

import json
import hashlib
import os
import random
import re
import threading
import time
from dataclasses import dataclass
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


class ConnectorCircuitOpen(ConnectorError):
    """Raised when a provider is temporarily cooled down after repeated failures."""


@dataclass(frozen=True)
class _CircuitState:
    failures: int = 0
    opened_at: float | None = None


class RequestCoordinator:
    """Coordinate public API requests without coupling policy to one connector.

    The coordinator is deliberately host-scoped: several connectors may share a
    provider host, and they should not collectively exceed that provider's rate.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_request: dict[str, float] = {}
        self._circuits: dict[str, _CircuitState] = {}

    @staticmethod
    def _float_env(name: str, default: float) -> float:
        try:
            return max(0.0, float(os.getenv(name, str(default))))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        try:
            return max(1, int(os.getenv(name, str(default))))
        except (TypeError, ValueError):
            return default

    def before_request(self, host: str) -> None:
        now = time.monotonic()
        cooldown = self._float_env("SCIDATA_CONNECTOR_CIRCUIT_COOLDOWN_SECONDS", 60.0)
        with self._lock:
            circuit = self._circuits.get(host, _CircuitState())
            if circuit.opened_at is not None and now - circuit.opened_at < cooldown:
                remaining = max(0.0, cooldown - (now - circuit.opened_at))
                raise ConnectorCircuitOpen(
                    f"provider circuit open for host={host}; retry after {remaining:.1f}s"
                )
            if circuit.opened_at is not None:
                self._circuits[host] = _CircuitState()

            interval = self._float_env("SCIDATA_CONNECTOR_MIN_INTERVAL_SECONDS", 0.25)
            next_allowed = max(now, self._last_request.get(host, 0.0) + interval)
            self._last_request[host] = next_allowed
        delay = next_allowed - now
        if delay > 0:
            time.sleep(delay)

    def success(self, host: str) -> None:
        with self._lock:
            self._circuits[host] = _CircuitState()

    def transient_failure(self, host: str) -> None:
        threshold = self._int_env("SCIDATA_CONNECTOR_CIRCUIT_FAILURE_THRESHOLD", 3)
        with self._lock:
            current = self._circuits.get(host, _CircuitState())
            failures = current.failures + 1
            opened_at = current.opened_at
            if failures >= threshold:
                opened_at = time.monotonic()
            self._circuits[host] = _CircuitState(failures=failures, opened_at=opened_at)

    def reset(self) -> None:
        """Clear state for tests and explicit process-level recovery."""
        with self._lock:
            self._last_request.clear()
            self._circuits.clear()


REQUEST_COORDINATOR = RequestCoordinator()


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
        host = urlsplit(full_url).hostname or "unknown"
        try:
            REQUEST_COORDINATOR.before_request(host)
            with safe_urlopen(request, timeout=timeout, allowed_hosts={host} if host else None) as response:
                payload = json.loads(response.read().decode("utf-8"))
                REQUEST_COORDINATOR.success(host)
                return payload
        except HTTPError as exc:  # pragma: no cover - depends on public network state.
            last_exc = exc
            retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            if retryable:
                REQUEST_COORDINATOR.transient_failure(host)
            if not retryable or attempt >= attempts:
                break
        except (TimeoutError, URLError) as exc:  # pragma: no cover - depends on public network state.
            last_exc = exc
            REQUEST_COORDINATOR.transient_failure(host)
            if attempt >= attempts:
                break
        except ConnectorCircuitOpen as exc:
            last_exc = exc
            break
        except Exception as exc:  # pragma: no cover - depends on public network state.
            last_exc = exc
            break
        base_delay = min(retry_sleep_seconds * (2 ** (attempt - 1)), 8.0)
        jitter_ratio = RequestCoordinator._float_env("SCIDATA_CONNECTOR_JITTER_RATIO", 0.25)
        time.sleep(base_delay + random.uniform(0.0, base_delay * jitter_ratio))
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
    """Return the strongest canonical identity key for a source."""
    return source_identity_keys(source)[0]


def source_identity_keys(source: DiscoveredSource) -> list[str]:
    """Return stable identity keys used to cluster provider records.

    Strong identifiers are kept alongside a conservative title/author/year
    key. This lets a Crossref record with a DOI match an OpenAlex record whose
    DOI is represented as a URL, while still retaining a deterministic fallback
    for sources that expose no persistent identifier.
    """
    keys: list[str] = []
    doi = normalize_doi(source.metadata.get("doi") or source.metadata.get("DOI"))
    if doi:
        keys.append(f"doi:{doi}")

    arxiv_id = extract_arxiv_id(
        source.url
        or source.metadata.get("arxiv_id")
        or source.metadata.get("arxiv_url")
        or ""
    )
    if arxiv_id:
        keys.append(f"arxiv:{arxiv_id}")

    record_id = source.metadata.get("record_id") or source.metadata.get("zenodo_id")
    if record_id:
        provider = str(source.metadata.get("provider") or source.source_type).strip().lower()
        keys.append(f"record:{provider}:{str(record_id).strip().lower()}")

    for candidate in (
        source.url,
        source.metadata.get("pdf_url"),
        source.metadata.get("open_access_url"),
    ):
        normalized_url = normalize_source_url(candidate)
        if normalized_url:
            keys.append(f"url:{normalized_url}")

    title_key = _title_author_year_key(source)
    if title_key:
        keys.append(f"title:{title_key}")
    return list(dict.fromkeys(keys)) or [f"source:{source.source_id.lower()}"]


def normalize_doi(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    text = text.rstrip(".,;:)]}")
    return text or None


def extract_arxiv_id(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    match = re.search(r"(?:arxiv[/:]|abs/|pdf/)?(\d{4}\.\d{4,5}(?:v\d+)?)", text)
    return match.group(1) if match else None


def normalize_source_url(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    doi = normalize_doi(text)
    if text.lower().startswith(("doi:", "http://doi.org/", "https://doi.org/", "https://dx.doi.org/")) and doi:
        return f"doi:{doi}"
    try:
        parsed = urlsplit(text)
    except ValueError:
        return text.lower().rstrip("/") or None
    if not parsed.netloc:
        return text.lower().rstrip("/") or None
    host = parsed.netloc.lower().removeprefix("www.")
    path = re.sub(r"/{2,}", "/", parsed.path).rstrip("/")
    arxiv_id = extract_arxiv_id(text) if "arxiv.org" in host else None
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    return f"{host}{path}" or None


def source_cluster_id(source: DiscoveredSource) -> str:
    """Create a deterministic cluster id from the strongest source key."""
    if source.source_cluster_id:
        return source.source_cluster_id
    digest = hashlib.sha1(source_key(source).encode("utf-8")).hexdigest()[:12]
    return f"cluster_{digest}"


def _title_author_year_key(source: DiscoveredSource) -> str:
    title = re.sub(r"[^a-z0-9]+", " ", str(source.title or "").lower()).strip()
    if not title or title.startswith("untitled "):
        return ""
    authors = source.metadata.get("authors") or source.metadata.get("creators") or []
    first_author = ""
    if isinstance(authors, list) and authors:
        first_author = re.sub(r"[^a-z0-9]+", " ", str(authors[0]).lower()).strip()
    elif authors:
        first_author = re.sub(r"[^a-z0-9]+", " ", str(authors).lower()).strip()
    year = source.metadata.get("publication_year") or source.metadata.get("year")
    if not first_author and not year:
        return title
    return "|".join(part for part in (title, first_author, str(year or "")) if part)
