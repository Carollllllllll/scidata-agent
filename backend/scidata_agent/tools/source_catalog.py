from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from scidata_agent.agent.schemas import (
    AgentState,
    SourceArtifact,
    SourceCatalogEntry,
    SourceInsight,
)


def build_source_catalog(state: AgentState) -> list[SourceCatalogEntry]:
    """Build a normalized source/artifact view from the current agent state.

    Discovery, selection, triage, ingestion, and parsing predate this catalog
    and keep their existing models. This adapter makes their state explicit
    without changing the behavior of any connector or parser.
    """
    sources = state.source_discovery_plan.candidate_sources if state.source_discovery_plan else []
    selections = {
        decision.source_id: decision
        for decision in (state.source_selection_plan.decisions if state.source_selection_plan else [])
    }
    triage = {decision.source_id: decision for decision in state.source_triage_decisions}
    insights_by_source: dict[str, list[SourceInsight]] = {}
    for insight in state.source_insights:
        insights_by_source.setdefault(insight.source_id, []).append(insight)

    parsed_paths = {
        _normalise_path(block.source_path)
        for block in state.parsed_sources.text_blocks
        if block.source_path
    }
    parsed_paths.update(
        _normalise_path(table.source_path)
        for table in state.parsed_sources.tables
        if table.source_path
    )
    parsed_paths.update(
        _normalise_path(asset.source_path)
        for asset in state.parsed_sources.figure_assets
        if asset.source_path
    )
    known_files = [file.path for file in state.files]

    catalog: list[SourceCatalogEntry] = []
    for source in sources:
        artifact_metadata = _combined_source_metadata(source.metadata)
        selection = selections.get(source.source_id)
        triage_decision = triage.get(source.source_id)
        source_insights = insights_by_source.get(source.source_id, [])
        status, failure_reason = _source_status(
            source,
            selection_action=selection.decision if selection else None,
            triage_action=triage_decision.recommended_action if triage_decision else None,
            insights=source_insights,
            parsed_paths=parsed_paths,
            metadata=artifact_metadata,
        )
        artifacts = _build_artifacts(
            source_id=source.source_id,
            source_cluster_id=source.source_cluster_id,
            provider=(triage_decision.provider if triage_decision else None) or source.metadata.get("provider"),
            source_type=source.source_type,
            source_url=source.url,
            metadata=artifact_metadata,
            source_status=status,
            source_insights=source_insights,
            parsed_paths=parsed_paths,
            known_files=known_files,
        )
        if any(artifact.status == "parsed" for artifact in artifacts):
            status = "parsed"
        elif any(artifact.status == "downloaded" for artifact in artifacts):
            status = "downloaded"

        catalog.append(
            SourceCatalogEntry(
                source_id=source.source_id,
                source_cluster_id=source.source_cluster_id,
                title=source.title,
                source_type=source.source_type,
                provider=(
                    (triage_decision.provider if triage_decision else None)
                    or source.metadata.get("provider")
                ),
                url=source.url,
                status=status,
                relevance_score=(
                    triage_decision.relevance_score
                    if triage_decision
                    else selection.priority_score if selection else source.confidence
                ),
                selection_action=selection.decision if selection else None,
                triage_action=triage_decision.recommended_action if triage_decision else None,
                reason=(
                    selection.reason if selection else None
                ) or (triage_decision.reason if triage_decision else None) or source.reason,
                failure_reason=failure_reason,
                artifacts=artifacts,
                metadata=_safe_source_metadata(source.metadata),
            )
        )

    # Uploaded files are valid research materials even when source discovery
    # was not used or could not associate them with a discovered source.
    catalog_paths = {
        _normalise_path(artifact.local_path)
        for entry in catalog
        for artifact in entry.artifacts
        if artifact.local_path
    }
    for uploaded_file in state.files:
        path = _normalise_path(uploaded_file.path)
        if path in catalog_paths:
            continue
        exists = uploaded_file.path.exists()
        artifact_status = "parsed" if path in parsed_paths else "downloaded" if exists else "failed"
        failure_reason = None if exists else "Uploaded file does not exist at execution time."
        artifact = SourceArtifact(
            artifact_id=_stable_artifact_id(
                uploaded_file.file_id,
                _artifact_type(uploaded_file.filename, "unknown"),
                None,
                str(uploaded_file.path),
            ),
            source_id=uploaded_file.file_id,
            artifact_type=_artifact_type(uploaded_file.filename, "unknown"),
            local_path=str(uploaded_file.path),
            content_type=uploaded_file.content_type,
            status=artifact_status,
            parser="pdf_parser" if artifact_status == "parsed" and uploaded_file.path.suffix.lower() == ".pdf" else None,
            failure_reason=failure_reason,
            metadata={"filename": uploaded_file.filename, "origin": "uploaded_file"},
        )
        catalog.append(
            SourceCatalogEntry(
                source_id=uploaded_file.file_id,
                source_cluster_id=None,
                title=uploaded_file.filename,
                source_type="uploaded_file",
                provider="local_upload",
                status=artifact_status,
                artifacts=[artifact],
                metadata={"filename": uploaded_file.filename, "origin": "uploaded_file"},
            )
        )
    return catalog


def refresh_source_catalog(state: AgentState) -> list[SourceCatalogEntry]:
    """Rebuild the shared catalog from the latest agent state.

    The catalog is intentionally derived from state rather than incrementally
    mutated. This keeps status transitions deterministic when a connector,
    downloader, or parser updates an existing source in place.
    """
    state.source_catalog = build_source_catalog(state)
    return state.source_catalog


def source_catalog_summary(catalog: list[SourceCatalogEntry]) -> dict[str, Any]:
    """Return compact counts for monitor events and processing logs."""
    source_statuses: dict[str, int] = {}
    artifact_statuses: dict[str, int] = {}
    artifact_count = 0
    for entry in catalog:
        source_statuses[entry.status] = source_statuses.get(entry.status, 0) + 1
        for artifact in entry.artifacts:
            artifact_count += 1
            artifact_statuses[artifact.status] = artifact_statuses.get(artifact.status, 0) + 1
    return {
        "source_catalog_count": len(catalog),
        "source_artifacts_count": artifact_count,
        "source_catalog_statuses": source_statuses,
        "source_artifact_statuses": artifact_statuses,
    }


def source_catalog_rows(catalog: list[SourceCatalogEntry]) -> list[dict[str, Any]]:
    """Flatten catalog entries for a human-readable CSV export."""
    rows: list[dict[str, Any]] = []
    for entry in catalog:
        artifacts = entry.artifacts or [None]
        for artifact in artifacts:
            rows.append(
                {
                    "source_id": entry.source_id,
                    "source_cluster_id": entry.source_cluster_id,
                    "title": entry.title,
                    "source_type": entry.source_type,
                    "provider": entry.provider,
                    "source_url": entry.url,
                    "source_status": entry.status,
                    "relevance_score": entry.relevance_score,
                    "selection_action": entry.selection_action,
                    "triage_action": entry.triage_action,
                    "source_reason": entry.reason,
                    "source_failure_reason": entry.failure_reason,
                    "artifact_id": artifact.artifact_id if artifact else None,
                    "artifact_type": artifact.artifact_type if artifact else None,
                    "artifact_source_cluster_id": artifact.source_cluster_id if artifact else None,
                    "artifact_provider": artifact.provider if artifact else None,
                    "artifact_name": artifact.name if artifact else None,
                    "artifact_size_bytes": artifact.size_bytes if artifact else None,
                    "artifact_url": artifact.url if artifact else None,
                    "local_path": artifact.local_path if artifact else None,
                    "content_type": artifact.content_type if artifact else None,
                    "artifact_status": artifact.status if artifact else None,
                    "parser": artifact.parser if artifact else None,
                    "artifact_failure_reason": artifact.failure_reason if artifact else None,
                }
            )
    return rows


def _source_status(
    source,
    *,
    selection_action: str | None,
    triage_action: str | None,
    insights: list[SourceInsight],
    parsed_paths: set[str],
    metadata: dict[str, Any],
) -> tuple[str, str | None]:
    if selection_action in {"reject"} or triage_action == "skip":
        return "skipped", None
    failures = [insight.content for insight in insights if insight.insight_type == "download_error"]
    if failures and not any(
        insight.insight_type in {"downloaded_file", "readme", "source_summary"}
        for insight in insights
    ):
        return "failed", failures[-1]
    local_paths = _metadata_paths(metadata)
    if any(_normalise_path(path) in parsed_paths for path in local_paths):
        return "parsed", None
    if local_paths:
        return "downloaded", None
    if insights:
        return "metadata_read", None
    if selection_action or triage_action:
        return "selected", None
    return "discovered", None


def _build_artifacts(
    *,
    source_id: str,
    source_cluster_id: str | None,
    provider: str | None,
    source_type: str,
    source_url: str | None,
    metadata: dict[str, Any],
    source_status: str,
    source_insights: list[SourceInsight],
    parsed_paths: set[str],
    known_files: list[Path],
) -> list[SourceArtifact]:
    artifacts: list[SourceArtifact] = []
    seen: set[tuple[str, str | None, str | None]] = set()

    def add(artifact: SourceArtifact) -> None:
        artifact.source_cluster_id = source_cluster_id
        artifact.provider = artifact.provider or provider
        artifact.name = artifact.name or _artifact_name_from_value(artifact.local_path or artifact.url)
        if artifact.size_bytes is None and artifact.local_path:
            try:
                artifact.size_bytes = Path(artifact.local_path).stat().st_size
            except (OSError, ValueError):
                pass
        artifact.artifact_id = _stable_artifact_id(
            artifact.source_id,
            artifact.artifact_type,
            artifact.url,
            artifact.local_path,
        )
        assessments = metadata.get("artifact_relevance_assessments")
        if isinstance(assessments, dict):
            assessment = assessments.get(artifact.artifact_id)
            if isinstance(assessment, dict):
                artifact.relevance_score = assessment.get("overall_score")
                artifact.field_scores = assessment.get("field_scores") or {}
                artifact.relevance_reason = assessment.get("rationale")
                artifact.evidence_types = assessment.get("evidence_types") or []
        key = (artifact.artifact_type, artifact.url, artifact.local_path)
        if key in seen:
            return
        seen.add(key)
        artifacts.append(artifact)

    if source_url:
        add(
            SourceArtifact(
                source_id=source_id,
                artifact_type="landing_page",
                url=source_url,
                status="metadata_read" if source_status not in {"skipped", "failed"} else source_status,
                metadata={"source_url": source_url},
            )
        )

    pdf_url = _first_text(metadata.get("pdf_url"))
    open_access_url = _first_text(metadata.get("open_access_url"))
    for url in _unique_texts(
        [
            pdf_url,
            open_access_url,
            *(_as_text_list(metadata.get("pdf_urls"))),
            *(_as_text_list(metadata.get("open_access_urls"))),
        ]
    ):
        local_path = _match_local_path(url, metadata, known_files)
        artifact_provider = (
            metadata.get("artifact_providers", {}).get(url)
            if isinstance(metadata.get("artifact_providers"), dict)
            else None
        ) or provider
        add(
            _artifact_for_path(
                source_id=source_id,
                source_cluster_id=source_cluster_id,
                provider=artifact_provider,
                url=url,
                local_path=local_path,
                source_type=source_type,
                source_status=source_status,
                parsed_paths=parsed_paths,
            )
        )

    existing_local_paths = {
        _normalise_path(artifact.local_path)
        for artifact in artifacts
        if artifact.local_path
    }
    for path in _metadata_paths(metadata):
        # A public PDF URL plus its downloaded_path describes one artifact.
        if _normalise_path(path) in existing_local_paths:
            continue
        add(
            _artifact_for_path(
                source_id=source_id,
                source_cluster_id=source_cluster_id,
                provider=provider,
                url=None,
                local_path=path,
                source_type=source_type,
                source_status=source_status,
                parsed_paths=parsed_paths,
            )
        )
        existing_local_paths.add(_normalise_path(path))

    files = metadata.get("files", [])
    if isinstance(files, list):
        for item in files:
            if isinstance(item, dict):
                url = _first_text(item.get("url") or item.get("download_url"))
                name = _first_text(item.get("name") or item.get("filename"))
                add(
                    SourceArtifact(
                        source_id=source_id,
                        source_cluster_id=source_cluster_id,
                        provider=_first_text(item.get("provider")) or provider,
                        name=name,
                        size_bytes=_item_size_bytes(item),
                        artifact_type=_artifact_type(name or url, source_type),
                        url=url,
                        status="skipped" if source_status == "skipped" else "planned",
                        metadata={key: value for key, value in item.items() if key not in {"url", "download_url"}},
                    )
                )
            elif item:
                add(
                    SourceArtifact(
                        source_id=source_id,
                        source_cluster_id=source_cluster_id,
                        provider=provider,
                        name=str(item),
                        artifact_type=_artifact_type(str(item), source_type),
                        url=str(item) if str(item).startswith(("http://", "https://")) else None,
                        status="skipped" if source_status == "skipped" else "planned",
                        metadata={"name": str(item)},
                    )
                )

    for insight in source_insights:
        if insight.insight_type == "readme":
            add(
                SourceArtifact(
                    source_id=source_id,
                    artifact_type="readme",
                    url=insight.url,
                    status="parsed",
                    parser="readme_text",
                )
            )
        elif insight.insight_type == "file_manifest":
            add(
                SourceArtifact(
                    source_id=source_id,
                    artifact_type="file_manifest",
                    url=insight.url,
                    status="metadata_read",
                    parser="source_manifest",
                )
            )
        elif insight.insight_type == "download_error":
            add(
                SourceArtifact(
                    source_id=source_id,
                    artifact_type="unknown",
                    url=insight.url,
                    status="failed",
                    failure_reason=insight.content,
                )
            )

    if not artifacts:
        add(
            SourceArtifact(
                source_id=source_id,
                artifact_type="landing_page",
                url=source_url,
                status=source_status if source_status in {"discovered", "selected", "skipped", "failed"} else "discovered",
            )
        )
    return artifacts


def _artifact_for_path(
    *,
    source_id: str,
    source_cluster_id: str | None,
    provider: str | None,
    url: str | None,
    local_path: str | None,
    source_type: str,
    source_status: str,
    parsed_paths: set[str],
) -> SourceArtifact:
    if source_status == "skipped":
        status = "skipped"
    elif local_path:
        status = "parsed" if _normalise_path(local_path) in parsed_paths else "downloaded"
    elif source_status == "failed":
        status = "failed"
    elif source_status in {"selected", "metadata_read"}:
        status = "planned"
    else:
        status = "discovered"
    return SourceArtifact(
        source_id=source_id,
        source_cluster_id=source_cluster_id,
        provider=provider,
        name=_artifact_name_from_value(local_path or url),
        artifact_type=_artifact_type(local_path or url, source_type),
        url=url,
        local_path=local_path,
        status=status,
        parser="pdf_parser" if status == "parsed" and str(local_path).lower().endswith(".pdf") else None,
    )


def _metadata_paths(metadata: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("downloaded_path", "downloaded_paths"):
        value = metadata.get(key)
        if isinstance(value, list):
            paths.extend(str(item) for item in value if item)
        elif value:
            paths.append(str(value))
    return paths


def _combined_source_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Flatten canonical and provider-record metadata for artifact discovery."""
    views: list[dict[str, Any]] = [metadata]
    for record in metadata.get("source_records", []) if isinstance(metadata.get("source_records"), list) else []:
        if not isinstance(record, dict):
            continue
        record_metadata = record.get("metadata")
        if isinstance(record_metadata, dict):
            views.append({**record_metadata, "provider": record.get("provider") or record_metadata.get("provider")})

    combined: dict[str, Any] = {}
    files: list[Any] = []
    downloaded_paths: list[str] = []
    pdf_urls: list[str] = []
    open_access_urls: list[str] = []
    artifact_providers: dict[str, str] = {}
    for view in views:
        for key, value in view.items():
            if key in {"files", "downloaded_paths"}:
                continue
            if value not in (None, "", []):
                combined.setdefault(key, value)
        view_provider = view.get("provider")
        for value in view.get("files", []):
            if value in (None, ""):
                continue
            if isinstance(value, dict) and view_provider and not value.get("provider"):
                files.append({**value, "provider": view_provider})
            else:
                files.append(value)
        paths = view.get("downloaded_paths", [])
        if isinstance(paths, list):
            downloaded_paths.extend(str(path) for path in paths if path)
        elif paths:
            downloaded_paths.append(str(paths))
        if view.get("downloaded_path"):
            downloaded_paths.append(str(view["downloaded_path"]))
        view_provider = _first_text(view.get("provider"))
        for key, destination in (("pdf_url", pdf_urls), ("open_access_url", open_access_urls)):
            for url in _as_text_list(view.get(key)):
                destination.append(url)
                if view_provider:
                    artifact_providers.setdefault(url, view_provider)
        for key, destination in (("pdf_urls", pdf_urls), ("open_access_urls", open_access_urls)):
            for url in _as_text_list(view.get(key)):
                destination.append(url)
                if view_provider:
                    artifact_providers.setdefault(url, view_provider)

    combined["files"] = _unique_file_items(files)
    combined["downloaded_paths"] = list(dict.fromkeys(downloaded_paths))
    combined["pdf_urls"] = list(dict.fromkeys(pdf_urls))
    combined["open_access_urls"] = list(dict.fromkeys(open_access_urls))
    combined["artifact_providers"] = artifact_providers
    if combined["downloaded_paths"]:
        combined["downloaded_path"] = combined["downloaded_paths"][0]
    return combined


def _unique_file_items(items: list[Any]) -> list[Any]:
    unique: list[Any] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            url = str(item.get("download_url") or item.get("url") or "").strip().lower()
            identity = url or "|".join(
                str(item.get(key) or "").strip().lower()
                for key in ("name", "filename", "key")
            )
        else:
            identity = str(item).strip().lower()
        if not identity or identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
    return unique


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _unique_texts(values: list[str | None]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _match_local_path(url: str | None, metadata: dict[str, Any], known_files: list[Path]) -> str | None:
    downloaded_artifacts = metadata.get("downloaded_artifacts")
    if url and isinstance(downloaded_artifacts, dict):
        mapped = downloaded_artifacts.get(url)
        if mapped:
            return str(mapped)
    paths = _metadata_paths(metadata)
    if paths:
        return paths[0]
    if not url:
        return None
    token = Path(urlparse(url).path).name.replace(".", "_").lower()
    for path in known_files:
        if token and token in path.name.lower().replace(".", "_"):
            return str(path)
    return None


def _artifact_type(value: str | None, source_type: str) -> str:
    suffix = Path(urlparse(value).path if value and "://" in value else value or "").suffix.lower()
    if suffix == ".pdf":
        lowered = str(value or "").lower()
        return "supplementary_pdf" if source_type == "supplementary_material" or any(token in lowered for token in ("supplement", "supp_", "appendix")) else "pdf"
    if suffix in {".csv", ".tsv", ".xlsx", ".xls", ".json", ".xml"}:
        return suffix.lstrip(".")
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if suffix in {".zip", ".tar", ".gz", ".tgz"}:
        return "code_archive" if source_type == "repository" else "unknown"
    if value and "readme" in value.lower():
        return "readme"
    if source_type == "repository":
        return "code_archive"
    if source_type in {"dataset", "open_database"}:
        return "file_manifest"
    return "html" if value and "://" in value else "unknown"


def _normalise_path(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve()).casefold()


def _artifact_name_from_value(value: str | Path | None) -> str | None:
    if not value:
        return None
    raw_value = str(value)
    if "://" in raw_value:
        raw_value = urlparse(raw_value).path
    return Path(raw_value).name or None


def _first_text(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            text = _first_text(item)
            if text:
                return text
        return None
    if value in (None, ""):
        return None
    return str(value)


def _safe_source_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in {"downloaded_path", "downloaded_paths"}
    }


def _stable_artifact_id(
    source_id: str,
    artifact_type: str,
    url: str | None,
    local_path: str | None,
) -> str:
    identity = "|".join(
        [
            str(source_id),
            str(artifact_type),
            str(url or ""),
            _normalise_path(local_path) if local_path else "",
        ]
    )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    return f"artifact_{digest}"


def _item_size_bytes(item: dict[str, Any]) -> int | None:
    for key in ("size_bytes", "size", "filesize", "file_size"):
        value = item.get(key)
        try:
            if value not in (None, ""):
                return int(value)
        except (TypeError, ValueError):
            continue
    return None
