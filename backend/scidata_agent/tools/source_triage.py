from __future__ import annotations

import re
from typing import Any

from scidata_agent.agent.schemas import DiscoveredSource, SourceSelectionDecision, SourceSelectionPlan, SourceTriageDecision


SMALL_FILE_BYTES = 10 * 1024 * 1024
MEDIUM_FILE_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_AUTO_RESOURCES: int | None = None
TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".xml"}
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md"}
AUTO_RESOURCE_ACTIONS = {
    "download_pdf",
    "download_small_table",
    "download_small_supplement",
    "read_readme",
}


def triage_sources(
    sources: list[DiscoveredSource],
    research_question: str,
    max_pdf_downloads: int | None = DEFAULT_MAX_AUTO_RESOURCES,
) -> list[SourceTriageDecision]:
    """Decide lightweight handling actions before any expensive downloads."""
    scored = [(source, _relevance_score(source, research_question)) for source in sources]
    eligible_pdf_sources = sorted(
        [
            (source, score)
            for source, score in scored
            if source.metadata.get("pdf_url")
            and str(source.metadata.get("provider") or "").lower() in {"arxiv", "openalex", "semantic_scholar"}
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    selected_pdf_ids = {
        source.source_id
        for source, _score in (
            eligible_pdf_sources
            if max_pdf_downloads is None or max_pdf_downloads <= 0
            else eligible_pdf_sources[:max_pdf_downloads]
        )
    }
    decisions = []
    for source, score in scored:
        decision = _triage_one_source(source, score, selected_pdf_ids)
        source.metadata["triage_action"] = decision.recommended_action
        source.metadata["triage_relevance_score"] = decision.relevance_score
        source.metadata["triage_should_ingest"] = decision.should_ingest
        source.metadata["triage_reason"] = decision.reason
        decisions.append(decision)
    return decisions


def triage_sources_from_selection(
    sources: list[DiscoveredSource],
    selection_plan: SourceSelectionPlan,
    max_pdf_downloads: int | None = None,
    max_auto_resources: int | None = DEFAULT_MAX_AUTO_RESOURCES,
) -> list[SourceTriageDecision]:
    """Convert LLM semantic source selections into safe executable triage actions."""
    selections = {decision.source_id: decision for decision in selection_plan.decisions}
    selected_pdf_ids = _selected_pdf_ids_from_llm(sources, selections, max_pdf_downloads=max_pdf_downloads)
    decisions = []
    for source in sources:
        selection = selections.get(source.source_id)
        if selection is None:
            decision = _triage_unselected_source(source)
        else:
            decision = _triage_one_source_from_selection(source, selection, selected_pdf_ids)
        _attach_triage_metadata(source, decision)
        decisions.append(decision)
    return _apply_auto_resource_cap(sources, decisions, max_auto_resources=max_auto_resources)


def ingestible_pdf_source_ids(decisions: list[SourceTriageDecision]) -> set[str]:
    return {
        decision.source_id
        for decision in decisions
        if decision.recommended_action == "download_pdf" and decision.should_ingest
    }


def ingestible_arxiv_source_ids(decisions: list[SourceTriageDecision]) -> set[str]:
    return {
        decision.source_id
        for decision in decisions
        if decision.provider == "arxiv"
        and decision.recommended_action == "download_pdf"
        and decision.should_ingest
    }


def _selected_pdf_ids_from_llm(
    sources: list[DiscoveredSource],
    selections: dict[str, SourceSelectionDecision],
    max_pdf_downloads: int | None,
) -> set[str]:
    eligible = []
    for source in sources:
        selection = selections.get(source.source_id)
        provider = str(source.metadata.get("provider") or "").strip().lower()
        if (
            selection
            and selection.decision == "deep_read"
            and source.metadata.get("pdf_url")
            and provider in {"arxiv", "openalex", "semantic_scholar"}
        ):
            eligible.append((source, selection))
    eligible.sort(key=lambda item: _selection_sort_key(item[1], item[0]), reverse=True)
    if max_pdf_downloads is None or max_pdf_downloads <= 0:
        return {source.source_id for source, _selection in eligible}
    return {source.source_id for source, _selection in eligible[: max(0, max_pdf_downloads)]}


def _apply_auto_resource_cap(
    sources: list[DiscoveredSource],
    decisions: list[SourceTriageDecision],
    max_auto_resources: int | None,
) -> list[SourceTriageDecision]:
    """Keep only the best N decisions that trigger remote reads or downloads."""
    if max_auto_resources is None or max_auto_resources <= 0:
        source_by_id = {source.source_id: source for source in sources}
        resource_decisions = [
            decision
            for decision in decisions
            if decision.recommended_action in AUTO_RESOURCE_ACTIONS and _counts_as_auto_resource(decision)
        ]
        allowed_ids = {decision.source_id for decision in resource_decisions}
    else:
        source_by_id = {source.source_id: source for source in sources}
        resource_decisions = [
            decision
            for decision in decisions
            if decision.recommended_action in AUTO_RESOURCE_ACTIONS and _counts_as_auto_resource(decision)
        ]
        ranked = sorted(
            resource_decisions,
            key=lambda decision: _resource_decision_sort_key(decision, source_by_id.get(decision.source_id)),
            reverse=True,
        )
        allowed_ids = {decision.source_id for decision in ranked[:max_auto_resources]}

    capped: list[SourceTriageDecision] = []
    source_by_id = {source.source_id: source for source in sources}
    for decision in decisions:
        if decision.recommended_action not in AUTO_RESOURCE_ACTIONS or not _counts_as_auto_resource(decision):
            capped.append(decision)
            continue
        if decision.source_id in allowed_ids:
            capped.append(decision)
            continue
        capped_decision = _defer_resource_decision(decision, source_by_id.get(decision.source_id), max_auto_resources)
        if capped_decision.recommended_action == "download_pdf":
            capped_decision = capped_decision.model_copy(update={"recommended_action": "read_metadata", "should_ingest": False})
        capped.append(capped_decision)

    for source, decision in zip(sources, capped, strict=False):
        _attach_triage_metadata(source, decision)
    return capped


def _counts_as_auto_resource(decision: SourceTriageDecision) -> bool:
    if decision.recommended_action == "read_readme":
        return True
    return decision.should_ingest


def _resource_decision_sort_key(decision: SourceTriageDecision, source: DiscoveredSource | None) -> tuple[float, float, float]:
    action_boost = {
        "download_pdf": 0.3,
        "download_small_table": 0.25,
        "download_small_supplement": 0.2,
        "read_readme": 0.1,
    }.get(decision.recommended_action, 0.0)
    selection_priority = str(decision.metadata.get("selection_priority") or "").lower()
    priority_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}.get(selection_priority, 0.0)
    source_confidence = source.confidence if source is not None else 0.0
    return (decision.relevance_score, priority_boost + action_boost, source_confidence)


def _defer_resource_decision(
    decision: SourceTriageDecision,
    source: DiscoveredSource | None,
    max_auto_resources: int,
) -> SourceTriageDecision:
    provider = (decision.provider or "").lower()
    if provider in {"arxiv", "openalex", "semantic_scholar", "crossref"}:
        fallback_action = "read_metadata"
    elif source and source.source_type in {"dataset", "open_database", "supplementary_material", "table", "image"}:
        fallback_action = "read_file_manifest"
    else:
        fallback_action = "record_only"
    reason = (
        f"{decision.reason or ''} Deferred by automatic resource cap: "
        f"max_auto_resources={max_auto_resources}."
    ).strip()
    metadata = dict(decision.metadata)
    metadata["resource_cap_deferred"] = True
    metadata["resource_cap"] = max_auto_resources
    metadata["original_recommended_action"] = decision.recommended_action
    return decision.model_copy(
        update={
            "recommended_action": fallback_action,
            "should_ingest": False,
            "reason": reason,
            "metadata": metadata,
        }
    )


def _triage_one_source_from_selection(
    source: DiscoveredSource,
    selection: SourceSelectionDecision,
    selected_pdf_ids: set[str],
) -> SourceTriageDecision:
    provider = str(source.metadata.get("provider") or "").strip().lower() or None
    estimated_size = _estimated_download_size(source)
    action = "record_only"
    should_ingest = False
    reason = selection.reason
    risk = _risk_from_selection(selection)
    cost = _estimated_cost(estimated_size)

    if selection.decision == "reject":
        action = "skip"
        should_ingest = False
        if not reason:
            reason = "LLM rejected this source as off-topic, stale, duplicate, or low value."
    elif selection.decision == "ask_user":
        action = "ask_user"
        should_ingest = False
        risk = "high"
        if not reason:
            reason = "LLM requested user confirmation before ingestion."
    elif selection.decision == "metadata_only":
        action = "read_metadata" if provider in {"arxiv", "openalex", "semantic_scholar", "crossref"} else "record_only"
        should_ingest = False
    elif selection.decision == "read_readme":
        action = "read_readme" if provider == "github" else "record_only"
        should_ingest = False
        if provider != "github":
            reason = f"{selection.reason} Executor downgraded read_readme because provider is {provider or 'unknown'}."
    elif selection.decision == "read_file_manifest":
        action = "read_file_manifest"
        should_ingest = False
    elif selection.decision in {"download_small_table", "download_small_supplement"}:
        file_decision = _file_manifest_decision(source)
        if file_decision["action"] == selection.decision:
            action = file_decision["action"]
            should_ingest = file_decision["should_ingest"]
            estimated_size = file_decision.get("size", estimated_size)
            cost = _estimated_cost(estimated_size)
            risk = max(risk, file_decision["risk"], key=_risk_rank)
            reason = f"{selection.reason} Executor confirmed a safe small file candidate."
        else:
            action = file_decision["action"]
            should_ingest = file_decision["should_ingest"] if action in {"download_small_table", "download_small_supplement"} else False
            estimated_size = file_decision.get("size", estimated_size)
            cost = _estimated_cost(estimated_size)
            risk = max(risk, file_decision["risk"], key=_risk_rank)
            reason = f"{selection.reason} Executor adjusted file action: {file_decision['reason']}"
    elif selection.decision == "deep_read":
        if provider in {"arxiv", "openalex", "semantic_scholar"} and source.metadata.get("pdf_url"):
            if source.source_id in selected_pdf_ids:
                action = "download_pdf"
                should_ingest = True
                cost = "medium"
            else:
                action = "read_metadata"
                should_ingest = False
                reason = f"{selection.reason} Deferred by PDF download budget."
        elif provider in {"openalex", "semantic_scholar", "crossref", "arxiv"}:
            action = "read_metadata"
            should_ingest = False
            reason = f"{selection.reason} Full text was requested by LLM, but no safe PDF URL is available."
            risk = max(risk, "medium", key=_risk_rank)
        elif provider == "github":
            action = "read_readme"
            should_ingest = False
            reason = f"{selection.reason} Source is a repository, so executor starts from README metadata."
        elif source.source_type in {"dataset", "open_database", "supplementary_material", "table", "image"}:
            file_decision = _file_manifest_decision(source)
            action = file_decision["action"]
            should_ingest = file_decision["should_ingest"]
            estimated_size = file_decision.get("size", estimated_size)
            cost = _estimated_cost(estimated_size)
            risk = max(risk, file_decision["risk"], key=_risk_rank)
            reason = f"{selection.reason} Executor mapped deep_read to safe data/supplement inspection."
        else:
            action = "record_only"
            should_ingest = False
            reason = f"{selection.reason} Executor kept it as a record because no safe ingestion route exists."

    return SourceTriageDecision(
        source_id=source.source_id,
        title=source.title,
        provider=provider,
        source_type=source.source_type,
        relevance_score=round(selection.priority_score, 4),
        recommended_action=action,
        reason=reason,
        estimated_download_size=estimated_size,
        estimated_cost=cost,
        risk=risk,
        should_ingest=should_ingest,
        metadata={
            "url": source.url,
            "query": source.query,
            "doi": source.metadata.get("doi") or source.metadata.get("DOI"),
            "pdf_url": source.metadata.get("pdf_url"),
            "open_access_url": source.metadata.get("open_access_url"),
            "selection_decision": selection.decision,
            "selection_priority": selection.priority,
            "selection_role": selection.source_role,
            "selection_reason": selection.reason,
            "selection_matched_requirements": selection.matched_requirements,
            "selection_expected_extractable_fields": selection.expected_extractable_fields,
            "selection_risk_notes": selection.risk_notes,
        },
    )


def _triage_unselected_source(source: DiscoveredSource) -> SourceTriageDecision:
    provider = str(source.metadata.get("provider") or "").strip().lower() or None
    decision = SourceTriageDecision(
        source_id=source.source_id,
        title=source.title,
        provider=provider,
        source_type=source.source_type,
        relevance_score=0.0,
        recommended_action="skip",
        reason="LLM Source Selector did not choose this source for ingestion or metadata reading.",
        estimated_download_size=_estimated_download_size(source),
        estimated_cost=_estimated_cost(_estimated_download_size(source)),
        risk="low",
        should_ingest=False,
        metadata={
            "url": source.url,
            "query": source.query,
            "doi": source.metadata.get("doi") or source.metadata.get("DOI"),
            "pdf_url": source.metadata.get("pdf_url"),
            "open_access_url": source.metadata.get("open_access_url"),
            "selection_decision": "omitted",
        },
    )
    return decision


def _attach_triage_metadata(source: DiscoveredSource, decision: SourceTriageDecision) -> None:
    source.metadata["triage_action"] = decision.recommended_action
    source.metadata["triage_relevance_score"] = decision.relevance_score
    source.metadata["triage_should_ingest"] = decision.should_ingest
    source.metadata["triage_reason"] = decision.reason
    for key in [
        "selection_decision",
        "selection_priority",
        "selection_role",
        "selection_reason",
        "selection_matched_requirements",
        "selection_expected_extractable_fields",
        "selection_risk_notes",
    ]:
        if key in decision.metadata:
            source.metadata[key] = decision.metadata[key]


def _selection_sort_key(selection: SourceSelectionDecision, source: DiscoveredSource) -> tuple[float, float, float]:
    priority_boost = {"high": 1.0, "medium": 0.5, "low": 0.0}.get(selection.priority, 0.5)
    role_boost = {
        "primary_paper": 0.3,
        "dataset": 0.2,
        "supplementary_material": 0.15,
        "supporting_paper": 0.1,
        "metadata_reference": 0.0,
        "code_repository": 0.0,
        "noise": -0.5,
    }.get(selection.source_role, 0.0)
    return (selection.priority_score, priority_boost + role_boost, source.confidence)


def _risk_from_selection(selection: SourceSelectionDecision) -> str:
    notes = " ".join(selection.risk_notes).lower()
    if any(token in notes for token in ["large", "too big", "paywall", "403", "unclear", "stale", "off-topic", "uncertain"]):
        return "medium"
    return "low"


def _risk_rank(value: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(value, 0)


def _triage_one_source(
    source: DiscoveredSource,
    relevance_score: float,
    selected_pdf_ids: set[str],
) -> SourceTriageDecision:
    provider = str(source.metadata.get("provider") or "").strip().lower() or None
    estimated_size = _estimated_download_size(source)
    action = "record_only"
    should_ingest = False
    reason = "Keep as a discovered source; no expensive ingestion is needed yet."
    risk = "low"
    cost = _estimated_cost(estimated_size)

    if provider in {"arxiv", "openalex", "semantic_scholar"} and source.metadata.get("pdf_url"):
        if source.source_id in selected_pdf_ids:
            action = "download_pdf"
            should_ingest = True
            cost = "medium"
            reason = f"High-priority {provider} paper with a PDF URL; selected for deep PDF parsing."
        else:
            action = "read_metadata"
            reason = f"{provider} paper was found, but not selected under the PDF download budget."
    elif provider in {"openalex", "semantic_scholar", "crossref"}:
        action = "read_metadata"
        reason = "Paper index result is useful as bibliographic metadata; defer PDF download unless later selected."
    elif provider == "github":
        action = "read_readme"
        should_ingest = False
        reason = "Repository content can be large; start with metadata/README instead of cloning or downloading files."
    elif provider in {"zenodo", "figshare"}:
        file_decision = _file_manifest_decision(source)
        action = file_decision["action"]
        should_ingest = file_decision["should_ingest"]
        reason = file_decision["reason"]
        estimated_size = file_decision.get("size", estimated_size)
        cost = _estimated_cost(estimated_size)
        risk = file_decision["risk"]
    elif source.source_type in {"dataset", "open_database", "supplementary_material", "table", "image"}:
        action = "read_file_manifest"
        reason = "Data-like source should be inspected through metadata or a file manifest before download."
    elif source.source_type == "repository":
        action = "record_only"
        should_ingest = False
        reason = "Repository-like source is not a normalized GitHub connector result; keep it as a search hint only."

    return SourceTriageDecision(
        source_id=source.source_id,
        title=source.title,
        provider=provider,
        source_type=source.source_type,
        relevance_score=round(relevance_score, 4),
        recommended_action=action,
        reason=reason,
        estimated_download_size=estimated_size,
        estimated_cost=cost,
        risk=risk,
        should_ingest=should_ingest,
        metadata={
            "url": source.url,
            "query": source.query,
            "doi": source.metadata.get("doi") or source.metadata.get("DOI"),
            "pdf_url": source.metadata.get("pdf_url"),
            "open_access_url": source.metadata.get("open_access_url"),
        },
    )


def _file_manifest_decision(source: DiscoveredSource) -> dict[str, Any]:
    files = source.metadata.get("files")
    if not isinstance(files, list) or not files:
        return {
            "action": "read_file_manifest",
            "should_ingest": False,
            "reason": "Dataset/supplementary source has no normalized file list yet; inspect manifest before download.",
            "risk": "low",
            "size": None,
        }
    best_file = _best_small_file(files)
    if best_file is None:
        total_size = _sum_file_sizes(files)
        action = "ask_user" if total_size and total_size > MEDIUM_FILE_BYTES else "read_file_manifest"
        return {
            "action": action,
            "should_ingest": False,
            "reason": "Files are missing size/type hints or appear too large; do not download automatically.",
            "risk": "high" if action == "ask_user" else "medium",
            "size": total_size,
        }

    extension = _file_extension(best_file)
    size = _file_size(best_file)
    if extension in TABLE_EXTENSIONS:
        return {
            "action": "download_small_table",
            "should_ingest": True,
            "reason": f"Small structured file selected for future table ingestion: {best_file.get('name') or best_file.get('key') or extension}.",
            "risk": "low",
            "size": size,
        }
    if extension in DOCUMENT_EXTENSIONS:
        return {
            "action": "download_small_supplement",
            "should_ingest": True,
            "reason": f"Small document-like supplement selected for future ingestion: {best_file.get('name') or best_file.get('key') or extension}.",
            "risk": "low",
            "size": size,
        }
    return {
        "action": "read_file_manifest",
        "should_ingest": False,
        "reason": "File list is useful, but no safe small structured/document file was selected.",
        "risk": "medium",
        "size": size,
    }


def _best_small_file(files: list[Any]) -> dict[str, Any] | None:
    candidates = []
    for file_item in files:
        if not isinstance(file_item, dict):
            continue
        size = _file_size(file_item)
        extension = _file_extension(file_item)
        if size is not None and size > SMALL_FILE_BYTES:
            continue
        if extension in TABLE_EXTENSIONS or extension in DOCUMENT_EXTENSIONS:
            priority = 0 if extension in TABLE_EXTENSIONS else 1
            candidates.append((priority, size or 0, file_item))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1]))[0][2]


def _relevance_score(source: DiscoveredSource, research_question: str) -> float:
    query_tokens = _tokens(research_question)
    if not query_tokens:
        return source.confidence
    source_text = " ".join(
        str(value or "")
        for value in [
            source.title,
            source.description,
            source.reason,
            source.query,
            " ".join(str(item) for item in source.metadata.get("keywords", []) if item)
            if isinstance(source.metadata.get("keywords"), list)
            else "",
            " ".join(str(item) for item in source.metadata.get("topics", []) if item)
            if isinstance(source.metadata.get("topics"), list)
            else "",
        ]
    )
    source_tokens = _tokens(source_text)
    overlap = len(query_tokens & source_tokens) / max(1, len(query_tokens))
    provider_boost = 0.08 if source.metadata.get("provider") in {"arxiv", "semantic_scholar", "openalex"} else 0.0
    score = 0.55 * source.confidence + 0.37 * overlap + provider_boost
    return max(0.0, min(1.0, score))


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", text.lower())
        if token not in {"paper", "dataset", "source", "data", "the", "and", "with", "from"}
    }


def _estimated_download_size(source: DiscoveredSource) -> int | None:
    files = source.metadata.get("files")
    if isinstance(files, list):
        return _sum_file_sizes(files)
    for key in ["size", "download_size", "content_length"]:
        value = source.metadata.get(key)
        try:
            return int(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            continue
    return None


def _sum_file_sizes(files: list[Any]) -> int | None:
    total = 0
    found = False
    for file_item in files:
        if not isinstance(file_item, dict):
            continue
        size = _file_size(file_item)
        if size is not None:
            total += size
            found = True
    return total if found else None


def _file_size(file_item: dict[str, Any]) -> int | None:
    for key in ["size", "filesize", "file_size"]:
        value = file_item.get(key)
        try:
            return int(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            continue
    return None


def _file_extension(file_item: dict[str, Any]) -> str:
    name = str(file_item.get("name") or file_item.get("key") or file_item.get("download_url") or file_item.get("url") or "")
    match = re.search(r"(\.[A-Za-z0-9]+)(?:$|\?)", name.lower())
    if match:
        return match.group(1)
    file_type = str(file_item.get("type") or "").lower()
    if file_type in {"csv", "tsv", "xlsx", "xls", "json", "xml", "pdf", "txt", "md"}:
        return f".{file_type}"
    return ""


def _estimated_cost(size: int | None) -> str:
    if size is None:
        return "unknown"
    if size <= SMALL_FILE_BYTES:
        return "low"
    if size <= MEDIUM_FILE_BYTES:
        return "medium"
    return "high"
