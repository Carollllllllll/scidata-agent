from __future__ import annotations

from collections import defaultdict
import hashlib
import os
import re
from typing import Any

from scidata_agent.agent.schemas import (
    AgentState,
    CoverageGap,
    CoverageItem,
    CoverageReport,
    FieldGroupCoverage,
)


def build_coverage_report(state: AgentState) -> CoverageReport:
    """Audit extraction coverage without trusting the planner's stop claim.

    The audit is task-agnostic: requirements come from the dynamic extraction
    plan and source requirements, while evidence comes from parsed records and
    artifact statuses. It deliberately treats an unparsed relevant artifact as
    an open requirement rather than inferring that its metadata is sufficient.
    """
    requirements = _requirement_specs(state)
    field_coverage = {
        str(name).strip(): float(value or 0.0)
        for name, value in state.quality_report.field_coverage.items()
    }
    field_evidence = _record_field_evidence(state)
    items: list[CoverageItem] = []
    missing: list[str] = []
    gaps: list[CoverageGap] = []
    weighted_total = 0.0
    weighted_covered = 0.0

    record_field_counts = _record_field_counts(state)
    for name, priority in requirements:
        coverage = max(
            _matching_coverage(name, field_coverage),
            _matching_coverage(name, field_evidence),
        )
        evidence_count = _matching_count(name, record_field_counts)
        evidence_types = _evidence_types_for_label(name)
        if coverage >= 0.75:
            status = "covered"
        elif coverage > 0.0 or evidence_count > 0:
            status = "partial"
        else:
            status = "missing"
        gap_status = status
        if status == "missing" and _requirement_unavailable(state, evidence_types):
            gap_status = "unavailable"
        gap_actions = _actions_for_requirement(state, evidence_types)
        item = CoverageItem(
            name=name,
            priority=priority,
            status=gap_status,
            evidence_count=evidence_count,
            evidence_types=evidence_types,
            reason=(None if status == "covered" else _requirement_reason(gap_status)),
            coverage_score=coverage,
        )
        items.append(item)
        weight = {"high": 3.0, "medium": 2.0, "low": 1.0}[priority]
        weighted_total += weight
        weighted_covered += weight * min(1.0, coverage)
        if gap_status in {"missing", "partial"} and priority != "low":
            missing.append(name)
        if status in {"missing", "partial"}:
            covered_types = _covered_evidence_types(state)
            gaps.append(
                CoverageGap(
                    gap_id=_stable_gap_id("requirement", name),
                    requirement_name=name,
                    priority=priority,
                    status=gap_status,
                    missing_fields=[name],
                    missing_evidence_types=[
                        evidence_type for evidence_type in evidence_types
                        if evidence_type not in covered_types
                    ],
                    evidence_count=evidence_count,
                    reason=_requirement_reason(gap_status),
                    recommended_actions=gap_actions,
                )
            )

    required_evidence = _required_evidence_types(state)
    covered_evidence = _covered_evidence_types(state)
    missing_evidence = [item for item in required_evidence if item not in covered_evidence]
    relevant_unprocessed = _unprocessed_relevant_artifacts(state)
    field_groups = _field_group_coverage(state, items)

    reasons: list[str] = []
    actions: list[str] = []
    if missing:
        reasons.append(f"Required fields are missing or partial: {', '.join(missing)}.")
        actions.extend(["download_artifact", "parse_pdf_sections", "parse_table", "parse_figure"])
    if missing_evidence:
        reasons.append(f"Required evidence types are not covered: {', '.join(missing_evidence)}.")
        actions.append("download_artifact")
        if _search_more_available(state):
            actions.append("search_more")
        for evidence_type in missing_evidence:
            gaps.append(
                CoverageGap(
                    gap_id=_stable_gap_id("evidence", evidence_type),
                    requirement_name=f"Required evidence: {evidence_type}",
                    priority="high",
                    status=("unavailable" if _evidence_type_unavailable(state, evidence_type) else "missing"),
                    missing_evidence_types=[evidence_type],
                    evidence_count=0,
                    reason=f"No parsed evidence of type {evidence_type!r} is available.",
                    recommended_actions=_actions_for_requirement(state, [evidence_type]),
                )
            )
    if relevant_unprocessed:
        reasons.append(
            f"High-relevance artifacts remain unprocessed: {', '.join(relevant_unprocessed[:12])}."
        )
        actions.append("download_artifact")
        gaps.append(
            CoverageGap(
                gap_id="unprocessed_relevant_artifacts",
                requirement_name="High-relevance artifacts",
                priority="high",
                status="missing",
                reason=(
                    "Relevant artifacts remain undiscovered by the content pipeline: "
                    + ", ".join(relevant_unprocessed[:12])
                ),
                recommended_actions=["download_artifact"],
            )
        )
    searchable_groups = [
        group for group in field_groups
        if group.status == "pending"
        or (
            group.status == "insufficient"
            and group.search_more_count < group.search_more_limit
        )
    ]
    if searchable_groups and not relevant_unprocessed:
        actions.append("search_more")
        reasons.append(
            "Field groups still need evidence: "
            + ", ".join(group.label for group in searchable_groups[:8])
            + "."
        )
    if not items and not covered_evidence and state.source_catalog:
        reasons.append("The catalog exists, but no content evidence has been parsed yet.")
        content_unavailable = _all_catalog_content_unavailable(state)
        if not content_unavailable:
            actions.extend(["download_artifact", "parse_pdf_sections", "read_metadata"])
        gaps.append(
            CoverageGap(
                gap_id="content_evidence_not_started",
                requirement_name="Content evidence",
                priority="high",
                status="unavailable" if content_unavailable else "missing",
                reason=(
                    "Content exists only in remote artifacts and downloads are disabled by policy."
                    if content_unavailable
                    else "The source catalog exists, but no content evidence has been parsed yet."
                ),
                recommended_actions=(
                    []
                    if content_unavailable
                    else ["download_artifact", "parse_pdf_sections", "read_metadata"]
                ),
            )
        )

    # An unavailable gap is still reported, but it cannot be repaired by
    # another planner turn.  Treating it as blocking creates an infinite
    # search/stop-rejection loop with no possible state transition.
    blocking_gaps = [
        gap for gap in gaps
        if gap.priority != "low" and gap.status != "unavailable"
    ]
    field_group_work_complete = (
        bool(field_groups)
        and not relevant_unprocessed
        and all(
            group.initial_search_completed
            and group.status in {"sufficient", "exhausted"}
            for group in field_groups
        )
        and _workflow_batch_ready(state, relevant_unprocessed)
    )
    decision = (
        "allow_stop"
        if not blocking_gaps
        or (field_group_work_complete and not relevant_unprocessed)
        else "continue"
    )
    coverage_score = weighted_covered / weighted_total if weighted_total else (1.0 if covered_evidence else 0.0)
    return CoverageReport(
        decision=decision,
        coverage_score=coverage_score,
        requirements=items,
        gaps=gaps,
        missing_requirements=missing,
        required_evidence_types=required_evidence,
        covered_evidence_types=covered_evidence,
        unprocessed_relevant_artifacts=relevant_unprocessed,
        field_groups=field_groups,
        reasons=reasons,
        recommended_actions=list(dict.fromkeys(actions)),
    )


def _workflow_batch_ready(
    state: AgentState,
    unprocessed_relevant_artifacts: list[str],
) -> bool:
    """Ensure terminal group coverage refers to a fully processed revision."""

    if state.runtime_requires_source_discovery:
        completed = {
            str(item.get("tool_name"))
            for item in state.tool_result_history
            if isinstance(item, dict)
            and item.get("status") in {"completed", "partial", "skipped"}
            and int(item.get("workflow_revision") or 0) == state.workflow_revision
        }
        if not completed.intersection({"search_sources", "search_more"}):
            return False
        if state.source_selection_plan is None or "triage_sources" not in completed:
            return False
        if state.runtime_auto_download_sources:
            providers = {
                str(decision.provider or "").strip().casefold()
                for decision in state.source_triage_decisions
                if decision.should_ingest
            }
            if any(provider != "arxiv" for provider in providers) and "ingest_sources" not in completed:
                return False
            if "arxiv" in providers and "ingest_arxiv_pdfs" not in completed:
                return False
    # Keep the coverage decision aligned with the runtime's batch fingerprint
    # chain. The import is local to avoid a module-initialization cycle.
    from scidata_agent.agent.action_executor import next_required_derived_stage

    return next_required_derived_stage(
        state,
        unprocessed_relevant_artifacts=unprocessed_relevant_artifacts,
    ) is None


def _field_group_coverage(
    state: AgentState,
    items: list[CoverageItem],
) -> list[FieldGroupCoverage]:
    """Summarize evidence and bounded retrieval state per dynamic table."""

    plan = state.dynamic_extraction_plan
    if plan is None:
        return []
    item_by_name = {item.name.strip().casefold(): item for item in items}
    initial = {
        str(group_id).strip().casefold()
        for group_id in state.runtime_group_initial_searches
        if str(group_id).strip()
    }
    retry_counts = {
        str(group_id).strip().casefold(): max(0, int(count))
        for group_id, count in state.runtime_group_search_more_counts.items()
        if str(group_id).strip()
    }
    target = _coverage_float_env("SCIDATA_FIELD_GROUP_COVERAGE_TARGET", 0.60)
    minimum_sources = _coverage_int_env("SCIDATA_FIELD_GROUP_MIN_SOURCES", 3)
    retry_limit = max(0, int(state.runtime_search_more_limit))
    reports: list[FieldGroupCoverage] = []
    non_search_fields = {
        "source_file",
        "source_type",
        "page",
        "evidence_text",
        "confidence",
        "warnings",
    }
    seen_fields: set[str] = set()

    for table in plan.dynamic_tables:
        group_id = _field_group_id(table.table_name)
        fields = [
            field.name.strip()
            for field in table.fields
            if field.name.strip()
            and field.name.strip().casefold() not in non_search_fields
            and field.name.strip().casefold() not in seen_fields
            and field.evidence_required
        ]
        if not fields:
            fields = [
                field.name.strip()
                for field in table.fields
                if field.name.strip()
                and field.name.strip().casefold() not in non_search_fields
                and field.name.strip().casefold() not in seen_fields
            ]
        if not fields:
            continue
        seen_fields.update(name.casefold() for name in fields)
        field_items = [
            item_by_name[name.casefold()]
            for name in fields
            if name.casefold() in item_by_name
        ]
        weighted_total = 0.0
        weighted_score = 0.0
        for item in field_items:
            weight = {"high": 3.0, "medium": 2.0, "low": 1.0}[item.priority]
            weighted_total += weight
            weighted_score += weight * item.coverage_score
        score = weighted_score / weighted_total if weighted_total else 0.0
        required_fields = [
            field.name.strip()
            for field in table.fields
            if field.name.strip() and field.required and field.name.strip() in fields
        ]
        missing_fields = [
            name
            for name in fields
            if (item := item_by_name.get(name.casefold())) is None
            or item.status in {"missing", "partial"}
        ]
        source_files = {
            record.source_file
            for record in state.dynamic_records
            if record.table_name == table.table_name
            and any(value not in (None, "", [], {}) for value in record.fields.values())
        }
        search_more_count = retry_counts.get(group_id, 0)
        initial_completed = group_id in initial
        required_missing = any(name in missing_fields for name in required_fields)
        enough_evidence = (
            score >= target
            and len(source_files) >= minimum_sources
            and not required_missing
        )
        if not initial_completed:
            status = "pending"
        elif enough_evidence:
            status = "sufficient"
        elif search_more_count >= retry_limit:
            status = "exhausted"
        else:
            status = "insufficient"
        reports.append(
            FieldGroupCoverage(
                group_id=group_id,
                label=table.table_name,
                fields=fields,
                required_fields=required_fields,
                missing_fields=missing_fields,
                coverage_score=score,
                evidence_count=sum(item.evidence_count for item in field_items),
                source_count=len(source_files),
                initial_search_completed=initial_completed,
                search_more_count=search_more_count,
                search_more_limit=retry_limit,
                status=status,
            )
        )
    return reports


def _field_group_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
    return slug or "field_group"


def _coverage_float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(0.0, min(1.0, value))


def _coverage_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(0, value)


def _stable_gap_id(prefix: str, value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{slug or 'item'}_{digest}"


def _requirement_reason(status: str) -> str:
    if status == "unavailable":
        return "Candidate artifacts exist, but all matching evidence is failed or skipped."
    if status == "partial":
        return "Some field-level evidence exists, but it is not sufficiently covered."
    return "No sufficient field-level evidence has been extracted."


def _actions_for_requirement(state: AgentState, evidence_types: list[str]) -> list[str]:
    actions: list[str] = []
    mapping = {
        "paper_full_text": ["download_artifact", "parse_pdf_sections"],
        "supplementary_material": ["download_artifact", "parse_pdf_sections"],
        "table": ["download_artifact", "parse_table"],
        "figure": ["download_artifact", "parse_figure"],
        "code_repository": ["download_artifact", "read_file_manifest", "read_readme"],
        "dataset": ["download_artifact", "read_file_manifest", "parse_csv"],
    }
    for evidence_type in evidence_types:
        actions.extend(mapping.get(evidence_type, []))
    if not actions:
        artifact_types = {
            artifact.artifact_type
            for entry in state.source_catalog
            for artifact in entry.artifacts
            if artifact.status not in {"failed", "skipped"}
        }
        if artifact_types & {"pdf", "html", "supplementary_pdf"}:
            actions.extend(["download_artifact", "parse_pdf_sections"])
        if artifact_types & {"csv", "tsv", "xlsx", "json", "xml"}:
            actions.extend(["download_artifact", "parse_table"])
        if artifact_types & {"image"}:
            actions.extend(["download_artifact", "parse_figure"])
        if artifact_types & {"readme", "code_archive", "file_manifest"}:
            actions.extend(["download_artifact", "read_file_manifest", "read_readme"])
    if not actions and _search_more_available(state):
        actions.append("search_more")
    return list(dict.fromkeys(actions))


def _search_more_available(state: AgentState) -> bool:
    limit = int(getattr(state, "runtime_search_more_limit", 2))
    plan = state.dynamic_extraction_plan
    if plan and plan.dynamic_tables:
        counts = state.runtime_group_search_more_counts
        return any(
            int(counts.get(_field_group_id(table.table_name), 0)) < limit
            for table in plan.dynamic_tables
        )
    return int(getattr(state, "runtime_search_more_count", 0)) < limit


def _requirement_unavailable(state: AgentState, evidence_types: list[str]) -> bool:
    return bool(evidence_types) and all(
        _evidence_type_unavailable(state, evidence_type) for evidence_type in evidence_types
    )


def _evidence_type_unavailable(state: AgentState, evidence_type: str) -> bool:
    candidates = [
        artifact
        for entry in state.source_catalog
        for artifact in entry.artifacts
        if evidence_type in _artifact_evidence_types(artifact.artifact_type)
    ]
    if not candidates:
        # Once supplemental discovery is exhausted, the absence of any
        # candidate of this type is an honest unavailable result rather than a
        # permanent reason to keep the Agent alive.
        return not _search_more_available(state)
    return all(
        artifact.status in {"failed", "skipped"}
        or (
            not state.runtime_auto_download_sources
            and not artifact.local_path
        )
        for artifact in candidates
    )


def _all_catalog_content_unavailable(state: AgentState) -> bool:
    artifacts = [
        artifact
        for entry in state.source_catalog
        for artifact in entry.artifacts
    ]
    return bool(artifacts) and all(
        artifact.status in {"failed", "skipped"}
        or (not state.runtime_auto_download_sources and not artifact.local_path)
        for artifact in artifacts
    )


def _artifact_evidence_types(artifact_type: str) -> tuple[str, ...]:
    if artifact_type in {"pdf", "html"}:
        return ("paper_full_text",)
    if artifact_type == "supplementary_pdf":
        return ("supplementary_material",)
    if artifact_type in {"csv", "tsv", "xlsx", "json", "xml"}:
        # A parsed structured file is both a tabular representation and a
        # dataset artifact. Treating it only as a table made dataset coverage
        # impossible to satisfy.
        return ("table", "dataset")
    if artifact_type == "image":
        return ("figure",)
    if artifact_type in {"readme", "code_archive", "file_manifest"}:
        return ("code_repository",)
    return ()


def _requirement_specs(state: AgentState) -> list[tuple[str, str]]:
    plan = state.dynamic_extraction_plan
    values: list[tuple[str, str]] = []
    if plan:
        field_requirements: list[tuple[str, str]] = []
        for table in plan.dynamic_tables:
            for field in table.fields:
                if field.evidence_required and field.name:
                    # evidence_required is an evidence contract, not a
                    # presence contract. Optional fields may legitimately be
                    # absent from a source; report their gap without blocking
                    # completion. Only explicitly required fields block.
                    field_requirements.append((field.name, "high" if field.required else "low"))
        # Information-need labels are narrative planning guidance and often do
        # not share names with the concrete schema fields that satisfy them.
        # When a concrete schema exists, use its measurable field contract and
        # avoid duplicate, permanently-unmatchable blockers.
        if field_requirements:
            values.extend(field_requirements)
        else:
            values.extend(
                (need.need_name, need.priority)
                for need in plan.information_needs
                if need.need_name
            )
    if not values and state.task_plan:
        values.extend((field, "medium") for field in state.task_plan.target_fields if field)
    deduped: list[tuple[str, str]] = []
    index_by_name: dict[str, int] = {}
    priority_rank = {"low": 0, "medium": 1, "high": 2}
    for name, priority in values:
        key = str(name).strip().casefold()
        if not key:
            continue
        existing_index = index_by_name.get(key)
        if existing_index is None:
            index_by_name[key] = len(deduped)
            deduped.append((str(name).strip(), priority))
            continue
        existing_name, existing_priority = deduped[existing_index]
        if priority_rank[priority] > priority_rank[existing_priority]:
            deduped[existing_index] = (existing_name, priority)
    return deduped


def _record_field_evidence(state: AgentState) -> dict[str, float]:
    counts = _record_field_counts(state)
    total = max(1, len(state.dynamic_records) or len(state.final_records))
    result = {key: min(1.0, value / total) for key, value in counts.items()}

    # Dynamic fields belong to particular output tables. Their denominator
    # must be the records from those tables, not every record from every table.
    # Otherwise a field used in one of four tables can never reach 75% even
    # when it is present in every applicable record.
    plan = state.dynamic_extraction_plan
    if plan:
        field_tables: dict[str, set[str]] = defaultdict(set)
        for table in plan.dynamic_tables:
            for field in table.fields:
                if field.name:
                    field_tables[field.name].add(table.table_name)
        for field_name, table_names in field_tables.items():
            applicable = [
                record
                for record in state.dynamic_records
                if record.table_name in table_names
            ]
            if not applicable:
                result[field_name] = 0.0
                continue
            present = sum(
                1
                for record in applicable
                if any(
                    _labels_match(field_name, key) and value not in (None, "", [], {})
                    for key, value in record.fields.items()
                )
            )
            result[field_name] = min(1.0, present / len(applicable))
    return result


def _record_field_counts(state: AgentState) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in state.dynamic_records:
        for key, value in record.fields.items():
            if value not in (None, "", [], {}):
                counts[str(key).strip()] += 1
    for record in state.final_records:
        if record.metric_name and record.metric_value not in (None, ""):
            counts[str(record.metric_name).strip()] += 1
    return dict(counts)


def _required_evidence_types(state: AgentState) -> list[str]:
    requirements = state.dynamic_extraction_plan.source_requirements if state.dynamic_extraction_plan else []
    result: list[str] = []
    for requirement in requirements:
        value = str(requirement).strip().casefold()
        if any(token in value for token in ("paper", "full_text", "正文")):
            normalized = "paper_full_text"
        elif any(token in value for token in ("table", "表格")):
            normalized = "table"
        elif any(token in value for token in ("supplement", "附件", "补充")):
            normalized = "supplementary_material"
        elif any(token in value for token in ("figure", "chart", "image", "图表")):
            normalized = "figure"
        elif any(token in value for token in ("code", "repository", "代码")):
            normalized = "code_repository"
        elif any(token in value for token in ("dataset", "database", "data", "数据")):
            normalized = "dataset"
        else:
            normalized = value
        if normalized and normalized not in result:
            result.append(normalized)
    normalized_result = [
        evidence_type
        for item in result
        if (evidence_type := _evidence_type_for_label(item))
    ]
    return list(dict.fromkeys(normalized_result))


def _matching_coverage(label: str, values: dict[str, float]) -> float:
    return max(
        (float(value or 0.0) for key, value in values.items() if _labels_match(label, key)),
        default=0.0,
    )


def _matching_count(label: str, values: dict[str, int]) -> int:
    return sum(int(value or 0) for key, value in values.items() if _labels_match(label, key))


def _labels_match(left: str, right: str) -> bool:
    left_tokens = _label_tokens(left)
    right_tokens = _label_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    if left_tokens == right_tokens:
        return True
    if left_tokens <= right_tokens or right_tokens <= left_tokens:
        return True
    shared = left_tokens & right_tokens
    return bool(shared) and any(len(token) >= 6 for token in shared)


def _label_tokens(value: Any) -> set[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    normalized: set[str] = set()
    for token in tokens:
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        normalized.add(token)
    return normalized


def _evidence_types_for_label(label: str) -> list[str]:
    evidence_type = _evidence_type_for_label(label)
    return [evidence_type] if evidence_type else []


def _evidence_type_for_label(value: Any) -> str | None:
    tokens = _label_tokens(value)
    if not tokens:
        return None
    if tokens & {"paper", "article", "publication", "fulltext", "text"}:
        return "paper_full_text"
    if tokens & {"table", "spreadsheet", "csv", "tsv", "excel"}:
        return "table"
    if tokens & {"supplement", "supplementary", "appendix", "supporting"}:
        return "supplementary_material"
    if tokens & {"figure", "chart", "image", "plot", "visual"}:
        return "figure"
    if tokens & {"code", "repository", "repo", "github", "software"}:
        return "code_repository"
    if tokens & {"dataset", "database", "data", "benchmark"}:
        return "dataset"
    return None


def _covered_evidence_types(state: AgentState) -> list[str]:
    result: list[str] = []
    for entry in state.source_catalog:
        for artifact in entry.artifacts:
            if artifact.status != "parsed":
                continue
            for normalized in _artifact_evidence_types(artifact.artifact_type):
                if normalized not in result:
                    result.append(normalized)
    if state.parsed_sources.tables and "table" not in result:
        result.append("table")
    if state.parsed_sources.figure_assets and "figure" not in result:
        result.append("figure")
    if state.parsed_sources.text_blocks and "paper_full_text" not in result:
        result.append("paper_full_text")
    return result


def _unprocessed_relevant_artifacts(state: AgentState) -> list[str]:
    result: list[str] = []
    for entry in state.source_catalog:
        for artifact in entry.artifacts:
            if artifact.relevance_score is not None and artifact.relevance_score >= 3.0:
                if not state.runtime_auto_download_sources and not artifact.local_path:
                    continue
                # Metadata inspection is not content extraction.  Keep a
                # downloadable/local high-relevance artifact pending until its
                # content is parsed; completed_operations prevents repeating
                # the same metadata action while the planner chooses download
                # or a type-specific parser next.
                metadata_url = next(
                    (
                        artifact.metadata.get(key)
                        for key in ("download_url", "source_url", "html_url", "url")
                        if artifact.metadata.get(key)
                    ),
                    None,
                )
                if artifact.status == "inspected" and not (
                    artifact.local_path or artifact.url or metadata_url
                ):
                    continue
                if artifact.status not in {"parsed", "failed", "skipped"}:
                    result.append(artifact.artifact_id)
    return result
