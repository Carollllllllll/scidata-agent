from __future__ import annotations

from collections import defaultdict
import hashlib
import re
from typing import Any

from scidata_agent.agent.schemas import AgentState, CoverageGap, CoverageItem, CoverageReport


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
        )
        items.append(item)
        weight = {"high": 3.0, "medium": 2.0, "low": 1.0}[priority]
        weighted_total += weight
        weighted_covered += weight * min(1.0, coverage)
        if status in {"missing", "partial"} and priority != "low":
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

    reasons: list[str] = []
    actions: list[str] = []
    if missing:
        reasons.append(f"Required fields are missing or partial: {', '.join(missing)}.")
        actions.extend(["download_artifact", "parse_pdf_sections", "parse_table", "parse_figure"])
    if missing_evidence:
        reasons.append(f"Required evidence types are not covered: {', '.join(missing_evidence)}.")
        actions.extend(["download_artifact", "search_more"])
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
    if not items and not covered_evidence and state.source_catalog:
        reasons.append("The catalog exists, but no content evidence has been parsed yet.")
        actions.extend(["download_artifact", "parse_pdf_sections", "read_metadata"])
        gaps.append(
            CoverageGap(
                gap_id="content_evidence_not_started",
                requirement_name="Content evidence",
                priority="high",
                status="missing",
                reason="The source catalog exists, but no content evidence has been parsed yet.",
                recommended_actions=["download_artifact", "parse_pdf_sections", "read_metadata"],
            )
        )

    blocking_gaps = [gap for gap in gaps if gap.priority != "low"]
    decision = "allow_stop" if not blocking_gaps else "continue"
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
        reasons=reasons,
        recommended_actions=list(dict.fromkeys(actions)),
    )


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
    if not actions:
        actions.append("search_more")
    return list(dict.fromkeys(actions))


def _requirement_unavailable(state: AgentState, evidence_types: list[str]) -> bool:
    return bool(evidence_types) and all(
        _evidence_type_unavailable(state, evidence_type) for evidence_type in evidence_types
    )


def _evidence_type_unavailable(state: AgentState, evidence_type: str) -> bool:
    candidates = [
        artifact
        for entry in state.source_catalog
        for artifact in entry.artifacts
        if _artifact_evidence_type(artifact.artifact_type) == evidence_type
    ]
    return bool(candidates) and all(artifact.status in {"failed", "skipped"} for artifact in candidates)


def _artifact_evidence_type(artifact_type: str) -> str | None:
    if artifact_type in {"pdf", "html"}:
        return "paper_full_text"
    if artifact_type == "supplementary_pdf":
        return "supplementary_material"
    if artifact_type in {"csv", "tsv", "xlsx", "json", "xml"}:
        return "table"
    if artifact_type == "image":
        return "figure"
    if artifact_type in {"readme", "code_archive", "file_manifest"}:
        return "code_repository"
    return None


def _requirement_specs(state: AgentState) -> list[tuple[str, str]]:
    plan = state.dynamic_extraction_plan
    values: list[tuple[str, str]] = []
    if plan:
        values.extend((need.need_name, need.priority) for need in plan.information_needs if need.need_name)
        for table in plan.dynamic_tables:
            for field in table.fields:
                if field.evidence_required and field.name:
                    # evidence_required is an evidence contract, not a
                    # presence contract. Optional fields may legitimately be
                    # absent from a source; report their gap without blocking
                    # completion. Only explicitly required fields block.
                    values.append((field.name, "high" if field.required else "low"))
    if not values and state.task_plan:
        values.extend((field, "medium") for field in state.task_plan.target_fields if field)
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, priority in values:
        key = str(name).strip().casefold()
        if key and key not in seen:
            deduped.append((str(name).strip(), priority))
            seen.add(key)
    return deduped


def _record_field_evidence(state: AgentState) -> dict[str, float]:
    counts = _record_field_counts(state)
    total = max(1, len(state.dynamic_records) or len(state.final_records))
    return {key: min(1.0, value / total) for key, value in counts.items()}


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
            if artifact.artifact_type in {"pdf", "html"}:
                normalized = "paper_full_text"
            elif artifact.artifact_type == "supplementary_pdf":
                normalized = "supplementary_material"
            elif artifact.artifact_type in {"csv", "tsv", "xlsx", "json", "xml"}:
                normalized = "table"
            elif artifact.artifact_type in {"image"}:
                normalized = "figure"
            elif artifact.artifact_type in {"readme", "code_archive"}:
                normalized = "code_repository"
            else:
                continue
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
                if artifact.status not in {"parsed", "failed", "skipped"}:
                    result.append(artifact.artifact_id)
    return result
