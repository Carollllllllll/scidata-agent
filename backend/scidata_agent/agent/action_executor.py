from __future__ import annotations

import html
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any

from scidata_agent.agent.action_registry import artifact_type_supported, get_action_capability, is_global_action
from scidata_agent.agent.schemas import (
    AgentState,
    ArtifactAction,
    ArtifactActionPlan,
    ArtifactActionResult,
    FigureAsset,
    SourceArtifact,
    SourceInsight,
    SourceType,
    TextBlock,
    UploadedFile,
)
from scidata_agent.llm.nodes import QwenAgentNodes
from scidata_agent.tools.connectors.registry import execute_multi_source_search, merge_sources


class ArtifactActionExecutor:
    """Execute validated artifact actions against the current AgentState.

    The executor is intentionally bounded to one plan. It does not decide what
    to do next and it does not run a hidden fallback. The planner or outer
    workflow owns iteration; this class only routes actions to existing tools.
    """

    def __init__(self, llm_nodes: QwenAgentNodes | None = None):
        self.llm_nodes = llm_nodes

    def execute_plan(
        self,
        plan: ArtifactActionPlan,
        state: AgentState,
        *,
        max_pdf_pages: int | None = None,
        max_figures_per_action: int = 6,
    ) -> list[ArtifactActionResult]:
        results: list[ArtifactActionResult] = []
        for action in plan.actions:
            result = self.execute_action(
                action,
                state,
                max_pdf_pages=max_pdf_pages,
                max_figures_per_action=max_figures_per_action,
            )
            results.append(result)
            state.processing_log.append(
                f"Artifact action {action.action_id}: action={action.action}, "
                f"status={result.status}, artifact_id={action.artifact_id or 'global'}, "
                f"message={result.message}"
            )
            if action.action == "stop":
                break
        return results

    def execute_action(
        self,
        action: ArtifactAction,
        state: AgentState,
        *,
        max_pdf_pages: int | None = None,
        max_figures_per_action: int = 6,
    ) -> ArtifactActionResult:
        try:
            capability = get_action_capability(action.action)
        except ValueError as exc:
            return self._failed(action, str(exc))

        if is_global_action(action.action):
            return self._execute_global(action, state)

        artifact = self._find_artifact(action.artifact_id, state)
        if artifact is None:
            return self._failed(action, f"Artifact not found: {action.artifact_id!r}")

        if action.action in {
            "parse_pdf_text",
            "parse_pdf_sections",
            "parse_table",
            "parse_csv",
            "parse_figure",
            "parse_html",
            "read_readme",
        } and artifact.status == "parsed":
            return self._skipped(
                action,
                "Artifact content was already parsed; duplicate parsing was skipped.",
            )

        effective_type = _effective_artifact_type(artifact)
        if not artifact_type_supported(action.action, effective_type):
            return self._skipped(
                action,
                f"Action {action.action!r} does not support artifact_type={effective_type!r}.",
            )
        if capability.requires_local_path:
            path_result = self._require_local_file(action, artifact)
            if path_result is not None:
                return path_result

        try:
            if action.action == "download_artifact":
                return self._download_artifact(action, state, artifact)
            if action.action == "read_metadata":
                return self._read_metadata(action, state, artifact)
            if action.action == "parse_pdf_text":
                return self._parse_pdf_text(action, state, artifact, max_pdf_pages)
            if action.action == "parse_pdf_sections":
                return self._parse_pdf_sections(action, state, artifact, max_pdf_pages)
            if action.action == "parse_table":
                return self._parse_table(action, state, artifact, effective_type, max_pdf_pages)
            if action.action == "parse_csv":
                return self._parse_csv(action, state, artifact)
            if action.action == "parse_figure":
                return self._parse_figure(action, state, artifact, effective_type, max_pdf_pages, max_figures_per_action)
            if action.action in {"parse_html", "read_readme"}:
                return self._read_text_artifact(action, state, artifact)
            if action.action == "read_file_manifest":
                return self._read_file_manifest(action, state, artifact)
            return self._failed(action, f"No executor handler registered for {action.action!r}.")
        except Exception as exc:
            return self._failed(action, f"Artifact action failed: {exc}", error=repr(exc))

    def _execute_global(self, action: ArtifactAction, state: AgentState) -> ArtifactActionResult:
        if action.action == "stop":
            return self._result(action, "no_op", "Planner requested workflow stop.")
        if action.action == "search_more":
            return self._search_more(action, state)
        if action.action == "validate_evidence":
            return self._validate_evidence(action, state)
        return self._failed(action, f"Unsupported global action: {action.action!r}")

    def _search_more(self, action: ArtifactAction, state: AgentState) -> ArtifactActionResult:
        if self.llm_nodes is None:
            return self._failed(action, "search_more requires an LLM node to create a new search plan.")
        if state.source_discovery_plan is None:
            return self._failed(action, "search_more requires an existing source discovery plan.")

        try:
            candidates = list(state.source_discovery_plan.candidate_sources)
            batch_size = _positive_env_int("SCIDATA_SEARCH_MORE_BATCH_SIZE", 40)
            batches = [
                candidates[start : start + batch_size]
                for start in range(0, len(candidates), batch_size)
            ] or [[]]
            plans = []
            planning_errors: list[str] = []
            for index, batch in enumerate(batches, start=1):
                batch_plan = state.source_discovery_plan.model_copy(update={"candidate_sources": batch})
                try:
                    plans.append(
                        _plan_search_batch(
                            self.llm_nodes,
                            state.research_question,
                            batch_plan,
                            candidate_context_limit=batch_size,
                            batch_label=f"batch {index}/{len(batches)}",
                        )
                    )
                except Exception as exc:
                    planning_errors.append(f"batch {index}/{len(batches)}: {exc}")
            if not plans:
                raise RuntimeError(
                    "All search_more planning batches failed: " + "; ".join(planning_errors)
                )
            plan = _merge_search_plans(plans)
            if planning_errors:
                plan.notes.append(
                    f"{len(planning_errors)} search-planning batch(es) failed; successful batches were executed."
                )
                state.processing_log.extend(
                    f"search_more planning warning: {error}" for error in planning_errors
                )
            state.multi_source_search_plan = plan
            search_kwargs: dict[str, Any] = {}
            try:
                parameters = inspect.signature(execute_multi_source_search).parameters
                accepts_kwargs = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
                if "cache_dir" in parameters or accepts_kwargs:
                    search_kwargs["cache_dir"] = state.output_dir / "_cache" / "source_search"
            except (TypeError, ValueError):
                # Some test doubles or extension callables do not expose a signature.
                # Keep the legacy one-argument call in that case.
                pass
            found, status = execute_multi_source_search(plan, **search_kwargs)
            merged, added = merge_sources(
                state.source_discovery_plan.candidate_sources,
                found,
            )
            state.source_discovery_plan.candidate_sources = merged
            state.connector_status.extend(status.get("connector_status", []))
            state.source_discovery_plan.notes.append(
                f"Artifact search_more status: {status}."
            )
            state.processing_log.append(
                "Artifact search_more completed: "
                f"requests={status.get('searched', 0)}, new_sources={added}, "
                f"failed_requests={status.get('failed', 0)}, status={status.get('status')}."
            )
            return self._result(
                action,
                "completed",
                "LLM generated and executed a new multi-source search plan.",
                search_requests=len(plan.search_requests),
                new_sources=added,
                failed_requests=int(status.get("failed", 0)),
                planning_batches=len(batches),
                failed_planning_batches=len(planning_errors),
            )
        except Exception as exc:
            return self._failed(action, f"search_more failed: {exc}", error=repr(exc))

    def _validate_evidence(self, action: ArtifactAction, state: AgentState) -> ArtifactActionResult:
        if self.llm_nodes is None:
            return self._failed(action, "validate_evidence requires an LLM node.")
        if not state.final_records:
            return self._skipped(
                action,
                "Evidence validation was requested before final records existed; the quality node will validate later.",
            )
        try:
            from scidata_agent.tools.quality import build_quality_report

            llm_issues = self.llm_nodes.validate_records(state.final_records)
            target_fields = state.task_plan.target_fields if state.task_plan else None
            state.quality_report = build_quality_report(
                state.final_records,
                llm_issues=llm_issues,
                target_fields=target_fields,
                mutate_records=True,
            )
            state.processing_log.append(
                "Artifact evidence validation completed: "
                f"records={state.quality_report.record_count}, "
                f"issues={state.quality_report.issue_count}, "
                f"conflicts={state.quality_report.conflict_count}."
            )
            return self._result(
                action,
                "completed",
                "Validated final records with LLM and deterministic quality checks.",
                records=state.quality_report.record_count,
                issues=state.quality_report.issue_count,
                conflicts=state.quality_report.conflict_count,
            )
        except Exception as exc:
            return self._failed(action, f"validate_evidence failed: {exc}", error=repr(exc))

    def _parse_pdf_text(
        self, action: ArtifactAction, state: AgentState, artifact: SourceArtifact, max_pdf_pages: int | None
    ) -> ArtifactActionResult:
        from scidata_agent.tools.parser import parse_pdf

        uploaded = _uploaded_file(artifact)
        blocks = parse_pdf(uploaded, max_pages=max_pdf_pages)
        state.parsed_sources.text_blocks.extend(blocks)
        artifact.status = "parsed"
        artifact.parser = "pdf_text"
        return self._result(
            action,
            "completed",
            f"Parsed PDF text from {uploaded.filename}.",
            text_blocks=len(blocks),
        )

    def _download_artifact(
        self,
        action: ArtifactAction,
        state: AgentState,
        artifact: SourceArtifact,
    ) -> ArtifactActionResult:
        from scidata_agent.tools.source_ingestion import download_source_artifact

        target_dir = state.output_dir / state.task_id / "downloads" / "artifacts"
        raw_limit = action.parameters.get("max_bytes")
        max_bytes = int(raw_limit) if raw_limit not in (None, "") else None
        try:
            path = download_source_artifact(
                artifact,
                target_dir,
                max_bytes=max_bytes,
            )
        except Exception as exc:
            artifact.status = "failed"
            artifact.failure_reason = str(exc)
            return self._failed(action, f"Artifact download failed: {exc}", error=repr(exc))

        source = self._find_discovered_source(artifact.source_id, state)
        if source is not None:
            paths = source.metadata.setdefault("downloaded_paths", [])
            if str(path) not in paths:
                paths.append(str(path))
            source.metadata["downloaded_path"] = source.metadata.get("downloaded_path") or str(path)
            downloads = source.metadata.setdefault("downloaded_artifacts", {})
            if artifact.url:
                downloads[artifact.url] = str(path)

        return self._result(
            action,
            "completed",
            f"Downloaded artifact {artifact.name or artifact.artifact_id}.",
            bytes=artifact.size_bytes or 0,
        )

    def _parse_pdf_sections(
        self, action: ArtifactAction, state: AgentState, artifact: SourceArtifact, max_pdf_pages: int | None
    ) -> ArtifactActionResult:
        from scidata_agent.tools.parser import (
            build_section_blocks_from_plan,
            extract_heading_candidates,
            parse_pdf,
        )

        if self.llm_nodes is None:
            return self._failed(action, "parse_pdf_sections requires an LLM node for section interpretation.")
        uploaded = _uploaded_file(artifact)
        blocks = parse_pdf(uploaded, max_pages=max_pdf_pages)
        headings = extract_heading_candidates(uploaded, blocks, max_pages=max_pdf_pages)
        section_plan = self.llm_nodes.interpret_sections(state.research_question, headings)
        section_blocks = build_section_blocks_from_plan(blocks, section_plan)
        state.parsed_sources.text_blocks.extend(blocks)
        state.parsed_sources.heading_candidates.extend(headings)
        state.parsed_sources.section_plan = section_plan
        state.parsed_sources.section_blocks.extend(section_blocks)
        artifact.status = "parsed"
        artifact.parser = "pdf_sections"
        return self._result(
            action,
            "completed",
            f"Parsed PDF text and interpreted sections from {uploaded.filename}.",
            text_blocks=len(blocks),
            heading_candidates=len(headings),
            section_blocks=len(section_blocks),
        )

    def _parse_table(
        self,
        action: ArtifactAction,
        state: AgentState,
        artifact: SourceArtifact,
        artifact_type: str,
        max_pdf_pages: int | None,
    ) -> ArtifactActionResult:
        from scidata_agent.tools.parser import parse_csv, parse_excel, parse_pdf_tables

        uploaded = _uploaded_file(artifact)
        if artifact_type in {"pdf", "supplementary_pdf"}:
            tables = parse_pdf_tables(uploaded, max_pages=max_pdf_pages)
        elif artifact_type in {"csv", "tsv"}:
            tables = [parse_csv(uploaded)]
        elif artifact_type in {"xlsx"}:
            tables = parse_excel(uploaded)
        else:
            return self._skipped(action, f"No structured parser is registered for {artifact_type!r}.")
        state.parsed_sources.tables.extend(tables)
        artifact.status = "parsed"
        artifact.parser = "table_parser"
        return self._result(action, "completed", f"Parsed {len(tables)} table(s) from {uploaded.filename}.", tables=len(tables))

    def _parse_csv(self, action: ArtifactAction, state: AgentState, artifact: SourceArtifact) -> ArtifactActionResult:
        from scidata_agent.tools.parser import parse_csv

        uploaded = _uploaded_file(artifact)
        table = parse_csv(uploaded)
        state.parsed_sources.tables.append(table)
        artifact.status = "parsed"
        artifact.parser = "csv_parser"
        return self._result(action, "completed", f"Parsed CSV/TSV table from {uploaded.filename}.", tables=1, rows=len(table.rows))

    def _parse_figure(
        self,
        action: ArtifactAction,
        state: AgentState,
        artifact: SourceArtifact,
        artifact_type: str,
        max_pdf_pages: int | None,
        max_figures: int,
    ) -> ArtifactActionResult:
        from scidata_agent.tools.chart_locator import locate_figures
        from scidata_agent.tools.chart_validator import validate_chart_extraction

        if self.llm_nodes is None:
            return self._failed(action, "parse_figure requires Qwen-VL through QwenAgentNodes.")
        uploaded = _uploaded_file(artifact)
        if artifact_type in {"pdf", "supplementary_pdf"}:
            figures_dir = state.output_dir / state.task_id / "figures"
            assets = locate_figures(
                uploaded,
                figures_dir,
                max_pages=max_pdf_pages,
                max_figures=max_figures,
            )
        else:
            assets = [
                FigureAsset(
                    source_file=uploaded.filename,
                    source_path=str(uploaded.path),
                    page=1,
                    image_path=str(uploaded.path),
                    detection_method="artifact",
                )
            ]
        state.parsed_sources.figure_assets.extend(assets)
        artifact.status = "parsed"
        artifact.parser = "figure_parser"
        extracted = 0
        skipped = 0
        for figure in assets:
            classification = self.llm_nodes.classify_chart(figure)
            if not classification.get("contains_data"):
                skipped += 1
                continue
            chart = self.llm_nodes.extract_chart_data(figure, classification.get("chart_type", "unknown"))
            state.chart_extractions.append(chart)
            state.chart_validations.append(validate_chart_extraction(chart, figure))
            extracted += 1
        return self._result(
            action,
            "completed",
            f"Processed {len(assets)} figure asset(s) from {uploaded.filename}; quantitative charts={extracted}.",
            figures=len(assets),
            charts_extracted=extracted,
            non_data_figures=skipped,
        )

    def _read_text_artifact(
        self, action: ArtifactAction, state: AgentState, artifact: SourceArtifact
    ) -> ArtifactActionResult:
        uploaded = _uploaded_file(artifact)
        text = uploaded.path.read_text(encoding="utf-8", errors="ignore")
        if action.action == "parse_html":
            text = _html_to_text(text)
        block = TextBlock(
            source_file=uploaded.filename,
            source_path=str(uploaded.path),
            source_type=SourceType.UNKNOWN,
            page=None,
            text=text[:200000],
            chunk_id=f"{artifact.artifact_id}_text",
        )
        state.parsed_sources.text_blocks.append(block)
        artifact.status = "parsed"
        artifact.parser = "text_parser"
        return self._result(action, "completed", f"Read text artifact {uploaded.filename}.", text_blocks=1, characters=len(block.text))

    def _read_file_manifest(
        self, action: ArtifactAction, state: AgentState, artifact: SourceArtifact
    ) -> ArtifactActionResult:
        entry = self._find_source_entry(artifact.source_id, state)
        manifest = artifact.metadata or (entry.metadata if entry else {})
        content = json.dumps(manifest, ensure_ascii=False, indent=2)
        if entry:
            state.source_insights.append(
                SourceInsight(
                    source_id=entry.source_id,
                    title=entry.title,
                    provider=entry.provider,
                    source_type=entry.source_type,
                    insight_type="file_manifest",
                    content=content,
                    url=artifact.url or entry.url,
                    confidence=entry.relevance_score,
                )
            )
        return self._result(action, "completed", "Read file manifest metadata.", manifest_items=len(manifest))

    def _read_metadata(
        self, action: ArtifactAction, state: AgentState, artifact: SourceArtifact
    ) -> ArtifactActionResult:
        entry = self._find_source_entry(artifact.source_id, state)
        metadata = artifact.model_dump(mode="json")
        if entry:
            metadata["source_title"] = entry.title
            metadata["source_type"] = entry.source_type
            state.source_insights.append(
                SourceInsight(
                    source_id=entry.source_id,
                    title=entry.title,
                    provider=entry.provider,
                    source_type=entry.source_type,
                    insight_type="metadata",
                    content=json.dumps(metadata, ensure_ascii=False, indent=2),
                    url=artifact.url or entry.url,
                    confidence=entry.relevance_score,
                )
            )
        return self._result(action, "completed", "Read catalog metadata without downloading.", metadata_fields=len(metadata))

    def _find_artifact(self, artifact_id: str | None, state: AgentState) -> SourceArtifact | None:
        if not artifact_id:
            return None
        for entry in state.source_catalog:
            for artifact in entry.artifacts:
                if artifact.artifact_id == artifact_id:
                    return artifact
        return None

    def _find_source_entry(self, source_id: str, state: AgentState):
        return next((entry for entry in state.source_catalog if entry.source_id == source_id), None)

    def _find_discovered_source(self, source_id: str, state: AgentState):
        if not state.source_discovery_plan:
            return None
        return next(
            (source for source in state.source_discovery_plan.candidate_sources if source.source_id == source_id),
            None,
        )

    def _require_local_file(self, action: ArtifactAction, artifact: SourceArtifact) -> ArtifactActionResult | None:
        if not artifact.local_path:
            return self._skipped(action, "The selected artifact has no local_path; discovery/ingestion must materialize it first.")
        path = Path(artifact.local_path).expanduser().resolve()
        if not path.exists():
            return self._failed(action, f"Artifact local_path does not exist: {path}")
        if not path.is_file():
            return self._failed(action, f"Artifact local_path is not a file: {path}")
        return None

    def _result(self, action: ArtifactAction, status: str, message: str, **counts: int) -> ArtifactActionResult:
        return ArtifactActionResult(
            action_id=action.action_id,
            artifact_id=action.artifact_id,
            action=action.action,
            status=status,  # type: ignore[arg-type]
            message=message,
            output_counts={key: int(value) for key, value in counts.items()},
        )

    def _skipped(self, action: ArtifactAction, message: str) -> ArtifactActionResult:
        return self._result(action, "skipped", message)

    def _failed(self, action: ArtifactAction, message: str, error: str | None = None) -> ArtifactActionResult:
        result = self._result(action, "failed", message)
        result.error = error or message
        return result


def _uploaded_file(artifact: SourceArtifact) -> UploadedFile:
    if not artifact.local_path:
        raise ValueError(f"Artifact {artifact.artifact_id!r} has no local_path.")
    path = Path(artifact.local_path).expanduser().resolve()
    return UploadedFile(
        filename=path.name,
        path=path,
        content_type=artifact.content_type,
    )


def _effective_artifact_type(artifact: SourceArtifact) -> str:
    if artifact.artifact_type != "unknown":
        return artifact.artifact_type
    if artifact.local_path:
        suffix = Path(artifact.local_path).suffix.lower().lstrip(".")
        if suffix in {"pdf", "csv", "tsv", "xlsx", "json", "xml", "html"}:
            return suffix
    return artifact.artifact_type


def _html_to_text(value: str) -> str:
    value = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    value = re.sub(r"(?s)<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _plan_search_batch(
    llm_nodes: Any,
    research_question: str,
    source_discovery_plan: Any,
    *,
    candidate_context_limit: int,
    batch_label: str,
) -> Any:
    """Call the bounded planner while preserving compatibility with old adapters.

    ``search_more`` is an extension point: tests and downstream integrations may
    still expose the original two-argument planner. Inspecting the bound method
    signature avoids catching a real ``TypeError`` raised inside the LLM node.
    """
    planner = llm_nodes.plan_multi_source_search
    try:
        parameters = inspect.signature(planner).parameters
    except (TypeError, ValueError):
        parameters = {}
    supports_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    planner_kwargs = {}
    if supports_kwargs or "candidate_context_limit" in parameters:
        planner_kwargs["candidate_context_limit"] = candidate_context_limit
    if supports_kwargs or "batch_label" in parameters:
        planner_kwargs["batch_label"] = batch_label
    if planner_kwargs:
        return planner(research_question, source_discovery_plan, **planner_kwargs)
    return planner(research_question, source_discovery_plan)


def _merge_search_plans(plans: list[Any]) -> Any:
    """Combine successful batch plans without repeating identical searches."""
    if not plans:
        raise ValueError("At least one search plan is required.")
    requests = []
    seen: set[tuple[str, str, str]] = set()
    criteria: list[str] = []
    notes: list[str] = []
    for plan in plans:
        for request in plan.search_requests:
            key = (
                str(request.connector_name).casefold(),
                str(request.source_type).casefold(),
                " ".join(str(request.query).split()).casefold(),
            )
            if key not in seen:
                seen.add(key)
                requests.append(request)
        for value in [*plan.selection_criteria, *plan.notes]:
            if value and value not in criteria and value not in notes:
                if value in plan.selection_criteria:
                    criteria.append(value)
                else:
                    notes.append(value)
    return plans[0].model_copy(
        update={
            "should_search": any(plan.should_search for plan in plans),
            "search_requests": requests,
            "selection_criteria": criteria,
            "notes": notes,
        }
    )
