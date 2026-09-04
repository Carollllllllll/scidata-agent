from __future__ import annotations

import hashlib
import html
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

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
from scidata_agent.agent.tool_protocol import ToolCall, ToolResult
from scidata_agent.agent.tool_registry import build_artifact_tool_registry
from scidata_agent.agent.tool_runtime import ToolRuntime
from scidata_agent.llm.nodes import QwenAgentNodes
from scidata_agent.tools.connectors.registry import execute_multi_source_search, merge_sources


SEARCH_MORE_CANDIDATE_LIMIT = 100
SEARCH_MORE_MAX_PLANNING_BATCHES = 3


class ArtifactActionExecutor:
    """Execute validated artifact actions against the current AgentState.

    The executor is intentionally bounded to one plan. It does not decide what
    to do next and it does not run a hidden fallback. The planner or outer
    workflow owns iteration; this class routes artifact actions and, when
    supplied, outer workflow actions to existing tools.
    """

    def __init__(
        self,
        llm_nodes: QwenAgentNodes | None = None,
        workflow_handler: Callable[[ArtifactAction, AgentState], ArtifactActionResult] | None = None,
    ):
        self.llm_nodes = llm_nodes
        self.workflow_handler = workflow_handler
        self.tool_registry = build_artifact_tool_registry()
        self.tool_runtime = ToolRuntime(self.tool_registry, handler=self._handle_tool_call)

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
            tool_result = self.tool_runtime.execute(
                _tool_call_from_action(
                    action,
                    workflow_revision=state.workflow_revision,
                    parsed_content_fingerprint=workflow_stage_fingerprint(
                        state,
                        action.action,
                    ),
                ),
                context=state,
                options={
                    "max_pdf_pages": max_pdf_pages,
                    "max_figures_per_action": max_figures_per_action,
                },
            )
            result = _artifact_result_from_tool_result(action, tool_result)
            results.append(result)
            state.processing_log.append(
                f"Artifact action {action.action_id}: action={action.action}, "
                f"status={result.status}, artifact_id={action.artifact_id or 'global'}, "
                f"message={result.message}"
            )
            if action.action == "stop":
                break
        return results

    def _handle_tool_call(
        self,
        call: ToolCall,
        state: AgentState,
        options: dict[str, Any],
    ) -> ToolResult:
        """Bridge the new tool protocol to the legacy action implementation."""
        action = _action_from_tool_call(call)
        legacy_result = self.execute_action(
            action,
            state,
            max_pdf_pages=options.get("max_pdf_pages"),
            max_figures_per_action=int(options.get("max_figures_per_action", 6)),
        )
        tool_result = _tool_result_from_action_result(legacy_result, call)
        # search_more may advance the state revision while it executes.  Store
        # the result in the revision whose downstream work it created.
        tool_result.workflow_revision = max(0, int(state.workflow_revision))
        return tool_result

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

        if action.action in artifact.completed_operations:
            return self._skipped(
                action,
                f"Artifact operation {action.action!r} was already completed; duplicate work was skipped.",
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
        if self.workflow_handler is not None and action.action in {
            "plan_task",
            "plan_dynamic_schema",
            "discover_sources",
            "plan_multi_source_search",
            "search_sources",
            "select_sources",
            "triage_sources",
            "ingest_sources",
            "ingest_arxiv_pdfs",
            "parse_content",
            "parse_source_content",
            "extract_figures",
            "interpret_sections",
            "extract_dynamic_records",
            "extract_records",
            "normalize_records",
            "track_provenance",
            "validate_quality",
        }:
            return self.workflow_handler(action, state)
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
        search_strategy = _normalize_search_strategy(action.parameters)
        field_group_id = str(search_strategy.get("field_group_id") or "").strip().casefold()
        initial_group_search = bool(search_strategy.get("initial_group_search"))
        limit = int(getattr(state, "runtime_search_more_limit", 2))
        count = (
            int(state.runtime_group_search_more_counts.get(field_group_id, 0))
            if field_group_id
            else int(getattr(state, "runtime_search_more_count", 0))
        )
        if not initial_group_search and count >= limit:
            return self._skipped(
                action,
                "search_more limit exhausted; supplemental search was not executed "
                f"(field_group={field_group_id or 'legacy_global'}, limit={limit}).",
            )
        # Count attempts, including failed connector/planning attempts, so a
        # permanently unavailable search cannot consume the runtime forever.
        if initial_group_search and field_group_id:
            state.runtime_group_initial_searches = sorted(
                set(state.runtime_group_initial_searches).union({field_group_id})
            )
            state.processing_log.append(
                f"Initial retrieval attempt for field_group={field_group_id}."
            )
        else:
            state.runtime_search_more_count = int(state.runtime_search_more_count) + 1
            if field_group_id:
                state.runtime_group_search_more_counts[field_group_id] = count + 1
            state.processing_log.append(
                "search_more attempt "
                f"{count + 1}/{limit} for field_group={field_group_id or 'legacy_global'}."
            )

        try:
            # Supplemental search is intentionally a focused recovery action,
            # not a second full-catalog planning pass.  Bound both the input
            # candidates and LLM planning batches so one weak field group
            # cannot stall a task by re-planning thousands of sources.
            candidate_limit = min(
                _positive_env_int(
                    "SCIDATA_SEARCH_MORE_CANDIDATE_LIMIT",
                    SEARCH_MORE_CANDIDATE_LIMIT,
                ),
                SEARCH_MORE_CANDIDATE_LIMIT,
            )
            max_batches = min(
                _positive_env_int(
                    "SCIDATA_SEARCH_MORE_MAX_PLANNING_BATCHES",
                    SEARCH_MORE_MAX_PLANNING_BATCHES,
                ),
                SEARCH_MORE_MAX_PLANNING_BATCHES,
            )
            candidates = list(state.source_discovery_plan.candidate_sources)[:candidate_limit]
            batch_size = _positive_env_int("SCIDATA_SEARCH_MORE_BATCH_SIZE", 40)
            batches = ([
                candidates[start : start + batch_size]
                for start in range(0, len(candidates), batch_size)
            ][:max_batches] or [[]])
            state.processing_log.append(
                "search_more planning scope: "
                f"candidates={len(candidates)}/{candidate_limit}, batches={len(batches)}/{max_batches}."
            )
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
                            search_strategy=search_strategy,
                        )
                    )
                except Exception as exc:
                    planning_errors.append(f"batch {index}/{len(batches)}: {exc}")
            if not plans:
                raise RuntimeError(
                    "All search_more planning batches failed: " + "; ".join(planning_errors)
                )
            plan = _merge_search_plans(plans)
            plan = _apply_search_strategy(plan, search_strategy)
            if plan.should_search and not plan.search_requests:
                raise RuntimeError(
                    "Dynamic search strategy removed all search requests; choose another connector or source type."
                )
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
            if added > 0:
                # Downstream selection/triage/extraction must see the enlarged
                # source set.  Keep old results in history, but move subsequent
                # calls into a fresh idempotency namespace.
                state.workflow_revision += 1
                state.source_selection_plan = None
                state.source_triage_decisions = []
            state.connector_status.extend(status.get("connector_status", []))
            state.source_discovery_plan.notes.append(
                f"Artifact search_more status: {status}."
            )
            state.processing_log.append(
                "Artifact search_more completed: "
                f"requests={status.get('searched', 0)}, new_sources={added}, "
                f"failed_requests={status.get('failed', 0)}, status={status.get('status')}."
            )
            search_status = str(status.get("status") or "completed")
            result_status = (
                "failed"
                if search_status == "failed"
                else "partial"
                if search_status == "partial"
                else "completed"
            )
            output_counts = {
                "search_requests": len(plan.search_requests),
                "new_sources": added,
                "failed_requests": int(status.get("failed", 0)),
                "planning_batches": len(batches),
                "failed_planning_batches": len(planning_errors),
            }
            # Keep the established zero-candidate result shape stable for
            # integrations that compare the action counters exactly.  The
            # diagnostic is meaningful once a bounded candidate pool exists
            # (and remains available for cap regression tests).
            if candidates:
                output_counts["planning_candidates"] = len(candidates)
            return self._result(
                action,
                result_status,
                (
                    "LLM generated and executed a new multi-source search plan"
                    f" with status={search_status}."
                ),
                **output_counts,
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
        _mark_operation_completed(artifact, action.action)
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
        from scidata_agent.tools.source_ingestion import (
            download_source_artifact,
            unsupported_materialized_format_reason,
        )

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

        unsupported_reason = unsupported_materialized_format_reason(
            path,
            artifact.artifact_type,
        )
        if unsupported_reason:
            artifact.status = "skipped"
            artifact.parser = "unsupported_format"
            artifact.failure_reason = unsupported_reason
            return self._result(
                action,
                "skipped",
                f"Downloaded artifact was skipped: {unsupported_reason}.",
                bytes=artifact.size_bytes or 0,
                files_delta=0,
            )

        files_delta = 0
        resolved_path = path.expanduser().resolve()
        if not any(
            uploaded.path.expanduser().resolve() == resolved_path
            for uploaded in state.files
        ):
            state.files.append(
                UploadedFile(
                    filename=resolved_path.name,
                    path=resolved_path,
                    content_type=artifact.content_type,
                )
            )
            files_delta = 1

        return self._result(
            action,
            "completed",
            f"Downloaded artifact {artifact.name or artifact.artifact_id}.",
            bytes=artifact.size_bytes or 0,
            files_delta=files_delta,
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
        _mark_operation_completed(artifact, action.action)
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
        _mark_operation_completed(artifact, action.action)
        return self._result(action, "completed", f"Parsed {len(tables)} table(s) from {uploaded.filename}.", tables=len(tables))

    def _parse_csv(self, action: ArtifactAction, state: AgentState, artifact: SourceArtifact) -> ArtifactActionResult:
        from scidata_agent.tools.parser import parse_csv

        uploaded = _uploaded_file(artifact)
        table = parse_csv(uploaded)
        state.parsed_sources.tables.append(table)
        artifact.status = "parsed"
        artifact.parser = "csv_parser"
        _mark_operation_completed(artifact, action.action)
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
        _mark_operation_completed(artifact, action.action)
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
        _mark_operation_completed(artifact, action.action)
        return self._result(action, "completed", f"Read text artifact {uploaded.filename}.", text_blocks=1, characters=len(block.text))

    def _read_file_manifest(
        self, action: ArtifactAction, state: AgentState, artifact: SourceArtifact
    ) -> ArtifactActionResult:
        entry = self._find_source_entry(artifact.source_id, state)
        manifest = artifact.metadata or (entry.metadata if entry else {})
        content = json.dumps(manifest, ensure_ascii=False, indent=2)
        local_text: str | None = None
        parsed_html = False
        if artifact.local_path:
            uploaded = _uploaded_file(artifact)
            local_text = uploaded.path.read_text(encoding="utf-8", errors="ignore")[:200000]
            looks_like_html = (
                uploaded.path.suffix.lower() in {".html", ".htm"}
                or artifact.content_type == "text/html"
                or local_text.lstrip().lower().startswith(("<!doctype html", "<html"))
            )
            if looks_like_html:
                local_text = _html_to_text(local_text)
                artifact.artifact_type = "html"
                artifact.status = "parsed"
                artifact.parser = "html_manifest_reader"
                parsed_html = True
                state.parsed_sources.text_blocks.append(
                    TextBlock(
                        source_file=uploaded.filename,
                        source_path=str(uploaded.path),
                        source_type=SourceType.UNKNOWN,
                        page=None,
                        text=local_text,
                        chunk_id=f"{artifact.artifact_id}_manifest_html",
                    )
                )
            else:
                artifact.status = "inspected"
                artifact.parser = "file_manifest_reader"
            content = f"Local file manifest: {uploaded.filename}\n\n{local_text}"
        else:
            artifact.status = "inspected"
            artifact.parser = "metadata_manifest_reader"
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
        _mark_operation_completed(artifact, action.action)
        return self._result(
            action,
            "completed",
            "Read local file manifest." if local_text is not None else "Read file manifest metadata.",
            manifest_items=len(manifest),
            text_blocks=1 if parsed_html else 0,
        )

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
        artifact.status = "inspected"
        artifact.parser = "metadata_inspection"
        _mark_operation_completed(artifact, action.action)
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


def _tool_call_from_action(
    action: ArtifactAction,
    *,
    workflow_revision: int = 0,
    parsed_content_fingerprint: str | None = None,
) -> ToolCall:
    parameters = dict(action.parameters)
    # Every derived-data stage is tied to the evidence batch it consumed.  This
    # preserves idempotency inside one batch while reopening the complete chain
    # when parsed evidence or the high-relevance work set changes.
    if action.action in {
        "extract_figures",
        "interpret_sections",
        "extract_dynamic_records",
        "extract_records",
        "normalize_records",
        "track_provenance",
        "validate_quality",
    } and parsed_content_fingerprint:
        parameters["_extraction_batch_fingerprint"] = parsed_content_fingerprint
    return ToolCall(
        call_id=action.action_id,
        tool_name=action.action,
        arguments={
            "artifact_id": action.artifact_id,
            "parameters": parameters,
        },
        purpose=action.purpose,
        reason=action.reason,
        priority=action.priority,
        gap_ids=list(action.gap_ids),
        expected_evidence=list(action.expected_fields),
        workflow_revision=max(0, int(workflow_revision)),
    )


def _action_from_tool_call(call: ToolCall) -> ArtifactAction:
    arguments = dict(call.arguments)
    artifact_id = arguments.pop("artifact_id", None)
    raw_parameters = arguments.pop("parameters", None)
    parameters = dict(raw_parameters) if isinstance(raw_parameters, dict) else arguments
    return ArtifactAction(
        action_id=call.call_id,
        artifact_id=artifact_id,
        action=call.tool_name,
        purpose=call.purpose or call.reason or f"Invoke {call.tool_name}.",
        expected_fields=list(call.expected_evidence),
        priority=call.priority,
        reason=call.reason or f"Invoke {call.tool_name}.",
        gap_ids=list(call.gap_ids),
        parameters=parameters,
    )


def _tool_result_from_action_result(
    result: ArtifactActionResult,
    call: ToolCall,
) -> ToolResult:
    status = "completed" if result.status == "no_op" else result.status
    data: dict[str, Any] = {
        "message": result.message,
        "output_counts": dict(result.output_counts),
        "legacy_status": result.status,
    }
    errors = [result.error] if result.error else []
    return ToolResult(
        call_id=call.call_id,
        tool_name=call.tool_name,
        status=status,  # type: ignore[arg-type]
        data=data,
        warnings=list(result.warnings),
        errors=errors,
        workflow_revision=max(0, int(call.workflow_revision)),
        idempotency_key=call.effective_idempotency_key(),
    )


def parsed_content_fingerprint(state: AgentState) -> str:
    """Return a stable token for the effective extraction batch.

    The high-relevance artifact IDs are included because relevance is assigned
    incrementally by the artifact planner.  A previously parsed artifact that
    later crosses the relevance threshold must therefore reopen extraction even
    when its text itself has not changed.
    """
    parsed = getattr(state, "parsed_sources", None)
    source_blocks = effective_extraction_blocks(state)
    high_relevance_artifacts = sorted(
        artifact.artifact_id
        for entry in getattr(state, "source_catalog", [])
        for artifact in entry.artifacts
        if artifact.relevance_score is not None and artifact.relevance_score >= 3.0
    )
    payload = {
        "source_blocks": [
            block.model_dump(mode="json")
            for block in source_blocks
        ],
        "tables": [
            table.model_dump(mode="json")
            for table in getattr(parsed, "tables", [])
        ],
        "figure_assets": [
            figure.model_dump(mode="json")
            for figure in getattr(parsed, "figure_assets", [])
        ],
        "chart_extractions": [
            chart.model_dump(mode="json")
            for chart in getattr(state, "chart_extractions", [])
        ],
        "high_relevance_artifacts": high_relevance_artifacts,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_content_fingerprint(state: AgentState) -> str:
    """Return a stable token for figure/section preprocessing inputs.

    Figure assets and section blocks are outputs of preprocessing, so they are
    intentionally excluded. Including them would invalidate the stage that
    created them and schedule the same work forever.
    """
    parsed = getattr(state, "parsed_sources", None)
    high_relevance_artifacts = sorted(
        artifact.artifact_id
        for entry in getattr(state, "source_catalog", [])
        for artifact in entry.artifacts
        if artifact.relevance_score is not None and artifact.relevance_score >= 3.0
    )
    payload = {
        "files": [
            {
                "filename": uploaded.filename,
                "path": str(uploaded.path),
                "content_type": uploaded.content_type,
            }
            for uploaded in getattr(state, "files", [])
        ],
        "text_blocks": [
            block.model_dump(mode="json")
            for block in getattr(parsed, "text_blocks", [])
        ],
        "tables": [
            table.model_dump(mode="json")
            for table in getattr(parsed, "tables", [])
        ],
        "high_relevance_artifacts": high_relevance_artifacts,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def workflow_stage_fingerprint(state: AgentState, action: str) -> str:
    """Select the correct idempotency scope for one workflow stage."""
    if action in {"extract_figures", "interpret_sections"}:
        return source_content_fingerprint(state)
    return parsed_content_fingerprint(state)


def effective_extraction_blocks(state: AgentState) -> list[Any]:
    """Use section blocks where available and raw text for uncovered files.

    A global ``section_blocks or text_blocks`` choice drops newly parsed text
    whenever older files already have section blocks.  Combining by source file
    keeps the richer representation without omitting the new batch.
    """
    parsed = getattr(state, "parsed_sources", None)
    section_blocks = list(getattr(parsed, "section_blocks", []))
    text_blocks = list(getattr(parsed, "text_blocks", []))
    if not section_blocks:
        return text_blocks
    section_sources = {_block_source_key(block) for block in section_blocks}
    return section_blocks + [
        block for block in text_blocks
        if _block_source_key(block) not in section_sources
    ]


def next_required_derived_stage(
    state: AgentState,
    *,
    unprocessed_relevant_artifacts: list[str] | None = None,
) -> str | None:
    """Return the next mandatory content stage for the current evidence batch."""
    processed = getattr(state, "runtime_stage_fingerprints", {})
    has_content = bool(effective_extraction_blocks(state) or state.parsed_sources.tables)
    pending_artifacts = (
        state.coverage_report.unprocessed_relevant_artifacts
        if unprocessed_relevant_artifacts is None
        else unprocessed_relevant_artifacts
    )
    high_relevance_batch_ready = not pending_artifacts
    if not has_content:
        return None

    source_fingerprint = source_content_fingerprint(state)
    for stage in ("extract_figures", "interpret_sections"):
        if processed.get(stage) != source_fingerprint:
            return stage

    fingerprint = parsed_content_fingerprint(state)
    if (
        state.dynamic_extraction_plan is not None
        and high_relevance_batch_ready
        and processed.get("extract_dynamic_records") != fingerprint
    ):
        return "extract_dynamic_records"
    ordered_dependencies = (
        ("extract_dynamic_records", "extract_records"),
        ("extract_records", "normalize_records"),
        ("normalize_records", "track_provenance"),
        ("track_provenance", "validate_quality"),
    )
    for prerequisite, stage in ordered_dependencies:
        if processed.get(prerequisite) == fingerprint and processed.get(stage) != fingerprint:
            return stage
    return None


def _block_source_key(block: Any) -> str:
    value = getattr(block, "source_path", None) or getattr(block, "source_file", None) or ""
    return str(value).replace("\\", "/").casefold()


def _mark_operation_completed(artifact: SourceArtifact, operation: str) -> None:
    if operation not in artifact.completed_operations:
        artifact.completed_operations.append(operation)


def _artifact_result_from_tool_result(
    action: ArtifactAction,
    result: ToolResult,
) -> ArtifactActionResult:
    legacy_status = result.data.get("legacy_status")
    if result.cached:
        status = "skipped"
        message = "Duplicate tool call skipped; the completed result was already available."
    elif legacy_status in {"completed", "partial", "skipped", "failed", "no_op"}:
        status = legacy_status
        message = str(result.data.get("message") or "")
    else:
        status = result.status
        message = str(result.data.get("message") or "")
    if not message:
        message = "; ".join(result.errors) if result.errors else f"Tool {action.action} returned {status}."
    return ArtifactActionResult(
        action_id=action.action_id,
        artifact_id=action.artifact_id,
        action=action.action,
        status=status,  # type: ignore[arg-type]
        message=message,
        output_counts={
            str(key): int(value)
            for key, value in (result.data.get("output_counts") or {}).items()
            if isinstance(value, (int, float))
        },
        warnings=list(result.warnings),
        error=(result.errors[0] if result.errors else None),
    )


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
    search_strategy: dict[str, Any] | None = None,
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
    if supports_kwargs or "search_strategy" in parameters:
        planner_kwargs["search_strategy"] = search_strategy or {}
    if planner_kwargs:
        return planner(research_question, source_discovery_plan, **planner_kwargs)
    return planner(research_question, source_discovery_plan)


def _normalize_search_strategy(parameters: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize model-authored search recovery hints without hard-coding a plan."""
    raw = parameters if isinstance(parameters, dict) else {}
    strategy: dict[str, Any] = {}
    for key in ("connector_names", "avoid_connectors", "source_types"):
        value = raw.get(key)
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            cleaned = [str(item).strip().casefold() for item in value if str(item).strip()]
            if cleaned:
                strategy[key] = list(dict.fromkeys(cleaned))
    for key in ("query_focus", "failure_reason"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            strategy[key] = value.strip()
    group_id = raw.get("field_group_id")
    if isinstance(group_id, str) and group_id.strip():
        strategy["field_group_id"] = group_id.strip().casefold()
    target_fields = raw.get("target_fields")
    if isinstance(target_fields, str):
        target_fields = [target_fields]
    if isinstance(target_fields, list):
        cleaned_fields = [
            str(field).strip()
            for field in target_fields
            if str(field).strip()
        ]
        if cleaned_fields:
            strategy["target_fields"] = list(dict.fromkeys(cleaned_fields))
    if raw.get("initial_group_search") is True:
        strategy["initial_group_search"] = True
    revised = raw.get("revised_queries")
    if isinstance(revised, dict):
        normalized: dict[str, list[str]] = {}
        for connector, queries in revised.items():
            if isinstance(queries, str):
                queries = [queries]
            if not isinstance(queries, list):
                continue
            values = [" ".join(str(query).split()) for query in queries if str(query).strip()]
            if values:
                normalized[str(connector).strip().casefold()] = list(dict.fromkeys(values))
        if normalized:
            strategy["revised_queries"] = normalized
    return strategy


def _apply_search_strategy(plan: Any, strategy: dict[str, Any]) -> Any:
    """Apply explicit recovery constraints after an LLM search plan is returned."""
    if not strategy:
        return plan
    allowed = set(strategy.get("connector_names", []))
    avoided = set(strategy.get("avoid_connectors", []))
    source_types = set(strategy.get("source_types", []))
    revised = strategy.get("revised_queries", {})
    requests = []
    for request in plan.search_requests:
        connector = str(request.connector_name).casefold()
        if allowed and connector not in allowed:
            continue
        if connector in avoided:
            continue
        if source_types and str(request.source_type).casefold() not in source_types:
            continue
        connector_queries = revised.get(connector, []) if isinstance(revised, dict) else []
        if connector_queries:
            requests.extend(
                request.model_copy(update={"query": query})
                for query in connector_queries
            )
        else:
            requests.append(request)
    field_group_id = str(strategy.get("field_group_id") or "").strip().casefold()
    target_fields = list(strategy.get("target_fields") or [])
    if field_group_id:
        requests = [
            request.model_copy(
                update={
                    "field_group_id": field_group_id,
                    "target_fields": target_fields or request.target_fields,
                }
            )
            for request in requests
        ]
    return plan.model_copy(update={"search_requests": _dedupe_search_requests(requests)})


def _dedupe_search_requests(requests: list[Any]) -> list[Any]:
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for request in requests:
        key = (
            str(request.connector_name).casefold(),
            str(request.source_type).casefold(),
            " ".join(str(request.query).split()).casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(request)
    return unique


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
