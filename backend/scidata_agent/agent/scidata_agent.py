from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from scidata_agent.agent.field_schema import FIELD_SCHEMA
from scidata_agent.agent.action_executor import ArtifactActionExecutor
from scidata_agent.agent.checkpoint import AgentCheckpointStore, build_run_fingerprint
from scidata_agent.agent.monitor import AgentMonitor
from scidata_agent.agent.planner import plan_task as fallback_plan_task
from scidata_agent.agent.schemas import (
    AgentResult,
    AgentState,
    AgentSummary,
    ArxivSearchPlan,
    ArtifactActionIteration,
    QualityIssue,
    ReviewQueueItem,
    ScientificRecord,
    SourceSelectionPlan,
    UploadedFile,
    timestamp_task_id,
)
from scidata_agent.llm.client import LLMConfigurationError, QwenBailianClient
from scidata_agent.llm.nodes import QwenAgentNodes
from scidata_agent.tools.chart_locator import locate_figures
from scidata_agent.tools.chart_validator import validate_chart_extraction
from scidata_agent.tools.coverage import build_coverage_report
from scidata_agent.tools.cross_modal import build_cross_modal_checks
from scidata_agent.tools.curator import curate_dynamic_records
from scidata_agent.tools.exporter import export_results
from scidata_agent.tools.evidence import build_evidence_traces
from scidata_agent.tools.connectors.arxiv import download_arxiv_pdfs, enrich_with_arxiv_results
from scidata_agent.tools.connectors.arxiv import (
    DEFAULT_ARXIV_BATCH_TIMEOUT_SECONDS,
    DEFAULT_PDF_TOTAL_TIMEOUT_SECONDS,
)
from scidata_agent.tools.connectors.registry import execute_multi_source_search, merge_sources
from scidata_agent.tools.normalizer import normalize_records, scientific_records_from_dynamic
from scidata_agent.tools.parser import build_section_blocks_from_plan, fallback_section_plan_from_candidates, parse_sources
from scidata_agent.tools.provenance import build_source_summaries
from scidata_agent.tools.quality import build_quality_report
from scidata_agent.tools.review import build_review_queue
from scidata_agent.tools.source_ingestion import ingest_triaged_sources
from scidata_agent.tools.source_catalog import refresh_source_catalog, source_catalog_summary
from scidata_agent.tools.source_triage import (
    DEFAULT_MAX_AUTO_RESOURCES,
    fallback_pdf_download_decisions,
    ingestible_arxiv_source_ids,
    triage_sources,
    triage_sources_from_selection,
)


CATALOG_REFRESH_STEPS = frozenset(
    {
        "source_discovery",
        "multi_source_search",
        "source_selection",
        "source_triage",
        "multi_source_ingestion",
        "arxiv_pdf_ingestion",
        "artifact_action_execution",
        "source_parsing",
        "figure_chart_extraction",
        "section_interpretation",
        "dynamic_extraction",
        "record_extraction",
        "normalization",
        "provenance_tracking",
        "quality_validation",
    }
)
LOGGER = logging.getLogger(__name__)


class AgentCancellationRequested(RuntimeError):
    """Raised between pipeline steps after an API cancellation request."""


def _worker_count(
    explicit: int | None,
    env_name: str,
    default: int,
    item_count: int,
) -> int:
    configured = explicit
    if configured is None:
        try:
            configured = int(os.getenv(env_name, str(default)))
        except ValueError:
            configured = default
    return max(1, min(int(configured), max(1, item_count)))


def _run_ordered_parallel(items, worker, max_workers: int, on_completed=None):
    """Run independent work concurrently and return results in input order."""
    if max_workers <= 1 or len(items) <= 1:
        results = []
        for index, item in enumerate(items):
            result = worker(item)
            results.append(result)
            if on_completed:
                on_completed(index, item, result, index + 1)
        return results
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="agent-worker",
    ) as executor:
        futures = {
            executor.submit(worker, item): index
            for index, item in enumerate(items)
        }
        results = [None] * len(items)
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            results[index] = result
            completed += 1
            if on_completed:
                on_completed(index, items[index], result, completed)
        return results


class SciDataAgent:
    """Qwen-powered Data Agent for multi-source scientific data integration.

    The workflow follows the competition video requirements:

    source discovery/ingestion -> PDF/table parsing -> field extraction ->
    schema alignment -> conflict/evidence validation -> CSV/JSON export.

    Official mode requires a configured Qwen/Bailian API key. Rule fallback is
    available only for local tool-chain tests and is explicitly logged.
    """

    def __init__(
        self,
        output_dir: str | Path,
        llm_client: QwenBailianClient | None = None,
        require_llm: bool = True,
        allow_rule_fallback: bool = False,
        monitor_console: bool = True,
        monitor_enabled: bool = True,
    ):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_client = llm_client or QwenBailianClient()
        self.require_llm = require_llm
        self.allow_rule_fallback = allow_rule_fallback
        self.llm_nodes = QwenAgentNodes(self.llm_client, allow_rule_fallback=allow_rule_fallback)
        self.monitor_console = monitor_console
        self.monitor_enabled = monitor_enabled
        self._checkpoint_store: AgentCheckpointStore | None = None
        self._checkpoint_fingerprint: str | None = None
        self._completed_checkpoint_steps: set[str] = set()
        self._resume_enabled = False

    def run(
        self,
        research_question: str,
        files: list[str | Path] | None = None,
        max_pdf_pages: int | None = None,
        auto_fetch_arxiv: bool = True,
        enable_live_search: bool | None = None,
        auto_download_sources: bool = True,
        discovery_only: bool = False,
        max_arxiv_papers: int | None = None,
        max_auto_resources: int | None = DEFAULT_MAX_AUTO_RESOURCES,
        max_dynamic_text_blocks: int | None = None,
        max_record_text_blocks: int | None = None,
        max_figures_per_pdf: int | None = None,
        max_pdf_parse_workers: int | None = None,
        max_chart_workers: int | None = None,
        max_text_extraction_workers: int | None = None,
        max_table_extraction_workers: int | None = None,
        max_artifact_action_iterations: int = 1,
        reuse_dynamic_records_for_metrics: bool = True,
        arxiv_pdf_timeout: int = DEFAULT_PDF_TOTAL_TIMEOUT_SECONDS,
        arxiv_download_batch_timeout: int = DEFAULT_ARXIV_BATCH_TIMEOUT_SECONDS,
        task_id: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
        resume: bool = False,
    ) -> AgentResult:
        self._completed_checkpoint_steps = set()
        max_pdf_pages = _unlimited_or_positive(max_pdf_pages)
        max_arxiv_papers = _unlimited_or_positive(max_arxiv_papers)
        max_auto_resources = _unlimited_or_positive(max_auto_resources)
        max_dynamic_text_blocks = _unlimited_or_positive(max_dynamic_text_blocks)
        max_record_text_blocks = _unlimited_or_positive(max_record_text_blocks)
        max_figures_per_pdf = _unlimited_or_positive(max_figures_per_pdf)
        resource_cap = max_arxiv_papers if max_arxiv_papers is not None else max_auto_resources
        # Preserve the library API's historical uploaded-file behavior unless
        # the caller makes an explicit choice.  The HTTP API always sends
        # ``enable_live_search`` and can therefore combine uploads with live
        # discovery without unexpectedly adding network work to local callers.
        live_search_enabled = (
            auto_fetch_arxiv and not (files or [])
            if enable_live_search is None
            else enable_live_search
        )
        artifact_action_iterations = max(1, int(max_artifact_action_iterations))
        uploaded_files = [UploadedFile(filename=Path(path).name, path=Path(path)) for path in (files or [])]
        run_options_for_fingerprint = {
            "max_pdf_pages": max_pdf_pages,
            "auto_fetch_arxiv": auto_fetch_arxiv,
            "enable_live_search": enable_live_search,
            "auto_download_sources": auto_download_sources,
            "discovery_only": discovery_only,
            "max_arxiv_papers": max_arxiv_papers,
            "max_auto_resources": max_auto_resources,
            "max_dynamic_text_blocks": max_dynamic_text_blocks,
            "max_record_text_blocks": max_record_text_blocks,
            "max_figures_per_pdf": max_figures_per_pdf,
            "max_pdf_parse_workers": max_pdf_parse_workers,
            "max_chart_workers": max_chart_workers,
            "max_text_extraction_workers": max_text_extraction_workers,
            "max_table_extraction_workers": max_table_extraction_workers,
            "max_artifact_action_iterations": artifact_action_iterations,
            "reuse_dynamic_records_for_metrics": reuse_dynamic_records_for_metrics,
            "arxiv_pdf_timeout": arxiv_pdf_timeout,
            "arxiv_download_batch_timeout": arxiv_download_batch_timeout,
        }
        resolved_task_id = task_id or timestamp_task_id()
        self._checkpoint_store = AgentCheckpointStore(self.output_dir / resolved_task_id)
        self._checkpoint_fingerprint = build_run_fingerprint(research_question, [path for path in (files or [])], run_options_for_fingerprint)
        self._resume_enabled = bool(resume)
        state = AgentState(
            task_id=resolved_task_id,
            research_question=research_question,
            files=uploaded_files,
            output_dir=self.output_dir,
        )
        if self._resume_enabled:
            checkpoint = self._checkpoint_store.load(fingerprint=self._checkpoint_fingerprint)
            if checkpoint is not None and checkpoint[0].task_id == resolved_task_id:
                state, self._completed_checkpoint_steps = checkpoint
                state.output_dir = self.output_dir
                state.processing_log.append(
                    "Resuming from checkpoint: "
                    f"completed_steps={len(self._completed_checkpoint_steps)}."
                )
            elif checkpoint is not None:
                self._checkpoint_store.last_load_reason = "task_id_mismatch"
        monitor = AgentMonitor(
            task_id=state.task_id,
            output_dir=state.output_dir,
            console=self.monitor_console,
            enabled=self.monitor_enabled,
            cancel_check=cancel_check,
        )
        state.monitor_log_path = monitor.log_path

        def on_llm_model_event(event: dict[str, Any]) -> None:
            event_name = event.get("event", "model_event")
            kind = event.get("kind", "unknown")
            if event_name == "model_switched":
                message = (
                    f"LLM model switched: kind={kind}, "
                    f"failed={event.get('failed_model')}, next={event.get('next_model')}."
                )
                status = "switched"
                step = "llm_model_failover"
            elif event_name == "node_retry":
                message = (
                    f"LLM node retry: node={event.get('node')}, kind={kind}, "
                    f"attempt={event.get('attempt')}/{event.get('max_attempts')}, "
                    f"error={event.get('error')}."
                )
                status = "retrying"
                step = "llm_retry"
            else:
                message = (
                    f"LLM model pool exhausted: kind={kind}, "
                    f"last_failed={event.get('failed_model')}."
                )
                status = "failed"
                step = "llm_model_failover"
            state.processing_log.append(message)
            monitor.emit("llm", step, status, message, event)

        trace_start_index = len(self.llm_client.traces)
        self.llm_client.set_event_callback(on_llm_model_event)
        monitor.task(
            "started",
            "Agent task started.",
            {
                "research_question": research_question,
                "files_count": len(uploaded_files),
                "enable_live_search": live_search_enabled,
                "auto_download_sources": auto_download_sources,
                "discovery_only": discovery_only,
                "max_auto_resources": resource_cap,
                "legacy_max_arxiv_papers": max_arxiv_papers,
                "max_pdf_pages": max_pdf_pages,
                "max_dynamic_text_blocks": max_dynamic_text_blocks,
                "max_record_text_blocks": max_record_text_blocks,
                "max_artifact_action_iterations": artifact_action_iterations,
                "max_pdf_parse_workers": max_pdf_parse_workers,
                "max_chart_workers": max_chart_workers,
                "max_text_extraction_workers": max_text_extraction_workers,
                "max_table_extraction_workers": max_table_extraction_workers,
                "reuse_dynamic_records_for_metrics": reuse_dynamic_records_for_metrics,
                "arxiv_pdf_timeout": arxiv_pdf_timeout,
                "arxiv_download_batch_timeout": arxiv_download_batch_timeout,
            },
        )

        try:
            self._run_step(monitor, "ensure_llm_ready", state, self._ensure_llm_ready)
            self._run_step(monitor, "task_planning", state, self._plan)
            self._run_step(monitor, "dynamic_schema_planning", state, self._plan_dynamic_schema)
            self._run_step(monitor, "source_discovery", state, self._discover_sources)
            # Uploaded files seed the analysis; they must not disable connector
            # search.  Search and download are separate controls so discovery-
            # only tasks can query live providers without fetching artifacts.
            if live_search_enabled:
                self._run_step(monitor, "multi_source_search_planning", state, self._plan_multi_source_search)
                self._run_step(monitor, "multi_source_search", state, self._execute_multi_source_search)
                self._run_step(
                    monitor,
                    "source_selection",
                    state,
                    self._select_sources,
                    max_auto_resources=resource_cap,
                )
                self._run_step(
                    monitor,
                    "source_triage",
                    state,
                    self._triage_sources,
                    max_auto_resources=resource_cap,
                )
                if auto_download_sources and not discovery_only:
                    self._run_step(monitor, "multi_source_ingestion", state, self._ingest_triaged_sources)
                    self._run_step(
                        monitor,
                        "arxiv_pdf_ingestion",
                        state,
                        self._ingest_arxiv_pdfs,
                        max_auto_resources=resource_cap,
                        step_monitor=monitor,
                        pdf_timeout=arxiv_pdf_timeout,
                        batch_timeout=arxiv_download_batch_timeout,
                    )
                else:
                    state.processing_log.append(
                        "Source download skipped by policy: live connector results remain metadata-only."
                    )
            if not discovery_only:
                self._run_artifact_action_iteration(
                    monitor,
                    state,
                    iteration=0,
                    max_auto_resources=resource_cap,
                    arxiv_pdf_timeout=arxiv_pdf_timeout,
                    arxiv_download_batch_timeout=arxiv_download_batch_timeout,
                )
                self._run_content_pipeline(
                    monitor,
                    state,
                    max_pdf_pages=max_pdf_pages,
                    max_dynamic_text_blocks=max_dynamic_text_blocks,
                    max_record_text_blocks=max_record_text_blocks,
                    max_figures_per_pdf=max_figures_per_pdf,
                    max_pdf_parse_workers=max_pdf_parse_workers,
                    max_chart_workers=max_chart_workers,
                    max_text_extraction_workers=max_text_extraction_workers,
                    max_table_extraction_workers=max_table_extraction_workers,
                    reuse_dynamic_records_for_metrics=reuse_dynamic_records_for_metrics,
                )
                self._refresh_coverage(state, "initial_content_pipeline")
                if state.coverage_report.decision == "continue" and artifact_action_iterations < 2:
                    artifact_action_iterations = 2
                    state.processing_log.append(
                        "Coverage remains incomplete after the initial content pipeline; "
                        "one follow-up artifact-planning iteration was added."
                    )
            if not discovery_only and artifact_action_iterations > 1:
                self._run_step(
                    monitor,
                    "quality_validation_before_artifact_followup",
                    state,
                    self._quality_check,
                )
            for iteration in range(1, artifact_action_iterations if not discovery_only else 1):
                if not state.artifact_action_plan or not state.artifact_action_plan.should_continue:
                    break
                if not state.artifact_action_plan.actions:
                    if state.coverage_report.decision != "continue":
                        state.processing_log.append(
                            "Artifact action loop stopped: planner requested continuation without actions."
                        )
                        break
                    state.processing_log.append(
                        "Artifact action planner returned no actions while coverage is incomplete; replanning."
                    )
                self._run_artifact_action_iteration(
                    monitor,
                    state,
                    iteration=iteration,
                    max_auto_resources=resource_cap,
                    arxiv_pdf_timeout=arxiv_pdf_timeout,
                    arxiv_download_batch_timeout=arxiv_download_batch_timeout,
                )
                if self._artifact_actions_need_content_refresh(state.artifact_action_results):
                    self._run_content_pipeline(
                        monitor,
                        state,
                        max_pdf_pages=max_pdf_pages,
                        max_dynamic_text_blocks=max_dynamic_text_blocks,
                        max_record_text_blocks=max_record_text_blocks,
                        max_figures_per_pdf=max_figures_per_pdf,
                        max_pdf_parse_workers=max_pdf_parse_workers,
                        max_chart_workers=max_chart_workers,
                        max_text_extraction_workers=max_text_extraction_workers,
                        max_table_extraction_workers=max_table_extraction_workers,
                        reuse_dynamic_records_for_metrics=reuse_dynamic_records_for_metrics,
                    )
                    self._refresh_coverage(state, f"artifact_followup_content_pipeline_{iteration}")
                    if iteration + 1 < artifact_action_iterations:
                        self._run_step(
                            monitor,
                            "quality_validation_before_artifact_followup",
                            state,
                            self._quality_check,
                        )
                if not state.artifact_action_plan or not state.artifact_action_plan.should_continue:
                    break
            if not discovery_only and artifact_action_iterations > 1 and len(state.artifact_action_history) >= artifact_action_iterations:
                if state.artifact_action_plan and state.artifact_action_plan.should_continue:
                    state.processing_log.append(
                        f"Artifact action loop reached configured cap={artifact_action_iterations}."
                    )
            self._run_step(monitor, "quality_validation", state, self._quality_check)
            self._append_llm_trace(state, start_index=trace_start_index)
            final_status = (
                "partial"
                if not discovery_only and state.coverage_report.decision != "allow_stop"
                else "completed"
            )
            if final_status == "partial":
                state.processing_log.append(
                    "Agent task produced partial results: coverage remains incomplete after all configured action iterations."
                )
            self._run_step(monitor, "export", state, self._export)
            result = self._build_result(state, status=final_status)
            monitor.task(
                final_status,
                "Agent task completed." if final_status == "completed" else "Agent task partially completed.",
                _result_snapshot(result),
            )
            return result
        except AgentCancellationRequested:
            message = "Task cancellation requested."
            state.processing_log.append(message)
            monitor.task("cancelled", message, _state_snapshot(state))
            self._append_llm_trace(state, start_index=trace_start_index)
            return self._build_result(state, status="failed")
        except Exception as exc:
            LOGGER.exception("SciData Agent task %s failed", state.task_id)
            state.processing_log.append(f"Task failed: {exc}")
            monitor.error("task", f"Agent task failed: {exc}", _state_snapshot(state))
            self._append_llm_trace(state, start_index=trace_start_index)
            result = self._build_result(state, status="failed")
            monitor.task("failed", "Agent task failed.", _result_snapshot(result))
            return result
        finally:
            self.llm_client.set_event_callback(None)

    def _run_step(self, monitor: AgentMonitor, step: str, state: AgentState, func, **kwargs) -> None:
        if monitor.cancel_requested():
            raise AgentCancellationRequested
        if self._resume_enabled and step in self._completed_checkpoint_steps:
            message = f"{step} skipped: restored from checkpoint."
            state.processing_log.append(message)
            monitor.emit("step", step, "resumed", message, _state_snapshot(state))
            return
        monitor.start(step, f"{step} started.", _state_snapshot(state))
        warning_start = len(self.llm_nodes.node_warnings)
        normalization_start = len(self.llm_nodes.normalization_events)
        try:
            func(state, **kwargs)
            if monitor.cancel_requested():
                raise AgentCancellationRequested
            if step in CATALOG_REFRESH_STEPS:
                self._refresh_catalog(state, step)
        except Exception as exc:
            state.processing_log.extend(self.llm_nodes.node_warnings[warning_start:])
            self._append_normalization_log(state, step, normalization_start)
            monitor.error(step, f"{step} failed: {exc}", _state_snapshot(state))
            raise
        state.processing_log.extend(self.llm_nodes.node_warnings[warning_start:])
        self._append_normalization_log(state, step, normalization_start)
        self._completed_checkpoint_steps.add(step)
        if self._checkpoint_store is not None and self._checkpoint_fingerprint is not None:
            self._checkpoint_store.save(
                state,
                fingerprint=self._checkpoint_fingerprint,
                completed_steps=self._completed_checkpoint_steps,
            )
        monitor.end(step, f"{step} completed.", _state_snapshot(state))

    def _append_normalization_log(
        self,
        state: AgentState,
        step: str,
        start_index: int,
    ) -> None:
        """Keep schema repairs auditable without dumping full LLM payloads."""
        events = self.llm_nodes.normalization_events[start_index:]
        if not events:
            return
        paths = []
        for event in events:
            path = event.get("path")
            if path and path not in paths:
                paths.append(path)
        preview = ", ".join(paths[:12])
        if len(paths) > 12:
            preview += f", ... (+{len(paths) - 12})"
        state.processing_log.append(
            f"LLM output normalization: step={step}, events={len(events)}, paths=[{preview}]."
        )

    def _refresh_catalog(self, state: AgentState, step: str) -> None:
        catalog = refresh_source_catalog(state)
        summary = source_catalog_summary(catalog)
        state.processing_log.append(
            "Source catalog refreshed after "
            f"{step}: sources={summary['source_catalog_count']}, "
            f"artifacts={summary['source_artifacts_count']}, "
            f"source_statuses={summary['source_catalog_statuses']}, "
            f"artifact_statuses={summary['source_artifact_statuses']}."
        )

    def _refresh_coverage(self, state: AgentState, step: str) -> None:
        state.coverage_report = build_coverage_report(state)
        state.processing_log.append(
            f"Coverage audit after {step}: decision={state.coverage_report.decision}, "
            f"score={state.coverage_report.coverage_score:.3f}, "
            f"missing={state.coverage_report.missing_requirements}, "
            f"gaps={len(state.coverage_report.gaps)}, "
            f"unprocessed_relevant={len(state.coverage_report.unprocessed_relevant_artifacts)}."
        )

    def _run_artifact_action_iteration(
        self,
        monitor: AgentMonitor,
        state: AgentState,
        *,
        iteration: int,
        max_auto_resources: int | None,
        arxiv_pdf_timeout: int,
        arxiv_download_batch_timeout: int,
    ) -> None:
        self._run_step(
            monitor,
            "artifact_action_planning",
            state,
            self._plan_artifact_actions,
            iteration=iteration,
        )
        self._run_step(monitor, "artifact_action_execution", state, self._execute_artifact_actions)
        self._refresh_catalog(state, "artifact_action_execution")
        self._refresh_coverage(state, "artifact_action_execution")
        if any(
            result.action == "search_more" and result.status == "completed"
            for result in state.artifact_action_results
        ):
            self._run_search_more_followup(
                monitor,
                state,
                max_auto_resources=max_auto_resources,
                arxiv_pdf_timeout=arxiv_pdf_timeout,
                arxiv_download_batch_timeout=arxiv_download_batch_timeout,
            )
            self._refresh_catalog(state, "artifact_search_more_followup")
            self._refresh_coverage(state, "artifact_search_more_followup")

    def _run_search_more_followup(
        self,
        monitor: AgentMonitor,
        state: AgentState,
        *,
        max_auto_resources: int | None,
        arxiv_pdf_timeout: int,
        arxiv_download_batch_timeout: int,
    ) -> None:
        """Turn an LLM-requested broader search into ingestible artifacts.

        ``search_more`` only discovers candidates in the executor. The normal
        source-selection and ingestion stages remain the single owner of
        download policy, so the follow-up reuses those stages and refreshes the
        catalog before the next planner iteration.
        """
        self._run_step(
            monitor,
            "artifact_search_more_source_selection",
            state,
            self._select_sources,
            max_auto_resources=max_auto_resources,
        )
        self._run_step(
            monitor,
            "artifact_search_more_source_triage",
            state,
            self._triage_sources,
            max_auto_resources=max_auto_resources,
        )
        self._run_step(
            monitor,
            "artifact_search_more_ingestion",
            state,
            self._ingest_triaged_sources,
        )
        self._run_step(
            monitor,
            "artifact_search_more_arxiv_ingestion",
            state,
            self._ingest_arxiv_pdfs,
            max_auto_resources=max_auto_resources,
            step_monitor=monitor,
            pdf_timeout=arxiv_pdf_timeout,
            batch_timeout=arxiv_download_batch_timeout,
        )
        self._refresh_catalog(state, "artifact_search_more_followup")
        state.processing_log.append(
            "Artifact search_more follow-up completed: source selection, triage, "
            "ingestion, arXiv ingestion, and catalog refresh were executed."
        )

    def _run_content_pipeline(
        self,
        monitor: AgentMonitor,
        state: AgentState,
        *,
        max_pdf_pages: int | None,
        max_dynamic_text_blocks: int | None,
        max_record_text_blocks: int | None,
        max_figures_per_pdf: int | None,
        max_pdf_parse_workers: int | None,
        max_chart_workers: int | None,
        max_text_extraction_workers: int | None,
        max_table_extraction_workers: int | None,
        reuse_dynamic_records_for_metrics: bool,
    ) -> None:
        if state.files or state.parsed_sources.text_blocks or state.parsed_sources.tables:
            self._run_step(
                monitor,
                "source_parsing",
                state,
                self._parse,
                max_pdf_pages=max_pdf_pages,
                max_workers=max_pdf_parse_workers,
            )
            self._run_step(
                monitor,
                "figure_chart_extraction",
                state,
                self._extract_charts,
                step_monitor=monitor,
                max_figures_per_pdf=max_figures_per_pdf,
                max_workers=max_chart_workers,
            )
            self._run_step(monitor, "section_interpretation", state, self._interpret_sections)
            self._run_step(
                monitor,
                "dynamic_extraction",
                state,
                self._extract_dynamic,
                step_monitor=monitor,
                max_text_blocks=max_dynamic_text_blocks,
                max_text_workers=max_text_extraction_workers,
                max_table_workers=max_table_extraction_workers,
            )
            self._run_step(
                monitor,
                "record_extraction",
                state,
                self._extract,
                step_monitor=monitor,
                max_text_blocks=max_record_text_blocks,
                reuse_dynamic_records=reuse_dynamic_records_for_metrics,
                max_text_workers=max_text_extraction_workers,
                max_table_workers=max_table_extraction_workers,
            )
            self._run_step(monitor, "normalization", state, self._normalize)
            self._run_step(monitor, "provenance_tracking", state, self._trace)
        else:
            state.processing_log.append(
                "No local files were provided. Agent completed research-goal planning and source discovery only."
            )
            monitor.emit(
                "step",
                "source_parsing",
                "skipped",
                "No local files are available after source discovery/arXiv ingestion.",
                {"files_count": 0},
            )

    @staticmethod
    def _artifact_actions_need_content_refresh(results: list[Any]) -> bool:
        content_actions = {
            "download_artifact",
            "parse_pdf_text",
            "parse_pdf_sections",
            "parse_table",
            "parse_figure",
            "parse_csv",
        }
        return any(
            result.status == "completed" and result.action in content_actions
            for result in results
        )

    def _plan_artifact_actions(self, state: AgentState, iteration: int = 0) -> None:
        if not state.source_catalog:
            self._refresh_catalog(state, "artifact_action_planning_input")
        self._refresh_coverage(state, "artifact_action_planning_input")
        state.artifact_action_plan = self.llm_nodes.plan_artifact_actions(
            state.research_question,
            state.source_catalog,
            dynamic_plan=state.dynamic_extraction_plan,
            quality_report=state.quality_report,
            coverage_report=state.coverage_report,
            processing_log=state.processing_log,
            connector_failures=[
                item
                for item in state.connector_status
                if item.get("status") in {"failed", "error"}
            ],
            iteration=iteration,
        )
        action_counts: dict[str, int] = {}
        for action in state.artifact_action_plan.actions:
            action_counts[action.action] = action_counts.get(action.action, 0) + 1
        self._persist_artifact_assessments(state)
        state.processing_log.append(
            "Qwen Artifact Action Planner completed: "
            f"iteration={state.artifact_action_plan.iteration}, "
            f"should_continue={state.artifact_action_plan.should_continue}, "
            f"actions={len(state.artifact_action_plan.actions)}, "
            f"action_counts={action_counts}."
        )

    @staticmethod
    def _persist_artifact_assessments(state: AgentState) -> None:
        if not state.source_discovery_plan or not state.artifact_action_plan:
            return
        source_by_id = {
            source.source_id: source
            for source in state.source_discovery_plan.candidate_sources
        }
        catalog_artifacts = {
            artifact.artifact_id: artifact
            for entry in state.source_catalog
            for artifact in entry.artifacts
        }
        for assessment in state.artifact_action_plan.artifact_assessments:
            artifact = catalog_artifacts.get(assessment.artifact_id)
            source = source_by_id.get(artifact.source_id) if artifact else None
            if source is None or artifact is None:
                continue
            stored = source.metadata.setdefault("artifact_relevance_assessments", {})
            stored[assessment.artifact_id] = assessment.model_dump(mode="json")

    def _execute_artifact_actions(self, state: AgentState) -> None:
        if not state.artifact_action_plan:
            state.processing_log.append("Artifact action execution skipped: plan is missing.")
            return
        state.artifact_action_results = ArtifactActionExecutor(self.llm_nodes).execute_plan(
            state.artifact_action_plan,
            state,
        )
        state.artifact_action_history.append(
            ArtifactActionIteration(
                iteration=state.artifact_action_plan.iteration,
                plan=state.artifact_action_plan,
                results=state.artifact_action_results,
            )
        )
        result_counts: dict[str, int] = {}
        for result in state.artifact_action_results:
            result_counts[result.status] = result_counts.get(result.status, 0) + 1
        state.processing_log.append(
            "Artifact action execution completed: "
            f"results={len(state.artifact_action_results)}, statuses={result_counts}."
        )

    def _ensure_llm_ready(self, state: AgentState) -> None:
        if self.require_llm:
            self.llm_client.require_configured()
            state.processing_log.append(
                f"Qwen/Bailian model pool configured: active={self.llm_client.model}, "
                f"text_models={','.join(self.llm_client.text_models)}, "
                f"vl_models={','.join(self.llm_client.vl_models)}. "
                "Core Agent nodes will call the real LLM."
            )
        elif self.allow_rule_fallback:
            state.processing_log.append(
                "Local tool-chain test mode: rule fallback is enabled. "
                "This mode must not be used as the official competition result."
            )
        else:
            raise LLMConfigurationError("LLM is not enabled and fallback is not allowed.")

    def _plan(self, state: AgentState) -> None:
        if self.require_llm or self.llm_client.configured:
            state.task_plan = self.llm_nodes.plan_task(state.research_question)
            state.processing_log.append(
                f"Qwen Task Planner completed: domain={state.task_plan.domain}; "
                f"target_fields={', '.join(state.task_plan.target_fields)}."
            )
        else:
            state.task_plan = fallback_plan_task(state.research_question)
            state.processing_log.append(f"Rule Task Planner completed: domain={state.task_plan.domain}.")

    def _plan_dynamic_schema(self, state: AgentState) -> None:
        state.dynamic_extraction_plan = self.llm_nodes.plan_dynamic_extraction(
            state.research_question,
            task_plan=state.task_plan,
        )
        table_names = [table.table_name for table in state.dynamic_extraction_plan.dynamic_tables]
        state.processing_log.append(
            f"Qwen Dynamic Schema Planner completed: domain={state.dynamic_extraction_plan.domain}, "
            f"task_type={state.dynamic_extraction_plan.task_type}, "
            f"tables={', '.join(table_names)}."
        )

    def _discover_sources(self, state: AgentState) -> None:
        state.source_discovery_plan = self.llm_nodes.discover_sources(state.research_question)
        raw_discovered_count = len(state.source_discovery_plan.candidate_sources)
        clustered_sources, clustered_count = merge_sources(
            [],
            state.source_discovery_plan.candidate_sources,
        )
        state.source_discovery_plan.candidate_sources = clustered_sources
        discovered_count = len(state.source_discovery_plan.candidate_sources)
        if state.task_plan:
            if not state.task_plan.dynamic_schema and state.source_discovery_plan.dynamic_schema:
                state.task_plan.dynamic_schema = dict(state.source_discovery_plan.dynamic_schema)
            if not state.task_plan.source_requirements and state.source_discovery_plan.target_data_types:
                state.task_plan.source_requirements = list(state.source_discovery_plan.target_data_types)
        state.processing_log.append(
            f"Source Discovery completed: domain={state.source_discovery_plan.domain}, "
            f"candidate_sources={discovered_count}, "
            f"source_clusters={discovered_count}, "
            f"duplicates_merged={raw_discovered_count - clustered_count}, "
            f"target_data_types={', '.join(state.source_discovery_plan.target_data_types)}."
        )

    def _plan_arxiv_search(self, state: AgentState) -> None:
        if not state.source_discovery_plan:
            state.processing_log.append("arXiv Search Planner skipped: source discovery plan is missing.")
            return
        state.arxiv_search_plan = self.llm_nodes.plan_arxiv_search(
            state.research_question,
            state.source_discovery_plan,
        )
        query_preview = [query.query for query in state.arxiv_search_plan.queries[:5]]
        state.processing_log.append(
            "Qwen arXiv Search Planner completed: "
            f"should_search_arxiv={state.arxiv_search_plan.should_search_arxiv}, "
            f"queries={len(state.arxiv_search_plan.queries)}, "
            f"query_preview={query_preview}, "
            f"selection_criteria={state.arxiv_search_plan.selection_criteria[:5]}."
        )

    def _plan_multi_source_search(self, state: AgentState) -> None:
        if not state.source_discovery_plan:
            state.processing_log.append("Multi-source Search Planner skipped: source discovery plan is missing.")
            return
        state.multi_source_search_plan = self.llm_nodes.plan_multi_source_search(
            state.research_question,
            state.source_discovery_plan,
        )
        state.arxiv_search_plan = _arxiv_plan_from_multi_source_plan(state)
        connectors = sorted(
            {request.connector_name for request in state.multi_source_search_plan.search_requests}
        )
        state.processing_log.append(
            "Qwen Multi-source Search Planner completed: "
            f"should_search={state.multi_source_search_plan.should_search}, "
            f"requests={len(state.multi_source_search_plan.search_requests)}, "
            f"connectors={connectors}, "
            f"selection_criteria={state.multi_source_search_plan.selection_criteria[:5]}."
        )

    def _execute_multi_source_search(self, state: AgentState) -> None:
        if not state.source_discovery_plan:
            state.processing_log.append("Multi-source search skipped: source discovery plan is missing.")
            return
        if not state.multi_source_search_plan:
            state.processing_log.append("Multi-source search skipped: multi-source search plan is missing.")
            return

        before = len(state.source_discovery_plan.candidate_sources)
        discovered_sources, status = execute_multi_source_search(
            state.multi_source_search_plan,
            cache_dir=state.output_dir / "_cache" / "source_search",
        )
        merged, added = merge_sources(state.source_discovery_plan.candidate_sources, discovered_sources)
        state.source_discovery_plan.candidate_sources = merged
        state.connector_status = list(status.get("connector_status", []))
        state.source_discovery_plan.notes.append(f"Multi-source search status: {status}.")
        for connector_status in state.connector_status[:40]:
            state.source_discovery_plan.notes.append(f"Connector status: {connector_status}.")
        state.processing_log.append(
            "Multi-source search completed: "
            f"requests={status.get('searched', 0)}, "
            f"new_sources={added}, "
            f"candidate_sources_before={before}, "
            f"candidate_sources_after={len(merged)}, "
            f"failed_requests={status.get('failed', 0)}, "
            f"status={status.get('status')}."
        )
        if status.get("failed"):
            failed_connectors = [
                item.get("connector")
                for item in state.connector_status
                if item.get("status") == "failed"
            ]
            state.processing_log.append(
                "Multi-source search needs review: "
                f"failed_connectors={failed_connectors}. See connector_status.csv/json for details."
            )

    def _select_sources(self, state: AgentState, max_auto_resources: int | None) -> None:
        if not state.source_discovery_plan:
            state.processing_log.append("Source selection skipped: source discovery plan is missing.")
            return
        if not state.source_discovery_plan.candidate_sources:
            state.processing_log.append("Source selection skipped: no candidate sources are available.")
            return
        candidates = list(state.source_discovery_plan.candidate_sources)
        batch_size = _positive_env_int("SCIDATA_SOURCE_SELECTION_BATCH_SIZE", 40)
        batches = [
            candidates[start : start + batch_size]
            for start in range(0, len(candidates), batch_size)
        ] or [[]]
        plans: list[SourceSelectionPlan] = []
        selection_errors: list[str] = []
        for index, batch in enumerate(batches, start=1):
            batch_plan = state.source_discovery_plan.model_copy(update={"candidate_sources": batch})
            try:
                plans.append(
                    self.llm_nodes.select_sources(
                        state.research_question,
                        batch_plan,
                        dynamic_plan=state.dynamic_extraction_plan,
                        multi_source_search_plan=state.multi_source_search_plan,
                        connector_status=state.connector_status,
                        max_auto_resources=max_auto_resources,
                        candidate_limit=batch_size,
                    )
                )
            except Exception as exc:
                selection_errors.append(f"batch {index}/{len(batches)}: {exc}")
        if not plans:
            raise RuntimeError(
                "All source-selection batches failed: " + "; ".join(selection_errors)
            )
        state.source_selection_plan = _merge_source_selection_plans(plans)
        if selection_errors:
            state.source_selection_plan.notes.append(
                f"{len(selection_errors)} source-selection batch(es) failed; successful batches were retained."
            )
            state.processing_log.extend(
                f"Source selection warning: {error}" for error in selection_errors
            )
        decision_counts: dict[str, int] = {}
        priority_counts: dict[str, int] = {}
        for decision in state.source_selection_plan.decisions:
            decision_counts[decision.decision] = decision_counts.get(decision.decision, 0) + 1
            priority_counts[decision.priority] = priority_counts.get(decision.priority, 0) + 1
        state.processing_log.append(
            "Qwen Source Selector completed: "
            f"candidate_sources={len(state.source_discovery_plan.candidate_sources)}, "
            f"decisions={len(state.source_selection_plan.decisions)}, "
            f"selection_batches={len(batches)}, "
            f"failed_selection_batches={len(selection_errors)}, "
            f"max_auto_resources={max_auto_resources}, "
            f"decision_counts={decision_counts}, "
            f"priority_counts={priority_counts}, "
            f"time_range={state.source_selection_plan.time_range_interpreted}."
        )

    def _triage_sources(self, state: AgentState, max_auto_resources: int | None) -> None:
        if not state.source_discovery_plan:
            state.processing_log.append("Source triage skipped: source discovery plan is missing.")
            return
        if state.source_selection_plan:
            state.source_triage_decisions = triage_sources_from_selection(
                state.source_discovery_plan.candidate_sources,
                state.source_selection_plan,
                max_pdf_downloads=None,
                max_auto_resources=max_auto_resources,
            )
        elif self.require_llm and not self.allow_rule_fallback:
            raise RuntimeError("Source triage requires an LLM Source Selection Plan in official mode.")
        else:
            state.source_triage_decisions = triage_sources(
                state.source_discovery_plan.candidate_sources,
                state.research_question,
                max_pdf_downloads=max_auto_resources,
            )
        action_counts: dict[str, int] = {}
        ingest_count = 0
        for decision in state.source_triage_decisions:
            action_counts[decision.recommended_action] = action_counts.get(decision.recommended_action, 0) + 1
            if decision.should_ingest:
                ingest_count += 1
        state.processing_log.append(
            "Source triage completed: "
            f"sources={len(state.source_triage_decisions)}, "
            f"ingest_selected={ingest_count}, "
            f"max_auto_resources={max_auto_resources}, "
            f"action_counts={action_counts}."
        )

    def _ingest_triaged_sources(self, state: AgentState) -> None:
        if not state.source_discovery_plan:
            state.processing_log.append("Multi-source ingestion skipped: source discovery plan is missing.")
            return
        if not state.source_triage_decisions:
            state.processing_log.append("Multi-source ingestion skipped: source triage decisions are missing.")
            return
        uploaded_files, text_blocks, insights, logs = ingest_triaged_sources(
            state.source_discovery_plan.candidate_sources,
            state.source_triage_decisions,
            state.output_dir,
            state.task_id,
        )
        state.files.extend(uploaded_files)
        state.parsed_sources.text_blocks.extend(text_blocks)
        state.source_insights.extend(insights)
        state.processing_log.extend(logs)
        state.processing_log.append(
            "Multi-source ingestion completed: "
            f"uploaded_files={len(uploaded_files)}, "
            f"source_text_blocks={len(text_blocks)}, "
            f"source_insights={len(insights)}."
        )
        failed_download_ids = {
            decision.source_id
            for decision in state.source_triage_decisions
            if decision.recommended_action in {"download_pdf", "download_small_table", "download_small_supplement"}
            and decision.should_ingest
            and next(
                (
                    source
                    for source in state.source_discovery_plan.candidate_sources
                    if source.source_id == decision.source_id
                ),
                None,
            ) is not None
            and next(
                (
                    source
                    for source in state.source_discovery_plan.candidate_sources
                    if source.source_id == decision.source_id
                ),
                None,
            ).metadata.get("last_ingestion_status") == "failed"
        }
        if not failed_download_ids:
            return
        attempted_ids = {
            source.source_id
            for source in state.source_discovery_plan.candidate_sources
            if int(source.metadata.get("ingestion_attempts") or 0) > 0
        }
        try:
            fallback_multiplier = max(1, int(os.getenv("SCIDATA_DOWNLOAD_FALLBACK_MULTIPLIER", "3")))
        except ValueError:
            fallback_multiplier = 3
        fallback_decisions = fallback_pdf_download_decisions(
            state.source_discovery_plan.candidate_sources,
            attempted_ids,
            failed_download_ids,
            multiplier=fallback_multiplier,
        )
        if not fallback_decisions:
            state.processing_log.append(
                "Download fallback found no unattempted PDF candidates after ingestion failures."
            )
            return
        state.source_triage_decisions.extend(fallback_decisions)
        state.processing_log.append(
            "Download fallback retry started: "
            f"failed_sources={len(failed_download_ids)}, replacement_candidates={len(fallback_decisions)}."
        )
        fallback_files, fallback_blocks, fallback_insights, fallback_logs = ingest_triaged_sources(
            state.source_discovery_plan.candidate_sources,
            fallback_decisions,
            state.output_dir,
            state.task_id,
        )
        state.files.extend(fallback_files)
        state.parsed_sources.text_blocks.extend(fallback_blocks)
        state.source_insights.extend(fallback_insights)
        state.processing_log.extend(fallback_logs)
        state.processing_log.append(
            "Download fallback retry completed: "
            f"uploaded_files={len(fallback_files)}, source_text_blocks={len(fallback_blocks)}, "
            f"source_insights={len(fallback_insights)}."
        )

    def _ingest_arxiv_pdfs(
        self,
        state: AgentState,
        max_auto_resources: int | None,
        step_monitor: AgentMonitor | None = None,
        pdf_timeout: int = DEFAULT_PDF_TOTAL_TIMEOUT_SECONDS,
        batch_timeout: int = DEFAULT_ARXIV_BATCH_TIMEOUT_SECONDS,
    ) -> None:
        if not state.source_discovery_plan:
            state.processing_log.append("arXiv PDF ingestion skipped: source discovery plan is missing.")
            return
        if not state.arxiv_search_plan and not state.multi_source_search_plan:
            state.processing_log.append("arXiv PDF ingestion skipped: arXiv search plan is missing.")
            return
        before_arxiv = len(state.source_discovery_plan.candidate_sources)
        arxiv_status = "already_searched_by_multi_source_registry"
        if not state.multi_source_search_plan and state.arxiv_search_plan:
            state.source_discovery_plan, arxiv_status = enrich_with_arxiv_results(
                state.source_discovery_plan,
                state.arxiv_search_plan,
            )
            # The dedicated arXiv path predates the multi-source registry. Run
            # the same source clustering pass here so arXiv records can merge
            # with existing metadata or uploaded-source hints as well.
            clustered_sources, _ = merge_sources(
                [],
                state.source_discovery_plan.candidate_sources,
            )
            state.source_discovery_plan.candidate_sources = clustered_sources
        discovered_count = len(state.source_discovery_plan.candidate_sources)
        allowed_source_ids = ingestible_arxiv_source_ids(state.source_triage_decisions) if state.source_triage_decisions else None
        if allowed_source_ids is not None and not allowed_source_ids:
            state.processing_log.append("arXiv PDF ingestion skipped: source triage selected no arXiv PDFs.")
            return
        download_dir = state.output_dir / "_cache" / "arxiv"
        reuse_dirs = _previous_arxiv_download_dirs(state.output_dir, state.task_id)
        downloaded_paths = download_arxiv_pdfs(
            state.source_discovery_plan,
            download_dir=download_dir,
            max_papers=max_auto_resources,
            allowed_source_ids=allowed_source_ids,
            total_timeout=pdf_timeout,
            batch_timeout=batch_timeout,
            progress_callback=_arxiv_download_progress_callback(step_monitor, state),
            reuse_dirs=reuse_dirs,
        )
        # A failed arXiv transport should not consume the whole paper-ingestion
        # opportunity. The downloader records per-source failures, so select
        # unattempted PDF candidates and give them the same bounded fallback
        # treatment as other multi-source downloads.
        attempted_arxiv_ids = {
            source.source_id
            for source in state.source_discovery_plan.candidate_sources
            if str(source.metadata.get("provider") or "").strip().lower() == "arxiv"
            and int(source.metadata.get("ingestion_attempts") or 0) > 0
        }
        failed_arxiv_ids = {
            source.source_id
            for source in state.source_discovery_plan.candidate_sources
            if source.source_id in attempted_arxiv_ids
            and str(source.metadata.get("provider") or "").strip().lower() == "arxiv"
            and source.metadata.get("last_ingestion_status") == "failed"
        }
        if failed_arxiv_ids:
            try:
                fallback_multiplier = max(
                    1,
                    int(os.getenv("SCIDATA_DOWNLOAD_FALLBACK_MULTIPLIER", "3")),
                )
            except ValueError:
                fallback_multiplier = 3
            fallback_decisions = fallback_pdf_download_decisions(
                state.source_discovery_plan.candidate_sources,
                attempted_arxiv_ids,
                failed_arxiv_ids,
                multiplier=fallback_multiplier,
            )
            fallback_arxiv_decisions = [
                decision
                for decision in fallback_decisions
                if decision.provider == "arxiv"
            ]
            if fallback_arxiv_decisions:
                state.source_triage_decisions.extend(fallback_arxiv_decisions)
                fallback_ids = {decision.source_id for decision in fallback_arxiv_decisions}
                fallback_paths = download_arxiv_pdfs(
                    state.source_discovery_plan,
                    download_dir=download_dir,
                    max_papers=len(fallback_ids),
                    allowed_source_ids=fallback_ids,
                    total_timeout=pdf_timeout,
                    batch_timeout=batch_timeout,
                    progress_callback=_arxiv_download_progress_callback(step_monitor, state),
                    reuse_dirs=reuse_dirs,
                )
                downloaded_paths.extend(fallback_paths)
                state.processing_log.append(
                    "arXiv download fallback completed: "
                    f"failed_sources={len(failed_arxiv_ids)}, "
                    f"replacement_candidates={len(fallback_arxiv_decisions)}, "
                    f"downloaded_pdfs={len(fallback_paths)}."
                )
            else:
                state.processing_log.append(
                    "arXiv download fallback found no unattempted PDF candidates."
                )
        for path in downloaded_paths:
            state.files.append(
                UploadedFile(
                    filename=path.name,
                    path=path,
                    content_type="application/pdf",
                )
            )
        state.processing_log.append(
            f"arXiv PDF ingestion completed: downloaded_pdfs={len(downloaded_paths)}, "
            f"max_auto_resources={max_auto_resources}, "
            f"arxiv_sources_added={discovered_count - before_arxiv}, "
            f"arxiv_status={arxiv_status}."
        )
        if max_auto_resources != 0 and not downloaded_paths:
            state.processing_log.append(
                "Deep paper ingestion warning: no arXiv PDFs were downloaded. "
                "The task may be based only on metadata, abstracts, manifests, or uploaded files."
            )

    def _parse(
        self,
        state: AgentState,
        max_pdf_pages: int | None,
        max_workers: int | None = None,
    ) -> None:
        parsed = parse_sources(
            state.files,
            max_pdf_pages=max_pdf_pages,
            max_workers=max_workers,
        )
        processed_text_paths = _completed_artifact_paths(
            state, {"parse_pdf_text", "parse_pdf_sections"}
        )
        processed_table_paths = _completed_artifact_paths(
            state, {"parse_table", "parse_csv"}
        )
        processed_section_paths = _completed_artifact_paths(
            state, {"parse_pdf_sections"}
        )

        existing_text_keys = {_text_block_key(block) for block in state.parsed_sources.text_blocks}
        for block in parsed.text_blocks:
            if _normalise_local_path(block.source_path) in processed_text_paths:
                continue
            if _text_block_key(block) not in existing_text_keys:
                state.parsed_sources.text_blocks.append(block)
                existing_text_keys.add(_text_block_key(block))

        existing_heading_keys = {
            (candidate.source_path, candidate.page, candidate.line_index, candidate.text)
            for candidate in state.parsed_sources.heading_candidates
        }
        for candidate in parsed.heading_candidates:
            if _normalise_local_path(candidate.source_path) in processed_section_paths:
                continue
            key = (candidate.source_path, candidate.page, candidate.line_index, candidate.text)
            if key not in existing_heading_keys:
                state.parsed_sources.heading_candidates.append(candidate)
                existing_heading_keys.add(key)

        existing_table_keys = {_table_key(table) for table in state.parsed_sources.tables}
        for table in parsed.tables:
            if _normalise_local_path(table.source_path) in processed_table_paths:
                continue
            if _table_key(table) not in existing_table_keys:
                state.parsed_sources.tables.append(table)
                existing_table_keys.add(_table_key(table))

        state.parsed_sources.file_titles.update(parsed.file_titles)
        for warning in parsed.parser_warnings:
            if warning not in state.parsed_sources.parser_warnings:
                state.parsed_sources.parser_warnings.append(warning)
                state.processing_log.append(f"Parser warning: {warning}")
        state.parsed_sources.table_extraction_status.extend(parsed.table_extraction_status)
        state.processing_log.append(
            f"Source parsing completed: text_blocks={len(state.parsed_sources.text_blocks)}, "
            f"heading_candidates={len(state.parsed_sources.heading_candidates)}, "
            f"tables={len(state.parsed_sources.tables)}, "
            f"parser_warnings={len(state.parsed_sources.parser_warnings)}."
        )

    def _extract_charts(
        self,
        state: AgentState,
        step_monitor: AgentMonitor | None = None,
        max_figures_per_pdf: int | None = None,
        max_workers: int | None = None,
    ) -> None:
        """Figure branch: locate -> classify (VL) -> extract (VL) -> validate.

        Applies to every PDF in state.files regardless of origin (user upload,
        arXiv download, or ingested supplement). Deterministic validation issues
        are merged into the quality report; suspicious charts are marked
        needs_review for the human-in-the-loop pass.
        """
        processed_figure_paths = _completed_artifact_paths(state, {"parse_figure"})
        pdf_files = [
            uploaded
            for uploaded in state.files
            if uploaded.path.suffix.lower() == ".pdf"
            and _normalise_local_path(uploaded.path) not in processed_figure_paths
        ]
        if not pdf_files:
            state.processing_log.append(
                "Figure chart extraction skipped: no unprocessed PDF files are available."
            )
            return
        if not self.llm_client.configured:
            state.processing_log.append(
                "Figure chart extraction skipped: Qwen-VL is not configured "
                "(set DASHSCOPE_API_KEY and QWEN_VL_MODEL)."
            )
            return

        figures_dir = state.output_dir / state.task_id / "figures"
        def locate_one(uploaded):
            try:
                return locate_figures(
                    uploaded,
                    figures_dir,
                    max_pages=None,
                    max_figures=max_figures_per_pdf,
                ), None
            except Exception as exc:
                return [], f"Figure location failed for {uploaded.filename}: {exc}"

        location_workers = _worker_count(
            max_workers,
            "SCIDATA_CHART_MAX_WORKERS",
            2,
            len(pdf_files),
        )
        location_results = _run_ordered_parallel(pdf_files, locate_one, location_workers)
        assets = []
        for located, warning in location_results:
            assets.extend(located)
            if warning:
                state.processing_log.append(warning)
        state.parsed_sources.figure_assets.extend(assets)
        if not assets:
            state.processing_log.append(
                f"Figure chart extraction completed: no figures with captions were located in {len(pdf_files)} PDF(s)."
            )
            return

        def process_one(asset):
            try:
                classification = self.llm_nodes.classify_chart(asset)
            except Exception as exc:
                return None, None, 0, (
                    f"Chart classification failed for {asset.label or asset.figure_id} "
                    f"({asset.source_file} p{asset.page}): {exc}"
                )
            if not classification.get("contains_data"):
                return None, None, 1, None
            try:
                extraction = self.llm_nodes.extract_chart_data(asset, classification["chart_type"])
            except Exception as exc:
                return None, None, 0, (
                    f"Chart extraction failed for {asset.label or asset.figure_id} "
                    f"({asset.source_file} p{asset.page}): {exc}"
                )
            return extraction, validate_chart_extraction(extraction, asset), 0, None

        chart_workers = _worker_count(
            max_workers,
            "SCIDATA_CHART_MAX_WORKERS",
            2,
            len(assets),
        )
        def on_chart_completed(index, asset, result, completed):
            if step_monitor:
                step_monitor.emit(
                    "progress",
                    "figure_chart_extraction",
                    "running",
                    f"figure {completed}/{len(assets)} completed: {asset.label or asset.figure_id} "
                    f"({asset.source_file} p{asset.page}).",
                    {
                        "progress_index": completed,
                        "progress_total": len(assets),
                        "source_file": asset.source_file,
                        "page": asset.page,
                    },
                )

        chart_results = _run_ordered_parallel(
            assets,
            process_one,
            chart_workers,
            on_completed=on_chart_completed,
        )

        extractions = []
        validations = []
        skipped_non_data = 0
        for extraction, validation, skipped, warning in chart_results:
            if warning:
                state.processing_log.append(warning)
            if extraction is not None and validation is not None:
                extractions.append(extraction)
                validations.append(validation)
            skipped_non_data += skipped

        state.chart_extractions.extend(extractions)
        state.chart_validations.extend(validations)
        needs_review = sum(1 for validation in validations if validation.needs_review)
        state.processing_log.append(
            "Figure chart extraction completed: "
            f"pdfs={len(pdf_files)}, figures_detected={len(assets)}, "
            f"non_data_figures_skipped={skipped_non_data}, charts_extracted={len(extractions)}, "
            f"charts_needs_review={needs_review}, vl_model={self.llm_client.vl_model}."
        )

    def _interpret_sections(self, state: AgentState) -> None:
        processed_section_paths = _completed_artifact_paths(state, {"parse_pdf_sections"})
        text_blocks = [
            block
            for block in state.parsed_sources.text_blocks
            if _normalise_local_path(block.source_path) not in processed_section_paths
        ]
        heading_candidates = [
            candidate
            for candidate in state.parsed_sources.heading_candidates
            if _normalise_local_path(candidate.source_path) not in processed_section_paths
        ]
        if not text_blocks:
            state.processing_log.append(
                "Section interpretation skipped: all available text was already section-parsed by an artifact action."
            )
            return
        if not state.parsed_sources.text_blocks:
            state.processing_log.append("Section interpretation skipped: no text blocks parsed.")
            return
        if not heading_candidates:
            state.parsed_sources.section_plan = fallback_section_plan_from_candidates([])
            new_blocks = build_section_blocks_from_plan(
                text_blocks,
                state.parsed_sources.section_plan,
            )
            state.parsed_sources.section_blocks.extend(new_blocks)
            state.processing_log.append(
                "Section interpretation used page fallback: no heading candidates were extracted; "
                f"section_blocks={len(state.parsed_sources.section_blocks)}."
            )
            return
        try:
            state.parsed_sources.section_plan = self.llm_nodes.interpret_sections(
                state.research_question,
                heading_candidates,
            )
        except Exception as exc:
            if not self.allow_rule_fallback:
                raise
            state.parsed_sources.section_plan = fallback_section_plan_from_candidates(
                heading_candidates
            )
            state.processing_log.append(
                "Qwen Section Interpreter failed; deterministic section fallback was used "
                f"for local testing only: {exc}"
            )

        new_blocks = build_section_blocks_from_plan(
            text_blocks,
            state.parsed_sources.section_plan,
        )
        state.parsed_sources.section_blocks.extend(new_blocks)
        section_types = sorted({block.section_type for block in state.parsed_sources.section_blocks})
        plan_sections = state.parsed_sources.section_plan.sections if state.parsed_sources.section_plan else []
        ignored = state.parsed_sources.section_plan.ignored_candidates if state.parsed_sources.section_plan else []
        state.processing_log.append(
            "Section interpretation completed: "
            f"heading_candidates={len(state.parsed_sources.heading_candidates)}, "
            f"planned_sections={len(plan_sections)}, "
            f"ignored_candidates={len(ignored)}, "
            f"section_blocks={len(state.parsed_sources.section_blocks)}, "
            f"section_types={', '.join(section_types) if section_types else 'none'}, "
            f"used_llm={state.parsed_sources.section_plan.used_llm if state.parsed_sources.section_plan else False}."
        )

    def _extract(
        self,
        state: AgentState,
        step_monitor: AgentMonitor | None = None,
        max_text_blocks: int | None = None,
        reuse_dynamic_records: bool = True,
        max_text_workers: int | None = None,
        max_table_workers: int | None = None,
    ) -> None:
        if reuse_dynamic_records:
            derived_records = scientific_records_from_dynamic(
                state.clean_dynamic_records or state.dynamic_records
            )
            if derived_records:
                state.candidate_records = derived_records
                self._backfill_paper_titles(state, state.candidate_records)
                state.processing_log.append(
                    "Metric extraction reused schema-driven dynamic records: "
                    f"derived_metrics={len(derived_records)}; duplicate PDF/table LLM pass skipped."
                )
                return
        if not state.task_plan:
            raise RuntimeError("Task plan missing before extraction.")
        source_blocks = state.parsed_sources.section_blocks or state.parsed_sources.text_blocks
        selected_blocks = _selected_block_count(len(source_blocks), max_text_blocks)
        if selected_blocks < len(source_blocks):
            state.processing_log.append(
                f"Record extraction limited to top-ranked {'section' if state.parsed_sources.section_blocks else 'text'} blocks: "
                f"selected={selected_blocks}, total={len(source_blocks)}."
            )
        if state.parsed_sources.section_blocks:
            state.processing_log.append("Record extraction using section-aware blocks.")
        text_records = self.llm_nodes.extract_from_text_blocks_limited(
            state.task_plan,
            source_blocks,
            max_blocks=max_text_blocks,
            progress_callback=_progress_callback(step_monitor, "record_extraction"),
            max_workers=max_text_workers,
        )
        table_records = self.llm_nodes.extract_from_tables(
            state.task_plan,
            state.parsed_sources.tables,
            max_workers=max_table_workers,
        )
        state.candidate_records = text_records + table_records
        self._backfill_paper_titles(state, state.candidate_records)
        state.processing_log.extend(self.llm_nodes.extraction_warnings)
        state.processing_log.append(
            f"Qwen Record Extractor completed: text_records={len(text_records)}, "
            f"table_records={len(table_records)}, candidates={len(state.candidate_records)}, "
            f"skipped_blocks={len(self.llm_nodes.extraction_warnings)}."
        )

    def _extract_dynamic(
        self,
        state: AgentState,
        step_monitor: AgentMonitor | None = None,
        max_text_blocks: int | None = None,
        max_text_workers: int | None = None,
        max_table_workers: int | None = None,
    ) -> None:
        if not state.dynamic_extraction_plan:
            raise RuntimeError("Dynamic extraction plan missing before dynamic extraction.")
        source_blocks = state.parsed_sources.section_blocks or state.parsed_sources.text_blocks
        selected_blocks = _selected_block_count(len(source_blocks), max_text_blocks)
        if selected_blocks < len(source_blocks):
            state.processing_log.append(
                f"Dynamic extraction limited to top-ranked {'section' if state.parsed_sources.section_blocks else 'text'} blocks: "
                f"selected={selected_blocks}, total={len(source_blocks)}."
            )
        if state.parsed_sources.section_blocks:
            state.processing_log.append("Dynamic extraction using section-aware blocks.")
        text_records = self.llm_nodes.extract_dynamic_from_text_blocks(
            state.dynamic_extraction_plan,
            source_blocks,
            max_blocks=max_text_blocks,
            progress_callback=_progress_callback(step_monitor, "dynamic_extraction"),
            max_workers=max_text_workers,
        )
        table_records = self.llm_nodes.extract_dynamic_from_tables(
            state.dynamic_extraction_plan,
            state.parsed_sources.tables,
            max_workers=max_table_workers,
        )
        state.dynamic_records = text_records + table_records
        self._backfill_paper_titles(state, state.dynamic_records)
        state.clean_dynamic_records, state.needs_review_records = curate_dynamic_records(
            state.dynamic_records,
            state.source_discovery_plan,
        )
        state.processing_log.extend(
            warning for warning in self.llm_nodes.extraction_warnings if "dynamic extraction" in warning
        )
        state.processing_log.append(
            f"Qwen Dynamic Extractor completed: dynamic_records={len(state.dynamic_records)}, "
            f"clean_dynamic_records={len(state.clean_dynamic_records)}, "
            f"needs_review={len(state.needs_review_records)}, "
            f"tables={len({record.table_name for record in state.clean_dynamic_records})}."
        )

    def _backfill_paper_titles(self, state: AgentState, records: list[Any]) -> None:
        """Fill empty paper_title fields from the per-file title map.

        Each PDF is parsed once and its full title is stored in
        state.parsed_sources.file_titles. Records extracted from that file may
        have left paper_title empty (especially table/rule-based records), so
        this pass propagates the known title where available.
        """
        file_titles = state.parsed_sources.file_titles if state.parsed_sources else {}
        if not file_titles:
            return
        for record in records:
            if getattr(record, "paper_title", None):
                continue
            source_file = getattr(record, "source_file", None)
            if not source_file:
                continue
            # Try exact filename first, then basename.
            title = file_titles.get(source_file) or file_titles.get(Path(source_file).name)
            if title:
                record.paper_title = title

    def _normalize(self, state: AgentState) -> None:
        before = len(state.candidate_records)
        state.final_records = normalize_records(state.candidate_records)
        after = len(state.final_records)
        state.processing_log.append(
            f"Schema alignment and normalization completed: candidates={before}, "
            f"strict_metric_records_after_cleaning={after}. "
            "Non-numeric or non-metric facts are kept in dynamic tables instead of result.csv."
        )

    def _trace(self, state: AgentState) -> None:
        state.sources = build_source_summaries(state.parsed_sources, state.final_records)
        state.processing_log.append(
            f"Provenance tracking completed: source_summaries={len(state.sources)}. "
            "Each record keeps source_file/source_type/page/evidence_text when available."
        )

    def _quality_check(self, state: AgentState) -> None:
        validation_records = _records_for_llm_validation(state.final_records)
        llm_issues = self.llm_nodes.validate_records(validation_records)
        if len(validation_records) < len(state.final_records):
            state.processing_log.append(
                "LLM quality validation used a risk-ranked sample: "
                f"selected={len(validation_records)}, total={len(state.final_records)}; "
                "deterministic validation still covered every record."
            )
        target_fields = state.task_plan.target_fields if state.task_plan else None
        state.quality_report = build_quality_report(
            state.final_records,
            llm_issues=llm_issues,
            target_fields=target_fields,
            dynamic_records=state.clean_dynamic_records,
            dynamic_plan=state.dynamic_extraction_plan,
            text_blocks=state.parsed_sources.text_blocks,
            table_blocks=state.parsed_sources.tables,
            mutate_records=True,
        )
        state.cross_modal_checks = build_cross_modal_checks(
            state.parsed_sources.text_blocks,
            state.parsed_sources.tables,
            state.parsed_sources.figure_assets,
            state.chart_extractions,
        )
        for check in state.cross_modal_checks:
            if check.status in {"partial", "not_comparable"}:
                for issue in check.issues:
                    state.quality_report.issues.append(
                        QualityIssue(
                            record_id=check.subject_id,
                            level="warning" if check.status == "partial" else "info",
                            field="cross_modal",
                            message=f"[Cross-modal/{check.status}] {issue}",
                        )
                    )
        if state.cross_modal_checks:
            state.quality_report.issue_count = len(state.quality_report.issues)
            state.quality_report.warning_count = sum(
                1 for issue in state.quality_report.issues if issue.level == "warning"
            )
            state.quality_report.error_count = sum(
                1 for issue in state.quality_report.issues if issue.level == "error"
            )
            state.quality_report.notes.append(
                f"Cross-modal audit merged: checks={len(state.cross_modal_checks)}, "
                f"supported={sum(1 for check in state.cross_modal_checks if check.status == 'supported')}, "
                f"partial={sum(1 for check in state.cross_modal_checks if check.status == 'partial')}, "
                f"not_comparable={sum(1 for check in state.cross_modal_checks if check.status == 'not_comparable')}."
            )
        review_issue_ids = {
            issue.record_id
            for issue in state.quality_report.issues
            if issue.record_id and issue.level in {"warning", "error"}
        }
        review_by_id = {record.record_id: record for record in state.needs_review_records}
        for record in state.clean_dynamic_records:
            if record.record_id in review_issue_ids or record.warnings:
                review_by_id[record.record_id] = record
        state.needs_review_records = list(review_by_id.values())
        for validation in state.chart_validations:
            for issue in validation.issues:
                state.quality_report.issues.append(
                    QualityIssue(
                        record_id=validation.figure_id,
                        level=issue.severity,
                        field="figure_chart",
                        message=f"[图表校验/{issue.code}] {issue.message}",
                    )
                )
        if state.chart_validations:
            state.quality_report.issue_count = len(state.quality_report.issues)
            state.quality_report.warning_count = sum(
                1 for issue in state.quality_report.issues if issue.level == "warning"
            )
            state.quality_report.error_count = sum(
                1 for issue in state.quality_report.issues if issue.level == "error"
            )
            state.quality_report.notes.append(
                f"Chart validation merged: figures={len(state.parsed_sources.figure_assets)}, "
                f"extracted={len(state.chart_extractions)}, "
                f"needs_review={sum(1 for v in state.chart_validations if v.needs_review)}."
            )
        # Build provenance before queueing review risks so record items can
        # point directly to their existing evidence trace.
        state.evidence_traces = build_evidence_traces(state)
        state.review_queue = build_review_queue(state)
        state.quality_report.review_count = len(state.review_queue)
        state.processing_log.append(
            f"Human review queue built: items={len(state.review_queue)}, "
            f"high={sum(1 for item in state.review_queue if item.priority == 'high')}, "
            f"medium={sum(1 for item in state.review_queue if item.priority == 'medium')}, "
            f"low={sum(1 for item in state.review_queue if item.priority == 'low')}."
        )
        state.processing_log.append(
            "Quality validation completed: "
            f"issues={state.quality_report.issue_count}, "
            f"warnings={state.quality_report.warning_count}, "
            f"errors={state.quality_report.error_count}, "
            f"conflicts={state.quality_report.conflict_count}, "
            f"evidence_coverage={state.quality_report.evidence_coverage}, "
            f"value_evidence_coverage={state.quality_report.value_evidence_coverage}."
        )

    def _export(self, state: AgentState) -> None:
        state.export_files = export_results(state)
        state.processing_log.append(
            "Export completed: generated CSV, JSON, source_selection, source_triage, processing log, and quality_report files."
        )

    def _append_llm_trace(self, state: AgentState, *, start_index: int = 0) -> None:
        traces = self.llm_client.traces[start_index:]
        if not traces:
            return
        for trace in traces:
            if trace.success:
                state.processing_log.append(
                    f"LLM trace: node={trace.node}, model={trace.model}, prompt_chars={trace.prompt_chars}, "
                    f"response_chars={trace.response_chars}, prompt_tokens={trace.prompt_tokens}, "
                    f"completion_tokens={trace.completion_tokens}, total_tokens={trace.total_tokens}, "
                    f"request_id={trace.request_id}, elapsed_ms={trace.elapsed_ms}."
                )
            else:
                state.processing_log.append(
                    f"LLM trace failed: node={trace.node}, model={trace.model}, error={trace.error}."
                )

    def _build_result(self, state: AgentState, status: str) -> AgentResult:
        warnings = state.quality_report.warning_count + state.quality_report.error_count
        return AgentResult(
            task_id=state.task_id,
            status=status,  # type: ignore[arg-type]
            research_question=state.research_question,
            task_plan=state.task_plan or fallback_plan_task(state.research_question),
            source_discovery_plan=state.source_discovery_plan,
            arxiv_search_plan=state.arxiv_search_plan,
            multi_source_search_plan=state.multi_source_search_plan,
            source_selection_plan=state.source_selection_plan,
            source_triage_decisions=state.source_triage_decisions,
            source_insights=state.source_insights,
            source_catalog=state.source_catalog,
            evidence_traces=state.evidence_traces,
            artifact_action_plan=state.artifact_action_plan,
            artifact_action_results=state.artifact_action_results,
            artifact_action_history=state.artifact_action_history,
            dynamic_extraction_plan=state.dynamic_extraction_plan,
            connector_status=state.connector_status,
            summary=AgentSummary(
                files_processed=len(state.files),
                text_blocks_processed=len(state.parsed_sources.text_blocks),
                heading_candidates_extracted=len(state.parsed_sources.heading_candidates),
                section_blocks_processed=len(state.parsed_sources.section_blocks),
                tables_processed=len(state.parsed_sources.tables),
                records_extracted=len(state.candidate_records),
                records_after_cleaning=len(state.final_records),
                dynamic_records_extracted=len(state.clean_dynamic_records or state.dynamic_records),
                dynamic_tables_count=len({record.table_name for record in (state.clean_dynamic_records or state.dynamic_records)}),
                figures_detected=len(state.parsed_sources.figure_assets),
                charts_extracted=len(state.chart_extractions),
                charts_needs_review=sum(1 for validation in state.chart_validations if validation.needs_review),
                warnings=warnings,
            ),
            records=state.final_records,
            dynamic_records=state.clean_dynamic_records or state.dynamic_records,
            dynamic_records_raw=state.dynamic_records,
            needs_review_records=state.needs_review_records,
            review_queue=state.review_queue,
            figures=state.parsed_sources.figure_assets,
            chart_extractions=state.chart_extractions,
            chart_validations=state.chart_validations,
            cross_modal_checks=state.cross_modal_checks,
            field_schema=FIELD_SCHEMA,
            sources=state.sources,
            processing_log=state.processing_log,
            quality_report=state.quality_report,
            coverage_report=state.coverage_report,
            export_files=state.export_files,
        )


def _arxiv_plan_from_multi_source_plan(state: AgentState) -> ArxivSearchPlan | None:
    if not state.multi_source_search_plan:
        return None
    arxiv_requests = [
        request
        for request in state.multi_source_search_plan.search_requests
        if request.connector_name == "arxiv"
    ]
    if not arxiv_requests:
        return None
    return ArxivSearchPlan(
        research_goal=state.multi_source_search_plan.research_goal,
        should_search_arxiv=state.multi_source_search_plan.should_search,
        search_intent="Synthesized from the LLM multi-source search plan.",
        queries=[
            {
                "query": request.query,
                "purpose": request.purpose,
                "max_results": request.max_results,
            }
            for request in arxiv_requests
        ],
        selection_criteria=state.multi_source_search_plan.selection_criteria,
        notes=[
            "This arXiv plan was derived from multi_source_search_plan for backward-compatible exports."
        ],
    )


def _state_snapshot(state: AgentState) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "files_count": len(state.files),
        "files": [
            {
                "filename": uploaded.filename,
                "path": str(uploaded.path),
                "content_type": uploaded.content_type,
            }
            for uploaded in state.files[:5]
        ],
        "text_blocks_count": len(state.parsed_sources.text_blocks),
        "heading_candidates_count": len(state.parsed_sources.heading_candidates),
        "section_blocks_count": len(state.parsed_sources.section_blocks),
        "tables_count": len(state.parsed_sources.tables),
        "figure_assets_count": len(state.parsed_sources.figure_assets),
        "chart_extractions_count": len(state.chart_extractions),
        "charts_needs_review_count": sum(1 for validation in state.chart_validations if validation.needs_review),
        "candidate_records_count": len(state.candidate_records),
        "final_records_count": len(state.final_records),
        "dynamic_records_count": len(state.dynamic_records),
        "clean_dynamic_records_count": len(state.clean_dynamic_records),
        "needs_review_records_count": len(state.needs_review_records),
        "source_summaries_count": len(state.sources),
        "source_insights_count": len(state.source_insights),
        **source_catalog_summary(state.source_catalog),
        "artifact_action_results_count": len(state.artifact_action_results),
        "artifact_action_iterations_count": len(state.artifact_action_history),
        "processing_log_tail": state.processing_log[-5:],
        "coverage_report": state.coverage_report.model_dump(mode="json"),
    }
    if state.artifact_action_plan:
        snapshot["artifact_action_plan"] = {
            "iteration": state.artifact_action_plan.iteration,
            "should_continue": state.artifact_action_plan.should_continue,
            "actions_count": len(state.artifact_action_plan.actions),
            "actions": [
                {
                    "action_id": action.action_id,
                    "artifact_id": action.artifact_id,
                    "action": action.action,
                    "priority": action.priority,
                }
                for action in state.artifact_action_plan.actions[:12]
            ],
        }
    if state.artifact_action_results:
        snapshot["artifact_action_results"] = [
            result.model_dump(mode="json") for result in state.artifact_action_results[:12]
        ]
    if state.task_plan:
        snapshot["task_plan"] = {
            "domain": state.task_plan.domain,
            "research_goal": state.task_plan.research_goal,
            "target_fields": state.task_plan.target_fields,
            "dynamic_schema_keys": list(state.task_plan.dynamic_schema.keys()),
            "source_requirements": state.task_plan.source_requirements,
        }
        snapshot["domain"] = state.task_plan.domain
    if state.dynamic_extraction_plan:
        snapshot["dynamic_extraction_plan"] = {
            "domain": state.dynamic_extraction_plan.domain,
            "task_type": state.dynamic_extraction_plan.task_type,
            "user_focus": state.dynamic_extraction_plan.user_focus,
            "tables": [
                {
                    "table_name": table.table_name,
                    "entity_type": table.entity_type,
                    "fields": [field.name for field in table.fields],
                }
                for table in state.dynamic_extraction_plan.dynamic_tables[:8]
            ],
        }
    if state.source_discovery_plan:
        arxiv_papers = [
            source
            for source in state.source_discovery_plan.candidate_sources
            if source.source_type == "paper" and source.metadata.get("provider") == "arxiv"
        ]
        downloaded = [
            source
            for source in arxiv_papers
            if source.metadata.get("downloaded_path")
        ]
        snapshot["source_discovery"] = {
            "domain": state.source_discovery_plan.domain,
            "recommended_keywords": state.source_discovery_plan.recommended_keywords[:8],
            "target_data_types": state.source_discovery_plan.target_data_types,
            "candidate_sources_count": len(state.source_discovery_plan.candidate_sources),
            "arxiv_papers_count": len(arxiv_papers),
            "downloaded_pdfs_count": len(downloaded),
            "sample_sources": [
                {
                    "title": source.title,
                    "source_type": source.source_type,
                    "url": source.url,
                    "pdf_url": source.metadata.get("pdf_url"),
                    "downloaded_path": source.metadata.get("downloaded_path"),
                }
                for source in state.source_discovery_plan.candidate_sources[:5]
            ],
            "notes_tail": state.source_discovery_plan.notes[-5:],
        }
        snapshot["candidate_sources_count"] = len(state.source_discovery_plan.candidate_sources)
        snapshot["arxiv_papers_count"] = len(arxiv_papers)
        snapshot["downloaded_pdfs_count"] = len(downloaded)
    if state.source_triage_decisions:
        action_counts: dict[str, int] = {}
        for decision in state.source_triage_decisions:
            action_counts[decision.recommended_action] = action_counts.get(decision.recommended_action, 0) + 1
        snapshot["source_triage"] = {
            "decisions_count": len(state.source_triage_decisions),
            "action_counts": action_counts,
            "ingest_selected": sum(1 for decision in state.source_triage_decisions if decision.should_ingest),
            "sample_decisions": [
                decision.model_dump(mode="json")
                for decision in state.source_triage_decisions[:8]
            ],
        }
    if state.source_insights:
        snapshot["source_insights"] = {
            "insights_count": len(state.source_insights),
            "insight_types": sorted({insight.insight_type for insight in state.source_insights}),
            "sample_insights": [
                {
                    "title": insight.title,
                    "provider": insight.provider,
                    "insight_type": insight.insight_type,
                    "chars": len(insight.content),
                    "preview": insight.content[:500],
                }
                for insight in state.source_insights[:6]
            ],
        }
    if state.multi_source_search_plan:
        snapshot["multi_source_search_plan"] = {
            "should_search": state.multi_source_search_plan.should_search,
            "requests_count": len(state.multi_source_search_plan.search_requests),
            "connectors": sorted({request.connector_name for request in state.multi_source_search_plan.search_requests}),
            "queries": [
                {
                    "connector": request.connector_name,
                    "source_type": request.source_type,
                    "query": request.query,
                    "max_results": request.max_results,
                }
                for request in state.multi_source_search_plan.search_requests[:8]
            ],
        }
    if state.connector_status:
        snapshot["connector_status"] = state.connector_status[:8]
    if state.source_selection_plan:
        decision_counts: dict[str, int] = {}
        priority_counts: dict[str, int] = {}
        for decision in state.source_selection_plan.decisions:
            decision_counts[decision.decision] = decision_counts.get(decision.decision, 0) + 1
            priority_counts[decision.priority] = priority_counts.get(decision.priority, 0) + 1
        snapshot["source_selection"] = {
            "decisions_count": len(state.source_selection_plan.decisions),
            "decision_counts": decision_counts,
            "priority_counts": priority_counts,
            "time_range_interpreted": state.source_selection_plan.time_range_interpreted,
            "selection_summary": state.source_selection_plan.selection_summary,
            "sample_decisions": [
                decision.model_dump(mode="json")
                for decision in state.source_selection_plan.decisions[:8]
            ],
        }
    if state.parsed_sources.text_blocks:
        snapshot["sample_text_blocks"] = [
            {
                "source_file": block.source_file,
                "page": block.page,
                "chars": len(block.text),
                "preview": block.text[:500],
            }
            for block in state.parsed_sources.text_blocks[:3]
        ]
    if state.parsed_sources.heading_candidates:
        snapshot["sample_heading_candidates"] = [
            candidate.model_dump(mode="json")
            for candidate in state.parsed_sources.heading_candidates[:5]
        ]
    if state.parsed_sources.section_blocks:
        snapshot["sample_section_blocks"] = [
            {
                "source_file": block.source_file,
                "section_title": block.section_title,
                "section_type": block.section_type,
                "page_start": block.page_start,
                "page_end": block.page_end,
                "chars": len(block.text),
                "preview": block.text[:500],
            }
            for block in state.parsed_sources.section_blocks[:5]
        ]
    if state.parsed_sources.tables:
        snapshot["sample_tables"] = [
            {
                "source_file": table.source_file,
                "columns": table.columns[:20],
                "rows_count": len(table.rows),
                "sample_rows": table.rows[:3],
            }
            for table in state.parsed_sources.tables[:3]
        ]
    if state.candidate_records:
        snapshot["sample_candidate_records"] = [
            record.model_dump(mode="json")
            for record in state.candidate_records[:5]
        ]
    if state.final_records:
        snapshot["sample_records"] = [
            record.model_dump(mode="json")
            for record in state.final_records[:5]
        ]
    if state.dynamic_records:
        snapshot["sample_dynamic_records"] = [
            record.model_dump(mode="json")
            for record in state.dynamic_records[:5]
        ]
    if state.clean_dynamic_records:
        snapshot["sample_clean_dynamic_records"] = [
            record.model_dump(mode="json")
            for record in state.clean_dynamic_records[:5]
        ]
    if state.sources:
        snapshot["sample_source_summaries"] = [
            source.model_dump(mode="json")
            for source in state.sources[:5]
        ]
    if state.quality_report:
        snapshot["quality_report"] = {
            "record_count": state.quality_report.record_count,
            "issue_count": state.quality_report.issue_count,
            "warning_count": state.quality_report.warning_count,
            "error_count": state.quality_report.error_count,
            "conflict_count": state.quality_report.conflict_count,
            "evidence_coverage": state.quality_report.evidence_coverage,
            "value_evidence_coverage": state.quality_report.value_evidence_coverage,
            "sample_issues": [
                issue.model_dump(mode="json")
                for issue in state.quality_report.issues[:5]
            ],
        }
        snapshot["issue_count"] = state.quality_report.issue_count
        snapshot["warning_count"] = state.quality_report.warning_count
        snapshot["error_count"] = state.quality_report.error_count
        snapshot["conflict_count"] = state.quality_report.conflict_count
    if state.export_files:
        exports = state.export_files.model_dump(mode="json", by_alias=True)
        exports = {key: value for key, value in exports.items() if value}
        if exports:
            snapshot["export_files"] = exports
    return snapshot


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _merge_source_selection_plans(plans: list[SourceSelectionPlan]) -> SourceSelectionPlan:
    """Merge independent source-selector batches without losing decisions."""
    if not plans:
        raise ValueError("At least one source-selection plan is required.")
    decisions_by_id = {}
    notes: list[str] = []
    summaries: list[str] = []
    time_range = None
    research_goal = plans[0].research_goal
    for plan in plans:
        time_range = time_range or plan.time_range_interpreted
        if plan.selection_summary and plan.selection_summary not in summaries:
            summaries.append(plan.selection_summary)
        for note in plan.notes:
            if note and note not in notes:
                notes.append(note)
        for decision in plan.decisions:
            existing = decisions_by_id.get(decision.source_id)
            if existing is None or decision.priority_score > existing.priority_score:
                decisions_by_id[decision.source_id] = decision
    return SourceSelectionPlan(
        research_goal=research_goal,
        selection_summary=" ".join(summaries) if summaries else None,
        time_range_interpreted=time_range,
        decisions=list(decisions_by_id.values()),
        notes=notes,
    )


def _unlimited_or_positive(value: int | None) -> int | None:
    """Treat omitted and legacy zero values as unlimited processing."""
    if value is None or value <= 0:
        return None
    return int(value)


def _selected_block_count(total_blocks: int, max_blocks: int | None) -> int:
    if max_blocks is None or max_blocks <= 0:
        return total_blocks
    return min(total_blocks, max_blocks)


def _arxiv_download_progress_callback(
    monitor: AgentMonitor | None,
    state: AgentState,
):
    reported_bytes: dict[str, int] = {}

    def callback(status: str, data: dict[str, Any]) -> None:
        if status == "progress":
            path = str(data.get("path") or "")
            current = int(data.get("bytes") or 0)
            previous = reported_bytes.get(path, 0)
            if current - previous < 5 * 1024 * 1024:
                return
            reported_bytes[path] = current
        title = str(data.get("title") or data.get("path") or "arXiv PDF")
        index = data.get("index")
        total = data.get("total")
        message = f"arXiv PDF {status}: {title}"
        if index is not None and total is not None:
            message = f"arXiv PDF {status} ({index}/{total}): {title}"
        if status == "progress":
            message += f"; bytes={data.get('bytes', 0)}"
        state.processing_log.append(message)
        if monitor:
            monitor.emit("progress", "arxiv_pdf_ingestion", status, message, data)

    return callback


def _previous_arxiv_download_dirs(output_dir: Path, current_task_id: str) -> list[Path]:
    """Find completed PDF directories from earlier timestamped tasks."""
    if not output_dir.is_dir():
        return []
    previous: list[Path] = []
    try:
        task_dirs = output_dir.iterdir()
    except OSError:
        return previous
    for task_dir in task_dirs:
        if not task_dir.is_dir() or task_dir.name in {current_task_id, "_cache"}:
            continue
        candidate = task_dir / "downloads" / "arxiv"
        if candidate.is_dir():
            previous.append(candidate)
    return previous


def _completed_artifact_paths(state: AgentState, actions: set[str]) -> set[str]:
    completed_ids = {
        result.artifact_id
        for result in [
            *state.artifact_action_results,
            *[
                result
                for iteration in state.artifact_action_history
                for result in iteration.results
            ],
        ]
        if result.status == "completed"
        and result.action in actions
        and result.artifact_id
    }
    paths: set[str] = set()
    for entry in state.source_catalog:
        for artifact in entry.artifacts:
            if artifact.artifact_id in completed_ids and artifact.local_path:
                paths.add(_normalise_local_path(artifact.local_path))
    return paths


def _normalise_local_path(value: str | Path | None) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(Path(value).expanduser().resolve()).casefold()
    except (OSError, RuntimeError, TypeError):
        return str(value).casefold()


def _text_block_key(block) -> tuple[str, int | None, str]:
    return (_normalise_local_path(block.source_path), block.page, block.text)


def _table_key(table) -> tuple[str, int | None, tuple[str, ...], str]:
    return (
        _normalise_local_path(table.source_path),
        table.page,
        tuple(table.columns),
        repr(table.rows[:3]),
    )


def _progress_callback(monitor: AgentMonitor | None, step: str):
    if monitor is None:
        return None

    def callback(index: int, total: int, block, records_so_far: int) -> None:
        monitor.emit(
            "progress",
            step,
            "running",
            f"{step} text block {index}/{total}.",
            {
                "progress_index": index,
                "progress_total": total,
                "source_file": getattr(block, "source_file", None),
                "page": getattr(block, "page", None),
                "section_title": getattr(block, "section_title", None),
                "section_type": getattr(block, "section_type", None),
                "page_start": getattr(block, "page_start", None),
                "page_end": getattr(block, "page_end", None),
                "chars": len(getattr(block, "text", "") or ""),
                "records_so_far": records_so_far,
            },
        )

    return callback


def _records_for_llm_validation(records: list[ScientificRecord]) -> list[ScientificRecord]:
    """Select a bounded, risk-ranked sample for the expensive LLM review.

    Deterministic quality checks still inspect the complete record set. This
    second-opinion pass prioritizes records already carrying extraction risk and
    then fills the remaining capacity in stable source order.
    """
    try:
        limit = int(os.getenv("SCIDATA_LLM_VALIDATE_MAX_RECORDS", "12"))
    except ValueError:
        limit = 12
    limit = max(0, min(limit, 100))
    if len(records) <= limit:
        return records

    def risk(item: tuple[int, ScientificRecord]) -> tuple[int, int]:
        index, record = item
        score = 0
        score += 4 if record.warnings else 0
        score += 3 if not record.evidence_text else 0
        score += 2 if record.page is None and record.source_type.value.startswith("pdf") else 0
        score += 2 if record.confidence < 0.75 else 0
        score += 1 if record.metric_value is None else 0
        return (-score, index)

    ranked = sorted(enumerate(records), key=risk)
    return [record for _, record in ranked[:limit]]


def _result_snapshot(result: AgentResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "summary": result.summary.model_dump(mode="json"),
        "export_files": result.export_files.model_dump(mode="json", by_alias=True),
        "quality_report": {
            "record_count": result.quality_report.record_count,
            "issue_count": result.quality_report.issue_count,
            "warning_count": result.quality_report.warning_count,
            "error_count": result.quality_report.error_count,
            "conflict_count": result.quality_report.conflict_count,
            "evidence_coverage": result.quality_report.evidence_coverage,
            "value_evidence_coverage": result.quality_report.value_evidence_coverage,
        },
        "sample_records": [
            record.model_dump(mode="json")
            for record in result.records[:5]
        ],
        "dynamic_tables": [
            table.table_name
            for table in result.dynamic_extraction_plan.dynamic_tables[:8]
        ] if result.dynamic_extraction_plan else [],
        "sample_dynamic_records": [
            record.model_dump(mode="json")
            for record in result.dynamic_records[:5]
        ],
    }
