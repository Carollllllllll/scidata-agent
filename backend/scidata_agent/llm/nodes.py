from __future__ import annotations

import json
import re
import time
from typing import Any

from pydantic import ValidationError

from scidata_agent.agent.field_schema import DEFAULT_TARGET_FIELDS
from scidata_agent.agent.planner import plan_task as fallback_plan_task
from scidata_agent.agent.schemas import (
    ArxivSearchPlan,
    ArtifactActionPlan,
    ChartExtraction,
    DynamicExtractionPlan,
    DynamicRecord,
    FigureAsset,
    HeadingCandidate,
    MultiSourceSearchPlan,
    QualityIssue,
    QualityReport,
    ScientificRecord,
    SectionPlan,
    SourceSearchRequest,
    SourceDiscoveryPlan,
    SourceCatalogEntry,
    SourceSelectionPlan,
    SourceType,
    SectionBlock,
    TableBlock,
    TaskPlan,
    TextBlock,
)
from scidata_agent.llm.client import LLMCallError, QwenBailianClient
from scidata_agent.llm.prompts import (
    ARXIV_SEARCH_PLANNER_SYSTEM,
    ARXIV_SEARCH_PLANNER_USER,
    ARTIFACT_ACTION_PLANNER_SYSTEM,
    ARTIFACT_ACTION_PLANNER_USER,
    CHART_CLASSIFIER_SYSTEM,
    CHART_CLASSIFIER_USER,
    CHART_EXTRACTOR_SYSTEM,
    CHART_EXTRACTOR_USER,
    DYNAMIC_EXTRACTOR_SYSTEM,
    DYNAMIC_EXTRACTOR_USER,
    DYNAMIC_PLANNER_SYSTEM,
    DYNAMIC_PLANNER_USER,
    MULTI_SOURCE_SEARCH_PLANNER_SYSTEM,
    MULTI_SOURCE_SEARCH_PLANNER_USER,
    RECORD_EXTRACTOR_SYSTEM,
    RECORD_EXTRACTOR_USER,
    SECTION_INTERPRETER_SYSTEM,
    SECTION_INTERPRETER_USER,
    SOURCE_DISCOVERY_SYSTEM,
    SOURCE_DISCOVERY_USER,
    SOURCE_SELECTOR_SYSTEM,
    SOURCE_SELECTOR_USER,
    TASK_PLANNER_SYSTEM,
    TASK_PLANNER_USER,
    VALIDATOR_SYSTEM,
    VALIDATOR_USER,
)
from scidata_agent.tools.extractor import extract_records_from_tables, extract_records_from_text_blocks
from scidata_agent.tools.source_discovery import fallback_discover_sources


SOURCE_SELECTION_CANDIDATE_LIMIT = 100
DEFAULT_MAX_AUTO_RESOURCES = 30


class QwenAgentNodes:
    """LLM-backed nodes used by the controlled SciData Agent workflow."""

    def __init__(self, client: QwenBailianClient, allow_rule_fallback: bool = False):
        self.client = client
        self.allow_rule_fallback = allow_rule_fallback
        self.node_warnings: list[str] = []
        self.extraction_warnings: list[str] = []

    def _generate_json_with_retries(
        self,
        node: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        attempts: int = 3,
        retry_delay_seconds: float = 2.0,
    ) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self.client.generate_json(node, system_prompt, user_prompt, temperature=temperature)
            except Exception as exc:
                last_exc = exc
                warning = f"LLM call failed: node={node}, attempt={attempt}/{attempts}, error={exc}"
                self.node_warnings.append(warning)
                if attempt < attempts:
                    time.sleep(min(retry_delay_seconds * attempt, 8.0))
        raise LLMCallError(f"LLM node failed after {attempts} attempts: node={node}, error={last_exc}") from last_exc

    def plan_task(self, research_question: str) -> TaskPlan:
        try:
            payload = self._generate_json_with_retries(
                "qwen_task_planner",
                TASK_PLANNER_SYSTEM,
                TASK_PLANNER_USER.format(research_question=research_question),
            )
            if not isinstance(payload, dict):
                raise LLMCallError("Task Planner did not return a JSON object.")
            payload["target_fields"] = _ensure_default_fields(payload.get("target_fields", []))
            payload["need_provenance"] = True
            payload.setdefault("output_format", ["csv", "json"])
            payload.setdefault("assumptions", [])
            payload.setdefault("schema_notes", [])
            return TaskPlan.model_validate(payload)
        except Exception:
            if not self.allow_rule_fallback:
                raise
            plan = fallback_plan_task(research_question)
            plan.assumptions.append("LLM task planning failed; rule fallback was used for local testing only.")
            return plan

    def discover_sources(self, research_question: str) -> SourceDiscoveryPlan:
        try:
            payload = self._generate_json_with_retries(
                "qwen_source_discovery",
                SOURCE_DISCOVERY_SYSTEM,
                SOURCE_DISCOVERY_USER.format(research_question=research_question),
            )
            if not isinstance(payload, dict):
                raise LLMCallError("Source Discovery did not return a JSON object.")
            payload.setdefault("research_goal", research_question)
            payload.setdefault("domain", "general science")
            payload.setdefault("recommended_keywords", [])
            payload.setdefault("target_data_types", [])
            payload.setdefault("dynamic_schema", {})
            payload.setdefault("candidate_sources", [])
            payload.setdefault("notes", [])
            _normalize_candidate_source_types(payload)
            return SourceDiscoveryPlan.model_validate(payload)
        except Exception:
            if not self.allow_rule_fallback:
                raise
            plan = fallback_discover_sources(research_question)
            plan.notes.append("LLM source discovery failed; deterministic fallback was used for local testing only.")
            return plan

    def plan_arxiv_search(
        self,
        research_question: str,
        source_discovery_plan: SourceDiscoveryPlan,
    ) -> ArxivSearchPlan:
        try:
            payload = self._generate_json_with_retries(
                "qwen_arxiv_search_planner",
                ARXIV_SEARCH_PLANNER_SYSTEM,
                ARXIV_SEARCH_PLANNER_USER.format(
                    research_question=research_question,
                    source_discovery_plan_json=source_discovery_plan.model_dump_json(),
                ),
            )
            if not isinstance(payload, dict):
                raise LLMCallError("arXiv Search Planner did not return a JSON object.")
            payload.setdefault("research_goal", research_question)
            payload.setdefault("should_search_arxiv", True)
            payload.setdefault("search_intent", None)
            payload.setdefault("queries", [])
            payload.setdefault("selection_criteria", [])
            payload.setdefault("notes", [])
            plan = ArxivSearchPlan.model_validate(payload)
            if plan.should_search_arxiv and not plan.queries:
                raise LLMCallError("arXiv Search Planner returned no queries.")
            return plan
        except Exception:
            if not self.allow_rule_fallback:
                raise
            return _fallback_arxiv_search_plan(research_question, source_discovery_plan)

    def plan_multi_source_search(
        self,
        research_question: str,
        source_discovery_plan: SourceDiscoveryPlan,
    ) -> MultiSourceSearchPlan:
        try:
            payload = self._generate_json_with_retries(
                "qwen_multi_source_search_planner",
                MULTI_SOURCE_SEARCH_PLANNER_SYSTEM,
                MULTI_SOURCE_SEARCH_PLANNER_USER.format(
                    research_question=research_question,
                    source_discovery_plan_json=source_discovery_plan.model_dump_json(),
                ),
            )
            if not isinstance(payload, dict):
                raise LLMCallError("Multi-source Search Planner did not return a JSON object.")
            payload.setdefault("research_goal", research_question)
            payload.setdefault("domain", source_discovery_plan.domain or "general science")
            payload.setdefault("should_search", True)
            payload.setdefault("search_requests", [])
            payload.setdefault("selection_criteria", [])
            payload.setdefault("notes", [])
            _normalize_multi_source_search_requests(payload)
            plan = MultiSourceSearchPlan.model_validate(payload)
            if plan.should_search and not plan.search_requests:
                raise LLMCallError("Multi-source Search Planner returned no search requests.")
            return plan
        except Exception:
            if not self.allow_rule_fallback:
                raise
            return _fallback_multi_source_search_plan(research_question, source_discovery_plan)

    def select_sources(
        self,
        research_question: str,
        source_discovery_plan: SourceDiscoveryPlan,
        dynamic_plan: DynamicExtractionPlan | None = None,
        multi_source_search_plan: MultiSourceSearchPlan | None = None,
        connector_status: list[dict[str, Any]] | None = None,
        max_auto_resources: int = DEFAULT_MAX_AUTO_RESOURCES,
        candidate_limit: int = SOURCE_SELECTION_CANDIDATE_LIMIT,
    ) -> SourceSelectionPlan:
        candidates = _source_candidate_summaries(source_discovery_plan, limit=candidate_limit)
        payload = self._generate_json_with_retries(
            "qwen_source_selector",
            SOURCE_SELECTOR_SYSTEM,
            SOURCE_SELECTOR_USER.format(
                research_question=research_question,
                dynamic_plan_json=dynamic_plan.model_dump_json() if dynamic_plan else "null",
                multi_source_search_plan_json=multi_source_search_plan.model_dump_json()
                if multi_source_search_plan
                else "null",
                connector_status_json=json.dumps(connector_status or [], ensure_ascii=False, indent=2),
                candidate_sources_json=json.dumps(candidates, ensure_ascii=False, indent=2),
                max_auto_resources=max_auto_resources,
                candidate_limit=candidate_limit,
            ),
            temperature=0.05,
        )
        if not isinstance(payload, dict):
            raise LLMCallError("Source Selector did not return a JSON object.")
        payload.setdefault("research_goal", research_question)
        payload.setdefault("selection_summary", None)
        payload.setdefault("time_range_interpreted", None)
        payload.setdefault("decisions", [])
        payload.setdefault("notes", [])
        _normalize_source_selection_decisions(payload)
        plan = SourceSelectionPlan.model_validate(payload)
        if not plan.decisions:
            raise LLMCallError("Source Selector returned no source decisions.")
        candidate_ids = {item["source_id"] for item in candidates}
        valid_decisions = [decision for decision in plan.decisions if decision.source_id in candidate_ids]
        if not valid_decisions:
            raise LLMCallError("Source Selector returned no decisions matching candidate source IDs.")
        if len(valid_decisions) != len(plan.decisions):
            plan.notes.append(
                f"Dropped {len(plan.decisions) - len(valid_decisions)} source selection decision(s) with unknown source_id."
            )
            plan.decisions = valid_decisions
        plan.notes.append(
            f"Source Selector compared up to {candidate_limit} candidates; executor resource cap is {max_auto_resources}."
        )
        return plan

    def plan_artifact_actions(
        self,
        research_question: str,
        source_catalog: list[SourceCatalogEntry],
        dynamic_plan: DynamicExtractionPlan | None = None,
        quality_report: QualityReport | None = None,
        processing_log: list[str] | None = None,
        connector_failures: list[dict[str, Any]] | None = None,
        iteration: int = 0,
    ) -> ArtifactActionPlan:
        """Ask Qwen which concrete artifact operations should run next."""
        catalog_payload = [
            {
                "source_id": source.source_id,
                "title": source.title,
                "source_type": source.source_type,
                "provider": source.provider,
                "status": source.status,
                "relevance_score": source.relevance_score,
                "artifacts": [artifact.model_dump(mode="json") for artifact in source.artifacts],
            }
            for source in source_catalog
        ]
        payload = self._generate_json_with_retries(
            "qwen_artifact_action_planner",
            ARTIFACT_ACTION_PLANNER_SYSTEM,
            ARTIFACT_ACTION_PLANNER_USER.format(
                research_question=research_question,
                dynamic_plan_json=dynamic_plan.model_dump_json() if dynamic_plan else "null",
                quality_report_json=quality_report.model_dump_json() if quality_report else "null",
                source_catalog_json=json.dumps(catalog_payload, ensure_ascii=False, indent=2),
                processing_log_json=json.dumps((processing_log or [])[-40:], ensure_ascii=False, indent=2),
                connector_failures_json=json.dumps(connector_failures or [], ensure_ascii=False, indent=2),
                iteration=iteration,
            ),
            temperature=0.05,
        )
        if not isinstance(payload, dict):
            raise LLMCallError("Artifact Action Planner did not return a JSON object.")
        payload.setdefault("research_goal", research_question)
        payload.setdefault("iteration", iteration)
        payload.setdefault("should_continue", True)
        payload.setdefault("stop_reason", None)
        payload.setdefault("actions", [])
        payload.setdefault("notes", [])
        plan = ArtifactActionPlan.model_validate(payload)

        artifact_ids = {
            artifact.artifact_id
            for source in source_catalog
            for artifact in source.artifacts
        }
        global_actions = {"search_more", "validate_evidence", "stop"}
        for action in plan.actions:
            if action.artifact_id is not None and action.artifact_id not in artifact_ids:
                raise LLMCallError(
                    f"Artifact Action Planner returned unknown artifact_id={action.artifact_id!r}."
                )
            if action.action in global_actions:
                if action.action == "stop" and action.artifact_id is not None:
                    raise LLMCallError("The stop action must use artifact_id=null.")
            elif action.artifact_id is None:
                raise LLMCallError(
                    f"Artifact action {action.action!r} requires a concrete artifact_id."
                )
        if any(action.action == "stop" for action in plan.actions) and plan.should_continue:
            raise LLMCallError("A plan containing stop must set should_continue=false.")
        return plan

    def plan_dynamic_extraction(self, research_question: str, task_plan: TaskPlan | None = None) -> DynamicExtractionPlan:
        try:
            payload = self._generate_json_with_retries(
                "qwen_dynamic_schema_planner",
                DYNAMIC_PLANNER_SYSTEM,
                DYNAMIC_PLANNER_USER.format(research_question=research_question),
            )
            if not isinstance(payload, dict):
                raise LLMCallError("Dynamic Schema Planner did not return a JSON object.")
            payload.setdefault("research_goal", research_question)
            payload.setdefault("domain", "general science")
            payload.setdefault("task_type", "literature_survey")
            payload.setdefault("user_focus", [])
            payload.setdefault("source_requirements", [])
            payload.setdefault("information_needs", [])
            payload.setdefault("dynamic_tables", [])
            payload.setdefault("quality_rules", [])
            payload.setdefault("missing_data_policy", "Use null for missing information; do not fabricate values.")
            plan = DynamicExtractionPlan.model_validate(payload)
            if not plan.dynamic_tables:
                raise LLMCallError("Dynamic Schema Planner returned no dynamic tables.")
            return plan
        except Exception as exc:
            if not self.allow_rule_fallback:
                raise
            plan = _fallback_dynamic_extraction_plan(research_question, task_plan=task_plan)
            plan.quality_rules.append(
                f"LLM dynamic schema planning failed; deterministic fallback was used for local testing only: {exc}"
            )
            return plan

    def interpret_sections(
        self,
        research_question: str,
        heading_candidates: list[HeadingCandidate],
    ) -> SectionPlan:
        candidates_payload = [
            {
                "candidate_id": candidate.candidate_id,
                "source_file": candidate.source_file,
                "page": candidate.page,
                "line_index": candidate.line_index,
                "text": candidate.text,
                "before_text": candidate.before_text,
                "after_text": candidate.after_text,
                "font_size": candidate.font_size,
                "is_bold": candidate.is_bold,
                "extraction_method": candidate.extraction_method,
                "score": candidate.score,
            }
            for candidate in heading_candidates[:80]
        ]
        payload = self._generate_json_with_retries(
            "qwen_section_interpreter",
            SECTION_INTERPRETER_SYSTEM,
            SECTION_INTERPRETER_USER.format(
                research_question=research_question,
                candidates_json=json.dumps(candidates_payload, ensure_ascii=False, indent=2),
            ),
        )
        if not isinstance(payload, dict):
            raise LLMCallError("Section Interpreter did not return a JSON object.")
        payload.setdefault("sections", [])
        payload.setdefault("ignored_candidates", [])
        payload.setdefault("warnings", [])
        _repair_section_sources(payload, heading_candidates)
        payload["used_llm"] = True
        return SectionPlan.model_validate(payload)

    def extract_dynamic_from_text_blocks(
        self,
        dynamic_plan: DynamicExtractionPlan,
        text_blocks: list[TextBlock | SectionBlock],
        max_blocks: int | None = None,
        progress_callback=None,
    ) -> list[DynamicRecord]:
        records: list[DynamicRecord] = []
        ranked_blocks = _limit_blocks(_rank_text_blocks(text_blocks), max_blocks)
        for index, block in enumerate(ranked_blocks, start=1):
            if progress_callback:
                progress_callback(index, len(ranked_blocks), block, len(records))
            user_prompt = DYNAMIC_EXTRACTOR_USER.format(
                dynamic_plan_json=dynamic_plan.model_dump_json(),
                source_file=block.source_file,
                source_type=block.source_type.value,
                page=block.page,
                page_json=json.dumps(block.page),
                section_title=json.dumps(getattr(block, "section_title", None), ensure_ascii=False),
                section_type=json.dumps(getattr(block, "section_type", None), ensure_ascii=False),
                page_range=_block_page_range(block),
                content=_trim_content_for_extraction(block.text),
            )
            try:
                payload = self._generate_json_with_retries("qwen_dynamic_record_extractor_pdf", DYNAMIC_EXTRACTOR_SYSTEM, user_prompt)
                records.extend(_dynamic_records_from_payload(payload, dynamic_plan, block.source_file, block.source_type, block.page, block=block))
            except Exception as exc:
                self.extraction_warnings.append(
                    f"Qwen dynamic extraction skipped one block: source_file={block.source_file}, "
                    f"page={block.page}, section={getattr(block, 'section_title', None)}, error={exc}"
                )
                continue
        return records

    def extract_dynamic_from_tables(
        self,
        dynamic_plan: DynamicExtractionPlan,
        tables: list[TableBlock],
    ) -> list[DynamicRecord]:
        records: list[DynamicRecord] = []
        for table in tables:
            content = json.dumps(
                {"columns": table.columns, "rows": table.rows[:80], "table_id": table.table_id},
                ensure_ascii=False,
                indent=2,
            )
            user_prompt = DYNAMIC_EXTRACTOR_USER.format(
                dynamic_plan_json=dynamic_plan.model_dump_json(),
                source_file=table.source_file,
                source_type=table.source_type.value,
                page=None,
                page_json="null",
                section_title="null",
                section_type="null",
                page_range="null",
                content=content,
            )
            try:
                payload = self._generate_json_with_retries("qwen_dynamic_record_extractor_table", DYNAMIC_EXTRACTOR_SYSTEM, user_prompt)
                table_records = _dynamic_records_from_payload(payload, dynamic_plan, table.source_file, table.source_type, None)
                for record in table_records:
                    record.raw.setdefault("table_id", table.table_id)
                records.extend(table_records)
            except Exception as exc:
                self.extraction_warnings.append(
                    f"Qwen dynamic extraction skipped one table: source_file={table.source_file}, "
                    f"table_id={table.table_id}, error={exc}"
                )
                continue
        return records

    def extract_from_text_blocks(self, task_plan: TaskPlan, text_blocks: list[TextBlock | SectionBlock]) -> list[ScientificRecord]:
        records: list[ScientificRecord] = []
        self.extraction_warnings.clear()
        for block in _rank_text_blocks(text_blocks):
            user_prompt = RECORD_EXTRACTOR_USER.format(
                task_plan_json=task_plan.model_dump_json(),
                source_file=block.source_file,
                source_type=block.source_type.value,
                page=block.page,
                page_json=json.dumps(block.page),
                section_title=json.dumps(getattr(block, "section_title", None), ensure_ascii=False),
                section_type=json.dumps(getattr(block, "section_type", None), ensure_ascii=False),
                page_range=_block_page_range(block),
                content=_trim_content_for_extraction(block.text),
            )
            try:
                payload = self._generate_json_with_retries("qwen_record_extractor_pdf", RECORD_EXTRACTOR_SYSTEM, user_prompt)
                records.extend(_records_from_payload(payload, block.source_file, block.source_type, block.page, block=block))
            except Exception as exc:
                warning = (
                    f"Qwen text extraction skipped one block: source_file={block.source_file}, "
                    f"page={block.page}, error={exc}"
                )
                self.extraction_warnings.append(warning)
                if self.allow_rule_fallback:
                    records.extend(extract_records_from_text_blocks([block]))
                continue
        return records

    def extract_from_text_blocks_limited(
        self,
        task_plan: TaskPlan,
        text_blocks: list[TextBlock | SectionBlock],
        max_blocks: int | None = None,
        progress_callback=None,
    ) -> list[ScientificRecord]:
        records: list[ScientificRecord] = []
        self.extraction_warnings.clear()
        ranked_blocks = _limit_blocks(_rank_text_blocks(text_blocks), max_blocks)
        for index, block in enumerate(ranked_blocks, start=1):
            if progress_callback:
                progress_callback(index, len(ranked_blocks), block, len(records))
            user_prompt = RECORD_EXTRACTOR_USER.format(
                task_plan_json=task_plan.model_dump_json(),
                source_file=block.source_file,
                source_type=block.source_type.value,
                page=block.page,
                page_json=json.dumps(block.page),
                section_title=json.dumps(getattr(block, "section_title", None), ensure_ascii=False),
                section_type=json.dumps(getattr(block, "section_type", None), ensure_ascii=False),
                page_range=_block_page_range(block),
                content=_trim_content_for_extraction(block.text),
            )
            try:
                payload = self._generate_json_with_retries("qwen_record_extractor_pdf", RECORD_EXTRACTOR_SYSTEM, user_prompt)
                records.extend(_records_from_payload(payload, block.source_file, block.source_type, block.page, block=block))
            except Exception as exc:
                warning = (
                    f"Qwen text extraction skipped one block: source_file={block.source_file}, "
                    f"page={block.page}, error={exc}"
                )
                self.extraction_warnings.append(warning)
                if self.allow_rule_fallback:
                    records.extend(extract_records_from_text_blocks([block]))
                continue
        return records

    def extract_from_tables(self, task_plan: TaskPlan, tables: list[TableBlock]) -> list[ScientificRecord]:
        records: list[ScientificRecord] = []
        for table in tables:
            preview_rows = table.rows[:80]
            content = json.dumps(
                {"columns": table.columns, "rows": preview_rows, "table_id": table.table_id},
                ensure_ascii=False,
                indent=2,
            )
            user_prompt = RECORD_EXTRACTOR_USER.format(
                task_plan_json=task_plan.model_dump_json(),
                source_file=table.source_file,
                source_type=table.source_type.value,
                page=None,
                page_json="null",
                section_title="null",
                section_type="null",
                page_range="null",
                content=content,
            )
            try:
                payload = self._generate_json_with_retries("qwen_record_extractor_table", RECORD_EXTRACTOR_SYSTEM, user_prompt)
                table_records = _records_from_payload(payload, table.source_file, table.source_type, None)
                for record in table_records:
                    record.raw.setdefault("table_id", table.table_id)
                records.extend(table_records)
            except Exception as exc:
                warning = (
                    f"Qwen table extraction skipped one table: source_file={table.source_file}, "
                    f"table_id={table.table_id}, error={exc}"
                )
                self.extraction_warnings.append(warning)
                if self.allow_rule_fallback:
                    records.extend(extract_records_from_tables([table]))
                continue
        return records

    def validate_records(self, records: list[ScientificRecord]) -> list[QualityIssue]:
        if not records:
            return []
        records_json = json.dumps([r.model_dump(mode="json") for r in records], ensure_ascii=False, indent=2)
        try:
            payload = self._generate_json_with_retries(
                "qwen_quality_validator",
                VALIDATOR_SYSTEM,
                VALIDATOR_USER.format(records_json=records_json),
            )
            return _issues_from_payload(payload)
        except Exception:
            if not self.allow_rule_fallback:
                raise
            return [
                QualityIssue(
                    level="warning",
                    message="LLM quality validation failed; only deterministic quality checks were used.",
                    field=None,
                )
            ]

    def _generate_vision_json_with_retries(
        self,
        node: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
        temperature: float = 0.1,
        attempts: int = 3,
        retry_delay_seconds: float = 2.0,
    ) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self.client.generate_vision_json(
                    node, system_prompt, user_prompt, image_paths, temperature=temperature
                )
            except Exception as exc:
                last_exc = exc
                warning = f"VL call failed: node={node}, attempt={attempt}/{attempts}, error={exc}"
                self.node_warnings.append(warning)
                if attempt < attempts:
                    time.sleep(min(retry_delay_seconds * attempt, 8.0))
        raise LLMCallError(f"VL node failed after {attempts} attempts: node={node}, error={last_exc}") from last_exc

    def classify_chart(self, figure: FigureAsset) -> dict[str, Any]:
        """Triage a rendered figure: does it contain extractable chart data?"""
        if not figure.image_path:
            return {
                "chart_type": "unknown",
                "contains_data": False,
                "reason": "figure has no rendered image",
                "confidence": 0.0,
            }
        payload = self._generate_vision_json_with_retries(
            "qwen_vl_chart_classifier",
            CHART_CLASSIFIER_SYSTEM,
            CHART_CLASSIFIER_USER.format(caption=figure.caption or "(no caption available)"),
            [figure.image_path],
        )
        if not isinstance(payload, dict):
            raise LLMCallError("Chart classifier did not return a JSON object.")
        return {
            "chart_type": str(payload.get("chart_type") or "unknown"),
            "contains_data": bool(payload.get("contains_data")),
            "reason": str(payload.get("reason") or ""),
            "confidence": _clamp_confidence(payload.get("confidence")),
        }

    def extract_chart_data(
        self,
        figure: FigureAsset,
        chart_type: str,
        max_points: int = 40,
    ) -> ChartExtraction:
        """Extract axis/legend/data-point JSON from a chart figure via Qwen-VL."""
        if not figure.image_path:
            raise LLMCallError("Chart extraction requires a rendered figure image.")
        payload = self._generate_vision_json_with_retries(
            "qwen_vl_chart_extractor",
            CHART_EXTRACTOR_SYSTEM,
            CHART_EXTRACTOR_USER.format(
                chart_type=chart_type,
                caption=figure.caption or "(no caption available)",
                max_points=max_points,
            ),
            [figure.image_path],
        )
        if not isinstance(payload, dict):
            raise LLMCallError("Chart extractor did not return a JSON object.")
        payload.setdefault("chart_type", chart_type)
        payload["contains_data"] = True
        payload["figure_id"] = figure.figure_id
        payload["source_file"] = figure.source_file
        payload["page"] = figure.page
        _repair_chart_extraction_payload(payload, max_points=max_points)
        return ChartExtraction.model_validate(payload)


def _clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _repair_chart_extraction_payload(payload: dict[str, Any], max_points: int = 40) -> None:
    """Normalize VL chart payloads before Pydantic validation.

    - Coerce axis ranges to floats, dropping unparseable values.
    - Keep only well-formed [x, y] numeric points, capped per series.
    - Normalize scale values outside the allowed enum to "unknown".
    """
    for axis_key in ("x_axis", "y_axis"):
        axis = payload.get(axis_key)
        if not isinstance(axis, dict):
            payload[axis_key] = {}
            continue
        if axis.get("scale") not in ("linear", "log", "unknown"):
            axis["scale"] = "unknown"
        for range_key in ("range_min", "range_max"):
            axis[range_key] = _coerce_float_or_none(axis.get(range_key))
    series_list = payload.get("series")
    if not isinstance(series_list, list):
        payload["series"] = []
        return
    cleaned_series = []
    for series in series_list:
        if not isinstance(series, dict):
            continue
        points = series.get("points")
        cleaned_points: list[list[float]] = []
        if isinstance(points, list):
            for point in points:
                if not isinstance(point, list | tuple) or len(point) < 2:
                    continue
                x_value = _coerce_float_or_none(point[0])
                y_value = _coerce_float_or_none(point[1])
                if x_value is None or y_value is None:
                    continue
                cleaned_points.append([x_value, y_value])
                if len(cleaned_points) >= max_points:
                    break
        series["points"] = cleaned_points
        cleaned_series.append(series)
    payload["series"] = cleaned_series
    payload["confidence"] = _clamp_confidence(payload.get("confidence"))


def _coerce_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _ensure_default_fields(fields: Any) -> list[str]:
    result = [str(field) for field in fields] if isinstance(fields, list) else []
    for field in DEFAULT_TARGET_FIELDS:
        if field not in result:
            result.append(field)
    return result


def _source_candidate_summaries(source_discovery_plan: SourceDiscoveryPlan, limit: int = 60) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for source in source_discovery_plan.candidate_sources[:limit]:
        metadata = source.metadata or {}
        files = metadata.get("files") if isinstance(metadata.get("files"), list) else []
        file_summaries = []
        for file_item in files[:8]:
            if not isinstance(file_item, dict):
                continue
            file_summaries.append(
                {
                    "name": file_item.get("name") or file_item.get("key"),
                    "type": file_item.get("type"),
                    "size": file_item.get("size") or file_item.get("filesize") or file_item.get("file_size"),
                    "download_url": file_item.get("download_url") or file_item.get("url"),
                }
            )
        summaries.append(
            {
                "source_id": source.source_id,
                "title": source.title,
                "source_type": source.source_type,
                "provider": metadata.get("provider"),
                "url": source.url,
                "query": source.query,
                "description": source.description,
                "reason": source.reason,
                "confidence": source.confidence,
                "published": metadata.get("published") or metadata.get("publication_date") or metadata.get("published_date"),
                "updated": metadata.get("updated"),
                "year": metadata.get("year") or metadata.get("publication_year"),
                "authors": metadata.get("authors") or metadata.get("creators"),
                "venue": metadata.get("venue"),
                "doi": metadata.get("doi") or metadata.get("DOI"),
                "pdf_url": metadata.get("pdf_url"),
                "open_access_url": metadata.get("open_access_url") or metadata.get("oa_url"),
                "abstract": _trim_candidate_text(metadata.get("abstract")),
                "keywords": metadata.get("keywords"),
                "topics": metadata.get("topics"),
                "repository": metadata.get("full_name"),
                "stars": metadata.get("stars"),
                "files": file_summaries,
            }
        )
    return summaries


def _trim_candidate_text(value: Any, limit: int = 900) -> str | None:
    if value in (None, ""):
        return None
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _normalize_source_selection_decisions(payload: dict[str, Any]) -> None:
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        payload["decisions"] = []
        return

    notes = payload.get("notes")
    if not isinstance(notes, list):
        notes = []
        payload["notes"] = notes

    decision_aliases = {
        "deep read": "deep_read",
        "deep-read": "deep_read",
        "full_text": "deep_read",
        "full-text": "deep_read",
        "download_pdf": "deep_read",
        "pdf": "deep_read",
        "metadata": "metadata_only",
        "metadata-only": "metadata_only",
        "record_only": "metadata_only",
        "read metadata": "metadata_only",
        "readme": "read_readme",
        "read_readme": "read_readme",
        "file_manifest": "read_file_manifest",
        "manifest": "read_file_manifest",
        "small_table": "download_small_table",
        "table": "download_small_table",
        "small_supplement": "download_small_supplement",
        "supplement": "download_small_supplement",
        "skip": "reject",
        "irrelevant": "reject",
        "noise": "reject",
        "discard": "reject",
        "ask": "ask_user",
        "ask-user": "ask_user",
    }
    priority_aliases = {
        "important": "high",
        "primary": "high",
        "top": "high",
        "mid": "medium",
        "normal": "medium",
        "minor": "low",
    }
    role_aliases = {
        "paper": "primary_paper",
        "primary": "primary_paper",
        "supporting": "supporting_paper",
        "data": "dataset",
        "repo": "code_repository",
        "code": "code_repository",
        "metadata": "metadata_reference",
        "irrelevant": "noise",
    }
    repaired = 0
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        raw_decision = str(decision.get("decision") or "metadata_only").strip().lower().replace(" ", "_")
        normalized_decision = decision_aliases.get(raw_decision, raw_decision)
        if normalized_decision != decision.get("decision"):
            decision["decision"] = normalized_decision
            repaired += 1

        raw_priority = str(decision.get("priority") or "medium").strip().lower()
        normalized_priority = priority_aliases.get(raw_priority, raw_priority)
        if normalized_priority != decision.get("priority"):
            decision["priority"] = normalized_priority
            repaired += 1

        raw_role = str(decision.get("source_role") or "unknown").strip().lower().replace(" ", "_")
        normalized_role = role_aliases.get(raw_role, raw_role)
        if normalized_role != decision.get("source_role"):
            decision["source_role"] = normalized_role
            repaired += 1

        decision.setdefault("reason", "LLM selected this handling action from candidate metadata.")
        decision.setdefault("matched_requirements", [])
        decision.setdefault("expected_extractable_fields", [])
        decision.setdefault("risk_notes", [])
    if repaired:
        notes.append(f"Normalized {repaired} source selector value(s) from LLM aliases.")


def _fallback_dynamic_extraction_plan(research_question: str, task_plan: TaskPlan | None = None) -> DynamicExtractionPlan:
    domain = task_plan.domain if task_plan else "general science"
    focus = _fallback_focus_from_task_plan(task_plan)
    dynamic_tables = _fallback_dynamic_tables_from_task_plan(task_plan)
    return DynamicExtractionPlan(
        research_goal=research_question,
        domain=domain,
        task_type="literature_survey",
        user_focus=focus,
        source_requirements=task_plan.source_requirements if task_plan and task_plan.source_requirements else ["papers", "tables", "supplementary_materials"],
        information_needs=[
            {"need_name": "paper overview", "reason": "Identify each source and its research goal.", "priority": "high"},
            {"need_name": "method details", "reason": "Capture how the work solves the research problem.", "priority": "high"},
            {"need_name": "experiment results", "reason": "Capture structured quantitative or qualitative findings.", "priority": "high"},
        ],
        dynamic_tables=dynamic_tables,
        quality_rules=[
            "Every important value must be supported by evidence_text.",
            "Use null when information is missing; do not fabricate.",
        ],
    )


def _fallback_arxiv_search_plan(
    research_question: str,
    source_discovery_plan: SourceDiscoveryPlan,
) -> ArxivSearchPlan:
    keywords = []
    for keyword in source_discovery_plan.recommended_keywords:
        cleaned = " ".join(str(keyword).split())
        if cleaned and cleaned.lower() not in {"dataset", "supplementary data", "table", "open database", "papers"}:
            keywords.append(cleaned)
        if len(keywords) >= 4:
            break
    query_text = " ".join(keywords) if keywords else " ".join(research_question.split())
    query = query_text if _looks_like_arxiv_query_text(query_text) else f"all:{query_text}"
    return ArxivSearchPlan(
        research_goal=research_question,
        should_search_arxiv=bool(query_text),
        search_intent="Fallback arXiv query built from LLM/source-discovery keywords for local testing only.",
        queries=[{"query": query, "purpose": "fallback keyword search", "max_results": 20}] if query_text else [],
        selection_criteria=["Use only papers that clearly match the user's research goal."],
        notes=["LLM arXiv search planning failed; deterministic fallback was used for local testing only."],
    )


def _fallback_multi_source_search_plan(
    research_question: str,
    source_discovery_plan: SourceDiscoveryPlan,
) -> MultiSourceSearchPlan:
    keywords = []
    for keyword in source_discovery_plan.recommended_keywords:
        cleaned = " ".join(str(keyword).split())
        if cleaned and cleaned.lower() not in {"dataset", "supplementary data", "table", "open database", "papers"}:
            keywords.append(cleaned)
        if len(keywords) >= 5:
            break
    query_text = " ".join(keywords) if keywords else " ".join(research_question.split())
    requests = []
    if query_text:
        requests = [
            SourceSearchRequest(
                connector_name="arxiv",
                source_type="paper",
                query=query_text if _looks_like_arxiv_query_text(query_text) else f"all:{query_text}",
                purpose="Fallback paper search for local testing only.",
                max_results=20,
            ),
            SourceSearchRequest(
                connector_name="openalex",
                source_type="paper_metadata",
                query=query_text,
                purpose="Fallback paper metadata search for local testing only.",
                max_results=20,
            ),
            SourceSearchRequest(
                connector_name="semantic_scholar",
                source_type="paper_metadata",
                query=query_text,
                purpose="Fallback semantic paper search for local testing only.",
                max_results=20,
            ),
            SourceSearchRequest(
                connector_name="crossref",
                source_type="paper_metadata",
                query=query_text,
                purpose="Fallback DOI metadata search for local testing only.",
                max_results=20,
            ),
            SourceSearchRequest(
                connector_name="zenodo",
                source_type="dataset",
                query=query_text,
                purpose="Fallback open data search for local testing only.",
                max_results=20,
            ),
            SourceSearchRequest(
                connector_name="figshare",
                source_type="dataset",
                query=query_text,
                purpose="Fallback dataset and supplementary material search for local testing only.",
                max_results=20,
            ),
            SourceSearchRequest(
                connector_name="github",
                source_type="repository",
                query=query_text,
                purpose="Fallback repository search for local testing only.",
                max_results=20,
            ),
        ]
    return MultiSourceSearchPlan(
        research_goal=research_question,
        domain=source_discovery_plan.domain,
        should_search=bool(query_text),
        search_requests=requests,
        selection_criteria=["Use only sources that clearly match the user's research goal."],
        notes=["LLM multi-source search planning failed; deterministic fallback was used for local testing only."],
    )


def _looks_like_arxiv_query_text(query: str) -> bool:
    return any(token in query for token in ["all:", "ti:", "abs:", "au:", "cat:", "submittedDate:", " AND ", " OR "])


def _normalize_candidate_source_types(payload: dict[str, Any]) -> None:
    candidates = payload.get("candidate_sources")
    if not isinstance(candidates, list):
        payload["candidate_sources"] = []
        return

    notes = payload.get("notes")
    if not isinstance(notes, list):
        notes = []
        payload["notes"] = notes

    allowed = {
        "paper",
        "paper_search",
        "paper_metadata",
        "open_database",
        "dataset",
        "supplementary_material",
        "table",
        "image",
        "webpage",
        "repository",
        "unknown",
    }
    aliases: dict[str, tuple[str, list[str]]] = {
        "article": ("paper", ["paper", "article"]),
        "academic_paper": ("paper", ["paper", "academic_paper"]),
        "paper_database": ("paper_search", ["paper", "search", "database"]),
        "metadata": ("paper_metadata", ["metadata"]),
        "paper_metadata": ("paper_metadata", ["paper_metadata"]),
        "literature_search": ("paper_search", ["literature", "search"]),
        "database": ("open_database", ["database"]),
        "dataset": ("dataset", ["dataset"]),
        "open_dataset": ("dataset", ["dataset", "open_dataset"]),
        "supplement": ("supplementary_material", ["supplementary_material"]),
        "supplementary": ("supplementary_material", ["supplementary_material"]),
        "supplemental_material": ("supplementary_material", ["supplementary_material"]),
        "supplementary_data": ("supplementary_material", ["supplementary_material", "supplementary_data"]),
        "tables": ("table", ["table"]),
        "figure": ("image", ["figure"]),
        "figures": ("image", ["figure"]),
        "chart": ("image", ["chart"]),
        "charts": ("image", ["chart"]),
        "image_or_chart": ("image", ["image", "chart"]),
        "images_or_charts": ("image", ["image", "chart"]),
        "images": ("image", ["image"]),
        "web": ("webpage", ["webpage"]),
        "website": ("webpage", ["webpage", "website"]),
        "project_page": ("webpage", ["webpage", "project_page"]),
        "official_project_page": ("webpage", ["webpage", "project_page", "official"]),
        "code": ("repository", ["code"]),
        "code_repository": ("repository", ["code", "repository"]),
        "github": ("repository", ["code", "repository", "github"]),
        "repo": ("repository", ["repository"]),
        "leaderboard": ("webpage", ["leaderboard"]),
        "benchmark_leaderboard": ("webpage", ["leaderboard", "benchmark"]),
    }

    normalized_count = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        raw_type = str(candidate.get("source_type") or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
        if raw_type in allowed:
            normalized = raw_type
            subtypes = [raw_type]
        else:
            normalized, subtypes = aliases.get(raw_type, ("unknown", [raw_type]))

        metadata = candidate.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            candidate["metadata"] = metadata

        existing_subtypes = metadata.get("source_subtypes")
        if isinstance(existing_subtypes, list):
            merged_subtypes = [str(subtype) for subtype in existing_subtypes if subtype not in (None, "")]
        elif existing_subtypes:
            merged_subtypes = [str(existing_subtypes)]
        else:
            merged_subtypes = []
        for subtype in subtypes:
            if subtype and subtype not in merged_subtypes:
                merged_subtypes.append(subtype)
        metadata["source_subtypes"] = merged_subtypes

        if normalized != raw_type:
            metadata["raw_source_type"] = raw_type
            candidate["source_type"] = normalized
            normalized_count += 1
        elif normalized not in allowed:
            candidate["source_type"] = "unknown"
            normalized_count += 1
    if normalized_count:
        notes.append(f"Normalized {normalized_count} candidate source_type value(s) from LLM aliases.")


def _normalize_multi_source_search_requests(payload: dict[str, Any]) -> None:
    requests = payload.get("search_requests")
    if not isinstance(requests, list):
        payload["search_requests"] = []
        return
    notes = payload.get("notes")
    if not isinstance(notes, list):
        notes = []
        payload["notes"] = notes
    connector_aliases = {
        "semantic scholar": "semantic_scholar",
        "semanticscholar": "semantic_scholar",
        "semantic-scholar": "semantic_scholar",
        "open alex": "openalex",
        "open-alex": "openalex",
        "cross ref": "crossref",
        "cross-ref": "crossref",
        "git hub": "github",
        "git-hub": "github",
    }
    type_aliases = {
        "papers": "paper",
        "article": "paper",
        "metadata": "paper_metadata",
        "paper_database": "paper_search",
        "literature_search": "paper_search",
        "database": "open_database",
        "open_dataset": "dataset",
        "supplement": "supplementary_material",
        "supplementary": "supplementary_material",
        "supplementary_data": "supplementary_material",
        "tables": "table",
        "figure": "image",
        "figures": "image",
        "chart": "image",
        "charts": "image",
        "images_or_charts": "image",
        "repo": "repository",
        "code": "repository",
        "code_repository": "repository",
    }
    normalized_count = 0
    for request in requests:
        if not isinstance(request, dict):
            continue
        raw_connector = str(request.get("connector_name") or "").strip().lower().replace("_", " ")
        connector = connector_aliases.get(raw_connector, raw_connector.replace(" ", "_").replace("-", "_"))
        if connector != request.get("connector_name"):
            request["connector_name"] = connector
            normalized_count += 1
        raw_type = str(request.get("source_type") or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
        source_type = type_aliases.get(raw_type, raw_type)
        if source_type != request.get("source_type"):
            request["source_type"] = source_type
            normalized_count += 1
    if normalized_count:
        notes.append(f"Normalized {normalized_count} multi-source search request value(s) from LLM aliases.")


def _repair_section_sources(payload: dict[str, Any], heading_candidates: list[HeadingCandidate]) -> None:
    candidates_by_key: dict[tuple[str, int], HeadingCandidate] = {}
    candidates_by_title: dict[str, list[HeadingCandidate]] = {}
    for candidate in heading_candidates:
        normalized_title = _normalize_heading_key(candidate.text)
        candidates_by_key[(normalized_title, candidate.page)] = candidate
        candidates_by_title.setdefault(normalized_title, []).append(candidate)

    repaired = 0
    for section in payload.get("sections", []):
        if not isinstance(section, dict) or section.get("source_file"):
            continue
        title = str(section.get("start_anchor") or section.get("section_title") or "")
        page = section.get("start_page")
        candidate = None
        try:
            candidate = candidates_by_key.get((_normalize_heading_key(title), int(page)))
        except (TypeError, ValueError):
            candidate = None
        if candidate is None:
            matches = candidates_by_title.get(_normalize_heading_key(title), [])
            if len(matches) == 1:
                candidate = matches[0]
        if candidate is not None:
            section["source_file"] = candidate.source_file
            repaired += 1
    if repaired:
        warnings = payload.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
            payload["warnings"] = warnings
        warnings.append(f"Repaired source_file for {repaired} section(s) from heading candidates.")


def _normalize_heading_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _fallback_focus_from_task_plan(task_plan: TaskPlan | None) -> list[str]:
    if not task_plan:
        return ["research question", "methods", "datasets or objects", "results", "evidence"]
    focus = []
    for field in task_plan.target_fields:
        lowered = str(field).lower()
        if any(token in lowered for token in ["architecture", "module", "backbone"]):
            focus.append("method architecture")
        elif any(token in lowered for token in ["dataset", "split", "material"]):
            focus.append("datasets or research objects")
        elif any(token in lowered for token in ["latency", "fps", "size", "throughput", "gpu"]):
            focus.append("deployment efficiency")
        elif any(token in lowered for token in ["metric", "result", "pce", "fid", "ssim", "rmse"]):
            focus.append("experimental results")
        elif "method" in lowered:
            focus.append("methods")
    focus.append("evidence")
    return list(dict.fromkeys(focus)) or ["research question", "methods", "datasets or objects", "results", "evidence"]


def _fallback_dynamic_tables_from_task_plan(task_plan: TaskPlan | None) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = [
            {
                "table_name": "paper_overview",
                "description": "Paper-level metadata, research problem, and contribution.",
                "entity_type": "paper",
                "priority": "high",
                "fields": [
                    {"name": "paper_title", "type": "string", "required": True, "evidence_required": True, "description": "Paper title"},
                    {"name": "research_problem", "type": "string", "required": False, "evidence_required": True, "description": "Problem studied by the paper"},
                    {"name": "main_contribution", "type": "string", "required": False, "evidence_required": True, "description": "Main contribution"},
                ],
            },
            {
                "table_name": "method_details",
                "description": "Method names, components, inputs, outputs, and training strategy.",
                "entity_type": "method",
                "priority": "high",
                "fields": [
                    {"name": "paper_title", "type": "string", "required": False, "evidence_required": True, "description": "Paper title"},
                    {"name": "method_name", "type": "string", "required": True, "evidence_required": True, "description": "Method name"},
                    {"name": "key_modules", "type": "list[string]", "required": False, "evidence_required": True, "description": "Key technical modules"},
                    {"name": "input", "type": "string", "required": False, "evidence_required": True, "description": "Input data or objects"},
                    {"name": "output", "type": "string", "required": False, "evidence_required": True, "description": "Output data or objects"},
                ],
            },
            {
                "table_name": "experiment_results",
                "description": "Datasets, settings, metrics, and values.",
                "entity_type": "experiment",
                "priority": "high",
                "fields": [
                    {"name": "paper_title", "type": "string", "required": False, "evidence_required": True, "description": "Paper title"},
                    {"name": "method_name", "type": "string", "required": False, "evidence_required": True, "description": "Method name"},
                    {"name": "dataset_or_object", "type": "string", "required": False, "evidence_required": True, "description": "Dataset, material, object, or sample"},
                    {"name": "metric_name", "type": "string", "required": False, "evidence_required": True, "description": "Metric name"},
                    {"name": "metric_value", "type": "number|string|null", "required": False, "evidence_required": True, "description": "Metric value"},
                ],
            },
    ]
    target_fields = {str(field).lower() for field in task_plan.target_fields} if task_plan else set()
    if any(any(token in field for token in ["latency", "fps", "throughput", "gpu", "model_size", "size", "deployment"]) for field in target_fields):
        tables.append(
            {
                "table_name": "deployment_efficiency",
                "description": "Deployment targets, runtime, model size, and hardware information.",
                "entity_type": "experiment",
                "priority": "high",
                "fields": [
                    {"name": "paper_title", "type": "string", "required": False, "evidence_required": True, "description": "Paper title"},
                    {"name": "method_name", "type": "string", "required": False, "evidence_required": True, "description": "Method name"},
                    {"name": "target_device", "type": "string", "required": False, "evidence_required": True, "description": "Target deployment device"},
                    {"name": "latency", "type": "number|string|null", "required": False, "evidence_required": True, "description": "Inference latency or runtime"},
                    {"name": "throughput_fps", "type": "number|string|null", "required": False, "evidence_required": True, "description": "Throughput in FPS"},
                    {"name": "model_size", "type": "number|string|null", "required": False, "evidence_required": True, "description": "Model size or parameters"},
                    {"name": "hardware", "type": "string", "required": False, "evidence_required": True, "description": "Hardware or GPU"},
                ],
            }
        )
    if any(any(token in field for token in ["dataset", "split", "material"]) for field in target_fields):
        tables.append(
            {
                "table_name": "dataset_usage",
                "description": "Datasets, splits, objects, or materials used by the study.",
                "entity_type": "dataset",
                "priority": "high",
                "fields": [
                    {"name": "paper_title", "type": "string", "required": False, "evidence_required": True, "description": "Paper title"},
                    {"name": "dataset_or_object", "type": "string", "required": True, "evidence_required": True, "description": "Dataset, material, object, or sample"},
                    {"name": "used_for", "type": "string", "required": False, "evidence_required": True, "description": "Training, validation, testing, benchmark, or analysis usage"},
                    {"name": "split", "type": "string", "required": False, "evidence_required": True, "description": "Dataset split or setting"},
                    {"name": "resolution_or_scale", "type": "string", "required": False, "evidence_required": True, "description": "Resolution, sample count, or scale"},
                ],
            }
        )
    return tables


def _rank_text_blocks(text_blocks: list[TextBlock | SectionBlock]) -> list[TextBlock | SectionBlock]:
    return sorted(text_blocks, key=_text_block_score, reverse=True)


def _limit_blocks(text_blocks: list[TextBlock | SectionBlock], max_blocks: int | None) -> list[TextBlock | SectionBlock]:
    if max_blocks is None or max_blocks <= 0:
        return text_blocks
    return text_blocks[:max_blocks]


def _text_block_score(block: TextBlock | SectionBlock) -> int:
    text = block.text.lower()
    score = 0
    keywords = [
        "table",
        "dataset",
        "benchmark",
        "experiment",
        "evaluation",
        "results",
        "metric",
        "fid",
        "lpips",
        "ssim",
        "kid",
        "fps",
        "user study",
        "ablation",
        "method",
        "architecture",
    ]
    for keyword in keywords:
        if keyword in text:
            score += 10
    score += min(len(re.findall(r"\d+(?:\.\d+)?", text)), 20)
    if block.page is not None and block.page <= 2:
        score += 5
    section_type = str(getattr(block, "section_type", "") or "").lower()
    section_boosts = {
        "abstract": 6,
        "method": 18,
        "implementation": 16,
        "experiments": 18,
        "results": 20,
        "ablation": 16,
        "data": 14,
        "discussion": 6,
    }
    score += section_boosts.get(section_type, 0)
    return score


def _trim_content_for_extraction(text: str, limit: int = 3200) -> str:
    compact = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(compact) <= limit:
        return compact
    head = compact[: limit // 2]
    tail = compact[-limit // 2 :]
    return f"{head}\n\n[...content truncated for extraction...]\n\n{tail}"


def _records_from_payload(
    payload: Any,
    source_file: str,
    source_type: SourceType,
    page: int | None,
    block: TextBlock | SectionBlock | None = None,
) -> list[ScientificRecord]:
    if not isinstance(payload, list):
        raise LLMCallError("Record Extractor did not return a JSON array.")
    records: list[ScientificRecord] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        item.setdefault("source_file", source_file)
        item.setdefault("source_type", source_type.value)
        item.setdefault("page", page)
        item.setdefault("confidence", 0.65)
        _attach_section_raw(item, block)
        if not item.get("metric_name"):
            continue
        try:
            records.append(ScientificRecord.model_validate(item))
        except ValidationError:
            repaired = _repair_record_payload(item, source_file, source_type, page)
            try:
                records.append(ScientificRecord.model_validate(repaired))
            except ValidationError:
                continue
    return records


def _dynamic_records_from_payload(
    payload: Any,
    dynamic_plan: DynamicExtractionPlan,
    source_file: str,
    source_type: SourceType,
    page: int | None,
    block: TextBlock | SectionBlock | None = None,
) -> list[DynamicRecord]:
    if not isinstance(payload, list):
        raise LLMCallError("Dynamic Record Extractor did not return a JSON array.")
    table_specs = {table.table_name: table for table in dynamic_plan.dynamic_tables}
    records: list[DynamicRecord] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        table_name = str(item.get("table_name") or "").strip()
        if table_name not in table_specs:
            continue
        allowed_fields = {field.name for field in table_specs[table_name].fields}
        raw_fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
        fields = {key: value for key, value in raw_fields.items() if key in allowed_fields}
        extra_fields = {key: value for key, value in raw_fields.items() if key not in allowed_fields}
        warnings = list(item.get("warnings") or [])
        if extra_fields:
            warnings.append("unknown dynamic fields moved to raw.extra_fields")
        for field_spec in table_specs[table_name].fields:
            if field_spec.required and fields.get(field_spec.name) in (None, "", []):
                warnings.append(f"required dynamic field missing: {field_spec.name}")
        raw = dict(item.get("raw") or {})
        _attach_section_raw_to_raw(raw, block)
        if extra_fields:
            raw["extra_fields"] = extra_fields
        try:
            records.append(
                DynamicRecord(
                    table_name=table_name,
                    fields=fields,
                    source_file=str(item.get("source_file") or source_file),
                    source_type=str(item.get("source_type") or source_type.value),
                    page=item.get("page", page),
                    evidence_text=item.get("evidence_text"),
                    confidence=item.get("confidence") or 0.65,
                    warnings=warnings,
                    raw=raw,
                )
            )
        except ValidationError:
            continue
    return records


def _repair_record_payload(item: dict[str, Any], source_file: str, source_type: SourceType, page: int | None) -> dict[str, Any]:
    repaired = dict(item)
    repaired["raw"] = dict(repaired.get("raw") or {})
    repaired["warnings"] = list(repaired.get("warnings") or [])
    repaired["source_file"] = str(repaired.get("source_file") or source_file)
    repaired["source_type"] = str(repaired.get("source_type") or source_type.value)
    repaired["page"] = repaired.get("page", page)
    repaired["metric_name"] = str(repaired.get("metric_name") or "unknown_metric")
    try:
        repaired["confidence"] = float(repaired.get("confidence") or 0.5)
    except (TypeError, ValueError):
        repaired["confidence"] = 0.5
        repaired["warnings"].append("confidence repaired from non-numeric value")
    value = repaired.get("metric_value")
    repaired_value, value_warning = _coerce_metric_value(value)
    if value_warning:
        repaired["raw"]["metric_value_raw"] = value
        repaired["warnings"].append(value_warning)
    repaired["metric_value"] = repaired_value
    return repaired


def _attach_section_raw(item: dict[str, Any], block: TextBlock | SectionBlock | None) -> None:
    if block is None:
        return
    raw = dict(item.get("raw") or {})
    _attach_section_raw_to_raw(raw, block)
    item["raw"] = raw


def _attach_section_raw_to_raw(raw: dict[str, Any], block: TextBlock | SectionBlock | None) -> None:
    if block is None:
        return
    section_title = getattr(block, "section_title", None)
    section_type = getattr(block, "section_type", None)
    page_start = getattr(block, "page_start", None)
    page_end = getattr(block, "page_end", None)
    if section_title is not None:
        raw.setdefault("section_title", section_title)
    if section_type is not None:
        raw.setdefault("section_type", section_type)
    if page_start is not None:
        raw.setdefault("page_start", page_start)
    if page_end is not None:
        raw.setdefault("page_end", page_end)


def _block_page_range(block: TextBlock | SectionBlock) -> str:
    page_start = getattr(block, "page_start", None)
    page_end = getattr(block, "page_end", None)
    if page_start is not None or page_end is not None:
        return json.dumps({"page_start": page_start, "page_end": page_end}, ensure_ascii=False)
    return json.dumps({"page_start": block.page, "page_end": block.page}, ensure_ascii=False)


def _coerce_metric_value(value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, "metric_value repaired from boolean value"
    if isinstance(value, int | float):
        return float(value), None
    if not isinstance(value, str):
        return None, "metric_value repaired from unsupported value type"

    text = value.strip()
    if text.lower() in {"", "null", "none", "n/a", "na", "nan", "-", "unknown", "not reported"}:
        return None, "metric_value repaired from empty/null-like string"

    numeric_text = text.replace(",", "")
    if numeric_text.endswith("%"):
        numeric_text = numeric_text[:-1].strip()
    try:
        return float(numeric_text), "metric_value converted from numeric string"
    except ValueError:
        pass

    normalized = text.replace("脳", "x").replace("×", "x").replace("*", "x")
    if re.search(r"\d\s*x\s*\d", normalized, flags=re.IGNORECASE):
        return None, "metric_value repaired from non-scalar dimension string"

    if re.search(r"\d\s*[-~–]\s*\d", normalized):
        return None, "metric_value repaired from range string"

    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", normalized.replace(",", ""))
    if len(numbers) == 1:
        return float(numbers[0]), "metric_value extracted from mixed numeric string"
    if len(numbers) > 1:
        return None, "metric_value repaired from multi-number string"
    return None, "metric_value repaired from non-numeric string"


def _issues_from_payload(payload: Any) -> list[QualityIssue]:
    if not isinstance(payload, list):
        raise LLMCallError("Validator did not return a JSON array.")
    issues: list[QualityIssue] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            issues.append(QualityIssue.model_validate(item))
        except ValidationError:
            issues.append(
                QualityIssue(
                    record_id=item.get("record_id"),
                    level=item.get("level") if item.get("level") in {"info", "warning", "error"} else "warning",
                    field=item.get("field"),
                    message=str(item.get("message") or "LLM returned an incomplete quality issue."),
                )
            )
    return issues
