from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

from scidata_agent.agent.schemas import (
    AgentState,
    ArxivSearchPlan,
    DiscoveredSource,
    DynamicExtractionPlan,
    DynamicRecord,
    MultiSourceSearchPlan,
    ScientificRecord,
    SourceDiscoveryPlan,
    SourceSearchRequest,
    SourceSelectionPlan,
    SourceSelectionDecision,
    SourceType,
    TableBlock,
)
from scidata_agent.agent.scidata_agent import SciDataAgent, _records_for_llm_validation
from scidata_agent.llm.client import QwenBailianClient
from scidata_agent.llm import nodes as llm_nodes_module
from scidata_agent.llm.nodes import (
    QwenAgentNodes,
    _dynamic_records_from_payload,
    _issues_from_payload,
    _records_from_payload,
)
from scidata_agent.tools.quality import build_quality_report
from scidata_agent.tools.connectors.arxiv import (
    download_arxiv_pdfs,
    enrich_with_arxiv_results,
    normalize_arxiv_query,
    select_arxiv_papers,
)
from scidata_agent.tools.connectors.github import github_repo_to_source
from scidata_agent.tools.connectors.openalex import openalex_work_to_source
from scidata_agent.tools.connectors.registry import execute_multi_source_search, merge_sources
from scidata_agent.tools.connectors.zenodo import zenodo_record_to_source
from scidata_agent.tools.source_discovery import fallback_discover_sources
from scidata_agent.tools.source_ingestion import ingest_triaged_sources
from scidata_agent.tools.source_triage import ingestible_arxiv_source_ids, ingestible_pdf_source_ids, triage_sources, triage_sources_from_selection
from scidata_agent.tools.curator import curate_dynamic_records
from scidata_agent.tools.exporter import build_paper_survey_records
from scidata_agent.tools.normalizer import scientific_records_from_dynamic
from scidata_agent.tools.parser import build_section_blocks_from_plan
from tests.create_fixtures import create_csv_fixture, create_pdf_fixture


ROOT = Path(__file__).resolve().parents[2]


def test_source_selection_batches_large_candidate_pool(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        DiscoveredSource(
            title=f"candidate-{index}",
            source_type="paper_metadata",
            url=f"https://example.org/{index}",
            metadata={"provider": "openalex"},
        )
        for index in range(85)
    ]
    state = AgentState(
        research_question="Compare glucose changes before and after meals.",
        files=[],
        output_dir=tmp_path,
        source_discovery_plan=SourceDiscoveryPlan(
            research_goal="Compare glucose changes before and after meals.",
            candidate_sources=candidates,
        ),
    )
    calls: list[int] = []

    class BatchingSelector:
        def select_sources(self, research_question, source_discovery_plan, **kwargs):
            calls.append(len(source_discovery_plan.candidate_sources))
            return SourceSelectionPlan(
                research_goal=research_question,
                decisions=[
                    SourceSelectionDecision(
                        source_id=source.source_id,
                        decision="metadata_only",
                        priority="medium",
                        source_role="supporting_paper",
                        priority_score=0.5,
                        reason="batch smoke test",
                    )
                    for source in source_discovery_plan.candidate_sources
                ],
            )

    agent = SciDataAgent(
        output_dir=tmp_path / "outputs",
        llm_client=MockQwenClient(),
        require_llm=True,
        monitor_console=False,
        monitor_enabled=False,
    )
    agent.llm_nodes = BatchingSelector()
    monkeypatch.setenv("SCIDATA_SOURCE_SELECTION_BATCH_SIZE", "40")

    agent._select_sources(state, max_auto_resources=None)

    assert calls == [40, 40, 5]
    assert len(state.source_selection_plan.decisions) == 85
    assert "selection_batches=3/3" in state.processing_log[-1]


def test_source_selection_hard_caps_large_candidate_pool(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        DiscoveredSource(
            title=f"candidate-{index}",
            source_type="paper_metadata",
            url=f"https://example.org/{index}",
            metadata={"provider": "openalex"},
        )
        for index in range(160)
    ]
    state = AgentState(
        research_question="Bounded source selection test.",
        files=[],
        output_dir=tmp_path,
        source_discovery_plan=SourceDiscoveryPlan(
            research_goal="Bounded source selection test.",
            candidate_sources=candidates,
        ),
    )
    calls: list[int] = []

    class BoundedSelector:
        def select_sources(self, research_question, source_discovery_plan, **kwargs):
            calls.append(len(source_discovery_plan.candidate_sources))
            return SourceSelectionPlan(research_goal=research_question)

    agent = SciDataAgent(
        output_dir=tmp_path / "outputs",
        llm_client=MockQwenClient(),
        require_llm=True,
        monitor_console=False,
        monitor_enabled=False,
    )
    agent.llm_nodes = BoundedSelector()
    monkeypatch.setenv("SCIDATA_SOURCE_SELECTION_BATCH_SIZE", "40")
    monkeypatch.setenv("SCIDATA_SOURCE_SELECTION_CANDIDATE_LIMIT", "999")
    monkeypatch.setenv("SCIDATA_SOURCE_SELECTION_MAX_BATCHES", "999")

    agent._select_sources(state, max_auto_resources=None)

    assert calls == [40, 40, 20]
    assert "candidate_sources=100/100" in state.processing_log[-1]
    assert "selection_batches=3/3" in state.processing_log[-1]


class MockQwenClient(QwenBailianClient):
    def __init__(self):
        super().__init__(api_key="mock-key", model="qwen-mock")

    @property
    def configured(self) -> bool:
        return True

    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        if node == "qwen_artifact_action_planner":
            return {
                "research_goal": "Extract materials science metrics from uploaded sources.",
                "iteration": 0,
                "should_continue": False,
                "stop_reason": "The existing MVP test controls parsing directly.",
                "actions": [
                    {
                        "action_id": "action_stop",
                        "artifact_id": None,
                        "action": "stop",
                        "purpose": "End the bounded artifact-planning test iteration.",
                        "expected_fields": [],
                        "priority": "low",
                        "reason": "The integration test exercises the new node without duplicating the legacy parser.",
                        "parameters": {},
                    }
                ],
                "notes": ["Mock artifact planner response."],
            }
        if node == "qwen_source_discovery":
            return {
                "research_goal": "Extract materials science metrics from uploaded sources.",
                "domain": "materials science",
                "recommended_keywords": ["perovskite", "PCE", "stability", "supplementary data"],
                "target_data_types": ["papers", "tables", "supplementary_materials", "open_databases"],
                "dynamic_schema": {
                    "entity": "string",
                    "entity_type": "material",
                    "metric_name": "string",
                    "metric_value": "number|string|null",
                    "composition": "string|null",
                },
                "candidate_sources": [
                    {
                        "title": "Materials Project",
                        "source_type": "open_database",
                        "url": "https://materialsproject.org",
                        "query": "perovskite PCE stability",
                        "description": "Materials structure and property database.",
                        "reason": "Relevant for materials data discovery.",
                        "confidence": 0.82,
                        "metadata": {},
                    },
                    {
                        "title": "arXiv",
                        "source_type": "paper_search",
                        "url": "https://arxiv.org",
                        "query": "perovskite solar cell PCE stability",
                        "description": "Open preprint search.",
                        "reason": "Finds related papers.",
                        "confidence": 0.75,
                        "metadata": {},
                    },
                ],
                "notes": ["Mock source discovery plan."],
            }
        if node == "qwen_arxiv_search_planner":
            return {
                "research_goal": "Extract materials science metrics from uploaded sources.",
                "should_search_arxiv": True,
                "search_intent": "Find papers that contain perovskite solar cell efficiency and stability metrics.",
                "queries": [
                    {
                        "query": 'all:"perovskite solar cell" AND all:PCE AND all:stability',
                        "purpose": "Find materials papers matching the requested metrics.",
                        "max_results": 3,
                    }
                ],
                "selection_criteria": ["Prefer papers with tables or explicit performance metrics."],
                "notes": ["Mock arXiv search plan."],
            }
        if node == "qwen_multi_source_search_planner":
            return {
                "research_goal": "Extract materials science metrics from uploaded sources.",
                "domain": "materials science",
                "should_search": True,
                "search_requests": [
                    {
                        "connector_name": "arxiv",
                        "source_type": "paper",
                        "query": 'all:"perovskite solar cell" AND all:PCE AND all:stability',
                        "purpose": "Find open preprints with requested performance metrics.",
                        "max_results": 3,
                        "must_have": ["perovskite", "PCE"],
                        "nice_to_have": ["stability", "tables"],
                    },
                    {
                        "connector_name": "openalex",
                        "source_type": "paper_metadata",
                        "query": "perovskite solar cell PCE stability",
                        "purpose": "Find bibliographic metadata and DOI links.",
                        "max_results": 3,
                        "must_have": ["perovskite"],
                        "nice_to_have": ["open access"],
                    },
                    {
                        "connector_name": "zenodo",
                        "source_type": "dataset",
                        "query": "perovskite solar cell stability dataset",
                        "purpose": "Find open datasets or supplementary records.",
                        "max_results": 3,
                        "must_have": ["perovskite"],
                        "nice_to_have": ["csv", "supplementary"],
                    },
                ],
                "selection_criteria": ["Prefer sources with extractable metrics and evidence."],
                "notes": ["Mock multi-source search plan."],
            }
        if node == "qwen_source_selector":
            candidate_json = user_prompt.split("Candidate source summaries:", 1)[1].split("PDF/full-text download budget:", 1)[0]
            candidates = json.loads(candidate_json)
            decisions = []
            for candidate in candidates:
                provider = candidate.get("provider")
                source_type = candidate.get("source_type")
                if provider == "arxiv" and candidate.get("pdf_url"):
                    decision = "deep_read"
                    priority = "high"
                    role = "primary_paper"
                    score = 0.9
                elif provider == "github":
                    decision = "read_readme"
                    priority = "medium"
                    role = "code_repository"
                    score = 0.75
                elif source_type in {"dataset", "supplementary_material", "table"}:
                    decision = "read_file_manifest"
                    priority = "medium"
                    role = "dataset"
                    score = 0.7
                elif provider in {"openalex", "semantic_scholar", "crossref"}:
                    decision = "metadata_only"
                    priority = "low"
                    role = "metadata_reference"
                    score = 0.45
                else:
                    decision = "reject"
                    priority = "low"
                    role = "noise"
                    score = 0.1
                decisions.append(
                    {
                        "source_id": candidate["source_id"],
                        "decision": decision,
                        "priority": priority,
                        "source_role": role,
                        "priority_score": score,
                        "reason": "Mock LLM source selector decision based on candidate metadata.",
                        "matched_requirements": ["mock evidence need"],
                        "expected_extractable_fields": ["method", "metric_name", "metric_value"],
                        "risk_notes": [],
                    }
                )
            return {
                "research_goal": "Extract materials science metrics from uploaded sources.",
                "selection_summary": "Mock LLM selected sources for safe downstream ingestion.",
                "time_range_interpreted": None,
                "decisions": decisions,
                "notes": ["Mock source selection plan."],
            }
        if node == "qwen_task_planner":
            return {
                "domain": "materials science",
                "target_fields": [
                    "paper_title",
                    "material",
                    "method",
                    "metric_name",
                    "metric_value",
                    "unit",
                    "condition",
                    "source_file",
                    "source_type",
                    "page",
                    "evidence_text",
                    "confidence",
                ],
                "output_format": ["csv", "json"],
                "need_provenance": True,
                "assumptions": ["Mock Qwen returns a structured task plan for tests."],
                "schema_notes": ["Keep provenance and evidence for each extracted value."],
                "dynamic_schema": {"entity": "string", "metric_name": "string", "metric_value": "number|null"},
                # The shared mock fixture only guarantees a local paper. Tests
                # that exercise missing table or supplementary evidence declare
                # those requirements explicitly in their own plans.
                "source_requirements": ["papers"],
                "validation_rules": ["metric_value must be supported by evidence_text"],
            }
        if node == "qwen_dynamic_schema_planner":
            return {
                "research_goal": "Extract material science information with dynamic tables.",
                "domain": "materials science",
                "task_type": "data_extraction",
                "user_focus": ["materials", "fabrication methods", "performance metrics"],
                "time_range": None,
                "source_requirements": ["papers"],
                "information_needs": [
                    {
                        "need_name": "device structure and material",
                        "reason": "The user needs material identities and fabrication context.",
                        "priority": "high",
                    }
                ],
                "dynamic_tables": [
                    {
                        "table_name": "material_methods",
                        "description": "Materials and preparation methods.",
                        "entity_type": "method",
                        "priority": "high",
                        "fields": [
                            {
                                "name": "material",
                                "type": "string",
                                "required": True,
                                "evidence_required": True,
                                "description": "Material name",
                                "examples": ["MAPbI3"],
                            },
                            {
                                "name": "method",
                                "type": "string",
                                "required": False,
                                "evidence_required": True,
                                "description": "Preparation method",
                                "examples": ["spin coating"],
                            },
                            {
                                "name": "performance_focus",
                                "type": "string",
                                "required": False,
                                "evidence_required": True,
                                "description": "Performance metric or target",
                                "examples": ["PCE"],
                            },
                        ],
                    },
                    {
                        "table_name": "performance_results",
                        "description": "Structured performance values.",
                        "entity_type": "experiment",
                        "priority": "high",
                        "fields": [
                            {
                                "name": "material",
                                "type": "string",
                                "required": True,
                                "evidence_required": True,
                                "description": "Material name",
                                "examples": ["MAPbI3"],
                            },
                            {
                                "name": "metric_name",
                                "type": "string",
                                "required": True,
                                "evidence_required": True,
                                "description": "Metric name",
                                "examples": ["PCE"],
                            },
                            {
                                "name": "metric_value",
                                "type": "number|string|null",
                                "required": False,
                                "evidence_required": True,
                                "description": "Metric value",
                                "examples": ["21.3"],
                            },
                        ],
                    },
                ],
                "quality_rules": ["Use null when evidence is missing."],
                "missing_data_policy": "Use null for missing information; do not fabricate values.",
            }
        if node == "qwen_section_interpreter":
            candidates = json.loads(user_prompt.split("Heading candidates extracted from PDF text/layout:", 1)[1].split("Return JSON:", 1)[0])
            sections = []
            ignored = []
            for candidate in candidates:
                text = candidate.get("text", "")
                lowered = text.lower()
                if "abstract" == lowered:
                    section_type = "abstract"
                elif "method" in lowered or "approach" in lowered:
                    section_type = "method"
                elif "result" in lowered:
                    section_type = "results"
                else:
                    ignored.append({"text": text, "page": candidate.get("page"), "reason": "not needed in mock"})
                    continue
                sections.append(
                    {
                        "source_file": candidate.get("source_file"),
                        "section_title": text,
                        "section_type": section_type,
                        "start_page": candidate.get("page", 1),
                        "start_anchor": text,
                        "confidence": 0.9,
                        "reason": "Mock section interpreter classified this heading from candidate context.",
                    }
                )
            return {"sections": sections, "ignored_candidates": ignored, "warnings": []}
        if node == "qwen_record_extractor_pdf":
            if "MAPbI3" in user_prompt:
                return [
                    {
                        "paper_title": "Demo Perovskite Solar Cell Study",
                        "material": "MAPbI3",
                        "method": "spin coating",
                        "metric_name": "PCE",
                        "metric_value": 21.3,
                        "unit": "%",
                        "condition": "AM 1.5G illumination",
                        "source_file": "demo_scientific_paper.pdf",
                        "source_type": "pdf_text",
                        "page": 1,
                        "evidence_text": "The MAPbI3 device prepared by spin coating achieved a PCE of 21.3% under AM 1.5G illumination.",
                        "confidence": 0.92,
                    }
                ]
            return []
        if node == "qwen_dynamic_record_extractor_pdf":
            if "MAPbI3" in user_prompt:
                return [
                    {
                        "table_name": "material_methods",
                        "fields": {
                            "material": "MAPbI3",
                            "method": "spin coating",
                            "performance_focus": "PCE",
                        },
                        "source_file": "demo_scientific_paper.pdf",
                        "source_type": "pdf_text",
                        "page": 1,
                        "evidence_text": "The MAPbI3 device prepared by spin coating achieved a PCE of 21.3% under AM 1.5G illumination.",
                        "confidence": 0.9,
                        "warnings": [],
                    },
                    {
                        "table_name": "performance_results",
                        "fields": {
                            "material": "MAPbI3",
                            "metric_name": "PCE",
                            "metric_value": 21.3,
                        },
                        "source_file": "demo_scientific_paper.pdf",
                        "source_type": "pdf_text",
                        "page": 1,
                        "evidence_text": "The MAPbI3 device prepared by spin coating achieved a PCE of 21.3% under AM 1.5G illumination.",
                        "confidence": 0.9,
                        "warnings": [],
                    },
                ]
            return []
        if node == "qwen_dynamic_record_extractor_table":
            return [
                {
                    "table_name": "performance_results",
                    "fields": {
                        "material": "FAPbI3",
                        "metric_name": "PCE",
                        "metric_value": 23.1,
                    },
                    "source_file": "perovskite_metrics.csv",
                    "source_type": "csv",
                    "page": None,
                    "evidence_text": "Material=FAPbI3; Method=annealing; PCE (%)=23.1; Condition=after 1000 h stability test",
                    "confidence": 0.9,
                    "warnings": [],
                }
            ]
        if node == "qwen_record_extractor_table":
            return [
                {
                    "paper_title": "Demo perovskite solar cells",
                    "material": "FAPbI3",
                    "method": "annealing",
                    "metric_name": "PCE",
                    "metric_value": 23.1,
                    "unit": "%",
                    "condition": "after 1000 h stability test",
                    "source_file": "perovskite_metrics.csv",
                    "source_type": "csv",
                    "page": None,
                    "evidence_text": "Material=FAPbI3; Method=annealing; PCE (%)=23.1; Condition=after 1000 h stability test",
                    "confidence": 0.9,
                }
            ]
        if node == "qwen_quality_validator":
            return []
        return []


class TimeoutThenSuccessQwenClient(MockQwenClient):
    def __init__(self):
        super().__init__()
        self.pdf_calls = 0

    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        if node == "qwen_record_extractor_pdf":
            self.pdf_calls += 1
            if self.pdf_calls == 1:
                raise TimeoutError("simulated timeout")
            return [
                {
                    "paper_title": "Demo Perovskite Solar Cell Study",
                    "material": "MAPbI3",
                    "method": "spin coating",
                    "metric_name": "PCE",
                    "metric_value": 21.3,
                    "unit": "%",
                    "condition": "AM 1.5G illumination",
                    "source_file": "demo_scientific_paper.pdf",
                    "source_type": "pdf_text",
                    "page": 1,
                    "evidence_text": "The MAPbI3 device prepared by spin coating achieved a PCE of 21.3% under AM 1.5G illumination.",
                    "confidence": 0.92,
                }
            ]
        return super().generate_json(node, system_prompt, user_prompt, temperature=temperature)


class DynamicPlannerTimeoutThenSuccessQwenClient(MockQwenClient):
    def __init__(self):
        super().__init__()
        self.dynamic_calls = 0

    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        if node == "qwen_dynamic_schema_planner":
            self.dynamic_calls += 1
            if self.dynamic_calls == 1:
                raise TimeoutError("simulated dynamic planner timeout")
        return super().generate_json(node, system_prompt, user_prompt, temperature=temperature)


class DynamicPlannerAlwaysTimeoutQwenClient(MockQwenClient):
    def __init__(self):
        super().__init__()
        self.dynamic_calls = 0

    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        if node == "qwen_dynamic_schema_planner":
            self.dynamic_calls += 1
            raise TimeoutError("simulated dynamic planner timeout")
        return super().generate_json(node, system_prompt, user_prompt, temperature=temperature)


class NestedDynamicSchemaQwenClient(MockQwenClient):
    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        payload = super().generate_json(node, system_prompt, user_prompt, temperature=temperature)
        if node == "qwen_task_planner" and isinstance(payload, dict):
            payload["dynamic_schema"] = {
                "paper": {
                    "title": "string",
                    "authors": "list[string]",
                    "published_year": "number|null",
                },
                "method": {
                    "architecture": "string|null",
                    "key_modules": "list[string]",
                    "deployment_efficiency": {
                        "latency": "number|string|null",
                        "fps": "number|string|null",
                    },
                },
                "provenance": {
                    "source_id": "string",
                    "evidence_text": "string",
                    "evidence_score": "number",
                },
            }
        return payload


class NumericDynamicFieldExamplesQwenClient(MockQwenClient):
    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        payload = super().generate_json(node, system_prompt, user_prompt, temperature=temperature)
        if node == "qwen_dynamic_schema_planner" and isinstance(payload, dict):
            payload["dynamic_tables"][1]["fields"][2]["examples"] = [124.5, 890.2, None]
        return payload


class SourceTypeAliasQwenClient(MockQwenClient):
    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        payload = super().generate_json(node, system_prompt, user_prompt, temperature=temperature)
        if node == "qwen_source_discovery" and isinstance(payload, dict):
            payload["candidate_sources"].append(
                {
                    "title": "Figure and chart data from virtual try-on papers",
                    "source_type": "images_or_charts",
                    "url": "https://example.org/figures",
                    "query": "virtual try-on charts",
                    "description": "Figures and chart-like evidence in papers.",
                    "reason": "The user asked for source evidence and visual data.",
                    "confidence": 0.6,
                    "metadata": {},
                }
            )
        return payload


def _without_retry_sleep(func):
    original_sleep = llm_nodes_module.time.sleep
    llm_nodes_module.time.sleep = lambda seconds: None
    try:
        return func()
    finally:
        llm_nodes_module.time.sleep = original_sleep


def test_qwen_agent_pipeline_with_mock_client() -> None:
    csv_path = create_csv_fixture()
    pdf_path = create_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs"
    agent = SciDataAgent(output_dir=output_dir, llm_client=MockQwenClient(), require_llm=True, monitor_console=False)

    result = agent.run(
        "Extract material, method, PCE, stability, RMSE, absorption wavelength, and source evidence.",
        [csv_path, pdf_path],
        max_pdf_pages=5,
    )

    assert result.status == "completed"
    assert re.fullmatch(r"\d{8}_\d{6}_\d{3}_[0-9a-f]{4}", result.task_id)
    assert result.summary.files_processed == 2
    assert result.source_discovery_plan is not None
    assert result.source_discovery_plan.domain == "materials science"
    assert result.dynamic_extraction_plan is not None
    assert any(table.table_name == "material_methods" for table in result.dynamic_extraction_plan.dynamic_tables)
    assert result.dynamic_records
    assert result.summary.dynamic_records_extracted >= 2
    assert result.summary.dynamic_tables_count >= 2
    assert result.source_discovery_plan.candidate_sources
    assert "entity" in result.task_plan.dynamic_schema
    assert result.summary.records_after_cleaning >= 2
    assert result.sources
    assert result.processing_log
    assert result.export_files.csv and Path(result.export_files.csv).exists()
    assert result.export_files.json_file and Path(result.export_files.json_file).exists()
    assert result.export_files.quality_report and Path(result.export_files.quality_report).exists()
    assert result.export_files.source_discovery_plan and Path(result.export_files.source_discovery_plan).exists()
    assert result.export_files.monitor_log and Path(result.export_files.monitor_log).exists()
    assert result.export_files.paper_survey_csv and Path(result.export_files.paper_survey_csv).exists()
    assert result.export_files.paper_survey_json and Path(result.export_files.paper_survey_json).exists()
    assert result.export_files.dynamic_schema and Path(result.export_files.dynamic_schema).exists()
    assert result.export_files.dynamic_records and Path(result.export_files.dynamic_records).exists()
    assert result.export_files.dynamic_records_csv and Path(result.export_files.dynamic_records_csv).exists()
    assert result.export_files.clean_dynamic_records and Path(result.export_files.clean_dynamic_records).exists()
    assert result.export_files.needs_review and Path(result.export_files.needs_review).exists()
    assert result.export_files.dynamic_tables_dir and Path(result.export_files.dynamic_tables_dir).exists()
    assert result.export_files.final_report and Path(result.export_files.final_report).exists()
    assert result.export_files.summary_json and Path(result.export_files.summary_json).exists()
    assert (Path(result.export_files.dynamic_tables_dir) / "material_methods.csv").exists()
    assert (Path(result.export_files.dynamic_tables_dir) / "performance_results.csv").exists()
    assert Path(result.export_files.csv).parent.name == result.task_id
    exported_json = json.loads(Path(result.export_files.json_file).read_text(encoding="utf-8"))
    dynamic_schema = json.loads(Path(result.export_files.dynamic_schema).read_text(encoding="utf-8"))
    dynamic_records = json.loads(Path(result.export_files.dynamic_records).read_text(encoding="utf-8"))
    clean_dynamic_records = json.loads(Path(result.export_files.clean_dynamic_records).read_text(encoding="utf-8"))
    dynamic_records_csv = Path(result.export_files.dynamic_records_csv).read_text(encoding="utf-8-sig")
    assert exported_json["summary"]["files_processed"] == 2
    assert exported_json["summary"]["records_after_cleaning"] >= 2
    assert exported_json["summary"]["dynamic_records_extracted"] >= 2
    assert dynamic_schema["dynamic_tables"]
    assert dynamic_records
    assert clean_dynamic_records
    assert "record_id" in dynamic_records_csv
    assert exported_json["dynamic_records_raw"]
    assert result.quality_report.record_count == result.summary.records_after_cleaning
    assert result.quality_report.evidence_coverage > 0
    assert result.quality_report.value_evidence_coverage > 0
    assert all(record.source_file for record in result.records)
    assert any(record.evidence_text for record in result.records)
    assert any("Metric extraction reused schema-driven dynamic records" in log for log in result.processing_log)
    monitor_events = [
        json.loads(line)
        for line in Path(result.export_files.monitor_log).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event["step"] == "task_planning" and event["status"] == "completed" for event in monitor_events)
    assert any(event["step"] == "dynamic_schema_planning" and event["status"] == "completed" for event in monitor_events)
    assert any(event["step"] == "dynamic_extraction" and event["status"] == "completed" for event in monitor_events)
    assert any(event["step"] == "record_extraction" and event["status"] == "completed" for event in monitor_events)
    assert any(event["step"] == "export" and event["status"] == "completed" for event in monitor_events)


def test_dynamic_record_curation_prefers_arxiv_metadata_and_flags_review() -> None:
    plan = SourceDiscoveryPlan(
        research_goal="virtual try-on",
        candidate_sources=[
            DiscoveredSource(
                title="Reliable arXiv Title",
                source_type="paper",
                url="http://arxiv.org/abs/1234.56789v1",
                metadata={
                    "provider": "arxiv",
                    "authors": ["Ada Lovelace", "Grace Hopper"],
                    "published": "2026-01-02T00:00:00Z",
                    "updated": "2026-01-03T00:00:00Z",
                    "pdf_url": "https://arxiv.org/pdf/1234.56789v1",
                    "downloaded_path": "downloads/arxiv/arxiv_1234_56789v1_reliable.pdf",
                },
            )
        ],
    )
    raw_records = [
        DynamicRecord(
            table_name="paper_metadata",
            fields={"title": "Guessed Title", "authors": ["unknown"], "publication_date": None, "venue": "arXiv"},
            source_file="arxiv_1234_56789v1_reliable.pdf",
            source_type=SourceType.PDF_TEXT,
            page=5,
            evidence_text="A random middle page.",
            confidence=0.65,
        ),
        DynamicRecord(
            table_name="deployment_efficiency",
            fields={"inference_latency_ms": 0.3, "model_size_mb": 1698.4, "memory_footprint_mb": None, "fps": None},
            source_file="arxiv_1234_56789v1_reliable.pdf",
            source_type=SourceType.PDF_TEXT,
            page=7,
            evidence_text="415M parameters are reported, but no MB model size is stated.",
            confidence=0.8,
            warnings=["model_size_mb should be null because MB is not explicitly given"],
        ),
    ]

    clean, needs_review = curate_dynamic_records(raw_records, plan)

    metadata = [record for record in clean if record.table_name == "paper_metadata"][0]
    assert metadata.fields["title"] == "Reliable arXiv Title"
    assert metadata.fields["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert metadata.fields["publication_date"] == "2026-01-02"
    assert metadata.confidence == 1.0
    assert any(record.table_name == "deployment_efficiency" for record in needs_review)
    deployment = [record for record in clean if record.table_name == "deployment_efficiency"][0]
    assert deployment.fields["model_size_mb"] is None
    assert deployment.raw["repaired_values"]["model_size_mb"] == 1698.4


def test_paper_survey_uses_meaningful_method_summary() -> None:
    from scidata_agent.agent.schemas import AgentState

    state = AgentState(
        research_question="survey tryon papers",
        files=[],
        output_dir=ROOT / "outputs" / "test-runs",
        final_records=[
            ScientificRecord(
                paper_title="Mobile-VTON: High-Fidelity On-Device Virtual Try-On",
                material="VITON-HD",
                method="Ours",
                metric_name="LPIPS",
                metric_value=0.088,
                unit="dimensionless",
                source_file="mobile_vton.pdf",
                source_type=SourceType.PDF_TEXT,
                page=7,
            ),
            ScientificRecord(
                paper_title="Mobile-VTON: High-Fidelity On-Device Virtual Try-On",
                material="VITON-HD",
                method="baseline (no TCG, no LC)",
                metric_name="SSIM",
                metric_value=0.893,
                unit="dimensionless",
                source_file="mobile_vton.pdf",
                source_type=SourceType.PDF_TEXT,
                page=7,
            ),
        ],
        clean_dynamic_records=[
            DynamicRecord(
                table_name="method_architecture",
                fields={
                    "architecture_paradigm": "diffusion",
                    "backbone_networks": ["TryonNet", "GarmentNet", "Light-Adapter"],
                },
                source_file="mobile_vton.pdf",
                source_type=SourceType.PDF_TEXT,
            ),
            DynamicRecord(
                table_name="key_modules",
                fields={"module_name": "TCG"},
                source_file="mobile_vton.pdf",
                source_type=SourceType.PDF_TEXT,
            ),
        ],
    )

    rows = build_paper_survey_records(state)

    assert rows[0]["methods"]
    assert "Mobile-VTON" in rows[0]["methods"]
    assert "diffusion" in rows[0]["methods"]
    assert "TryonNet" in rows[0]["methods"]
    assert "Ours" not in rows[0]["methods"]
    assert "baseline" not in rows[0]["methods"]


def test_paper_survey_separates_baselines_from_proposed_method() -> None:
    from scidata_agent.agent.schemas import AgentState

    state = AgentState(
        research_question="survey tryon papers",
        files=[],
        output_dir=ROOT / "outputs" / "test-runs",
        final_records=[
            ScientificRecord(
                paper_title="FW-VTON: Flattening-and-Warping for Person-to-Person Virtual Try-on",
                material="image",
                method="DCI-VTON",
                metric_name="FID",
                metric_value=7.2,
                unit="dimensionless",
                source_file="fw_vton.pdf",
                source_type=SourceType.PDF_TEXT,
                page=7,
            ),
            ScientificRecord(
                paper_title="FW-VTON: Flattening-and-Warping for Person-to-Person Virtual Try-on",
                material="image",
                method="CatVTON",
                metric_name="KID",
                metric_value=1.1,
                unit="dimensionless",
                source_file="fw_vton.pdf",
                source_type=SourceType.PDF_TEXT,
                page=7,
            ),
        ],
        clean_dynamic_records=[
            DynamicRecord(
                table_name="method_architecture",
                fields={"method_name": "FW-VTON", "backbone_network": "dual U-Net"},
                source_file="fw_vton.pdf",
                source_type=SourceType.PDF_TEXT,
            ),
            DynamicRecord(
                table_name="evaluation_metrics",
                fields={"baseline_comparison": "DCI-VTON, CatVTON, IDM-VTON"},
                source_file="fw_vton.pdf",
                source_type=SourceType.PDF_TEXT,
            ),
        ],
    )

    rows = build_paper_survey_records(state)

    assert rows[0]["methods"] == "FW-VTON"
    assert "DCI-VTON" in rows[0]["baselines"]
    assert "CatVTON" in rows[0]["baselines"]
    assert "DCI-VTON" not in rows[0]["methods"]


def test_section_aware_pipeline_exports_section_plan() -> None:
    pdf_path = create_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs"
    agent = SciDataAgent(
        output_dir=output_dir,
        llm_client=MockQwenClient(),
        require_llm=True,
        monitor_console=False,
    )

    result = agent.run(
        "Extract material, method, PCE, results, and evidence by paper section.",
        [pdf_path],
        max_pdf_pages=2,
        max_dynamic_text_blocks=3,
        max_record_text_blocks=3,
    )

    assert result.status == "completed"
    assert result.summary.heading_candidates_extracted >= 1
    assert result.summary.section_blocks_processed >= 1
    assert result.export_files.section_plan and Path(result.export_files.section_plan).exists()
    section_payload = json.loads(Path(result.export_files.section_plan).read_text(encoding="utf-8"))
    assert section_payload["heading_candidates"]
    assert section_payload["section_plan"]["used_llm"] is True
    assert section_payload["section_blocks"]
    assert any(block["section_type"] in {"abstract", "method", "results"} for block in section_payload["section_blocks"])
    assert any("Section interpretation completed" in log for log in result.processing_log)
    assert any("section-aware blocks" in log for log in result.processing_log)
    assert any(record.raw.get("section_type") for record in result.records)


def test_section_builder_keeps_sections_within_source_file() -> None:
    from scidata_agent.agent.schemas import SectionPlan, TextBlock

    blocks = [
        TextBlock(
            source_file="paper_a.pdf",
            source_path="paper_a.pdf",
            source_type=SourceType.PDF_TEXT,
            page=1,
            text="Abstract\nA abstract.\nMethod\nA method text.",
            chunk_id="a1",
        ),
        TextBlock(
            source_file="paper_b.pdf",
            source_path="paper_b.pdf",
            source_type=SourceType.PDF_TEXT,
            page=1,
            text="Abstract\nB abstract.\nResults\nB result text.",
            chunk_id="b1",
        ),
    ]
    plan = SectionPlan.model_validate(
        {
            "sections": [
                {
                    "source_file": "paper_a.pdf",
                    "section_title": "Method",
                    "section_type": "method",
                    "start_page": 1,
                    "start_anchor": "Method",
                    "confidence": 0.9,
                },
                {
                    "source_file": "paper_b.pdf",
                    "section_title": "Results",
                    "section_type": "results",
                    "start_page": 1,
                    "start_anchor": "Results",
                    "confidence": 0.9,
                },
            ],
            "used_llm": True,
        }
    )

    sections = build_section_blocks_from_plan(blocks, plan)

    assert any(block.source_file == "paper_a.pdf" and block.section_type == "method" for block in sections)
    assert any(block.source_file == "paper_b.pdf" and block.section_type == "results" for block in sections)
    assert not any(block.source_file == "paper_a.pdf" and block.section_type == "results" for block in sections)
    assert not any(block.source_file == "paper_b.pdf" and block.section_type == "method" for block in sections)


def test_uploaded_file_does_not_disable_live_multi_source_search(tmp_path, monkeypatch) -> None:
    pdf_path = create_pdf_fixture()
    agent = SciDataAgent(
        output_dir=tmp_path / "outputs",
        llm_client=MockQwenClient(),
        require_llm=True,
        monitor_console=False,
    )
    called: list[str] = []

    def mark(name: str):
        def callback(state, **kwargs):
            called.append(name)
        return callback

    monkeypatch.setattr(agent, "_plan_multi_source_search", mark("plan"))
    monkeypatch.setattr(agent, "_execute_multi_source_search", mark("search"))
    monkeypatch.setattr(agent, "_select_sources", mark("select"))
    monkeypatch.setattr(agent, "_triage_sources", mark("triage"))
    monkeypatch.setattr(agent, "_ingest_triaged_sources", mark("ingest"))
    monkeypatch.setattr(agent, "_ingest_arxiv_pdfs", mark("download"))

    result = agent.run(
        "Search related sources for the uploaded scientific paper.",
        [pdf_path],
        auto_fetch_arxiv=False,
        enable_live_search=True,
        auto_download_sources=False,
        discovery_only=True,
    )

    assert result.status == "completed"
    assert called == ["plan", "search", "select", "triage"]
    assert any("metadata-only" in item for item in result.processing_log)


def test_section_builder_matches_normalized_anchor_and_preserves_chunk_pages() -> None:
    from scidata_agent.agent.schemas import SectionPlan, TextBlock

    blocks = [
        TextBlock(
            source_file="paper.pdf",
            source_path="paper.pdf",
            source_type=SourceType.PDF_TEXT,
            page=4,
            text="3.2. Unified AutoEncoder Method details on page four.",
            chunk_id="p4",
        ),
        TextBlock(
            source_file="paper.pdf",
            source_path="paper.pdf",
            source_type=SourceType.PDF_TEXT,
            page=5,
            text="Additional method evidence on page five.",
            chunk_id="p5",
        ),
        TextBlock(
            source_file="paper.pdf",
            source_path="paper.pdf",
            source_type=SourceType.PDF_TEXT,
            page=6,
            text="4 Experiments Benchmark details on page six.",
            chunk_id="p6",
        ),
    ]
    plan = SectionPlan.model_validate(
        {
            "sections": [
                {
                    "source_file": "paper.pdf",
                    "section_title": "Unified AutoEncoder",
                    "section_type": "method",
                    "start_page": 4,
                    "start_anchor": "preceding paragraph merged here 3.2 Unified AutoEncoder",
                    "confidence": 0.9,
                },
                {
                    "source_file": "paper.pdf",
                    "section_title": "Experiments",
                    "section_type": "experiments",
                    "start_page": 6,
                    "start_anchor": "4. Experiments",
                    "confidence": 0.9,
                },
            ],
            "used_llm": True,
        }
    )

    sections = build_section_blocks_from_plan(blocks, plan)
    method_blocks = [block for block in sections if block.section_type == "method"]

    assert [block.page for block in method_blocks] == [4, 5]
    assert all(block.page_start == block.page_end == block.page for block in method_blocks)
    assert method_blocks[0].raw["section_page_start"] == 4
    assert method_blocks[0].raw["section_page_end"] == 5


def test_record_page_is_corrected_to_source_block_page() -> None:
    from scidata_agent.agent.schemas import TextBlock

    block = TextBlock(
        source_file="paper.pdf",
        source_path="paper.pdf",
        source_type=SourceType.PDF_TEXT,
        page=7,
        text="UAE reports FID 1.23.",
        chunk_id="p7",
    )
    records = _records_from_payload(
        [
            {
                "metric_name": "FID",
                "metric_value": 1.23,
                "page": 1,
                "evidence_text": "UAE reports FID 1.23.",
            }
        ],
        "paper.pdf",
        SourceType.PDF_TEXT,
        7,
        block=block,
    )

    assert records[0].page == 7
    assert records[0].raw["llm_reported_page"] == 1
    assert any("page corrected" in warning for warning in records[0].warnings)


def test_task_planner_accepts_nested_dynamic_schema() -> None:
    pdf_path = create_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs"
    agent = SciDataAgent(
        output_dir=output_dir,
        llm_client=NestedDynamicSchemaQwenClient(),
        require_llm=True,
        monitor_console=False,
    )

    result = agent.run("Survey recent try-on papers and extract rich method details.", [pdf_path], max_pdf_pages=2)

    assert result.status == "completed"
    assert isinstance(result.task_plan.dynamic_schema["provenance"], dict)
    assert result.task_plan.dynamic_schema["provenance"]["source_id"] == "string"
    assert result.export_files.dynamic_schema and Path(result.export_files.dynamic_schema).exists()


def test_dynamic_schema_planner_accepts_numeric_field_examples() -> None:
    pdf_path = create_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs"
    agent = SciDataAgent(
        output_dir=output_dir,
        llm_client=NumericDynamicFieldExamplesQwenClient(),
        require_llm=True,
        monitor_console=False,
    )

    result = agent.run("Survey recent try-on papers and extract numeric metrics.", [pdf_path], max_pdf_pages=2)

    assert result.status == "completed"
    assert result.dynamic_extraction_plan is not None
    numeric_examples = result.dynamic_extraction_plan.dynamic_tables[1].fields[2].examples
    assert numeric_examples == [124.5, 890.2, None]
    assert result.export_files.dynamic_schema and Path(result.export_files.dynamic_schema).exists()


def test_dynamic_schema_plan_enforces_provenance_and_metric_context_contract() -> None:
    class ContractViolatingClient(MockQwenClient):
        def generate_json(
            self,
            node: str,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.1,
        ) -> Any:
            payload = super().generate_json(node, system_prompt, user_prompt, temperature=temperature)
            if node == "qwen_dynamic_schema_planner":
                metric_table = payload["dynamic_tables"][1]
                value_field = next(field for field in metric_table["fields"] if field["name"] == "metric_value")
                value_field["name"] = "value"
                value_field["examples"] = "21.3"
                metric_table["fields"].extend(
                    [
                        {"name": "evidence_text", "type": "string", "required": True},
                        {"name": "paper", "type": "url", "required": True},
                    ]
                )
            return payload

    plan = QwenAgentNodes(ContractViolatingClient()).plan_dynamic_extraction("Extract metrics")
    metric_table = next(table for table in plan.dynamic_tables if table.table_name == "performance_results")
    fields = {field.name: field for field in metric_table.fields}

    assert "evidence_text" not in fields
    assert fields["paper"].required is False
    assert "condition" in fields
    assert fields["value"].examples == ["21.3"]
    assert any("provenance fields" in rule for rule in plan.quality_rules)


def test_dynamic_warning_normalization_keeps_one_required_field_issue() -> None:
    plan = DynamicExtractionPlan.model_validate(
        {
            "research_goal": "Extract scores",
            "dynamic_tables": [
                {
                    "table_name": "scores",
                    "fields": [
                        {"name": "score", "type": "number", "required": True},
                        {"name": "paper", "type": "url", "required": False},
                    ],
                }
            ],
        }
    )
    records = _dynamic_records_from_payload(
        [
            {
                "table_name": "scores",
                "fields": {"score": None, "paper": None},
                "evidence_text": "The score was not reported.",
                "confidence": 0.9,
                "warnings": [
                    "Missing paper URL",
                    "required dynamic field missing: score",
                    "record contains extraction warnings that require review",
                ],
            }
        ],
        plan,
        "scores.csv",
        SourceType.CSV,
        None,
    )

    assert records[0].warnings == [
        "required dynamic field missing: score",
        "record contains extraction warnings that require review",
    ]
    report = build_quality_report([], dynamic_records=records, dynamic_plan=plan)
    assert report.warning_count == 1
    assert report.review_count == 1
    assert report.issues[0].field == "score"


def test_dynamic_context_alias_is_aligned_without_false_unknown_warning() -> None:
    plan = DynamicExtractionPlan.model_validate(
        {
            "research_goal": "Extract metrics",
            "dynamic_tables": [
                {
                    "table_name": "performance_metrics",
                    "fields": [
                        {"name": "metric_name", "required": True},
                        {"name": "value", "type": "number|string|null"},
                        {"name": "condition", "type": "string|null"},
                    ],
                }
            ],
        }
    )

    records = _dynamic_records_from_payload(
        [
            {
                "table_name": "performance_metrics",
                "fields": {"metric_name": "PSNR", "value": "31.00", "dataset_name": "ImageNet-1K"},
                "evidence_text": "On ImageNet-1K, PSNR reaches 31.00.",
                "confidence": 0.9,
                "warnings": [],
            }
        ],
        plan,
        "paper.pdf",
        SourceType.PDF_TEXT,
        7,
    )

    assert records[0].fields["condition"] == "ImageNet-1K"
    assert records[0].raw["field_aliases"] == {"dataset_name": "condition"}
    assert "extra_fields" not in records[0].raw
    assert not any("unknown dynamic fields" in warning for warning in records[0].warnings)


def test_source_discovery_normalizes_llm_source_type_aliases() -> None:
    output_dir = ROOT / "outputs" / "test-runs"
    agent = SciDataAgent(
        output_dir=output_dir,
        llm_client=SourceTypeAliasQwenClient(),
        require_llm=True,
        monitor_console=False,
    )

    result = agent.run(
        "Survey recent try-on papers and collect figure/chart evidence.",
        auto_fetch_arxiv=False,
        discovery_only=True,
    )

    assert result.status == "completed"
    assert result.source_discovery_plan is not None
    image_chart_source = next(
        source
        for source in result.source_discovery_plan.candidate_sources
        if source.metadata.get("raw_source_type") == "images_or_charts"
    )
    assert image_chart_source.source_type == "image"
    assert image_chart_source.metadata["source_subtypes"] == ["image", "chart"]
    assert any(
        source.source_type == "paper_search" and source.metadata.get("source_subtypes") == ["paper_search"]
        for source in result.source_discovery_plan.candidate_sources
    )
    assert any("Normalized" in note for note in result.source_discovery_plan.notes)


def test_dynamic_schema_planner_retries_timeout_and_recovers() -> None:
    pdf_path = create_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs"
    client = DynamicPlannerTimeoutThenSuccessQwenClient()
    agent = SciDataAgent(
        output_dir=output_dir,
        llm_client=client,
        require_llm=True,
        allow_rule_fallback=False,
        monitor_console=False,
    )

    result = _without_retry_sleep(
        lambda: agent.run("Extract material, method, PCE, and evidence.", [pdf_path], max_pdf_pages=2)
    )

    assert result.status == "completed"
    assert client.dynamic_calls == 2
    assert result.dynamic_extraction_plan is not None
    assert result.export_files.dynamic_schema and Path(result.export_files.dynamic_schema).exists()
    assert any(
        "node=qwen_dynamic_schema_planner" in log and "attempt=1/2" in log
        for log in result.processing_log
    )


def test_dynamic_schema_planner_fails_explicitly_after_retries_in_official_mode() -> None:
    pdf_path = create_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs"
    client = DynamicPlannerAlwaysTimeoutQwenClient()
    agent = SciDataAgent(
        output_dir=output_dir,
        llm_client=client,
        require_llm=True,
        allow_rule_fallback=False,
        monitor_console=False,
    )

    result = _without_retry_sleep(
        lambda: agent.run("Extract material, method, PCE, and evidence.", [pdf_path], max_pdf_pages=2)
    )

    assert result.status == "failed"
    assert client.dynamic_calls == 2
    assert result.dynamic_extraction_plan is None
    assert result.export_files.dynamic_schema is None
    assert any(
        "node=qwen_dynamic_schema_planner" in log and "attempt=2/2" in log
        for log in result.processing_log
    )
    assert any("Task failed" in log and "qwen_dynamic_schema_planner" in log for log in result.processing_log)


def test_dynamic_schema_planner_rule_fallback_only_when_explicitly_allowed() -> None:
    pdf_path = create_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs"
    client = DynamicPlannerAlwaysTimeoutQwenClient()
    agent = SciDataAgent(
        output_dir=output_dir,
        llm_client=client,
        require_llm=True,
        allow_rule_fallback=True,
        monitor_console=False,
    )

    result = _without_retry_sleep(
        lambda: agent.run("Extract material, method, PCE, and evidence.", [pdf_path], max_pdf_pages=2)
    )

    assert result.status == "partial"
    assert result.coverage_report.decision == "continue"
    assert result.coverage_report.missing_requirements
    assert client.dynamic_calls == 2
    assert result.dynamic_extraction_plan is not None
    assert result.dynamic_extraction_plan.dynamic_tables
    assert any(
        "local testing only" in rule.lower()
        for rule in result.dynamic_extraction_plan.quality_rules
    )


def test_qwen_extraction_timeout_retries_and_exports() -> None:
    pdf_path = create_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs"
    agent = SciDataAgent(
        output_dir=output_dir,
        llm_client=TimeoutThenSuccessQwenClient(),
        require_llm=True,
        monitor_console=False,
    )

    result = agent.run(
        "Extract material, method, PCE, and evidence.",
        [pdf_path],
        max_pdf_pages=2,
        reuse_dynamic_records_for_metrics=False,
    )

    assert result.status == "completed"
    assert result.summary.records_after_cleaning >= 1
    assert result.export_files.csv and Path(result.export_files.csv).exists()
    assert any(
        "node=qwen_record_extractor_pdf" in log and "attempt=1/2" in log
        for log in result.processing_log
    )
    assert any("skipped_blocks=0" in log for log in result.processing_log)


def test_extraction_limits_text_blocks_and_logs_progress() -> None:
    pdf_path = create_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs"
    agent = SciDataAgent(
        output_dir=output_dir,
        llm_client=MockQwenClient(),
        require_llm=True,
        monitor_console=False,
    )

    result = agent.run(
        "Extract material, method, PCE, and evidence.",
        [pdf_path],
        max_pdf_pages=2,
        max_dynamic_text_blocks=1,
        max_record_text_blocks=1,
    )

    assert result.status == "completed"
    assert any("Dynamic extraction limited to top-ranked" in log for log in result.processing_log)
    assert any("duplicate PDF/table LLM pass skipped" in log for log in result.processing_log)
    assert result.export_files.monitor_log
    monitor_events = [
        json.loads(line)
        for line in Path(result.export_files.monitor_log).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        event["event_type"] == "progress"
        and event["step"] == "dynamic_extraction"
        and event["data"]["progress_index"] == 1
        and event["data"]["progress_total"] == 1
        for event in monitor_events
    )
    assert any(
        event["event_type"] == "step"
        and event["step"] == "record_extraction"
        and event["status"] == "completed"
        for event in monitor_events
    )


def test_question_only_source_discovery_mode_with_mock_client() -> None:
    output_dir = ROOT / "outputs" / "test-runs"
    agent = SciDataAgent(output_dir=output_dir, llm_client=MockQwenClient(), require_llm=True, monitor_console=False)

    result = agent.run(
        "我希望研究 Ia 型超新星光变曲线",
        auto_fetch_arxiv=False,
        discovery_only=True,
    )

    assert result.status == "completed"
    assert result.summary.files_processed == 0
    assert result.summary.records_after_cleaning == 0
    assert result.source_discovery_plan is not None
    assert result.source_discovery_plan.candidate_sources
    assert result.export_files.source_discovery_plan
    assert Path(result.export_files.source_discovery_plan).exists()
    assert any("source discovery only" in log.lower() for log in result.processing_log)


def test_missing_qwen_key_fails_in_official_mode() -> None:
    pdf_path = create_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs"
    client = QwenBailianClient(api_key="")
    agent = SciDataAgent(output_dir=output_dir, llm_client=client, require_llm=True, monitor_console=False)

    result = agent.run("Please extract scientific data from the paper.", [pdf_path], max_pdf_pages=1)

    assert result.status == "failed"
    assert any("Task failed" in log or "Qwen" in log or "api key" in log.lower() for log in result.processing_log)


def test_rule_fallback_is_explicitly_marked() -> None:
    csv_path = create_csv_fixture()
    pdf_path = create_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs"
    agent = SciDataAgent(
        output_dir=output_dir,
        llm_client=QwenBailianClient(api_key=""),
        require_llm=False,
        allow_rule_fallback=True,
        monitor_console=False,
    )

    result = agent.run("Local tool-chain test: extract metrics from PDF and CSV.", [csv_path, pdf_path], max_pdf_pages=5)

    assert result.status == "partial"
    assert result.coverage_report.decision == "continue"
    assert result.coverage_report.missing_requirements
    assert result.summary.records_after_cleaning >= 1
    assert any("fallback" in log.lower() for log in result.processing_log)


def test_quality_report_flags_weak_evidence_and_dimensionless_units() -> None:
    records = [
        ScientificRecord(
            material="Model A",
            method="baseline",
            metric_name="FID",
            metric_value=11.2,
            unit=None,
            source_file="paper.pdf",
            source_type=SourceType.PDF_TEXT,
            page=3,
            evidence_text="The lower FID score indicates better generation quality.",
            confidence=0.9,
        )
    ]

    report = build_quality_report(records, target_fields=["metric_name", "metric_value", "unit", "evidence_text"])

    assert records[0].unit is None
    assert report.record_count == 1
    assert report.warning_count >= 1
    assert report.value_evidence_coverage == 0
    assert any(issue.field == "evidence_text" for issue in report.issues)


def test_quality_report_links_dynamic_warning_and_validates_pdf_page() -> None:
    from scidata_agent.agent.schemas import DynamicExtractionPlan, TextBlock

    dynamic_plan = DynamicExtractionPlan.model_validate(
        {
            "research_goal": "Extract model results",
            "dynamic_tables": [
                {
                    "table_name": "metric_result",
                    "entity_type": "metric",
                    "fields": [
                        {"name": "model", "required": True},
                        {"name": "score", "required": True},
                    ],
                }
            ],
        }
    )
    dynamic_record = DynamicRecord(
        table_name="metric_result",
        fields={"model": "UAE", "score": None},
        source_file="paper.pdf",
        source_type=SourceType.PDF_TEXT,
        page=7,
        evidence_text="UAE reports an FID score of 1.23.",
        confidence=0.8,
        warnings=["required dynamic field missing: score"],
    )
    block = TextBlock(
        source_file="paper.pdf",
        source_path="paper.pdf",
        source_type=SourceType.PDF_TEXT,
        page=7,
        text="Table 1. UAE reports an FID score of 1.23.",
        chunk_id="p7",
    )

    report = build_quality_report(
        [],
        dynamic_records=[dynamic_record],
        dynamic_plan=dynamic_plan,
        text_blocks=[block],
    )

    assert report.dynamic_record_count == 1
    assert report.total_record_count == 1
    assert report.evidence_text_coverage == 1
    assert report.provenance_page_coverage == 1
    assert report.review_count == 1
    assert any(issue.record_id == dynamic_record.record_id for issue in report.issues)


def test_pdf_page_validation_accepts_ordered_table_header_and_row_only() -> None:
    from scidata_agent.agent.schemas import TextBlock

    record = ScientificRecord(
        paper_title="UAE",
        method="SiT",
        metric_name="gFID",
        metric_value=8.61,
        source_file="paper.pdf",
        source_type=SourceType.PDF_TEXT,
        page=8,
        evidence_text="Methods gFID IS Prec Rec SiT 8.61 131.7 0.68 0.67",
        confidence=0.9,
    )
    block = TextBlock(
        source_file="paper.pdf",
        source_path="paper.pdf",
        source_type=SourceType.PDF_TEXT,
        page=8,
        text=(
            "Methods gFID IS Prec Rec\n"
            "DiT 9.62 121.5 0.67 0.67\n"
            "SiT 8.61 131.7 0.68 0.67"
        ),
        chunk_id="p8",
    )

    report = build_quality_report([record], text_blocks=[block])
    assert report.provenance_page_coverage == 1
    assert not any(issue.field == "page" for issue in report.issues)

    unsupported = record.model_copy(
        update={"record_id": "rec_unsupported", "evidence_text": "Methods gFID SiT 999.0"},
        deep=True,
    )
    unsupported_report = build_quality_report([unsupported], text_blocks=[block])
    assert unsupported_report.provenance_page_coverage == 0
    assert any(issue.field == "page" for issue in unsupported_report.issues)


def test_pdf_table_page_validation_uses_structured_table_rows() -> None:
    record = ScientificRecord(
        paper_title="UAE",
        method="SiT",
        metric_name="gFID",
        metric_value=8.61,
        source_file="paper.pdf",
        source_type=SourceType.PDF_TABLE,
        page=8,
        evidence_text='column_1: "SiT", column_2: "8.61", column_3: "131.7"',
        confidence=0.9,
    )
    table = TableBlock(
        source_file="paper.pdf",
        source_path="paper.pdf",
        source_type=SourceType.PDF_TABLE,
        columns=["column_1", "column_2", "column_3"],
        rows=[{"column_1": "SiT", "column_2": "8.61", "column_3": "131.7"}],
        table_id="table_p8_1",
        page=8,
    )

    report = build_quality_report([record], text_blocks=[], table_blocks=[table])

    assert report.provenance_page_coverage == 1
    assert not any(issue.field == "page" for issue in report.issues)


def test_llm_quality_missing_unit_is_warning_not_error() -> None:
    issues = _issues_from_payload(
        [
            {
                "record_id": "rec_latency",
                "level": "error",
                "field": "unit",
                "message": "指标 Latency 缺少单位，应明确时间单位。",
            },
            {
                "record_id": "rec_bad_value",
                "level": "error",
                "field": "metric_value",
                "message": "数值与原文矛盾。",
            },
        ]
    )

    assert issues[0].level == "warning"
    assert issues[1].level == "error"


def test_curator_routes_any_warned_dynamic_record_to_review() -> None:
    record = DynamicRecord(
        table_name="metric_result",
        fields={"model": "UAE"},
        source_file="paper.pdf",
        source_type=SourceType.PDF_TEXT,
        page=7,
        evidence_text="UAE result.",
        confidence=0.8,
        warnings=["required dynamic field missing: score"],
    )

    clean, needs_review = curate_dynamic_records([record])

    assert len(clean) == 1
    assert len(needs_review) == 1
    assert needs_review[0].record_id == record.record_id


def test_dynamic_metric_records_are_reused_without_losing_provenance() -> None:
    dynamic = DynamicRecord(
        table_name="experiment_results",
        fields={"model_name": "UAE", "metric_name": "FID", "metric_value": "4.20"},
        paper_title="Unified Auto-Encoding",
        source_file="uae.pdf",
        source_type=SourceType.PDF_TEXT,
        page=7,
        evidence_text="UAE reaches an FID of 4.20 on the benchmark.",
        confidence=0.91,
    )

    records = scientific_records_from_dynamic([dynamic])

    assert len(records) == 1
    assert records[0].metric_name == "FID"
    assert records[0].metric_value == 4.2
    assert records[0].method == "UAE"
    assert records[0].page == 7
    assert records[0].raw["derived_from_dynamic_record_id"] == dynamic.record_id


def test_dynamic_metric_value_alias_is_reused_with_condition() -> None:
    dynamic = DynamicRecord(
        table_name="performance_metrics",
        fields={
            "model_name": "UAE",
            "metric_name": "FID",
            "value": "1.52",
            "condition": "ImageNet-1K",
        },
        paper_title="Unified Auto-Encoding",
        source_file="uae.pdf",
        source_type=SourceType.PDF_TABLE,
        page=7,
        evidence_text='metric_name: "FID", value: "1.52", condition: "ImageNet-1K"',
        confidence=0.93,
    )

    records = scientific_records_from_dynamic([dynamic])

    assert len(records) == 1
    assert records[0].metric_name == "FID"
    assert records[0].metric_value == 1.52
    assert records[0].condition == "ImageNet-1K"
    assert records[0].source_type == SourceType.PDF_TABLE


def test_llm_quality_sample_prioritizes_risky_records(monkeypatch) -> None:
    monkeypatch.setenv("SCIDATA_LLM_VALIDATE_MAX_RECORDS", "1")
    safe = ScientificRecord(
        metric_name="FID",
        metric_value=4.2,
        source_file="safe.pdf",
        source_type=SourceType.PDF_TEXT,
        page=2,
        evidence_text="FID was 4.2.",
        confidence=0.95,
    )
    risky = ScientificRecord(
        metric_name="FID",
        metric_value=4.5,
        source_file="risky.pdf",
        source_type=SourceType.PDF_TEXT,
        confidence=0.5,
        warnings=["page missing"],
    )

    assert _records_for_llm_validation([safe, risky]) == [risky]


def test_quality_report_detects_conflicts() -> None:
    records = [
        ScientificRecord(
            material="MAPbI3",
            method="spin coating",
            metric_name="PCE",
            metric_value=21.3,
            unit="%",
            source_file="paper_a.pdf",
            source_type=SourceType.PDF_TEXT,
            page=1,
            evidence_text="MAPbI3 achieved a PCE of 21.3%.",
            confidence=0.9,
        ),
        ScientificRecord(
            material="MAPbI3",
            method="spin coating",
            metric_name="PCE",
            metric_value=19.8,
            unit="%",
            source_file="paper_b.pdf",
            source_type=SourceType.PDF_TEXT,
            page=2,
            evidence_text="MAPbI3 achieved a PCE of 19.8%.",
            confidence=0.9,
        ),
    ]

    report = build_quality_report(records, target_fields=["metric_name", "metric_value", "evidence_text"])

    assert report.conflict_count == 1
    assert report.conflicts[0].metric_name == "pce"


def test_quality_report_does_not_flag_different_experimental_contexts_as_conflicts() -> None:
    records = [
        ScientificRecord(
            material="VITON-HD",
            method="Mobile-VTON",
            metric_name="FID",
            metric_value=10.211,
            unit="dimensionless",
            condition="paired setting",
            source_file="paper.pdf",
            source_type=SourceType.PDF_TEXT,
            page=7,
            evidence_text="Mobile-VTON obtains FID 10.211 in the paired setting.",
            confidence=0.9,
        ),
        ScientificRecord(
            material="VITON-HD",
            method="Mobile-VTON",
            metric_name="FID",
            metric_value=12.095,
            unit="dimensionless",
            condition="unpaired setting",
            source_file="paper.pdf",
            source_type=SourceType.PDF_TEXT,
            page=8,
            evidence_text="Mobile-VTON obtains FID 12.095 in the unpaired setting.",
            confidence=0.9,
        ),
    ]

    report = build_quality_report(records, target_fields=["metric_name", "metric_value", "condition"])

    assert report.conflict_count == 0


def test_quality_report_does_not_claim_conflict_when_context_is_missing() -> None:
    records = [
        ScientificRecord(
            paper_title="UAE",
            metric_name="PSNR",
            metric_value=value,
            unit="dB",
            source_file="paper.pdf",
            source_type=SourceType.PDF_TEXT,
            page=7,
            evidence_text=f"The reported PSNR is {value} dB.",
            confidence=0.9,
        )
        for value in (31.0, 31.19)
    ]

    report = build_quality_report(records, target_fields=["metric_name", "metric_value"])

    assert report.conflict_count == 0


def test_record_payload_repair_handles_non_scalar_metric_value() -> None:
    records = _records_from_payload(
        [
            {
                "paper_title": "Mobile-VTON",
                "method": "on-device virtual try-on",
                "metric_name": "input resolution",
                "metric_value": "1024脳768",
                "unit": "pixels",
                "evidence_text": "The model processes images at 1024x768 resolution.",
                "confidence": "0.83",
            }
        ],
        source_file="mobile_vton.pdf",
        source_type=SourceType.PDF_TEXT,
        page=2,
    )

    assert len(records) == 1
    assert records[0].metric_value is None
    assert records[0].raw["metric_value_raw"] == "1024脳768"
    assert any("non-scalar dimension" in warning for warning in records[0].warnings)


def test_fallback_source_discovery_is_general_across_domains() -> None:
    astronomy = fallback_discover_sources("我希望研究 Ia 型超新星光变曲线")
    materials = fallback_discover_sources("我希望研究钙钛矿太阳能电池的 PCE 和稳定性")
    ml = fallback_discover_sources("我希望比较虚拟试衣模型在 VITON-HD 数据集上的 FID 和 LPIPS")

    assert astronomy.domain == "astronomy"
    assert any("VizieR" in source.title for source in astronomy.candidate_sources)
    assert "mjd" in astronomy.dynamic_schema

    assert materials.domain == "materials science"
    assert any("Materials Project" in source.title for source in materials.candidate_sources)
    assert "composition" in materials.dynamic_schema

    assert ml.domain == "machine learning"
    assert any("Papers with Code" in source.title for source in ml.candidate_sources)
    assert "dataset" in ml.dynamic_schema


def test_arxiv_connector_enriches_source_discovery_plan_without_network() -> None:
    plan = fallback_discover_sources("Type Ia supernova light curve")
    arxiv_plan = ArxivSearchPlan(
        research_goal=plan.research_goal,
        search_intent="Find Type Ia supernova light-curve papers.",
        queries=[
            {
                "query": 'all:"Type Ia supernova" AND all:"light curve"',
                "purpose": "LLM-planned astronomy search.",
                "max_results": 3,
            }
        ],
        selection_criteria=["Prefer papers with light-curve data or tables."],
    )

    # Keep the test fully offline by injecting a deterministic arXiv result.
    def offline_searcher(query: str, max_results: int):
        from scidata_agent.agent.schemas import DiscoveredSource

        return [
            DiscoveredSource(
                title="Light curves of Type Ia supernovae from an offline fixture",
                source_type="paper",
                url="https://arxiv.org/abs/0000.00000",
                query=query,
                description="Offline arXiv-like result for tests.",
                reason="Matched by fake arXiv searcher.",
                confidence=0.8,
                metadata={"provider": "arxiv", "pdf_url": "https://arxiv.org/pdf/0000.00000"},
            )
        ]

    enriched, status = enrich_with_arxiv_results(plan, arxiv_plan, max_results=3, searcher=offline_searcher)

    assert 'all:"Type Ia supernova"' in arxiv_plan.queries[0].query
    assert status == "added=1,searched=1,failed=0"
    assert any(source.source_type == "paper" for source in enriched.candidate_sources)
    assert any(source.metadata.get("provider") == "arxiv" for source in enriched.candidate_sources)
    assert any("arXiv search completed" in note for note in enriched.notes)


def test_arxiv_connector_does_not_expand_domain_terms_itself() -> None:
    raw_query = "tryon"
    normalized = normalize_arxiv_query(raw_query)

    assert normalized == "all:tryon"
    assert "virtual try-on" not in normalized
    assert "VTON" not in normalized


def test_arxiv_search_plan_can_execute_multiple_llm_queries_without_network() -> None:
    plan = fallback_discover_sources("Compare catalysts for CO2 reduction")
    arxiv_plan = ArxivSearchPlan(
        research_goal=plan.research_goal,
        search_intent="Find papers about CO2 reduction catalysts.",
        queries=[
            {"query": 'all:"CO2 reduction" AND all:catalyst', "purpose": "broad catalyst query", "max_results": 2},
            {"query": 'ti:"carbon dioxide reduction" AND abs:catalyst', "purpose": "title/abstract precision query", "max_results": 2},
        ],
        selection_criteria=["Prefer papers with catalyst identity and quantitative performance."],
    )
    seen_queries = []

    def offline_searcher(query: str, max_results: int):
        seen_queries.append((query, max_results))
        return [
            DiscoveredSource(
                title=f"Offline result for {query}",
                source_type="paper",
                url=f"https://arxiv.org/abs/{len(seen_queries):04d}.00000",
                query=query,
                description="Offline arXiv-like result.",
                reason="Matched by fake arXiv searcher.",
                confidence=0.8,
                metadata={"provider": "arxiv", "pdf_url": f"https://arxiv.org/pdf/{len(seen_queries):04d}.00000"},
            )
        ]

    enriched, status = enrich_with_arxiv_results(plan, arxiv_plan, max_results=5, searcher=offline_searcher)

    assert len(seen_queries) == 2
    assert seen_queries[0] == ('all:"CO2 reduction" AND all:catalyst', 2)
    assert seen_queries[1] == ('ti:"carbon dioxide reduction" AND abs:catalyst', 2)
    assert status == "added=2,searched=2,failed=0"
    assert sum(1 for source in enriched.candidate_sources if source.metadata.get("provider") == "arxiv") == 2


def test_arxiv_pdf_download_selection_without_network() -> None:
    plan = fallback_discover_sources("Type Ia supernova light curve")
    arxiv_plan = ArxivSearchPlan(
        research_goal=plan.research_goal,
        queries=[
            {
                "query": 'all:"Type Ia supernova" AND all:"light curve"',
                "purpose": "LLM-planned astronomy search.",
                "max_results": 2,
            }
        ],
    )

    def offline_searcher(query: str, max_results: int):
        from scidata_agent.agent.schemas import DiscoveredSource

        return [
            DiscoveredSource(
                title="Offline arXiv Paper A",
                source_type="paper",
                url="https://arxiv.org/abs/0000.00001",
                query=query,
                description="Offline arXiv-like result A.",
                reason="Matched by fake arXiv searcher.",
                confidence=0.8,
                metadata={"provider": "arxiv", "pdf_url": "https://arxiv.org/pdf/0000.00001"},
            ),
            DiscoveredSource(
                title="Offline arXiv Paper B",
                source_type="paper",
                url="https://arxiv.org/abs/0000.00002",
                query=query,
                description="Offline arXiv-like result B.",
                reason="Matched by fake arXiv searcher.",
                confidence=0.8,
                metadata={"provider": "arxiv", "pdf_url": "https://arxiv.org/pdf/0000.00002"},
            ),
        ]

    enriched, _ = enrich_with_arxiv_results(plan, arxiv_plan, max_results=2, searcher=offline_searcher)
    assert len(select_arxiv_papers(enriched, max_papers=1)) == 1

    download_dir = ROOT / "outputs" / "test-runs" / "offline-arxiv-downloads"

    def offline_downloader(url: str, target_path: Path, timeout: int):
        fixture_pdf = create_pdf_fixture()
        target_path.write_bytes(fixture_pdf.read_bytes())

    downloaded = download_arxiv_pdfs(
        enriched,
        download_dir=download_dir,
        max_papers=1,
        downloader=offline_downloader,
    )

    assert len(downloaded) == 1
    assert downloaded[0].exists()
    assert downloaded[0].suffix == ".pdf"
    assert any("downloaded_path" in source.metadata for source in enriched.candidate_sources)


def test_arxiv_pdf_download_records_failure_and_continues_with_other_candidates(tmp_path: Path) -> None:
    from scidata_agent.tools.connectors.arxiv import download_arxiv_pdfs

    plan = fallback_discover_sources("offline arXiv fallback test")
    sources = [
        DiscoveredSource(
            title=f"Fallback paper {index}",
            source_type="paper",
            url=f"https://arxiv.org/abs/9999.0000{index}",
            metadata={
                "provider": "arxiv",
                "pdf_url": f"https://arxiv.org/pdf/9999.0000{index}",
            },
            confidence=0.9 - index * 0.1,
        )
        for index in range(3)
    ]
    plan.candidate_sources = sources

    def flaky_downloader(url: str, target_path: Path, timeout: int) -> None:
        if url.endswith("0000"):
            raise TimeoutError("fixture timeout")
        target_path.write_bytes(b"%PDF-1.4\nfixture\n%%EOF")

    downloaded = download_arxiv_pdfs(
        plan,
        download_dir=tmp_path / "arxiv-fallback",
        max_papers=3,
        downloader=flaky_downloader,
        retries=0,
        max_workers=1,
    )

    assert len(downloaded) == 2
    assert sources[0].metadata["last_ingestion_status"] == "failed"
    assert sources[0].metadata["last_ingestion_error"] == "fixture timeout"
    assert sources[0].metadata["ingestion_attempts"] == 1
    assert all(source.metadata["last_ingestion_status"] == "completed" for source in sources[1:])


def test_multi_source_search_plan_executes_all_connectors_without_network() -> None:
    plan = MultiSourceSearchPlan(
        research_goal="Survey catalyst datasets and papers.",
        search_requests=[
            SourceSearchRequest(connector_name="arxiv", source_type="paper", query="all:catalyst", max_results=2),
            SourceSearchRequest(connector_name="openalex", source_type="paper_metadata", query="catalyst", max_results=2),
            SourceSearchRequest(connector_name="github", source_type="repository", query="catalyst dataset", max_results=2),
        ],
    )
    seen = []

    def fake_searcher(request: SourceSearchRequest):
        seen.append(request.connector_name)
        return [
            DiscoveredSource(
                title=f"{request.connector_name} source",
                source_type=request.source_type if request.source_type != "paper_search" else "paper",
                url=f"https://example.org/{request.connector_name}",
                query=request.query,
                metadata={"provider": request.connector_name},
            )
        ]

    sources, status = execute_multi_source_search(
        plan,
        searchers={"arxiv": fake_searcher, "openalex": fake_searcher, "github": fake_searcher},
    )

    assert set(seen) == {"arxiv", "openalex", "github"}
    assert status["status"] == "completed"
    assert status["added"] == 3
    assert len(sources) == 3
    assert {source.metadata["provider"] for source in sources} == {"arxiv", "openalex", "github"}


def test_connector_registry_merges_duplicate_sources() -> None:
    existing = [
        DiscoveredSource(
            title="Duplicate paper",
            source_type="paper",
            url="https://doi.org/10.123/demo",
            metadata={"provider": "crossref", "doi": "10.123/demo"},
        )
    ]
    new_sources = [
        DiscoveredSource(
            title="Duplicate paper from OpenAlex",
            source_type="paper_metadata",
            url="https://openalex.org/W1",
            metadata={"provider": "openalex", "doi": "10.123/demo"},
        ),
        DiscoveredSource(
            title="New dataset",
            source_type="dataset",
            url="https://zenodo.org/records/1",
            metadata={"provider": "zenodo"},
        ),
    ]

    merged, added = merge_sources(existing, new_sources)

    assert added == 1
    assert len(merged) == 2
    assert any(source.title == "New dataset" for source in merged)


def test_openalex_connector_maps_work_payload() -> None:
    request = SourceSearchRequest(connector_name="openalex", source_type="paper_metadata", query="wavelet image generation")
    source = openalex_work_to_source(
        {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.555/demo",
            "title": "Wavelet Methods for Image Generation",
            "publication_year": 2025,
            "cited_by_count": 42,
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
            "primary_location": {
                "pdf_url": "https://example.org/paper.pdf",
                "source": {"display_name": "Demo Journal"},
            },
            "open_access": {"oa_url": "https://example.org/open"},
        },
        request,
    )

    assert source.title == "Wavelet Methods for Image Generation"
    assert source.metadata["provider"] == "openalex"
    assert source.metadata["publication_year"] == 2025
    assert source.metadata["authors"] == ["Ada Lovelace"]
    assert source.metadata["pdf_url"] == "https://example.org/paper.pdf"


def test_zenodo_connector_maps_record_payload() -> None:
    request = SourceSearchRequest(connector_name="zenodo", source_type="dataset", query="supernova light curve")
    source = zenodo_record_to_source(
        {
            "id": 123,
            "links": {"html": "https://zenodo.org/records/123"},
            "metadata": {
                "title": "Type Ia Supernova Light Curve Dataset",
                "doi": "10.5281/zenodo.123",
                "publication_date": "2025-01-01",
                "creators": [{"name": "Demo Author"}],
                "keywords": ["supernova", "light curve"],
            },
            "files": [{"key": "data.csv", "type": "csv", "size": 100, "links": {"self": "https://example.org/data.csv"}}],
        },
        request,
    )

    assert source.source_type == "dataset"
    assert source.metadata["provider"] == "zenodo"
    assert source.metadata["doi"] == "10.5281/zenodo.123"
    assert source.metadata["files"][0]["key"] == "data.csv"


def test_github_connector_maps_repo_payload() -> None:
    request = SourceSearchRequest(connector_name="github", source_type="repository", query="wavelet diffusion")
    source = github_repo_to_source(
        {
            "id": 9,
            "full_name": "demo/wavelet-diffusion",
            "html_url": "https://github.com/demo/wavelet-diffusion",
            "description": "Code for wavelet diffusion experiments.",
            "language": "Python",
            "stargazers_count": 100,
            "forks_count": 8,
            "owner": {"login": "demo"},
            "topics": ["diffusion", "wavelet"],
        },
        request,
    )

    assert source.source_type == "repository"
    assert source.metadata["provider"] == "github"
    assert source.metadata["full_name"] == "demo/wavelet-diffusion"
    assert source.metadata["stars"] == 100


def test_source_triage_selects_only_budgeted_arxiv_pdfs() -> None:
    sources = [
        DiscoveredSource(
            title="Relevant arXiv paper A",
            source_type="paper",
            url="https://arxiv.org/abs/0000.00001",
            query="wavelet image generation",
            confidence=0.9,
            metadata={"provider": "arxiv", "pdf_url": "https://arxiv.org/pdf/0000.00001"},
        ),
        DiscoveredSource(
            title="Relevant arXiv paper B",
            source_type="paper",
            url="https://arxiv.org/abs/0000.00002",
            query="wavelet image generation",
            confidence=0.8,
            metadata={"provider": "arxiv", "pdf_url": "https://arxiv.org/pdf/0000.00002"},
        ),
        DiscoveredSource(
            title="Relevant arXiv paper C",
            source_type="paper",
            url="https://arxiv.org/abs/0000.00003",
            query="wavelet image generation",
            confidence=0.7,
            metadata={"provider": "arxiv", "pdf_url": "https://arxiv.org/pdf/0000.00003"},
        ),
    ]

    decisions = triage_sources(sources, "wavelet image generation", max_pdf_downloads=2)

    assert sum(1 for decision in decisions if decision.recommended_action == "download_pdf") == 2
    assert len(ingestible_arxiv_source_ids(decisions)) == 2
    assert sources[0].metadata["triage_action"] == "download_pdf"
    assert sources[2].metadata["triage_action"] == "read_metadata"


def test_source_triage_downloads_open_pdfs_across_providers_with_budget() -> None:
    sources = [
        DiscoveredSource(
            title="OpenAlex metadata result",
            source_type="paper_metadata",
            url="https://openalex.org/W123",
            confidence=0.8,
            metadata={"provider": "openalex", "doi": "10.555/demo", "pdf_url": "https://example.org/demo.pdf"},
        ),
        DiscoveredSource(
            title="GitHub repository result",
            source_type="repository",
            url="https://github.com/demo/repo",
            confidence=0.8,
            metadata={"provider": "github", "stars": 100},
        ),
    ]

    decisions = triage_sources(sources, "demo research", max_pdf_downloads=1)

    by_provider = {decision.provider: decision for decision in decisions}
    assert by_provider["openalex"].recommended_action == "download_pdf"
    assert by_provider["openalex"].should_ingest
    assert len(ingestible_pdf_source_ids(decisions)) == 1
    assert by_provider["github"].recommended_action == "read_readme"
    assert not by_provider["github"].should_ingest


def test_source_triage_uses_llm_selection_and_rejects_off_topic_sources() -> None:
    recent_relevant = DiscoveredSource(
        title="A2RD: Agentic Autoregressive Diffusion for Long Video Consistency",
        source_type="paper",
        url="https://arxiv.org/abs/2601.00001",
        query="long video consistency",
        confidence=0.82,
        metadata={"provider": "arxiv", "published": "2026-01-10", "pdf_url": "https://arxiv.org/pdf/2601.00001"},
    )
    stale_paper = DiscoveredSource(
        title="Deep Video Prior",
        source_type="paper",
        url="https://arxiv.org/abs/2201.00001",
        query="long video consistency",
        confidence=0.95,
        metadata={"provider": "arxiv", "published": "2022-01-10", "pdf_url": "https://arxiv.org/pdf/2201.00001"},
    )
    generic_proceedings = DiscoveredSource(
        title="Computer Analysis of Images and Patterns, 2007 proceedings",
        source_type="paper_metadata",
        url="https://example.org/caip-2007",
        query="long video consistency",
        confidence=0.9,
        metadata={"provider": "semantic_scholar", "publication_year": 2007, "pdf_url": "https://example.org/caip.pdf"},
    )
    github = DiscoveredSource(
        title="demo/long-video-consistency",
        source_type="repository",
        url="https://github.com/demo/long-video-consistency",
        confidence=0.7,
        metadata={"provider": "github", "full_name": "demo/long-video-consistency"},
    )
    plan = SourceSelectionPlan.model_validate(
        {
            "research_goal": "Investigate past-year long video generation consistency papers and code.",
            "selection_summary": "Prefer recent directly related papers and code repositories.",
            "time_range_interpreted": "2025-07-21 to 2026-07-21",
            "decisions": [
                {
                    "source_id": recent_relevant.source_id,
                    "decision": "deep_read",
                    "priority": "high",
                    "source_role": "primary_paper",
                    "priority_score": 0.95,
                    "reason": "Recent paper directly matches long video consistency.",
                    "matched_requirements": ["past year", "long video consistency"],
                    "expected_extractable_fields": ["method architecture", "datasets", "metrics"],
                },
                {
                    "source_id": stale_paper.source_id,
                    "decision": "reject",
                    "priority": "low",
                    "source_role": "noise",
                    "priority_score": 0.1,
                    "reason": "Outside the requested past-year time range.",
                    "risk_notes": ["stale"],
                },
                {
                    "source_id": generic_proceedings.source_id,
                    "decision": "reject",
                    "priority": "low",
                    "source_role": "noise",
                    "priority_score": 0.05,
                    "reason": "Generic proceedings page is not a direct paper about the requested topic.",
                },
                {
                    "source_id": github.source_id,
                    "decision": "read_readme",
                    "priority": "medium",
                    "source_role": "code_repository",
                    "priority_score": 0.75,
                    "reason": "Repository may contain code and reproducibility details.",
                },
            ],
        }
    )

    decisions = triage_sources_from_selection(
        [recent_relevant, stale_paper, generic_proceedings, github],
        plan,
        max_pdf_downloads=1,
    )

    by_title = {decision.title: decision for decision in decisions}
    assert by_title[recent_relevant.title].recommended_action == "download_pdf"
    assert by_title[recent_relevant.title].should_ingest
    assert by_title[stale_paper.title].recommended_action == "skip"
    assert not by_title[stale_paper.title].should_ingest
    assert by_title[generic_proceedings.title].recommended_action == "skip"
    assert by_title[github.title].recommended_action == "read_readme"
    assert recent_relevant.metadata["selection_decision"] == "deep_read"
    assert stale_paper.metadata["selection_decision"] == "reject"


def test_source_triage_caps_llm_selected_auto_resources_at_30() -> None:
    sources = [
        DiscoveredSource(
            title=f"Relevant arXiv paper {index:02d}",
            source_type="paper",
            url=f"https://arxiv.org/abs/2601.{index:05d}",
            query="broad survey",
            confidence=0.9 - index * 0.001,
            metadata={"provider": "arxiv", "pdf_url": f"https://arxiv.org/pdf/2601.{index:05d}"},
        )
        for index in range(35)
    ]
    plan = SourceSelectionPlan.model_validate(
        {
            "research_goal": "Broad survey that finds many relevant papers.",
            "selection_summary": "All papers are relevant enough for comparison.",
            "decisions": [
                {
                    "source_id": source.source_id,
                    "decision": "deep_read",
                    "priority": "high",
                    "source_role": "primary_paper",
                    "priority_score": 1.0 - index * 0.01,
                    "reason": "Directly relevant and should be deep-read if within the resource cap.",
                    "matched_requirements": ["broad survey"],
                    "expected_extractable_fields": ["method", "dataset", "metrics"],
                }
                for index, source in enumerate(sources)
            ],
        }
    )

    decisions = triage_sources_from_selection(sources, plan, max_auto_resources=30)

    assert sum(1 for decision in decisions if decision.recommended_action == "download_pdf") == 30
    assert len(ingestible_arxiv_source_ids(decisions)) == 30
    assert decisions[0].recommended_action == "download_pdf"
    assert decisions[29].recommended_action == "download_pdf"
    assert decisions[30].recommended_action == "read_metadata"
    assert decisions[30].metadata["resource_cap_deferred"]
    assert decisions[30].metadata["original_recommended_action"] == "download_pdf"


def test_source_triage_rejects_solar_physics_when_perovskite_is_required() -> None:
    off_topic = DiscoveredSource(
        title="Solar cycle variation in solar f-mode frequencies and radius",
        source_type="paper",
        url="https://arxiv.org/abs/2401.00001",
        query="perovskite solar-cell stability",
        confidence=0.95,
        metadata={"provider": "arxiv", "pdf_url": "https://arxiv.org/pdf/2401.00001"},
    )
    on_topic = DiscoveredSource(
        title="Photo Stabilization of p-i-n Perovskite Solar Cells",
        source_type="paper",
        url="https://arxiv.org/abs/2401.00002",
        query="perovskite solar-cell stability",
        confidence=0.9,
        metadata={"provider": "arxiv", "pdf_url": "https://arxiv.org/pdf/2401.00002"},
    )
    plan = SourceSelectionPlan(
        research_goal="研究钙钛矿太阳能电池的稳定性和效率。",
        decisions=[
            SourceSelectionDecision(
                source_id=source.source_id,
                decision="deep_read",
                priority="high",
                source_role="primary_paper",
                priority_score=0.9,
                reason="The source appears relevant.",
            )
            for source in (off_topic, on_topic)
        ],
    )

    decisions = triage_sources_from_selection(
        [off_topic, on_topic],
        plan,
        research_question=plan.research_goal,
        max_auto_resources=2,
    )

    by_title = {decision.title: decision for decision in decisions}
    assert by_title[off_topic.title].recommended_action == "skip"
    assert not by_title[off_topic.title].should_ingest
    assert by_title[off_topic.title].metadata["topic_guard"] == "missing_required_material_term"
    assert by_title[on_topic.title].recommended_action == "download_pdf"


def test_source_triage_keeps_paper_indexes_as_metadata_without_pdf() -> None:
    sources = [
        DiscoveredSource(
            title="Crossref metadata result",
            source_type="paper_metadata",
            url="https://doi.org/10.555/demo",
            confidence=0.8,
            metadata={"provider": "crossref", "doi": "10.555/demo"},
        )
    ]

    decisions = triage_sources(sources, "demo research", max_pdf_downloads=2)

    assert decisions[0].recommended_action == "read_metadata"
    assert not decisions[0].should_ingest


def test_source_triage_selects_small_tables_and_blocks_large_archives() -> None:
    small = DiscoveredSource(
        title="Small Figshare table",
        source_type="dataset",
        url="https://figshare.com/articles/1",
        confidence=0.8,
        metadata={
            "provider": "figshare",
            "files": [{"name": "results.csv", "size": 1024, "download_url": "https://example.org/results.csv"}],
        },
    )
    large = DiscoveredSource(
        title="Large Zenodo archive",
        source_type="dataset",
        url="https://zenodo.org/records/1",
        confidence=0.8,
        metadata={
            "provider": "zenodo",
            "files": [{"key": "archive.zip", "size": 200 * 1024 * 1024, "url": "https://example.org/archive.zip"}],
        },
    )

    decisions = triage_sources([small, large], "dataset results", max_pdf_downloads=2)

    by_title = {decision.title: decision for decision in decisions}
    assert by_title["Small Figshare table"].recommended_action == "download_small_table"
    assert by_title["Small Figshare table"].should_ingest
    assert by_title["Large Zenodo archive"].recommended_action == "ask_user"
    assert not by_title["Large Zenodo archive"].should_ingest


def test_multi_source_ingestion_creates_research_blocks_and_downloads_small_tables() -> None:
    github = DiscoveredSource(
        title="demo/wavelet-diffusion",
        source_type="repository",
        url="https://github.com/demo/wavelet-diffusion",
        confidence=0.8,
        metadata={"provider": "github", "full_name": "demo/wavelet-diffusion"},
    )
    figshare = DiscoveredSource(
        title="Small Figshare table",
        source_type="dataset",
        url="https://figshare.com/articles/1",
        confidence=0.8,
        metadata={
            "provider": "figshare",
            "files": [{"name": "results.csv", "size": 1024, "download_url": "https://example.org/results.csv"}],
        },
    )
    decisions = triage_sources([github, figshare], "wavelet diffusion results dataset", max_pdf_downloads=1)
    output_dir = ROOT / "outputs" / "test-runs"

    def fake_fetcher(url: str, max_bytes: int, headers: dict[str, str] | None = None) -> str:
        assert "readme" in url.lower()
        return "# Wavelet Diffusion\nThis repository contains training code and experiment configs."

    def fake_downloader(url: str, target_path: Path, max_bytes: int) -> None:
        assert url == "https://example.org/results.csv"
        target_path.write_text("method,FID\nWaveletDiffusion,12.3\n", encoding="utf-8")

    files, text_blocks, insights, logs = ingest_triaged_sources(
        [github, figshare],
        decisions,
        output_dir=output_dir,
        task_id="offline_multi_source_ingestion",
        downloader=fake_downloader,
        text_fetcher=fake_fetcher,
    )

    assert any(insight.insight_type == "readme" for insight in insights)
    assert any(insight.insight_type == "downloaded_file" for insight in insights)
    assert any("Wavelet Diffusion" in block.text for block in text_blocks)
    assert len(files) == 1
    assert files[0].path.suffix == ".csv"
    assert any("Downloaded source file" in line for line in logs)
