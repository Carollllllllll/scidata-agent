from __future__ import annotations

import threading
import time
from pathlib import Path

from scidata_agent.agent.schemas import (
    AgentState,
    ChartExtraction,
    FigureAsset,
    SourceType,
    TableBlock,
    TaskPlan,
    TextBlock,
    UploadedFile,
)
from scidata_agent.agent.scidata_agent import SciDataAgent
from scidata_agent.llm.nodes import QwenAgentNodes


class _ConfiguredClient:
    configured = True
    vl_model = "test-vl"


def _tracking_worker(delay: float = 0.06):
    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    def enter() -> None:
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])

    def leave() -> None:
        with lock:
            state["active"] -= 1

    def run(prompt: str) -> list[dict]:
        enter()
        try:
            time.sleep(delay)
            if "bad-source" in prompt:
                raise RuntimeError("simulated block failure")
            return [{"metric_name": "test_metric", "metric_value": 1.0}]
        finally:
            leave()

    return state, run


def _text_blocks() -> list[TextBlock]:
    return [
        TextBlock(
            source_file=f"paper-{index}.pdf",
            source_path=f"paper-{index}.pdf",
            page=index,
            text="Experimental results report the test metric.",
            chunk_id=f"chunk-{index}",
        )
        for index in range(4)
    ]


def test_text_extraction_is_parallel_and_ordered() -> None:
    state, fake_generate = _tracking_worker()
    nodes = QwenAgentNodes(_ConfiguredClient())
    nodes._generate_json_with_retries = lambda node, system, user_prompt, **kwargs: fake_generate(user_prompt)

    records = nodes.extract_from_text_blocks_limited(
        TaskPlan(target_fields=["metric_name", "metric_value"]),
        _text_blocks(),
        max_workers=4,
    )

    assert state["max_active"] >= 2
    assert [record.source_file for record in records] == [
        "paper-0.pdf",
        "paper-1.pdf",
        "paper-2.pdf",
        "paper-3.pdf",
    ]


def test_text_extraction_isolates_one_worker_failure() -> None:
    state, fake_generate = _tracking_worker()
    nodes = QwenAgentNodes(_ConfiguredClient())
    nodes._generate_json_with_retries = lambda node, system, user_prompt, **kwargs: fake_generate(user_prompt)
    blocks = _text_blocks()
    blocks[1].source_file = "bad-source.pdf"

    records = nodes.extract_from_text_blocks_limited(
        TaskPlan(target_fields=["metric_name", "metric_value"]),
        blocks,
        max_workers=4,
    )

    assert state["max_active"] >= 2
    assert len(records) == 3
    assert len(nodes.extraction_warnings) == 1
    assert "bad-source.pdf" in nodes.extraction_warnings[0]


def test_table_extraction_is_parallel_and_ordered() -> None:
    state, fake_generate = _tracking_worker()
    nodes = QwenAgentNodes(_ConfiguredClient())
    nodes._generate_json_with_retries = lambda node, system, user_prompt, **kwargs: fake_generate(user_prompt)
    tables = [
        TableBlock(
            source_file=f"table-{index}.pdf",
            source_path=f"table-{index}.pdf",
            source_type=SourceType.PDF_TABLE,
            columns=["metric"],
            rows=[{"metric": index}],
            table_id=f"table-{index}",
            page=index,
        )
        for index in range(4)
    ]

    records = nodes.extract_from_tables(
        TaskPlan(target_fields=["metric_name", "metric_value"]),
        tables,
        max_workers=4,
    )

    assert state["max_active"] >= 2
    assert [record.raw["table_id"] for record in records] == [
        "table-0",
        "table-1",
        "table-2",
        "table-3",
    ]


class _ChartNodes:
    def __init__(self):
        self._state = {"active": 0, "max_active": 0}
        self._lock = threading.Lock()

    def _run(self):
        with self._lock:
            self._state["active"] += 1
            self._state["max_active"] = max(self._state["max_active"], self._state["active"])
        try:
            time.sleep(0.06)
        finally:
            with self._lock:
                self._state["active"] -= 1

    def classify_chart(self, asset):
        self._run()
        return {"chart_type": "line", "contains_data": True}

    def extract_chart_data(self, asset, chart_type):
        self._run()
        return ChartExtraction(
            figure_id=asset.figure_id,
            source_file=asset.source_file,
            page=asset.page,
            chart_type=chart_type,
            contains_data=True,
            confidence=0.9,
        )


def test_chart_pipeline_processes_figures_in_parallel(monkeypatch, tmp_path: Path) -> None:
    assets = [
        FigureAsset(
            figure_id=f"figure-{index}",
            source_file="paper.pdf",
            source_path="paper.pdf",
            page=index + 1,
            label=f"Figure {index + 1}",
        )
        for index in range(4)
    ]
    monkeypatch.setattr(
        "scidata_agent.agent.scidata_agent.locate_figures",
        lambda *args, **kwargs: assets,
    )
    chart_nodes = _ChartNodes()
    agent = SciDataAgent(
        output_dir=tmp_path,
        llm_client=_ConfiguredClient(),
        monitor_console=False,
        monitor_enabled=False,
    )
    agent.llm_nodes = chart_nodes
    state = AgentState(
        task_id="parallel-chart-test",
        research_question="test chart extraction",
        files=[UploadedFile(filename="paper.pdf", path=tmp_path / "paper.pdf")],
        output_dir=tmp_path,
    )

    agent._extract_charts(state, max_workers=4)

    assert chart_nodes._state["max_active"] >= 2
    assert [item.figure_id for item in state.chart_extractions] == [
        "figure-0",
        "figure-1",
        "figure-2",
        "figure-3",
    ]
    assert len(state.chart_validations) == 4
