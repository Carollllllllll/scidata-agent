from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from scidata_agent.agent.schemas import ChartExtraction, ChartSeries, UploadedFile
from scidata_agent.agent.scidata_agent import SciDataAgent
from scidata_agent.llm.nodes import QwenAgentNodes
from scidata_agent.tools.chart_locator import locate_figures
from scidata_agent.tools.chart_validator import validate_chart_extraction
from tests.create_fixtures import create_pdf_fixture
from tests.test_agent_mvp import MockQwenClient


ROOT = Path(__file__).resolve().parents[2]


def create_chart_pdf_fixture() -> Path:
    """A one-page PDF with a simple line chart and a Figure 1 caption."""
    out_dir = ROOT / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "demo_chart_paper.pdf"
    if path.exists():
        return path
    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(72, 740, "Demo Chart Paper")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(72, 710, "We measured the PCE retention of the device over time.")
    # Chart graphics: plot box, axes, and a polyline.
    pdf.rect(120, 420, 300, 200)
    pdf.line(120, 420, 120, 620)
    pdf.line(120, 420, 420, 420)
    pdf.line(130, 440, 200, 500)
    pdf.line(200, 500, 300, 560)
    pdf.line(300, 560, 410, 600)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(110, 412, "0")
    pdf.drawString(400, 405, "10 days")
    pdf.drawString(90, 615, "25 %")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(72, 380, "Figure 1: PCE retention versus time for the MAPbI3 device.")
    pdf.showPage()
    pdf.setFont("Helvetica", 10)
    pdf.drawString(72, 740, "No figures on this page.")
    pdf.save()
    return path


class MockVisionQwenClient(MockQwenClient):
    """Mock client that also answers Qwen-VL chart nodes without network calls."""

    def generate_vision_json(
        self,
        node: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str | Path],
        temperature: float = 0.1,
    ) -> Any:
        if node == "qwen_vl_chart_classifier":
            return {"chart_type": "line", "contains_data": True, "reason": "mock line chart", "confidence": 0.9}
        if node == "qwen_vl_chart_extractor":
            return {
                "title": "PCE retention",
                "x_axis": {"label": "time", "unit": "days", "scale": "linear", "range_min": 0, "range_max": 10},
                "y_axis": {"label": "PCE retention", "unit": "%", "scale": "linear", "range_min": 0, "range_max": 25},
                "series": [
                    {"name": "MAPbI3", "points": [[0, 5.0], [5, 15.0], [10, 21.0]], "point_style": "line"}
                ],
                "notes": ["mock extraction"],
                "confidence": 0.8,
            }
        raise AssertionError(f"unexpected vision node: {node}")


def test_chart_locator_renders_figure_png() -> None:
    pdf_path = create_chart_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs" / "chart-locator"
    uploaded = UploadedFile(filename=pdf_path.name, path=pdf_path)

    assets = locate_figures(uploaded, output_dir)

    assert len(assets) == 1
    asset = assets[0]
    assert asset.label == "Figure 1"
    assert asset.caption and "PCE retention" in asset.caption
    assert asset.page == 1
    assert asset.image_path and Path(asset.image_path).exists()
    assert asset.bbox is not None and (asset.bbox[3] - asset.bbox[1]) > 80


def test_chart_locator_ignores_pdf_without_figures() -> None:
    pdf_path = create_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs" / "chart-locator-empty"
    uploaded = UploadedFile(filename=pdf_path.name, path=pdf_path)

    assets = locate_figures(uploaded, output_dir)

    assert assets == []


def test_chart_nodes_with_mock_vision() -> None:
    pdf_path = create_chart_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs" / "chart-nodes"
    uploaded = UploadedFile(filename=pdf_path.name, path=pdf_path)
    asset = locate_figures(uploaded, output_dir)[0]

    nodes = QwenAgentNodes(MockVisionQwenClient())
    classification = nodes.classify_chart(asset)
    assert classification["contains_data"] is True
    assert classification["chart_type"] == "line"

    extraction = nodes.extract_chart_data(asset, classification["chart_type"])
    assert extraction.figure_id == asset.figure_id
    assert extraction.source_file == asset.source_file
    assert extraction.page == 1
    assert extraction.chart_type == "line"
    assert extraction.x_axis.unit == "days"
    assert len(extraction.series) == 1
    assert extraction.series[0].points == [[0.0, 5.0], [5.0, 15.0], [10.0, 21.0]]
    assert extraction.approximate is True


def test_chart_validator_detects_axis_range_mismatch() -> None:
    extraction = ChartExtraction(
        figure_id="fig_test",
        source_file="demo.pdf",
        page=1,
        chart_type="line",
        contains_data=True,
        confidence=0.8,
    )
    extraction.x_axis.range_min = 0.0
    extraction.x_axis.range_max = 10.0
    extraction.y_axis.range_min = 0.0
    extraction.y_axis.range_max = 25.0
    extraction.series = [ChartSeries(name="s1", points=[[1.0, 5.0], [50.0, 90.0], [80.0, 120.0]])]

    result = validate_chart_extraction(extraction)

    assert result.passed is False
    assert result.needs_review is True
    assert any(issue.code == "axis_range_mismatch" and issue.severity == "error" for issue in result.issues)


def test_chart_validator_passes_consistent_extraction() -> None:
    extraction = ChartExtraction(
        figure_id="fig_ok",
        source_file="demo.pdf",
        page=1,
        chart_type="line",
        contains_data=True,
        confidence=0.85,
    )
    extraction.x_axis.range_min = 0.0
    extraction.x_axis.range_max = 10.0
    extraction.y_axis.range_min = 0.0
    extraction.y_axis.range_max = 25.0
    extraction.series = [ChartSeries(name="s1", points=[[1.0, 5.0], [5.0, 15.0], [9.0, 21.0]])]

    result = validate_chart_extraction(extraction)

    assert result.passed is True
    assert result.needs_review is False
    assert result.issues == []


def test_chart_validator_flags_caption_unit_conflict() -> None:
    pdf_path = create_chart_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs" / "chart-validator-unit"
    uploaded = UploadedFile(filename=pdf_path.name, path=pdf_path)
    asset = locate_figures(uploaded, output_dir)[0]
    asset.caption = "Figure 1: The light curve peaks at 550 nm after 2 days."

    extraction = ChartExtraction(
        figure_id=asset.figure_id,
        source_file=asset.source_file,
        page=1,
        chart_type="line",
        contains_data=True,
        confidence=0.8,
    )
    extraction.x_axis.range_min = 0.0
    extraction.x_axis.range_max = 10.0
    extraction.x_axis.unit = "days"
    extraction.y_axis.range_min = 0.0
    extraction.y_axis.range_max = 25.0
    extraction.y_axis.unit = "%"
    extraction.series = [ChartSeries(name="s1", points=[[1.0, 5.0], [5.0, 15.0]])]

    result = validate_chart_extraction(extraction, asset)

    assert any(issue.code == "unit_suspect" for issue in result.issues)


def test_chart_pipeline_end_to_end_with_mock_vision() -> None:
    pdf_path = create_chart_pdf_fixture()
    output_dir = ROOT / "outputs" / "test-runs"
    agent = SciDataAgent(
        output_dir=output_dir,
        llm_client=MockVisionQwenClient(),
        require_llm=True,
        monitor_console=False,
    )

    result = agent.run(
        "Extract PCE retention metrics and chart data from the uploaded paper.",
        [pdf_path],
        max_pdf_pages=3,
    )

    assert result.status == "completed"
    assert result.summary.figures_detected == 1
    assert result.summary.charts_extracted == 1
    assert result.summary.charts_needs_review == 0
    assert result.export_files.chart_extractions_json and Path(result.export_files.chart_extractions_json).exists()
    assert result.export_files.chart_validation_json and Path(result.export_files.chart_validation_json).exists()
    assert result.export_files.figures_dir and Path(result.export_files.figures_dir).exists()
    assert result.export_files.chart_tables_dir and Path(result.export_files.chart_tables_dir).exists()

    chart_payload = json.loads(Path(result.export_files.chart_extractions_json).read_text(encoding="utf-8"))
    assert len(chart_payload["figures"]) == 1
    assert len(chart_payload["extractions"]) == 1
    assert chart_payload["extractions"][0]["series"][0]["points"]

    chart_index = Path(result.export_files.chart_tables_dir) / "chart_data_index.csv"
    assert chart_index.exists()
    chart_csvs = list(Path(result.export_files.chart_tables_dir).glob("chart_data_chart_*.csv"))
    assert chart_csvs, "per-chart long-format CSV should be exported"

    validations = json.loads(Path(result.export_files.chart_validation_json).read_text(encoding="utf-8"))
    assert validations and validations[0]["passed"] is True
    assert any("Chart validation merged" in note for note in result.quality_report.notes)

    monitor_events = [
        json.loads(line)
        for line in Path(result.export_files.monitor_log).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        event["step"] == "figure_chart_extraction" and event["status"] == "completed"
        for event in monitor_events
    )
