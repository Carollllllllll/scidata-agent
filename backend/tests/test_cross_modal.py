from __future__ import annotations

from scidata_agent.agent.schemas import (
    ChartExtraction,
    ChartSeries,
    FigureAsset,
    SourceType,
    TableBlock,
    TextBlock,
)
from scidata_agent.tools.cross_modal import build_cross_modal_checks


def test_chart_is_supported_by_same_page_text_and_table() -> None:
    figure = FigureAsset(
        figure_id="figure-1",
        source_file="paper.pdf",
        source_path="paper.pdf",
        page=3,
        caption="Performance curve; reported value 0.80.",
    )
    extraction = ChartExtraction(
        figure_id="figure-1",
        source_file="paper.pdf",
        page=3,
        contains_data=True,
        series=[ChartSeries(name="model", points=[[1.0, 0.80]])],
    )
    text = TextBlock(
        source_file="paper.pdf",
        source_path="paper.pdf",
        source_type=SourceType.PDF_TEXT,
        page=3,
        text="The model reaches 0.80 on the benchmark.",
        chunk_id="page-3",
    )
    table = TableBlock(
        source_file="paper.pdf",
        source_path="paper.pdf",
        source_type=SourceType.PDF_TABLE,
        columns=["score"],
        rows=[{"score": 0.80}],
        table_id="table-1",
        page=3,
    )

    checks = build_cross_modal_checks([text], [table], [figure], [extraction])

    figure_check = next(check for check in checks if check.subject_id == "figure-1")
    assert figure_check.status == "supported"
    assert figure_check.modalities == ["figure", "text", "table"]
    assert figure_check.matched_value_count >= 1


def test_qualitative_figure_is_not_comparable_not_a_failure() -> None:
    figure = FigureAsset(
        figure_id="figure-qualitative",
        source_file="paper.pdf",
        source_path="paper.pdf",
        page=4,
        caption="Overview of the proposed architecture.",
    )
    extraction = ChartExtraction(
        figure_id="figure-qualitative",
        source_file="paper.pdf",
        page=4,
        contains_data=False,
    )

    checks = build_cross_modal_checks([], [], [figure], [extraction])

    assert checks[0].status == "not_comparable"
    assert checks[0].confidence == 0
    assert "qualitative" in checks[0].issues[0]


def test_chart_without_numeric_correspondence_is_partial() -> None:
    figure = FigureAsset(
        figure_id="figure-partial",
        source_file="paper.pdf",
        source_path="paper.pdf",
        page=5,
    )
    extraction = ChartExtraction(
        figure_id="figure-partial",
        source_file="paper.pdf",
        page=5,
        contains_data=True,
        series=[ChartSeries(points=[[1.0, 0.80]])],
    )
    text = TextBlock(
        source_file="paper.pdf",
        source_path="paper.pdf",
        source_type=SourceType.PDF_TEXT,
        page=5,
        text="The experiment uses the standard benchmark.",
        chunk_id="page-5",
    )

    checks = build_cross_modal_checks([text], [], [figure], [extraction])

    assert checks[0].status == "partial"
    assert checks[0].matched_value_count == 0
