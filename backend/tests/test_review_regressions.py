from __future__ import annotations

from threading import Barrier

import pytest

from scidata_agent.agent.schemas import (
    DiscoveredSource,
    DynamicExtractionPlan,
    MultiSourceSearchPlan,
    ScientificRecord,
    SourceSearchRequest,
    SourceType,
    TaskPlan,
    TextBlock,
    UploadedFile,
)
from scidata_agent.llm.nodes import QwenAgentNodes, _dynamic_records_from_payload
from scidata_agent.tools.extractor import _coerce_number_and_unit
from scidata_agent.tools.connectors.registry import execute_multi_source_search
from scidata_agent.tools.connectors import arxiv
from scidata_agent.tools.parser import parse_csv, parse_sources
from scidata_agent.tools.quality import _value_supported_by_evidence, build_quality_report


def _uploaded(path):
    return UploadedFile(filename=path.name, path=path)


def test_parse_sources_continues_after_malformed_csv(tmp_path) -> None:
    malformed = tmp_path / "malformed.csv"
    malformed.write_text('name,value\nbroken,"unterminated\n', encoding="utf-8")
    valid = tmp_path / "valid.csv"
    valid.write_text("name,value\nalpha,3.2\n", encoding="utf-8")

    parsed = parse_sources([_uploaded(malformed), _uploaded(valid)])

    assert [table.source_file for table in parsed.tables] == ["valid.csv"]
    assert any("malformed.csv" in warning for warning in parsed.parser_warnings)


def test_parse_csv_accepts_gb18030_input(tmp_path) -> None:
    path = tmp_path / "legacy.csv"
    path.write_bytes("名称,数值\n效率,25.1\n".encode("gb18030"))

    table = parse_csv(_uploaded(path))

    assert table.columns == ["名称", "数值"]
    assert table.rows[0]["名称"] == "效率"


def test_measurement_parser_ignores_digits_embedded_in_model_names() -> None:
    assert _coerce_number_and_unit("GPT-4") == (None, None)
    assert _coerce_number_and_unit("model_v2") == (None, None)
    assert _coerce_number_and_unit("PCE = -4.2 %") == (-4.2, "%")


def test_evidence_number_matching_respects_numeric_boundaries() -> None:
    unsupported = ScientificRecord(
        metric_name="score",
        metric_value=0.5,
        source_file="paper.pdf",
        source_type=SourceType.PDF_TEXT,
        evidence_text="The reported score was 10.55.",
    )
    supported = unsupported.model_copy(update={"evidence_text": "The reported score was 0.5."})

    assert _value_supported_by_evidence(unsupported) is False
    assert _value_supported_by_evidence(supported) is True


def test_quality_report_is_pure_by_default_and_explicit_mutation_is_idempotent() -> None:
    record = ScientificRecord(
        metric_name="FID",
        metric_value=1.2,
        unit=None,
        source_file="paper.pdf",
        evidence_text="FID is discussed without a numeric result.",
        confidence=0.9,
    )

    build_quality_report([record])
    assert record.unit is None
    assert record.confidence == 0.9
    assert record.warnings == []

    build_quality_report([record], mutate_records=True)
    first_confidence = record.confidence
    first_warnings = list(record.warnings)
    assert record.unit == "dimensionless"
    build_quality_report([record], mutate_records=True)
    assert record.confidence == first_confidence
    assert record.warnings == first_warnings


def test_dynamic_confidence_keeps_zero_and_recovers_invalid_value() -> None:
    plan = DynamicExtractionPlan.model_validate(
        {
            "research_goal": "Extract scores",
            "dynamic_tables": [
                {
                    "table_name": "scores",
                    "fields": [{"name": "score", "type": "number", "required": True}],
                }
            ],
        }
    )
    records = _dynamic_records_from_payload(
        [
            {"table_name": "scores", "fields": {"score": 1}, "confidence": 0},
            {"table_name": "scores", "fields": {"score": 2}, "confidence": "not-a-number"},
        ],
        plan,
        "scores.csv",
        SourceType.CSV,
        None,
    )

    assert [record.confidence for record in records] == [0.0, 0.65]
    assert any("invalid confidence" in warning for warning in records[1].warnings)


def test_connector_requests_execute_concurrently_in_deterministic_result_order() -> None:
    barrier = Barrier(2, timeout=1)
    plan = MultiSourceSearchPlan(
        research_goal="Find papers",
        search_requests=[
            SourceSearchRequest(connector_name="arxiv", source_type="paper", query="alpha"),
            SourceSearchRequest(connector_name="openalex", source_type="paper", query="beta"),
        ],
    )

    def search(request: SourceSearchRequest):
        barrier.wait()
        return [
            DiscoveredSource(
                title=request.query,
                source_type="paper",
                url=f"https://example.org/{request.query}",
            )
        ]

    sources, status = execute_multi_source_search(
        plan,
        searchers={"arxiv": search, "openalex": search},
    )

    assert [source.title for source in sources] == ["alpha", "beta"]
    assert [item["connector"] for item in status["connector_status"]] == ["arxiv", "openalex"]


def test_text_block_llm_extraction_runs_with_bounded_concurrency(monkeypatch) -> None:
    barrier = Barrier(2, timeout=1)

    class Client:
        configured = True

        def generate_json(self, *_args, **_kwargs):
            barrier.wait()
            return [{"metric_name": "score", "metric_value": 1, "evidence_text": "score 1"}]

    monkeypatch.setenv("SCIDATA_LLM_BLOCK_WORKERS", "2")
    nodes = QwenAgentNodes(Client())  # type: ignore[arg-type]
    blocks = [
        TextBlock(
            source_file="paper.pdf",
            source_path="paper.pdf",
            source_type=SourceType.PDF_TEXT,
            page=page,
            text=f"score 1 on page {page}",
            chunk_id=f"p{page}",
        )
        for page in (1, 2)
    ]

    records = nodes.extract_from_text_blocks_limited(TaskPlan(), blocks, max_blocks=2)

    assert [record.page for record in records] == [1, 2]


def test_arxiv_pdf_download_stops_at_byte_limit(monkeypatch, tmp_path) -> None:
    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            return b"%PDF-" + (b"x" * 20)

    monkeypatch.setattr(arxiv, "safe_urlopen", lambda *_args, **_kwargs: Response())
    target = tmp_path / "large.pdf"

    with pytest.raises(arxiv.ArxivConnectorError, match="size limit"):
        arxiv.download_pdf("https://arxiv.org/pdf/1234.5678", target, max_bytes=10)

    assert not target.exists()
    assert not target.with_name("large.pdf.part").exists()
