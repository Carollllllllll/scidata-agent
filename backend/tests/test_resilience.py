from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from scidata_agent.agent.checkpoint import AgentCheckpointStore, build_run_fingerprint
from scidata_agent.agent.schemas import (
    AgentState,
    DiscoveredSource,
    MultiSourceSearchPlan,
    SourceType,
    SourceSearchRequest,
    TableBlock,
    TextBlock,
)
from scidata_agent.tools.connectors.registry import execute_multi_source_search
from scidata_agent.llm.nodes import _expand_table_rows_for_extraction, _expand_text_blocks_for_extraction
from scidata_agent.tools.source_triage import triage_sources_from_selection


def test_checkpoint_round_trip_and_fingerprint_guard() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        state = AgentState(task_id="resume-test", research_question="q", files=[], output_dir=root)
        fingerprint = build_run_fingerprint("q", [], {"max_pages": None})
        store = AgentCheckpointStore(root / state.task_id)

        store.save(state, fingerprint=fingerprint, completed_steps={"task_planning"})

        restored = store.load(fingerprint=fingerprint)
        assert restored is not None
        restored_state, completed_steps = restored
        assert restored_state.task_id == state.task_id
        assert completed_steps == {"task_planning"}
        assert store.load(fingerprint="wrong") is None
        assert store.last_load_reason == "fingerprint_mismatch"


def test_failed_connector_is_retried_and_success_is_cached() -> None:
    plan = MultiSourceSearchPlan(
        research_goal="q",
        search_requests=[SourceSearchRequest(connector_name="arxiv", query="q", max_results=50)],
    )
    calls = {"count": 0}

    def eventually_succeeds(_request: SourceSearchRequest) -> list[DiscoveredSource]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary provider failure")
        return [DiscoveredSource(title="cached paper", source_type="paper", url="https://example.org/paper")]

    with TemporaryDirectory() as directory:
        sources, status = execute_multi_source_search(
            plan,
            searchers={"arxiv": eventually_succeeds},
            max_workers=1,
            cache_dir=directory,
            retry_failed_requests=1,
        )
        assert len(sources) == 1
        assert status["failed"] == 0
        assert calls["count"] == 2

        def should_not_run(_request: SourceSearchRequest) -> list[DiscoveredSource]:
            raise AssertionError("a successful search should be served by the cache")

        cached_sources, cached_status = execute_multi_source_search(
            plan,
            searchers={"arxiv": should_not_run},
            max_workers=1,
            cache_dir=directory,
            retry_failed_requests=0,
        )
        assert len(cached_sources) == 1
        assert cached_status["connector_status"][0]["cache_hit"] is True


def test_zero_resource_cap_means_unlimited() -> None:
    sources = [
        DiscoveredSource(
            title=f"paper-{index}",
            source_type="paper",
            metadata={"provider": "arxiv", "pdf_url": f"https://arxiv.org/pdf/{index}"},
        )
        for index in range(4)
    ]
    from scidata_agent.agent.schemas import SourceSelectionDecision, SourceSelectionPlan

    plan = SourceSelectionPlan(
        research_goal="q",
        decisions=[
            SourceSelectionDecision(
                source_id=source.source_id,
                decision="deep_read",
                source_role="primary_paper",
                reason="relevant",
            )
            for source in sources
        ],
    )
    decisions = triage_sources_from_selection(sources, plan, max_auto_resources=0)
    assert sum(decision.should_ingest for decision in decisions) == len(sources)


def test_long_text_is_split_without_dropping_content() -> None:
    block = TextBlock(
        source_file="paper.pdf",
        source_path="paper.pdf",
        source_type=SourceType.PDF_TEXT,
        page=1,
        text="\n".join(f"line-{index}" for index in range(1200)),
        chunk_id="page-1",
    )
    chunks = _expand_text_blocks_for_extraction([block])
    assert len(chunks) > 1
    assert "".join(chunk.text for chunk in chunks) == block.text
    assert all(len(chunk.text) <= 3200 for chunk in chunks)


def test_long_table_is_batched_without_dropping_rows() -> None:
    rows = [{"row_id": index} for index in range(205)]
    table = TableBlock(
        source_file="paper.pdf",
        source_path="paper.pdf",
        source_type=SourceType.PDF_TABLE,
        columns=["row_id"],
        rows=rows,
        table_id="table-1",
    )
    chunks = _expand_table_rows_for_extraction([table])
    assert [row for chunk in chunks for row in chunk.rows] == rows
    assert [chunk.raw["extraction_row_range"] for chunk in chunks] == [
        {"start": 0, "end": 80, "total": 205},
        {"start": 80, "end": 160, "total": 205},
        {"start": 160, "end": 205, "total": 205},
    ]
