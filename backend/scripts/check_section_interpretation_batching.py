from __future__ import annotations

"""Offline regression test for incremental, truncation-safe section parsing.

Run from the repository root:

    D:\software\anaconda3\envs\scidata-agent\python.exe backend\scripts\check_section_interpretation_batching.py

The script uses a fake LLM and performs no network requests.  Exit code 0
means that oversized batches were split and already interpreted sources were
not sent to the model again.
"""

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scidata_agent.agent.schemas import AgentState, HeadingCandidate, TextBlock
from scidata_agent.agent.scidata_agent import SciDataAgent
from scidata_agent.llm.client import LLMCallError
from scidata_agent.llm.nodes import QwenAgentNodes


class TruncatingSectionClient:
    """Simulate a provider that truncates responses above a tiny threshold."""

    configured = True

    def __init__(self, max_candidates: int = 2) -> None:
        self.max_candidates = max_candidates
        self.calls: list[dict[str, object]] = []
        self.traces: list[object] = []
        self.model_events: list[dict[str, object]] = []

    def generate_json(
        self,
        node: str,
        _system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> dict[str, object]:
        del temperature
        assert node == "qwen_section_interpreter"
        marker = "Heading candidates extracted from PDF text/layout:\n"
        candidates_json = user_prompt.split(marker, 1)[1].split("\n\nReturn JSON:", 1)[0]
        candidates = json.loads(candidates_json)
        sources = sorted({str(item["source_file"]) for item in candidates})
        self.calls.append({"size": len(candidates), "sources": sources})
        if len(candidates) > self.max_candidates:
            raise LLMCallError("Qwen output truncated at max_tokens=8192.")
        return {
            "sections": [
                {
                    "source_file": item["source_file"],
                    "section_title": item["text"],
                    "section_type": "results" if "result" in item["text"].casefold() else "method",
                    "start_page": item["page"],
                    "start_anchor": item["text"],
                    "confidence": 0.9,
                    "reason": "Synthetic section used by the offline regression test.",
                }
                for item in candidates
            ],
            "ignored_candidates": [],
            "warnings": [],
        }


def _source_fixture(source_name: str, count: int = 4) -> tuple[list[TextBlock], list[HeadingCandidate]]:
    source_path = str(Path("fixtures") / source_name)
    blocks: list[TextBlock] = []
    headings: list[HeadingCandidate] = []
    for index in range(1, count + 1):
        title = f"{index}. {'Results' if index == count else 'Method'} {index}"
        blocks.append(
            TextBlock(
                source_file=source_name,
                source_path=source_path,
                page=index,
                text=f"{title}\nScientific content for {source_name}, page {index}.",
                chunk_id=f"{source_name}-page-{index}",
            )
        )
        headings.append(
            HeadingCandidate(
                source_file=source_name,
                source_path=source_path,
                page=index,
                line_index=0,
                text=title,
                score=9.0,
            )
        )
    return blocks, headings


def _assert_adaptive_split() -> dict[str, object]:
    client = TruncatingSectionClient(max_candidates=2)
    nodes = QwenAgentNodes(client)  # type: ignore[arg-type]
    _, headings = _source_fixture("large.pdf", count=8)

    plan = nodes.interpret_sections("Inspect a large paper.", headings)

    call_sizes = [int(call["size"]) for call in client.calls]
    assert len(plan.sections) == 8, f"Expected 8 merged sections, got {len(plan.sections)}"
    assert call_sizes == [8, 4, 2, 2, 4, 2, 2], (
        "Truncated batches must be bisected immediately instead of retried unchanged: "
        f"{call_sizes}"
    )
    assert any("split" in warning.casefold() for warning in nodes.node_warnings), (
        "The truncation split was not recorded in node warnings."
    )
    return {"call_sizes": call_sizes, "merged_sections": len(plan.sections)}


def _assert_incremental_sources(output_dir: Path) -> dict[str, object]:
    client = TruncatingSectionClient(max_candidates=2)
    nodes = QwenAgentNodes(client)  # type: ignore[arg-type]
    agent = SciDataAgent(
        output_dir=output_dir,
        llm_client=client,  # type: ignore[arg-type]
        require_llm=True,
        monitor_console=False,
        monitor_enabled=False,
    )
    agent.llm_nodes = nodes
    state = AgentState(
        research_question="Interpret all paper sections incrementally.",
        files=[],
        output_dir=output_dir,
    )

    for source_name in ("paper_a.pdf", "paper_b.pdf"):
        blocks, headings = _source_fixture(source_name)
        state.parsed_sources.text_blocks.extend(blocks)
        state.parsed_sources.heading_candidates.extend(headings)
    agent._interpret_sections(state)

    before_counts = Counter(block.source_file for block in state.parsed_sources.section_blocks)
    calls_before_new_source = len(client.calls)
    blocks, headings = _source_fixture("paper_c.pdf")
    state.parsed_sources.text_blocks.extend(blocks)
    state.parsed_sources.heading_candidates.extend(headings)
    agent._interpret_sections(state)

    after_counts = Counter(block.source_file for block in state.parsed_sources.section_blocks)
    new_calls = client.calls[calls_before_new_source:]
    assert before_counts["paper_a.pdf"] == after_counts["paper_a.pdf"]
    assert before_counts["paper_b.pdf"] == after_counts["paper_b.pdf"]
    assert after_counts["paper_c.pdf"] > 0
    assert new_calls and all(call["sources"] == ["paper_c.pdf"] for call in new_calls), new_calls

    calls_before_noop = len(client.calls)
    blocks_before_noop = len(state.parsed_sources.section_blocks)
    agent._interpret_sections(state)
    assert len(client.calls) == calls_before_noop, "A no-op rerun called the LLM again."
    assert len(state.parsed_sources.section_blocks) == blocks_before_noop

    changed_blocks, changed_headings = _source_fixture("paper_c.pdf", count=5)
    state.parsed_sources.text_blocks = [
        block
        for block in state.parsed_sources.text_blocks
        if block.source_file != "paper_c.pdf"
    ] + changed_blocks
    state.parsed_sources.heading_candidates = [
        heading
        for heading in state.parsed_sources.heading_candidates
        if heading.source_file != "paper_c.pdf"
    ] + changed_headings
    calls_before_changed_source = len(client.calls)
    agent._interpret_sections(state)
    changed_calls = client.calls[calls_before_changed_source:]
    changed_counts = Counter(block.source_file for block in state.parsed_sources.section_blocks)
    assert changed_counts["paper_a.pdf"] == after_counts["paper_a.pdf"]
    assert changed_counts["paper_b.pdf"] == after_counts["paper_b.pdf"]
    assert changed_counts["paper_c.pdf"] == 5
    assert changed_calls and all(
        call["sources"] == ["paper_c.pdf"] for call in changed_calls
    ), changed_calls

    return {
        "source_block_counts": dict(changed_counts),
        "new_source_call_sizes": [int(call["size"]) for call in new_calls],
        "changed_source_call_sizes": [int(call["size"]) for call in changed_calls],
        "total_llm_calls": len(client.calls),
    }


def main() -> int:
    previous_batch_size = os.environ.get("SCIDATA_SECTION_INTERPRETER_BATCH_SIZE")
    os.environ["SCIDATA_SECTION_INTERPRETER_BATCH_SIZE"] = "8"
    try:
        adaptive = _assert_adaptive_split()
        with tempfile.TemporaryDirectory(prefix="scidata-section-batch-") as temp_dir:
            incremental = _assert_incremental_sources(Path(temp_dir))
    finally:
        if previous_batch_size is None:
            os.environ.pop("SCIDATA_SECTION_INTERPRETER_BATCH_SIZE", None)
        else:
            os.environ["SCIDATA_SECTION_INTERPRETER_BATCH_SIZE"] = previous_batch_size

    print(
        json.dumps(
            {
                "status": "passed",
                "adaptive_split": adaptive,
                "incremental_sources": incremental,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
