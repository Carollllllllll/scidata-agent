from __future__ import annotations

"""Offline check that per-PDF figure limits survive dynamic source updates."""

import sys
import tempfile
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import scidata_agent.agent.scidata_agent as agent_module
from scidata_agent.agent.schemas import AgentState, FigureAsset, UploadedFile
from scidata_agent.agent.scidata_agent import SciDataAgent


class FakeClient:
    configured = True
    vl_model = "fake-vl"


class FakeNodes:
    @staticmethod
    def classify_chart(_asset: FigureAsset) -> dict[str, object]:
        return {"contains_data": False, "chart_type": "diagram"}


def main() -> int:
    calls: list[str] = []

    def fake_locate(uploaded, _figures_dir, max_pages=None, max_figures=None):
        del max_pages
        calls.append(uploaded.filename)
        count = 7 if max_figures in (None, 0) else min(7, max_figures)
        return [
            FigureAsset(
                source_file=uploaded.filename,
                source_path=str(uploaded.path),
                page=index + 1,
                label=f"Figure {index + 1}",
                image_path=str(uploaded.path.with_suffix(f".figure-{index + 1}.png")),
                bbox=[0.0, 0.0, 100.0, 100.0],
                detection_method="test",
            )
            for index in range(count)
        ]

    original_locator = agent_module.locate_figures
    agent_module.locate_figures = fake_locate
    try:
        with tempfile.TemporaryDirectory(prefix="scidata-figure-limit-") as temp_dir:
            output_dir = Path(temp_dir)
            paths = [output_dir / name for name in ("a.pdf", "b.pdf", "c.pdf")]
            for path in paths:
                path.write_bytes(b"pdf fixture")
            agent = SciDataAgent(
                output_dir=output_dir,
                llm_client=FakeClient(),  # type: ignore[arg-type]
                monitor_console=False,
                monitor_enabled=False,
            )
            agent.llm_nodes = FakeNodes()  # type: ignore[assignment]
            state = AgentState(
                research_question="Check incremental figure limits.",
                files=[UploadedFile(filename=path.name, path=path) for path in paths[:2]],
                output_dir=output_dir,
            )

            agent._extract_charts(state, max_figures_per_pdf=5)
            assert len(state.parsed_sources.figure_assets) == 10
            assert calls == ["a.pdf", "b.pdf"]

            state.files.append(UploadedFile(filename=paths[2].name, path=paths[2]))
            agent._extract_charts(state, max_figures_per_pdf=5)
            assert len(state.parsed_sources.figure_assets) == 15
            assert calls == ["a.pdf", "b.pdf", "c.pdf"]

            agent._extract_charts(state, max_figures_per_pdf=5)
            assert calls == ["a.pdf", "b.pdf", "c.pdf"]
            assert len(state.parsed_sources.figure_assets) == 15

            paths[2].write_bytes(b"changed pdf fixture")
            agent._extract_charts(state, max_figures_per_pdf=5)
            assert calls == ["a.pdf", "b.pdf", "c.pdf", "c.pdf"]
            assert len(state.parsed_sources.figure_assets) == 15
            assert all(
                sum(asset.source_file == filename for asset in state.parsed_sources.figure_assets) == 5
                for filename in ("a.pdf", "b.pdf", "c.pdf")
            )
    finally:
        agent_module.locate_figures = original_locator

    print("PASS: figure cap is per PDF; unchanged PDFs are not processed again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
