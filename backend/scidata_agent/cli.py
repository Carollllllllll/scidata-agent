from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scidata_agent.agent.scidata_agent import SciDataAgent
from scidata_agent.config import load_dotenv


def main() -> None:
    _configure_stdio_utf8()
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the Qwen-powered SciData Agent.")
    parser.add_argument("--question", help="Research question or data extraction request.")
    parser.add_argument(
        "--question-file",
        help=(
            "UTF-8 text file containing the research question. "
            "Recommended on Windows/PowerShell when Chinese command-line text becomes mojibake."
        ),
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Optional input PDF/CSV/Excel files. If omitted, the agent runs source discovery only.",
    )
    parser.add_argument("--output-dir", default="../outputs", help="Directory for exported results.")
    parser.add_argument("--max-pdf-pages", type=int, default=0, help="Maximum pages per PDF; 0 means all pages.")
    parser.add_argument(
        "--max-arxiv-papers",
        type=int,
        default=None,
        help="Deprecated compatibility option. Omit or use 0 to process every selected resource.",
    )
    parser.add_argument(
        "--max-pdf-downloads",
        type=int,
        default=None,
        help="Optional explicit PDF download cap across providers; omit or use 0 for all selected PDFs.",
    )
    parser.add_argument(
        "--max-dynamic-text-blocks",
        type=int,
        default=0,
        help="Maximum ranked text blocks for dynamic extraction. Use 0 for no limit.",
    )
    parser.add_argument(
        "--max-record-text-blocks",
        type=int,
        default=0,
        help="Maximum ranked text blocks for metric record extraction. Use 0 for no limit.",
    )
    parser.add_argument(
        "--no-arxiv-download",
        action="store_true",
        help="Disable automatic arXiv PDF download. Source discovery results are still exported.",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only plan target schema and discover candidate sources; ignore local files if provided.",
    )
    parser.add_argument(
        "--max-figures-per-pdf",
        type=int,
        default=0,
        help="Maximum figures per PDF sent to Qwen-VL; 0 means all detected figures.",
    )
    parser.add_argument(
        "--max-artifact-action-iterations",
        type=int,
        default=1,
        help="Maximum LLM artifact-planning iterations. Default 1; explicitly set a larger value when needed.",
    )
    runtime_group = parser.add_mutually_exclusive_group()
    runtime_group.add_argument(
        "--dynamic-runtime",
        dest="dynamic_runtime",
        action="store_true",
        help="Enable the Observation-Decision-Action runtime (the default).",
    )
    runtime_group.add_argument(
        "--legacy-runtime",
        dest="dynamic_runtime",
        action="store_false",
        help="Use the compatibility pipeline instead of the dynamic Agent runtime.",
    )
    parser.set_defaults(dynamic_runtime=True)
    parser.add_argument(
        "--max-agent-iterations",
        type=int,
        default=None,
        help="Safety budget for dynamic Agent turns; this does not cap sources or downloaded resources.",
    )
    parser.add_argument(
        "--task-id",
        help="Reuse a specific task ID when resuming from its checkpoint.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load the checkpoint for --task-id and continue the previous task.",
    )
    parser.add_argument(
        "--arxiv-pdf-timeout",
        type=int,
        default=600,
        help="Maximum total seconds for one arXiv PDF attempt. Default 600 seconds.",
    )
    parser.add_argument(
        "--arxiv-download-batch-timeout",
        type=int,
        default=3600,
        help="Maximum total seconds for the arXiv PDF download stage. Default 3600 seconds.",
    )
    parser.add_argument(
        "--allow-rule-fallback",
        action="store_true",
        help="Local testing only. Allows rule fallback when Qwen is not configured. Do not use for official results.",
    )
    args = parser.parse_args()
    question = _load_question(args.question, args.question_file)

    agent = SciDataAgent(
        output_dir=Path(args.output_dir),
        require_llm=not args.allow_rule_fallback,
        allow_rule_fallback=args.allow_rule_fallback,
    )
    files = [] if args.discover_only else args.files
    result = agent.run(
        question,
        files,
        max_pdf_pages=args.max_pdf_pages,
        auto_fetch_arxiv=not args.no_arxiv_download,
        discovery_only=args.discover_only,
        max_arxiv_papers=args.max_pdf_downloads if args.max_pdf_downloads is not None else args.max_arxiv_papers,
        max_dynamic_text_blocks=args.max_dynamic_text_blocks,
        max_record_text_blocks=args.max_record_text_blocks,
        max_figures_per_pdf=args.max_figures_per_pdf,
        max_artifact_action_iterations=args.max_artifact_action_iterations,
        arxiv_pdf_timeout=args.arxiv_pdf_timeout,
        arxiv_download_batch_timeout=args.arxiv_download_batch_timeout,
        enable_dynamic_runtime=args.dynamic_runtime,
        max_agent_iterations=args.max_agent_iterations,
        task_id=args.task_id,
        resume=args.resume,
    )
    print(json.dumps(result.model_dump(mode="json", by_alias=True), ensure_ascii=False, indent=2))
    if getattr(result, "status", None) == "failed":
        raise SystemExit(1)


def _configure_stdio_utf8() -> None:
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _load_question(question: str | None, question_file: str | None) -> str:
    if question and question_file:
        raise SystemExit("Use either --question or --question-file, not both.")
    if question_file:
        path = Path(question_file).expanduser().resolve()
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise SystemExit(f"--question-file must be UTF-8 encoded: {path}") from exc
        question = " ".join(text.split())
    if not question or not question.strip():
        raise SystemExit("A research question is required. Use --question or --question-file.")
    return question.strip()


if __name__ == "__main__":
    main()
