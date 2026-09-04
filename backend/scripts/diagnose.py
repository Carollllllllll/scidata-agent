from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Launched as ``python scripts/diagnose.py``, Python adds ``scripts`` to
# sys.path rather than the backend directory. This script only reads task
# output files, so it does not import the project package; the paths below are
# resolved relative to the repository root regardless of the caller's cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "runtime" / "outputs"


def _load_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        return rows
    return rows


def _find_task_dir(task_id: str | None, outputs_root: Path) -> Path | None:
    if task_id:
        candidate = outputs_root / task_id
        return candidate if candidate.is_dir() else None
    dirs = [p for p in outputs_root.iterdir() if p.is_dir() and p.name != "_cache"]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _terminal_runtime(monitor_rows: list[dict]) -> dict:
    for row in reversed(monitor_rows):
        if row.get("step") != "task":
            continue
        data = row.get("data") or {}
        runtime = data.get("runtime") or {}
        if runtime:
            return runtime
    return {}


def _failed_tools(tool_history: list | None) -> list[dict]:
    if not isinstance(tool_history, list):
        return []
    failures = []
    for item in tool_history:
        if not isinstance(item, dict):
            continue
        if item.get("status") == "failed":
            errors = item.get("errors") or []
            failures.append(
                {
                    "tool_name": item.get("tool_name") or "unknown",
                    "errors": [str(e) for e in errors] if isinstance(errors, list) else [str(errors)],
                    "retry_count": item.get("retry_count"),
                }
            )
    return failures


def _coverage_summary(coverage: dict | None) -> dict:
    if not isinstance(coverage, dict):
        return {}
    required_evidence = {str(e) for e in (coverage.get("required_evidence_types") or [])}
    covered_evidence = {str(e) for e in (coverage.get("covered_evidence_types") or [])}
    return {
        "decision": coverage.get("decision"),
        "coverage_score": coverage.get("coverage_score"),
        "missing_requirements_high_medium": sorted(
            str(r) for r in (coverage.get("missing_requirements") or [])
        ),
        "missing_evidence_types": sorted(required_evidence - covered_evidence),
        "unprocessed_artifacts": coverage.get("unprocessed_relevant_artifacts") or [],
        "reasons": coverage.get("reasons") or [],
        "recommended_actions": coverage.get("recommended_actions") or [],
    }


def _print_report(report: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("SciData Agent 任务终止诊断报告")
    lines.append("=" * 72)
    lines.append(f"任务ID   : {report.get('task_id')}")
    lines.append(f"研究问题 : {report.get('research_question')}")
    lines.append(f"输出目录 : {report.get('task_dir')}")
    lines.append("")

    status = str(report.get("status") or "unknown")
    runtime_status = str(report.get("runtime_status") or "unknown")
    lines.append("## 1. 结论")
    lines.append(f"- 最终状态: {status}  (runtime: {runtime_status})")
    lines.append(f"- 停止原因: {report.get('stop_reason')}")
    runtime = report.get("runtime") or {}
    if runtime:
        lines.append(
            f"- 迭代进度: {runtime.get('iteration')} / {runtime.get('iteration_budget')}"
        )
        lines.append(
            f"- 决策/工具/轨迹计数: {runtime.get('decision_count')} / "
            f"{runtime.get('tool_result_count')} / {runtime.get('trace_count')}"
        )
    lines.append("")

    cov = report.get("coverage") or {}
    if cov:
        lines.append("## 2. 覆盖率缺口（决定\"能否算完成\"）")
        lines.append(f"- 覆盖率决策: {cov.get('decision')}  （allow_stop=可完成，continue=不允许停止）")
        score = cov.get("coverage_score")
        if isinstance(score, (int, float)):
            lines.append(f"- 覆盖率分数: {score * 100:.2f}%")
        missing_fields = cov.get("missing_requirements_high_medium") or []
        missing_evidence = cov.get("missing_evidence_types") or []
        unprocessed = cov.get("unprocessed_artifacts") or []
        if missing_fields:
            lines.append(f"- 缺失的高/中优先级字段 ({len(missing_fields)} 个):")
            for name in missing_fields:
                lines.append(f"    - {name}")
        if missing_evidence:
            lines.append(f"- 缺失的证据类型: {', '.join(missing_evidence)}")
        if unprocessed:
            lines.append(f"- 未处理的高相关来源: {', '.join(str(a) for a in unprocessed)}")
        reasons = cov.get("reasons") or []
        if reasons:
            lines.append("- 官方给出的未完成理由:")
            for reason in reasons:
                lines.append(f"    - {reason}")
        actions = cov.get("recommended_actions") or []
        if actions:
            lines.append(f"- 建议的后续动作: {', '.join(str(a) for a in actions)}")
        lines.append("")

    failures = report.get("failed_tools") or []
    if failures:
        lines.append("## 3. 失败的工具/操作")
        by_tool: Counter = Counter()
        for f in failures:
            by_tool[f["tool_name"]] += 1
        lines.append("- 失败次数统计: " + ", ".join(f"{k} x{v}" for k, v in sorted(by_tool.items())))
        lines.append("- 失败原因明细:")
        seen = set()
        for f in failures:
            for err in f["errors"]:
                key = (f["tool_name"], err)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"    - [{f['tool_name']}] {err}")
        lines.append("")

    alerts = report.get("workflow_alerts") or []
    if alerts:
        lines.append("## 4. 工作流告警")
        for alert in alerts:
            lines.append(f"- {alert}")
        lines.append("")

    runtime = report.get("runtime") or {}
    if runtime and runtime.get("no_progress_streak"):
        lines.append("## 5. 无进展停机详情")
        lines.append(f"- 连续无进展: {runtime.get('no_progress_streak')} / {runtime.get('no_progress_limit')}")
        lines.append(f"- 最后一次有进展的迭代: {runtime.get('last_progress_iteration')}")
        lines.append("- 含义: 从该迭代之后，模型反复选择无效动作，被安全护栏中止")
        lines.append("")

    quality = report.get("quality") or {}
    if quality:
        lines.append("## 6. 质量指标")
        lines.append(
            f"- 证据覆盖率: {quality.get('evidence_coverage')}  "
            f"值证据覆盖率: {quality.get('value_evidence_coverage')}"
        )
        lines.append(
            f"- issues/warnings/errors: {quality.get('issues')} / "
            f"{quality.get('warnings')} / {quality.get('errors')}"
        )
        lines.append("")

    lines.append("## 7. 如何判定 completed vs partial")
    lines.append("只有当覆盖率决策为 allow_stop（所有高/中优先级字段与证据类型都被满足）")
    lines.append("且 Agent 主动请求 stop 并通过 stop-gate 时，任务才会 completed。")
    lines.append("上面第 2 节的缺口就是本次被判定为 partial 的根本原因。")
    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def build_report(task_dir: Path) -> dict:
    task_id = task_dir.name
    task_state = _load_json(task_dir / "task_state.json")
    summary = _load_json(task_dir / "summary.json")
    coverage = _load_json(task_dir / "coverage_report.json")
    tool_history = _load_json(task_dir / "tool_history.json")
    monitor_rows = _load_jsonl(task_dir / "agent_monitor.jsonl")

    research_question = ""
    if isinstance(summary, dict):
        research_question = str(summary.get("research_question") or "")
    if not research_question and isinstance(task_state, dict):
        research_question = str(task_state.get("research_question") or "")

    status = None
    if isinstance(summary, dict):
        status = summary.get("status")
    if not status and isinstance(task_state, dict):
        status = task_state.get("status")

    runtime_status = None
    stop_reason = None
    quality = None
    workflow_alerts = []
    if isinstance(summary, dict):
        runtime_status = summary.get("runtime_status")
        stop_reason = summary.get("runtime_stop_reason")
        quality = summary.get("quality") or {}
        workflow_alerts = summary.get("workflow_alerts") or []
    if not stop_reason and isinstance(task_state, dict):
        error = task_state.get("error") or {}
        if isinstance(error, dict):
            stop_reason = error.get("message") or error.get("code")

    runtime = _terminal_runtime(monitor_rows)

    return {
        "task_id": task_id,
        "task_dir": str(task_dir),
        "research_question": research_question,
        "status": status,
        "runtime_status": runtime_status,
        "stop_reason": stop_reason,
        "runtime": runtime,
        "coverage": _coverage_summary(coverage),
        "failed_tools": _failed_tools(tool_history),
        "workflow_alerts": workflow_alerts,
        "quality": quality,
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(
        description=(
            "Read a completed task output directory and explain why it ended as "
            "partial instead of completed. This script only reads files and never "
            "modifies the running program."
        )
    )
    parser.add_argument(
        "--task-id",
        help="Diagnose a specific task id under the outputs root",
    )
    parser.add_argument(
        "--task-dir",
        help="Diagnose a specific task output directory (absolute or relative)",
    )
    parser.add_argument(
        "--outputs-root",
        default=str(DEFAULT_OUTPUTS_ROOT),
        help="Root of runtime/outputs (defaults to <repo>/runtime/outputs)",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the report as a text file",
    )
    args = parser.parse_args()

    if args.task_dir:
        task_dir = Path(args.task_dir).expanduser().resolve()
    else:
        outputs_root = Path(args.outputs_root).expanduser().resolve()
        task_dir = _find_task_dir(args.task_id, outputs_root)
        if task_dir is None:
            print(f"未找到任务目录: {outputs_root}", file=sys.stderr)
            return 1
        if args.task_id and not task_dir.is_dir():
            print(f"任务不存在: {task_dir}", file=sys.stderr)
            return 1

    if not task_dir.is_dir():
        print(f"任务目录不存在: {task_dir}", file=sys.stderr)
        return 1

    report = build_report(task_dir)
    text = _print_report(report)

    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"\n[diagnose] 报告已写入: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
