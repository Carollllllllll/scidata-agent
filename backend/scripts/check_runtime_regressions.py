from __future__ import annotations

"""Fast, offline regression checks for dynamic-runtime stalls found in tasks
20260902_204208_491_5961 and 20260902_212225_343_d6b7.

This script intentionally avoids network requests, LLM calls, and full task
execution. Run it after changing the runtime or connectors:

    python backend/scripts/check_runtime_regressions.py

Exit code 0 means all targeted regressions are fixed; exit code 1 means at
least one check still fails.
"""

import argparse
import ast
import inspect
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _activate_project_python() -> None:
    """Re-exec with a usable project interpreter when one is available."""

    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    conda_relative = Path("python.exe") if os.name == "nt" else Path("bin/python")
    candidates = [
        BACKEND_ROOT / ".venv" / relative,
        Path(sys.prefix) / "envs" / "scidata-agent" / conda_relative,
    ]
    if os.environ.get("SCIDATA_REGRESSION_REEXEC") == "1":
        return
    for project_python in candidates:
        if not project_python.is_file():
            continue
        try:
            if Path(sys.executable).resolve() == project_python.resolve():
                return
            # A copied Windows venv can contain python.exe while pyvenv.cfg
            # still points at Python from the old machine. Probe both startup
            # and the core runtime dependencies before selecting it.
            probe = subprocess.run(
                [
                    str(project_python),
                    "-c",
                    "import pandas, pydantic, pymupdf; print('ready')",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            continue
        if probe.returncode != 0:
            continue
        child_env = os.environ.copy()
        child_env["SCIDATA_REGRESSION_REEXEC"] = "1"
        completed = subprocess.run(
            [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            env=child_env,
            check=False,
        )
        raise SystemExit(completed.returncode)


_activate_project_python()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    elapsed_ms: float


def _run_check(name: str, check: Callable[[], str]) -> CheckResult:
    started = time.perf_counter()
    try:
        detail = check()
    except AssertionError as exc:
        return CheckResult(name, False, str(exc), (time.perf_counter() - started) * 1000)
    except Exception as exc:  # Keep the smoke check useful during refactors.
        return CheckResult(
            name,
            False,
            f"检查本身异常: {type(exc).__name__}: {exc}",
            (time.perf_counter() - started) * 1000,
        )
    return CheckResult(name, True, detail, (time.perf_counter() - started) * 1000)


def _artifact_state(tmp_path: Path, *, artifact_type: str, status: str, metadata: dict | None = None):
    from scidata_agent.agent.schemas import AgentState, SourceArtifact, SourceCatalogEntry

    artifact = SourceArtifact(
        artifact_id="artifact_smoke_test",
        source_id="source_smoke_test",
        artifact_type=artifact_type,
        status=status,
        relevance_score=4.0,
        metadata=metadata or {},
    )
    state = AgentState(
        research_question="Offline runtime regression check.",
        files=[],
        output_dir=tmp_path / "outputs",
        source_catalog=[
            SourceCatalogEntry(
                source_id="source_smoke_test",
                title="Smoke-test source",
                source_type="dataset",
                relevance_score=1.0,
                artifacts=[artifact],
            )
        ],
    )
    return state, artifact


def check_manifest_reads_local_content() -> str:
    """A local manifest action must inspect the file, not echo model metadata."""

    from scidata_agent.agent.action_executor import ArtifactActionExecutor
    from scidata_agent.agent.schemas import ArtifactAction

    sentinel = "SCIDATA_LOCAL_MANIFEST_SENTINEL"
    state, artifact = _artifact_state(
        BACKEND_ROOT,
        artifact_type="file_manifest",
        status="downloaded",
        metadata={"note": "metadata does not contain the file sentinel"},
    )
    # Reuse this script as a read-only fixture so the check leaves no temporary
    # files behind and works in restricted Windows environments.
    artifact.local_path = str(Path(__file__).resolve())
    result = ArtifactActionExecutor().execute_action(
        ArtifactAction(
            action_id="check_manifest",
            artifact_id=artifact.artifact_id,
            action="read_file_manifest",
            purpose="Read the real local manifest.",
            reason="Regression check.",
        ),
        state,
    )
    observed = "\n".join(item.content for item in state.source_insights)
    observed += "\n" + "\n".join(item.text for item in state.parsed_sources.text_blocks)

    assert result.status == "completed", f"read_file_manifest 未完成: {result.status} / {result.message}"
    assert sentinel in observed, (
        "read_file_manifest 仍未读取本地文件内容；当前实现可能只是在序列化 artifact.metadata。"
    )
    return "read_file_manifest 已读取真实本地内容"


def check_metadata_action_keeps_content_work_pending_without_repeating_it() -> str:
    """Metadata-only work must advance to content without repeating itself."""

    from scidata_agent.agent.action_executor import ArtifactActionExecutor
    from scidata_agent.agent.schemas import ArtifactAction
    from scidata_agent.tools.coverage import _unprocessed_relevant_artifacts

    state, artifact = _artifact_state(
        BACKEND_ROOT,
        artifact_type="landing_page",
        status="downloaded",
        metadata={"source_url": "https://example.org/catalog"},
    )
    result = ArtifactActionExecutor().execute_action(
        ArtifactAction(
            action_id="check_metadata",
            artifact_id=artifact.artifact_id,
            action="read_metadata",
            purpose="Inspect metadata once.",
            reason="Regression check.",
        ),
        state,
    )
    pending = _unprocessed_relevant_artifacts(state)

    assert result.status == "completed", f"read_metadata 未完成: {result.status} / {result.message}"
    assert artifact.artifact_id in pending, (
        "高相关 artifact 只读取 metadata 后被错误视为完成，后续不会再下载和解析正文。"
    )
    assert "read_metadata" in artifact.completed_operations, (
        "read_metadata 未写入独立操作账本，Planner 可能重复执行相同 metadata 动作。"
    )
    return "metadata 动作不重复，但高相关 artifact 会继续进入正文下载/解析"


def check_catalog_refresh_preserves_inspection_status() -> str:
    """Refreshing the derived catalog must not undo a completed inspection."""

    from scidata_agent.agent.schemas import (
        AgentState,
        DiscoveredSource,
        SourceArtifact,
        SourceCatalogEntry,
        SourceDiscoveryPlan,
        SourceInsight,
    )
    from scidata_agent.tools.source_catalog import refresh_source_catalog

    source = DiscoveredSource(
        source_id="source_inspected_landing_page",
        title="Inspected landing page",
        source_type="webpage",
        url="https://example.org/inspected",
    )
    state = AgentState(
        research_question="Offline catalog-status regression check.",
        files=[],
        output_dir=BACKEND_ROOT / "outputs",
        source_discovery_plan=SourceDiscoveryPlan(
            research_goal="Keep terminal artifact states across catalog refreshes.",
            candidate_sources=[source],
        ),
        source_insights=[
            SourceInsight(
                source_id=source.source_id,
                title=source.title,
                source_type=source.source_type,
                insight_type="metadata",
                content="Metadata was inspected successfully.",
                url=source.url,
            )
        ],
        source_catalog=[
            SourceCatalogEntry(
                source_id=source.source_id,
                title=source.title,
                source_type=source.source_type,
                url=source.url,
                status="inspected",
                artifacts=[
                    SourceArtifact(
                        artifact_id="prior_inspected_landing_page",
                        source_id=source.source_id,
                        artifact_type="landing_page",
                        url=source.url,
                        status="inspected",
                    )
                ],
            )
        ],
    )

    refresh_source_catalog(state)
    refreshed = next(
        artifact
        for entry in state.source_catalog
        for artifact in entry.artifacts
        if artifact.artifact_type == "landing_page" and artifact.url == source.url
    )
    assert refreshed.status == "inspected", (
        "refresh_source_catalog 把已 inspected 的落地页重置为 "
        f"{refreshed.status!r}；这会让同一 artifact 被重复规划。"
    )
    return "catalog 刷新会保留 inspected 终态"


def check_catalog_manifest_file_keeps_type_and_download_state() -> str:
    """A provider manifest key must route /content URLs and retain local state."""

    from scidata_agent.agent.schemas import AgentState, DiscoveredSource, SourceDiscoveryPlan
    from scidata_agent.tools.source_catalog import refresh_source_catalog

    with tempfile.TemporaryDirectory(prefix="scidata-regression-") as directory:
        fixture = Path(directory) / "yse_table.csv"
        fixture.write_text("sn_identifier,band_filter\nSN2020abc,g\n", encoding="utf-8")
        url = "https://example.org/records/1/files/yse_table.csv/content"
        source = DiscoveredSource(
            source_id="source_manifest_csv",
            title="Manifest CSV fixture",
            source_type="dataset",
            metadata={
                "files": [
                    {
                        "key": "yse_table.csv",
                        "url": url,
                        "size": fixture.stat().st_size,
                    }
                ],
                "downloaded_artifacts": {url: str(fixture)},
                "downloaded_path": str(fixture),
            },
        )
        state = AgentState(
            research_question="Offline manifest-catalog regression check.",
            files=[],
            output_dir=Path(directory),
            source_discovery_plan=SourceDiscoveryPlan(
                research_goal="Keep a materialized CSV routable.",
                candidate_sources=[source],
            ),
        )
        refresh_source_catalog(state)
        artifact = next(
            artifact
            for entry in state.source_catalog
            for artifact in entry.artifacts
            if artifact.url == url
        )
        first_id = artifact.artifact_id
        refresh_source_catalog(state)
        refreshed = next(
            artifact
            for entry in state.source_catalog
            for artifact in entry.artifacts
            if artifact.url == url
        )

    assert artifact.name == "yse_table.csv" and artifact.artifact_type == "csv", (
        "数据集文件仍按 /content 或 dataset 回退为 file_manifest；"
        "应优先使用 manifest 的 key/filename 识别 CSV。"
    )
    assert artifact.local_path == str(fixture) and artifact.status == "downloaded", (
        "已下载的 manifest 文件没有绑定回其具体 URL artifact。"
    )
    assert refreshed.artifact_id == first_id and refreshed.local_path == str(fixture), (
        "catalog 刷新改变了远程 artifact 身份或丢失 local_path，"
        "会导致 planner 重复下载、而 policy 以已完成调用拒绝。"
    )
    return "manifest CSV 类型、下载状态与 artifact ID 会跨 catalog 刷新保留"


def _workflow_ready_state(*, artifact: object | None = None):
    """Create the smallest state that has completed the planning contract."""

    from scidata_agent.agent.schemas import (
        AgentState,
        DynamicExtractionPlan,
        SourceCatalogEntry,
        TaskPlan,
    )

    catalog = []
    if artifact is not None:
        catalog.append(
            SourceCatalogEntry(
                source_id="source_policy_check",
                title="Policy-check source",
                source_type="webpage",
                artifacts=[artifact],
            )
        )
    return AgentState(
        research_question="Offline policy regression check.",
        files=[],
        output_dir=BACKEND_ROOT / "outputs",
        task_plan=TaskPlan(research_goal="Check runtime prerequisites."),
        dynamic_extraction_plan=DynamicExtractionPlan(
            research_goal="Check runtime prerequisites."
        ),
        source_catalog=catalog,
        runtime_requires_source_discovery=True,
        tool_result_history=[{"tool_name": "search_sources", "status": "completed"}],
    )


def check_triage_requires_source_selection() -> str:
    """Official dynamic runs must reject triage until selection has succeeded."""

    from scidata_agent.agent.decision import AgentDecision
    from scidata_agent.agent.policy import AgentPolicy
    from scidata_agent.agent.tool_protocol import ToolCall
    from scidata_agent.agent.tool_registry import build_artifact_tool_registry

    result = AgentPolicy(build_artifact_tool_registry()).validate(
        AgentDecision(
            decision="continue",
            tool_calls=[ToolCall(call_id="triage-without-selection", tool_name="triage_sources")],
        ),
        _workflow_ready_state(),
    )
    assert result.allowed is False and any(
        "source selection" in violation.casefold() for violation in result.violations
    ), (
        "triage_sources 在没有 Source Selection Plan 时仍会通过 policy；"
        "官方模式会在执行阶段报错并浪费一个迭代。"
    )
    return "triage_sources 会在选择计划缺失时被 policy 拦截"


def check_parser_requires_materialized_local_artifact() -> str:
    """A parser must be rejected before it can produce a skipped no-op."""

    from scidata_agent.agent.decision import AgentDecision
    from scidata_agent.agent.policy import AgentPolicy
    from scidata_agent.agent.schemas import SourceArtifact
    from scidata_agent.agent.tool_protocol import ToolCall
    from scidata_agent.agent.tool_registry import build_artifact_tool_registry

    artifact = SourceArtifact(
        artifact_id="landing_page_without_file",
        source_id="source_policy_check",
        artifact_type="landing_page",
        url="https://example.org/no-local-file",
        status="metadata_read",
    )
    state = _workflow_ready_state(artifact=artifact)
    result = AgentPolicy(build_artifact_tool_registry()).validate(
        AgentDecision(
            decision="continue",
            tool_calls=[
                ToolCall(
                    call_id="parse-without-local-file",
                    tool_name="parse_html",
                    arguments={"artifact_id": artifact.artifact_id},
                )
            ],
        ),
        state,
    )
    assert result.allowed is False and any(
        "local_path" in violation for violation in result.violations
    ), (
        "parse_html 在 artifact 没有 local_path 时仍会通过 policy；"
        "随后只会得到 skipped，造成无进展循环。"
    )
    return "无 local_path 的解析动作会在执行前被 policy 拦截"


def check_policy_allows_selected_landing_page_download() -> str:
    """A selected content-bearing landing page must have a materialization path."""

    from scidata_agent.agent.decision import AgentDecision
    from scidata_agent.agent.policy import AgentPolicy
    from scidata_agent.agent.schemas import SourceArtifact
    from scidata_agent.agent.tool_protocol import ToolCall
    from scidata_agent.agent.tool_registry import build_artifact_tool_registry

    artifact = SourceArtifact(
        artifact_id="landing_page_download_check",
        source_id="source_policy_check",
        artifact_type="landing_page",
        url="https://example.org/repository",
        status="metadata_read",
    )
    result = AgentPolicy(build_artifact_tool_registry()).validate(
        AgentDecision(
            decision="continue",
            tool_calls=[
                ToolCall(
                    call_id="download-landing-page",
                    tool_name="download_artifact",
                    arguments={"artifact_id": artifact.artifact_id},
                )
            ],
        ),
        _workflow_ready_state(artifact=artifact),
    )
    assert result.allowed is True, (
        "download_artifact(landing_page) 没有可用的受控下载路径；"
        "运行时只能重复 read_metadata 或无本地文件的 parse_html。"
    )
    return "已选择的 landing_page 可通过受控下载路径物化"


def check_download_registers_materialized_file() -> str:
    """Direct artifact downloads must enter state.files for the parser to see them."""

    from scidata_agent.agent.action_executor import ArtifactActionExecutor
    from scidata_agent.agent.schemas import ArtifactAction
    from scidata_agent.tools import source_ingestion

    with tempfile.TemporaryDirectory(prefix="scidata-regression-") as directory:
        fixture = Path(directory) / "landing-page.html"
        fixture.write_text("<html><body>local evidence</body></html>", encoding="utf-8")
        state, artifact = _artifact_state(
            Path(directory),
            artifact_type="landing_page",
            status="metadata_read",
            metadata={"source_url": "https://example.org/landing"},
        )
        state.output_dir = Path(directory)
        artifact.url = "https://example.org/landing"
        original = source_ingestion.download_source_artifact

        def fake_download(current_artifact, _target_dir, *, max_bytes=None, downloader=None):
            current_artifact.local_path = str(fixture)
            current_artifact.name = fixture.name
            current_artifact.content_type = "text/html"
            current_artifact.artifact_type = "html"
            current_artifact.size_bytes = fixture.stat().st_size
            current_artifact.status = "downloaded"
            return fixture

        source_ingestion.download_source_artifact = fake_download
        try:
            result = ArtifactActionExecutor().execute_action(
                ArtifactAction(
                    action_id="check-direct-download",
                    artifact_id=artifact.artifact_id,
                    action="download_artifact",
                    purpose="Materialize one local test artifact.",
                    reason="Regression check.",
                ),
                state,
            )
        finally:
            source_ingestion.download_source_artifact = original

    assert result.status == "completed", f"直接下载未完成: {result.status} / {result.message}"
    assert result.output_counts.get("files_delta") == 1 and len(state.files) == 1, (
        "download_artifact 成功后没有将新文件登记到 state.files；"
        "下一轮 parse_source_content 看不到该文件。"
    )
    assert artifact.artifact_type == "html", "下载后没有按本地内容将 landing_page 规范为 html。"
    return "直接下载会登记本地文件并按内容更新 artifact 类型"


def check_unsupported_download_is_terminally_skipped() -> str:
    """Downloaded archives must not enter parser, coverage, or planner retry loops."""

    from scidata_agent.agent.action_executor import ArtifactActionExecutor
    from scidata_agent.agent.schemas import (
        ArtifactAction,
        DiscoveredSource,
        SourceDiscoveryPlan,
    )
    from scidata_agent.llm.nodes import _artifact_planner_catalog_payload
    from scidata_agent.tools import source_ingestion
    from scidata_agent.tools.coverage import _unprocessed_relevant_artifacts
    from scidata_agent.tools.source_catalog import refresh_source_catalog

    with tempfile.TemporaryDirectory(prefix="scidata-regression-") as directory:
        fixture = Path(directory) / "photometry_bundle.tgz"
        fixture.write_bytes(b"not expanded during this offline check")
        spreadsheet = Path(directory) / "supported_legacy_table.xls"
        spreadsheet.write_bytes(b"legacy spreadsheet fixture")
        image = Path(directory) / "supported_chart.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        state, artifact = _artifact_state(
            Path(directory),
            artifact_type="unknown",
            status="planned",
        )
        state.output_dir = Path(directory)
        artifact.url = "https://example.org/photometry_bundle.tgz"
        original = source_ingestion.download_source_artifact

        def fake_download(current_artifact, _target_dir, *, max_bytes=None, downloader=None):
            current_artifact.local_path = str(fixture)
            current_artifact.name = fixture.name
            current_artifact.content_type = "application/gzip"
            current_artifact.artifact_type = "unknown"
            current_artifact.size_bytes = fixture.stat().st_size
            current_artifact.status = "downloaded"
            return fixture

        source_ingestion.download_source_artifact = fake_download
        try:
            result = ArtifactActionExecutor().execute_action(
                ArtifactAction(
                    action_id="check-unsupported-archive",
                    artifact_id=artifact.artifact_id,
                    action="download_artifact",
                    purpose="Classify a downloaded archive.",
                    reason="Regression check.",
                ),
                state,
            )
        finally:
            source_ingestion.download_source_artifact = original

        state.source_discovery_plan = SourceDiscoveryPlan(
            research_goal="Keep skipped archive terminal across catalog refreshes.",
            candidate_sources=[
                DiscoveredSource(
                    source_id=artifact.source_id,
                    title="Archive fixture",
                    source_type="dataset",
                    metadata={
                        "files": [{"key": fixture.name, "url": artifact.url}],
                        "downloaded_artifacts": {artifact.url: str(fixture)},
                    },
                )
            ],
        )
        refresh_source_catalog(state)
        refreshed = next(
            item
            for entry in state.source_catalog
            for item in entry.artifacts
            if item.url == artifact.url
        )

    payload, total = _artifact_planner_catalog_payload(state.source_catalog)
    assert result.status == "skipped" and artifact.status == "skipped", (
        "不支持的压缩包下载后没有进入 skipped 终态。"
    )
    assert artifact.parser == "unsupported_format" and "archive" in (artifact.failure_reason or ""), (
        "不支持格式没有保留可审计的原因。"
    )
    assert refreshed.status == "skipped" and refreshed.parser == "unsupported_format", (
        "catalog 刷新后丢失了不支持格式的 skipped 终态。"
    )
    assert not state.files and artifact.artifact_id not in _unprocessed_relevant_artifacts(state), (
        "不支持的压缩包仍被加入解析列表或 coverage 待处理列表。"
    )
    assert total == 0 and not payload, "Planner 输入仍包含已跳过的不支持格式。"
    assert source_ingestion._detected_artifact_type(spreadsheet, None) == "xlsx", (
        "已支持的 .xls 被误判为不支持格式。"
    )
    assert source_ingestion.unsupported_materialized_format_reason(spreadsheet, "xlsx") is None, (
        "已支持的 .xls 不应被跳过。"
    )
    assert source_ingestion._detected_artifact_type(image, None) == "image", (
        "已支持的图片被误判为不支持格式。"
    )
    assert source_ingestion.unsupported_materialized_format_reason(image, "image") is None, (
        "已支持的图片不应被跳过。"
    )
    return "不支持压缩包会终态跳过，不进入解析、coverage 或 Planner 重试"


def check_ingestion_routes_archives_and_dat_files() -> str:
    """The ingestion path must share terminal skip and .dat table routing."""

    from scidata_agent.agent.schemas import (
        AgentState,
        DiscoveredSource,
        SourceDiscoveryPlan,
        SourceTriageDecision,
    )
    from scidata_agent.llm.nodes import _artifact_planner_catalog_payload
    from scidata_agent.tools.parser import parse_sources
    from scidata_agent.tools.source_catalog import refresh_source_catalog
    from scidata_agent.tools.source_ingestion import ingest_triaged_sources

    archive_url = "https://example.org/CSP_Photometry_DR3.tgz"
    data_url = "https://example.org/Pantheon_SH0ES.dat"
    archive_source = DiscoveredSource(
        source_id="archive-ingestion-check",
        title="Archive fixture",
        source_type="supplementary_material",
        url=archive_url,
        metadata={"provider": "fixture"},
    )
    data_source = DiscoveredSource(
        source_id="dat-ingestion-check",
        title="Whitespace table fixture",
        source_type="dataset",
        metadata={
            "provider": "fixture",
            "files": [{"key": "Pantheon_SH0ES.dat", "url": data_url, "size": 128}],
        },
    )
    decisions = [
        SourceTriageDecision(
            source_id=archive_source.source_id,
            title=archive_source.title,
            provider="fixture",
            source_type=archive_source.source_type,
            relevance_score=0.9,
            recommended_action="download_pdf",
            should_ingest=True,
        ),
        SourceTriageDecision(
            source_id=data_source.source_id,
            title=data_source.title,
            provider="fixture",
            source_type=data_source.source_type,
            relevance_score=0.9,
            recommended_action="download_small_table",
            should_ingest=True,
        ),
    ]

    with tempfile.TemporaryDirectory(prefix="scidata-regression-") as directory:
        root = Path(directory)

        def fake_downloader(url: str, target_path: Path, _max_bytes: int) -> None:
            if url == archive_url:
                target_path.write_bytes(b"archive fixture")
            elif url == data_url:
                target_path.write_text(
                    "# Pantheon fixture\nSNID mB x1\nSN_001 14.25 0.12\n",
                    encoding="utf-8",
                )
            else:
                raise AssertionError(f"Unexpected fixture URL: {url}")

        files, _blocks, _insights, logs = ingest_triaged_sources(
            [archive_source, data_source],
            decisions,
            output_dir=root,
            task_id="ingestion-format-check",
            downloader=fake_downloader,
        )
        parsed = parse_sources(files)
        state = AgentState(
            research_question="Offline ingestion format regression check.",
            files=files,
            output_dir=root,
            source_discovery_plan=SourceDiscoveryPlan(
                research_goal="Keep archive skips terminal and route .dat tables.",
                candidate_sources=[archive_source, data_source],
            ),
            source_triage_decisions=decisions,
        )
        refresh_source_catalog(state)
        archive_artifact = next(
            artifact
            for entry in state.source_catalog
            for artifact in entry.artifacts
            if artifact.source_id == archive_source.source_id and artifact.url == archive_url
        )
        dat_artifact = next(
            artifact
            for entry in state.source_catalog
            for artifact in entry.artifacts
            if artifact.source_id == data_source.source_id and artifact.url == data_url
        )

    planner_payload, planner_total = _artifact_planner_catalog_payload(state.source_catalog)
    assert not any(file.path.suffix.lower() == ".tgz" for file in files), (
        "来源摄取仍把不支持的压缩包加入 state.files。"
    )
    assert archive_artifact.status == "skipped" and archive_artifact.parser == "unsupported_format", (
        "来源摄取后的压缩包没有保留 skipped 终态。"
    )
    assert not any(
        item["source_id"] == archive_source.source_id for item in planner_payload
    ), "Planner 仍接收到来源摄取后已跳过的压缩包。"
    assert dat_artifact.artifact_type == "csv" and dat_artifact.status == "downloaded", (
        ".dat 数据集没有路由为可下载、可解析的表格。"
    )
    assert len(files) == 1 and files[0].path.suffix.lower() == ".dat" and len(parsed.tables) == 1, (
        ".dat 数据集没有进入空白分隔表格解析器。"
    )
    assert planner_total == 1 and any("skipped" in line for line in logs), (
        "来源摄取没有记录压缩包跳过，或 Planner 候选数不正确。"
    )
    return "来源摄取会跳过不支持压缩包，并将 .dat 路由到表格解析"


def check_artifact_plan_preflight_keeps_valid_calls() -> str:
    """One stale action must not invalidate the executable remainder of a plan."""

    from scidata_agent.agent.action_preflight import preflight_artifact_action_plan
    from scidata_agent.agent.schemas import (
        AgentState,
        ArtifactAction,
        ArtifactActionPlan,
        SourceArtifact,
        SourceCatalogEntry,
    )

    skipped_archive = SourceArtifact(
        artifact_id="terminal-archive",
        source_id="preflight-source",
        artifact_type="unknown",
        status="skipped",
        parser="unsupported_format",
    )
    remote_table = SourceArtifact(
        artifact_id="remote-table",
        source_id="preflight-source",
        artifact_type="csv",
        status="planned",
        url="https://example.org/table.dat",
    )
    local_pdf = SourceArtifact(
        artifact_id="local-pdf",
        source_id="preflight-source",
        artifact_type="pdf",
        status="downloaded",
        local_path=str(Path(__file__).resolve()),
    )
    state = AgentState(
        research_question="Offline action preflight regression check.",
        files=[],
        output_dir=BACKEND_ROOT,
        source_catalog=[
            SourceCatalogEntry(
                source_id="preflight-source",
                title="Preflight fixture",
                source_type="dataset",
                artifacts=[skipped_archive, remote_table, local_pdf],
            )
        ],
    )
    plan = ArtifactActionPlan(
        research_goal=state.research_question,
        actions=[
            ArtifactAction(
                action_id="drop-terminal",
                artifact_id=skipped_archive.artifact_id,
                action="read_file_manifest",
                purpose="Should never revisit a terminal archive.",
                reason="Regression check.",
            ),
            ArtifactAction(
                action_id="drop-unmaterialized-parser",
                artifact_id=remote_table.artifact_id,
                action="parse_csv",
                purpose="Parsing cannot precede materialization.",
                reason="Regression check.",
            ),
            ArtifactAction(
                action_id="keep-download",
                artifact_id=remote_table.artifact_id,
                action="download_artifact",
                purpose="The valid action must survive filtering.",
                reason="Regression check.",
            ),
            ArtifactAction(
                action_id="drop-wrong-parser",
                artifact_id=local_pdf.artifact_id,
                action="parse_csv",
                purpose="A PDF cannot use the CSV parser.",
                reason="Regression check.",
            ),
        ],
    )
    dropped = preflight_artifact_action_plan(plan, state)

    assert [action.action_id for action in plan.actions] == ["keep-download"], (
        "预检没有剔除终态、未物化或类型不匹配动作，或误删了合法下载动作。"
    )
    assert len(dropped) == 3 and any("terminal" in reason for reason in dropped), (
        "预检没有留下足够的可审计剔除原因。"
    )
    return "动作预检会剔除无效调用，同时保留同批合法下载动作"


def check_download_triggers_content_parse() -> str:
    """A newly downloaded direct artifact must trigger the parser, like source ingest."""

    from types import SimpleNamespace

    from scidata_agent.agent.scidata_agent import SciDataAgent

    result = SimpleNamespace(
        action="download_artifact",
        status="completed",
        output_counts={"files_delta": 1},
    )
    assert SciDataAgent._materialization_actions_need_source_parse([result]), (
        "download_artifact 成功后没有触发自动解析；HTML/PDF 会滞留在本地缓存。"
    )
    return "直接下载完成会自动进入解析链路"


def check_materialized_html_enters_parser() -> str:
    """The shared parser must turn a downloaded landing page into text evidence."""

    from scidata_agent.agent.schemas import UploadedFile
    from scidata_agent.tools.parser import parse_sources

    with tempfile.TemporaryDirectory(prefix="scidata-regression-") as directory:
        fixture = Path(directory) / "landing-page.html"
        fixture.write_text(
            "<html><head><script>ignore()</script></head><body>HTML evidence sentinel</body></html>",
            encoding="utf-8",
        )
        parsed = parse_sources([UploadedFile(filename=fixture.name, path=fixture)])
    assert len(parsed.text_blocks) == 1 and "HTML evidence sentinel" in parsed.text_blocks[0].text, (
        "共享解析器没有将 HTML landing page 转换为文本证据；"
        "下载后的网页仍会停留在缓存而无法参与动态抽取。"
    )
    return "物化的 HTML 会进入共享解析器并产生文本证据"


def check_dynamic_runtime_forces_source_selection() -> str:
    """After search, selection and triage must precede free-form artifact planning."""

    from scidata_agent.agent.schemas import (
        AgentState,
        DynamicExtractionPlan,
        MultiSourceSearchPlan,
        SourceDiscoveryPlan,
        SourceSelectionPlan,
        TaskPlan,
        TextBlock,
    )
    from scidata_agent.agent.action_executor import (
        parsed_content_fingerprint,
        source_content_fingerprint,
    )
    from scidata_agent.agent.scidata_agent import SciDataAgent

    state = AgentState(
        research_question="Offline workflow-stage regression check.",
        files=[],
        output_dir=BACKEND_ROOT / "outputs",
        task_plan=TaskPlan(research_goal="Require selection after search."),
        dynamic_extraction_plan=DynamicExtractionPlan(
            research_goal="Require selection after search."
        ),
        source_discovery_plan=SourceDiscoveryPlan(research_goal="Require selection after search."),
        multi_source_search_plan=MultiSourceSearchPlan(
            research_goal="Require selection after search."
        ),
        runtime_requires_source_discovery=True,
        tool_result_history=[{"tool_name": "search_sources", "status": "partial"}],
    )
    assert SciDataAgent._required_dynamic_workflow_actions(state) == ["select_sources"], (
        "搜索完成但没有选择计划时，运行时没有强制 select_sources。"
    )
    state.source_selection_plan = SourceSelectionPlan(
        research_goal="Require triage after selection."
    )
    assert SciDataAgent._required_dynamic_workflow_actions(state) == ["triage_sources"], (
        "来源选择完成后，运行时没有强制 triage_sources。"
    )
    return "搜索后会按 select_sources → triage_sources 顺序推进"


def check_dynamic_runtime_finishes_record_pipeline() -> str:
    """Dynamic extraction must continue through metric conversion and normalization."""

    from scidata_agent.agent.schemas import (
        AgentState,
        DynamicExtractionPlan,
        MultiSourceSearchPlan,
        SourceDiscoveryPlan,
        SourceSelectionPlan,
        TaskPlan,
        TextBlock,
    )
    from scidata_agent.agent.action_executor import (
        parsed_content_fingerprint,
        source_content_fingerprint,
    )
    from scidata_agent.agent.scidata_agent import SciDataAgent

    state = AgentState(
        research_question="Offline record-pipeline regression check.",
        files=[],
        output_dir=BACKEND_ROOT / "outputs",
        task_plan=TaskPlan(research_goal="Finish the record pipeline."),
        dynamic_extraction_plan=DynamicExtractionPlan(research_goal="Finish the record pipeline."),
        source_discovery_plan=SourceDiscoveryPlan(research_goal="Finish the record pipeline."),
        multi_source_search_plan=MultiSourceSearchPlan(research_goal="Finish the record pipeline."),
        source_selection_plan=SourceSelectionPlan(research_goal="Finish the record pipeline."),
        runtime_requires_source_discovery=True,
        tool_result_history=[
            {"tool_name": "search_sources", "status": "completed"},
            {"tool_name": "triage_sources", "status": "completed"},
            {"tool_name": "ingest_sources", "status": "completed"},
        ],
    )
    state.parsed_sources.text_blocks = [
        TextBlock(
            source_file="paper.txt",
            source_path="paper.txt",
            text="metric value 1.0",
            chunk_id="record-pipeline",
        )
    ]
    source_fingerprint = source_content_fingerprint(state)
    state.runtime_stage_fingerprints["extract_figures"] = source_fingerprint
    state.runtime_stage_fingerprints["interpret_sections"] = source_fingerprint
    fingerprint = parsed_content_fingerprint(state)
    state.runtime_stage_fingerprints["extract_dynamic_records"] = fingerprint
    state.dynamic_records = [object()]
    assert SciDataAgent._required_dynamic_workflow_actions(state) == ["extract_records"], (
        "动态记录生成后没有强制转换为候选指标记录。"
    )
    state.runtime_stage_fingerprints["extract_records"] = fingerprint
    state.candidate_records = [object()]
    assert SciDataAgent._required_dynamic_workflow_actions(state) == ["normalize_records"], (
        "候选指标记录生成后没有强制进入规范化。"
    )
    state.runtime_stage_fingerprints["normalize_records"] = fingerprint
    state.final_records = [object()]
    assert SciDataAgent._required_dynamic_workflow_actions(state) == ["track_provenance"], (
        "规范化记录生成后没有强制写入溯源。"
    )
    state.runtime_stage_fingerprints["track_provenance"] = fingerprint
    assert SciDataAgent._required_dynamic_workflow_actions(state) == ["validate_quality"], (
        "溯源完成后没有强制质量验证。"
    )
    return "动态记录会完成转换 → 规范化 → 溯源 → 质量验证"


def check_dynamic_runtime_requires_multimodal_preprocessing() -> str:
    """Every parsed evidence batch must pass figures and sections first."""

    from scidata_agent.agent.action_executor import source_content_fingerprint
    from scidata_agent.agent.schemas import AgentState, DynamicExtractionPlan, TaskPlan, TextBlock
    from scidata_agent.agent.scidata_agent import SciDataAgent

    state = AgentState(
        research_question="Extract multimodal evidence.",
        files=[],
        output_dir=BACKEND_ROOT / "outputs",
        task_plan=TaskPlan(research_goal="Extract multimodal evidence."),
        dynamic_extraction_plan=DynamicExtractionPlan(
            research_goal="Extract multimodal evidence."
        ),
    )
    state.parsed_sources.text_blocks = [
        TextBlock(
            source_file="paper.pdf",
            source_path="paper.pdf",
            text="Figure 1 reports the requested result.",
            chunk_id="multimodal-stage",
        )
    ]
    fingerprint = source_content_fingerprint(state)
    assert SciDataAgent._required_dynamic_workflow_actions(state) == ["extract_figures"], (
        "解析正文后没有强制进入图表解析。"
    )
    state.runtime_stage_fingerprints["extract_figures"] = fingerprint
    assert SciDataAgent._required_dynamic_workflow_actions(state) == ["interpret_sections"], (
        "图表解析后没有强制进入章节解析。"
    )
    return "每批解析内容都会先执行图表解析和章节解析"


def check_default_agent_budget_is_one_hundred() -> str:
    """The default runtime budget must match the public API maximum."""

    from scidata_agent.agent.scidata_agent import _agent_iteration_budget

    previous = os.environ.pop("SCIDATA_AGENT_MAX_ITERATIONS", None)
    try:
        assert _agent_iteration_budget(None) == 100, "Agent 默认最大轮次不是 100。"
    finally:
        if previous is not None:
            os.environ["SCIDATA_AGENT_MAX_ITERATIONS"] = previous
    return "Agent 默认最大工作流轮次为 100"


def check_zero_record_stages_do_not_loop() -> str:
    """A successful zero-row extraction is complete, not a retry signal."""

    from scidata_agent.agent.schemas import (
        AgentState,
        DynamicExtractionPlan,
        MultiSourceSearchPlan,
        SourceDiscoveryPlan,
        SourceSelectionPlan,
        TaskPlan,
        TextBlock,
    )
    from scidata_agent.agent.action_executor import (
        parsed_content_fingerprint,
        source_content_fingerprint,
    )
    from scidata_agent.agent.scidata_agent import SciDataAgent

    state = AgentState(
        research_question="Zero-row completion regression check.",
        files=[],
        output_dir=BACKEND_ROOT / "outputs",
        task_plan=TaskPlan(research_goal="Allow an honest zero-row result."),
        dynamic_extraction_plan=DynamicExtractionPlan(research_goal="Allow an honest zero-row result."),
        source_discovery_plan=SourceDiscoveryPlan(research_goal="Allow an honest zero-row result."),
        multi_source_search_plan=MultiSourceSearchPlan(research_goal="Allow an honest zero-row result."),
        source_selection_plan=SourceSelectionPlan(research_goal="Allow an honest zero-row result."),
        runtime_requires_source_discovery=True,
        tool_result_history=[
            {"tool_name": "search_sources", "status": "completed", "workflow_revision": 0},
            {"tool_name": "triage_sources", "status": "completed", "workflow_revision": 0},
        ],
    )
    state.parsed_sources.text_blocks = [
        TextBlock(
            source_file="empty.txt",
            source_path="empty.txt",
            text="no requested metric is present",
            chunk_id="zero-record",
        )
    ]
    source_fingerprint = source_content_fingerprint(state)
    state.runtime_stage_fingerprints["extract_figures"] = source_fingerprint
    state.runtime_stage_fingerprints["interpret_sections"] = source_fingerprint
    fingerprint = parsed_content_fingerprint(state)
    state.runtime_stage_fingerprints["extract_dynamic_records"] = fingerprint
    assert SciDataAgent._required_dynamic_workflow_actions(state) == ["extract_records"]
    state.runtime_stage_fingerprints["extract_records"] = fingerprint
    assert SciDataAgent._required_dynamic_workflow_actions(state) == ["normalize_records"]
    state.runtime_stage_fingerprints["normalize_records"] = fingerprint
    assert SciDataAgent._required_dynamic_workflow_actions(state) == ["track_provenance"]
    state.runtime_stage_fingerprints["track_provenance"] = fingerprint
    assert SciDataAgent._required_dynamic_workflow_actions(state) == ["validate_quality"]
    state.runtime_stage_fingerprints["validate_quality"] = fingerprint
    assert SciDataAgent._required_dynamic_workflow_actions(state) == [], (
        "零记录抽取已经成功完成，运行时仍把空列表误判为未执行并重复调度。"
    )
    return "零记录阶段按执行账本结束，不再因空列表循环"


def check_tool_idempotency_is_revision_scoped() -> str:
    """The same global tool may run again after search_more adds sources."""

    from scidata_agent.agent.tool_protocol import ToolCall

    first = ToolCall(call_id="first", tool_name="select_sources", workflow_revision=0)
    second = ToolCall(call_id="second", tool_name="select_sources", workflow_revision=1)
    assert first.effective_idempotency_key() != second.effective_idempotency_key(), (
        "全局工具幂等键仍跨工作流修订冲突，新来源加入后 select/triage 无法重跑。"
    )
    return "工具幂等键已按工作流修订隔离"


def check_artifact_parse_operations_are_independent() -> str:
    """PDF text completion must not suppress a later table parse."""

    from scidata_agent.agent.action_preflight import preflight_artifact_action_plan
    from scidata_agent.agent.schemas import ArtifactAction, ArtifactActionPlan

    state, artifact = _artifact_state(BACKEND_ROOT, artifact_type="pdf", status="parsed")
    artifact.local_path = str(Path(__file__).resolve())
    artifact.completed_operations = ["parse_pdf_text"]
    plan = ArtifactActionPlan(
        research_goal="Check independent parse modalities.",
        actions=[
            ArtifactAction(
                action_id="duplicate_text",
                artifact_id=artifact.artifact_id,
                action="parse_pdf_text",
                purpose="Duplicate text parse.",
                reason="Regression check.",
            ),
            ArtifactAction(
                action_id="new_table",
                artifact_id=artifact.artifact_id,
                action="parse_table",
                purpose="Extract tables after text.",
                reason="Regression check.",
            ),
        ],
    )
    preflight_artifact_action_plan(plan, state)
    assert [action.action for action in plan.actions] == ["parse_table"], (
        "artifact.status=parsed 仍然阻止不同模态解析，或未过滤同一解析动作。"
    )
    return "PDF 文本、表格和图片解析按独立操作记账"


def check_unavailable_coverage_does_not_block_stop() -> str:
    """An impossible evidence gap remains visible without forcing a loop."""

    from scidata_agent.agent.schemas import AgentState, DynamicExtractionPlan, InformationNeed
    from scidata_agent.tools.coverage import build_coverage_report

    state, artifact = _artifact_state(BACKEND_ROOT, artifact_type="image", status="failed")
    state.dynamic_extraction_plan = DynamicExtractionPlan(
        research_goal="Unavailable evidence regression check.",
        information_needs=[InformationNeed(need_name="figure", priority="high")],
        source_requirements=["figure"],
    )
    report = build_coverage_report(state)
    assert any(gap.status == "unavailable" for gap in report.gaps)
    assert report.decision == "allow_stop", (
        "所有候选图像均失败后，unavailable gap 仍阻塞 stop，运行时会无限 search_more。"
    )
    return "不可获取证据仍被报告，但不再阻塞工作流结束"


def check_terminal_harness_result_is_preserved() -> str:
    """The one-turn adapter must not rewrite terminal partial as running."""

    from types import SimpleNamespace

    from scidata_agent.agent.schemas import AgentState
    from scidata_agent.agent.scidata_agent import SciDataAgent

    state = AgentState(
        research_question="Terminal harness regression check.",
        files=[],
        output_dir=BACKEND_ROOT / "outputs",
    )

    class TerminalHarness:
        def run(self, *_args, **_kwargs):
            return SimpleNamespace(
                trace=[],
                status="partial",
                terminal=True,
                stop_reason="Repeated stop rejection.",
            )

    SciDataAgent._execute_shared_agent_harness_iteration(
        state, harness=TerminalHarness(), iteration=0
    )
    assert state.runtime_status == "partial" and state.runtime_stop_reason, (
        "共享 harness 已返回终态，但适配器仍将其覆盖为 running。"
    )
    return "共享 harness 的终态不会再被覆盖为 running"


def check_llm_timeout_triggers_model_failover() -> str:
    from scidata_agent.llm.client import LLMCallError, QwenBailianClient

    assert QwenBailianClient._should_failover(LLMCallError("request timed out")), (
        "LLM 超时没有触发模型池切换。"
    )
    return "LLM 超时会切换到下一候选模型"


def check_disabled_download_policy_can_finish() -> str:
    """Metadata-only runs must not schedule downloads or block forever."""

    from scidata_agent.agent.decision import AgentDecision
    from scidata_agent.agent.policy import AgentPolicy
    from scidata_agent.agent.schemas import (
        DynamicExtractionPlan,
        MultiSourceSearchPlan,
        SourceDiscoveryPlan,
        SourceSelectionPlan,
        SourceTriageDecision,
        TaskPlan,
    )
    from scidata_agent.agent.scidata_agent import SciDataAgent
    from scidata_agent.agent.tool_protocol import ToolCall
    from scidata_agent.agent.tool_registry import build_artifact_tool_registry
    from scidata_agent.tools.coverage import build_coverage_report

    state, artifact = _artifact_state(BACKEND_ROOT, artifact_type="pdf", status="selected")
    artifact.url = "https://example.org/remote-only.pdf"
    state.task_plan = TaskPlan(research_goal="Metadata-only run.")
    state.dynamic_extraction_plan = DynamicExtractionPlan(research_goal="Metadata-only run.")
    state.source_discovery_plan = SourceDiscoveryPlan(research_goal="Metadata-only run.")
    state.multi_source_search_plan = MultiSourceSearchPlan(research_goal="Metadata-only run.")
    state.source_selection_plan = SourceSelectionPlan(research_goal="Metadata-only run.")
    state.source_triage_decisions = [
        SourceTriageDecision(
            source_id=artifact.source_id,
            title="Remote paper",
            provider="crossref",
            recommended_action="download_pdf",
            should_ingest=True,
        )
    ]
    state.runtime_requires_source_discovery = True
    state.runtime_auto_download_sources = False
    state.tool_result_history = [
        {"tool_name": "search_sources", "status": "completed", "workflow_revision": 0},
        {"tool_name": "triage_sources", "status": "completed", "workflow_revision": 0},
    ]
    assert SciDataAgent._required_dynamic_workflow_actions(state) == []
    policy = AgentPolicy(build_artifact_tool_registry()).validate(
        AgentDecision(
            decision="continue",
            reason="Try a forbidden remote download.",
            tool_calls=[
                ToolCall(
                    call_id="remote_download",
                    tool_name="download_artifact",
                    arguments={"artifact_id": artifact.artifact_id},
                )
            ],
        ),
        state,
    )
    assert not policy.allowed
    assert build_coverage_report(state).decision == "allow_stop", (
        "auto_download_sources=False 时，远程内容仍被当作可修复 blocker。"
    )
    return "禁用下载时不会调度远程下载，metadata-only 工作流可结束"


def check_artifact_planner_catalog_is_bounded() -> str:
    """Planner input must cap artifact count and omit unbounded raw metadata."""

    from scidata_agent.agent.schemas import SourceArtifact, SourceCatalogEntry
    from scidata_agent.llm.nodes import _artifact_planner_catalog_payload

    catalog = [
        SourceCatalogEntry(
            source_id=f"source_{index}",
            title=f"Source {index}",
            source_type="paper",
            artifacts=[
                SourceArtifact(
                    artifact_id=f"artifact_{index}",
                    source_id=f"source_{index}",
                    artifact_type="pdf",
                    url=f"https://example.org/{index}.pdf",
                    status="planned",
                    metadata={"large_untrusted_payload": "x" * 10000},
                )
            ],
        )
        for index in range(101)
    ]
    payload, total = _artifact_planner_catalog_payload(catalog)
    artifacts = [artifact for source in payload for artifact in source["artifacts"]]
    assert total == 101 and len(artifacts) <= 100, (
        f"Artifact planner 未限制输入：total={total}, selected={len(artifacts)}。"
    )
    assert all("metadata" not in artifact for artifact in artifacts), (
        "Artifact planner 输入仍包含原始 artifact.metadata，可能无限膨胀提示词。"
    )
    return f"artifact planner 输入已裁剪为 {len(artifacts)}/{total} 条"


def check_artifact_planner_context_is_bounded() -> str:
    """Planner context must omit tool schemas and cap noisy runtime history."""

    from scidata_agent.llm.nodes import (
        _artifact_planner_observation_payload,
        _artifact_planner_processing_log_payload,
    )

    observation = {
        "iteration": 9,
        "task": {"research_question": "x" * 5000, "target_fields": list(range(100))},
        "sources": {"items": [{"source_id": index} for index in range(100)]},
        "artifacts": {"items": [{"artifact_id": index} for index in range(100)]},
        "available_tools": [{"schema": "x" * 100000}],
        "recent_results": [{"message": "x" * 20000} for _ in range(20)],
    }
    compact = _artifact_planner_observation_payload(json.dumps(observation))
    decoded = json.loads(compact)
    logs = _artifact_planner_processing_log_payload(["x" * 2000 for _ in range(40)])
    assert "available_tools" not in decoded, "artifact planner 仍接收完整工具 schema。"
    assert len(decoded["sources"]["items"]) <= 16 and len(decoded["artifacts"]["items"]) <= 24, (
        "artifact planner observation 未限制来源或 artifact 列表。"
    )
    assert len(logs) <= 16 and all(len(item) <= 420 for item in logs), (
        "artifact planner processing log 未限制条数或单条长度。"
    )
    return "artifact planner observation、日志和工具 schema 已裁剪"


def check_exporter_receives_final_status_and_refreshes_catalog() -> str:
    """Export must preserve catalog status and never hard-code completed."""

    from scidata_agent.tools import exporter

    source = inspect.getsource(exporter.export_results)
    signature = inspect.signature(exporter.export_results)
    assert "final_status" in signature.parameters, "export_results 未接收最终任务状态。"
    assert "refresh_source_catalog(state)" in source, "导出仍绕过 refresh_source_catalog。"
    assert '"status": final_status or _export_status(state)' in source, (
        "result.json 状态仍可能被固定写成 completed。"
    )
    return "导出会保留 catalog 状态并写入最终任务状态"


def check_arxiv_ingest_triggers_content_refresh() -> str:
    """A successful arXiv ingest must enter the parsing path automatically."""

    from types import SimpleNamespace

    from scidata_agent.agent.scidata_agent import SciDataAgent

    result = SimpleNamespace(
        action="ingest_arxiv_pdfs",
        status="completed",
        output_counts={"files_delta": 1},
    )
    needs_refresh = SciDataAgent._artifact_actions_need_content_refresh([result])
    needs_parse = SciDataAgent._ingestion_actions_need_source_parse([result])
    assert needs_refresh and needs_parse, (
        "ingest_arxiv_pdfs 成功后未触发自动解析链路；已下载 PDF 会滞留在缓存，"
        "随后由无关 artifact 操作耗尽迭代。"
    )
    return "arXiv 摄取完成会自动进入解析链路"


def check_zenodo_anonymous_page_size() -> str:
    """The anonymous Zenodo records API must never receive size > 25."""

    from scidata_agent.agent.schemas import SourceSearchRequest
    from scidata_agent.tools.connectors import zenodo

    captured: dict = {}
    original = zenodo.fetch_json

    def fake_fetch_json(url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return {"hits": {"hits": []}}

    zenodo.fetch_json = fake_fetch_json
    try:
        zenodo.ZenodoConnector().search(
            SourceSearchRequest(
                connector_name="zenodo",
                source_type="dataset",
                query="type ia supernova light curve",
                max_results=100,
            )
        )
    finally:
        zenodo.fetch_json = original

    size = int((captured.get("params") or {}).get("size") or 0)
    assert 1 <= size <= 25, (
        f"Zenodo 匿名搜索仍发送 size={size or 'missing'}；应在连接器内限制到 1..25。"
    )
    return f"Zenodo 匿名搜索 size={size}"


def _call_name(node: ast.Call) -> str:
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def check_figshare_uses_post_json() -> str:
    """Figshare search must be POST /articles/search with a JSON body."""

    from scidata_agent.tools.connectors import figshare

    source = textwrap.dedent(inspect.getsource(figshare._fetch_figshare_search))
    tree = ast.parse(source)
    has_post_call = False
    has_post_method = False
    has_json_body = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node).casefold()
        if "post" in call_name:
            has_post_call = True
        for keyword in node.keywords:
            if keyword.arg == "method" and isinstance(keyword.value, ast.Constant):
                has_post_method = str(keyword.value.value).upper() == "POST"
            if keyword.arg in {"json", "json_body", "body", "data"}:
                has_json_body = True

    source_lower = source.casefold()
    endpoint_is_search = "figshare_search_url" in source_lower or "/articles/search" in source_lower
    post_is_explicit = has_post_call or has_post_method or 'method="post"' in source_lower or "method='post'" in source_lower
    assert endpoint_is_search and post_is_explicit and (has_json_body or has_post_call), (
        "Figshare 搜索仍不像 POST JSON 请求；应调用 POST /v2/articles/search，并把 search_for/page_size 放入 JSON body。"
    )
    return "Figshare 搜索已使用 POST JSON"


def check_tls_uses_explicit_ca_bundle() -> str:
    """Pinned HTTPS connections should load an explicit, existing CA bundle."""

    from scidata_agent.tools import url_safety

    module_source = inspect.getsource(url_safety).casefold()
    mentions_bundle = "certifi" in module_source or "ssl_cert_file" in module_source
    loads_bundle = "cafile=" in module_source or "load_verify_locations" in module_source
    assert mentions_bundle and loads_bundle, (
        "HTTPS 安全连接仍只使用默认 CA；应显式加载 certifi.where() 或 SSL_CERT_FILE 指向的 CA bundle。"
    )
    return "HTTPS 安全连接已显式加载 CA bundle"


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("manifest 真实内容读取", check_manifest_reads_local_content),
    ("metadata 后继续正文处理", check_metadata_action_keeps_content_work_pending_without_repeating_it),
    ("catalog 刷新保留 inspected 状态", check_catalog_refresh_preserves_inspection_status),
    ("manifest 文件类型与下载状态保留", check_catalog_manifest_file_keeps_type_and_download_state),
    ("triage 必须先完成来源选择", check_triage_requires_source_selection),
    ("解析动作必须已有本地文件", check_parser_requires_materialized_local_artifact),
    ("landing page 受控下载路径", check_policy_allows_selected_landing_page_download),
    ("直接下载登记本地文件", check_download_registers_materialized_file),
    ("不支持格式终态跳过", check_unsupported_download_is_terminally_skipped),
    ("来源摄取格式路由", check_ingestion_routes_archives_and_dat_files),
    ("artifact 动作预检", check_artifact_plan_preflight_keeps_valid_calls),
    ("直接下载触发解析链路", check_download_triggers_content_parse),
    ("HTML 下载进入共享解析器", check_materialized_html_enters_parser),
    ("动态来源阶段强制推进", check_dynamic_runtime_forces_source_selection),
    ("动态多模态阶段强制推进", check_dynamic_runtime_requires_multimodal_preprocessing),
    ("动态记录完成指标管线", check_dynamic_runtime_finishes_record_pipeline),
    ("零记录阶段防循环", check_zero_record_stages_do_not_loop),
    ("工具幂等键按修订隔离", check_tool_idempotency_is_revision_scoped),
    ("artifact 多模态解析独立", check_artifact_parse_operations_are_independent),
    ("不可获取 coverage 不阻塞", check_unavailable_coverage_does_not_block_stop),
    ("harness 终态保留", check_terminal_harness_result_is_preserved),
    ("默认 Agent 轮次上限", check_default_agent_budget_is_one_hundred),
    ("LLM 超时模型切换", check_llm_timeout_triggers_model_failover),
    ("禁用下载分支可结束", check_disabled_download_policy_can_finish),
    ("arXiv 摄取触发解析链路", check_arxiv_ingest_triggers_content_refresh),
    ("artifact planner 输入上限", check_artifact_planner_catalog_is_bounded),
    ("artifact planner 上下文裁剪", check_artifact_planner_context_is_bounded),
    ("导出状态与 catalog 一致", check_exporter_receives_final_status_and_refreshes_catalog),
    ("Zenodo 匿名分页限制", check_zenodo_anonymous_page_size),
    ("Figshare POST JSON 搜索", check_figshare_uses_post_json),
    ("TLS 显式 CA bundle", check_tls_uses_explicit_ca_bundle),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="离线快速检查 SciData Agent 本次任务暴露出的关键回归问题。"
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON。")
    args = parser.parse_args()

    started = time.perf_counter()
    results = [_run_check(name, check) for name, check in CHECKS]
    elapsed_ms = (time.perf_counter() - started) * 1000
    passed = sum(item.passed for item in results)
    failed = len(results) - passed

    if args.json:
        print(
            json.dumps(
                {
                    "passed": passed,
                    "failed": failed,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "checks": [asdict(item) for item in results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("=" * 68)
        print("SciData Agent 快速回归检查（离线，无 LLM/网络调用）")
        print("=" * 68)
        for item in results:
            marker = "PASS" if item.passed else "FAIL"
            print(f"[{marker}] {item.name} ({item.elapsed_ms:.1f} ms)")
            print(f"       {item.detail}")
        print("-" * 68)
        print(f"结果: {passed}/{len(results)} 通过，{failed} 失败；耗时 {elapsed_ms:.1f} ms")
        if failed:
            print("结论: 关键问题尚未全部解决。根据 FAIL 项修复后重新运行本脚本。")
        else:
            print("结论: 本次任务暴露出的关键回归检查已全部通过，可以再跑一次小型集成案例。")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
