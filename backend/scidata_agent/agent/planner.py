from __future__ import annotations

import re

from scidata_agent.agent.field_schema import DEFAULT_TARGET_FIELDS
from scidata_agent.agent.schemas import TaskPlan


def plan_task(research_question: str) -> TaskPlan:
    """Rule fallback planner used only for local testing or graceful failure."""

    lowered = research_question.lower()
    domain = "scientific data extraction"
    if any(token in lowered for token in ["perovskite", "solar", "battery", "catalyst", "material", "材料", "电池", "催化"]):
        domain = "materials science"
    elif any(token in lowered for token in ["paper", "pdf", "论文", "文献"]):
        domain = "scientific literature"
    elif any(token in lowered for token in ["model", "dataset", "accuracy", "benchmark", "模型", "数据集"]):
        domain = "machine learning research"

    assumptions = [
        "规则 fallback 仅用于本地工具链测试；正式参赛结果应由 Qwen Task Planner 生成。",
        "未在文本或表格中出现的信息不会被编造，缺失字段保留为空。",
    ]
    if re.search(r"图|chart|figure|image", research_question, re.IGNORECASE):
        assumptions.append("当前 fallback 不做图像坐标数据还原，仅处理 PDF 文本、CSV 和 Excel。")

    return TaskPlan(
        domain=domain,
        target_fields=list(DEFAULT_TARGET_FIELDS),
        output_format=["csv", "json"],
        need_provenance=True,
        assumptions=assumptions,
    )

