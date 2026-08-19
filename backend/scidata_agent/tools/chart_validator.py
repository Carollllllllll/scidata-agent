from __future__ import annotations

import re

from scidata_agent.agent.schemas import ChartExtraction, ChartValidationIssue, ChartValidationResult, FigureAsset

# Numbers embedded in captions, e.g. "peaks at 550 nm", " declines by 2.5 mag".
_CAPTION_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
# Unit-like tokens that sometimes follow numbers in captions/axis labels.
_UNIT_TOKEN_RE = re.compile(
    r"(?i)\b(nm|um|µm|angstrom|å|hz|khz|mhz|ghz|kev|mev|gev|mjy|jy|mag|days?|hours?|sec(?:onds?)?|"
    r"m/s|km/s|k|mk|pc|kpc|mpc|msun|lsun|kg|g|cm|mm|m)\b"
)

_AXIS_TOLERANCE = 0.08  # data points may exceed the stated axis range by 8%
_LOW_CONFIDENCE = 0.45


def validate_chart_extraction(
    extraction: ChartExtraction,
    figure: FigureAsset | None = None,
) -> ChartValidationResult:
    """Deterministic stage-1 checks for a VL chart extraction.

    Checks (no LLM calls):
    1. Data points must fall inside the declared axis ranges (with tolerance).
    2. Series must contain usable points.
    3. Units mentioned in the caption should not contradict axis units.
    4. Low-confidence or axis-calibration-missing extractions are flagged.

    The self-correction loop (re-asking the VL model with the issues as
    feedback) is stage 2; for now suspicious results are marked needs_review.
    """
    issues: list[ChartValidationIssue] = []

    if not extraction.series:
        issues.append(
            ChartValidationIssue(
                severity="error",
                code="no_series",
                message="图表提取结果没有任何数据序列。",
                suggestion="重新提取或标记人工复核。",
            )
        )

    x_axis, y_axis = extraction.x_axis, extraction.y_axis
    if x_axis.range_min is None or x_axis.range_max is None:
        issues.append(
            ChartValidationIssue(
                severity="warning",
                code="axis_range_missing",
                message="x 轴刻度范围未能读取，数据点坐标无法校验。",
                suggestion="检查坐标轴解析是否错误，必要时人工复核。",
            )
        )
    if y_axis.range_min is None or y_axis.range_max is None:
        issues.append(
            ChartValidationIssue(
                severity="warning",
                code="axis_range_missing",
                message="y 轴刻度范围未能读取，数据点坐标无法校验。",
                suggestion="检查坐标轴解析是否错误，必要时人工复核。",
            )
        )

    total_points = 0
    out_of_range = 0
    for series in extraction.series:
        total_points += len(series.points)
        for x_value, y_value in series.points:
            if not _within_axis(x_value, x_axis.range_min, x_axis.range_max) or not _within_axis(
                y_value, y_axis.range_min, y_axis.range_max
            ):
                out_of_range += 1

    if extraction.series and total_points == 0:
        issues.append(
            ChartValidationIssue(
                severity="error",
                code="no_points",
                message="数据序列存在但没有任何有效数据点。",
                suggestion="重新提取或标记人工复核。",
            )
        )
    if total_points and out_of_range / total_points > 0.2:
        issues.append(
            ChartValidationIssue(
                severity="error",
                code="axis_range_mismatch",
                message=(
                    f"{out_of_range}/{total_points} 个数据点超出声明的坐标轴范围，"
                    "坐标轴刻度很可能被错误解析。"
                ),
                suggestion="重新读取坐标轴刻度（注意线性/对数轴），或人工复核。",
            )
        )
    elif out_of_range:
        issues.append(
            ChartValidationIssue(
                severity="warning",
                code="points_near_axis_edge",
                message=f"{out_of_range} 个数据点略超坐标轴范围，可能是读数误差。",
            )
        )

    caption = (figure.caption if figure else None) or ""
    unit_issue = _caption_unit_check(caption, extraction)
    if unit_issue:
        issues.append(unit_issue)

    if extraction.confidence < _LOW_CONFIDENCE:
        issues.append(
            ChartValidationIssue(
                severity="warning",
                code="low_confidence",
                message=f"VL 提取置信度偏低（{extraction.confidence:.2f}）。",
                suggestion="建议人工复核该图表的坐标轴与图例解析。",
            )
        )

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    return ChartValidationResult(
        figure_id=extraction.figure_id,
        passed=error_count == 0,
        issues=issues,
        needs_review=error_count > 0 or warning_count >= 2 or extraction.confidence < _LOW_CONFIDENCE,
    )


def _within_axis(value: float, range_min: float | None, range_max: float | None) -> bool:
    if range_min is None or range_max is None:
        return True
    span = abs(range_max - range_min) or 1.0
    margin = span * _AXIS_TOLERANCE
    lower, upper = (range_min, range_max) if range_min <= range_max else (range_max, range_min)
    return lower - margin <= value <= upper + margin


def _caption_unit_check(caption: str, extraction: ChartExtraction) -> ChartValidationIssue | None:
    """Flag when the caption's unit family contradicts the parsed axis units."""
    if not caption:
        return None
    caption_units = {token.lower() for token in _UNIT_TOKEN_RE.findall(caption)}
    if not caption_units:
        return None
    axis_units = {
        unit
        for unit in [extraction.x_axis.unit, extraction.y_axis.unit]
        if unit
    }
    normalized_axis = {_normalize_unit(unit) for unit in axis_units}
    wavelength_units = {"nm", "um", "µm", "angstrom", "å"}
    time_units = {"day", "days", "hour", "hours", "sec", "second", "seconds"}
    families = [("wavelength", wavelength_units), ("time", time_units)]
    for family_name, family in families:
        if caption_units & family and normalized_axis and not (normalized_axis & family):
            return ChartValidationIssue(
                severity="warning",
                code="unit_suspect",
                message=(
                    f"caption 中出现了{family_name}单位 {sorted(caption_units & family)}，"
                    f"但坐标轴单位为 {sorted(axis_units)}，可能存在单位或图例解析错误。"
                ),
                suggestion="核对坐标轴标签与图例，必要时人工复核。",
            )
    return None


def _normalize_unit(unit: str) -> str:
    return unit.strip().lower().replace("μ", "µ").rstrip("s") if unit else ""
