from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any

from scidata_agent.agent.schemas import ChartExtraction, CrossModalCheck, FigureAsset, TableBlock, TextBlock


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def build_cross_modal_checks(
    text_blocks: list[TextBlock],
    tables: list[TableBlock],
    figures: list[FigureAsset],
    chart_extractions: list[ChartExtraction],
) -> list[CrossModalCheck]:
    """Audit whether text, table, and figure evidence can corroborate each other."""
    text_by_location = _group_by_location(text_blocks)
    table_by_location = _group_by_location(tables)
    figures_by_id = {figure.figure_id: figure for figure in figures}
    checks: list[CrossModalCheck] = []
    for extraction in chart_extractions:
        figure = figures_by_id.get(extraction.figure_id)
        if figure is None:
            continue
        location = _location_key(figure.source_file, figure.page)
        checks.append(_check_figure(
            figure,
            extraction,
            text_by_location.get(location, []),
            table_by_location.get(location, []),
        ))
    for table in tables:
        checks.append(_check_table(
            table,
            text_by_location.get(_location_key(table.source_file, table.page), []),
        ))
    return checks


def _check_figure(figure: FigureAsset, extraction: ChartExtraction, nearby_text: list[Any], nearby_tables: list[Any]) -> CrossModalCheck:
    modalities = ["figure"]
    if nearby_text:
        modalities.append("text")
    if nearby_tables:
        modalities.append("table")
    chart_numbers = _chart_numbers(extraction)
    reference_numbers = _numbers_from_items([figure.caption or "", *nearby_text, *nearby_tables])
    matches = _compatible_number_matches(chart_numbers, reference_numbers)
    issues: list[str] = []
    if not extraction.contains_data or not chart_numbers:
        status = "not_comparable"
        issues.append("Figure is qualitative or contains no extractable numeric series.")
    elif not nearby_text and not nearby_tables:
        status = "not_comparable"
        issues.append("No same-page text or table evidence was available for comparison.")
    elif matches:
        status = "supported"
    else:
        status = "partial"
        issues.append("Chart values were not numerically corroborated by nearby text/table evidence.")
    return CrossModalCheck(
        check_id=_check_id("figure", figure.source_file, figure.page, figure.figure_id),
        source_file=figure.source_file,
        page=figure.page,
        subject_id=figure.figure_id,
        modalities=modalities,
        status=status,
        matched_value_count=matches,
        candidate_value_count=len(chart_numbers),
        evidence_refs=[figure.figure_id, *[table.table_id for table in nearby_tables]],
        issues=issues,
        confidence=_confidence(status, matches, len(chart_numbers)),
    )


def _check_table(table: TableBlock, nearby_text: list[Any]) -> CrossModalCheck:
    modalities = ["table"] + (["text"] if nearby_text else [])
    table_numbers = _numbers_from_items(table.rows)
    text_numbers = _numbers_from_items(nearby_text)
    matches = _compatible_number_matches(table_numbers, text_numbers)
    if not nearby_text:
        status = "not_comparable"
        issues = ["No same-page text evidence was available for comparison."]
    elif matches:
        status = "supported"
        issues = []
    else:
        status = "partial"
        issues = ["Table values were not numerically corroborated by nearby text evidence."]
    return CrossModalCheck(
        check_id=_check_id("table", table.source_file, table.page, table.table_id),
        source_file=table.source_file,
        page=table.page,
        subject_id=table.table_id,
        modalities=modalities,
        status=status,
        matched_value_count=matches,
        candidate_value_count=len(table_numbers),
        evidence_refs=[table.table_id],
        issues=issues,
        confidence=_confidence(status, matches, len(table_numbers)),
    )


def _group_by_location(items: Iterable[Any]) -> dict[tuple[str, int | None], list[Any]]:
    grouped: dict[tuple[str, int | None], list[Any]] = {}
    for item in items:
        grouped.setdefault(_location_key(item.source_file, item.page), []).append(item)
    return grouped


def _location_key(source_file: str, page: int | None) -> tuple[str, int | None]:
    return (str(source_file).replace("\\", "/").casefold(), page)


def _chart_numbers(extraction: ChartExtraction) -> list[float]:
    values: list[float] = []
    for series in extraction.series:
        for point in series.points:
            values.extend(float(value) for value in point if isinstance(value, (int, float)))
    return values


def _numbers_from_items(items: Iterable[Any]) -> list[float]:
    values: list[float] = []
    for item in items:
        if isinstance(item, dict):
            values.extend(_numbers_from_items(item.values()))
        elif isinstance(item, (list, tuple, set)):
            values.extend(_numbers_from_items(item))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            values.append(float(item))
        else:
            for match in _NUMBER_RE.findall(str(item or "")):
                try:
                    values.append(float(match))
                except ValueError:
                    continue
    return values


def _compatible_number_matches(values: list[float], references: list[float]) -> int:
    if not values or not references:
        return 0
    matches = 0
    for value in values:
        tolerance = max(0.02, abs(value) * 0.03)
        if any(abs(value - reference) <= tolerance for reference in references):
            matches += 1
    return matches


def _confidence(status: str, matches: int, candidate_count: int) -> float:
    if status == "supported":
        return min(1.0, 0.6 + 0.4 * matches / max(1, candidate_count))
    if status == "partial":
        return 0.35
    return 0.0


def _check_id(kind: str, source_file: str, page: int | None, subject_id: str) -> str:
    digest = hashlib.sha1(f"{kind}|{source_file}|{page}|{subject_id}".encode("utf-8")).hexdigest()[:10]
    return f"cross_modal_{digest}"
