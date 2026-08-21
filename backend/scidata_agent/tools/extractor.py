from __future__ import annotations

import re
from typing import Any

from scidata_agent.agent.schemas import ScientificRecord, SourceType, TableBlock, TextBlock


METRIC_ALIASES = {
    "pce": "PCE",
    "power conversion efficiency": "PCE",
    "efficiency": "efficiency",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "F1",
    "rmse": "RMSE",
    "mae": "MAE",
    "psnr": "PSNR",
    "ssim": "SSIM",
    "fid": "FID",
    "absorption": "absorption wavelength",
    "wavelength": "wavelength",
    "bandgap": "bandgap",
    "temperature": "temperature",
    "stability": "stability",
    "yield": "yield",
}

MATERIAL_PATTERNS = [
    r"\b(?:MAPbI3|FAPbI3|CsPbI3|CsPbBr3|TiO2|SnO2|perovskite|graphene|silicon|catalyst|chromophore)\b",
    r"\b[A-Z][a-z]?[A-Z]?[a-z]?\d?(?:[A-Z][a-z]?\d?){1,4}\b",
]

METHOD_PATTERNS = [
    "spin coating",
    "annealing",
    "sol-gel",
    "hydrothermal",
    "random forest",
    "transformer",
    "diffusion",
    "normalizing flow",
    "fine-tuning",
]


def extract_records_from_text_blocks(text_blocks: list[TextBlock]) -> list[ScientificRecord]:
    records: list[ScientificRecord] = []
    for block in text_blocks:
        records.extend(_extract_records_from_text_block(block))
    return records


def extract_records_from_tables(tables: list[TableBlock]) -> list[ScientificRecord]:
    records: list[ScientificRecord] = []
    for table in tables:
        records.extend(_extract_records_from_table(table))
    return records


def _extract_records_from_text_block(block: TextBlock) -> list[ScientificRecord]:
    records: list[ScientificRecord] = []
    sentences = _split_sentences(block.text)
    for sentence in sentences:
        lowered = sentence.lower()
        metrics = [alias for alias in METRIC_ALIASES if alias in lowered]
        if not metrics:
            continue
        numbers = _find_numbers_with_units(sentence)
        if not numbers:
            continue
        material = _guess_material(sentence)
        method = _guess_method(sentence)
        for metric_alias in metrics[:2]:
            metric_name = METRIC_ALIASES[metric_alias]
            relevant_numbers = _select_metric_numbers(sentence, metric_alias, numbers)
            for value, unit in relevant_numbers[:1]:
                records.append(
                    ScientificRecord(
                        paper_title=_guess_title(block.text),
                        material=material,
                        method=method,
                        metric_name=metric_name,
                        metric_value=value,
                        unit=unit,
                        condition=_guess_condition(sentence),
                        source_file=block.source_file,
                        source_type=SourceType.PDF_TEXT,
                        page=block.page,
                        evidence_text=_truncate(sentence, 500),
                        confidence=0.72,
                    )
                )
    return records


def _extract_records_from_table(table: TableBlock) -> list[ScientificRecord]:
    records: list[ScientificRecord] = []
    columns = table.columns
    normalized_columns = {column: _normalize_key(column) for column in columns}

    metric_columns = [
        column
        for column, normalized in normalized_columns.items()
        if _column_looks_metric(normalized)
    ]
    if not metric_columns:
        metric_columns = [column for column in columns if _first_numeric_value(table.rows, column) is not None]

    material_column = _find_column(normalized_columns, ["material", "compound", "sample", "molecule", "model", "method"])
    method_column = _find_column(normalized_columns, ["method", "preparation", "process", "architecture", "algorithm"])
    condition_column = _find_column(normalized_columns, ["condition", "dataset", "solvent", "temperature", "test", "environment"])
    title_column = _find_column(normalized_columns, ["paper", "title", "source"])
    unit_column = _find_column(normalized_columns, ["unit"])

    for row_index, row in enumerate(table.rows, 1):
        for metric_column in metric_columns:
            raw_value = row.get(metric_column)
            value, unit_from_value = _coerce_number_and_unit(raw_value)
            if value is None:
                continue
            metric_name = _metric_name_from_column(metric_column)
            unit = unit_from_value or _unit_from_column(metric_column) or (_value_as_str(row.get(unit_column)) if unit_column else None)
            evidence = "; ".join(f"{key}={_value_as_str(value)}" for key, value in row.items() if value is not None)
            records.append(
                ScientificRecord(
                    paper_title=_value_as_str(row.get(title_column)) if title_column else None,
                    material=_value_as_str(row.get(material_column)) if material_column else None,
                    method=_value_as_str(row.get(method_column)) if method_column else None,
                    metric_name=metric_name,
                    metric_value=value,
                    unit=unit,
                    condition=_value_as_str(row.get(condition_column)) if condition_column else None,
                    source_file=table.source_file,
                    source_type=table.source_type,
                    page=table.page,
                    evidence_text=f"row {row_index}: {evidence}",
                    confidence=0.84,
                    raw={"row_index": row_index, "table_id": table.table_id},
                )
            )
    return records


def _split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?。！？])\s+", text)
    return [piece.strip() for piece in pieces if 20 <= len(piece.strip()) <= 900]


def _find_numbers_with_units(text: str) -> list[tuple[float, str | None]]:
    pattern = re.compile(
        r"(?<![A-Za-z0-9])([-+]?\d+(?:\.\d+)?)\s*(%|percent|nm|K|°C|C|eV|mA/cm2|mA cm-2|V|mV|g|mg|kg|s|ms|h|hours|days|mmol|mol|M|AU|dB)?",
        re.IGNORECASE,
    )
    matches: list[tuple[float, str | None]] = []
    for raw_value, raw_unit in pattern.findall(text):
        try:
            value = float(raw_value)
        except ValueError:
            continue
        unit = _normalize_unit(raw_unit) if raw_unit else None
        if _number_is_probably_reference(value, unit):
            continue
        matches.append((value, unit))
    return matches


def _select_metric_numbers(sentence: str, metric_alias: str, numbers: list[tuple[float, str | None]]) -> list[tuple[float, str | None]]:
    metric_index = sentence.lower().find(metric_alias)
    if metric_index < 0:
        return numbers
    candidates: list[tuple[int, tuple[float, str | None]]] = []
    for value, unit in numbers:
        raw = str(int(value)) if value.is_integer() else str(value)
        index = sentence.find(raw)
        if index == -1:
            index = sentence.find(f"{value:g}")
        distance = abs(index - metric_index) if index != -1 else 9999
        if unit and unit.lower() == "g" and "am 1.5g" in sentence.lower():
            distance += 5000
        candidates.append((distance, (value, unit)))
    return [item for _, item in sorted(candidates, key=lambda pair: pair[0])]


def _number_is_probably_reference(value: float, unit: str | None) -> bool:
    return unit is None and value.is_integer() and (value > 1900 or value < 0)


def _guess_material(text: str) -> str | None:
    for pattern in MATERIAL_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def _guess_method(text: str) -> str | None:
    lowered = text.lower()
    for method in METHOD_PATTERNS:
        if method in lowered:
            return method
    return None


def _guess_title(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    first = lines[0]
    if 8 <= len(first) <= 180 and len(first.split()) >= 3:
        return first
    return None


def _guess_condition(text: str) -> str | None:
    condition_markers = ["under", "at ", "in ", "on ", "dataset", "test set", "AM 1.5", "solvent"]
    lowered = text.lower()
    for marker in condition_markers:
        index = lowered.find(marker)
        if index != -1:
            return _truncate(text[index:], 180)
    return None


def _coerce_number_and_unit(value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        return float(value), None
    text = str(value)
    # Do not treat digits embedded in identifiers such as ``GPT-4`` or
    # ``model_v2`` as scientific measurements.  Including sign characters in
    # both boundaries is intentional: otherwise the engine can skip the
    # optional ``-`` and still match the ``4`` in ``GPT-4``.
    match = re.search(
        r"(?<![\w.+-])([-+]?\d+(?:\.\d+)?)(?![\w.+-])\s*([A-Za-z%/°0-9.-]+)?",
        text,
    )
    if not match:
        return None, None
    return float(match.group(1)), _normalize_unit(match.group(2)) if match.group(2) else None


def _metric_name_from_column(column: str) -> str:
    normalized = _normalize_key(column)
    for alias, canonical in METRIC_ALIASES.items():
        if alias.replace(" ", "") in normalized:
            return canonical
    return column.strip()


def _unit_from_column(column: str) -> str | None:
    match = re.search(r"\(([^)]+)\)", column)
    if match:
        return _normalize_unit(match.group(1))
    return None


def _column_looks_metric(normalized_column: str) -> bool:
    metric_tokens = [
        "pce",
        "efficiency",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "rmse",
        "mae",
        "psnr",
        "ssim",
        "fid",
        "yield",
        "value",
        "score",
        "temperature",
        "wavelength",
        "bandgap",
        "stability",
    ]
    return any(token in normalized_column for token in metric_tokens)


def _first_numeric_value(rows: list[dict[str, Any]], column: str) -> float | None:
    for row in rows:
        value, _ = _coerce_number_and_unit(row.get(column))
        if value is not None:
            return value
    return None


def _find_column(normalized_columns: dict[str, str], tokens: list[str]) -> str | None:
    for column, normalized in normalized_columns.items():
        if any(token in normalized for token in tokens):
            return column
    return None


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    normalized = unit.strip()
    aliases = {
        "percent": "%",
        "hours": "h",
        "days": "day",
        "mA cm-2": "mA/cm2",
    }
    return aliases.get(normalized, normalized)


def _value_as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
