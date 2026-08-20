from __future__ import annotations

import re

from scidata_agent.agent.schemas import DynamicRecord, ScientificRecord


METRIC_CANONICAL = {
    "power conversion efficiency": "PCE",
    "photoelectric conversion efficiency": "PCE",
    "光电转换效率": "PCE",
    "pce": "PCE",
    "efficiency": "efficiency",
    "acc": "accuracy",
    "accuracy": "accuracy",
    "f1-score": "F1",
    "f1": "F1",
    "root mean square error": "RMSE",
    "rmse": "RMSE",
    "mean absolute error": "MAE",
    "mae": "MAE",
    "fid": "FID",
    "kid": "KID",
    "ssim": "SSIM",
    "lpips": "LPIPS",
    "psnr": "PSNR",
    "clip score": "CLIP score",
    "absorption": "absorption wavelength",
    "absorption wavelength": "absorption wavelength",
    "wavelength": "wavelength",
    "stability": "stability",
    "bandgap": "bandgap",
    "number of parameters": "number of parameters",
    "parameters": "number of parameters",
}

UNIT_CANONICAL = {
    "percent": "%",
    "percentage": "%",
    "pct": "%",
    "％": "%",
    "hours": "h",
    "hour": "h",
    "hrs": "h",
    "days": "day",
    "℃": "degC",
    "°c": "degC",
    "c": "degC",
    "mA cm-2": "mA/cm2",
    "ma cm-2": "mA/cm2",
    "ma/cm2": "mA/cm2",
}

DIMENSIONLESS = {
    "accuracy",
    "F1",
    "FID",
    "KID",
    "SSIM",
    "LPIPS",
    "CLIP score",
    "number of parameters",
}

STRICT_NUMERIC_METRICS = {
    "PCE",
    "efficiency",
    "accuracy",
    "F1",
    "RMSE",
    "MAE",
    "FID",
    "KID",
    "SSIM",
    "LPIPS",
    "PSNR",
    "CLIP score",
    "absorption wavelength",
    "wavelength",
    "stability",
    "bandgap",
    "number of parameters",
    "Latency",
    "latency",
    "inference_time_ms",
    "fps",
    "memory",
    "model size",
    "MSE",
    "Preprocessing time",
    "KM Blending time",
    "Delta E CIE2000",
    "DISTS",
    "User Study",
    "short-sleeved synthesis accuracy",
    "normal output rate",
}

_DYNAMIC_METRIC_PAIRS = (
    ("metric_name", "metric_value"),
    ("metric_name", "value"),
    ("primary_metric", "reported_score"),
    ("metric", "score"),
    ("metric_name", "score"),
    ("metric", "value"),
)

_KNOWN_METRIC_FIELDS = {
    "fid",
    "rfid",
    "kid",
    "ssim",
    "psnr",
    "lpips",
    "accuracy",
    "pce",
    "f1",
    "mae",
    "rmse",
    "latency",
    "fps",
}


def scientific_records_from_dynamic(records: list[DynamicRecord]) -> list[ScientificRecord]:
    """Reuse the schema-driven extraction pass for the strict metric export.

    This avoids sending the same PDF block to a second LLM extractor.  The
    deterministic adapter only emits numeric metric/value pairs; if none are
    present the caller can still fall back to the dedicated metric extractor.
    """
    scientific: list[ScientificRecord] = []
    for record in records:
        fields = record.fields
        pairs: list[tuple[str, object]] = []
        for metric_field, value_field in _DYNAMIC_METRIC_PAIRS:
            metric_name = fields.get(metric_field)
            if metric_name not in (None, "", []) and fields.get(value_field) not in (None, "", []):
                pairs.append((str(metric_name), fields[value_field]))
        for name, value in fields.items():
            if _norm_key(name) in _KNOWN_METRIC_FIELDS and value not in (None, "", []):
                pairs.append((name, value))

        seen: set[tuple[str, float]] = set()
        for metric_name, raw_value in pairs:
            metric_value = _dynamic_numeric_value(raw_value)
            if metric_value is None:
                continue
            key = (_norm_key(metric_name), metric_value)
            if key in seen:
                continue
            seen.add(key)
            scientific.append(
                ScientificRecord(
                    paper_title=record.paper_title,
                    material=_first_dynamic_text(fields, "material", "composition", "entity"),
                    method=_first_dynamic_text(
                        fields,
                        "method",
                        "method_name",
                        "model_name",
                        "variant_name",
                        "proposed_solution",
                    ),
                    metric_name=metric_name,
                    metric_value=metric_value,
                    unit=_dynamic_metric_unit(metric_name, raw_value, fields),
                    condition=_first_dynamic_text(fields, "condition", "test_condition", "dataset_name", "dataset", "task"),
                    source_file=record.source_file,
                    source_type=record.source_type,
                    page=record.page,
                    evidence_text=record.evidence_text,
                    confidence=record.confidence,
                    warnings=list(record.warnings),
                    raw={
                        **record.raw,
                        "derived_from_dynamic_record_id": record.record_id,
                        "derived_from_dynamic_table": record.table_name,
                    },
                )
            )
    return scientific


def _dynamic_numeric_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))(?:\s*%)?\s*", value.replace(",", ""))
    return float(match.group(1)) if match else None


def _first_dynamic_text(fields: dict[str, object], *names: str) -> str | None:
    for name in names:
        value = fields.get(name)
        if value not in (None, "", []):
            return " ".join(str(value).split())
    return None


def _dynamic_metric_unit(metric_name: str, raw_value: object, fields: dict[str, object]) -> str | None:
    explicit = fields.get("unit") or fields.get("metric_unit")
    if explicit not in (None, "", []):
        return str(explicit)
    if isinstance(raw_value, str) and "%" in raw_value:
        return "%"
    if _norm_key(metric_name) in {"pce", "power conversion efficiency", "efficiency"}:
        return "%"
    return None


def normalize_records(records: list[ScientificRecord]) -> list[ScientificRecord]:
    normalized = [_normalize_record(record) for record in records]
    strict_records = []
    for record in normalized:
        if _is_strict_metric_record(record):
            strict_records.append(record)
        else:
            record.raw["filtered_from_result_csv"] = True
            record.warnings.append(
                "Filtered from strict metric output because the record is not a numeric metric; "
                "use dynamic tables for non-metric facts."
            )
    return _deduplicate(strict_records)


def _normalize_record(record: ScientificRecord) -> ScientificRecord:
    metric_key = _norm_key(record.metric_name)
    record.metric_name = METRIC_CANONICAL.get(metric_key, record.metric_name.strip())

    if record.unit:
        unit_key = record.unit.strip().lower()
        record.unit = UNIT_CANONICAL.get(unit_key, record.unit.strip())

    if not record.unit and record.metric_name in DIMENSIONLESS:
        record.unit = "dimensionless"

    if record.evidence_text:
        record.evidence_text = " ".join(record.evidence_text.split())

    if record.material:
        record.material = " ".join(record.material.split())
    if record.method:
        record.method = " ".join(record.method.split())
    if record.condition:
        record.condition = " ".join(record.condition.split())

    return record


def _deduplicate(records: list[ScientificRecord]) -> list[ScientificRecord]:
    """Deduplicate records while keeping measurements from different papers separate.

    The deduplication key uses the source file (paper identity) so that two
    different papers reporting the same metric value are kept as distinct
    records. Within a single paper, records that share the semantic identity
    (material/method/metric/value/unit/condition) are merged into one entry;
    the first record wins and duplicates are dropped.
    """
    seen: set[tuple] = set()
    deduped: list[ScientificRecord] = []
    for record in records:
        key = (
            record.source_file.lower(),
            _norm_key(record.material or ""),
            _norm_key(record.method or ""),
            record.metric_name.lower(),
            round(record.metric_value, 8) if record.metric_value is not None else None,
            (record.unit or "").lower(),
            _norm_key(record.condition or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _is_strict_metric_record(record: ScientificRecord) -> bool:
    if record.metric_value is None:
        return False
    metric_name = record.metric_name.strip()
    if metric_name in STRICT_NUMERIC_METRICS:
        return True
    metric_key = metric_name.lower()
    return any(
        token in metric_key
        for token in [
            "fid",
            "kid",
            "ssim",
            "lpips",
            "clip",
            "latency",
            "fps",
            "time",
            "memory",
            "size",
            "parameter",
            "accuracy",
            "rate",
            "mse",
            "psnr",
            "dists",
        ]
    )


def _norm_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
