from __future__ import annotations

import re

from scidata_agent.agent.schemas import ScientificRecord


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
