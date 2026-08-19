DEFAULT_TARGET_FIELDS = [
    "paper_title",
    "material",
    "method",
    "metric_name",
    "metric_value",
    "unit",
    "condition",
    "source_file",
    "source_type",
    "page",
    "evidence_text",
    "confidence",
]


FIELD_SCHEMA = [
    {"field": "paper_title", "type": "string|null", "description": "Title of the paper or data source, if available."},
    {"field": "material", "type": "string|null", "description": "Research object, material, model, dataset, compound, sample, or entity name."},
    {"field": "method", "type": "string|null", "description": "Experiment, preparation, training, measurement, or analysis method."},
    {"field": "metric_name", "type": "string", "description": "Normalized scientific metric name, such as PCE, RMSE, FID, SSIM, stability, or absorption wavelength."},
    {"field": "metric_value", "type": "number|null", "description": "Numeric value of the metric. Use null if the value cannot be reliably parsed."},
    {"field": "unit", "type": "string|null", "description": "Metric unit. Use dimensionless for unitless metrics such as FID, SSIM, LPIPS, or accuracy."},
    {"field": "condition", "type": "string|null", "description": "Experimental condition, dataset, test setting, sample context, or other relevant context."},
    {"field": "source_file", "type": "string", "description": "Source file name."},
    {"field": "source_type", "type": "string", "description": "Source type, e.g. pdf_text, csv, excel."},
    {"field": "page", "type": "integer|null", "description": "PDF page number. Table sources can use null and record row_index in raw."},
    {"field": "evidence_text", "type": "string|null", "description": "Original evidence text, table row text, or source snippet supporting the record."},
    {"field": "confidence", "type": "number", "description": "Extraction confidence between 0 and 1."},
]
