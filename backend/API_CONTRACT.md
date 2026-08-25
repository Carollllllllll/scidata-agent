# SciData Agent API Contract

The API adapter owns HTTP requests, uploads, task state, public asset URLs, and
downloads. The scientific workflow remains inside `SciDataAgent`. API version
`0.3.0` uses one stable task envelope for queued, running, completed, partial,
failed, and cancelled states.

## Health and task history

- `GET /api/health`: service, model configuration flag, model pools, and Agent stages.
- `GET /api/tasks?limit=20&status=completed`: recent persisted task summaries.
  Use `status=partial` to find tasks that produced inspectable outputs but did
  not satisfy the deterministic coverage audit.

Health data never returns the API key.

## Start a task

`POST /api/analyze` with `multipart/form-data`:

- `research_question`: required text, 3-4000 characters
- `files`: optional repeated PDF/CSV/TSV/XLSX/XLS files
- `max_pdf_pages`: optional non-negative integer, default `null`; omitted or `0` means all pages
- `max_arxiv_papers`: optional non-negative integer; omitted or `0` means all selected PDFs
- `max_auto_resources`: optional non-negative integer; omitted or `0` means all LLM-selected resources
- `enable_live_search`: boolean, default `true`
- `auto_download_sources`: boolean, default `true`
- `max_dynamic_text_blocks`: optional non-negative integer; omitted or `0` means all ranked blocks
- `max_record_text_blocks`: optional non-negative integer; omitted or `0` means all ranked blocks
- `max_figures_per_pdf`: optional non-negative integer; omitted or `0` means all detected figures

Scientific-data limits are opt-in. The API does not silently truncate discovered
sources, selected downloads, PDF pages, text blocks, tables, or figures. Positive
values are explicit operator limits and are recorded in the task state and
monitor log. Upload count, upload bytes, queue size, and per-file download byte
limits remain service-safety guardrails.
- `max_pdf_parse_workers`: optional integer 1-16; overrides `SCIDATA_PDF_PARSE_MAX_WORKERS`
- `max_chart_workers`: optional integer 1-16; overrides `SCIDATA_CHART_MAX_WORKERS`
- `max_text_extraction_workers`: optional integer 1-16; overrides `SCIDATA_TEXT_EXTRACTION_MAX_WORKERS`
- `max_table_extraction_workers`: optional integer 1-16; overrides `SCIDATA_TABLE_EXTRACTION_MAX_WORKERS`
- `reuse_dynamic_records_for_metrics`: boolean, default `true`

The default guardrails are 20 uploaded files, 50 MiB per file, and 200 MiB per
request. They can be changed with `SCIDATA_MAX_UPLOAD_FILES`,
`SCIDATA_MAX_UPLOAD_BYTES`, and `SCIDATA_MAX_UPLOAD_TOTAL_BYTES`.

The endpoint returns HTTP `202` immediately:

```json
{
  "task_id": "20260820_120000_123_abcd",
  "status": "queued",
  "status_url": "/api/tasks/20260820_120000_123_abcd",
  "events_url": "/api/tasks/20260820_120000_123_abcd/events"
}
```

`POST /api/discover` accepts `research_question` and creates a source-discovery
task without local files.

## Poll task state

`GET /api/tasks/{task_id}` always returns the same envelope:

```json
{
  "task_id": "20260820_120000_123_abcd",
  "status": "running",
  "research_question": "...",
  "current_step": "source_parsing",
  "message": "...",
  "progress": {"current": 2, "total": 5},
  "created_at": "...",
  "updated_at": "...",
  "error": null,
  "uploads": [],
  "event": {},
  "result": null,
  "summary": null,
  "quality_report": null,
  "download_urls": {}
}
```

On completion or partial completion, `result` contains the Agent payload. A
`partial` task is terminal for polling, but its coverage report and processing
log must be inspected before treating the research result as complete.
Important frontend fields
include:

- `dynamic_extraction_plan`
- `dynamic_records` (cleaned canonical records)
- `dynamic_records_raw`
- `needs_review_records`
- `review_queue` (risk-ranked review items with stable `review_id` values;
  items may represent records, figures, cross-modal checks, source conflicts,
  or evidence-coverage gaps)
- `source_catalog` and `connector_status`
- `evidence_traces` (record-to-source/page/section/table/figure provenance)
- `coverage_report` (deterministic stop decision, coverage score, structured
  evidence gaps, and recommended next actions)
- `figures`, `chart_extractions`, and `chart_validations`
- `quality_report`

`cross_modal_checks` and the `cross_modal_validation` export report whether
same-page text, table, and figure evidence numerically corroborate one another.
Qualitative figures and missing comparison material are reported as
`not_comparable`, not as extraction failures.

Conflict entries in `quality_report.conflicts` include the aligned context,
the fields used for comparison, and a resolution marker. The current policy is
`preserve_all`: conflicting values from different sources are retained and
shown for review; the backend never silently chooses one value.

The top-level `summary` and `quality_report` mirror the corresponding result
fields for fast rendering. `GET /api/tasks/{task_id}/events?tail=80` reads a
bounded window from the end of the monitor file and omits event `data` by
default. Add `include_data=true` only for detailed diagnostics.

Lifecycle and review mutations:

- `POST /api/tasks/{task_id}/cancel`: cancel a queued task immediately, or request
  cooperative cancellation of a running task at the next safe pipeline checkpoint.
- `POST /api/tasks/{task_id}/retry`: copy safe uploaded inputs and create a new task.
- `POST /api/tasks/{task_id}/resume`: reuse the same task directory and resume from
  the latest valid Agent checkpoint; successful stages and downloads are reused.
- `POST /api/tasks/{task_id}/reviews/{review_id}`: persist `approved`,
  `needs_changes`, or `rejected` with an optional note. The endpoint accepts
  legacy record IDs too. Decisions are stored separately from extraction
  output and include the reviewed subject type and ID.

## Assets and downloads

Internal filesystem paths are removed recursively. Paths belonging to the task
are exposed as scoped URLs such as:

```text
/api/tasks/{task_id}/assets/upload/{relative_path}
/api/tasks/{task_id}/assets/output/{relative_path}
```

The asset endpoint rejects absolute paths and traversal outside the task roots.
The frontend should only use returned `asset_url`, `source_url`, and `image_url`
fields.

`GET /api/tasks/{task_id}/export?format=review_queue` downloads the structured
human-review queue. `GET /api/tasks/{task_id}/export?format=csv` uses a
server-side allowlist. Use
the returned `download_urls`; never reconstruct a server path in the browser.

## Error contract

HTTP errors use FastAPI's `detail` envelope with a stable code and readable
message:

```json
{
  "detail": {
    "code": "TASK_NOT_FOUND",
    "message": "任务不存在。"
  }
}
```

Important statuses are `400` for unsupported export formats, `404` for missing
tasks/assets or not-yet-generated exports, `413` for upload limits, `415` for
unsupported files, and `422` for invalid form parameters.

Deployments may set `SCIDATA_API_TOKEN` to require a Bearer token on API routes
except health checks. Mutation requests are rate-limited, and pending task count
is bounded by `SCIDATA_MAX_PENDING_TASKS`.

## Frontend integration rules

1. Submit once and retain `task_id` in the URL.
2. Poll the task every 1-3 seconds while status is `queued` or `running`.
3. Stop polling on `completed`, `partial`, `failed`, or `cancelled`.
4. Render dynamic columns from `dynamic_extraction_plan`; do not hard-code a domain schema.
5. Keep source lifecycle status separate from record quality status.
6. When `coverage_report.decision` is `continue`, show `coverage_report.gaps`
   and their recommended actions instead of treating the result as complete.
6. Use `download_urls` and public asset URLs only.
7. Do not fabricate placeholder scientific results for empty responses.
8. Render `evidence_traces` as the record-to-source audit trail. An
   `unresolved` locator means the backend did not have enough evidence to claim
   a precise page, section, table, or figure location.
