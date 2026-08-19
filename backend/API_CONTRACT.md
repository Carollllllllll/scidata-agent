# SciData Agent API Contract

The API adapter owns HTTP requests, uploads, task state, and downloads. The
scientific workflow remains inside `SciDataAgent`.

## Start a task

`POST /api/analyze` with `multipart/form-data`:

- `research_question`: required text
- `files`: optional repeated PDF/CSV/TSV/XLSX files
- `max_pdf_pages`: optional integer, default `8`
- `max_arxiv_papers`: optional integer
- `max_dynamic_text_blocks`: optional integer, default `20`
- `max_record_text_blocks`: optional integer, default `20`
- `max_figures_per_pdf`: optional integer, default `6`

The endpoint returns HTTP `202` immediately:

```json
{
  "task_id": "20260819_120000_123_abcd",
  "status": "queued",
  "status_url": "/api/tasks/20260819_120000_123_abcd",
  "events_url": "/api/tasks/20260819_120000_123_abcd/events"
}
```

`POST /api/discover` accepts `research_question` and creates a discovery task.

## Poll task state

`GET /api/tasks/{task_id}` returns `queued`, `running`, `completed`, or
`failed`. While running, the frontend can use `current_step`, `message`, and
`progress`. When completed, the same response contains the Agent result,
`summary`, `records`, `dynamic_records`, `source_catalog`, `quality_report`,
and `download_urls`.

`GET /api/tasks/{task_id}/events?tail=100` returns recent JSON monitoring events.

## Download results

`GET /api/tasks/{task_id}/export?format=csv` supports the server-side export
allowlist. The frontend should use the `download_urls` returned by the API and
must not use server filesystem paths.

Supported formats include `csv`, `json`, `quality_report`, `processing_log`,
`source_catalog`, `paper_survey`, `dynamic_schema`, `dynamic_records`,
`needs_review`, `chart_extractions`, `chart_validation`, `summary`, and
`final_report`.

## Frontend integration rules

1. Submit once and retain `task_id`.
2. Poll `status_url` every 1-3 seconds while status is `queued` or `running`.
3. Render `summary` and `quality_report` before loading large record lists.
4. Use `source_catalog` and record provenance for the evidence panel.
5. Use `download_urls` for downloads; do not depend on local Windows paths.
