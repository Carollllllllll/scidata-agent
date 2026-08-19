from __future__ import annotations

from collections import defaultdict

from scidata_agent.agent.schemas import ParsedSources, ScientificRecord, SourceSummary


def build_source_summaries(parsed_sources: ParsedSources, records: list[ScientificRecord]) -> list[SourceSummary]:
    summaries: dict[tuple[str, str, str], SourceSummary] = {}
    for block in parsed_sources.text_blocks:
        key = (block.source_file, block.source_path, block.source_type.value)
        if key not in summaries:
            summaries[key] = SourceSummary(
                source_file=block.source_file,
                source_path=block.source_path,
                source_type=block.source_type,
            )
        if block.page and block.page not in summaries[key].pages_processed:
            summaries[key].pages_processed.append(block.page)

    for table in parsed_sources.tables:
        key = (table.source_file, table.source_path, table.source_type.value)
        if key not in summaries:
            summaries[key] = SourceSummary(
                source_file=table.source_file,
                source_path=table.source_path,
                source_type=table.source_type,
            )
        summaries[key].tables_processed += 1

    counts = defaultdict(int)
    for record in records:
        for key in list(summaries):
            if key[0] == record.source_file and key[2] == record.source_type.value:
                counts[key] += 1
                break

    for key, count in counts.items():
        summaries[key].records_count = count
    for summary in summaries.values():
        summary.pages_processed = sorted(summary.pages_processed)
        if summary.records_count == 0:
            summary.notes.append("已解析该来源，但未抽取到符合当前 schema 的数值记录。")
    return list(summaries.values())

