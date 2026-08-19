from __future__ import annotations

from typing import Any

from scidata_agent.agent.schemas import DiscoveredSource, SourceSearchRequest
from scidata_agent.tools.connectors.base import BaseConnector, compact_text, fetch_json, first_text, pick_date


CROSSREF_WORKS_URL = "https://api.crossref.org/works"


class CrossrefConnector(BaseConnector):
    name = "crossref"
    supported_source_types = ("paper", "paper_metadata", "paper_search")

    def search(self, request: SourceSearchRequest) -> list[DiscoveredSource]:
        payload = fetch_json(CROSSREF_WORKS_URL, params={"query": request.query, "rows": request.max_results})
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        return [crossref_work_to_source(item, request) for item in message.get("items", []) if isinstance(item, dict)]


def crossref_work_to_source(item: dict[str, Any], request: SourceSearchRequest) -> DiscoveredSource:
    title = first_text(item.get("title")) or "Untitled Crossref work"
    authors = []
    for author in item.get("author") or []:
        if not isinstance(author, dict):
            continue
        name = " ".join(part for part in [author.get("given"), author.get("family")] if part)
        if name:
            authors.append(name)
    doi = compact_text(item.get("DOI"))
    metadata = {
        "provider": "crossref",
        "doi": doi,
        "type": item.get("type"),
        "authors": authors,
        "venue": first_text(item.get("container-title")),
        "publisher": item.get("publisher"),
        "published": pick_date(item.get("published-print") or item.get("published-online") or item.get("published")),
        "reference_count": item.get("reference-count"),
        "is_referenced_by_count": item.get("is-referenced-by-count"),
    }
    return DiscoveredSource(
        title=title,
        source_type="paper_metadata",
        url=compact_text(item.get("URL") or (f"https://doi.org/{doi}" if doi else None)),
        query=request.query,
        description=compact_text(item.get("abstract") or item.get("subject")),
        reason=request.purpose or "Matched by Crossref metadata search.",
        confidence=0.68,
        metadata={key: value for key, value in metadata.items() if value not in (None, "", [])},
    )
