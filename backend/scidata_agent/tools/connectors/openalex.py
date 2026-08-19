from __future__ import annotations

from typing import Any

from scidata_agent.agent.schemas import DiscoveredSource, SourceSearchRequest
from scidata_agent.tools.connectors.base import BaseConnector, compact_text, fetch_json


OPENALEX_WORKS_URL = "https://api.openalex.org/works"


class OpenAlexConnector(BaseConnector):
    name = "openalex"
    supported_source_types = ("paper", "paper_metadata", "paper_search")

    def search(self, request: SourceSearchRequest) -> list[DiscoveredSource]:
        payload = fetch_json(
            OPENALEX_WORKS_URL,
            params={"search": request.query, "per-page": request.max_results},
        )
        return [openalex_work_to_source(item, request) for item in payload.get("results", []) if isinstance(item, dict)]


def openalex_work_to_source(item: dict[str, Any], request: SourceSearchRequest) -> DiscoveredSource:
    title = compact_text(item.get("title") or item.get("display_name")) or "Untitled OpenAlex work"
    primary_location = item.get("primary_location") if isinstance(item.get("primary_location"), dict) else {}
    source = primary_location.get("source") if isinstance(primary_location.get("source"), dict) else {}
    open_access = item.get("open_access") if isinstance(item.get("open_access"), dict) else {}
    authors = []
    for authorship in item.get("authorships") or []:
        author = authorship.get("author") if isinstance(authorship, dict) else None
        name = author.get("display_name") if isinstance(author, dict) else None
        if name:
            authors.append(name)
    doi = compact_text(item.get("doi"))
    pdf_url = compact_text(primary_location.get("pdf_url") or open_access.get("oa_url"))
    metadata = {
        "provider": "openalex",
        "openalex_id": item.get("id"),
        "doi": doi,
        "publication_year": item.get("publication_year"),
        "authors": authors,
        "venue": source.get("display_name"),
        "cited_by_count": item.get("cited_by_count"),
        "open_access_url": open_access.get("oa_url"),
        "pdf_url": pdf_url,
    }
    return DiscoveredSource(
        title=title,
        source_type="paper_metadata",
        url=compact_text(item.get("doi") or item.get("id")),
        query=request.query,
        description=compact_text(item.get("abstract")),
        reason=request.purpose or "Matched by OpenAlex work search.",
        confidence=0.72,
        metadata={key: value for key, value in metadata.items() if value not in (None, "", [])},
    )
