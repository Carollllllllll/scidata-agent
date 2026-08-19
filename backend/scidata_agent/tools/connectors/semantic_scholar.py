from __future__ import annotations

from typing import Any

from scidata_agent.agent.schemas import DiscoveredSource, SourceSearchRequest
from scidata_agent.tools.connectors.base import BaseConnector, compact_text, fetch_json


SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarConnector(BaseConnector):
    name = "semantic_scholar"
    supported_source_types = ("paper", "paper_metadata", "paper_search")

    def search(self, request: SourceSearchRequest) -> list[DiscoveredSource]:
        payload = fetch_json(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            params={
                "query": request.query,
                "limit": request.max_results,
                "fields": "title,abstract,authors,year,venue,url,openAccessPdf,externalIds,citationCount",
            },
        )
        return [
            semantic_scholar_paper_to_source(item, request)
            for item in payload.get("data", [])
            if isinstance(item, dict)
        ]


def semantic_scholar_paper_to_source(item: dict[str, Any], request: SourceSearchRequest) -> DiscoveredSource:
    title = compact_text(item.get("title")) or "Untitled Semantic Scholar paper"
    external_ids = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
    pdf = item.get("openAccessPdf") if isinstance(item.get("openAccessPdf"), dict) else {}
    authors = [author.get("name") for author in item.get("authors") or [] if isinstance(author, dict) and author.get("name")]
    metadata = {
        "provider": "semantic_scholar",
        "paper_id": item.get("paperId"),
        "doi": external_ids.get("DOI"),
        "arxiv_id": external_ids.get("ArXiv"),
        "year": item.get("year"),
        "authors": authors,
        "venue": item.get("venue"),
        "citation_count": item.get("citationCount"),
        "pdf_url": pdf.get("url"),
        "external_ids": external_ids,
    }
    return DiscoveredSource(
        title=title,
        source_type="paper_metadata",
        url=compact_text(item.get("url") or pdf.get("url")),
        query=request.query,
        description=compact_text(item.get("abstract")),
        reason=request.purpose or "Matched by Semantic Scholar paper search.",
        confidence=0.72,
        metadata={key: value for key, value in metadata.items() if value not in (None, "", [], {})},
    )
