from __future__ import annotations

from typing import Any

from scidata_agent.agent.schemas import DiscoveredSource, SourceSearchRequest
from scidata_agent.tools.connectors.base import BaseConnector, compact_text, fetch_json


FIGSHARE_ARTICLES_URL = "https://api.figshare.com/v2/articles"
FIGSHARE_SEARCH_URL = "https://api.figshare.com/v2/articles/search"
FIGSHARE_ARTICLE_URL = "https://api.figshare.com/v2/articles/{article_id}"


class FigshareConnector(BaseConnector):
    name = "figshare"
    supported_source_types = ("dataset", "open_database", "supplementary_material", "table", "image")

    def search(self, request: SourceSearchRequest) -> list[DiscoveredSource]:
        payload = _fetch_figshare_search(request)
        sources: list[DiscoveredSource] = []
        for item in payload if isinstance(payload, list) else []:
            if not isinstance(item, dict):
                continue
            detail = item
            article_id = item.get("id")
            if article_id:
                try:
                    detail_payload = fetch_json(FIGSHARE_ARTICLE_URL.format(article_id=article_id))
                    if isinstance(detail_payload, dict):
                        detail = {**item, **detail_payload}
                except Exception:
                    detail = item
            sources.append(figshare_article_to_source(detail, request))
        return sources


def _fetch_figshare_search(request: SourceSearchRequest) -> Any:
    params = {"search_for": request.query, "page_size": request.max_results}
    return fetch_json(
        FIGSHARE_SEARCH_URL,
        method="POST",
        json_body=params,
    )


def figshare_article_to_source(item: dict[str, Any], request: SourceSearchRequest) -> DiscoveredSource:
    files = []
    for file_item in item.get("files") or []:
        if isinstance(file_item, dict):
            files.append(
                {
                    "name": file_item.get("name"),
                    "size": file_item.get("size"),
                    "download_url": file_item.get("download_url"),
                    "computed_md5": file_item.get("computed_md5"),
                }
            )
    authors = [
        author.get("full_name")
        for author in item.get("authors") or []
        if isinstance(author, dict) and author.get("full_name")
    ]
    metadata = {
        "provider": "figshare",
        "article_id": item.get("id"),
        "doi": item.get("doi"),
        "defined_type": item.get("defined_type_name") or item.get("defined_type"),
        "published_date": item.get("published_date"),
        "modified_date": item.get("modified_date"),
        "authors": authors,
        "tags": item.get("tags"),
        "files": files,
    }
    return DiscoveredSource(
        title=compact_text(item.get("title")) or "Untitled Figshare article",
        source_type=request.source_type if request.source_type in {"dataset", "open_database", "supplementary_material", "table", "image"} else "dataset",
        url=compact_text(item.get("url") or item.get("figshare_url")),
        query=request.query,
        description=compact_text(item.get("description")),
        reason=request.purpose or "Matched by Figshare article search.",
        confidence=0.68,
        metadata={key: value for key, value in metadata.items() if value not in (None, "", [], {})},
    )
