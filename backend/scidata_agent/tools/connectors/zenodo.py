from __future__ import annotations

from typing import Any

from scidata_agent.agent.schemas import DiscoveredSource, SourceSearchRequest
from scidata_agent.tools.connectors.base import BaseConnector, compact_text, fetch_json


ZENODO_RECORDS_URL = "https://zenodo.org/api/records"


class ZenodoConnector(BaseConnector):
    name = "zenodo"
    supported_source_types = ("dataset", "open_database", "supplementary_material", "table", "image")

    def search(self, request: SourceSearchRequest) -> list[DiscoveredSource]:
        # Zenodo permits at most 25 records per anonymous Records API request.
        # The connector does not attach a token, so clamp planner-provided limits.
        page_size = max(1, min(int(request.max_results), 25))
        payload = fetch_json(ZENODO_RECORDS_URL, params={"q": request.query, "size": page_size})
        hits = payload.get("hits") if isinstance(payload.get("hits"), dict) else {}
        return [zenodo_record_to_source(item, request) for item in hits.get("hits", []) if isinstance(item, dict)]


def zenodo_record_to_source(item: dict[str, Any], request: SourceSearchRequest) -> DiscoveredSource:
    metadata_payload = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    creators = [
        creator.get("name")
        for creator in metadata_payload.get("creators") or []
        if isinstance(creator, dict) and creator.get("name")
    ]
    files = item.get("files") or []
    file_links = []
    for file_item in files:
        if not isinstance(file_item, dict):
            continue
        links = file_item.get("links") if isinstance(file_item.get("links"), dict) else {}
        file_links.append(
            {
                "key": file_item.get("key"),
                "type": file_item.get("type"),
                "size": file_item.get("size"),
                "url": links.get("self") or links.get("download"),
            }
        )
    doi = compact_text(metadata_payload.get("doi") or item.get("doi"))
    metadata = {
        "provider": "zenodo",
        "record_id": item.get("id"),
        "doi": doi,
        "publication_date": metadata_payload.get("publication_date"),
        "resource_type": metadata_payload.get("resource_type"),
        "creators": creators,
        "keywords": metadata_payload.get("keywords"),
        "files": file_links,
    }
    return DiscoveredSource(
        title=compact_text(metadata_payload.get("title")) or "Untitled Zenodo record",
        source_type=request.source_type if request.source_type in {"dataset", "open_database", "supplementary_material", "table", "image"} else "dataset",
        url=compact_text(item.get("links", {}).get("html") if isinstance(item.get("links"), dict) else None),
        query=request.query,
        description=compact_text(metadata_payload.get("description")),
        reason=request.purpose or "Matched by Zenodo records search.",
        confidence=0.7,
        metadata={key: value for key, value in metadata.items() if value not in (None, "", [], {})},
    )
