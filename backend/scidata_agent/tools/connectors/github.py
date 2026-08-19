from __future__ import annotations

from typing import Any

from scidata_agent.agent.schemas import DiscoveredSource, SourceSearchRequest
from scidata_agent.tools.connectors.base import BaseConnector, compact_text, fetch_json


GITHUB_REPOSITORY_SEARCH_URL = "https://api.github.com/search/repositories"


class GitHubConnector(BaseConnector):
    name = "github"
    supported_source_types = ("repository", "dataset", "webpage")

    def search(self, request: SourceSearchRequest) -> list[DiscoveredSource]:
        payload = fetch_json(
            GITHUB_REPOSITORY_SEARCH_URL,
            params={"q": request.query, "per_page": request.max_results, "sort": "stars", "order": "desc"},
            headers={"X-GitHub-Api-Version": "2022-11-28"},
        )
        return [github_repo_to_source(item, request) for item in payload.get("items", []) if isinstance(item, dict)]


def github_repo_to_source(item: dict[str, Any], request: SourceSearchRequest) -> DiscoveredSource:
    owner = item.get("owner") if isinstance(item.get("owner"), dict) else {}
    metadata = {
        "provider": "github",
        "repo_id": item.get("id"),
        "full_name": item.get("full_name"),
        "owner": owner.get("login"),
        "language": item.get("language"),
        "stars": item.get("stargazers_count"),
        "forks": item.get("forks_count"),
        "open_issues": item.get("open_issues_count"),
        "license": item.get("license", {}).get("spdx_id") if isinstance(item.get("license"), dict) else None,
        "updated_at": item.get("updated_at"),
        "pushed_at": item.get("pushed_at"),
        "topics": item.get("topics"),
    }
    return DiscoveredSource(
        title=compact_text(item.get("full_name") or item.get("name")) or "Untitled GitHub repository",
        source_type="repository",
        url=compact_text(item.get("html_url")),
        query=request.query,
        description=compact_text(item.get("description")),
        reason=request.purpose or "Matched by GitHub repository search.",
        confidence=0.64,
        metadata={key: value for key, value in metadata.items() if value not in (None, "", [], {})},
    )
