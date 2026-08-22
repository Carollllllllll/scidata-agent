from scidata_agent.agent.schemas import DiscoveredSource, MultiSourceSearchPlan, SourceSearchRequest
from scidata_agent.tools.connectors.registry import execute_multi_source_search, merge_sources


def test_source_cluster_merges_provider_records_by_canonical_doi() -> None:
    crossref = DiscoveredSource(
        title="A Study of Scientific Data",
        source_type="paper_metadata",
        url="https://doi.org/10.1234/example",
        metadata={
            "provider": "crossref",
            "doi": "10.1234/example",
            "authors": ["Ada Lovelace"],
            "published": "2025-01-01",
        },
    )
    openalex = DiscoveredSource(
        title="A Study of Scientific Data",
        source_type="paper_metadata",
        url="https://openalex.org/W123",
        metadata={
            "provider": "openalex",
            "doi": "https://doi.org/10.1234/example",
            "cited_by_count": 42,
            "pdf_url": "https://example.org/paper.pdf",
        },
    )

    merged, added = merge_sources([crossref], [openalex])

    assert added == 0
    assert len(merged) == 1
    source = merged[0]
    assert source.source_cluster_id
    assert source.metadata["providers"] == ["crossref", "openalex"]
    assert source.metadata["source_ids"] == [crossref.source_id, openalex.source_id]
    assert source.metadata["cited_by_count"] == 42
    assert "https://openalex.org/W123" in source.metadata["alternate_urls"]
    assert "https://example.org/paper.pdf" in source.metadata["alternate_urls"]
    assert [item["provider"] for item in source.metadata["source_records"]] == ["crossref", "openalex"]


def test_source_cluster_merges_records_without_persistent_id_by_title_author_year() -> None:
    first = DiscoveredSource(
        title="Reliable Results for Scientific Agents",
        source_type="paper",
        metadata={
            "provider": "arxiv",
            "authors": ["Grace Hopper"],
            "publication_year": 2025,
        },
    )
    second = DiscoveredSource(
        title="Reliable Results for Scientific Agents",
        source_type="paper_metadata",
        metadata={
            "provider": "openalex",
            "authors": ["Grace Hopper"],
            "publication_year": 2025,
            "venue": "Demo Conference",
        },
    )

    merged, added = merge_sources([first], [second])

    assert added == 0
    assert len(merged) == 1
    assert merged[0].metadata["venue"] == "Demo Conference"
    assert merged[0].source_cluster_id == first.source_cluster_id


def test_source_cluster_preserves_conflicting_provider_values() -> None:
    first = DiscoveredSource(
        title="Conflicting Publication Metadata",
        source_type="paper_metadata",
        metadata={"provider": "crossref", "doi": "10.1234/conflict", "published": "2024-01-01"},
    )
    second = DiscoveredSource(
        title="Conflicting Publication Metadata",
        source_type="paper_metadata",
        metadata={"provider": "openalex", "doi": "10.1234/conflict", "published": "2025-01-01"},
    )

    merged, _ = merge_sources([first], [second])

    conflicts = merged[0].metadata["source_conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0]["field"] == "published"
    assert conflicts[0]["source_ids"] == [first.source_id, second.source_id]


def test_multi_source_search_clusters_duplicate_provider_results() -> None:
    plan = MultiSourceSearchPlan(
        research_goal="find one paper across providers",
        search_requests=[
            SourceSearchRequest(connector_name="crossref", source_type="paper_metadata", query="agent"),
            SourceSearchRequest(connector_name="openalex", source_type="paper_metadata", query="agent"),
        ],
    )

    def fake_search(request: SourceSearchRequest) -> list[DiscoveredSource]:
        provider = request.connector_name
        return [
            DiscoveredSource(
                title="One Shared Paper",
                source_type="paper_metadata",
                url="https://doi.org/10.1234/shared",
                metadata={"provider": provider, "doi": "10.1234/shared"},
            )
        ]

    sources, status = execute_multi_source_search(
        plan,
        searchers={"crossref": fake_search, "openalex": fake_search},
        max_workers=1,
    )

    assert status["status"] == "completed"
    assert status["added"] == 1
    assert len(sources) == 1
    assert sources[0].metadata["providers"] == ["crossref", "openalex"]
