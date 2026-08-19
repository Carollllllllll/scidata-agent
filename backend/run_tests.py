from __future__ import annotations

from tests.test_agent_mvp import (
    test_arxiv_connector_enriches_source_discovery_plan_without_network,
    test_arxiv_connector_does_not_expand_domain_terms_itself,
    test_arxiv_pdf_download_selection_without_network,
    test_arxiv_search_plan_can_execute_multiple_llm_queries_without_network,
    test_connector_registry_merges_duplicate_sources,
    test_dynamic_schema_planner_fails_explicitly_after_retries_in_official_mode,
    test_dynamic_schema_planner_accepts_numeric_field_examples,
    test_dynamic_schema_planner_retries_timeout_and_recovers,
    test_dynamic_schema_planner_rule_fallback_only_when_explicitly_allowed,
    test_dynamic_record_curation_prefers_arxiv_metadata_and_flags_review,
    test_extraction_limits_text_blocks_and_logs_progress,
    test_fallback_source_discovery_is_general_across_domains,
    test_missing_qwen_key_fails_in_official_mode,
    test_multi_source_ingestion_creates_research_blocks_and_downloads_small_tables,
    test_multi_source_search_plan_executes_all_connectors_without_network,
    test_openalex_connector_maps_work_payload,
    test_source_discovery_normalizes_llm_source_type_aliases,
    test_source_triage_downloads_open_pdfs_across_providers_with_budget,
    test_source_triage_keeps_paper_indexes_as_metadata_without_pdf,
    test_source_triage_caps_llm_selected_auto_resources_at_30,
    test_source_triage_selects_only_budgeted_arxiv_pdfs,
    test_source_triage_selects_small_tables_and_blocks_large_archives,
    test_source_triage_uses_llm_selection_and_rejects_off_topic_sources,
    test_task_planner_accepts_nested_dynamic_schema,
    test_qwen_extraction_timeout_retries_and_exports,
    test_question_only_source_discovery_mode_with_mock_client,
    test_quality_report_does_not_flag_different_experimental_contexts_as_conflicts,
    test_quality_report_detects_conflicts,
    test_quality_report_flags_weak_evidence_and_dimensionless_units,
    test_record_payload_repair_handles_non_scalar_metric_value,
    test_paper_survey_separates_baselines_from_proposed_method,
    test_qwen_agent_pipeline_with_mock_client,
    test_rule_fallback_is_explicitly_marked,
    test_section_aware_pipeline_exports_section_plan,
    test_section_builder_keeps_sections_within_source_file,
    test_zenodo_connector_maps_record_payload,
    test_github_connector_maps_repo_payload,
)
from tests.test_chart_extraction import (
    test_chart_locator_ignores_pdf_without_figures,
    test_chart_locator_renders_figure_png,
    test_chart_nodes_with_mock_vision,
    test_chart_pipeline_end_to_end_with_mock_vision,
    test_chart_validator_detects_axis_range_mismatch,
    test_chart_validator_flags_caption_unit_conflict,
    test_chart_validator_passes_consistent_extraction,
)
from tests.test_pdf_table_extraction import (
    test_parse_pdf_tables_extracts_structured_table,
    test_parse_pdf_tables_returns_empty_for_text_only_pdf,
    test_parse_sources_includes_pdf_tables,
    test_pdf_table_quality_filter_rejects_fragment,
)


def main() -> None:
    test_qwen_agent_pipeline_with_mock_client()
    print("qwen_agent_pipeline_with_mock_client: PASS")
    test_dynamic_record_curation_prefers_arxiv_metadata_and_flags_review()
    print("dynamic_record_curation_prefers_arxiv_metadata_and_flags_review: PASS")
    test_task_planner_accepts_nested_dynamic_schema()
    print("task_planner_accepts_nested_dynamic_schema: PASS")
    test_dynamic_schema_planner_accepts_numeric_field_examples()
    print("dynamic_schema_planner_accepts_numeric_field_examples: PASS")
    test_dynamic_schema_planner_retries_timeout_and_recovers()
    print("dynamic_schema_planner_retries_timeout_and_recovers: PASS")
    test_dynamic_schema_planner_fails_explicitly_after_retries_in_official_mode()
    print("dynamic_schema_planner_fails_explicitly_after_retries_in_official_mode: PASS")
    test_dynamic_schema_planner_rule_fallback_only_when_explicitly_allowed()
    print("dynamic_schema_planner_rule_fallback_only_when_explicitly_allowed: PASS")
    test_source_discovery_normalizes_llm_source_type_aliases()
    print("source_discovery_normalizes_llm_source_type_aliases: PASS")
    test_section_aware_pipeline_exports_section_plan()
    print("section_aware_pipeline_exports_section_plan: PASS")
    test_section_builder_keeps_sections_within_source_file()
    print("section_builder_keeps_sections_within_source_file: PASS")
    test_paper_survey_separates_baselines_from_proposed_method()
    print("paper_survey_separates_baselines_from_proposed_method: PASS")
    test_extraction_limits_text_blocks_and_logs_progress()
    print("extraction_limits_text_blocks_and_logs_progress: PASS")
    test_qwen_extraction_timeout_retries_and_exports()
    print("qwen_extraction_timeout_retries_and_exports: PASS")
    test_question_only_source_discovery_mode_with_mock_client()
    print("question_only_source_discovery_mode_with_mock_client: PASS")
    test_missing_qwen_key_fails_in_official_mode()
    print("missing_qwen_key_fails_in_official_mode: PASS")
    test_rule_fallback_is_explicitly_marked()
    print("rule_fallback_is_explicitly_marked: PASS")
    test_quality_report_flags_weak_evidence_and_dimensionless_units()
    print("quality_report_flags_weak_evidence_and_dimensionless_units: PASS")
    test_quality_report_detects_conflicts()
    print("quality_report_detects_conflicts: PASS")
    test_quality_report_does_not_flag_different_experimental_contexts_as_conflicts()
    print("quality_report_does_not_flag_different_experimental_contexts_as_conflicts: PASS")
    test_record_payload_repair_handles_non_scalar_metric_value()
    print("record_payload_repair_handles_non_scalar_metric_value: PASS")
    test_fallback_source_discovery_is_general_across_domains()
    print("fallback_source_discovery_is_general_across_domains: PASS")
    test_arxiv_connector_enriches_source_discovery_plan_without_network()
    print("arxiv_connector_enriches_source_discovery_plan_without_network: PASS")
    test_arxiv_connector_does_not_expand_domain_terms_itself()
    print("arxiv_connector_does_not_expand_domain_terms_itself: PASS")
    test_arxiv_search_plan_can_execute_multiple_llm_queries_without_network()
    print("arxiv_search_plan_can_execute_multiple_llm_queries_without_network: PASS")
    test_arxiv_pdf_download_selection_without_network()
    print("arxiv_pdf_download_selection_without_network: PASS")
    test_multi_source_search_plan_executes_all_connectors_without_network()
    print("multi_source_search_plan_executes_all_connectors_without_network: PASS")
    test_connector_registry_merges_duplicate_sources()
    print("connector_registry_merges_duplicate_sources: PASS")
    test_openalex_connector_maps_work_payload()
    print("openalex_connector_maps_work_payload: PASS")
    test_zenodo_connector_maps_record_payload()
    print("zenodo_connector_maps_record_payload: PASS")
    test_github_connector_maps_repo_payload()
    print("github_connector_maps_repo_payload: PASS")
    test_source_triage_selects_only_budgeted_arxiv_pdfs()
    print("source_triage_selects_only_budgeted_arxiv_pdfs: PASS")
    test_source_triage_downloads_open_pdfs_across_providers_with_budget()
    print("source_triage_downloads_open_pdfs_across_providers_with_budget: PASS")
    test_source_triage_uses_llm_selection_and_rejects_off_topic_sources()
    print("source_triage_uses_llm_selection_and_rejects_off_topic_sources: PASS")
    test_source_triage_caps_llm_selected_auto_resources_at_30()
    print("source_triage_caps_llm_selected_auto_resources_at_30: PASS")
    test_source_triage_keeps_paper_indexes_as_metadata_without_pdf()
    print("source_triage_keeps_paper_indexes_as_metadata_without_pdf: PASS")
    test_source_triage_selects_small_tables_and_blocks_large_archives()
    print("source_triage_selects_small_tables_and_blocks_large_archives: PASS")
    test_multi_source_ingestion_creates_research_blocks_and_downloads_small_tables()
    print("multi_source_ingestion_creates_research_blocks_and_downloads_small_tables: PASS")
    test_chart_locator_renders_figure_png()
    print("chart_locator_renders_figure_png: PASS")
    test_chart_locator_ignores_pdf_without_figures()
    print("chart_locator_ignores_pdf_without_figures: PASS")
    test_chart_nodes_with_mock_vision()
    print("chart_nodes_with_mock_vision: PASS")
    test_chart_validator_passes_consistent_extraction()
    print("chart_validator_passes_consistent_extraction: PASS")
    test_chart_validator_detects_axis_range_mismatch()
    print("chart_validator_detects_axis_range_mismatch: PASS")
    test_chart_validator_flags_caption_unit_conflict()
    print("chart_validator_flags_caption_unit_conflict: PASS")
    test_chart_pipeline_end_to_end_with_mock_vision()
    print("chart_pipeline_end_to_end_with_mock_vision: PASS")
    test_parse_pdf_tables_extracts_structured_table()
    print("parse_pdf_tables_extracts_structured_table: PASS")
    test_parse_pdf_tables_returns_empty_for_text_only_pdf()
    print("parse_pdf_tables_returns_empty_for_text_only_pdf: PASS")
    test_parse_sources_includes_pdf_tables()
    print("parse_sources_includes_pdf_tables: PASS")
    test_pdf_table_quality_filter_rejects_fragment()
    print("pdf_table_quality_filter_rejects_fragment: PASS")


if __name__ == "__main__":
    main()
