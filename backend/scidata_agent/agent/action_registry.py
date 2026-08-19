from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionCapability:
    """Static capability metadata used to validate artifact routing."""

    action: str
    description: str
    artifact_types: frozenset[str] = frozenset()
    requires_local_path: bool = False
    global_action: bool = False


_GLOBAL_ARTIFACT_ACTIONS = frozenset({"search_more", "validate_evidence", "stop"})

ACTION_CAPABILITIES: dict[str, ActionCapability] = {
    "read_metadata": ActionCapability(
        action="read_metadata",
        description="Read catalog metadata without downloading the artifact.",
    ),
    "parse_pdf_text": ActionCapability(
        action="parse_pdf_text",
        description="Extract text blocks from a PDF artifact.",
        artifact_types=frozenset({"pdf", "supplementary_pdf"}),
        requires_local_path=True,
    ),
    "parse_pdf_sections": ActionCapability(
        action="parse_pdf_sections",
        description="Extract PDF text and create LLM-interpreted section blocks.",
        artifact_types=frozenset({"pdf", "supplementary_pdf"}),
        requires_local_path=True,
    ),
    "parse_table": ActionCapability(
        action="parse_table",
        description="Extract a PDF table with TATR/pdfplumber or read a spreadsheet table.",
        artifact_types=frozenset({"pdf", "supplementary_pdf", "csv", "tsv", "xlsx", "json", "xml"}),
        requires_local_path=True,
    ),
    "parse_figure": ActionCapability(
        action="parse_figure",
        description="Locate and interpret quantitative figure/chart evidence with VL.",
        artifact_types=frozenset({"pdf", "supplementary_pdf", "image"}),
        requires_local_path=True,
    ),
    "parse_html": ActionCapability(
        action="parse_html",
        description="Read a local HTML artifact as text evidence.",
        artifact_types=frozenset({"html", "landing_page"}),
        requires_local_path=True,
    ),
    "parse_csv": ActionCapability(
        action="parse_csv",
        description="Read a CSV or TSV artifact as a structured table.",
        artifact_types=frozenset({"csv", "tsv"}),
        requires_local_path=True,
    ),
    "read_readme": ActionCapability(
        action="read_readme",
        description="Read a local README or code artifact as text evidence.",
        artifact_types=frozenset({"readme", "code_archive"}),
        requires_local_path=True,
    ),
    "read_file_manifest": ActionCapability(
        action="read_file_manifest",
        description="Inspect the file manifest already attached to a source artifact.",
        artifact_types=frozenset({"file_manifest", "landing_page", "code_archive"}),
    ),
    "search_more": ActionCapability(
        action="search_more",
        description="Request another discovery/search iteration.",
        global_action=True,
    ),
    "validate_evidence": ActionCapability(
        action="validate_evidence",
        description="Run the existing evidence and quality validation node.",
        global_action=True,
    ),
    "stop": ActionCapability(
        action="stop",
        description="Stop the current planner iteration.",
        global_action=True,
    ),
}


def get_action_capability(action: str) -> ActionCapability:
    try:
        return ACTION_CAPABILITIES[action]
    except KeyError as exc:
        raise ValueError(f"Unsupported artifact action: {action!r}") from exc


def list_action_capabilities() -> list[ActionCapability]:
    return list(ACTION_CAPABILITIES.values())


def is_global_action(action: str) -> bool:
    return action in _GLOBAL_ARTIFACT_ACTIONS


def artifact_type_supported(action: str, artifact_type: str) -> bool:
    capability = get_action_capability(action)
    if capability.global_action or action == "read_metadata":
        return True
    return artifact_type in capability.artifact_types
