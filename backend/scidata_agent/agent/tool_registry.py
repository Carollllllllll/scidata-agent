from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from scidata_agent.agent.action_registry import list_action_capabilities
from scidata_agent.agent.tool_protocol import ToolCall, ToolSpec


class ToolRegistry:
    """Registry of discoverable tool contracts used by the Agent runtime."""

    def __init__(self, specs: Iterable[ToolSpec] | None = None) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._specs:
            raise ValueError(f"Tool already registered: {spec.name!r}")
        self._specs[spec.name] = spec
        return spec

    def replace(self, spec: ToolSpec) -> ToolSpec:
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def require(self, name: str) -> ToolSpec:
        spec = self.get(name)
        if spec is None:
            raise ValueError(f"Unknown tool: {name!r}")
        return spec

    def list(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def describe(self, *, artifact_type: str | None = None) -> list[dict[str, Any]]:
        specs = self.list()
        if artifact_type is not None:
            specs = [
                spec
                for spec in specs
                if spec.global_action
                or not spec.artifact_types
                or artifact_type in spec.artifact_types
            ]
        return [spec.model_dump(mode="json") for spec in specs]

    def validate_call(self, call: ToolCall) -> list[str]:
        errors: list[str] = []
        spec = self.get(call.tool_name)
        if spec is None:
            return [f"Unknown tool: {call.tool_name!r}"]
        if not isinstance(call.arguments, dict):
            errors.append("Tool arguments must be a JSON object.")
        if spec.requires_artifact and not str(call.arguments.get("artifact_id") or "").strip():
            errors.append(f"Tool {call.tool_name!r} requires arguments.artifact_id.")
        required = spec.input_schema.get("required", [])
        properties = call.arguments
        for field in required:
            if field not in properties or properties[field] in (None, ""):
                errors.append(f"Missing required tool argument: {field}")
        return errors


def build_artifact_tool_registry() -> ToolRegistry:
    """Expose legacy artifact actions through MCP-compatible tool metadata."""
    specs = []
    for capability in list_action_capabilities():
        is_global = capability.global_action
        parameters_schema: dict[str, Any] = {"type": "object"}
        if capability.action == "search_more":
            parameters_schema = {
                "type": "object",
                "properties": {
                    "connector_names": {"type": "array", "items": {"type": "string"}},
                    "avoid_connectors": {"type": "array", "items": {"type": "string"}},
                    "query_focus": {"type": "string"},
                    "revised_queries": {"type": "object"},
                    "source_types": {"type": "array", "items": {"type": "string"}},
                    "failure_reason": {"type": "string"},
                },
                "additionalProperties": True,
            }
        specs.append(
            ToolSpec(
                name=capability.action,
                description=capability.description,
                input_schema={
                    "type": "object",
                    "properties": {
                        "artifact_id": {"type": ["string", "null"]},
                        "parameters": parameters_schema,
                    },
                    "required": [] if is_global else ["artifact_id"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "data": {"type": "object"},
                        "warnings": {"type": "array"},
                        "errors": {"type": "array"},
                    },
                },
                artifact_types=capability.artifact_types,
                requires_artifact=not is_global,
                requires_local_path=capability.requires_local_path,
                global_action=is_global,
                side_effects=(
                    frozenset({"network"})
                    if capability.action in {
                        "search_more",
                        "search_sources",
                        "download_artifact",
                        "ingest_sources",
                        "ingest_arxiv_pdfs",
                    }
                    else frozenset()
                ),
                can_run_parallel=capability.action in {
                    "read_metadata",
                    "parse_pdf_text",
                    "parse_table",
                    "parse_csv",
                    "parse_figure",
                },
            )
        )
    return ToolRegistry(specs)
