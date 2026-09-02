from __future__ import annotations

import json
import re
import types
from collections.abc import Mapping, Sequence, Set
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel


_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def normalize_payload_for_model(
    payload: Any,
    model_type: type[BaseModel],
) -> tuple[Any, list[dict[str, Any]]]:
    """Apply schema-driven, lossless type normalization to an LLM payload.

    This helper handles harmless representation drift such as ``"0.92"`` for
    a float or ``"one item"`` for a list of strings. It deliberately does not
    invent missing fields, rename domain-specific fields, or discard unknown
    fields; Pydantic validation and the node's existing business normalizers
    remain responsible for those decisions.
    """
    events: list[dict[str, Any]] = []
    normalized = _normalize_value(payload, model_type, "$", events)
    return normalized, events


def _normalize_value(value: Any, annotation: Any, path: str, events: list[dict[str, Any]]) -> Any:
    annotation = _unwrap_annotation(annotation)
    if annotation is Any:
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)

    if value is None:
        if origin in (list, Sequence):
            _event(events, path, "null_to_empty_list", value, [])
            return []
        if origin in (dict, Mapping):
            _event(events, path, "null_to_empty_dict", value, {})
            return {}
        return None

    if origin in (Union, types.UnionType):
        candidates = [candidate for candidate in args if candidate is not type(None)]
        for candidate in candidates:
            if _looks_compatible(value, candidate):
                return _normalize_value(value, candidate, path, events)
        return _normalize_value(value, candidates[0], path, events) if candidates else value

    if origin is Literal:
        return _normalize_literal(value, args, path, events)

    if origin in (list, Sequence):
        item_annotation = args[0] if args else Any
        values = value
        if isinstance(value, str):
            parsed = _parse_json_container(value, list, path, events)
            values = parsed if parsed is not None else [value]
            if parsed is None:
                _event(events, path, "scalar_to_list", value, values)
        elif isinstance(value, (tuple, Set)):
            values = list(value)
            _event(events, path, "sequence_to_list", value, values)
        elif not isinstance(value, list):
            values = [value]
            _event(events, path, "scalar_to_list", value, values)
        return [
            _normalize_value(item, item_annotation, f"{path}[{index}]", events)
            for index, item in enumerate(values)
        ]

    if origin in (dict, Mapping):
        key_annotation = args[0] if args else Any
        value_annotation = args[1] if len(args) > 1 else Any
        mapping = value
        if isinstance(value, str):
            mapping = _parse_json_container(value, dict, path, events)
        if not isinstance(mapping, Mapping):
            return value
        return {
            _normalize_value(key, key_annotation, f"{path}.<key>", events): _normalize_value(
                item, value_annotation, f"{path}.{key}", events
            )
            for key, item in mapping.items()
        }

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        model_value = value
        if isinstance(value, str):
            model_value = _parse_json_container(value, dict, path, events)
        if not isinstance(model_value, Mapping):
            return value
        return _normalize_model_mapping(model_value, annotation, path, events)

    if annotation is bool and isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "yes", "y", "1"}:
            _event(events, path, "string_to_bool", value, True)
            return True
        if lowered in {"false", "no", "n", "0"}:
            _event(events, path, "string_to_bool", value, False)
            return False

    if annotation is int and isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        converted = int(value.strip())
        _event(events, path, "numeric_string_to_int", value, converted)
        return converted

    if annotation is float and isinstance(value, str) and _NUMBER_RE.fullmatch(value.strip()):
        converted = float(value.strip())
        _event(events, path, "numeric_string_to_float", value, converted)
        return converted

    if annotation is str and not isinstance(value, str):
        converted = (
            json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            if isinstance(value, (dict, list, tuple, set))
            else str(value)
        )
        _event(events, path, "value_to_string", value, converted)
        return converted

    return value


def _normalize_model_mapping(
    value: Mapping[Any, Any],
    model_type: type[BaseModel],
    path: str,
    events: list[dict[str, Any]],
) -> dict[Any, Any]:
    result = dict(value)
    try:
        annotations = get_type_hints(model_type)
    except Exception:
        annotations = {name: field.annotation for name, field in model_type.model_fields.items()}
    for field_name, field in model_type.model_fields.items():
        if field_name in result:
            annotation = annotations.get(field_name, field.annotation)
            result[field_name] = _normalize_value(
                result[field_name], annotation, f"{path}.{field_name}", events
            )
    return result


def _unwrap_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if str(origin) == "<class 'typing.Annotated'>":
        args = get_args(annotation)
        return args[0] if args else annotation
    return annotation


def _looks_compatible(value: Any, annotation: Any) -> bool:
    annotation = _unwrap_annotation(annotation)
    origin = get_origin(annotation)
    if origin is Literal:
        return value in get_args(annotation) or isinstance(value, str)
    if origin in (list, Sequence):
        return isinstance(value, (list, tuple, set, str)) or not isinstance(value, Mapping)
    if origin in (dict, Mapping):
        return isinstance(value, Mapping) or isinstance(value, str)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return isinstance(value, (Mapping, str))
    if annotation is bool:
        return isinstance(value, (bool, str))
    if annotation in (int, float):
        return isinstance(value, (int, float, str)) and not isinstance(value, bool)
    return isinstance(value, annotation) if isinstance(annotation, type) else True


def _normalize_literal(value: Any, choices: tuple[Any, ...], path: str, events: list[dict[str, Any]]) -> Any:
    if value in choices:
        return value
    if isinstance(value, str):
        canonical = _canonical_token(value)
        for choice in choices:
            if isinstance(choice, str) and _canonical_token(choice) == canonical:
                _event(events, path, "literal_token_normalization", value, choice)
                return choice
        semantic = _semantic_literal_alias(canonical, choices)
        if semantic is not None:
            _event(events, path, "literal_semantic_normalization", value, semantic)
            return semantic
    return value


def _semantic_literal_alias(canonical: str, choices: tuple[Any, ...]) -> str | None:
    """Map an action-shaped classification to its allowed semantic label.

    LLM planners occasionally return the concrete operation they would run
    (for example ``download_artifact``) where the schema asks for the
    assessment category ``process``. This normalization is deliberately
    limited to the assessment vocabulary; arbitrary unsupported literals are
    still rejected by Pydantic and therefore remain visible to callers.
    """

    string_choices = {choice for choice in choices if isinstance(choice, str)}
    if not {"process", "inspect_metadata", "skip", "unknown"}.issubset(string_choices):
        return None

    process_tokens = {
        "process",
        "process_artifact",
        "download",
        "download_artifact",
        "parse",
        "parse_pdf",
        "parse_pdf_text",
        "parse_pdf_sections",
        "parse_table",
        "parse_figure",
        "parse_html",
        "parse_csv",
        "read_readme",
        "read_file_manifest",
        "extract_figures",
        "extract_records",
        "extract_dynamic_records",
    }
    metadata_tokens = {
        "inspect",
        "inspect_metadata",
        "metadata",
        "read_metadata",
        "read_source_metadata",
    }
    skip_tokens = {"skip", "ignore", "irrelevant", "reject", "do_not_process"}
    unknown_tokens = {"unknown", "unclear", "undecided", "not_sure", "n_a", "na"}
    if canonical in process_tokens:
        return "process"
    if canonical in metadata_tokens:
        return "inspect_metadata"
    if canonical in skip_tokens:
        return "skip"
    if canonical in unknown_tokens:
        return "unknown"
    return None


def _canonical_token(value: str) -> str:
    return re.sub(r"[_\-\s]+", "_", value.strip().casefold())


def _parse_json_container(value: str, expected_type: type, path: str, events: list[dict[str, Any]]) -> Any | None:
    text = value.strip()
    if not text or text[0] not in "[{":
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    if expected_type is list and not isinstance(parsed, list):
        return None
    if expected_type is dict and not isinstance(parsed, dict):
        return None
    _event(events, path, "json_string_to_container", value, parsed)
    return parsed


def _event(events: list[dict[str, Any]], path: str, rule: str, original: Any, normalized: Any) -> None:
    events.append(
        {
            "path": path,
            "rule": rule,
            "original_type": type(original).__name__,
            "normalized_type": type(normalized).__name__,
            "original_preview": _preview(original),
            "normalized_preview": _preview(normalized),
        }
    )


def _preview(value: Any, limit: int = 180) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."
