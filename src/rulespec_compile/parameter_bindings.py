"""Parameter bundle helpers for generic compilation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ParameterBindingError(ValueError):
    """Raised when parameter bindings are malformed."""


@dataclass(frozen=True)
class ParameterBinding:
    """A resolved external parameter binding."""

    values: dict[int, float]
    source: str = ""
    description: str | None = None
    unit: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class ParameterBundle:
    """A structured parameter bundle file."""

    schema_version: int = 1
    parameters: dict[str, ParameterBinding] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_parameter_overrides(
    value: dict[str, Any] | ParameterBundle | None,
) -> dict[str, ParameterBinding]:
    """Normalize user-supplied parameter overrides or bundles."""
    if value is None:
        return {}
    if isinstance(value, ParameterBundle):
        return value.parameters
    if not isinstance(value, dict):
        raise ParameterBindingError(
            "Parameter overrides must be a dict or ParameterBundle."
        )

    normalized: dict[str, ParameterBinding] = {}
    for name, raw in value.items():
        normalized[name] = _normalize_parameter_binding(name, raw)
    return normalized


def merge_parameter_overrides(
    *sources: dict[str, Any] | ParameterBundle | None,
) -> dict[str, ParameterBinding]:
    """Merge parameter override sources with later sources winning."""
    merged: dict[str, ParameterBinding] = {}
    for source in sources:
        normalized = normalize_parameter_overrides(source)
        for name, binding in normalized.items():
            if name in merged:
                merged[name] = _merge_parameter_binding(merged[name], binding)
            else:
                merged[name] = binding
    return merged


def load_parameter_overrides_file(path: Path | None) -> ParameterBundle:
    """Load parameter overrides from a JSON file."""
    if path is None:
        return ParameterBundle()

    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ParameterBindingError(f"Parameter file '{path}' was not found.") from exc
    except json.JSONDecodeError as exc:
        raise ParameterBindingError(
            f"Parameter file '{path}' is not valid JSON: {exc.msg}."
        ) from exc

    if not isinstance(raw, dict):
        raise ParameterBindingError(
            f"Parameter file '{path}' must contain a JSON object at the top level."
        )

    if _looks_like_ambiguous_bundle(raw):
        raise ParameterBindingError(
            f"Parameter file '{path}' is ambiguous between a structured bundle "
            "and a plain parameter map. Add a dict-valued 'metadata' field to "
            "disambiguate the structured bundle form, or avoid reserved top-level "
            "keys in the plain map."
        )

    if _looks_like_structured_bundle(raw):
        schema_version = raw.get("schema_version", 1)
        if schema_version != 1:
            raise ParameterBindingError(
                f"Parameter file '{path}' uses unsupported schema_version "
                f"{schema_version}. Expected 1."
            )
        parameters = raw.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ParameterBindingError(
                f"Parameter file '{path}' has a non-object 'parameters' field."
            )
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ParameterBindingError(
                f"Parameter file '{path}' has a non-object 'metadata' field."
            )
        return ParameterBundle(
            schema_version=1,
            parameters=normalize_parameter_overrides(parameters),
            metadata=metadata,
        )

    return ParameterBundle(parameters=normalize_parameter_overrides(raw))


def _looks_like_structured_bundle(raw: dict[str, Any]) -> bool:
    """Return whether a JSON object looks like a structured bundle."""
    if "parameters" not in raw:
        return False
    if not isinstance(raw["parameters"], dict):
        return False
    if set(raw) - {"schema_version", "metadata", "parameters"}:
        return False
    if "metadata" in raw and isinstance(raw["metadata"], dict):
        return True
    if _looks_like_single_parameter_binding_payload(raw["parameters"]):
        return False
    return _looks_like_parameter_mapping(raw["parameters"])


def _looks_like_ambiguous_bundle(raw: dict[str, Any]) -> bool:
    """Return whether the top-level shape is ambiguous between two formats."""
    if "schema_version" not in raw or "parameters" not in raw:
        return False
    if not isinstance(raw["parameters"], dict):
        return False
    if set(raw) - {"schema_version", "metadata", "parameters"}:
        return False
    if isinstance(raw.get("metadata"), dict):
        return False
    if _looks_like_single_parameter_binding_payload(raw["parameters"]):
        return False
    return not _looks_like_parameter_mapping(raw["parameters"])


def _looks_like_single_parameter_binding_payload(raw: dict[str, Any]) -> bool:
    """Return whether a dict looks like one binding payload, not a param map."""
    return ("value" in raw or "values" in raw) and _looks_like_binding_object(raw)


def _looks_like_parameter_mapping(raw: dict[str, Any]) -> bool:
    """Return whether a mapping looks like named parameters rather than indices."""
    return not raw or any(not _looks_like_numeric_index(key) for key in raw)


def _looks_like_numeric_index(value: Any) -> bool:
    """Return whether a key looks like a numeric parameter index."""
    return isinstance(value, int) or (isinstance(value, str) and value.isdigit())


def _normalize_parameter_binding(name: str, raw: Any) -> ParameterBinding:
    """Normalize a single parameter binding payload."""
    if isinstance(raw, ParameterBinding):
        return raw
    if isinstance(raw, (int, float)):
        return ParameterBinding(values={0: float(raw)})
    if isinstance(raw, (list, tuple)):
        return ParameterBinding(values=_coerce_sequence_values(name, raw))
    if isinstance(raw, dict):
        if _looks_like_binding_object(raw):
            values = _normalize_binding_values(
                name, raw.get("values", raw.get("value"))
            )
            return ParameterBinding(
                values=values,
                source=str(raw.get("source", "")),
                description=_optional_string(raw.get("description")),
                unit=_optional_string(raw.get("unit")),
                reference=_optional_string(raw.get("reference")),
            )
        return ParameterBinding(values=_coerce_indexed_values(name, raw))

    raise ParameterBindingError(
        f"Unsupported parameter override for '{name}': {type(raw).__name__}"
    )


def _merge_parameter_binding(
    earlier: ParameterBinding,
    later: ParameterBinding,
) -> ParameterBinding:
    """Merge two bindings with later values and metadata taking precedence."""
    values = dict(earlier.values)
    values.update(later.values)
    return ParameterBinding(
        values=values,
        source=later.source or earlier.source,
        description=later.description or earlier.description,
        unit=later.unit or earlier.unit,
        reference=later.reference or earlier.reference,
    )


def _looks_like_binding_object(raw: dict[str, Any]) -> bool:
    """Return whether a mapping is a structured binding object."""
    return any(
        key in raw
        for key in (
            "value",
            "values",
            "source",
            "description",
            "unit",
            "reference",
        )
    )


def _normalize_binding_values(name: str, raw: Any) -> dict[int, float]:
    """Normalize the value payload of a structured binding object."""
    if raw is None:
        raise ParameterBindingError(
            f"Structured parameter binding for '{name}' must include "
            "'value' or 'values'."
        )
    if isinstance(raw, (int, float)):
        return {0: float(raw)}
    if isinstance(raw, (list, tuple)):
        return _coerce_sequence_values(name, raw)
    if isinstance(raw, dict):
        return _coerce_indexed_values(name, raw)
    raise ParameterBindingError(
        f"Structured parameter binding for '{name}' has unsupported values type "
        f"{type(raw).__name__}."
    )


def _coerce_indexed_values(name: str, raw: dict[Any, Any]) -> dict[int, float]:
    """Convert an indexed binding mapping into numeric indices and values."""
    values: dict[int, float] = {}
    for index, entry in raw.items():
        try:
            values[int(index)] = float(entry)
        except (TypeError, ValueError) as exc:
            raise ParameterBindingError(
                f"Parameter binding for '{name}' must use numeric indices and "
                "numeric values."
            ) from exc
    return values


def _coerce_sequence_values(
    name: str,
    raw: list[Any] | tuple[Any, ...],
) -> dict[int, float]:
    """Convert a sequence binding payload into indexed numeric values."""
    values: dict[int, float] = {}
    for index, entry in enumerate(raw):
        try:
            values[index] = float(entry)
        except (TypeError, ValueError) as exc:
            raise ParameterBindingError(
                f"Parameter binding for '{name}' must use numeric values."
            ) from exc
    return values


def _optional_string(value: Any) -> str | None:
    """Normalize an optional metadata field."""
    if value is None:
        return None
    return str(value)
