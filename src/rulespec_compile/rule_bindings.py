"""External rule binding bundle and resolver helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml


class RuleBindingError(ValueError):
    """Raised when external rule bindings are malformed or ambiguous."""


@dataclass(frozen=True)
class RuleBindingTarget:
    """One external rule identity targeted by a binding."""

    module_identity: str = ""
    symbol: str = ""

    @property
    def display_name(self) -> str:
        """Return the user-facing target name."""
        if self.module_identity:
            return f"{self.module_identity}.{self.symbol}"
        return self.symbol

    @classmethod
    def parse(cls, value: str) -> "RuleBindingTarget":
        """Parse `module_identity.symbol` or a bare symbol target."""
        stripped = value.strip()
        if not stripped:
            raise RuleBindingError("Rule binding target cannot be empty.")
        if "." not in stripped:
            if not re.fullmatch(r"[A-Za-z_]\w*", stripped):
                raise RuleBindingError(f"Invalid rule binding target '{value}'.")
            return cls(symbol=stripped)

        module_identity, symbol = stripped.rsplit(".", 1)
        if not module_identity or not re.fullmatch(r"[A-Za-z_]\w*", symbol):
            raise RuleBindingError(f"Invalid rule binding target '{value}'.")
        return cls(module_identity=module_identity, symbol=symbol)


@dataclass(frozen=True)
class RuleBinding:
    """A resolved external rule value/table binding."""

    values: dict[int, float]
    source: str = ""
    description: str | None = None
    unit: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class RuleBindingEntry:
    """One dated or undated binding for one external rule."""

    target: RuleBindingTarget
    binding: RuleBinding
    effective_date: date | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable entry payload."""
        payload: dict[str, Any] = {
            "module_identity": self.target.module_identity,
            "symbol": self.target.symbol,
            "values": {
                str(index): value for index, value in self.binding.values.items()
            },
        }
        if self.effective_date is not None:
            payload["effective_date"] = self.effective_date.isoformat()
        if self.binding.source:
            payload["source"] = self.binding.source
        if self.binding.description is not None:
            payload["description"] = self.binding.description
        if self.binding.unit is not None:
            payload["unit"] = self.binding.unit
        if self.binding.reference is not None:
            payload["reference"] = self.binding.reference
        return payload


@dataclass(frozen=True)
class RuleBindingBundle:
    """A structured external-rule binding bundle."""

    schema_version: int = 1
    bindings: tuple[RuleBindingEntry, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    allow_unused_entries: bool = False

    def to_resolver(self) -> "RuleResolver":
        """Build a resolver from this bundle."""
        return RuleResolver(bindings=self.bindings)


@dataclass(frozen=True)
class RuleResolver:
    """Resolve external rule bindings by identity and effective date."""

    bindings: tuple[RuleBindingEntry, ...] = ()

    def resolve(
        self,
        *,
        module_identity: str,
        symbol: str,
        effective_date: date | None = None,
    ) -> RuleBinding | None:
        """Resolve one external rule binding for one compile date."""
        target = RuleBindingTarget(module_identity=module_identity, symbol=symbol)
        candidates = [
            entry.binding
            for entry in self.bindings
            if entry.target == target and entry.effective_date is None
        ]
        dated_candidates = sorted(
            [
                entry
                for entry in self.bindings
                if entry.target == target and entry.effective_date is not None
            ],
            key=lambda entry: entry.effective_date or date.min,
        )

        if effective_date is None:
            if candidates:
                return candidates[-1]
            if dated_candidates:
                dates = ", ".join(
                    entry.effective_date.isoformat()
                    for entry in dated_candidates
                    if entry.effective_date is not None
                )
                raise RuleBindingError(
                    f"External rule '{target.display_name}' has only effective-dated "
                    f"bindings ({dates}). Compile with --effective-date or provide "
                    "an undated binding."
                )
            return None

        applicable = [
            entry
            for entry in dated_candidates
            if entry.effective_date is not None
            and entry.effective_date <= effective_date
        ]
        if applicable:
            return applicable[-1].binding
        if candidates:
            return candidates[-1]
        if dated_candidates:
            earliest = dated_candidates[0].effective_date
            raise RuleBindingError(
                f"External rule '{target.display_name}' has no binding effective as "
                f"of {effective_date.isoformat()}. Earliest available binding starts "
                f"on {earliest.isoformat()}."
            )
        return None


def normalize_rule_bindings(
    value: dict[str, Any] | RuleBindingBundle | RuleResolver | None,
) -> RuleBindingBundle:
    """Normalize user-supplied external rule bindings or bundles."""
    if value is None:
        return RuleBindingBundle()
    if isinstance(value, RuleBindingBundle):
        return value
    if isinstance(value, RuleResolver):
        return RuleBindingBundle(bindings=value.bindings)
    if not isinstance(value, dict):
        raise RuleBindingError(
            "Rule bindings must be a dict, RuleBindingBundle, or RuleResolver."
        )
    if _looks_like_rule_bundle(value):
        return _parse_rule_binding_bundle(value, source_label="Rule bindings")
    return RuleBindingBundle(bindings=tuple(_parse_plain_rule_map(value)))


def merge_rule_bindings(
    *sources: dict[str, Any] | RuleBindingBundle | RuleResolver | None,
) -> RuleBindingBundle:
    """Merge several binding sources with later sources winning."""
    merged_entries: list[RuleBindingEntry] = []
    entry_index: dict[tuple[str, str, date | None], int] = {}
    metadata: dict[str, Any] = {}
    allow_unused_entries = False
    for source in sources:
        bundle = normalize_rule_bindings(source)
        metadata.update(bundle.metadata)
        allow_unused_entries = allow_unused_entries or bundle.allow_unused_entries
        for entry in bundle.bindings:
            key = (
                entry.target.module_identity,
                entry.target.symbol,
                entry.effective_date,
            )
            existing_index = entry_index.get(key)
            if existing_index is None:
                entry_index[key] = len(merged_entries)
                merged_entries.append(entry)
                continue
            existing = merged_entries[existing_index]
            merged_entries[existing_index] = RuleBindingEntry(
                target=existing.target,
                effective_date=existing.effective_date,
                binding=_merge_rule_binding(existing.binding, entry.binding),
            )
    return RuleBindingBundle(
        bindings=tuple(merged_entries),
        metadata=metadata,
        allow_unused_entries=allow_unused_entries,
    )


def load_rule_bindings_file(path: Path | None) -> RuleBindingBundle:
    """Load external rule bindings from JSON or YAML bundle files."""
    if path is None:
        return RuleBindingBundle()
    try:
        raw = _load_rule_binding_mapping(path)
    except FileNotFoundError as exc:
        raise RuleBindingError(f"Rule binding file '{path}' was not found.") from exc
    if not isinstance(raw, dict):
        raise RuleBindingError(
            f"Rule binding file '{path}' must contain an object at the top level."
        )
    if _looks_like_rule_bundle(raw):
        return _parse_rule_binding_bundle(
            raw,
            source_label=f"Rule binding file '{path}'",
        )
    raise RuleBindingError(
        f"Rule binding file '{path}' is not a supported rule-binding file. "
        "Expected a schema_version: 1 bundle with a 'bindings' list."
    )


def _load_rule_binding_mapping(path: Path) -> Any:
    """Load one raw mapping payload from a binding file path."""
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuleBindingError(
                f"Rule binding file '{path}' is not valid JSON: {exc.msg}."
            ) from exc

    if suffix in {".yaml", ".yml"}:
        return _load_yaml_binding_mapping(path, text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _load_yaml_binding_mapping(path, text)


def _load_yaml_binding_mapping(path: Path, text: str) -> Any:
    """Load a YAML-compatible rule-binding payload."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RuleBindingError(
            f"Rule binding file '{path}' is not valid YAML: {exc}."
        ) from exc


def _parse_rule_binding_bundle(
    raw: dict[str, Any],
    *,
    source_label: str,
) -> RuleBindingBundle:
    """Parse a structured bundle with explicit binding entries."""
    schema_version = raw.get("schema_version", 1)
    if schema_version != 1:
        raise RuleBindingError(
            f"{source_label} uses unsupported schema_version {schema_version}. "
            "Expected 1."
        )
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise RuleBindingError(f"{source_label} has a non-object 'metadata' field.")
    raw_bindings = raw.get("bindings", [])
    if not isinstance(raw_bindings, list):
        raise RuleBindingError(f"{source_label} has a non-list 'bindings' field.")
    bindings = tuple(
        _parse_rule_binding_entry(entry, source_label=source_label)
        for entry in raw_bindings
    )
    return RuleBindingBundle(schema_version=1, bindings=bindings, metadata=metadata)


def _parse_plain_rule_map(raw: dict[str, Any]) -> list[RuleBindingEntry]:
    """Parse a plain map keyed by rule target name."""
    entries: list[RuleBindingEntry] = []
    for name, payload in raw.items():
        target = RuleBindingTarget.parse(str(name))
        entries.append(
            RuleBindingEntry(
                target=target,
                binding=_normalize_rule_binding(target.display_name, payload),
            )
        )
    return entries


def _parse_rule_binding_entry(
    raw: Any,
    *,
    source_label: str,
) -> RuleBindingEntry:
    """Parse one structured binding entry."""
    if not isinstance(raw, dict):
        raise RuleBindingError(f"{source_label} contains a non-object binding entry.")
    if "target" in raw:
        target = RuleBindingTarget.parse(str(raw["target"]))
    else:
        symbol = raw.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            raise RuleBindingError(
                f"{source_label} binding entry must include 'symbol' or 'target'."
            )
        module_identity = raw.get("module_identity", "")
        if module_identity is None:
            module_identity = ""
        target = RuleBindingTarget(
            module_identity=str(module_identity),
            symbol=symbol,
        )
    raw_effective_date = raw.get("effective_date")
    effective_date: date | None = None
    if raw_effective_date is not None:
        if not isinstance(raw_effective_date, str):
            raise RuleBindingError(
                f"{source_label} binding entry for '{target.display_name}' has "
                "a non-string effective_date."
            )
        try:
            effective_date = date.fromisoformat(raw_effective_date)
        except ValueError as exc:
            raise RuleBindingError(
                f"{source_label} binding entry for '{target.display_name}' has "
                f"invalid effective_date '{raw_effective_date}'."
            ) from exc
    binding = _normalize_rule_binding(
        target.display_name,
        {
            "values": raw.get("values", raw.get("value")),
            "source": raw.get("source", ""),
            "description": raw.get("description"),
            "unit": raw.get("unit"),
            "reference": raw.get("reference"),
        },
    )
    return RuleBindingEntry(
        target=target,
        binding=binding,
        effective_date=effective_date,
    )


def _looks_like_rule_bundle(raw: dict[str, Any]) -> bool:
    """Return whether a mapping looks like a structured rule-binding bundle."""
    if "bindings" not in raw:
        return False
    if not isinstance(raw["bindings"], list):
        return False
    return not (set(raw) - {"schema_version", "metadata", "bindings"})


def _normalize_rule_binding(name: str, raw: Any) -> RuleBinding:
    """Normalize one rule-binding payload."""
    if isinstance(raw, RuleBinding):
        return raw
    if isinstance(raw, (int, float)):
        return RuleBinding(values={0: float(raw)})
    if isinstance(raw, (list, tuple)):
        return RuleBinding(values=_coerce_sequence_values(name, raw))
    if isinstance(raw, dict):
        if _looks_like_binding_object(raw):
            values = _normalize_binding_values(
                name,
                raw.get("values", raw.get("value")),
            )
            return RuleBinding(
                values=values,
                source=str(raw.get("source", "")),
                description=_optional_string(raw.get("description")),
                unit=_optional_string(raw.get("unit")),
                reference=_optional_string(raw.get("reference")),
            )
        return RuleBinding(values=_coerce_indexed_values(name, raw))
    raise RuleBindingError(
        f"Unsupported rule binding for '{name}': {type(raw).__name__}"
    )


def _merge_rule_binding(earlier: RuleBinding, later: RuleBinding) -> RuleBinding:
    """Merge two bindings with later values and metadata taking precedence."""
    values = dict(earlier.values)
    values.update(later.values)
    return RuleBinding(
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
        raise RuleBindingError(
            f"Structured rule binding for '{name}' must include 'value' or 'values'."
        )
    if isinstance(raw, (int, float)):
        return {0: float(raw)}
    if isinstance(raw, (list, tuple)):
        return _coerce_sequence_values(name, raw)
    if isinstance(raw, dict):
        return _coerce_indexed_values(name, raw)
    raise RuleBindingError(
        f"Structured rule binding for '{name}' has unsupported values type "
        f"{type(raw).__name__}."
    )


def _coerce_indexed_values(name: str, raw: dict[Any, Any]) -> dict[int, float]:
    """Convert an indexed binding mapping into integer indices and numeric values."""
    values: dict[int, float] = {}
    for index, entry in raw.items():
        try:
            coerced_index = _coerce_integer_index(name, index)
            values[coerced_index] = float(entry)
        except (TypeError, ValueError) as exc:
            raise RuleBindingError(
                f"Rule binding for '{name}' must use integer indices and "
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
            raise RuleBindingError(
                f"Rule binding for '{name}' must use numeric values."
            ) from exc
    return values


def _optional_string(value: Any) -> str | None:
    """Normalize an optional metadata field."""
    if value is None:
        return None
    return str(value)


def _coerce_integer_index(name: str, index: Any) -> int:
    """Coerce one binding-table key into an integer index."""
    if isinstance(index, bool):
        raise RuleBindingError(
            f"Rule binding for '{name}' must use integer indices and numeric values."
        )
    if isinstance(index, int):
        return index
    if isinstance(index, float):
        if not index.is_integer():
            raise RuleBindingError(
                f"Rule binding for '{name}' must use integer indices "
                "and numeric values."
            )
        return int(index)
    if isinstance(index, str) and re.fullmatch(r"-?\d+", index.strip()):
        return int(index.strip())
    raise RuleBindingError(
        f"Rule binding for '{name}' must use integer indices and numeric values."
    )
