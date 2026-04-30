"""Parser for RuleSpec files."""

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

import yaml

from .rule_bindings import (
    RuleBindingBundle,
    RuleBindingEntry,
    RuleBindingError,
    RuleBindingTarget,
    RuleResolver,
    merge_rule_bindings,
    normalize_rule_bindings,
)


class ParserError(ValueError):
    """Raised when a RuleSpec file cannot be parsed safely."""


@dataclass
class SourceBlock:
    """Parsed source block."""

    lawarchive: Optional[str] = None
    citation: Optional[str] = None
    accessed: Optional[str] = None


@dataclass
class TemporalEntry:
    """A temporal entry: a from-date with either a scalar value or code."""

    from_date: str
    value: Optional[float] = None
    code: Optional[str] = None


@dataclass(frozen=True)
class ImportSpec:
    """One top-level file import."""

    path: str
    alias: str | None = None
    symbols: tuple["ImportSymbolSpec", ...] = ()


@dataclass(frozen=True)
class ImportSymbolSpec:
    """One imported symbol, with an optional local alias."""

    name: str
    alias: str | None = None


@dataclass(frozen=True)
class ExportSpec:
    """One exported symbol, with an optional public alias."""

    name: str
    alias: str | None = None

    @property
    def public_name(self) -> str:
        """Return the exported public name."""
        return self.alias or self.name


@dataclass(frozen=True)
class ReExportSpec:
    """One exported symbol list forwarded from another module."""

    path: str
    symbols: tuple[ImportSymbolSpec, ...] = ()


@dataclass
class ParameterDef:
    """Parsed parameter definition."""

    name: str = ""
    symbol_name: str = ""
    source: str = ""
    values: dict[int, float] = field(default_factory=dict)
    temporal: list[TemporalEntry] = field(default_factory=list)
    description: Optional[str] = None
    unit: Optional[str] = None
    reference: Optional[str] = None
    module_identity: str = ""


@dataclass
class VariableBlock:
    """Parsed variable block."""

    name: str
    symbol_name: str = ""
    entity: Optional[str] = None
    period: Optional[str] = None
    dtype: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    unit: Optional[str] = None
    reference: Optional[str] = None
    source: str = ""
    default: Any = None
    import_specs: list[ImportSpec] = field(default_factory=list)
    formula: str = ""
    temporal: list[TemporalEntry] = field(default_factory=list)
    source_citation: str = ""
    unqualified_bindings: dict[str, str] = field(default_factory=dict)
    qualified_bindings: dict[str, dict[str, str]] = field(default_factory=dict)
    module_identity: str = ""

    @property
    def effective_formula(self) -> str:
        """Return formula, falling back to the latest temporal code entry."""
        if self.formula:
            return self.formula
        code_entries = [t for t in self.temporal if t.code]
        if code_entries:
            return max(code_entries, key=lambda t: t.from_date).code
        return ""


@dataclass(frozen=True)
class RuleDecl:
    """Unified parsed rule view used by the compiler surface."""

    name: str
    symbol_name: str = ""
    entity: Optional[str] = None
    period: Optional[str] = None
    dtype: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    unit: Optional[str] = None
    reference: Optional[str] = None
    source: str = ""
    default: Any = None
    import_specs: tuple[ImportSpec, ...] = ()
    temporal: tuple[TemporalEntry, ...] = ()
    values: dict[int, float] = field(default_factory=dict)
    source_citation: str = ""
    module_identity: str = ""
    declared_as: str = ""

    @property
    def effective_formula(self) -> str:
        """Return formula-like code when this rule is computed."""
        code_entries = [entry for entry in self.temporal if entry.code]
        if code_entries:
            return max(code_entries, key=lambda entry: entry.from_date).code or ""
        return ""

    @property
    def has_formula(self) -> bool:
        """Return whether this rule compiles as a computed formula."""
        return bool(self.effective_formula.strip())

    @property
    def is_external_rule(self) -> bool:
        """Return whether this rule behaves like an external scalar or table value."""
        return not self.has_formula and (
            self.source
            or self.values
            or any(entry.value is not None for entry in self.temporal)
        )

    @property
    def is_input_rule(self) -> bool:
        """Return whether this rule behaves like a free calculator input."""
        return not self.has_formula and not self.is_external_rule

    @property
    def is_computed_rule(self) -> bool:
        """Return whether this rule compiles from code."""
        return self.has_formula


@dataclass
class RuleSpecFile:
    """Parsed .yaml file."""

    source: Optional[SourceBlock] = None
    statute_text: Optional[str] = None
    imports: list[str] = field(default_factory=list)
    import_specs: list[ImportSpec] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    export_specs: list[ExportSpec] = field(default_factory=list)
    re_export_specs: list[ReExportSpec] = field(default_factory=list)
    parameters: dict[str, ParameterDef] = field(default_factory=dict)
    variables: list[VariableBlock] = field(default_factory=list)
    rules: list[RuleDecl] = field(default_factory=list)
    origin: Path | None = None
    module_identity: str = ""

    @property
    def resolved_module_identity(self) -> str:
        """Return the stable rule/module identity for this file."""
        return self.module_identity or _derive_module_identity(self.origin)

    @property
    def rule_decls(self) -> list[RuleDecl]:
        """Return unified parsed rules, preserving parse order when available."""
        if self.rules:
            return list(self.rules)
        return _definitions_to_rule_decls([*self.parameters.values(), *self.variables])

    @property
    def computed_rules(self) -> list[RuleDecl]:
        """Return the parsed rules that compile as formulas."""
        return [rule for rule in self.rule_decls if rule.is_computed_rule]

    @property
    def external_rules(self) -> list[RuleDecl]:
        """Return the parsed rules that behave as external scalar or table values."""
        return [rule for rule in self.rule_decls if rule.is_external_rule]

    @property
    def input_rules(self) -> list[RuleDecl]:
        """Return the parsed rules that behave as free inputs."""
        return [rule for rule in self.rule_decls if rule.is_input_rule]

    def resolve_output_bindings(
        self,
        outputs: list[str] | None = None,
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Resolve requested outputs to internal names plus public bindings."""
        from .compile_model import CompilationError

        variable_names = [variable.name for variable in self.variables]
        variable_set = set(variable_names)
        output_variable_names = [rule.name for rule in self.computed_rules]
        output_variable_set = set(output_variable_names)
        symbol_names = {rule.name for rule in self.rule_decls}

        if self.export_specs:
            exported_bindings: dict[str, str] = {}
            ordered_public_names: list[str] = []
            for export_spec in self.export_specs:
                if export_spec.name not in symbol_names:
                    raise CompilationError(
                        f"File exports unknown symbol '{export_spec.name}'."
                    )
                if export_spec.name not in output_variable_set:
                    raise CompilationError(
                        f"File exports '{export_spec.name}', but that rule has no "
                        "compiled formula. Generic compilation currently exposes "
                        "only formula-backed variables as public outputs."
                    )
                public_name = export_spec.public_name
                existing = exported_bindings.get(public_name)
                if existing is not None and existing != export_spec.name:
                    raise CompilationError(
                        f"File exports public name '{public_name}' more than once."
                    )
                if existing is None:
                    ordered_public_names.append(public_name)
                exported_bindings[public_name] = export_spec.name

            if not exported_bindings:
                raise CompilationError(
                    "This file exports no variables. Export at least one variable "
                    "to compile a public output."
                )

            requested_public = _ordered_unique(outputs or ordered_public_names)
            unknown = [
                name for name in requested_public if name not in exported_bindings
            ]
            if unknown:
                names = ", ".join(unknown)
                raise CompilationError(f"Unknown exported output variable(s): {names}.")
            return [
                exported_bindings[public_name] for public_name in requested_public
            ], [
                (public_name, exported_bindings[public_name])
                for public_name in requested_public
            ]

        requested_internal = _ordered_unique(outputs or output_variable_names)
        unknown = [name for name in requested_internal if name not in variable_set]
        if unknown:
            names = ", ".join(unknown)
            raise CompilationError(f"Unknown output variable(s): {names}.")
        unsupported = [
            name for name in requested_internal if name not in output_variable_set
        ]
        if unsupported:
            names = ", ".join(unsupported)
            raise CompilationError(
                f"Output variable(s) {names} have no compiled formula. Generic "
                "compilation currently exposes them as inputs, not public outputs."
            )
        return requested_internal, [
            (internal_name, internal_name) for internal_name in requested_internal
        ]

    def resolve_rule_bindings(
        self,
        rule_bindings: dict[str, Any] | RuleBindingBundle | RuleResolver | None = None,
    ) -> RuleResolver:
        """Resolve external rule bindings against this file graph safely."""
        bundle = normalize_rule_bindings(rule_bindings)
        if not bundle.bindings:
            return RuleResolver()

        external_rules = {rule.name: rule for rule in self.external_rules}
        exact_names = set(external_rules)
        qualified_targets: dict[str, str] = {}
        bare_targets: dict[str, list[str]] = {}
        available_targets: dict[str, RuleBindingTarget] = {}
        for internal_name, rule in external_rules.items():
            symbol_name = rule.symbol_name or internal_name
            bare_targets.setdefault(symbol_name, []).append(internal_name)
            available_targets[internal_name] = RuleBindingTarget(
                module_identity=rule.module_identity,
                symbol=symbol_name,
            )
            if rule.module_identity:
                qualified_targets[available_targets[internal_name].display_name] = (
                    internal_name
                )

        resolved_entries: list[RuleBindingEntry] = []
        for entry in bundle.bindings:
            try:
                internal_name = _resolve_rule_binding_name(
                    entry.target,
                    exact_names=exact_names,
                    qualified_targets=qualified_targets,
                    bare_targets=bare_targets,
                    available_targets=available_targets,
                )
            except RuleBindingError as exc:
                if bundle.allow_unused_entries and str(exc).startswith(
                    "Unknown rule binding target"
                ):
                    continue
                raise
            resolved_entries.append(
                RuleBindingEntry(
                    target=available_targets[internal_name],
                    binding=entry.binding,
                    effective_date=entry.effective_date,
                )
            )
        return merge_rule_bindings(
            RuleBindingBundle(bindings=tuple(resolved_entries))
        ).to_resolver()

    def to_compile_model(
        self,
        effective_date: date | str | None = None,
        rule_bindings: dict[str, Any] | RuleBindingBundle | RuleResolver | None = None,
        outputs: list[str] | None = None,
    ):
        """Convert to the shared compile model for generic compilation."""
        from .compile_model import CompilationError, CompileContext, CompiledModule
        from .program import load_rulespec_program

        if self.imports or self.re_export_specs:
            if self.origin is None:
                raise CompilationError(
                    "This RuleSpec file imports other files. Load it from disk with "
                    "load_rulespec_program() or parse it with an origin path."
                )
            return load_rulespec_program(self.origin).to_compile_model(
                effective_date=effective_date,
                rule_bindings=rule_bindings,
                outputs=outputs,
            )

        selected_outputs, public_output_bindings = self.resolve_output_bindings(outputs)
        return CompiledModule.from_rulespec_file(
            self,
            compile_context=CompileContext(
                effective_date=_normalize_effective_date(effective_date),
                external_rule_resolver=self.resolve_rule_bindings(rule_bindings),
            ),
            selected_outputs=selected_outputs,
        ).with_public_outputs(public_output_bindings)

    def to_lowered_program(
        self,
        effective_date: date | str | None = None,
        rule_bindings: dict[str, Any] | RuleBindingBundle | RuleResolver | None = None,
        outputs: list[str] | None = None,
    ):
        """Convert to a serializable lowered program bundle."""
        return self.to_compile_model(
            effective_date=effective_date,
            rule_bindings=rule_bindings,
            outputs=outputs,
        ).to_lowered_program()

    def to_js_generator(
        self,
        effective_date: date | str | None = None,
        rule_bindings: dict[str, Any] | RuleBindingBundle | RuleResolver | None = None,
        outputs: list[str] | None = None,
    ):
        """Convert to JSCodeGenerator for JS output."""
        return self.to_compile_model(
            effective_date=effective_date,
            rule_bindings=rule_bindings,
            outputs=outputs,
        ).to_js_generator()

    def to_python_generator(
        self,
        effective_date: date | str | None = None,
        rule_bindings: dict[str, Any] | RuleBindingBundle | RuleResolver | None = None,
        outputs: list[str] | None = None,
    ):
        """Convert to PythonCodeGenerator for Python output."""
        return self.to_compile_model(
            effective_date=effective_date,
            rule_bindings=rule_bindings,
            outputs=outputs,
        ).to_python_generator()

    def to_rust_generator(
        self,
        effective_date: date | str | None = None,
        rule_bindings: dict[str, Any] | RuleBindingBundle | RuleResolver | None = None,
        outputs: list[str] | None = None,
    ):
        """Convert to RustCodeGenerator for Rust output."""
        return self.to_compile_model(
            effective_date=effective_date,
            rule_bindings=rule_bindings,
            outputs=outputs,
        ).to_rust_generator()


def parse_rulespec(content: str, origin: Path | str | None = None) -> RuleSpecFile:
    """Parse a `format: rulespec/v1` YAML file into a RuleSpecFile."""
    resolved_origin = Path(origin).resolve() if origin is not None else None
    return _parse_rulespec_v1(content, resolved_origin)


def _derive_module_identity(origin: Path | None) -> str:
    """Derive one rule/module identity from its canonical RuleSpec path when present."""
    if origin is None:
        return ""
    resolved = origin.resolve()
    parts = resolved.parts
    for root_name in ("statutes", "regulations", "policies"):
        if root_name not in parts:
            continue
        root_index = parts.index(root_name)
        return str(Path(*parts[root_index:]).with_suffix(""))
    return resolved.stem


def _parse_rulespec_v1(content: str, origin: Path | None) -> RuleSpecFile:
    """Parse the current `format: rulespec/v1` YAML envelope."""
    try:
        payload = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ParserError(f"Could not parse RuleSpec v1 YAML: {exc}.") from exc
    if not isinstance(payload, dict):
        raise ParserError("RuleSpec v1 payload must be a mapping.")
    if payload.get("format") != "rulespec/v1":
        raise ParserError("RuleSpec files must declare `format: rulespec/v1`.")
    _reject_unknown_fields(
        payload,
        {"format", "module", "source", "imports", "exports", "re_exports", "rules"},
        "top-level RuleSpec v1 payload",
    )

    result = RuleSpecFile(origin=origin)
    module_identity = result.resolved_module_identity
    result.module_identity = module_identity

    module = payload.get("module")
    if isinstance(module, dict):
        summary = module.get("summary")
        if isinstance(summary, str) and summary.strip():
            result.statute_text = summary.strip()

    if "source" in payload:
        source = payload.get("source")
        if not isinstance(source, dict):
            raise ParserError("RuleSpec v1 field source must be a mapping.")
        if not source:
            raise ParserError("source: block must contain at least one field.")
        _reject_unknown_fields(
            source,
            {"lawarchive", "citation", "accessed"},
            "source block",
        )
        result.source = SourceBlock(
            lawarchive=_optional_string(source.get("lawarchive")),
            citation=_optional_string(source.get("citation")),
            accessed=_optional_string(source.get("accessed")),
        )

    result.import_specs = _parse_rulespec_v1_imports(
        payload.get("imports"),
        field="imports",
    )
    result.imports = [import_spec.path for import_spec in result.import_specs]
    result.export_specs = _parse_rulespec_v1_exports(payload.get("exports"))
    result.exports = [export_spec.public_name for export_spec in result.export_specs]
    result.re_export_specs = _parse_rulespec_v1_re_exports(
        payload.get("re_exports"),
    )
    for re_export in result.re_export_specs:
        result.exports.extend(
            symbol.alias or symbol.name for symbol in re_export.symbols
        )

    rules = payload.get("rules", [])
    if not isinstance(rules, list):
        raise ParserError("RuleSpec v1 payload field rules must be a list.")
    if (
        not rules
        and not result.import_specs
        and not result.export_specs
        and not result.re_export_specs
    ):
        raise ParserError("RuleSpec v1 payload must define at least one rule.")

    parsed_definitions: list[ParameterDef | VariableBlock] = []
    for index, raw_rule in enumerate(rules):
        if not isinstance(raw_rule, dict):
            raise ParserError(f"RuleSpec v1 rules[{index}] must be a mapping.")
        definition = _parse_rulespec_v1_rule(raw_rule, module_identity, index)
        if isinstance(definition, ParameterDef):
            result.parameters[definition.name] = definition
        else:
            result.variables.append(definition)
        parsed_definitions.append(definition)

    result.rules = _definitions_to_rule_decls(parsed_definitions)
    return result


def _parse_rulespec_v1_rule(
    raw_rule: dict[str, Any],
    module_identity: str,
    index: int,
) -> ParameterDef | VariableBlock:
    """Parse one rule object from the current RuleSpec v1 envelope."""
    _reject_unknown_fields(
        raw_rule,
        {
            "name",
            "kind",
            "entity",
            "period",
            "dtype",
            "label",
            "description",
            "unit",
            "source",
            "source_url",
            "default",
            "imports",
            "values",
            "versions",
        },
        f"rules[{index}]",
    )
    name = _required_identifier(raw_rule.get("name"), f"rules[{index}].name")
    kind = _required_string(raw_rule.get("kind"), f"rules[{index}].kind")
    source = _optional_string(raw_rule.get("source")) or ""
    source_url = _optional_string(raw_rule.get("source_url"))
    if kind != "parameter" and "values" in raw_rule:
        raise ParserError(
            f"RuleSpec v1 {kind} rule '{name}' cannot define parameter values."
        )

    temporal = _parse_rulespec_v1_versions(raw_rule, name, kind)

    if kind == "input":
        if temporal:
            raise ParserError(
                f"RuleSpec v1 input rule '{name}' cannot define versions."
            )
        return VariableBlock(
            name=name,
            symbol_name=name,
            entity=_optional_string(raw_rule.get("entity")),
            period=_optional_string(raw_rule.get("period")),
            dtype=_optional_string(raw_rule.get("dtype")),
            label=_optional_string(raw_rule.get("label")),
            description=_optional_string(raw_rule.get("description")),
            unit=_optional_string(raw_rule.get("unit")),
            reference=source_url,
            source=source or (source_url or ""),
            default=raw_rule.get("default"),
            source_citation=source,
            module_identity=module_identity,
        )

    if kind == "parameter":
        values = _parse_rulespec_v1_values(
            raw_rule.get("values"), f"rules[{index}].values"
        )
        if values and temporal:
            raise ParserError(
                f"RuleSpec v1 parameter rule '{name}' cannot mix values and versions."
            )
        return ParameterDef(
            name=name,
            symbol_name=name,
            source=source,
            values=values,
            temporal=temporal,
            description=_optional_string(raw_rule.get("description")),
            unit=_optional_string(raw_rule.get("unit")),
            reference=source_url,
            module_identity=module_identity,
        )

    if kind not in {"derived", "relation"}:
        raise ParserError(
            f"RuleSpec v1 rule '{name}' has unsupported kind {kind!r}. "
            "Supported kinds are input, parameter, derived, and relation."
        )

    return VariableBlock(
        name=name,
        symbol_name=name,
        entity=_optional_string(raw_rule.get("entity")),
        period=_optional_string(raw_rule.get("period")),
        dtype=_optional_string(raw_rule.get("dtype")),
        label=_optional_string(raw_rule.get("label")),
        description=_optional_string(raw_rule.get("description")),
        unit=_optional_string(raw_rule.get("unit")),
        reference=source_url,
        source=source or (source_url or ""),
        default=raw_rule.get("default"),
        import_specs=_parse_rulespec_v1_imports(
            raw_rule.get("imports"),
            field=f"rules[{index}].imports",
        ),
        temporal=temporal,
        source_citation=source,
        module_identity=module_identity,
    )


def _parse_rulespec_v1_versions(
    raw_rule: dict[str, Any],
    rule_name: str,
    kind: str,
) -> list[TemporalEntry]:
    """Parse `versions` entries from a RuleSpec v1 rule object."""
    versions = raw_rule.get("versions", [])
    if versions is None:
        versions = []
    if not isinstance(versions, list):
        raise ParserError(
            f"RuleSpec v1 rule '{rule_name}' field versions must be a list."
        )
    if kind in {"derived", "relation"} and not versions:
        raise ParserError(f"RuleSpec v1 rule '{rule_name}' must define versions.")
    if kind in {"parameter", "input"} and not versions:
        return []

    temporal: list[TemporalEntry] = []
    for version_index, version in enumerate(versions):
        if not isinstance(version, dict):
            raise ParserError(
                f"RuleSpec v1 rule '{rule_name}' versions[{version_index}] "
                "must be a mapping."
            )
        _reject_unknown_fields(
            version,
            {"effective_from", "formula", "value"},
            f"rules[{rule_name}].versions[{version_index}]",
        )
        effective_from = _required_string(
            version.get("effective_from"),
            f"rules[{rule_name}].versions[{version_index}].effective_from",
        )
        formula = version.get("formula", version.get("value"))
        if kind == "parameter":
            numeric_value = _numeric_formula_value(formula)
            if numeric_value is not None:
                temporal.append(
                    TemporalEntry(from_date=effective_from, value=numeric_value)
                )
                continue
        if formula is None:
            raise ParserError(
                f"RuleSpec v1 rule '{rule_name}' versions[{version_index}] "
                "must define formula."
            )
        temporal.append(TemporalEntry(from_date=effective_from, code=str(formula)))
    return temporal


def _parse_rulespec_v1_values(value: Any, field: str) -> dict[int, float]:
    """Parse a RuleSpec v1 indexed numeric value table."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ParserError(f"RuleSpec v1 field {field} must be a mapping.")
    values: dict[int, float] = {}
    for raw_index, raw_value in value.items():
        try:
            index = _parse_non_negative_integer_key(raw_index)
            values[index] = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ParserError(
                f"RuleSpec v1 field {field} must map integer indices to numbers."
            ) from exc
    return values


def _parse_rulespec_v1_imports(value: Any, *, field: str) -> list[ImportSpec]:
    """Parse top-level or per-rule RuleSpec v1 imports."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ParserError(f"RuleSpec v1 field {field} must be a list.")
    imports: list[ImportSpec] = []
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        if isinstance(item, str):
            path = _required_string(item, item_field)
            imports.append(ImportSpec(path=path))
            continue
        if not isinstance(item, dict):
            raise ParserError(
                f"RuleSpec v1 field {item_field} must be a string or mapping."
            )
        _reject_unknown_fields(item, {"path", "alias", "symbols"}, item_field)
        path = _required_string(item.get("path"), f"{item_field}.path")
        alias = _optional_identifier(item.get("alias"), f"{item_field}.alias")
        symbols = tuple(
            _parse_rulespec_v1_import_symbols(
                item.get("symbols"),
                field=f"{item_field}.symbols",
            )
        )
        if alias and symbols:
            raise ParserError(
                f"RuleSpec v1 field {item_field} cannot define both alias and symbols."
            )
        imports.append(ImportSpec(path=path, alias=alias, symbols=symbols))
    return imports


def _parse_rulespec_v1_import_symbols(
    value: Any,
    *,
    field: str,
) -> list[ImportSymbolSpec]:
    """Parse a RuleSpec v1 import/re-export symbol list."""
    if value is None:
        return []
    if not isinstance(value, list) or not value:
        raise ParserError(f"RuleSpec v1 field {field} must be a non-empty list.")
    symbols: list[ImportSymbolSpec] = []
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        if isinstance(item, str):
            symbols.append(
                ImportSymbolSpec(name=_required_identifier(item, item_field))
            )
            continue
        if not isinstance(item, dict):
            raise ParserError(
                f"RuleSpec v1 field {item_field} must be a string or mapping."
            )
        _reject_unknown_fields(item, {"name", "alias"}, item_field)
        symbols.append(
            ImportSymbolSpec(
                name=_required_identifier(item.get("name"), f"{item_field}.name"),
                alias=_optional_identifier(item.get("alias"), f"{item_field}.alias"),
            )
        )
    return symbols


def _parse_rulespec_v1_exports(value: Any) -> list[ExportSpec]:
    """Parse RuleSpec v1 public exports."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ParserError("RuleSpec v1 field exports must be a list.")
    exports: list[ExportSpec] = []
    for index, item in enumerate(value):
        item_field = f"exports[{index}]"
        if isinstance(item, str):
            exports.append(ExportSpec(name=_required_identifier(item, item_field)))
            continue
        if not isinstance(item, dict):
            raise ParserError(
                f"RuleSpec v1 field {item_field} must be a string or mapping."
            )
        _reject_unknown_fields(item, {"name", "alias"}, item_field)
        exports.append(
            ExportSpec(
                name=_required_identifier(item.get("name"), f"{item_field}.name"),
                alias=_optional_identifier(item.get("alias"), f"{item_field}.alias"),
            )
        )
    return exports


def _parse_rulespec_v1_re_exports(value: Any) -> list[ReExportSpec]:
    """Parse RuleSpec v1 re-export declarations."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ParserError("RuleSpec v1 field re_exports must be a list.")
    re_exports: list[ReExportSpec] = []
    for index, item in enumerate(value):
        item_field = f"re_exports[{index}]"
        if not isinstance(item, dict):
            raise ParserError(f"RuleSpec v1 field {item_field} must be a mapping.")
        _reject_unknown_fields(item, {"path", "symbols"}, item_field)
        path = _required_string(item.get("path"), f"{item_field}.path")
        symbols = tuple(
            _parse_rulespec_v1_import_symbols(
                item.get("symbols"),
                field=f"{item_field}.symbols",
            )
        )
        if not symbols:
            raise ParserError(f"RuleSpec v1 field {item_field}.symbols is required.")
        re_exports.append(ReExportSpec(path=path, symbols=symbols))
    return re_exports


def _numeric_formula_value(value: Any) -> float | None:
    """Return a numeric value for scalar RuleSpec v1 formulas when possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+(?:\.\d+)?", value.strip()):
        return float(value.strip())
    return None


def _parse_non_negative_integer_key(value: Any) -> int:
    """Parse one non-negative integer mapping key."""
    if isinstance(value, bool):
        raise ValueError
    if isinstance(value, int):
        if value < 0:
            raise ValueError
        return value
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        return int(value.strip())
    raise ValueError


def _optional_string(value: Any) -> str | None:
    """Return a stripped string or None for absent values."""
    if value is None:
        return None
    return str(value).strip()


def _optional_identifier(value: Any, field: str) -> str | None:
    """Return an optional identifier value."""
    if value is None:
        return None
    return _required_identifier(value, field)


def _required_string(value: Any, field: str) -> str:
    """Require a non-empty string-like value."""
    normalized = _optional_string(value)
    if not normalized:
        raise ParserError(f"RuleSpec v1 field {field} is required.")
    return normalized


def _required_identifier(value: Any, field: str) -> str:
    """Require a safe RuleSpec rule identifier."""
    normalized = _required_string(value, field)
    if re.fullmatch(r"[A-Za-z_]\w*", normalized) is None:
        raise ParserError(f"RuleSpec v1 field {field} must be an identifier.")
    return normalized


def _reject_unknown_fields(
    payload: dict[str, Any],
    allowed: set[str],
    subject: str,
) -> None:
    """Reject unknown RuleSpec v1 fields."""
    unknown = sorted(set(payload) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ParserError(f"Unknown field(s) in {subject}: {names}.")


def _resolve_rule_binding_name(
    target: RuleBindingTarget,
    *,
    exact_names: set[str],
    qualified_targets: dict[str, str],
    bare_targets: dict[str, list[str]],
    available_targets: dict[str, RuleBindingTarget],
) -> str:
    """Resolve a user-facing rule binding target to one internal name."""
    requested_name = target.display_name
    if not target.module_identity and target.symbol in exact_names:
        return target.symbol
    if target.module_identity and requested_name in qualified_targets:
        return qualified_targets[requested_name]

    candidates = bare_targets.get(target.symbol, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        qualified = ", ".join(
            sorted(available_targets[name].display_name for name in candidates)
        )
        raise RuleBindingError(
            f"Rule binding target '{requested_name}' is ambiguous across "
            f"multiple modules: {qualified}. Use module_identity.symbol."
        )

    available_names = set(exact_names) | set(qualified_targets)
    available_names.update(
        symbol for symbol, matched in bare_targets.items() if len(matched) == 1
    )
    available = ", ".join(sorted(available_names)) or "none"
    raise RuleBindingError(
        f"Unknown rule binding target '{requested_name}'. Available targets: "
        f"{available}."
    )


def _normalize_effective_date(value: date | str | None) -> date | None:
    """Normalize an optional effective date argument."""
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _ordered_unique(names: list[str]) -> list[str]:
    """Return names in first-seen order with duplicates removed."""
    return list(dict.fromkeys(names))


def _definitions_to_rule_decls(
    definitions: list[ParameterDef | VariableBlock],
) -> list[RuleDecl]:
    """Convert parsed definition objects into the unified rule view."""
    rule_decls: list[RuleDecl] = []
    for definition in definitions:
        if isinstance(definition, ParameterDef):
            rule_decls.append(
                RuleDecl(
                    name=definition.name or definition.symbol_name,
                    symbol_name=definition.symbol_name,
                    description=definition.description,
                    unit=definition.unit,
                    reference=definition.reference,
                    source=definition.source,
                    temporal=tuple(definition.temporal),
                    values=dict(definition.values),
                    module_identity=definition.module_identity,
                    declared_as="parameter",
                )
            )
            continue
        rule_decls.append(
            RuleDecl(
                name=definition.name,
                symbol_name=definition.symbol_name,
                entity=definition.entity,
                period=definition.period,
                dtype=definition.dtype,
                label=definition.label,
                description=definition.description,
                unit=definition.unit,
                reference=definition.reference,
                source=definition.source,
                default=definition.default,
                import_specs=tuple(definition.import_specs),
                temporal=tuple(definition.temporal),
                source_citation=definition.source_citation,
                module_identity=definition.module_identity,
                declared_as="variable",
            )
        )
    return rule_decls
