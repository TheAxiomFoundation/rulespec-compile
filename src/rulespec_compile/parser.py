"""Parser for RuleSpec files."""

import re
import textwrap
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

import yaml

from .rule_bindings import (
    RuleBinding,
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

    def resolve_parameter_overrides(
        self,
        parameter_overrides: (
            dict[str, Any] | RuleBindingBundle | RuleResolver | None
        ) = None,
    ) -> dict[str, RuleBinding]:
        """Compatibility wrapper returning undated rule bindings by display name."""
        resolver = self.resolve_rule_bindings(parameter_overrides)
        return {
            entry.target.display_name: entry.binding
            for entry in resolver.bindings
            if entry.effective_date is None
        }

    def to_compile_model(
        self,
        effective_date: date | str | None = None,
        rule_bindings: dict[str, Any] | RuleBindingBundle | RuleResolver | None = None,
        parameter_overrides: (
            dict[str, Any] | RuleBindingBundle | RuleResolver | None
        ) = None,
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
                parameter_overrides=parameter_overrides,
                outputs=outputs,
            )

        selected_outputs, public_output_bindings = self.resolve_output_bindings(outputs)
        merged_rule_bindings = merge_rule_bindings(rule_bindings, parameter_overrides)
        return CompiledModule.from_rulespec_file(
            self,
            compile_context=CompileContext(
                effective_date=_normalize_effective_date(effective_date),
                external_rule_resolver=self.resolve_rule_bindings(merged_rule_bindings),
            ),
            selected_outputs=selected_outputs,
        ).with_public_outputs(public_output_bindings)

    def to_lowered_program(
        self,
        effective_date: date | str | None = None,
        rule_bindings: dict[str, Any] | RuleBindingBundle | RuleResolver | None = None,
        parameter_overrides: (
            dict[str, Any] | RuleBindingBundle | RuleResolver | None
        ) = None,
        outputs: list[str] | None = None,
    ):
        """Convert to a serializable lowered program bundle."""
        return self.to_compile_model(
            effective_date=effective_date,
            rule_bindings=rule_bindings,
            parameter_overrides=parameter_overrides,
            outputs=outputs,
        ).to_lowered_program()

    def to_js_generator(
        self,
        effective_date: date | str | None = None,
        rule_bindings: dict[str, Any] | RuleBindingBundle | RuleResolver | None = None,
        parameter_overrides: (
            dict[str, Any] | RuleBindingBundle | RuleResolver | None
        ) = None,
        outputs: list[str] | None = None,
    ):
        """Convert to JSCodeGenerator for JS output."""
        return self.to_compile_model(
            effective_date=effective_date,
            rule_bindings=rule_bindings,
            parameter_overrides=parameter_overrides,
            outputs=outputs,
        ).to_js_generator()

    def to_python_generator(
        self,
        effective_date: date | str | None = None,
        rule_bindings: dict[str, Any] | RuleBindingBundle | RuleResolver | None = None,
        parameter_overrides: (
            dict[str, Any] | RuleBindingBundle | RuleResolver | None
        ) = None,
        outputs: list[str] | None = None,
    ):
        """Convert to PythonCodeGenerator for Python output."""
        return self.to_compile_model(
            effective_date=effective_date,
            rule_bindings=rule_bindings,
            parameter_overrides=parameter_overrides,
            outputs=outputs,
        ).to_python_generator()

    def to_rust_generator(
        self,
        effective_date: date | str | None = None,
        rule_bindings: dict[str, Any] | RuleBindingBundle | RuleResolver | None = None,
        parameter_overrides: (
            dict[str, Any] | RuleBindingBundle | RuleResolver | None
        ) = None,
        outputs: list[str] | None = None,
    ):
        """Convert to RustCodeGenerator for Rust output."""
        return self.to_compile_model(
            effective_date=effective_date,
            rule_bindings=rule_bindings,
            parameter_overrides=parameter_overrides,
            outputs=outputs,
        ).to_rust_generator()


_IMPORT_PATTERN = re.compile(
    r'^import\s+["\']([^"\']+)["\'](?:\s+as\s+([A-Za-z_]\w*))?\s*$'
)
_SELECTIVE_IMPORT_PATTERN = re.compile(r'^from\s+["\']([^"\']+)["\']\s+import\s+(.+)$')
_RE_EXPORT_PATTERN = re.compile(
    r'^export\s+from\s+["\']([^"\']+)["\']\s+import\s+(.+)$'
)
_EXPORT_PATTERN = re.compile(r"^export\s+(.+)$")
_DATE_PATTERN = re.compile(r"^from\s+(\d{4}-\d{2}-\d{2})\s*:\s*$")
_DATE_SCALAR_PATTERN = re.compile(
    r"^from\s+(\d{4}-\d{2}-\d{2})\s*:\s*(-?\d+(?:\.\d+)?)\s*$"
)


def parse_rulespec(content: str, origin: Path | str | None = None) -> RuleSpecFile:
    """Parse .yaml file content into a RuleSpecFile."""
    resolved_origin = Path(origin).resolve() if origin is not None else None
    if _looks_like_rulespec_v1(content):
        return _parse_rulespec_v1(content, resolved_origin)

    result = RuleSpecFile(origin=resolved_origin)

    lines = content.split("\n")
    i = 0
    parsed_definitions: list[ParameterDef | VariableBlock] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        if stripped.startswith('"""'):
            text_lines = []
            rest = stripped[3:]
            if '"""' in rest:
                result.statute_text = rest[: rest.index('"""')]
                i += 1
                continue
            text_lines.append(rest)
            i += 1
            while i < len(lines):
                if '"""' in lines[i]:
                    before_close = lines[i][: lines[i].index('"""')]
                    text_lines.append(before_close)
                    break
                text_lines.append(lines[i])
                i += 1
            result.statute_text = textwrap.dedent("\n".join(text_lines)).strip()
            i += 1
            continue

        if stripped == "source:":
            i, result.source = _parse_source_definition(lines, i + 1)
            i += 1
            continue

        if (
            stripped.startswith("source {")
            or stripped.startswith("parameters {")
            or re.match(r"parameter\s+\w+\s*\{", stripped)
            or re.match(r"variable\s+\w+\s*\{", stripped)
        ):
            raise ParserError(
                "Legacy brace syntax is no longer supported. Rewrite this file "
                "using .yaml blocks such as 'source:' and 'name:'."
            )

        if stripped == "imports:":
            i, import_specs = _parse_variable_imports_block(lines, i + 1)
            for import_spec in import_specs:
                result.imports.append(import_spec.path)
                result.import_specs.append(import_spec)
            continue

        import_match = _IMPORT_PATTERN.match(stripped)
        if import_match:
            import_path = import_match.group(1)
            import_alias = import_match.group(2)
            result.imports.append(import_path)
            result.import_specs.append(ImportSpec(path=import_path, alias=import_alias))
            i += 1
            continue

        selective_import_match = _SELECTIVE_IMPORT_PATTERN.match(stripped)
        if selective_import_match:
            import_path = selective_import_match.group(1)
            try:
                symbols = _parse_import_symbol_list(selective_import_match.group(2))
            except ValueError as exc:
                raise ParserError(str(exc)) from exc
            result.imports.append(import_path)
            result.import_specs.append(
                ImportSpec(path=import_path, symbols=tuple(symbols))
            )
            i += 1
            continue

        re_export_match = _RE_EXPORT_PATTERN.match(stripped)
        if re_export_match:
            export_path = re_export_match.group(1)
            try:
                symbols = _parse_import_symbol_list(re_export_match.group(2))
            except ValueError as exc:
                raise ParserError(str(exc)) from exc
            result.re_export_specs.append(
                ReExportSpec(path=export_path, symbols=tuple(symbols))
            )
            result.exports.extend(symbol.alias or symbol.name for symbol in symbols)
            i += 1
            continue

        export_match = _EXPORT_PATTERN.match(stripped)
        if export_match:
            try:
                export_specs = _parse_export_list(export_match.group(1))
            except ValueError as exc:
                raise ParserError(str(exc)) from exc
            result.export_specs.extend(export_specs)
            result.exports.extend(
                export_spec.public_name for export_spec in export_specs
            )
            i += 1
            continue

        unified_match = re.match(r"^(\w+)\s*:\s*$", line)
        if unified_match:
            name = unified_match.group(1)
            i += 1
            i, definition = _parse_unified_definition(name, lines, i)
            if isinstance(definition, ParameterDef):
                result.parameters[name] = definition
            elif isinstance(definition, VariableBlock):
                result.variables.append(definition)
            parsed_definitions.append(definition)
            continue

        i += 1

    module_identity = result.resolved_module_identity
    result.module_identity = module_identity
    source_citation = result.source.citation if result.source else ""
    for name, parameter in result.parameters.items():
        if not parameter.name:
            parameter.name = name
        if not parameter.symbol_name:
            parameter.symbol_name = name
        if not parameter.module_identity:
            parameter.module_identity = module_identity
    for variable in result.variables:
        if not variable.symbol_name:
            variable.symbol_name = variable.name
        if not variable.module_identity:
            variable.module_identity = module_identity
        if not variable.source_citation:
            variable.source_citation = source_citation
    result.rules = _definitions_to_rule_decls(parsed_definitions)

    return result


def _derive_module_identity(origin: Path | None) -> str:
    """Derive one rule/module identity from its canonical RuleSpec path when present."""
    if origin is None:
        return ""
    resolved = origin.resolve()
    parts = resolved.parts
    for root_name in (
        "statutes",
        "regulations",
        "policies",
        "statute",
        "regulation",
        "legislation",
    ):
        if root_name not in parts:
            continue
        root_index = parts.index(root_name)
        return str(Path(*parts[root_index:]).with_suffix(""))
    return resolved.stem


def _looks_like_rulespec_v1(content: str) -> bool:
    """Return whether content uses the current structured RuleSpec v1 envelope."""
    try:
        payload = yaml.safe_load(content)
    except yaml.YAMLError:
        return False
    return isinstance(payload, dict) and payload.get("format") == "rulespec/v1"


def _parse_rulespec_v1(content: str, origin: Path | None) -> RuleSpecFile:
    """Parse the current `format: rulespec/v1` YAML envelope."""
    try:
        payload = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ParserError(f"Could not parse RuleSpec v1 YAML: {exc}.") from exc
    if not isinstance(payload, dict):
        raise ParserError("RuleSpec v1 payload must be a mapping.")

    result = RuleSpecFile(origin=origin)
    module_identity = result.resolved_module_identity
    result.module_identity = module_identity

    module = payload.get("module")
    if isinstance(module, dict):
        summary = module.get("summary")
        if isinstance(summary, str) and summary.strip():
            result.statute_text = summary.strip()

    source = payload.get("source")
    if isinstance(source, dict):
        result.source = SourceBlock(
            lawarchive=_optional_string(source.get("lawarchive")),
            citation=_optional_string(source.get("citation")),
            accessed=_optional_string(source.get("accessed")),
        )

    imports = payload.get("imports")
    if isinstance(imports, list):
        for item in imports:
            if not isinstance(item, str):
                raise ParserError("RuleSpec v1 imports must be strings.")
            result.imports.append(item)
            result.import_specs.append(ImportSpec(path=item))

    rules = payload.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ParserError("RuleSpec v1 payload must define a non-empty rules list.")

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
    name = _required_identifier(raw_rule.get("name"), f"rules[{index}].name")
    kind = _required_string(raw_rule.get("kind"), f"rules[{index}].kind")
    source = _optional_string(raw_rule.get("source")) or ""
    source_url = _optional_string(raw_rule.get("source_url"))
    temporal = _parse_rulespec_v1_versions(raw_rule, name, kind)

    if kind == "parameter":
        return ParameterDef(
            name=name,
            symbol_name=name,
            source=source,
            temporal=temporal,
            description=_optional_string(raw_rule.get("description")),
            unit=_optional_string(raw_rule.get("unit")),
            reference=source_url,
            module_identity=module_identity,
        )

    if kind not in {"derived", "relation"}:
        raise ParserError(
            f"RuleSpec v1 rule '{name}' has unsupported kind {kind!r}. "
            "Supported kinds are parameter, derived, and relation."
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
    versions = raw_rule.get("versions")
    if not isinstance(versions, list) or not versions:
        raise ParserError(f"RuleSpec v1 rule '{rule_name}' must define versions.")

    temporal: list[TemporalEntry] = []
    for version_index, version in enumerate(versions):
        if not isinstance(version, dict):
            raise ParserError(
                f"RuleSpec v1 rule '{rule_name}' versions[{version_index}] "
                "must be a mapping."
            )
        effective_from = _required_string(
            version.get("effective_from"),
            f"rules[{rule_name}].versions[{version_index}].effective_from",
        )
        formula = version.get("formula")
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


def _numeric_formula_value(value: Any) -> float | None:
    """Return a numeric value for scalar RuleSpec v1 formulas when possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+(?:\.\d+)?", value.strip()):
        return float(value.strip())
    return None


def _optional_string(value: Any) -> str | None:
    """Return a stripped string or None for absent values."""
    if value is None:
        return None
    return str(value).strip()


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


def _resolve_rule_binding_name(
    target: RuleBindingTarget,
    *,
    exact_names: set[str],
    qualified_targets: dict[str, str],
    bare_targets: dict[str, list[str]],
    available_targets: dict[str, RuleBindingTarget],
) -> str:
    """Resolve a user-facing parameter binding target to one internal name."""
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


def _parse_import_symbol_list(value: str) -> list[ImportSymbolSpec]:
    """Parse `name` and `name as alias` entries from a selective import."""
    symbols: list[ImportSymbolSpec] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        match = re.fullmatch(
            r"([A-Za-z_]\w*)(?:\s+as\s+([A-Za-z_]\w*))?",
            stripped,
        )
        if match is None:
            raise ValueError(f"Invalid selective import item '{stripped}'.")
        symbols.append(ImportSymbolSpec(name=match.group(1), alias=match.group(2)))
    if not symbols:
        raise ValueError("Selective imports must name at least one symbol.")
    return symbols


def _parse_export_list(value: str) -> list[ExportSpec]:
    """Parse a comma-separated export declaration."""
    names: list[ExportSpec] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        match = re.fullmatch(
            r"([A-Za-z_]\w*)(?:\s+as\s+([A-Za-z_]\w*))?",
            stripped,
        )
        if match is None:
            raise ValueError(f"Invalid export name '{stripped}'.")
        names.append(ExportSpec(name=match.group(1), alias=match.group(2)))
    if not names:
        raise ValueError("Export declarations must name at least one symbol.")
    return names


def _ordered_unique(names: list[str]) -> list[str]:
    """Return names in first-seen order with duplicates removed."""
    return list(dict.fromkeys(names))


def _collect_indented_block(lines: list[str], start: int) -> tuple[int, str]:
    """Collect one indented block and return the next unread line index."""
    block_lines: list[str] = []
    block_indent: int | None = None
    i = start

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            if block_indent is not None:
                block_lines.append("")
            i += 1
            continue

        indent = len(line) - len(line.lstrip())
        if block_indent is None:
            if indent == 0:
                break
            block_indent = indent
        elif indent < block_indent:
            break

        block_lines.append(line)
        i += 1

    while block_lines and not block_lines[-1].strip():
        block_lines.pop()

    return i, textwrap.dedent("\n".join(block_lines)).strip()


def _parse_unified_definition(
    name: str, lines: list[str], start: int
) -> tuple[int, "ParameterDef | VariableBlock"]:
    """Parse a unified definition block starting after the 'name:' line.

    Returns (next line index, parsed ParameterDef or VariableBlock).
    """
    attrs: dict[str, str] = {}
    temporal: list[TemporalEntry] = []
    import_specs: list[ImportSpec] = []
    values: dict[int, float] = {}
    i = start

    while i < len(lines):
        line = lines[i]
        if line and not line[0].isspace() and line.strip():
            break

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        scalar_match = _DATE_SCALAR_PATTERN.match(stripped)
        if scalar_match:
            temporal.append(
                TemporalEntry(
                    from_date=scalar_match.group(1),
                    value=float(scalar_match.group(2)),
                )
            )
            i += 1
            continue

        date_match = _DATE_PATTERN.match(stripped)
        if date_match:
            date = date_match.group(1)
            i += 1
            i, entry = _collect_temporal_block(date, lines, i)
            temporal.append(entry)
            continue

        if stripped == "values:":
            i, values = _parse_values_block(lines, i + 1)
            continue

        if stripped == "imports:":
            i, import_specs = _parse_variable_imports_block(lines, i + 1)
            continue

        attr_match = re.match(r"(\w+)\s*:\s*(.+)", stripped)
        if attr_match:
            attrs[attr_match.group(1)] = attr_match.group(2).strip().strip('"')
            i += 1
            continue

        i += 1

    has_temporal_code = any(entry.code for entry in temporal)
    if (
        any(k in attrs for k in ("entity", "period", "dtype", "formula", "default"))
        or import_specs
        or has_temporal_code
    ):
        if values:
            raise ParserError(
                f"Variable '{name}' cannot define a parameter values block."
            )
        return i, VariableBlock(
            name=name,
            symbol_name=name,
            entity=attrs.get("entity"),
            period=attrs.get("period"),
            dtype=attrs.get("dtype"),
            label=attrs.get("label"),
            description=attrs.get("description"),
            unit=attrs.get("unit"),
            reference=attrs.get("reference"),
            source=attrs.get("source", attrs.get("reference", "")),
            default=_parse_default_value(attrs["default"])
            if "default" in attrs
            else None,
            import_specs=list(import_specs),
            formula=attrs.get("formula", ""),
            temporal=temporal,
        )

    if values and temporal:
        raise ParserError(
            f"Parameter '{name}' cannot mix a values block with temporal entries."
        )

    return i, ParameterDef(
        name=name,
        symbol_name=name,
        source=attrs.get("source", attrs.get("reference", "")),
        description=attrs.get("description"),
        unit=attrs.get("unit"),
        reference=attrs.get("reference"),
        temporal=temporal,
        values=values
        or {
            idx: entry.value
            for idx, entry in enumerate(temporal)
            if entry.value is not None
        },
    )


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


def _collect_temporal_block(
    date: str, lines: list[str], start: int
) -> tuple[int, TemporalEntry]:
    """Collect an indented code block under a 'from date:' line."""
    i, code = _collect_indented_block(lines, start)

    try:
        return i, TemporalEntry(from_date=date, value=float(code))
    except ValueError:
        return i, TemporalEntry(from_date=date, code=code)


def _parse_source_block(content: str) -> SourceBlock:
    """Parse source block content."""
    source = SourceBlock()
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"')

            if key == "lawarchive":
                source.lawarchive = value
            elif key == "citation":
                source.citation = value
            elif key == "accessed":
                source.accessed = value

    return source


def _parse_source_definition(lines: list[str], start: int) -> tuple[int, SourceBlock]:
    """Parse a top-level `source:` block."""
    i, block = _collect_indented_block(lines, start)
    if not block:
        raise ParserError("source: block must contain at least one field.")
    return i - 1, _parse_source_block(block)


def _parse_values_block(lines: list[str], start: int) -> tuple[int, dict[int, float]]:
    """Parse an indented parameter values block."""
    i, block = _collect_indented_block(lines, start)
    if not block:
        raise ParserError("values: block must contain at least one indexed value.")

    values: dict[int, float] = {}
    for line in block.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.fullmatch(r"(\d+)\s*:\s*(-?\d+(?:\.\d+)?)", stripped)
        if match is None:
            raise ParserError(
                f"Invalid parameter values entry '{stripped}'. Use INDEX: NUMBER."
            )
        values[int(match.group(1))] = float(match.group(2))

    if not values:
        raise ParserError("values: block must contain at least one indexed value.")

    return i, values


def _parse_default_value(value: str) -> Any:
    """Parse one variable default value from inline RuleSpec metadata."""
    normalized = value.strip()
    lowered = normalized.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"-?\d+", normalized):
        return int(normalized)
    if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
        return float(normalized)
    return normalized


def _parse_variable_imports_block(
    lines: list[str],
    start: int,
) -> tuple[int, list[ImportSpec]]:
    """Parse one per-variable `imports:` block."""
    i, block = _collect_indented_block(lines, start)
    if not block:
        raise ParserError("imports: block must contain at least one import.")

    specs: list[ImportSpec] = []
    for line in block.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not stripped.startswith("- "):
            raise ParserError(
                f"Invalid imports entry '{stripped}'. Use '- path#symbol' syntax."
            )
        item = stripped[2:].strip()
        match = re.fullmatch(
            r"([^#\s]+)#([A-Za-z_]\w*)(?:\s+as\s+([A-Za-z_]\w*))?",
            item,
        )
        if match is None:
            raise ParserError(
                f"Invalid imports entry '{item}'. Use 'path#symbol' or "
                "'path#symbol as alias'."
            )
        specs.append(
            ImportSpec(
                path=match.group(1),
                symbols=(ImportSymbolSpec(name=match.group(2), alias=match.group(3)),),
            )
        )

    if not specs:
        raise ParserError("imports: block must contain at least one import.")
    return i, specs
