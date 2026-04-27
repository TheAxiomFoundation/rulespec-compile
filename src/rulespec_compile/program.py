"""File-graph loading for multi-file RuleSpec compilation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .compile_model import CompilationError
from .module_resolution import (
    ImportResolver,
    ModuleResolutionError,
    build_import_resolver,
)
from .parser import (
    ImportSpec,
    ParameterDef,
    ParserError,
    RuleSpecFile,
    VariableBlock,
    parse_rulespec,
)


@dataclass(frozen=True)
class RuleSpecProgram:
    """A loaded RuleSpec program rooted at one entry file."""

    entrypoint: Path
    entry_file: RuleSpecFile
    files: list[RuleSpecFile] = field(default_factory=list)
    resolver: ImportResolver | None = None

    @property
    def default_outputs(self) -> list[str]:
        """Return the public outputs exported by the entry file."""
        _, public_output_bindings = self._resolve_output_bindings()
        return [public_name for public_name, _ in public_output_bindings]

    def to_compile_model(
        self,
        effective_date=None,
        rule_bindings: dict[str, Any] | None = None,
        parameter_overrides: dict[str, Any] | None = None,
        outputs: list[str] | None = None,
    ):
        """Compile the merged program graph through the shared compile model."""
        merged, entry_exports, output_variables = self._merged_rulespec_file()
        selected_outputs, public_output_bindings = self._resolve_output_bindings(
            outputs=outputs,
            entry_exports=entry_exports,
            output_variables=output_variables,
        )
        return merged.to_compile_model(
            effective_date=effective_date,
            rule_bindings=rule_bindings,
            parameter_overrides=parameter_overrides,
            outputs=selected_outputs,
        ).with_public_outputs(public_output_bindings)

    def to_js_generator(
        self,
        effective_date=None,
        rule_bindings: dict[str, Any] | None = None,
        parameter_overrides: dict[str, Any] | None = None,
        outputs: list[str] | None = None,
    ):
        """Build a JS generator for the merged program graph."""
        return self.to_compile_model(
            effective_date=effective_date,
            rule_bindings=rule_bindings,
            parameter_overrides=parameter_overrides,
            outputs=outputs,
        ).to_js_generator()

    def to_lowered_program(
        self,
        effective_date=None,
        rule_bindings: dict[str, Any] | None = None,
        parameter_overrides: dict[str, Any] | None = None,
        outputs: list[str] | None = None,
    ):
        """Build a lowered program bundle for the merged graph."""
        return self.to_compile_model(
            effective_date=effective_date,
            rule_bindings=rule_bindings,
            parameter_overrides=parameter_overrides,
            outputs=outputs,
        ).to_lowered_program()

    def to_python_generator(
        self,
        effective_date=None,
        rule_bindings: dict[str, Any] | None = None,
        parameter_overrides: dict[str, Any] | None = None,
        outputs: list[str] | None = None,
    ):
        """Build a Python generator for the merged program graph."""
        return self.to_compile_model(
            effective_date=effective_date,
            rule_bindings=rule_bindings,
            parameter_overrides=parameter_overrides,
            outputs=outputs,
        ).to_python_generator()

    def to_rust_generator(
        self,
        effective_date=None,
        rule_bindings: dict[str, Any] | None = None,
        parameter_overrides: dict[str, Any] | None = None,
        outputs: list[str] | None = None,
    ):
        """Build a Rust generator for the merged program graph."""
        return self.to_compile_model(
            effective_date=effective_date,
            rule_bindings=rule_bindings,
            parameter_overrides=parameter_overrides,
            outputs=outputs,
        ).to_rust_generator()

    def _merged_rulespec_file(self) -> tuple[RuleSpecFile, dict[str, str], set[str]]:
        """Merge the loaded files into one synthetic RuleSpec file and its exports."""
        merged = RuleSpecFile(
            source=self.entry_file.source,
            statute_text=self.entry_file.statute_text,
            origin=self.entrypoint,
            module_identity=self.entry_file.resolved_module_identity,
        )
        module_identities = _build_module_identities(self.files, self.entrypoint)
        internal_symbols = {
            _file_origin(
                rulespec_file, self.entrypoint
            ): _build_module_internal_symbols(
                rulespec_file,
                ""
                if _file_origin(rulespec_file, self.entrypoint) == self.entrypoint
                else module_identities[_file_origin(rulespec_file, self.entrypoint)],
            )
            for rulespec_file in self.files
        }
        _validate_internal_symbol_uniqueness(internal_symbols)
        module_exports: dict[Path, dict[str, str]] = {}
        output_variables: set[str] = set()

        for rulespec_file in self.files:
            origin = _file_origin(rulespec_file, self.entrypoint)
            computed_rule_names = {rule.name for rule in rulespec_file.computed_rules}
            module_exports[origin] = _build_module_exports(
                rulespec_file=rulespec_file,
                origin=origin,
                internal_symbols=internal_symbols[origin],
                module_exports=module_exports,
                resolver=self.resolver,
            )
            base_unqualified_bindings = dict(internal_symbols[origin])
            base_qualified_bindings: dict[str, dict[str, str]] = {}
            _apply_import_specs(
                importer=origin,
                import_specs=_local_import_specs(rulespec_file),
                resolver=self.resolver,
                module_exports=module_exports,
                unqualified_bindings=base_unqualified_bindings,
                qualified_bindings=base_qualified_bindings,
            )

            for name, parameter in rulespec_file.parameters.items():
                merged.parameters[internal_symbols[origin][name]] = _copy_parameter(
                    parameter,
                    internal_name=internal_symbols[origin][name],
                )

            for variable in rulespec_file.variables:
                internal_name = internal_symbols[origin][variable.name]
                variable_unqualified_bindings = dict(base_unqualified_bindings)
                variable_qualified_bindings = {
                    alias: dict(bindings)
                    for alias, bindings in base_qualified_bindings.items()
                }
                _apply_import_specs(
                    importer=origin,
                    import_specs=variable.import_specs,
                    resolver=self.resolver,
                    module_exports=module_exports,
                    unqualified_bindings=variable_unqualified_bindings,
                    qualified_bindings=variable_qualified_bindings,
                )
                merged.variables.append(
                    _copy_variable(
                        variable,
                        internal_name=internal_name,
                        unqualified_bindings=variable_unqualified_bindings,
                        qualified_bindings=variable_qualified_bindings,
                    )
                )
                if variable.name in computed_rule_names:
                    output_variables.add(internal_name)

        return merged, module_exports[self.entrypoint], output_variables

    def _resolve_output_bindings(
        self,
        outputs: list[str] | None = None,
        entry_exports: dict[str, str] | None = None,
        output_variables: set[str] | None = None,
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Resolve requested public outputs against the entry module surface."""
        if entry_exports is None or output_variables is None:
            _, entry_exports, output_variables = self._merged_rulespec_file()

        explicit_public_surface = bool(
            self.entry_file.export_specs or self.entry_file.re_export_specs
        )
        public_variable_exports = {
            public_name: internal_name
            for public_name, internal_name in entry_exports.items()
            if internal_name in output_variables
        }

        if explicit_public_surface and not public_variable_exports:
            raise CompilationError(
                "This entry file exports no variables. Export at least one "
                "variable to compile a public output."
            )

        requested_public = _ordered_unique(outputs or list(public_variable_exports))
        unknown = [
            name for name in requested_public if name not in public_variable_exports
        ]
        if unknown:
            names = ", ".join(unknown)
            subject = (
                "exported output variable(s)"
                if explicit_public_surface
                else ("output variable(s)")
            )
            raise CompilationError(f"Unknown {subject}: {names}.")
        return [
            public_variable_exports[public_name] for public_name in requested_public
        ], [
            (public_name, public_variable_exports[public_name])
            for public_name in requested_public
        ]


def load_rulespec_program(
    entry_path: str | Path,
    module_roots: list[Path] | tuple[Path, ...] | None = None,
    module_packages: dict[str, Path] | None = None,
) -> RuleSpecProgram:
    """Load a RuleSpec program rooted at one entry file and its imports."""
    resolved_entry = Path(entry_path).expanduser().resolve()
    _validate_rulespec_path(resolved_entry, subject="Entry file")
    try:
        resolver = build_import_resolver(
            resolved_entry,
            module_roots=module_roots,
            module_packages=module_packages,
        )
    except ModuleResolutionError as exc:
        raise CompilationError(str(exc)) from exc
    loaded: dict[Path, RuleSpecFile] = {}
    visiting: list[Path] = []
    ordered: list[RuleSpecFile] = []

    def visit(path: Path) -> None:
        if path in loaded:
            return
        if path in visiting:
            cycle = " -> ".join(str(part) for part in [*visiting, path])
            raise CompilationError(f"Import cycle detected: {cycle}.")
        _validate_rulespec_path(path, subject="Imported file")

        try:
            content = path.read_text()
        except FileNotFoundError as exc:
            importer = visiting[-1] if visiting else resolved_entry
            raise CompilationError(
                f"Import '{path}' referenced from '{importer}' was not found."
            ) from exc
        except OSError as exc:
            raise CompilationError(
                f"Could not read imported RuleSpec file '{path}': {exc}."
            ) from exc

        visiting.append(path)
        try:
            rulespec_file = parse_rulespec(content, origin=path)
        except ParserError as exc:
            raise CompilationError(
                f"Could not parse RuleSpec file '{path}': {exc}."
            ) from exc
        for import_path in _dependency_import_paths(rulespec_file):
            visit(_resolve_import_path(import_path, path, resolver))
        visiting.pop()

        loaded[path] = rulespec_file
        ordered.append(rulespec_file)

    visit(resolved_entry)
    return RuleSpecProgram(
        entrypoint=resolved_entry,
        entry_file=loaded[resolved_entry],
        files=ordered,
        resolver=resolver,
    )


def _validate_rulespec_path(path: Path, *, subject: str) -> None:
    """Require program files to use the current `.yaml` extension."""
    if path.suffix != ".yaml":
        raise CompilationError(f"{subject} '{path}' must use the .yaml extension.")


def _build_module_identities(
    files: list[RuleSpecFile], entrypoint: Path
) -> dict[Path, str]:
    """Assign stable leaf-derived identities to each loaded file."""
    identities: dict[Path, str] = {}
    seen_by_identity: dict[str, Path] = {}
    seen_by_key: dict[str, tuple[str, Path]] = {}
    for rulespec_file in files:
        origin = _file_origin(rulespec_file, entrypoint)
        module_identity = rulespec_file.resolved_module_identity
        if not module_identity:
            raise CompilationError(
                f"Could not derive a module identity for '{origin}'."
            )
        existing = seen_by_identity.get(module_identity)
        if existing is not None and existing != origin:
            raise CompilationError(
                "Module identity collision: two RuleSpec files resolve to the same "
                f"module identity '{module_identity}'.\n"
                f"  - first file:  {existing}\n"
                f"  - second file: {origin}\n"
                "Module identities must be unique within one loaded program "
                "because they key into imports, bindings, lowered metadata, and "
                "citations. To resolve, rename one of the files (for canonical "
                "RuleSpec trees) or move one into a distinct subsection leaf so the "
                "leaf-derived identity differs."
            )
        seen_by_identity[module_identity] = origin
        module_key = _module_key(module_identity)
        existing_key = seen_by_key.get(module_key)
        if existing_key is not None and existing_key[1] != origin:
            raise CompilationError(
                "Module identity collision after normalization: two distinct "
                "module identities collapse to the same internal symbol prefix "
                f"'{module_key}'.\n"
                f"  - identity: '{existing_key[0]}' (file: {existing_key[1]})\n"
                f"  - identity: '{module_identity}' (file: {origin})\n"
                "Identities are normalized by replacing non-word characters "
                "with underscores to form internal symbol prefixes. To resolve, "
                "rename one of the modules so the two identities differ after "
                "normalization (e.g. change a character that only differs by "
                "punctuation or separator)."
            )
        seen_by_key[module_key] = (module_identity, origin)
        identities[origin] = module_identity
    return identities


def _build_module_internal_symbols(
    rulespec_file: RuleSpecFile,
    module_identity: str,
) -> dict[str, str]:
    """Build the full internal symbol map for one file."""
    symbols: dict[str, str] = {}
    module_key = _module_key(module_identity) if module_identity else ""
    for name in rulespec_file.parameters:
        symbols[name] = _export_name(symbols, name, module_key, subject="parameter")
    for variable in rulespec_file.variables:
        symbols[variable.name] = _export_name(
            symbols,
            variable.name,
            module_key,
            subject="variable",
        )
    return symbols


def _build_module_exports(
    rulespec_file: RuleSpecFile,
    origin: Path,
    internal_symbols: dict[str, str],
    module_exports: dict[Path, dict[str, str]],
    resolver: ImportResolver | None,
) -> dict[str, str]:
    """Build the exported symbol map for one file."""
    if not rulespec_file.export_specs:
        exports = dict(internal_symbols)
    else:
        exports = {}
        for export_spec in rulespec_file.export_specs:
            public_name = export_spec.public_name
            try:
                internal_name = internal_symbols[export_spec.name]
            except KeyError as exc:
                raise CompilationError(
                    f"File '{origin}' exports unknown symbol '{export_spec.name}'."
                ) from exc
            existing = exports.get(public_name)
            if existing is not None and existing != internal_name:
                raise CompilationError(
                    f"File '{origin}' exports public name '{public_name}' "
                    "more than once."
                )
            exports[public_name] = internal_name

    for re_export_spec in rulespec_file.re_export_specs:
        target = _resolve_import_path(re_export_spec.path, origin, resolver)
        try:
            target_exports = module_exports[target]
        except KeyError as exc:
            raise CompilationError(
                f"Could not resolve re-export target '{target}' from '{origin}'."
            ) from exc
        _merge_re_export_bindings(
            exporter=origin,
            re_export_path=target,
            current_exports=exports,
            target_exports=target_exports,
            symbol_specs=re_export_spec.symbols,
        )
    return exports


def _export_name(
    exports: dict[str, str],
    name: str,
    module_key: str,
    subject: str,
) -> str:
    """Allocate one exported name, rejecting in-file collisions."""
    if name in exports:
        raise CompilationError(
            f"Imported file defines '{name}' as more than one top-level {subject}. "
            "Imported programs must avoid in-file symbol collisions."
        )
    if not module_key:
        return name
    return f"{module_key}_{name}"


def _merge_plain_import_bindings(
    importer: Path,
    imported_path: Path,
    current_bindings: dict[str, str],
    imported_bindings: dict[str, str],
) -> None:
    """Merge a plain import into one file's unqualified scope."""
    for name, internal_name in imported_bindings.items():
        existing = current_bindings.get(name)
        if existing is None:
            current_bindings[name] = internal_name
            continue
        if existing != internal_name:
            raise CompilationError(
                f"Plain import scope collision in '{importer}': symbol '{name}' is "
                f"available from more than one source, including '{imported_path}'. "
                "Use import aliases to disambiguate."
            )


def _add_qualified_import_binding(
    importer: Path,
    alias: str,
    target: Path,
    unqualified_bindings: dict[str, str],
    qualified_bindings: dict[str, dict[str, str]],
    target_exports: dict[str, str],
) -> None:
    """Add an aliased import scope to one file."""
    if alias in qualified_bindings:
        raise CompilationError(
            f"File '{importer}' imports more than one module as '{alias}'. "
            "Import aliases must be unique within a file."
        )
    if alias in unqualified_bindings:
        raise CompilationError(
            f"Import alias '{alias}' in '{importer}' conflicts with an existing "
            "top-level symbol or plain-imported name."
        )
    qualified_bindings[alias] = dict(target_exports)


def _merge_selective_import_bindings(
    importer: Path,
    imported_path: Path,
    current_bindings: dict[str, str],
    qualified_bindings: dict[str, dict[str, str]],
    import_spec: ImportSpec,
    imported_bindings: dict[str, str],
) -> None:
    """Merge explicitly selected imported symbols into local scope."""
    for symbol in import_spec.symbols:
        try:
            internal_name = imported_bindings[symbol.name]
        except KeyError as exc:
            raise CompilationError(
                f"Selective import from '{imported_path}' references unknown or "
                f"non-exported symbol '{symbol.name}'."
            ) from exc
        local_name = symbol.alias or symbol.name
        if local_name in qualified_bindings:
            raise CompilationError(
                f"Selective import name '{local_name}' in '{importer}' conflicts "
                "with an existing module alias."
            )
        existing = current_bindings.get(local_name)
        if existing is not None and existing != internal_name:
            raise CompilationError(
                f"Selective import scope collision in '{importer}': local name "
                f"'{local_name}' already refers to another symbol."
            )
        current_bindings[local_name] = internal_name


def _copy_parameter(
    parameter: ParameterDef,
    *,
    internal_name: str,
) -> ParameterDef:
    """Copy a parsed parameter definition."""
    return ParameterDef(
        name=internal_name,
        symbol_name=parameter.symbol_name,
        source=parameter.source,
        values=dict(parameter.values),
        temporal=list(parameter.temporal),
        description=parameter.description,
        unit=parameter.unit,
        reference=parameter.reference,
        module_identity=parameter.module_identity,
    )


def _copy_variable(
    variable: VariableBlock,
    internal_name: str,
    unqualified_bindings: dict[str, str],
    qualified_bindings: dict[str, dict[str, str]],
) -> VariableBlock:
    """Copy a parsed variable with graph-specific name bindings."""
    return VariableBlock(
        name=internal_name,
        symbol_name=variable.symbol_name,
        entity=variable.entity,
        period=variable.period,
        dtype=variable.dtype,
        label=variable.label,
        description=variable.description,
        unit=variable.unit,
        reference=variable.reference,
        source=variable.source,
        default=variable.default,
        import_specs=list(variable.import_specs),
        formula=variable.formula,
        temporal=list(variable.temporal),
        source_citation=variable.source_citation,
        unqualified_bindings=dict(unqualified_bindings),
        qualified_bindings={
            alias: dict(bindings) for alias, bindings in qualified_bindings.items()
        },
        module_identity=variable.module_identity,
    )


def _file_origin(rulespec_file: RuleSpecFile, entrypoint: Path) -> Path:
    """Return the on-disk path for a loaded file."""
    return rulespec_file.origin or entrypoint


def _local_import_specs(rulespec_file: RuleSpecFile) -> list[ImportSpec]:
    """Return normalized local import specs for a parsed file."""
    if rulespec_file.import_specs:
        return list(rulespec_file.import_specs)
    return [ImportSpec(path=path) for path in rulespec_file.imports]


def _dependency_import_paths(rulespec_file: RuleSpecFile) -> list[str]:
    """Return all imported file paths, including re-export dependencies."""
    paths = [import_spec.path for import_spec in _local_import_specs(rulespec_file)]
    for variable in rulespec_file.variables:
        paths.extend(import_spec.path for import_spec in variable.import_specs)
    paths.extend(
        re_export_spec.path for re_export_spec in rulespec_file.re_export_specs
    )
    return _ordered_unique(paths)


def _merge_re_export_bindings(
    exporter: Path,
    re_export_path: Path,
    current_exports: dict[str, str],
    target_exports: dict[str, str],
    symbol_specs,
) -> None:
    """Merge explicitly re-exported symbols into one module's public surface."""
    for symbol in symbol_specs:
        try:
            internal_name = target_exports[symbol.name]
        except KeyError as exc:
            raise CompilationError(
                f"Re-export from '{re_export_path}' references unknown or "
                f"non-exported symbol '{symbol.name}'."
            ) from exc
        public_name = symbol.alias or symbol.name
        existing = current_exports.get(public_name)
        if existing is not None and existing != internal_name:
            raise CompilationError(
                f"File '{exporter}' exports public name '{public_name}' more than once."
            )
        current_exports[public_name] = internal_name


def _ordered_unique(names: list[str]) -> list[str]:
    """Return names in first-seen order with duplicates removed."""
    return list(dict.fromkeys(names))


def _resolve_import_path(
    import_path: str,
    importer: Path,
    resolver: ImportResolver | None,
) -> Path:
    """Resolve one import string against the importing file."""
    if resolver is not None:
        try:
            return resolver.resolve(import_path, importer)
        except ModuleResolutionError as exc:
            citation_relative = _resolve_citation_relative_import_path(
                import_path,
                importer,
            )
            if citation_relative is not None:
                return citation_relative
            raise CompilationError(str(exc)) from exc
    citation_relative = _resolve_citation_relative_import_path(import_path, importer)
    if citation_relative is not None:
        return citation_relative
    candidate = Path(import_path)
    if not candidate.is_absolute():
        candidate = importer.parent / candidate
    return candidate.resolve()


def _apply_import_specs(
    *,
    importer: Path,
    import_specs: list[ImportSpec],
    resolver: ImportResolver | None,
    module_exports: dict[Path, dict[str, str]],
    unqualified_bindings: dict[str, str],
    qualified_bindings: dict[str, dict[str, str]],
) -> None:
    """Resolve and merge a list of import specs into one file or variable scope."""
    for import_spec in import_specs:
        target = _resolve_import_path(import_spec.path, importer, resolver)
        target_exports = module_exports[target]
        if import_spec.symbols:
            _merge_selective_import_bindings(
                importer=importer,
                imported_path=target,
                current_bindings=unqualified_bindings,
                qualified_bindings=qualified_bindings,
                import_spec=import_spec,
                imported_bindings=target_exports,
            )
            continue
        if import_spec.alias is None:
            _merge_plain_import_bindings(
                importer=importer,
                imported_path=target,
                current_bindings=unqualified_bindings,
                imported_bindings=target_exports,
            )
            continue
        _add_qualified_import_binding(
            importer=importer,
            alias=import_spec.alias,
            target=target,
            unqualified_bindings=unqualified_bindings,
            qualified_bindings=qualified_bindings,
            target_exports=target_exports,
        )


def _resolve_citation_relative_import_path(
    import_path: str,
    importer: Path,
) -> Path | None:
    """Resolve spec-style title-root import paths like `26/32/a`."""
    candidate = Path(import_path)
    if candidate.is_absolute() or import_path.startswith(".") or candidate.suffix:
        return None

    root_names = ("statute", "regulation", "legislation")
    importer_parts = importer.resolve().parts
    candidate_parts = candidate.parts
    if candidate_parts and candidate_parts[0] in root_names:
        anchor_indices = [
            importer_parts.index(anchor)
            for anchor in (*root_names, "source", "external", "usda")
            if anchor in importer_parts
        ]
        if anchor_indices:
            repo_root = Path(*importer_parts[: min(anchor_indices)])
            return (repo_root / candidate).with_suffix(".yaml").resolve()

    for root_name in root_names:
        if root_name not in importer_parts:
            continue
        root_index = importer_parts.index(root_name)
        repo_root = Path(*importer_parts[:root_index])
        target_path = candidate
        if len(candidate.parts) == 2:
            title, section = candidate.parts
            target_path = Path(title) / section / section
        return (repo_root / root_name / target_path).with_suffix(".yaml").resolve()
    return None


def _validate_internal_symbol_uniqueness(
    internal_symbols: dict[Path, dict[str, str]],
) -> None:
    """Reject merged programs whose internal names still collide."""
    seen: dict[str, Path] = {}
    for origin, symbols in internal_symbols.items():
        for internal_name in symbols.values():
            existing = seen.get(internal_name)
            if existing is None:
                seen[internal_name] = origin
                continue
            if existing == origin:
                continue
            raise CompilationError(
                "Loaded program contains a symbol collision after module-identity "
                f"normalization: '{internal_name}' appears in '{existing}' and "
                f"'{origin}'. Rename the entry-file symbol or one imported module "
                "so the merged graph stays unambiguous."
            )


def _module_key(module_identity: str) -> str:
    """Normalize one module identity for internal symbol allocation."""
    normalized = re.sub(r"\W+", "_", module_identity).strip("_")
    if not normalized:
        raise CompilationError(
            f"Module identity '{module_identity}' cannot be normalized to an "
            "internal symbol prefix."
        )
    if normalized[0].isdigit():
        normalized = f"_{normalized}"
    return normalized
