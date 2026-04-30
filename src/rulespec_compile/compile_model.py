"""Shared compile model for generic RuleSpec compilation.

This module is the backend-neutral middle of the compiler. It takes the parsed
``RuleSpecProgram`` graph (produced by :mod:`rulespec_compile.program`) and produces a
``LoweredProgram`` that the JavaScript, Python, and Rust generators (and the
batch executor) can consume. It is intentionally large because it owns every
step that is not parsing and not target-specific emission.

At ~2.5k LOC this file is overdue for a split along the phase boundaries
described below, but splitting it is an architectural change that needs to
preserve the exact shape of ``LoweredProgram`` and the current
``CompiledModule`` public surface. Until that split lands, the sections below
are marked with ``# --- Phase X: name ---`` comments so contributors can
navigate the logical layers.

Logical phases (in roughly the order the code executes):

1. **Parse / IR types** -- classes describing the compile-time (``Compiled*``)
   and serializable lowered (``Lowered*``) shapes: ``CompilationError``,
   ``CompileContext``, ``CompiledInput``, ``CompiledParameter``,
   ``CompiledOutput``, ``LoweredInput``, ``LoweredParameter``,
   ``LoweredComputation``, ``LoweredOutput``, ``LoweredProgram`` (top of file
   through ``LoweredProgram``).
2. **Compile model** -- ``CompiledVariable`` and ``CompiledModule``, the main
   per-module compile entry point that orchestrates phases 3--6 against a
   single ``RuleSpecFile`` (``CompiledVariable`` through the end of
   ``CompiledModule``).
3. **Render helpers** -- target-specific formula rendering wrappers
   (``_render_js_formula``, ``_render_python_formula``).
4. **Resolve temporal + lower inputs** -- lowering ``CompiledInput`` to
   ``LoweredInput`` and validating explicit value kinds (``_lower_input``,
   ``_input_value_kind``, ``_normalize_value_kind``, and the parameter
   value/lookup-kind helpers).
5. **Resolve bindings + infer kinds** -- statement-level kind analysis
   (``_StatementKindAnalysis``, ``_build_variable_kind_hints``,
   ``_analyze_statement_kinds``, ``_infer_expression_value_kind``,
   ``_combine_value_kinds``). This is where declared kinds, inferred kinds,
   and bound external-rule kinds are reconciled.
6. **Lower / emit** -- ``LoweredProgram`` (de)serialization helpers
   (``_statement_to_dict`` / ``_from_dict``, ``_expression_to_dict`` /
   ``_from_dict``, ``_require_*`` payload validators). These bridge the
   in-memory compile model to the on-disk/IPC lowered form.
7. **Compile driver helpers** -- actually producing ``CompiledInput`` /
   ``CompiledParameter`` / ``CompiledVariable`` from the parsed AST:
   ``_compile_reachable_variables``, ``_build_declared_inputs``,
   ``_compile_declared_input``, ``_compile_parameter``,
   ``_compile_variable``, ``_resolve_variable_formula``,
   ``_resolve_temporal_entry``, ``_parse_formula_block``.
8. **Statement resolution + ordering** -- reference binding, dependency
   analysis, and topological ordering of variables
   (``_bind_statement_references``, ``_analyze_statement_dependencies``,
   ``_order_variables``, ``_infer_input``, ``_append_unique``,
   ``_ordered_unique``).
9. **External rule binding** -- resolving source-only external rules via the
   ``RuleResolver`` on ``CompileContext``
   (``_resolve_external_rule_binding``, ``_bound_external_rule_source``,
   ``_external_rule_binding_target``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from .expression_ir import (
    AssignStmt,
    BinaryExpr,
    BoolExpr,
    CallExpr,
    CompareExpr,
    ConditionalExpr,
    Expression,
    ExpressionParseError,
    IfStmt,
    LiteralExpr,
    NameExpr,
    ReturnStmt,
    Statement,
    SubscriptExpr,
    UnaryExpr,
    collect_assigned_names,
    collect_references,
    formula_has_branching,
    is_straight_line_formula,
    map_statement_names,
    parse_formula_statements,
    render_expression_js,
    render_expression_python,
    render_statement_block_js,
    render_statement_block_python,
)
from .js_generator import JSCodeGenerator
from .python_generator import PythonCodeGenerator
from .rule_bindings import RuleBinding, RuleBindingError, RuleResolver
from .rust_generator import RustCodeGenerator

if TYPE_CHECKING:
    from .parser import RuleDecl, RuleSpecFile, VariableBlock


# --- Phase 1: Parse / IR types ---


class CompilationError(ValueError):
    """Raised when generic compilation cannot produce correct output."""


@dataclass(frozen=True)
class CompileContext:
    """Context used to resolve compilation decisions."""

    effective_date: date | None = None
    external_rule_resolver: RuleResolver = field(default_factory=RuleResolver)


_BOOLEAN_PREFIXES = ("can_", "has_", "is_", "should_")
_INTEGER_PREFIXES = ("n_", "num_")
_INTEGER_SUFFIXES = ("_count", "_size", "_children")
_LOWERED_VALUE_KINDS = {"boolean", "integer", "number", "string"}
_LOWERED_INPUT_VALUE_KINDS = {"boolean", "integer", "number"}
_LOWERED_PARAMETER_VALUE_KINDS = {"integer", "number"}
_LOWERED_PARAMETER_LOOKUP_KINDS = {"scalar", "indexed"}
_LOWERED_PARAMETER_INDEX_VALUE_KINDS = {"integer"}


@dataclass
class CompiledInput:
    """An external calculator input discovered from formula references."""

    name: str
    default: Any
    js_type: str
    python_type: str
    public_name: str = ""
    module_identity: str = ""
    symbol_name: str = ""

    @property
    def external_name(self) -> str:
        """Return the user-facing input name exposed by generated targets."""
        return self.public_name or self.name


@dataclass
class CompiledParameter:
    """A concrete parameter available to target generators."""

    name: str
    values: dict[int, float]
    source: str = ""
    module_identity: str = ""
    value_kind: str = "number"
    lookup_kind: str = "scalar"
    index_value_kind: str | None = None


@dataclass(frozen=True)
class CompiledOutput:
    """One public output exposed by a compiled calculator."""

    name: str
    variable_name: str
    value_kind: str
    module_identity: str = ""


@dataclass(frozen=True)
class LoweredInput:
    """A backend-neutral public input in the lowered program bundle."""

    name: str
    default: Any
    value_kind: str
    public_name: str = ""
    module_identity: str = ""
    symbol_name: str = ""

    @property
    def external_name(self) -> str:
        """Return the user-facing input name exposed by generated targets."""
        return self.public_name or self.name

    @property
    def js_type(self) -> str:
        """Return the JavaScript type used for this input."""
        if self.value_kind == "boolean":
            return "boolean"
        return "number"

    @property
    def python_type(self) -> str:
        """Return the Python type used for this input."""
        return {
            "boolean": "bool",
            "integer": "int",
            "number": "float",
        }[self.value_kind]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "default": self.default,
            "value_kind": self.value_kind,
            "public_name": self.public_name,
            "module_identity": self.module_identity,
            "symbol_name": self.symbol_name,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LoweredInput":
        """Load one lowered input from JSON-compatible data."""
        payload = _require_object(payload, "Lowered input")
        try:
            name = payload["name"]
            default = payload["default"]
            raw_value_kind = payload["value_kind"]
        except KeyError as exc:
            raise CompilationError(
                f"Lowered input is missing required field {exc.args[0]!r}."
            ) from exc
        value_kind = _normalize_value_kind(
            raw_value_kind,
            subject=f"Lowered input '{name}'",
            allowed=_LOWERED_INPUT_VALUE_KINDS,
        )
        public_name = payload.get("public_name", name)
        module_identity = payload.get("module_identity", "")
        symbol_name = payload.get("symbol_name") or _infer_input_symbol_name(
            name=name,
            public_name=public_name,
            module_identity=module_identity,
        )
        return cls(
            name=name,
            default=default,
            value_kind=value_kind,
            public_name=public_name,
            module_identity=module_identity,
            symbol_name=symbol_name,
        )


@dataclass(frozen=True)
class LoweredParameter:
    """A resolved parameter in the lowered program bundle."""

    name: str
    values: dict[int, float]
    source: str = ""
    module_identity: str = ""
    value_kind: str = "number"
    lookup_kind: str = "scalar"
    index_value_kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "values": {str(index): value for index, value in self.values.items()},
            "source": self.source,
            "module_identity": self.module_identity,
            "value_kind": self.value_kind,
            "lookup_kind": self.lookup_kind,
            "index_value_kind": self.index_value_kind,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LoweredParameter":
        """Load one lowered parameter from JSON-compatible data."""
        payload = _require_object(payload, "Lowered parameter")
        try:
            name = payload["name"]
            raw_values = payload["values"]
            raw_value_kind = payload["value_kind"]
            raw_lookup_kind = payload["lookup_kind"]
        except KeyError as exc:
            raise CompilationError(
                f"Lowered parameter is missing required field {exc.args[0]!r}."
            ) from exc
        if not isinstance(raw_values, dict):
            raise CompilationError(
                f"Lowered parameter '{name}' must define values as an object."
            )
        values: dict[int, float] = {}
        for index, value in raw_values.items():
            try:
                values[int(index)] = float(value)
            except (TypeError, ValueError) as exc:
                raise CompilationError(
                    f"Lowered parameter '{name}' has invalid indexed value "
                    f"{index!r}: {value!r}."
                ) from exc
        value_kind = _normalize_value_kind(
            raw_value_kind,
            subject=f"Lowered parameter '{name}'",
            allowed=_LOWERED_PARAMETER_VALUE_KINDS,
        )
        lookup_kind = _normalize_parameter_lookup_kind(
            raw_lookup_kind,
            name=name,
        )
        index_value_kind = _normalize_parameter_index_value_kind(
            payload.get("index_value_kind"),
            name=name,
            lookup_kind=lookup_kind,
        )
        return cls(
            name=name,
            values=values,
            source=payload.get("source", ""),
            module_identity=payload.get("module_identity", ""),
            value_kind=value_kind,
            lookup_kind=lookup_kind,
            index_value_kind=index_value_kind,
        )


@dataclass(frozen=True)
class LoweredComputation:
    """A backend-neutral ordered computation in the lowered program bundle."""

    name: str
    statements: tuple[Statement, ...]
    local_names: tuple[str, ...]
    input_dependencies: tuple[str, ...]
    parameter_dependencies: tuple[str, ...]
    variable_dependencies: tuple[str, ...]
    local_value_kinds: dict[str, str] = field(default_factory=dict)
    value_kind: str = "number"
    label: str = ""
    citation: str = ""
    module_identity: str = ""

    def to_js_formula(self) -> str:
        """Render this computation as a JavaScript formula."""
        return _render_js_formula(
            self.statements,
            list(self.local_names),
            set(self.parameter_dependencies),
        )

    def to_python_formula(self) -> str:
        """Render this computation as a Python formula."""
        return _render_python_formula(
            self.statements,
            set(self.parameter_dependencies),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "statements": [
                _statement_to_dict(statement) for statement in self.statements
            ],
            "local_names": list(self.local_names),
            "local_value_kinds": dict(self.local_value_kinds),
            "input_dependencies": list(self.input_dependencies),
            "parameter_dependencies": list(self.parameter_dependencies),
            "variable_dependencies": list(self.variable_dependencies),
            "value_kind": self.value_kind,
            "label": self.label,
            "citation": self.citation,
            "module_identity": self.module_identity,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LoweredComputation":
        """Load one lowered computation from JSON-compatible data."""
        return cls._from_dict(payload)

    @classmethod
    def _from_dict(
        cls,
        payload: dict[str, Any],
        *,
        input_value_kinds: dict[str, str] | None = None,
        parameter_value_kinds: dict[str, str] | None = None,
        variable_kind_hints: dict[str, str] | None = None,
    ) -> "LoweredComputation":
        """Load one lowered computation with optional environment hints."""
        payload = _require_object(payload, "Lowered computation")
        try:
            name = payload["name"]
            raw_statements = payload["statements"]
        except KeyError as exc:
            raise CompilationError(
                f"Lowered computation is missing required field {exc.args[0]!r}."
            ) from exc
        if not isinstance(raw_statements, list):
            raise CompilationError(
                f"Lowered computation '{name}' must define statements as a list."
            )
        statements = tuple(
            _statement_from_dict(statement) for statement in raw_statements
        )
        local_names = tuple(
            payload.get("local_names") or collect_assigned_names(statements)
        )
        return cls(
            name=name,
            statements=statements,
            local_names=local_names,
            local_value_kinds=_normalize_local_value_kinds(
                payload.get("local_value_kinds"),
                computation_name=name,
                statements=statements,
                local_names=local_names,
                input_value_kinds=input_value_kinds or {},
                parameter_value_kinds=parameter_value_kinds or {},
                variable_kind_hints=variable_kind_hints or {},
            ),
            input_dependencies=tuple(payload.get("input_dependencies", [])),
            parameter_dependencies=tuple(payload.get("parameter_dependencies", [])),
            variable_dependencies=tuple(payload.get("variable_dependencies", [])),
            value_kind=_normalize_value_kind(
                payload.get("value_kind", "number"),
                subject=f"Lowered computation '{name}'",
            ),
            label=payload.get("label", ""),
            citation=payload.get("citation", ""),
            module_identity=payload.get("module_identity", ""),
        )


@dataclass(frozen=True)
class LoweredOutput:
    """One public output exposed by a lowered program bundle."""

    name: str
    variable_name: str
    value_kind: str = "number"
    module_identity: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "variable_name": self.variable_name,
            "value_kind": self.value_kind,
            "module_identity": self.module_identity,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        variable_module_identities: dict[str, str] | None = None,
    ) -> "LoweredOutput":
        """Load one lowered output from JSON-compatible data."""
        payload = _require_object(payload, "Lowered output")
        try:
            name = payload["name"]
            variable_name = payload["variable_name"]
            return cls(
                name=name,
                variable_name=variable_name,
                value_kind=_normalize_value_kind(
                    payload.get("value_kind", "number"),
                    subject=f"Lowered output '{name}'",
                ),
                module_identity=payload.get("module_identity")
                or (variable_module_identities or {}).get(variable_name, ""),
            )
        except KeyError as exc:
            raise CompilationError(
                f"Lowered output is missing required field {exc.args[0]!r}."
            ) from exc


@dataclass(frozen=True)
class LoweredProgram:
    """A serializable lowered artifact after graph resolution and pruning."""

    inputs: tuple[LoweredInput, ...] = ()
    parameters: tuple[LoweredParameter, ...] = ()
    computations: tuple[LoweredComputation, ...] = ()
    outputs: tuple[LoweredOutput, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable lowered-program payload."""
        return {
            "inputs": [compiled_input.to_dict() for compiled_input in self.inputs],
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "computations": [
                computation.to_dict() for computation in self.computations
            ],
            "outputs": [output.to_dict() for output in self.outputs],
        }

    def to_json(self, *, indent: int = 2, sort_keys: bool = True) -> str:
        """Serialize the lowered program as JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LoweredProgram":
        """Load a lowered program from JSON-compatible data."""
        payload = _require_object(payload, "Lowered program")
        inputs = tuple(
            LoweredInput.from_dict(item) for item in payload.get("inputs", [])
        )
        input_value_kinds = {
            compiled_input.name: compiled_input.value_kind for compiled_input in inputs
        }
        parameters = tuple(
            LoweredParameter.from_dict(item) for item in payload.get("parameters", [])
        )
        parameter_value_kinds = {
            parameter.name: parameter.value_kind for parameter in parameters
        }
        computations: list[LoweredComputation] = []
        variable_kind_hints: dict[str, str] = {}
        for item in payload.get("computations", []):
            computation = LoweredComputation._from_dict(
                item,
                input_value_kinds=input_value_kinds,
                parameter_value_kinds=parameter_value_kinds,
                variable_kind_hints=variable_kind_hints,
            )
            computations.append(computation)
            variable_kind_hints[computation.name] = computation.value_kind
        variable_module_identities = {
            computation.name: computation.module_identity
            for computation in computations
        }
        parameters = _validate_lowered_parameter_contracts(
            parameters,
            tuple(computations),
        )
        return cls(
            inputs=inputs,
            parameters=parameters,
            computations=tuple(computations),
            outputs=tuple(
                LoweredOutput.from_dict(
                    item,
                    variable_module_identities=variable_module_identities,
                )
                for item in payload.get("outputs", [])
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> "LoweredProgram":
        """Load a lowered program from serialized JSON."""
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise CompilationError(f"Could not parse lowered JSON: {exc}.") from exc
        if not isinstance(decoded, dict):
            raise CompilationError("Lowered JSON must decode to an object.")
        return cls.from_dict(decoded)

    def to_js_generator(self, module_name: str = "calculator") -> JSCodeGenerator:
        """Build a JavaScript generator from a lowered program bundle."""
        generator = JSCodeGenerator(module_name=module_name)
        for compiled_input in self.inputs:
            generator.add_input(
                compiled_input.name,
                compiled_input.default,
                compiled_input.js_type,
                public_name=compiled_input.external_name,
            )

        for parameter in self.parameters:
            generator.add_parameter(
                parameter.name,
                parameter.values,
                parameter.source,
                parameter.module_identity,
            )

        for computation in self.computations:
            generator.add_variable(
                name=computation.name,
                inputs=list(computation.input_dependencies),
                formula_js=computation.to_js_formula(),
                label=computation.label,
                citation=computation.citation,
                module_identity=computation.module_identity,
            )

        generator.set_outputs(list(self.outputs))
        return generator

    def to_python_generator(
        self, module_name: str = "calculator"
    ) -> PythonCodeGenerator:
        """Build a Python generator from a lowered program bundle."""
        generator = PythonCodeGenerator(module_name=module_name)
        for compiled_input in self.inputs:
            generator.add_input(
                compiled_input.name,
                compiled_input.default,
                compiled_input.python_type,
                public_name=compiled_input.external_name,
            )

        for parameter in self.parameters:
            generator.add_parameter(
                parameter.name,
                parameter.values,
                parameter.source,
                parameter.module_identity,
            )

        for computation in self.computations:
            generator.add_variable(
                name=computation.name,
                inputs=list(computation.input_dependencies),
                formula_python=computation.to_python_formula(),
                label=computation.label,
                citation=computation.citation,
                module_identity=computation.module_identity,
            )

        generator.set_outputs(list(self.outputs))
        return generator

    def to_rust_generator(self, module_name: str = "calculator") -> RustCodeGenerator:
        """Build a Rust generator from a lowered program bundle."""
        generator = RustCodeGenerator(module_name=module_name)
        for compiled_input in self.inputs:
            generator.add_input(
                compiled_input.name,
                compiled_input.default,
                compiled_input.value_kind,
                public_name=compiled_input.external_name,
            )

        for parameter in self.parameters:
            generator.add_parameter(
                parameter.name,
                parameter.values,
                parameter.source,
                parameter.module_identity,
                parameter.value_kind,
                parameter.lookup_kind,
                parameter.index_value_kind,
            )

        for computation in self.computations:
            generator.add_variable(
                name=computation.name,
                statements=computation.statements,
                local_names=computation.local_names,
                local_value_kinds=computation.local_value_kinds,
                parameter_dependencies=computation.parameter_dependencies,
                value_kind=computation.value_kind,
                label=computation.label,
                citation=computation.citation,
                module_identity=computation.module_identity,
            )

        generator.set_outputs(list(self.outputs))
        return generator


# --- Phase 2: Compile model (CompiledVariable, CompiledModule) ---


@dataclass
class CompiledVariable:
    """A target-neutral compiled variable."""

    name: str
    statements: tuple[Statement, ...]
    local_names: list[str]
    local_value_kinds: dict[str, str]
    input_dependencies: list[str]
    parameter_dependencies: list[str]
    variable_dependencies: list[str]
    value_kind: str = "number"
    label: str = ""
    citation: str = ""
    module_identity: str = ""

    def to_js_formula(self) -> str:
        """Render this variable's formula for the JS generator."""
        return _render_js_formula(
            self.statements,
            self.local_names,
            set(self.parameter_dependencies),
        )

    def to_python_formula(self) -> str:
        """Render this variable's formula for the Python generator."""
        return _render_python_formula(
            self.statements,
            set(self.parameter_dependencies),
        )

    def to_lowered_computation(self) -> LoweredComputation:
        """Convert this compiled variable into a lowered computation node."""
        return LoweredComputation(
            name=self.name,
            statements=self.statements,
            local_names=tuple(self.local_names),
            local_value_kinds=dict(self.local_value_kinds),
            input_dependencies=tuple(self.input_dependencies),
            parameter_dependencies=tuple(self.parameter_dependencies),
            variable_dependencies=tuple(self.variable_dependencies),
            value_kind=self.value_kind,
            label=self.label,
            citation=self.citation,
            module_identity=self.module_identity,
        )


@dataclass
class CompiledModule:
    """Shared compile model used by generic JS and Python output."""

    inputs: list[CompiledInput] = field(default_factory=list)
    parameters: list[CompiledParameter] = field(default_factory=list)
    variables: list[CompiledVariable] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    public_outputs: list[CompiledOutput] = field(default_factory=list)

    @classmethod
    def from_rulespec_file(
        cls,
        rulespec_file: RuleSpecFile,
        compile_context: CompileContext | None = None,
        selected_outputs: list[str] | None = None,
    ) -> "CompiledModule":
        """Compile a parsed RuleSpec file into a shared compile model."""
        compile_context = compile_context or CompileContext()
        rule_decls = rulespec_file.rule_decls
        all_rule_names = [rule.name for rule in rule_decls]
        external_rules = {rule.name: rule for rule in rulespec_file.external_rules}
        declared_inputs = _build_declared_inputs(rulespec_file.input_rules)
        computed_rule_names = {rule.name for rule in rulespec_file.computed_rules}
        computed_variables = [
            variable
            for variable in rulespec_file.variables
            if variable.name in computed_rule_names
        ]
        variable_names = [variable.name for variable in computed_variables]
        variable_kind_hints = _build_variable_kind_hints(rulespec_file.variables)
        parameter_kind_hints = _build_parameter_kind_hints(
            external_rules,
            compile_context,
        )
        duplicate_names = sorted(
            name for name in set(all_rule_names) if all_rule_names.count(name) > 1
        )
        if duplicate_names:
            names = ", ".join(duplicate_names)
            raise CompilationError(f"Rules cannot share the same name: {names}.")

        source_citation = rulespec_file.source.citation if rulespec_file.source else ""
        if selected_outputs is None:
            compiled_variables = [
                _compile_variable(
                    variable=variable,
                    parameter_names=set(external_rules),
                    parameter_kind_hints=parameter_kind_hints,
                    variable_names=set(variable_names),
                    variable_kind_hints=variable_kind_hints,
                    source_citation=source_citation,
                    compile_context=compile_context,
                )
                for variable in computed_variables
            ]
        else:
            compiled_variables = _compile_reachable_variables(
                rulespec_file=rulespec_file,
                compile_context=compile_context,
                source_citation=source_citation,
                selected_outputs=selected_outputs,
                parameter_kind_hints=parameter_kind_hints,
                variable_kind_hints=variable_kind_hints,
            )
        ordered_variables = _order_variables(compiled_variables)

        parameter_names = _ordered_unique(
            name
            for variable in ordered_variables
            for name in variable.parameter_dependencies
        )
        parameter_lookup_contracts = _infer_parameter_lookup_contracts(
            ordered_variables,
            set(parameter_names),
        )
        compiled_parameters = [
            _compile_parameter(
                name,
                external_rules[name],
                compile_context,
                lookup_kind=parameter_lookup_contracts[name],
            )
            for name in parameter_names
        ]

        inputs = [
            declared_inputs.get(name, _infer_input(name))
            for name in _ordered_unique(
                name
                for variable in ordered_variables
                for name in variable.input_dependencies
            )
        ]

        module = cls(
            inputs=inputs,
            parameters=compiled_parameters,
            variables=ordered_variables,
        )
        return module.select_outputs(selected_outputs)

    def select_outputs(self, output_names: list[str] | None = None) -> "CompiledModule":
        """Prune this module to the reachable subgraph for selected outputs."""
        requested = _ordered_unique(
            output_names
            if output_names is not None
            else [variable.name for variable in self.variables]
        )
        if not requested:
            raise CompilationError("Select at least one output variable to compile.")

        variables_by_name = {variable.name: variable for variable in self.variables}
        unknown = [name for name in requested if name not in variables_by_name]
        if unknown:
            names = ", ".join(unknown)
            raise CompilationError(f"Unknown output variable(s): {names}.")

        reachable: set[str] = set()
        stack = list(requested)
        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)
            stack.extend(variables_by_name[current].variable_dependencies)

        selected_variables = [
            variable for variable in self.variables if variable.name in reachable
        ]
        parameter_names = _ordered_unique(
            name
            for variable in selected_variables
            for name in variable.parameter_dependencies
        )
        input_names = _ordered_unique(
            name
            for variable in selected_variables
            for name in variable.input_dependencies
        )
        parameters_by_name = {
            parameter.name: parameter for parameter in self.parameters
        }
        inputs_by_name = {
            compiled_input.name: compiled_input for compiled_input in self.inputs
        }

        return CompiledModule(
            inputs=[inputs_by_name[name] for name in input_names],
            parameters=[parameters_by_name[name] for name in parameter_names],
            variables=selected_variables,
            outputs=requested,
            public_outputs=[
                CompiledOutput(
                    name=name,
                    variable_name=name,
                    value_kind=variables_by_name[name].value_kind,
                    module_identity=variables_by_name[name].module_identity,
                )
                for name in requested
            ],
        )

    def with_public_outputs(
        self,
        output_bindings: list[tuple[str, str]],
    ) -> "CompiledModule":
        """Replace public output names while keeping the same compiled graph."""
        variables_by_name = {variable.name: variable for variable in self.variables}
        compiled_outputs: list[CompiledOutput] = []
        seen_public_names: set[str] = set()

        for public_name, variable_name in output_bindings:
            if variable_name not in variables_by_name:
                raise CompilationError(
                    f"Unknown compiled output variable '{variable_name}'."
                )
            if public_name in seen_public_names:
                raise CompilationError(
                    f"Public output name '{public_name}' is defined more than once."
                )
            seen_public_names.add(public_name)
            compiled_outputs.append(
                CompiledOutput(
                    name=public_name,
                    variable_name=variable_name,
                    value_kind=variables_by_name[variable_name].value_kind,
                    module_identity=variables_by_name[variable_name].module_identity,
                )
            )

        return CompiledModule(
            inputs=self.inputs,
            parameters=self.parameters,
            variables=self.variables,
            outputs=[output.variable_name for output in compiled_outputs],
            public_outputs=compiled_outputs,
        )

    def to_lowered_program(self) -> LoweredProgram:
        """Lower this compiled module into a serializable backend-neutral bundle."""
        outputs = self.public_outputs or [
            CompiledOutput(
                name=variable.name,
                variable_name=variable.name,
                value_kind=variable.value_kind,
                module_identity=variable.module_identity,
            )
            for variable in self.variables
        ]
        return LoweredProgram(
            inputs=tuple(
                _lower_input(compiled_input) for compiled_input in self.inputs
            ),
            parameters=tuple(
                LoweredParameter(
                    name=parameter.name,
                    values=dict(parameter.values),
                    source=parameter.source,
                    module_identity=parameter.module_identity,
                    value_kind=parameter.value_kind,
                    lookup_kind=parameter.lookup_kind,
                    index_value_kind=parameter.index_value_kind,
                )
                for parameter in self.parameters
            ),
            computations=tuple(
                variable.to_lowered_computation() for variable in self.variables
            ),
            outputs=tuple(
                LoweredOutput(
                    name=output.name,
                    variable_name=output.variable_name,
                    value_kind=output.value_kind,
                    module_identity=output.module_identity,
                )
                for output in outputs
            ),
        )

    def to_js_generator(self, module_name: str = "calculator") -> JSCodeGenerator:
        """Build a JS generator from this compiled module."""
        return self.to_lowered_program().to_js_generator(module_name=module_name)

    def to_python_generator(
        self, module_name: str = "calculator"
    ) -> PythonCodeGenerator:
        """Build a Python generator from this compiled module."""
        return self.to_lowered_program().to_python_generator(module_name=module_name)

    def to_rust_generator(self, module_name: str = "calculator") -> RustCodeGenerator:
        """Build a Rust generator from this compiled module."""
        return self.to_lowered_program().to_rust_generator(module_name=module_name)


# --- Phase 3: Render helpers ---


def _render_js_formula(
    statements: tuple[Statement, ...],
    local_names: list[str] | tuple[str, ...],
    parameter_names: set[str],
) -> str:
    """Render shared statement IR as a JavaScript formula."""
    if len(statements) == 1 and isinstance(statements[0], ReturnStmt):
        return render_expression_js(statements[0].expression, parameter_names)

    if is_straight_line_formula(statements):
        lines = [
            f"const {statement.name} = "
            f"{render_expression_js(statement.expression, parameter_names)};"
            for statement in statements[:-1]
            if isinstance(statement, AssignStmt)
        ]
        result = statements[-1]
        if not isinstance(result, ReturnStmt):
            raise AssertionError("Straight-line formula must end with return.")
        lines.append(
            f"return {render_expression_js(result.expression, parameter_names)};"
        )
        return "\n".join(lines)

    lines: list[str] = []
    if formula_has_branching(statements) and local_names:
        lines.append(f"let {', '.join(local_names)};")
    lines.extend(render_statement_block_js(statements, parameter_names))
    return "\n".join(lines)


def _render_python_formula(
    statements: tuple[Statement, ...],
    parameter_names: set[str],
) -> str:
    """Render shared statement IR as a Python formula."""
    if len(statements) == 1 and isinstance(statements[0], ReturnStmt):
        return render_expression_python(
            statements[0].expression,
            parameter_names,
        )
    return "\n".join(render_statement_block_python(statements, parameter_names))


# --- Phase 4: Resolve temporal + lower inputs ---


def _lower_input(compiled_input: CompiledInput) -> LoweredInput:
    """Convert one compiled input into a lowered input."""
    value_kind = _input_value_kind(compiled_input)
    return LoweredInput(
        name=compiled_input.name,
        default=compiled_input.default,
        value_kind=value_kind,
        public_name=compiled_input.external_name,
        module_identity=compiled_input.module_identity,
        symbol_name=compiled_input.symbol_name or compiled_input.name,
    )


def _input_value_kind(compiled_input: CompiledInput) -> str:
    """Map compiled input target types to one backend-neutral kind."""
    if compiled_input.js_type == "boolean" and compiled_input.python_type == "bool":
        return "boolean"
    if compiled_input.js_type == "number" and compiled_input.python_type == "int":
        return "integer"
    if compiled_input.js_type == "number" and compiled_input.python_type == "float":
        return "number"
    raise CompilationError(
        f"Input '{compiled_input.name}' has unsupported target-type pair "
        f"{compiled_input.js_type!r}/{compiled_input.python_type!r}."
    )


def _normalize_value_kind(
    value_kind: str,
    *,
    subject: str,
    allowed: set[str] = _LOWERED_VALUE_KINDS,
) -> str:
    """Validate one lowered value kind."""
    if value_kind not in allowed:
        raise CompilationError(f"{subject} has unsupported value kind '{value_kind}'.")
    return value_kind


def _infer_parameter_value_kind_from_values(values: dict[int, float]) -> str:
    """Infer one parameter kind from its resolved numeric values."""
    if values and all(_is_integral_numeric(value) for value in values.values()):
        return "integer"
    return "number"


def _infer_parameter_lookup_kind_from_values(values: dict[int, float]) -> str:
    """Infer one conservative lookup contract from resolved parameter values."""
    if set(values) in (set(), {0}):
        return "scalar"
    return "indexed"


def _normalize_parameter_lookup_kind(raw_lookup_kind: Any, *, name: str) -> str:
    """Validate one lowered parameter lookup contract."""
    if not isinstance(raw_lookup_kind, str):
        raise CompilationError(
            f"Lowered parameter '{name}' has invalid lookup_kind {raw_lookup_kind!r}."
        )
    if raw_lookup_kind not in _LOWERED_PARAMETER_LOOKUP_KINDS:
        raise CompilationError(
            f"Lowered parameter '{name}' has unsupported lookup_kind "
            f"'{raw_lookup_kind}'."
        )
    return raw_lookup_kind


def _normalize_parameter_index_value_kind(
    raw_index_value_kind: Any,
    *,
    name: str,
    lookup_kind: str,
) -> str | None:
    """Validate one lowered parameter index-kind annotation."""
    if lookup_kind == "indexed":
        if raw_index_value_kind is None:
            raise CompilationError(
                f"Lowered parameter '{name}' must define index_value_kind when "
                "lookup_kind is 'indexed'."
            )
        if not isinstance(raw_index_value_kind, str):
            raise CompilationError(
                f"Lowered parameter '{name}' has invalid index_value_kind "
                f"{raw_index_value_kind!r}."
            )
        if raw_index_value_kind not in _LOWERED_PARAMETER_INDEX_VALUE_KINDS:
            raise CompilationError(
                f"Lowered parameter '{name}' has unsupported index_value_kind "
                f"'{raw_index_value_kind}'."
            )
        return raw_index_value_kind
    if raw_index_value_kind is not None:
        raise CompilationError(
            f"Lowered parameter '{name}' cannot define index_value_kind unless "
            "lookup_kind is 'indexed'."
        )
    return None


def _is_integral_numeric(value: Any) -> bool:
    """Return whether one numeric payload represents an exact integer."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False


def _normalize_local_value_kinds(
    raw_value_kinds: Any,
    *,
    computation_name: str,
    statements: tuple[Statement, ...],
    local_names: tuple[str, ...],
    input_value_kinds: dict[str, str],
    parameter_value_kinds: dict[str, str],
    variable_kind_hints: dict[str, str],
) -> dict[str, str]:
    """Validate or infer one computation's local value-kind map."""
    allowed_names = set(local_names)
    if raw_value_kinds is None:
        inferred = _infer_statement_local_value_kinds(
            statements,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_kind_hints=variable_kind_hints,
        )
        missing = sorted(allowed_names - set(inferred))
        if missing:
            names = ", ".join(missing)
            raise CompilationError(
                f"Lowered computation '{computation_name}' is missing inferable local "
                f"value kinds for: {names}."
            )
        return {name: inferred[name] for name in local_names}
    if not isinstance(raw_value_kinds, dict):
        raise CompilationError(
            f"Lowered computation '{computation_name}' must define local_value_kinds "
            "as an object."
        )
    normalized: dict[str, str] = {}
    for local_name, raw_kind in raw_value_kinds.items():
        if local_name not in allowed_names:
            raise CompilationError(
                f"Lowered computation '{computation_name}' defines a local value kind "
                f"for unknown local '{local_name}'."
            )
        normalized[local_name] = _normalize_value_kind(
            raw_kind,
            subject=(f"Lowered computation '{computation_name}' local '{local_name}'"),
        )
    missing = sorted(allowed_names - set(normalized))
    if missing:
        names = ", ".join(missing)
        raise CompilationError(
            f"Lowered computation '{computation_name}' is missing local value kinds "
            f"for: {names}."
        )
    return normalized


# --- Phase 5: Resolve bindings + infer kinds ---


@dataclass(frozen=True)
class _StatementKindAnalysis:
    """Shared value-kind analysis for one validated statement block."""

    local_value_kinds: dict[str, str]
    return_value_kind: str | None


def _build_variable_kind_hints(variables: list[VariableBlock]) -> dict[str, str]:
    """Collect declared value-kind hints from parsed variable dtypes."""
    hints: dict[str, str] = {}
    for variable in variables:
        declared = _declared_rule_value_kind(variable)
        if declared is not None:
            hints[variable.name] = declared
    return hints


def _build_parameter_kind_hints(
    rules: dict[str, "RuleDecl"],
    compile_context: CompileContext,
) -> dict[str, str]:
    """Collect inferred value kinds for parsed parameters."""
    hints: dict[str, str] = {}
    for name, rule in rules.items():
        try:
            hints[name] = _resolve_parameter_value_kind(
                name,
                rule,
                compile_context,
            )
        except (CompilationError, RuleBindingError):
            # Parameter resolution still happens later for referenced parameters.
            # Keep unused or unreachable source-only parameters from failing eager
            # kind collection, and fall back conservatively until the concrete
            # parameter is actually compiled.
            hints[name] = "number"
    return hints


def _declared_rule_value_kind(rule: VariableBlock | "RuleDecl") -> str | None:
    """Map one parsed rule dtype to one lowered value kind."""
    if not rule.dtype:
        return None

    normalized = rule.dtype.strip().lower()
    if normalized in {"bool", "boolean"}:
        return "boolean"
    if normalized in {"int", "integer", "count", "index"}:
        return "integer"
    if normalized in {
        "amount",
        "currency",
        "decimal",
        "float",
        "money",
        "number",
        "percent",
        "percentage",
        "rate",
    }:
        return "number"
    if normalized in {"str", "string", "text"}:
        return "string"
    return None


def _resolve_variable_value_kind(
    variable: VariableBlock,
    statements: tuple[Statement, ...],
    parameter_value_kinds: dict[str, str],
    variable_kind_hints: dict[str, str],
) -> str:
    """Resolve one compiled variable's backend-neutral result kind."""
    declared = _declared_rule_value_kind(variable)
    if declared is not None:
        return declared
    inferred = _analyze_statement_kinds(
        statements,
        input_value_kinds={},
        parameter_value_kinds=parameter_value_kinds,
        variable_kind_hints=variable_kind_hints,
    ).return_value_kind
    return inferred or "number"


def _infer_statement_local_value_kinds(
    statements: tuple[Statement, ...],
    *,
    input_value_kinds: dict[str, str] | None = None,
    parameter_value_kinds: dict[str, str],
    variable_kind_hints: dict[str, str],
) -> dict[str, str]:
    """Infer stable local slot kinds for one statement block."""
    return _analyze_statement_kinds(
        statements,
        input_value_kinds=input_value_kinds or {},
        parameter_value_kinds=parameter_value_kinds,
        variable_kind_hints=variable_kind_hints,
    ).local_value_kinds


def _infer_statement_value_kind(
    statements: tuple[Statement, ...],
    *,
    input_value_kinds: dict[str, str] | None = None,
    parameter_value_kinds: dict[str, str],
    variable_kind_hints: dict[str, str],
) -> str | None:
    """Infer the result kind of a statement block."""
    return _analyze_statement_kinds(
        statements,
        input_value_kinds=input_value_kinds or {},
        parameter_value_kinds=parameter_value_kinds,
        variable_kind_hints=variable_kind_hints,
    ).return_value_kind


def _analyze_statement_kinds(
    statements: tuple[Statement, ...],
    *,
    input_value_kinds: dict[str, str],
    parameter_value_kinds: dict[str, str],
    variable_kind_hints: dict[str, str],
) -> _StatementKindAnalysis:
    """Infer stable local-slot kinds plus one statement block return kind."""

    def walk_sequence(
        sequence: tuple[Statement, ...],
        local_kinds: dict[str, str],
    ) -> tuple[dict[str, str], dict[str, str], list[str]]:
        current = dict(local_kinds)
        observed: dict[str, str] = {}
        return_kinds: list[str] = []

        for statement in sequence:
            if isinstance(statement, AssignStmt):
                assignment_kind = _infer_expression_value_kind(
                    statement.expression,
                    local_kinds=current,
                    input_value_kinds=input_value_kinds,
                    parameter_value_kinds=parameter_value_kinds,
                    variable_kind_hints=variable_kind_hints,
                )
                current[statement.name] = assignment_kind
                observed[statement.name] = _merge_observed_local_kind(
                    statement.name,
                    observed.get(statement.name),
                    assignment_kind,
                )
                continue

            if isinstance(statement, ReturnStmt):
                return_kinds.append(
                    _infer_expression_value_kind(
                        statement.expression,
                        local_kinds=current,
                        input_value_kinds=input_value_kinds,
                        parameter_value_kinds=parameter_value_kinds,
                        variable_kind_hints=variable_kind_hints,
                    )
                )
                break

            if isinstance(statement, IfStmt):
                body_locals, body_observed, body_return_kinds = walk_sequence(
                    statement.body,
                    dict(current),
                )
                if statement.orelse:
                    else_locals, else_observed, else_return_kinds = walk_sequence(
                        statement.orelse,
                        dict(current),
                    )
                else:
                    else_locals, else_observed, else_return_kinds = (
                        dict(current),
                        {},
                        [],
                    )

                current = _merge_local_kinds(body_locals, else_locals)
                observed = _merge_observed_local_value_kinds(
                    observed,
                    body_observed,
                    else_observed,
                )
                return_kinds.extend(body_return_kinds)
                return_kinds.extend(else_return_kinds)
                continue

            raise AssertionError(
                f"Unhandled statement node: {type(statement).__name__}"
            )

        return current, observed, return_kinds

    _, observed_locals, return_kinds = walk_sequence(statements, {})
    return _StatementKindAnalysis(
        local_value_kinds=dict(observed_locals),
        return_value_kind=(
            _combine_value_kinds(return_kinds, fallback="number")
            if return_kinds
            else None
        ),
    )


def _merge_local_kinds(
    left: dict[str, str],
    right: dict[str, str],
) -> dict[str, str]:
    """Merge local-kind environments across control-flow branches."""
    merged: dict[str, str] = {}
    for name in left.keys() & right.keys():
        merged[name] = _merge_observed_local_kind(
            name,
            left[name],
            right[name],
        )
    return merged


def _merge_observed_local_kind(
    local_name: str,
    existing: str | None,
    new_kind: str,
) -> str:
    """Merge repeated assignments to one local into one stable inferred kind."""
    if existing is None:
        return new_kind
    if existing == new_kind:
        return existing
    if {existing, new_kind} <= {"integer", "number"}:
        return "number"
    raise CompilationError(
        f"Local '{local_name}' is assigned incompatible value kinds "
        f"'{existing}' and '{new_kind}'."
    )


def _merge_observed_local_value_kinds(
    *maps: dict[str, str],
) -> dict[str, str]:
    """Merge observed local assignment kinds across nested control flow."""
    merged: dict[str, str] = {}
    for current in maps:
        for name, kind in current.items():
            merged[name] = _merge_observed_local_kind(name, merged.get(name), kind)
    return merged


def _infer_expression_value_kind(
    expression: Expression,
    *,
    local_kinds: dict[str, str],
    input_value_kinds: dict[str, str],
    parameter_value_kinds: dict[str, str],
    variable_kind_hints: dict[str, str],
) -> str:
    """Infer one expression node's backend-neutral value kind."""
    if isinstance(expression, LiteralExpr):
        if isinstance(expression.value, bool):
            return "boolean"
        if isinstance(expression.value, int):
            return "integer"
        if isinstance(expression.value, float):
            return "number"
        if isinstance(expression.value, str):
            return "string"
        return "number"

    if isinstance(expression, NameExpr):
        if expression.name in local_kinds:
            return local_kinds[expression.name]
        if expression.name in input_value_kinds:
            return input_value_kinds[expression.name]
        if expression.name in parameter_value_kinds:
            return parameter_value_kinds[expression.name]
        if expression.name in variable_kind_hints:
            return variable_kind_hints[expression.name]
        return _input_value_kind(_infer_input(expression.name))

    if isinstance(expression, SubscriptExpr):
        if isinstance(expression.value, NameExpr):
            return parameter_value_kinds.get(expression.value.name, "number")
        return "number"

    if isinstance(expression, CallExpr):
        if expression.function in {"ceil", "floor", "round"}:
            return "integer"
        if expression.function in {"abs", "max", "min"}:
            argument_kinds = [
                _infer_expression_value_kind(
                    argument,
                    local_kinds=local_kinds,
                    input_value_kinds=input_value_kinds,
                    parameter_value_kinds=parameter_value_kinds,
                    variable_kind_hints=variable_kind_hints,
                )
                for argument in expression.arguments
            ]
            return _combine_value_kinds(argument_kinds, fallback="number")
        return "number"

    if isinstance(expression, UnaryExpr):
        if expression.operator == "not":
            return "boolean"
        return _infer_expression_value_kind(
            expression.operand,
            local_kinds=local_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_kind_hints=variable_kind_hints,
        )

    if isinstance(expression, BinaryExpr):
        left_kind = _infer_expression_value_kind(
            expression.left,
            local_kinds=local_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_kind_hints=variable_kind_hints,
        )
        right_kind = _infer_expression_value_kind(
            expression.right,
            local_kinds=local_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_kind_hints=variable_kind_hints,
        )
        if expression.operator == "+" and left_kind == right_kind == "string":
            return "string"
        if expression.operator in {"/", "**"}:
            return "number"
        if left_kind == right_kind == "integer" and expression.operator in {
            "+",
            "-",
            "*",
            "%",
        }:
            return "integer"
        return _combine_value_kinds([left_kind, right_kind], fallback="number")

    if isinstance(expression, BoolExpr | CompareExpr):
        return "boolean"

    if isinstance(expression, ConditionalExpr):
        true_kind = _infer_expression_value_kind(
            expression.if_true,
            local_kinds=local_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_kind_hints=variable_kind_hints,
        )
        false_kind = _infer_expression_value_kind(
            expression.if_false,
            local_kinds=local_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_kind_hints=variable_kind_hints,
        )
        return _combine_value_kinds([true_kind, false_kind], fallback=true_kind)

    raise AssertionError(f"Unhandled expression node: {type(expression).__name__}")


def _combine_value_kinds(kinds: list[str], fallback: str) -> str:
    """Combine several value kinds into one conservative shared kind."""
    unique = {kind for kind in kinds if kind}
    if not unique:
        return fallback
    if len(unique) == 1:
        return next(iter(unique))
    if unique <= {"integer", "number"}:
        return "number"
    return fallback


# --- Phase 6: Lower / emit (LoweredProgram serialization) ---


def _statement_to_dict(statement: Statement) -> dict[str, Any]:
    """Serialize one statement node."""
    if isinstance(statement, AssignStmt):
        return {
            "kind": "assign",
            "name": statement.name,
            "expression": _expression_to_dict(statement.expression),
        }
    if isinstance(statement, ReturnStmt):
        return {
            "kind": "return",
            "expression": _expression_to_dict(statement.expression),
        }
    if isinstance(statement, IfStmt):
        return {
            "kind": "if",
            "condition": _expression_to_dict(statement.condition),
            "body": [_statement_to_dict(item) for item in statement.body],
            "orelse": [_statement_to_dict(item) for item in statement.orelse],
        }
    raise AssertionError(f"Unhandled statement node: {type(statement).__name__}")


def _statement_from_dict(payload: dict[str, Any]) -> Statement:
    """Deserialize one statement node."""
    payload = _require_object(payload, "Lowered statement")
    kind = payload.get("kind")
    if kind == "assign":
        return AssignStmt(
            name=_require_field(payload, "name", "Lowered assign statement"),
            expression=_expression_from_dict(
                _require_field(
                    payload,
                    "expression",
                    "Lowered assign statement",
                )
            ),
        )
    if kind == "return":
        return ReturnStmt(
            expression=_expression_from_dict(
                _require_field(
                    payload,
                    "expression",
                    "Lowered return statement",
                )
            ),
        )
    if kind == "if":
        return IfStmt(
            condition=_expression_from_dict(
                _require_field(payload, "condition", "Lowered if statement")
            ),
            body=tuple(
                _statement_from_dict(item)
                for item in _require_list(payload.get("body", []), "Lowered if body")
            ),
            orelse=tuple(
                _statement_from_dict(item)
                for item in _require_list(
                    payload.get("orelse", []),
                    "Lowered if orelse",
                )
            ),
        )
    raise CompilationError(f"Unknown lowered statement kind '{kind}'.")


def _expression_to_dict(expression: Expression) -> dict[str, Any]:
    """Serialize one expression node."""
    if isinstance(expression, LiteralExpr):
        return {"kind": "literal", "value": expression.value}
    if isinstance(expression, NameExpr):
        return {"kind": "name", "name": expression.name}
    if isinstance(expression, SubscriptExpr):
        return {
            "kind": "subscript",
            "value": _expression_to_dict(expression.value),
            "index": _expression_to_dict(expression.index),
        }
    if isinstance(expression, CallExpr):
        return {
            "kind": "call",
            "function": expression.function,
            "arguments": [
                _expression_to_dict(argument) for argument in expression.arguments
            ],
        }
    if isinstance(expression, UnaryExpr):
        return {
            "kind": "unary",
            "operator": expression.operator,
            "operand": _expression_to_dict(expression.operand),
        }
    if isinstance(expression, BinaryExpr):
        return {
            "kind": "binary",
            "left": _expression_to_dict(expression.left),
            "operator": expression.operator,
            "right": _expression_to_dict(expression.right),
        }
    if isinstance(expression, BoolExpr):
        return {
            "kind": "bool",
            "operator": expression.operator,
            "values": [_expression_to_dict(value) for value in expression.values],
        }
    if isinstance(expression, CompareExpr):
        return {
            "kind": "compare",
            "left": _expression_to_dict(expression.left),
            "operators": list(expression.operators),
            "comparators": [
                _expression_to_dict(value) for value in expression.comparators
            ],
        }
    if isinstance(expression, ConditionalExpr):
        return {
            "kind": "conditional",
            "condition": _expression_to_dict(expression.condition),
            "if_true": _expression_to_dict(expression.if_true),
            "if_false": _expression_to_dict(expression.if_false),
        }
    raise AssertionError(f"Unhandled expression node: {type(expression).__name__}")


def _expression_from_dict(payload: dict[str, Any]) -> Expression:
    """Deserialize one expression node."""
    payload = _require_object(payload, "Lowered expression")
    kind = payload.get("kind")
    if kind == "literal":
        return LiteralExpr(payload.get("value"))
    if kind == "name":
        return NameExpr(_require_field(payload, "name", "Lowered name expression"))
    if kind == "subscript":
        return SubscriptExpr(
            value=_expression_from_dict(
                _require_field(payload, "value", "Lowered subscript expression")
            ),
            index=_expression_from_dict(
                _require_field(payload, "index", "Lowered subscript expression")
            ),
        )
    if kind == "call":
        return CallExpr(
            function=_require_field(payload, "function", "Lowered call expression"),
            arguments=tuple(
                _expression_from_dict(argument)
                for argument in _require_list(
                    payload.get("arguments", []),
                    "Lowered call arguments",
                )
            ),
        )
    if kind == "unary":
        return UnaryExpr(
            operator=_require_field(payload, "operator", "Lowered unary expression"),
            operand=_expression_from_dict(
                _require_field(payload, "operand", "Lowered unary expression")
            ),
        )
    if kind == "binary":
        return BinaryExpr(
            left=_expression_from_dict(
                _require_field(payload, "left", "Lowered binary expression")
            ),
            operator=_require_field(payload, "operator", "Lowered binary expression"),
            right=_expression_from_dict(
                _require_field(payload, "right", "Lowered binary expression")
            ),
        )
    if kind == "bool":
        return BoolExpr(
            operator=_require_field(payload, "operator", "Lowered boolean expression"),
            values=tuple(
                _expression_from_dict(value)
                for value in _require_list(
                    payload.get("values", []),
                    "Lowered boolean values",
                )
            ),
        )
    if kind == "compare":
        return CompareExpr(
            left=_expression_from_dict(
                _require_field(payload, "left", "Lowered compare expression")
            ),
            operators=tuple(
                _require_list(
                    payload.get("operators", []),
                    "Lowered compare operators",
                )
            ),
            comparators=tuple(
                _expression_from_dict(value)
                for value in _require_list(
                    payload.get("comparators", []),
                    "Lowered compare comparators",
                )
            ),
        )
    if kind == "conditional":
        return ConditionalExpr(
            condition=_expression_from_dict(
                _require_field(payload, "condition", "Lowered conditional expression")
            ),
            if_true=_expression_from_dict(
                _require_field(payload, "if_true", "Lowered conditional expression")
            ),
            if_false=_expression_from_dict(
                _require_field(payload, "if_false", "Lowered conditional expression")
            ),
        )
    raise CompilationError(f"Unknown lowered expression kind '{kind}'.")


def _require_object(payload: Any, subject: str) -> dict[str, Any]:
    """Require one lowered JSON node to be an object."""
    if not isinstance(payload, dict):
        raise CompilationError(f"{subject} must be an object.")
    return payload


def _require_field(payload: dict[str, Any], name: str, subject: str) -> Any:
    """Require one field from a lowered JSON object."""
    try:
        return payload[name]
    except KeyError as exc:
        raise CompilationError(
            f"{subject} is missing required field '{name}'."
        ) from exc


def _require_list(value: Any, subject: str) -> list[Any]:
    """Require one lowered JSON field to be a list."""
    if not isinstance(value, list):
        raise CompilationError(f"{subject} must be a list.")
    return value


# --- Phase 7: Compile driver helpers ---


def _compile_reachable_variables(
    rulespec_file: RuleSpecFile,
    compile_context: CompileContext,
    source_citation: str,
    selected_outputs: list[str],
    parameter_kind_hints: dict[str, str],
    variable_kind_hints: dict[str, str],
) -> list[CompiledVariable]:
    """Compile only variables reachable from the requested outputs."""
    external_rules = {rule.name: rule for rule in rulespec_file.external_rules}
    computed_rule_names = {rule.name for rule in rulespec_file.computed_rules}
    variables_by_name = {
        variable.name: variable
        for variable in rulespec_file.variables
        if variable.name in computed_rule_names
    }
    unknown = [
        name
        for name in _ordered_unique(selected_outputs)
        if name not in variables_by_name
    ]
    if unknown:
        names = ", ".join(unknown)
        raise CompilationError(f"Unknown output variable(s): {names}.")

    parameter_names = set(external_rules)
    variable_names = set(variables_by_name)
    compiled_by_name: dict[str, CompiledVariable] = {}
    pending = list(_ordered_unique(selected_outputs))

    while pending:
        name = pending.pop()
        if name in compiled_by_name:
            continue
        compiled = _compile_variable(
            variable=variables_by_name[name],
            parameter_names=parameter_names,
            parameter_kind_hints=parameter_kind_hints,
            variable_names=variable_names,
            variable_kind_hints=variable_kind_hints,
            source_citation=source_citation,
            compile_context=compile_context,
        )
        compiled_by_name[name] = compiled
        pending.extend(
            dependency
            for dependency in compiled.variable_dependencies
            if dependency not in compiled_by_name
        )

    return list(compiled_by_name.values())


def _build_declared_inputs(
    rules: list["RuleDecl"],
) -> dict[str, CompiledInput]:
    """Collect typed declared-input rules from parsed variable blocks."""
    return {rule.name: _compile_declared_input(rule) for rule in rules}


def _input_public_name(rule: "RuleDecl") -> str:
    """Return the user-facing name for one compiled input rule."""
    symbol_name = rule.symbol_name or rule.name
    if rule.name == symbol_name:
        return symbol_name
    if rule.module_identity:
        return f"{rule.module_identity}.{symbol_name}"
    return symbol_name


def _infer_input_symbol_name(
    *,
    name: str,
    public_name: str,
    module_identity: str,
) -> str:
    """Infer one input symbol name when loading older lowered bundles."""
    if module_identity and public_name.startswith(f"{module_identity}."):
        return public_name[len(module_identity) + 1 :]
    return public_name or name


def _compile_declared_input(rule: "RuleDecl") -> CompiledInput:
    """Compile a no-formula variable declaration into one typed public input."""
    inferred = _infer_input(rule.name)
    value_kind = _declared_rule_value_kind(rule) or _input_value_kind(inferred)
    default = _resolve_declared_input_default(rule, value_kind, inferred.default)
    public_name = _input_public_name(rule)
    symbol_name = rule.symbol_name or rule.name
    if value_kind == "boolean":
        return CompiledInput(
            name=rule.name,
            default=default,
            js_type="boolean",
            python_type="bool",
            public_name=public_name,
            module_identity=rule.module_identity,
            symbol_name=symbol_name,
        )
    if value_kind == "integer":
        return CompiledInput(
            name=rule.name,
            default=default,
            js_type="number",
            python_type="int",
            public_name=public_name,
            module_identity=rule.module_identity,
            symbol_name=symbol_name,
        )
    if value_kind == "number":
        return CompiledInput(
            name=rule.name,
            default=default,
            js_type="number",
            python_type="float",
            public_name=public_name,
            module_identity=rule.module_identity,
            symbol_name=symbol_name,
        )
    raise CompilationError(
        f"Rule '{rule.name}' declares unsupported input dtype "
        f"'{rule.dtype}'. Generic compilation currently supports only "
        "boolean and numeric declared inputs."
    )


def _resolve_declared_input_default(
    rule: "RuleDecl",
    value_kind: str,
    fallback: Any,
) -> Any:
    """Normalize one declared input's explicit default or inferred fallback."""
    default = rule.default
    if default is None:
        return fallback
    if value_kind == "boolean":
        if isinstance(default, bool):
            return default
        if isinstance(default, (int, float)) and default in {0, 1, 0.0, 1.0}:
            return bool(default)
        raise CompilationError(
            f"Rule '{rule.name}' default must be boolean, got {default!r}."
        )
    if value_kind == "integer":
        if isinstance(default, bool):
            raise CompilationError(
                f"Rule '{rule.name}' integer default cannot be boolean."
            )
        if isinstance(default, int):
            return default
        if isinstance(default, float) and default.is_integer():
            return int(default)
        raise CompilationError(
            f"Rule '{rule.name}' default must be an integer, got {default!r}."
        )
    if value_kind == "number":
        if isinstance(default, bool):
            raise CompilationError(
                f"Rule '{rule.name}' numeric default cannot be boolean."
            )
        if isinstance(default, (int, float)):
            return default
        raise CompilationError(
            f"Rule '{rule.name}' default must be numeric, got {default!r}."
        )
    return default


def _compile_parameter(
    name: str,
    rule: "RuleDecl",
    compile_context: CompileContext,
    *,
    lookup_kind: str,
) -> CompiledParameter:
    """Compile a parameter into a concrete generator-friendly form."""
    value_kind = _resolve_parameter_value_kind(name, rule, compile_context)
    if rule.temporal:
        active_entry = _resolve_temporal_entry(
            name,
            list(rule.temporal),
            compile_context,
            subject="Parameter",
        )
        if active_entry.code:
            raise CompilationError(
                f"Parameter '{name}' uses code blocks in temporal entries. "
                "Generic compilation currently supports only inline numeric values."
            )
        if active_entry.value is None:
            raise CompilationError(
                f"Parameter '{name}' does not resolve to a numeric value."
            )
        values = {0: float(active_entry.value)}
        _validate_parameter_lookup_contract(name, values, lookup_kind)
        return CompiledParameter(
            name=name,
            values=values,
            source=rule.source,
            module_identity=rule.module_identity,
            value_kind=value_kind,
            lookup_kind=lookup_kind,
            index_value_kind=_index_value_kind_for_lookup(lookup_kind),
        )

    if rule.values:
        values = {index: float(value) for index, value in rule.values.items()}
        _validate_parameter_lookup_contract(name, values, lookup_kind)
        return CompiledParameter(
            name=name,
            values=values,
            source=rule.source,
            module_identity=rule.module_identity,
            value_kind=value_kind,
            lookup_kind=lookup_kind,
            index_value_kind=_index_value_kind_for_lookup(lookup_kind),
        )

    resolved_binding = _resolve_external_rule_binding(rule, compile_context)
    if resolved_binding is not None:
        _validate_parameter_lookup_contract(name, resolved_binding.values, lookup_kind)
        return CompiledParameter(
            name=name,
            values=resolved_binding.values,
            source=_bound_external_rule_source(rule.source, resolved_binding),
            module_identity=rule.module_identity,
            value_kind=value_kind,
            lookup_kind=lookup_kind,
            index_value_kind=_index_value_kind_for_lookup(lookup_kind),
        )

    binding_target = _external_rule_binding_target(name, rule)
    raise CompilationError(
        f"External rule '{name}' is referenced but has no inline numeric values. "
        "Supply a rule binding such as --binding "
        f"{binding_target}=VALUE, --binding-file bindings.json, or pass "
        f"rule_bindings={{'{binding_target}': VALUE}}."
    )


def _resolve_parameter_value_kind(
    name: str,
    rule: "RuleDecl",
    compile_context: CompileContext,
) -> str:
    """Resolve one parameter's lowered numeric kind."""
    if rule.temporal:
        active_entry = _resolve_temporal_entry(
            name,
            list(rule.temporal),
            compile_context,
            subject="Parameter",
        )
        if active_entry.code:
            raise CompilationError(
                f"Parameter '{name}' uses code blocks in temporal entries. "
                "Generic compilation currently supports only inline numeric values."
            )
        if active_entry.value is None:
            raise CompilationError(
                f"Parameter '{name}' does not resolve to a numeric value."
            )
        return _infer_parameter_value_kind_from_values({0: active_entry.value})

    if rule.values:
        return _infer_parameter_value_kind_from_values(rule.values)

    resolved_binding = _resolve_external_rule_binding(rule, compile_context)
    if resolved_binding is not None:
        return _infer_parameter_value_kind_from_values(resolved_binding.values)

    binding_target = _external_rule_binding_target(name, rule)
    raise CompilationError(
        f"External rule '{name}' is referenced but has no inline numeric values. "
        "Supply a rule binding such as --binding "
        f"{binding_target}=VALUE, --binding-file bindings.json, or pass "
        f"rule_bindings={{'{binding_target}': VALUE}}."
    )


def _index_value_kind_for_lookup(lookup_kind: str) -> str | None:
    """Return the index-kind annotation for one normalized lookup contract."""
    if lookup_kind == "indexed":
        return "integer"
    return None


def _validate_parameter_lookup_contract(
    name: str,
    values: dict[int, float],
    lookup_kind: str,
) -> None:
    """Validate that resolved values match the parameter lookup contract."""
    if lookup_kind != "scalar":
        return
    if set(values) != {0}:
        indexes = ", ".join(str(index) for index in sorted(values))
        raise CompilationError(
            f"Parameter '{name}' is used as a scalar value but resolves to indexed "
            f"entries ({indexes}). Use {name}[index] or bind a single scalar value."
        )


def _validate_lowered_parameter_contracts(
    parameters: tuple[LoweredParameter, ...],
    computations: tuple[LoweredComputation, ...],
) -> tuple[LoweredParameter, ...]:
    """Validate lowered parameter metadata against actual computation usage."""
    lookup_kinds = _infer_parameter_lookup_contracts(
        computations,
        {parameter.name for parameter in parameters},
    )
    normalized: list[LoweredParameter] = []
    for parameter in parameters:
        actual_lookup = lookup_kinds.get(parameter.name)
        if actual_lookup is not None and actual_lookup != parameter.lookup_kind:
            raise CompilationError(
                f"Lowered parameter '{parameter.name}' is declared as "
                f"lookup_kind='{parameter.lookup_kind}' but computations use it as "
                f"'{actual_lookup}'."
            )
        _validate_parameter_lookup_contract(
            parameter.name,
            parameter.values,
            parameter.lookup_kind,
        )
        normalized.append(
            LoweredParameter(
                name=parameter.name,
                values=dict(parameter.values),
                source=parameter.source,
                module_identity=parameter.module_identity,
                value_kind=parameter.value_kind,
                lookup_kind=parameter.lookup_kind,
                index_value_kind=_normalize_parameter_index_value_kind(
                    parameter.index_value_kind,
                    name=parameter.name,
                    lookup_kind=parameter.lookup_kind,
                ),
            )
        )
    return tuple(normalized)


def _infer_parameter_lookup_contracts(
    computations: list[CompiledVariable] | tuple[LoweredComputation, ...],
    parameter_names: set[str],
) -> dict[str, str]:
    """Infer one lookup contract per referenced parameter from computation IR."""
    lookup_kinds: dict[str, str] = {}
    for computation in computations:
        _collect_parameter_lookup_kinds_from_statements(
            computation.statements,
            parameter_names,
            lookup_kinds,
        )
    return lookup_kinds


def _collect_parameter_lookup_kinds_from_statements(
    statements: tuple[Statement, ...],
    parameter_names: set[str],
    lookup_kinds: dict[str, str],
) -> None:
    """Walk statement IR and record whether parameters are scalar or indexed."""

    def note_lookup(name: str, lookup_kind: str) -> None:
        existing = lookup_kinds.get(name)
        if existing is None:
            lookup_kinds[name] = lookup_kind
            return
        if existing != lookup_kind:
            raise CompilationError(
                f"Parameter '{name}' is used both as a scalar value and an indexed "
                "lookup. Use either name or name[index], not both."
            )

    def walk_expression(expression: Expression) -> None:
        if isinstance(expression, NameExpr):
            if expression.name in parameter_names:
                note_lookup(expression.name, "scalar")
            return
        if isinstance(expression, SubscriptExpr):
            if (
                isinstance(expression.value, NameExpr)
                and expression.value.name in parameter_names
            ):
                note_lookup(expression.value.name, "indexed")
                walk_expression(expression.index)
                return
            walk_expression(expression.value)
            walk_expression(expression.index)
            return
        if isinstance(expression, CallExpr):
            for argument in expression.arguments:
                walk_expression(argument)
            return
        if isinstance(expression, UnaryExpr):
            walk_expression(expression.operand)
            return
        if isinstance(expression, BinaryExpr):
            walk_expression(expression.left)
            walk_expression(expression.right)
            return
        if isinstance(expression, BoolExpr):
            for value in expression.values:
                walk_expression(value)
            return
        if isinstance(expression, CompareExpr):
            walk_expression(expression.left)
            for comparator in expression.comparators:
                walk_expression(comparator)
            return
        if isinstance(expression, ConditionalExpr):
            walk_expression(expression.condition)
            walk_expression(expression.if_true)
            walk_expression(expression.if_false)
            return

    def walk_statements(block: tuple[Statement, ...]) -> None:
        for statement in block:
            if isinstance(statement, AssignStmt):
                walk_expression(statement.expression)
                continue
            if isinstance(statement, ReturnStmt):
                walk_expression(statement.expression)
                continue
            if isinstance(statement, IfStmt):
                walk_expression(statement.condition)
                walk_statements(statement.body)
                walk_statements(statement.orelse)
                continue
            raise AssertionError(
                f"Unhandled statement node: {type(statement).__name__}"
            )

    walk_statements(statements)


def _compile_variable(
    variable: VariableBlock,
    parameter_names: set[str],
    parameter_kind_hints: dict[str, str],
    variable_names: set[str],
    variable_kind_hints: dict[str, str],
    source_citation: str,
    compile_context: CompileContext,
) -> CompiledVariable:
    """Compile a parsed variable into the shared model."""
    formula = _resolve_variable_formula(variable, compile_context).strip()
    if not formula:
        raise CompilationError(f"Variable '{variable.name}' has no formula to compile.")

    statements = _parse_formula_block(variable.name, formula)
    local_names = collect_assigned_names(statements)
    reserved_names = (
        set(variable.unqualified_bindings)
        | set(variable.qualified_bindings)
        | parameter_names
        | (variable_names - {variable.name})
    )
    for local_name in local_names:
        if local_name in reserved_names:
            raise CompilationError(
                f"Variable '{variable.name}' assigns to '{local_name}', which "
                "shadows a parameter, imported module alias, or another compiled "
                "variable."
            )
    statements = _bind_statement_references(
        variable_name=variable.name,
        statements=statements,
        unqualified_bindings=variable.unqualified_bindings,
        qualified_bindings=variable.qualified_bindings,
    )
    input_dependencies: list[str] = []
    parameter_dependencies: list[str] = []
    variable_dependencies: list[str] = []

    def record_reference(name: str) -> None:
        if name == variable.name:
            return
        if name in parameter_names:
            _append_unique(parameter_dependencies, name)
            return
        if name in variable_names:
            _append_unique(variable_dependencies, name)
            return
        _append_unique(input_dependencies, name)

    _analyze_statement_dependencies(
        variable_name=variable.name,
        statements=statements,
        local_names=set(local_names),
        record_reference=record_reference,
    )
    kind_analysis = _analyze_statement_kinds(
        statements=statements,
        input_value_kinds={},
        parameter_value_kinds=parameter_kind_hints,
        variable_kind_hints=variable_kind_hints,
    )
    value_kind = _declared_rule_value_kind(variable) or (
        kind_analysis.return_value_kind or "number"
    )

    return CompiledVariable(
        name=variable.name,
        statements=statements,
        local_names=local_names,
        local_value_kinds=kind_analysis.local_value_kinds,
        input_dependencies=input_dependencies,
        parameter_dependencies=parameter_dependencies,
        variable_dependencies=variable_dependencies,
        value_kind=value_kind,
        label=variable.label or "",
        citation=variable.source_citation or source_citation,
        module_identity=variable.module_identity,
    )


def _resolve_variable_formula(
    variable: VariableBlock,
    compile_context: CompileContext,
) -> str:
    """Resolve the active formula for a variable."""
    if variable.formula:
        return variable.formula
    if not variable.temporal:
        return ""
    entry = _resolve_temporal_entry(
        variable.name,
        variable.temporal,
        compile_context,
        subject="Variable",
    )
    if entry.code:
        return entry.code
    if entry.value is not None:
        return str(entry.value)
    return ""


def _resolve_temporal_entry(
    name: str,
    temporal_entries: list[Any],
    compile_context: CompileContext,
    subject: str,
):
    """Resolve a temporal entry against an optional effective date."""
    if len(temporal_entries) == 1:
        return temporal_entries[0]

    if compile_context.effective_date is None:
        raise CompilationError(
            f"{subject} '{name}' has multiple temporal entries. "
            "Pass an effective date to choose which version to compile."
        )

    eligible_entries = [
        entry
        for entry in temporal_entries
        if _parse_from_date(entry.from_date) <= compile_context.effective_date
    ]
    if not eligible_entries:
        raise CompilationError(
            f"{subject} '{name}' has no temporal entry active on "
            f"{compile_context.effective_date.isoformat()}."
        )

    return max(eligible_entries, key=lambda entry: entry.from_date)


def _parse_from_date(value: str) -> date:
    """Parse a RuleSpec from-date string."""
    return date.fromisoformat(value)


def _parse_formula_block(
    variable_name: str,
    formula: str,
) -> tuple[Statement, ...]:
    """Parse a formula block into validated statements."""
    if not formula.strip():
        raise CompilationError(f"Variable '{variable_name}' has an empty formula.")
    try:
        return parse_formula_statements(formula, variable_name)
    except ExpressionParseError as exc:
        raise CompilationError(str(exc)) from exc


# --- Phase 8: Statement resolution + ordering ---


def _bind_statement_references(
    variable_name: str,
    statements: tuple[Statement, ...],
    unqualified_bindings: dict[str, str],
    qualified_bindings: dict[str, dict[str, str]],
) -> tuple[Statement, ...]:
    """Resolve module-qualified and renamed top-level references."""

    def bind_name(name: str) -> str:
        if "." not in name:
            return unqualified_bindings.get(name, name)

        alias, _, symbol_name = name.partition(".")
        if not symbol_name:
            raise CompilationError(
                f"Variable '{variable_name}' uses attribute access in '{name}'. "
                "Attribute access is not supported by generic compilation."
            )
        if alias not in qualified_bindings:
            raise CompilationError(
                f"Variable '{variable_name}' uses attribute access in '{name}'. "
                "Attribute access is not supported by generic compilation."
            )
        if "." in symbol_name:
            raise CompilationError(
                f"Variable '{variable_name}' uses nested attribute access in '{name}'. "
                "Only module-qualified references like alias.value are supported."
            )
        try:
            return qualified_bindings[alias][symbol_name]
        except KeyError as exc:
            raise CompilationError(
                f"Variable '{variable_name}' references unknown imported symbol "
                f"'{name}'."
            ) from exc

    return map_statement_names(statements, bind_name)


def _analyze_statement_dependencies(
    variable_name: str,
    statements: tuple[Statement, ...],
    local_names: set[str],
    record_reference,
) -> None:
    """Validate statement flow and collect external references."""

    def note_reference(name: str, assigned: set[str]) -> None:
        if name in assigned or name == variable_name:
            return
        if name in local_names:
            raise CompilationError(
                f"Variable '{variable_name}' references local '{name}' before it is "
                "assigned on all reachable paths."
            )
        record_reference(name)

    def walk_sequence(
        sequence: tuple[Statement, ...],
        assigned: set[str],
    ) -> set[str] | None:
        current = set(assigned)
        for statement in sequence:
            if current is None:
                raise CompilationError(
                    f"Variable '{variable_name}' has statements after a guaranteed "
                    "return. Remove unreachable code."
                )

            if isinstance(statement, AssignStmt):
                for reference in collect_references(statement.expression):
                    note_reference(reference, current)
                current.add(statement.name)
                continue

            if isinstance(statement, ReturnStmt):
                for reference in collect_references(statement.expression):
                    note_reference(reference, current)
                current = None
                continue

            if isinstance(statement, IfStmt):
                for reference in collect_references(statement.condition):
                    note_reference(reference, current)

                body_assigned = walk_sequence(statement.body, set(current))
                if statement.orelse:
                    else_assigned = walk_sequence(statement.orelse, set(current))
                else:
                    else_assigned = set(current)

                if body_assigned is None and else_assigned is None:
                    current = None
                elif body_assigned is None:
                    current = else_assigned
                elif else_assigned is None:
                    current = body_assigned
                else:
                    current = body_assigned & else_assigned
                continue

            raise AssertionError(
                f"Unhandled statement node: {type(statement).__name__}"
            )

        return current

    final_assigned = walk_sequence(statements, set())
    if final_assigned is not None:
        raise CompilationError(
            f"Variable '{variable_name}' does not return a value on all reachable "
            "paths."
        )


def _order_variables(variables: list[CompiledVariable]) -> list[CompiledVariable]:
    """Order variables so each dependency is available before use."""
    ordered: list[CompiledVariable] = []
    available: set[str] = set()
    remaining = list(variables)

    while remaining:
        next_round: list[CompiledVariable] = []
        progressed = False
        for variable in remaining:
            if set(variable.variable_dependencies) <= available:
                ordered.append(variable)
                available.add(variable.name)
                progressed = True
            else:
                next_round.append(variable)

        if not progressed:
            names = ", ".join(sorted(variable.name for variable in next_round))
            raise CompilationError(
                "Variable dependencies are cyclic or depend on unsupported ordering: "
                f"{names}."
            )

        remaining = next_round

    return ordered


def _infer_input(name: str) -> CompiledInput:
    """Infer a simple public input shape from a free formula reference."""
    if name.startswith(_BOOLEAN_PREFIXES):
        return CompiledInput(
            name=name,
            default=False,
            js_type="boolean",
            python_type="bool",
            public_name=name,
            symbol_name=name,
        )
    if name.startswith(_INTEGER_PREFIXES) or name.endswith(_INTEGER_SUFFIXES):
        return CompiledInput(
            name=name,
            default=0,
            js_type="number",
            python_type="int",
            public_name=name,
            symbol_name=name,
        )
    return CompiledInput(
        name=name,
        default=0,
        js_type="number",
        python_type="float",
        public_name=name,
        symbol_name=name,
    )


def _append_unique(names: list[str], value: str) -> None:
    """Append a value to a list only once while preserving order."""
    if value not in names:
        names.append(value)


def _ordered_unique(values: Any) -> list[str]:
    """Return unique values in encounter order."""
    ordered: list[str] = []
    for value in values:
        _append_unique(ordered, value)
    return ordered


# --- Phase 9: External rule binding ---


def _resolve_external_rule_binding(
    rule: "RuleDecl",
    compile_context: CompileContext,
) -> RuleBinding | None:
    """Resolve one external rule from the compile context resolver."""
    return compile_context.external_rule_resolver.resolve(
        module_identity=rule.module_identity,
        symbol=rule.symbol_name or rule.name,
        effective_date=compile_context.effective_date,
    )


def _bound_external_rule_source(source: str, binding: RuleBinding) -> str:
    """Annotate a source-backed external rule that was bound externally."""
    if binding.source and source:
        return f"{source} [bound from {binding.source}]"
    if binding.source:
        return binding.source
    if source:
        return f"{source} [bound externally]"
    return "bound externally"


def _external_rule_binding_target(name: str, rule: "RuleDecl") -> str:
    """Return the user-facing binding key for one parsed external rule."""
    symbol_name = rule.symbol_name or name
    if rule.module_identity:
        return f"{rule.module_identity}.{symbol_name}"
    return symbol_name
