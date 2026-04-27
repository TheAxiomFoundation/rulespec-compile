"""
Rust code generation from lowered RuleSpec IR.

Generates standalone Rust calculators for the current validated numeric/boolean
generic compile subset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .expression_ir import (
    AssignStmt,
    BinaryExpr,
    BoolExpr,
    CallExpr,
    CompareExpr,
    ConditionalExpr,
    Expression,
    IfStmt,
    LiteralExpr,
    NameExpr,
    ReturnStmt,
    Statement,
    SubscriptExpr,
    UnaryExpr,
)

_RUST_KEYWORDS = {
    "as",
    "break",
    "const",
    "continue",
    "crate",
    "else",
    "enum",
    "extern",
    "false",
    "fn",
    "for",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "match",
    "mod",
    "move",
    "mut",
    "pub",
    "ref",
    "return",
    "Self",
    "self",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "type",
    "unsafe",
    "use",
    "where",
    "while",
}


def _compilation_error(message: str):
    """Construct the shared compilation error lazily to avoid circular imports."""
    from .compile_model import CompilationError

    return CompilationError(message)


@dataclass
class Parameter:
    """A lowered parameter ready for Rust generation."""

    name: str
    values: dict[int, float]
    source: str
    module_identity: str = ""
    value_kind: str = "number"
    lookup_kind: str = "scalar"
    index_value_kind: str | None = None


@dataclass
class Variable:
    """A lowered computation ready for Rust generation."""

    name: str
    statements: tuple[Statement, ...]
    local_names: tuple[str, ...]
    local_value_kinds: dict[str, str]
    parameter_dependencies: tuple[str, ...]
    value_kind: str = "number"
    label: str = ""
    citation: str = ""
    module_identity: str = ""


@dataclass
class Output:
    """A public output exposed by the generated Rust calculator."""

    name: str
    variable_name: str
    value_kind: str = "number"


class RustCodeGenerator:
    """
    Generate standalone Rust calculators from lowered RuleSpec IR.

    The current Rust backend intentionally targets the validated numeric/boolean
    subset used by the shared generic compiler.
    """

    def __init__(
        self,
        module_name: str = "calculator",
        include_provenance: bool = True,
    ):
        self.module_name = module_name
        self.include_provenance = include_provenance
        self.parameters: dict[str, Parameter] = {}
        self.variables: list[Variable] = []
        self.outputs: list[Output] | None = None
        self.inputs: dict[str, Any] = {}

    def add_input(
        self,
        name: str,
        default: Any = 0,
        value_kind: str = "number",
        *,
        public_name: str | None = None,
    ) -> None:
        """Add an input variable."""
        if value_kind not in {"boolean", "integer", "number"}:
            raise _compilation_error(
                f"Rust backend does not support input kind '{value_kind}' for '{name}'."
            )
        self.inputs[name] = {
            "default": default,
            "value_kind": value_kind,
            "public_name": public_name or name,
        }

    def add_parameter(
        self,
        name: str,
        values: dict[int, float],
        source: str = "",
        module_identity: str = "",
        value_kind: str = "number",
        lookup_kind: str = "scalar",
        index_value_kind: str | None = None,
    ) -> None:
        """Add a parameter with values indexed by bracket."""
        if value_kind not in {"integer", "number"}:
            raise _compilation_error(
                f"Rust backend does not support parameter kind '{value_kind}' for "
                f"'{name}'."
            )
        if lookup_kind not in {"scalar", "indexed"}:
            raise _compilation_error(
                f"Rust backend does not support parameter lookup kind "
                f"'{lookup_kind}' for '{name}'."
            )
        if lookup_kind == "indexed":
            if index_value_kind not in {None, "integer"}:
                raise _compilation_error(
                    f"Rust backend does not support parameter index kind "
                    f"'{index_value_kind}' for '{name}'."
                )
            normalized_index_value_kind = "integer"
        else:
            if index_value_kind is not None:
                raise _compilation_error(
                    f"Rust backend parameter '{name}' cannot define index_value_kind "
                    "unless lookup_kind is 'indexed'."
                )
            normalized_index_value_kind = None
        self.parameters[name] = Parameter(
            name=name,
            values=values,
            source=source,
            module_identity=module_identity,
            value_kind=value_kind,
            lookup_kind=lookup_kind,
            index_value_kind=normalized_index_value_kind,
        )

    def add_variable(
        self,
        name: str,
        statements: tuple[Statement, ...],
        local_names: tuple[str, ...],
        local_value_kinds: dict[str, str],
        parameter_dependencies: tuple[str, ...],
        value_kind: str = "number",
        label: str = "",
        citation: str = "",
        module_identity: str = "",
    ) -> None:
        """Add a lowered computation for Rust generation."""
        self.variables.append(
            Variable(
                name=name,
                statements=statements,
                local_names=local_names,
                local_value_kinds=dict(local_value_kinds),
                parameter_dependencies=parameter_dependencies,
                value_kind=value_kind,
                label=label,
                citation=citation,
                module_identity=module_identity,
            )
        )

    def set_outputs(self, output_names: list[Any]) -> None:
        """Set the public outputs returned by calculate()."""
        self.outputs = self._normalize_outputs(output_names)

    def generate(self) -> str:
        """Generate the complete Rust module."""
        lines: list[str] = []
        self._emit_header(lines)
        self._emit_runtime_types(lines)
        self._emit_numeric_helpers(lines)
        self._emit_parameters(lines)
        self._emit_input_struct(lines)
        self._emit_calculate(lines)
        self._emit_calculate_public(lines)
        return "\n".join(lines)

    def _emit_header(self, lines: list[str]) -> None:
        """Emit the generated-module header."""
        lines.append("// Auto-generated from RuleSpec")
        lines.append(f"// Module: {self.module_name}")
        lines.append("//")
        lines.append(
            "// This Rust output targets the current validated numeric/boolean"
        )
        lines.append("// generic compile subset.")
        if self.include_provenance:
            sources = {
                param.source for param in self.parameters.values() if param.source
            }
            for output, variable in self._resolved_outputs():
                if variable.citation:
                    sources.add(variable.citation)
            if sources:
                lines.append("//")
                lines.append("// Sources:")
                for source in sorted(sources):
                    lines.append(f"//   - {source}")
        lines.append("")

    def _emit_runtime_types(self, lines: list[str]) -> None:
        """Emit shared runtime structs and enums."""
        lines.extend(
            [
                "use std::collections::BTreeMap;",
                "",
                "#[derive(Debug, Clone, PartialEq)]",
                "pub enum RuleSpecValue {",
                "    Bool(bool),",
                "    Integer(i64),",
                "    Number(f64),",
                "    String(String),",
                "}",
                "",
                "impl From<bool> for RuleSpecValue {",
                "    fn from(value: bool) -> Self {",
                "        Self::Bool(value)",
                "    }",
                "}",
                "",
                "impl From<i64> for RuleSpecValue {",
                "    fn from(value: i64) -> Self {",
                "        Self::Integer(value)",
                "    }",
                "}",
                "",
                "impl From<f64> for RuleSpecValue {",
                "    fn from(value: f64) -> Self {",
                "        Self::Number(value)",
                "    }",
                "}",
                "",
                "impl From<String> for RuleSpecValue {",
                "    fn from(value: String) -> Self {",
                "        Self::String(value)",
                "    }",
                "}",
                "",
                "impl From<&str> for RuleSpecValue {",
                "    fn from(value: &str) -> Self {",
                "        Self::String(value.to_string())",
                "    }",
                "}",
                "",
                "#[derive(Debug, Clone, PartialEq)]",
                "pub struct Citation {",
                "    pub kind: &'static str,",
                "    pub name: &'static str,",
                "    pub module_identity: &'static str,",
                "    pub source: &'static str,",
                "}",
                "",
                "#[derive(Debug, Clone, PartialEq)]",
                "pub struct CalculationResult {",
                "    pub outputs: BTreeMap<&'static str, RuleSpecValue>,",
                "    pub citations: Vec<Citation>,",
                "}",
                "",
            ]
        )

    def _emit_numeric_helpers(self, lines: list[str]) -> None:
        """Emit numeric helper shims for supported builtin functions."""
        lines.extend(
            [
                "fn rulespec_abs(value: f64) -> f64 {",
                "    value.abs()",
                "}",
                "",
                "fn rulespec_ceil(value: f64) -> f64 {",
                "    value.ceil()",
                "}",
                "",
                "fn rulespec_floor(value: f64) -> f64 {",
                "    value.floor()",
                "}",
                "",
                "fn rulespec_round(value: f64) -> f64 {",
                "    value.round()",
                "}",
                "",
                "fn rulespec_max(values: &[f64]) -> f64 {",
                "    values",
                "        .iter()",
                "        .copied()",
                "        .reduce(f64::max)",
                '        .expect("rulespec_max requires at least one argument")',
                "}",
                "",
                "fn rulespec_min(values: &[f64]) -> f64 {",
                "    values",
                "        .iter()",
                "        .copied()",
                "        .reduce(f64::min)",
                '        .expect("rulespec_min requires at least one argument")',
                "}",
                "",
            ]
        )

    def _emit_parameters(self, lines: list[str]) -> None:
        """Emit parameter lookup helpers."""
        for parameter in self.parameters.values():
            function_name = _parameter_function_name(parameter.name)
            lines.append(
                f"fn {function_name}(index: i64) -> "
                f"{_rust_parameter_type(parameter.value_kind)} {{"
            )
            lines.append("    match index {")
            for index, value in sorted(parameter.values.items()):
                lines.append(
                    "        "
                    f"{index} => "
                    f"{_render_parameter_literal(value, parameter.value_kind)},"
                )
            lines.append(
                "        _ => panic!("
                + _rust_string_literal(
                    f"Unknown parameter index for '{parameter.name}': " + "{}"
                )
                + ", index),"
            )
            lines.append("    }")
            lines.append("}")
            lines.append("")

    def _emit_input_struct(self, lines: list[str]) -> None:
        """Emit the typed input struct and its defaults."""
        lines.append("#[derive(Debug, Clone, PartialEq)]")
        lines.append("pub struct CalculateInputs {")
        for name, info in self.inputs.items():
            lines.append(
                f"    pub {_rust_identifier(name)}: "
                f"{_rust_input_type(info['value_kind'])},"
            )
        lines.append("}")
        lines.append("")
        lines.append("impl Default for CalculateInputs {")
        lines.append("    fn default() -> Self {")
        lines.append("        Self {")
        for name, info in self.inputs.items():
            lines.append(
                f"            {_rust_identifier(name)}: "
                f"{_render_input_default(info['default'], info['value_kind'])},"
            )
        lines.append("        }")
        lines.append("    }")
        lines.append("}")
        lines.append("")

    def _emit_calculate(self, lines: list[str]) -> None:
        """Emit the main calculate() function."""
        lines.append("pub fn calculate(inputs: CalculateInputs) -> CalculationResult {")
        if self.inputs:
            lines.append("    let CalculateInputs {")
            for name in self.inputs:
                identifier = _rust_identifier(name)
                lines.append(f"        {identifier},")
            lines.append("    } = inputs;")
            lines.append("")
        else:
            lines.append("    let _ = inputs;")
            lines.append("")

        input_value_kinds = {
            name: info["value_kind"] for name, info in self.inputs.items()
        }
        parameter_value_kinds = {
            name: parameter.value_kind for name, parameter in self.parameters.items()
        }
        variable_value_kinds = {
            variable.name: variable.value_kind for variable in self.variables
        }
        for variable in self.variables:
            if variable.citation:
                lines.append(f"    // {variable.citation}")
            expression = _render_variable_expression(
                variable,
                input_value_kinds=input_value_kinds,
                parameter_value_kinds=parameter_value_kinds,
                variable_value_kinds=variable_value_kinds,
            )
            lines.append(f"    let {_rust_identifier(variable.name)} = {expression};")
            lines.append("")

        lines.append("    let mut outputs = BTreeMap::new();")
        for output, _ in self._resolved_outputs():
            lines.append(
                f"    outputs.insert({_rust_string_literal(output.name)}, "
                f"{_render_output_value(output)});"
            )
        lines.append("")
        lines.append("    CalculationResult {")
        lines.append("        outputs,")
        lines.append("        citations: vec![")
        for parameter in self.parameters.values():
            if parameter.source:
                lines.append(
                    "            Citation { "
                    f'kind: "param", name: {_rust_string_literal(parameter.name)}, '
                    "module_identity: "
                    f"{_rust_string_literal(parameter.module_identity)}, "
                    f"source: {_rust_string_literal(parameter.source)} "
                    "},"
                )
        for output, variable in self._resolved_outputs():
            if variable.citation:
                lines.append(
                    "            Citation { "
                    f'kind: "variable", name: {_rust_string_literal(output.name)}, '
                    "module_identity: "
                    f"{_rust_string_literal(variable.module_identity)}, "
                    f"source: {_rust_string_literal(variable.citation)} "
                    "},"
                )
        lines.append("        ],")
        lines.append("    }")
        lines.append("}")
        lines.append("")

    def _emit_calculate_public(self, lines: list[str]) -> None:
        """Emit a map-based public input entrypoint keyed by rule identity."""
        lines.append(
            "pub fn calculate_public(inputs: &BTreeMap<String, RuleSpecValue>) "
            "-> CalculationResult {"
        )
        lines.append("    let mut typed_inputs = CalculateInputs::default();")
        for name, info in self.inputs.items():
            public_name = info["public_name"]
            lookup = (
                f"inputs.get({_rust_string_literal(public_name)})"
                if public_name == name
                else (
                    f"inputs.get({_rust_string_literal(public_name)})"
                    f".or_else(|| inputs.get({_rust_string_literal(name)}))"
                )
            )
            lines.append(f"    if let Some(value) = {lookup} {{")
            lines.extend(
                _render_public_input_assignment_rust(
                    name=name,
                    public_name=public_name,
                    value_kind=info["value_kind"],
                    indent="        ",
                )
            )
            lines.append("    }")
        lines.append("    calculate(typed_inputs)")
        lines.append("}")

    def _normalize_outputs(self, output_names: list[Any]) -> list[Output]:
        """Normalize output bindings from strings or output-like objects."""
        outputs: list[Output] = []
        seen_public_names: set[str] = set()

        for output_name in output_names:
            if isinstance(output_name, str):
                output = Output(name=output_name, variable_name=output_name)
            else:
                public_name = getattr(output_name, "name")
                variable_name = getattr(output_name, "variable_name", public_name)
                value_kind = getattr(output_name, "value_kind", "number")
                output = Output(
                    name=public_name,
                    variable_name=variable_name,
                    value_kind=value_kind,
                )
            if output.name in seen_public_names:
                continue
            seen_public_names.add(output.name)
            outputs.append(output)
        return outputs

    def _resolved_outputs(self) -> list[tuple[Output, Variable]]:
        """Return the public outputs paired with their backing variables."""
        if self.outputs is None:
            return [
                (
                    Output(
                        name=variable.name,
                        variable_name=variable.name,
                        value_kind=variable.value_kind,
                    ),
                    variable,
                )
                for variable in self.variables
            ]
        variables_by_name = {variable.name: variable for variable in self.variables}
        return [
            (output, variables_by_name[output.variable_name]) for output in self.outputs
        ]


def _render_variable_expression(
    variable: Variable,
    *,
    input_value_kinds: dict[str, str],
    parameter_value_kinds: dict[str, str],
    variable_value_kinds: dict[str, str],
) -> str:
    """Render one computation as a Rust block expression."""
    label = _calculation_label(variable.name)
    local_bindings = {name: _local_slot_name(name) for name in variable.local_names}
    parameter_functions = {
        name: _parameter_function_name(name) for name in variable.parameter_dependencies
    }
    lines = [f"'{label}: {{"]
    for local_name in variable.local_names:
        slot_type = _rust_local_slot_type(
            _require_local_value_kind(variable, local_name)
        )
        lines.append(
            f"    let mut {local_bindings[local_name]}: Option<{slot_type}> = None;"
        )
    if variable.local_names:
        lines.append("")
    lines.extend(
        _render_statement_block_rust(
            variable.statements,
            indent="    ",
            label=label,
            local_bindings=local_bindings,
            local_value_kinds=variable.local_value_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_value_kinds=variable_value_kinds,
            parameter_functions=parameter_functions,
            computation_value_kind=variable.value_kind,
        )
    )
    lines.append("}")
    return "\n".join(lines)


def _render_output_value(output: Output) -> str:
    """Render one public output conversion into RuleSpecValue."""
    variable_name = _rust_identifier(output.variable_name)
    if output.value_kind == "boolean":
        return f"RuleSpecValue::Bool({variable_name})"
    if output.value_kind == "integer":
        return f"RuleSpecValue::Integer(({variable_name}) as i64)"
    if output.value_kind == "number":
        return f"RuleSpecValue::Number({variable_name})"
    if output.value_kind == "string":
        return f"RuleSpecValue::String({variable_name}.to_string())"
    raise _compilation_error(
        f"Rust backend does not support output kind '{output.value_kind}'."
    )


def _render_public_input_assignment_rust(
    *,
    name: str,
    public_name: str,
    value_kind: str,
    indent: str,
) -> list[str]:
    """Render Rust lines that coerce one public RuleSpecValue into a typed input."""
    target = f"typed_inputs.{_rust_identifier(name)}"
    if value_kind == "boolean":
        message = _rust_string_literal(f"Input '{public_name}' must be boolean.")
        return [
            f"{indent}{target} = match value {{",
            f"{indent}    RuleSpecValue::Bool(value) => *value,",
            f"{indent}    _ => panic!({message}),",
            f"{indent}}};",
        ]
    if value_kind == "integer":
        message = _rust_string_literal(f"Input '{public_name}' must be integer.")
        return [
            f"{indent}{target} = match value {{",
            f"{indent}    RuleSpecValue::Integer(value) => *value,",
            (
                f"{indent}    RuleSpecValue::Number(value) "
                "if value.fract() == 0.0 => *value as i64,"
            ),
            f"{indent}    _ => panic!({message}),",
            f"{indent}}};",
        ]
    if value_kind == "number":
        message = _rust_string_literal(f"Input '{public_name}' must be numeric.")
        return [
            f"{indent}{target} = match value {{",
            f"{indent}    RuleSpecValue::Integer(value) => *value as f64,",
            f"{indent}    RuleSpecValue::Number(value) => *value,",
            f"{indent}    _ => panic!({message}),",
            f"{indent}}};",
        ]
    raise _compilation_error(
        f"Rust backend does not support public input kind '{value_kind}' for '{name}'."
    )


def _render_statement_block_rust(
    statements: tuple[Statement, ...],
    *,
    indent: str,
    label: str,
    local_bindings: dict[str, str],
    local_value_kinds: dict[str, str],
    input_value_kinds: dict[str, str],
    parameter_value_kinds: dict[str, str],
    variable_value_kinds: dict[str, str],
    parameter_functions: dict[str, str],
    computation_value_kind: str,
) -> list[str]:
    """Render statement IR to Rust source lines."""
    lines: list[str] = []
    for statement in statements:
        lines.extend(
            _render_statement_rust(
                statement,
                indent=indent,
                label=label,
                local_bindings=local_bindings,
                local_value_kinds=local_value_kinds,
                input_value_kinds=input_value_kinds,
                parameter_value_kinds=parameter_value_kinds,
                variable_value_kinds=variable_value_kinds,
                parameter_functions=parameter_functions,
                computation_value_kind=computation_value_kind,
            )
        )
    return lines


def _render_statement_rust(
    statement: Statement,
    *,
    indent: str,
    label: str,
    local_bindings: dict[str, str],
    local_value_kinds: dict[str, str],
    input_value_kinds: dict[str, str],
    parameter_value_kinds: dict[str, str],
    variable_value_kinds: dict[str, str],
    parameter_functions: dict[str, str],
    computation_value_kind: str,
) -> list[str]:
    """Render one statement node to Rust source lines."""
    if isinstance(statement, AssignStmt):
        target = local_bindings[statement.name]
        target_kind = local_value_kinds[statement.name]
        expression = _render_expression_rust(
            statement.expression,
            local_bindings=local_bindings,
            local_value_kinds=local_value_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_value_kinds=variable_value_kinds,
            parameter_functions=parameter_functions,
            expected_kind=target_kind,
        )
        return [f"{indent}{target} = Some({expression});"]

    if isinstance(statement, ReturnStmt):
        expression = _render_expression_rust(
            statement.expression,
            local_bindings=local_bindings,
            local_value_kinds=local_value_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_value_kinds=variable_value_kinds,
            parameter_functions=parameter_functions,
            expected_kind=computation_value_kind,
        )
        return [f"{indent}break '{label} {expression};"]

    if isinstance(statement, IfStmt):
        condition = _render_expression_rust(
            statement.condition,
            local_bindings=local_bindings,
            local_value_kinds=local_value_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_value_kinds=variable_value_kinds,
            parameter_functions=parameter_functions,
            expected_kind="boolean",
        )
        lines = [f"{indent}if {condition} {{"]
        lines.extend(
            _render_statement_block_rust(
                statement.body,
                indent=indent + "    ",
                label=label,
                local_bindings=local_bindings,
                local_value_kinds=local_value_kinds,
                input_value_kinds=input_value_kinds,
                parameter_value_kinds=parameter_value_kinds,
                variable_value_kinds=variable_value_kinds,
                parameter_functions=parameter_functions,
                computation_value_kind=computation_value_kind,
            )
        )
        if statement.orelse:
            if len(statement.orelse) == 1 and isinstance(statement.orelse[0], IfStmt):
                nested = _render_statement_rust(
                    statement.orelse[0],
                    indent=indent,
                    label=label,
                    local_bindings=local_bindings,
                    local_value_kinds=local_value_kinds,
                    input_value_kinds=input_value_kinds,
                    parameter_value_kinds=parameter_value_kinds,
                    variable_value_kinds=variable_value_kinds,
                    parameter_functions=parameter_functions,
                    computation_value_kind=computation_value_kind,
                )
                first_nested = nested[0][len(indent) :]
                lines.append(f"{indent}}} else {first_nested}")
                lines.extend(nested[1:])
            else:
                lines.append(f"{indent}}} else {{")
                lines.extend(
                    _render_statement_block_rust(
                        statement.orelse,
                        indent=indent + "    ",
                        label=label,
                        local_bindings=local_bindings,
                        local_value_kinds=local_value_kinds,
                        input_value_kinds=input_value_kinds,
                        parameter_value_kinds=parameter_value_kinds,
                        variable_value_kinds=variable_value_kinds,
                        parameter_functions=parameter_functions,
                        computation_value_kind=computation_value_kind,
                    )
                )
                lines.append(f"{indent}}}")
        else:
            lines.append(f"{indent}}}")
        return lines

    raise AssertionError(f"Unhandled statement node: {type(statement).__name__}")


def _render_expression_rust(
    expression: Expression,
    *,
    local_bindings: dict[str, str],
    local_value_kinds: dict[str, str],
    input_value_kinds: dict[str, str],
    parameter_value_kinds: dict[str, str],
    variable_value_kinds: dict[str, str],
    parameter_functions: dict[str, str],
    expected_kind: str | None = None,
) -> str:
    """Render expression IR to Rust."""
    natural_kind = _infer_expression_value_kind_rust(
        expression,
        local_value_kinds=local_value_kinds,
        input_value_kinds=input_value_kinds,
        parameter_value_kinds=parameter_value_kinds,
        variable_value_kinds=variable_value_kinds,
    )
    if isinstance(expression, LiteralExpr):
        if isinstance(expression.value, str):
            raise _compilation_error(
                "Rust backend does not support string formula literals; use "
                "the Python or JavaScript backend. Encountered literal: "
                f"{expression.value!r}."
            )
        if isinstance(expression.value, bool):
            rendered = "true" if expression.value else "false"
        elif natural_kind == "integer":
            rendered = _render_integer_literal(expression.value)
        else:
            rendered = _render_number_literal(expression.value)
        return _coerce_rust_value(
            rendered,
            from_kind=natural_kind,
            expected_kind=expected_kind,
            subject="literal expression",
        )

    if isinstance(expression, NameExpr):
        if expression.name in local_bindings:
            slot = local_bindings[expression.name]
            message = _rust_string_literal(
                f"Local {expression.name!r} was referenced before assignment."
            )
            rendered = f"{slot}.clone().expect({message})"
            source_kind = local_value_kinds[expression.name]
        elif expression.name in parameter_functions:
            rendered = f"{parameter_functions[expression.name]}(0)"
            source_kind = parameter_value_kinds.get(expression.name, "number")
        elif expression.name in input_value_kinds:
            rendered = _rust_identifier(expression.name)
            source_kind = input_value_kinds[expression.name]
        else:
            rendered = _rust_identifier(expression.name)
            source_kind = variable_value_kinds.get(expression.name, natural_kind)
        return _coerce_rust_value(
            rendered,
            from_kind=source_kind,
            expected_kind=expected_kind,
            subject=f"reference '{expression.name}'",
        )

    if isinstance(expression, SubscriptExpr):
        if not isinstance(expression.value, NameExpr):
            raise _compilation_error(
                "Rust backend currently supports indexed access only for "
                "compiled parameters."
            )
        if expression.value.name not in parameter_functions:
            raise _compilation_error(
                "Rust backend currently supports indexed access only for "
                "compiled parameters."
            )
        index = _render_expression_rust(
            expression.index,
            local_bindings=local_bindings,
            local_value_kinds=local_value_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_value_kinds=variable_value_kinds,
            parameter_functions=parameter_functions,
            expected_kind="integer",
        )
        rendered = f"{parameter_functions[expression.value.name]}(({index}) as i64)"
        return _coerce_rust_value(
            rendered,
            from_kind=parameter_value_kinds.get(expression.value.name, "number"),
            expected_kind=expected_kind,
            subject="parameter lookup",
        )

    if isinstance(expression, CallExpr):
        if expression.function == "abs":
            argument = _render_expression_rust(
                expression.arguments[0],
                local_bindings=local_bindings,
                local_value_kinds=local_value_kinds,
                input_value_kinds=input_value_kinds,
                parameter_value_kinds=parameter_value_kinds,
                variable_value_kinds=variable_value_kinds,
                parameter_functions=parameter_functions,
                expected_kind=natural_kind,
            )
            rendered = f"({argument}).abs()"
            return _coerce_rust_value(
                rendered,
                from_kind=natural_kind,
                expected_kind=expected_kind,
                subject="abs() result",
            )
        if expression.function in {"ceil", "floor", "round"}:
            argument = _render_expression_rust(
                expression.arguments[0],
                local_bindings=local_bindings,
                local_value_kinds=local_value_kinds,
                input_value_kinds=input_value_kinds,
                parameter_value_kinds=parameter_value_kinds,
                variable_value_kinds=variable_value_kinds,
                parameter_functions=parameter_functions,
                expected_kind="number",
            )
            rendered = f"rulespec_{expression.function}({argument})"
            if natural_kind == "integer":
                rendered = f"(({rendered}) as i64)"
            return _coerce_rust_value(
                rendered,
                from_kind=natural_kind,
                expected_kind=expected_kind,
                subject=f"{expression.function}() result",
            )
        if expression.function in {"max", "min"}:
            if natural_kind == "integer":
                rendered = _render_integer_extrema_call(
                    expression.function,
                    expression.arguments,
                    local_bindings=local_bindings,
                    local_value_kinds=local_value_kinds,
                    input_value_kinds=input_value_kinds,
                    parameter_value_kinds=parameter_value_kinds,
                    variable_value_kinds=variable_value_kinds,
                    parameter_functions=parameter_functions,
                )
            else:
                rendered_args = [
                    _render_expression_rust(
                        argument,
                        local_bindings=local_bindings,
                        local_value_kinds=local_value_kinds,
                        input_value_kinds=input_value_kinds,
                        parameter_value_kinds=parameter_value_kinds,
                        variable_value_kinds=variable_value_kinds,
                        parameter_functions=parameter_functions,
                        expected_kind="number",
                    )
                    for argument in expression.arguments
                ]
                rendered = (
                    f"rulespec_{expression.function}(&[{', '.join(rendered_args)}])"
                )
            return _coerce_rust_value(
                rendered,
                from_kind=natural_kind,
                expected_kind=expected_kind,
                subject=f"{expression.function}() result",
            )
        raise _compilation_error(
            f"Rust backend does not support function '{expression.function}'."
        )

    if isinstance(expression, UnaryExpr):
        operand = _render_expression_rust(
            expression.operand,
            local_bindings=local_bindings,
            local_value_kinds=local_value_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_value_kinds=variable_value_kinds,
            parameter_functions=parameter_functions,
            expected_kind=("boolean" if expression.operator == "not" else natural_kind),
        )
        if expression.operator == "not":
            rendered = f"(!({operand}))"
        else:
            rendered = f"({expression.operator}{operand})"
        return _coerce_rust_value(
            rendered,
            from_kind=natural_kind,
            expected_kind=expected_kind,
            subject="unary expression",
        )

    if isinstance(expression, BinaryExpr):
        if natural_kind == "string":
            raise _compilation_error(
                "Rust backend does not support string formula literals; use "
                "the Python or JavaScript backend. Encountered string-valued "
                f"binary expression with operator {expression.operator!r}."
            )
        operand_kind = "number" if expression.operator in {"/", "**"} else natural_kind
        left = _render_expression_rust(
            expression.left,
            local_bindings=local_bindings,
            local_value_kinds=local_value_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_value_kinds=variable_value_kinds,
            parameter_functions=parameter_functions,
            expected_kind=operand_kind,
        )
        right = _render_expression_rust(
            expression.right,
            local_bindings=local_bindings,
            local_value_kinds=local_value_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_value_kinds=variable_value_kinds,
            parameter_functions=parameter_functions,
            expected_kind=operand_kind,
        )
        if expression.operator == "**":
            rendered = f"({left}).powf({right})"
        else:
            rendered = f"({left} {expression.operator} {right})"
        return _coerce_rust_value(
            rendered,
            from_kind=natural_kind,
            expected_kind=expected_kind,
            subject="binary expression",
        )

    if isinstance(expression, BoolExpr):
        operator = " || " if expression.operator == "or" else " && "
        rendered_values = [
            _render_expression_rust(
                value,
                local_bindings=local_bindings,
                local_value_kinds=local_value_kinds,
                input_value_kinds=input_value_kinds,
                parameter_value_kinds=parameter_value_kinds,
                variable_value_kinds=variable_value_kinds,
                parameter_functions=parameter_functions,
                expected_kind="boolean",
            )
            for value in expression.values
        ]
        rendered = "(" + operator.join(rendered_values) + ")"
        return _coerce_rust_value(
            rendered,
            from_kind="boolean",
            expected_kind=expected_kind,
            subject="boolean expression",
        )

    if isinstance(expression, CompareExpr):
        operand_kind = _comparison_operand_kind(
            expression,
            local_value_kinds=local_value_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_value_kinds=variable_value_kinds,
        )
        left = _render_expression_rust(
            expression.left,
            local_bindings=local_bindings,
            local_value_kinds=local_value_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_value_kinds=variable_value_kinds,
            parameter_functions=parameter_functions,
            expected_kind=operand_kind,
        )
        comparators = [
            _render_expression_rust(
                comparator,
                local_bindings=local_bindings,
                local_value_kinds=local_value_kinds,
                input_value_kinds=input_value_kinds,
                parameter_value_kinds=parameter_value_kinds,
                variable_value_kinds=variable_value_kinds,
                parameter_functions=parameter_functions,
                expected_kind=operand_kind,
            )
            for comparator in expression.comparators
        ]
        rendered_parts: list[str] = []
        previous = left
        for operator, comparator in zip(
            expression.operators,
            comparators,
            strict=True,
        ):
            rendered_parts.append(f"({previous} {operator} {comparator})")
            previous = comparator
        rendered = "(" + " && ".join(rendered_parts) + ")"
        return _coerce_rust_value(
            rendered,
            from_kind="boolean",
            expected_kind=expected_kind,
            subject="comparison expression",
        )

    if isinstance(expression, ConditionalExpr):
        condition = _render_expression_rust(
            expression.condition,
            local_bindings=local_bindings,
            local_value_kinds=local_value_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_value_kinds=variable_value_kinds,
            parameter_functions=parameter_functions,
            expected_kind="boolean",
        )
        if_true = _render_expression_rust(
            expression.if_true,
            local_bindings=local_bindings,
            local_value_kinds=local_value_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_value_kinds=variable_value_kinds,
            parameter_functions=parameter_functions,
            expected_kind=natural_kind,
        )
        if_false = _render_expression_rust(
            expression.if_false,
            local_bindings=local_bindings,
            local_value_kinds=local_value_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_value_kinds=variable_value_kinds,
            parameter_functions=parameter_functions,
            expected_kind=natural_kind,
        )
        rendered = f"(if {condition} {{ {if_true} }} else {{ {if_false} }})"
        return _coerce_rust_value(
            rendered,
            from_kind=natural_kind,
            expected_kind=expected_kind,
            subject="conditional expression",
        )

    raise AssertionError(f"Unhandled expression node: {type(expression).__name__}")


def _infer_expression_value_kind_rust(
    expression: Expression,
    *,
    local_value_kinds: dict[str, str],
    input_value_kinds: dict[str, str],
    parameter_value_kinds: dict[str, str],
    variable_value_kinds: dict[str, str],
) -> str:
    """Reuse shared compile-model kind inference for Rust lowering."""
    from .compile_model import _infer_expression_value_kind

    return _infer_expression_value_kind(
        expression,
        local_kinds=local_value_kinds,
        input_value_kinds=input_value_kinds,
        parameter_value_kinds=parameter_value_kinds,
        variable_kind_hints=variable_value_kinds,
    )


def _coerce_rust_value(
    rendered: str,
    *,
    from_kind: str,
    expected_kind: str | None,
    subject: str,
) -> str:
    """Coerce one rendered Rust expression into the expected lowered kind."""
    if expected_kind is None or expected_kind == from_kind:
        return rendered
    if from_kind == "integer" and expected_kind == "number":
        return f"(({rendered}) as f64)"
    raise _compilation_error(
        f"Rust backend cannot safely use {subject} of kind '{from_kind}' as "
        f"'{expected_kind}'."
    )


def _comparison_operand_kind(
    expression: CompareExpr,
    *,
    local_value_kinds: dict[str, str],
    input_value_kinds: dict[str, str],
    parameter_value_kinds: dict[str, str],
    variable_value_kinds: dict[str, str],
) -> str | None:
    """Choose one shared operand kind for a comparison chain."""
    operand_kinds = [
        _infer_expression_value_kind_rust(
            expression.left,
            local_value_kinds=local_value_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_value_kinds=variable_value_kinds,
        ),
        *[
            _infer_expression_value_kind_rust(
                comparator,
                local_value_kinds=local_value_kinds,
                input_value_kinds=input_value_kinds,
                parameter_value_kinds=parameter_value_kinds,
                variable_value_kinds=variable_value_kinds,
            )
            for comparator in expression.comparators
        ],
    ]
    unique = set(operand_kinds)
    if unique <= {"integer"}:
        return "integer"
    if unique <= {"integer", "number"}:
        return "number"
    if len(unique) == 1:
        return next(iter(unique))
    return None


def _render_integer_extrema_call(
    function_name: str,
    arguments: tuple[Expression, ...],
    *,
    local_bindings: dict[str, str],
    local_value_kinds: dict[str, str],
    input_value_kinds: dict[str, str],
    parameter_value_kinds: dict[str, str],
    variable_value_kinds: dict[str, str],
    parameter_functions: dict[str, str],
) -> str:
    """Render an exact integer max/min call without widening to f64."""
    rendered_args = [
        _render_expression_rust(
            argument,
            local_bindings=local_bindings,
            local_value_kinds=local_value_kinds,
            input_value_kinds=input_value_kinds,
            parameter_value_kinds=parameter_value_kinds,
            variable_value_kinds=variable_value_kinds,
            parameter_functions=parameter_functions,
            expected_kind="integer",
        )
        for argument in arguments
    ]
    if not rendered_args:
        raise _compilation_error(
            f"Rust backend requires at least one argument for {function_name}()."
        )
    current = rendered_args[0]
    reducer = "std::cmp::max" if function_name == "max" else "std::cmp::min"
    for argument in rendered_args[1:]:
        current = f"{reducer}({current}, {argument})"
    return current


def _require_local_value_kind(variable: Variable, local_name: str) -> str:
    """Require one lowered local slot kind before Rust generation."""
    try:
        return variable.local_value_kinds[local_name]
    except KeyError as exc:
        raise _compilation_error(
            f"Rust backend requires a value kind for local '{local_name}' in "
            f"computation '{variable.name}'."
        ) from exc


def _parameter_function_name(name: str) -> str:
    """Return the internal helper name for one parameter."""
    return _internal_identifier("param", name)


def _local_slot_name(name: str) -> str:
    """Return the internal Option slot name for one local."""
    return _internal_identifier("local", name)


def _calculation_label(name: str) -> str:
    """Return the labeled-block name for one computation."""
    return _internal_identifier("calc", name).removeprefix("r#")


def _internal_identifier(prefix: str, name: str) -> str:
    """Build one internal Rust identifier with a collision-resistant prefix."""
    return _rust_identifier(f"__{prefix}_{name}")


def _rust_identifier(name: str) -> str:
    """Render a RuleSpec identifier as a Rust identifier."""
    if name in _RUST_KEYWORDS:
        return f"r#{name}"
    return name


def _rust_string_literal(value: str) -> str:
    """Render one UTF-8 Rust string literal."""
    return json.dumps(value, ensure_ascii=False)


def _rust_input_type(value_kind: str) -> str:
    """Map one lowered input kind to a Rust input type."""
    if value_kind == "boolean":
        return "bool"
    if value_kind == "integer":
        return "i64"
    return "f64"


def _rust_parameter_type(value_kind: str) -> str:
    """Map one lowered parameter kind to a Rust lookup return type."""
    if value_kind == "integer":
        return "i64"
    if value_kind == "number":
        return "f64"
    raise _compilation_error(
        f"Rust backend does not support parameter kind '{value_kind}'."
    )


def _rust_local_slot_type(value_kind: str) -> str:
    """Map one lowered local kind to a conservative Rust Option slot type."""
    if value_kind == "boolean":
        return "bool"
    if value_kind == "integer":
        return "i64"
    if value_kind == "number":
        return "f64"
    if value_kind == "string":
        return "String"
    raise _compilation_error(
        f"Rust backend does not support local kind '{value_kind}'."
    )


def _render_input_default(default: Any, value_kind: str) -> str:
    """Render one input default value to Rust syntax."""
    if value_kind == "boolean":
        return "true" if bool(default) else "false"
    if value_kind == "integer":
        return _render_integer_literal(default)
    return _render_number_literal(default)


def _render_parameter_literal(value: Any, value_kind: str) -> str:
    """Render one lowered parameter value to Rust syntax."""
    if value_kind == "integer":
        return _render_integer_literal(value)
    if value_kind == "number":
        return _render_number_literal(value)
    raise _compilation_error(
        f"Rust backend does not support parameter kind '{value_kind}'."
    )


def _render_integer_literal(value: Any) -> str:
    """Render one integer literal to Rust syntax."""
    if isinstance(value, bool):
        raise _compilation_error(
            f"Rust backend expected an integer literal, got {value!r}."
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise _compilation_error(
            f"Rust backend expected an integer literal, got {value!r}."
        ) from exc
    if not numeric.is_integer():
        raise _compilation_error(
            f"Rust backend expected an exact integer literal, got {value!r}."
        )
    return str(int(numeric))


def _render_number_literal(value: Any) -> str:
    """Render one numeric literal to Rust syntax."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise _compilation_error(
            f"Rust backend expected a numeric literal, got {value!r}."
        ) from exc
    rendered = repr(numeric)
    if "e" not in rendered and "." not in rendered:
        rendered += ".0"
    return rendered
