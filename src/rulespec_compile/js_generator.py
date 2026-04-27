"""
JavaScript code generation from RuleSpec.

Generates standalone JS calculators that can run in browsers
without any backend - suitable for static inspection and demo sites.
"""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class Parameter:
    """A parameter value from statute or guidance."""

    name: str
    values: dict[int, float]  # key by number of children or bracket index
    source: str  # e.g., "26 USC 32(b)(1)" or "Rev. Proc. 2024-40"
    module_identity: str = ""


@dataclass
class Variable:
    """A compiled variable ready for JS generation."""

    name: str
    inputs: list[str]
    formula_js: str
    label: str = ""
    citation: str = ""
    module_identity: str = ""


@dataclass
class Output:
    """A public output exposed by the generated calculator."""

    name: str
    variable_name: str


class JSCodeGenerator:
    """
    Generate standalone JavaScript calculators from RuleSpec.

    Usage:
        gen = JSCodeGenerator()
        gen.add_parameter("credit_pct", {0: 7.65, 1: 34, ...}, "26 USC 32(b)(1)")
        gen.add_variable("eitc", ["earned_income", "agi", ...], formula_js)
        code = gen.generate()
    """

    def __init__(
        self,
        module_name: str = "calculator",
        include_provenance: bool = True,
        typescript: bool = False,
    ):
        self.module_name = module_name
        self.include_provenance = include_provenance
        self.typescript = typescript
        self.parameters: dict[str, Parameter] = {}
        self.variables: list[Variable] = []
        self.outputs: list[Output] | None = None
        self.inputs: dict[str, Any] = {}  # name -> default value

    def add_input(
        self,
        name: str,
        default: Any = 0,
        type_hint: str = "number",
        *,
        public_name: str | None = None,
    ) -> None:
        """Add an input variable."""
        # Convert Python booleans to JS
        if isinstance(default, bool):
            default = "true" if default else "false"
        self.inputs[name] = {
            "default": default,
            "type": type_hint,
            "public_name": public_name or name,
        }

    def add_parameter(
        self,
        name: str,
        values: dict[int, float],
        source: str = "",
        module_identity: str = "",
    ) -> None:
        """Add a parameter with values indexed by bracket."""
        self.parameters[name] = Parameter(
            name=name,
            values=values,
            source=source,
            module_identity=module_identity,
        )

    def add_variable(
        self,
        name: str,
        inputs: list[str],
        formula_js: str,
        label: str = "",
        citation: str = "",
        module_identity: str = "",
    ) -> None:
        """Add a calculated variable with its JS formula."""
        self.variables.append(
            Variable(
                name=name,
                inputs=inputs,
                formula_js=formula_js,
                label=label,
                citation=citation,
                module_identity=module_identity,
            )
        )

    def set_outputs(self, output_names: list[Any]) -> None:
        """Set the public outputs returned by calculate()."""
        self.outputs = self._normalize_outputs(output_names)

    def generate(self) -> str:
        """Generate the complete JavaScript module."""
        lines = []

        # Header with provenance
        lines.append("/**")
        lines.append(f" * {self.module_name} - Auto-generated from RuleSpec")
        lines.append(" * ")
        lines.append(" * This code runs entirely in the browser with full citation")
        lines.append(" * chain - every value traces back to authoritative law.")
        lines.append(" * ")
        if self.include_provenance:
            lines.append(" * Sources:")
            sources = set()
            for param in self.parameters.values():
                if param.source:
                    sources.add(param.source)
            for output, var in self._resolved_outputs():
                if var.citation:
                    sources.add(var.citation)
            for src in sorted(sources):
                lines.append(f" *   - {src}")
        lines.append(" */")
        lines.append("")

        # Parameters as const objects
        if self.parameters:
            lines.append("// Parameters from statute and guidance")
            lines.append("const PARAMS = {")
            for param in self.parameters.values():
                values_js = ", ".join(
                    f"{k}: {v}" for k, v in sorted(param.values.items())
                )
                comment = f"  // {param.source}" if param.source else ""
                lines.append(f"  {param.name}: {{ {values_js} }},{comment}")
            lines.append("};")
            lines.append("")
        else:
            lines.append("const PARAMS = {};")
            lines.append("")

        # Main calculate function
        if self.typescript:
            self._generate_typescript_function(lines)
        else:
            self._generate_js_function(lines)

        # ESM exports
        lines.append("")
        lines.append("export { calculate, PARAMS };")
        lines.append("export default calculate;")

        return "\n".join(lines)

    def _generate_js_function(self, lines: list[str]) -> None:
        """Generate JavaScript function."""
        # JSDoc
        lines.append("/**")
        lines.append(" * Calculate tax/benefit values with full citation chain.")
        lines.append(" *")
        for name, info in self.inputs.items():
            lines.append(f" * @param {{{info['type']}}} {info['public_name']}")
        lines.append(" * @returns {{result: number, citations: Array}}")
        lines.append(" */")

        # Function signature
        if self._can_use_destructured_js_inputs():
            params = ", ".join(
                self._render_js_input_binding(name, info)
                for name, info in self.inputs.items()
            )
            lines.append(f"function calculate({{ {params} }} = {{}}) {{")
        else:
            lines.append("function calculate(inputs = {}) {")
            for name, info in self.inputs.items():
                lines.append(
                    f"  const {name} = {self._render_js_input_lookup(name, info)};"
                )
            if self.inputs:
                lines.append("")

        # Add calculations
        for var in self.variables:
            lines.append(f"  // {var.citation}" if var.citation else "")
            lines.append(
                f"  const {var.name} = {self._render_formula_js(var.formula_js)};"
            )
            lines.append("")

        # Return with citations
        lines.append("  return {")
        for output, _ in self._resolved_outputs():
            lines.append(f"    {output.name}: {output.variable_name},")
        lines.append("    citations: [")
        for param in self.parameters.values():
            if param.source:
                lines.append(
                    "      { "
                    f'param: "{param.name}", '
                    f'module_identity: "{param.module_identity}", '
                    f'source: "{param.source}" '
                    "},"
                )
        for output, var in self._resolved_outputs():
            if var.citation:
                lines.append(
                    "      { "
                    f'variable: "{output.name}", '
                    f'module_identity: "{var.module_identity}", '
                    f'source: "{var.citation}" '
                    "},"
                )
        lines.append("    ],")
        lines.append("  };")
        lines.append("}")

    def _render_formula_js(self, formula_js: str) -> str:
        """Render an expression or statement block as valid JavaScript."""
        stripped = formula_js.strip()
        if (
            "\n" not in stripped
            and ";" not in stripped
            and not self._is_return_statement_js(stripped)
        ):
            return stripped

        lines = self._normalize_block_formula_js(stripped)
        indented = "\n".join(f"    {line}" for line in lines)
        return f"(() => {{\n{indented}\n  }})()"

    def _normalize_block_formula_js(self, formula_js: str) -> list[str]:
        """Ensure a JS statement block returns its final expression explicitly."""
        lines = self._split_inline_js_statements(formula_js)
        last_index = self._last_nonempty_line(lines)
        if last_index is None:
            raise ValueError("JavaScript block formulas cannot be empty.")

        stripped = lines[last_index].strip()
        if self._is_return_statement_js(stripped):
            return lines
        if self._is_expression_line_js(stripped):
            indent = lines[last_index][
                : len(lines[last_index]) - len(lines[last_index].lstrip())
            ]
            expression = stripped[:-1].rstrip() if stripped.endswith(";") else stripped
            lines[last_index] = f"{indent}return {expression};"
            return lines
        if stripped == "}" and self._final_statement_guarantees_return_js(lines):
            return lines
        raise ValueError(
            "JavaScript block formulas must end with an explicit return "
            "or a final expression."
        )

    def _is_expression_line_js(self, line: str) -> bool:
        """Return whether a JS line looks like an expression statement."""
        if not line or line in {"{", "}"} or line.endswith("{"):
            return False
        if re.match(
            r"(return|const|let|var|if|else|for|while|switch|case|default|break|"
            r"continue|throw|try|catch|finally|function|class|import|export)\b",
            line,
        ):
            return False
        if re.search(r"(?<![=!<>])=(?!=)|\+=|-=|\*=|/=|%=|&&=|\|\|=|\?\?=", line):
            return False
        return True

    def _final_statement_guarantees_return_js(self, lines: list[str]) -> bool:
        """Return whether the final top-level JS statement always returns."""
        ranges = self._top_level_statement_ranges_js(lines)
        if not ranges:
            return False
        start, end = ranges[-1]
        return self._statement_range_guarantees_return_js(lines, start, end)

    def _top_level_statement_ranges_js(self, lines: list[str]) -> list[tuple[int, int]]:
        """Split one JS block into top-level statement ranges."""
        ranges: list[tuple[int, int]] = []
        index = 0
        while index < len(lines):
            if not lines[index].strip():
                index += 1
                continue
            start = index
            depth = lines[index].count("{") - lines[index].count("}")
            if depth <= 0:
                ranges.append((start, index))
                index += 1
                continue
            index += 1
            while index < len(lines):
                depth += lines[index].count("{") - lines[index].count("}")
                if depth <= 0:
                    ranges.append((start, index))
                    index += 1
                    break
                index += 1
        return ranges

    def _statement_range_guarantees_return_js(
        self,
        lines: list[str],
        start: int,
        end: int,
    ) -> bool:
        """Return whether one top-level JS statement range always returns."""
        first = next(
            (
                lines[index].strip()
                for index in range(start, end + 1)
                if lines[index].strip()
            ),
            "",
        )
        if not first:
            return False
        if self._is_return_statement_js(first):
            return True
        return (
            first.startswith("if ") or first.startswith("if(")
        ) and self._if_chain_guarantees_return_js(lines, start, end)

    def _if_chain_guarantees_return_js(
        self,
        lines: list[str],
        start: int,
        end: int,
    ) -> bool:
        """Return whether one final JS if/else chain returns on every branch."""
        branch_ranges: list[tuple[int, int]] = []
        saw_final_else = False
        body_start = start + 1
        depth = 1
        index = start + 1

        while index <= end:
            stripped = lines[index].strip()
            if depth == 1 and stripped.startswith("} else if"):
                branch_ranges.append((body_start, index - 1))
                body_start = index + 1
                index += 1
                continue
            if depth == 1 and stripped.startswith("} else {"):
                branch_ranges.append((body_start, index - 1))
                body_start = index + 1
                saw_final_else = True
                index += 1
                continue

            depth += stripped.count("{") - stripped.count("}")
            if depth == 0:
                branch_ranges.append((body_start, index - 1))
                break
            index += 1

        if not saw_final_else or not branch_ranges:
            return False
        return all(
            self._sequence_guarantees_return_js(lines, branch_start, branch_end)
            for branch_start, branch_end in branch_ranges
        )

    def _sequence_guarantees_return_js(
        self,
        lines: list[str],
        start: int,
        end: int,
    ) -> bool:
        """Return whether one nested JS statement sequence always returns."""
        if end < start:
            return False
        ranges = self._top_level_statement_ranges_js(lines[start : end + 1])
        if not ranges:
            return False
        nested_start, nested_end = ranges[-1]
        return self._statement_range_guarantees_return_js(
            lines[start : end + 1],
            nested_start,
            nested_end,
        )

    def _last_nonempty_line(self, lines: list[str]) -> int | None:
        """Return the index of the last non-empty line in a block."""
        for index in range(len(lines) - 1, -1, -1):
            if lines[index].strip():
                return index
        return None

    def _is_return_statement_js(self, line: str) -> bool:
        """Return whether a JS line starts with a return statement."""
        return bool(re.match(r"return\b", line))

    def _split_inline_js_statements(self, formula_js: str) -> list[str]:
        """Split same-line JS statements on semicolons outside string literals."""
        lines: list[str] = []
        for raw_line in formula_js.split("\n"):
            indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
            content = raw_line[len(indent) :]
            current: list[str] = []
            quote: str | None = None
            escaped = False
            for char in content:
                if quote is not None:
                    current.append(char)
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == quote:
                        quote = None
                    continue
                if char in {"'", '"'}:
                    quote = char
                    current.append(char)
                    continue
                if char == ";":
                    statement = "".join(current).strip()
                    if statement:
                        lines.append(f"{indent}{statement};")
                    current = []
                    continue
                current.append(char)
            statement = "".join(current).strip()
            if statement:
                lines.append(f"{indent}{statement}")
            elif not lines or lines[-1].strip():
                lines.append("")
        return lines

    def _generate_typescript_function(self, lines: list[str]) -> None:
        """Generate TypeScript function with proper types."""
        # Interface for inputs
        lines.append("interface CalculatorInputs {")
        for name, info in self.inputs.items():
            public_name = info["public_name"]
            property_name = (
                public_name
                if _is_valid_js_identifier(public_name)
                else f'"{public_name}"'
            )
            lines.append(f"  {property_name}?: {info['type']};")
        lines.append("}")
        lines.append("")

        # Interface for result
        lines.append("interface CalculatorResult {")
        for output, _ in self._resolved_outputs():
            lines.append(f"  {output.name}: number;")
        citations_type = (
            "Array<{param?: string; variable?: string; module_identity?: string; "
            "source: string}>"
        )
        lines.append(f"  citations: {citations_type};")
        lines.append("}")
        lines.append("")

        # Function
        lines.append(
            "function calculate(inputs: CalculatorInputs = {}): CalculatorResult {"
        )
        if self._can_use_destructured_js_inputs():
            destructure = ", ".join(
                self._render_js_input_binding(name, info)
                for name, info in self.inputs.items()
            )
            lines.append(f"  const {{ {destructure} }} = inputs;")
        else:
            for name, info in self.inputs.items():
                lines.append(
                    f"  const {name} = {self._render_js_input_lookup(name, info)};"
                )
        lines.append("")

        # Calculations
        for var in self.variables:
            if var.citation:
                lines.append(f"  // {var.citation}")
            lines.append(f"  const {var.name}: number = {var.formula_js};")
            lines.append("")

        # Return
        lines.append("  return {")
        for output, _ in self._resolved_outputs():
            lines.append(f"    {output.name}: {output.variable_name},")
        lines.append("    citations: [")
        for param in self.parameters.values():
            if param.source:
                lines.append(
                    "      { "
                    f'param: "{param.name}", '
                    f'module_identity: "{param.module_identity}", '
                    f'source: "{param.source}" '
                    "},"
                )
        for output, var in self._resolved_outputs():
            if var.citation:
                lines.append(
                    "      { "
                    f'variable: "{output.name}", '
                    f'module_identity: "{var.module_identity}", '
                    f'source: "{var.citation}" '
                    "},"
                )
        lines.append("    ],")
        lines.append("  };")
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
                output = Output(name=public_name, variable_name=variable_name)
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
                    Output(name=variable.name, variable_name=variable.name),
                    variable,
                )
                for variable in self.variables
            ]
        variables_by_name = {variable.name: variable for variable in self.variables}
        return [
            (output, variables_by_name[output.variable_name]) for output in self.outputs
        ]

    def _can_use_destructured_js_inputs(self) -> bool:
        """Return whether public input names are safe for JS destructuring."""
        return all(
            _is_valid_js_identifier(info["public_name"])
            for info in self.inputs.values()
        )

    def _render_js_input_binding(self, name: str, info: dict[str, Any]) -> str:
        """Render one JS destructured input binding."""
        public_name = info["public_name"]
        if public_name == name:
            return f"{name} = {info['default']}"
        return f"{public_name}: {name} = {info['default']}"

    def _render_js_input_lookup(self, name: str, info: dict[str, Any]) -> str:
        """Render one JS lookup expression for a public input."""
        public_name = info["public_name"]
        default = info["default"]
        if public_name == name:
            return (
                "Object.prototype.hasOwnProperty.call(inputs, "
                f"{public_name!r}) ? inputs[{public_name!r}] : {default}"
            )
        return (
            "Object.prototype.hasOwnProperty.call(inputs, "
            f"{public_name!r}) ? inputs[{public_name!r}] : "
            "("
            "Object.prototype.hasOwnProperty.call(inputs, "
            f"{name!r}) ? inputs[{name!r}] : {default}"
            ")"
        )


def _is_valid_js_identifier(name: str) -> bool:
    """Return whether one name is safe to use as a JS identifier."""
    return bool(re.fullmatch(r"[A-Za-z_$][\w$]*", name))


def generate_eitc_calculator(tax_year: int = 2025) -> str:
    """
    Generate a standalone EITC calculator for the specified tax year.

    This is a pre-built calculator based on 26 USC 32.
    """
    gen = JSCodeGenerator(module_name=f"EITC Calculator (TY {tax_year})")

    # Inputs
    gen.add_input("earned_income", 0, "number")
    gen.add_input("agi", 0, "number")
    gen.add_input("n_children", 0, "number")
    gen.add_input("is_joint", False, "boolean")

    # Parameters from statute (fixed percentages)
    gen.add_parameter(
        "credit_pct",
        {0: 7.65, 1: 34, 2: 40, 3: 45},
        "26 USC 32(b)(1)",
    )
    gen.add_parameter(
        "phaseout_pct",
        {0: 7.65, 1: 15.98, 2: 21.06, 3: 21.06},
        "26 USC 32(b)(1)",
    )

    # Parameters from IRS guidance (inflation-adjusted for TY 2025)
    gen.add_parameter(
        "earned_income_amount",
        {0: 8260, 1: 12730, 2: 17880, 3: 17880},
        "Rev. Proc. 2024-40",
    )
    gen.add_parameter(
        "phaseout_single",
        {0: 10620, 1: 23350, 2: 23350, 3: 23350},
        "Rev. Proc. 2024-40",
    )
    gen.add_parameter(
        "phaseout_joint",
        {0: 17730, 1: 30470, 2: 30470, 3: 30470},
        "Rev. Proc. 2024-40",
    )

    # EITC formula - follows 26 USC 32(a)
    eitc_formula = """(() => {
    const n = Math.min(n_children, 3);
    const creditPct = PARAMS.credit_pct[n] / 100;
    const phaseoutPct = PARAMS.phaseout_pct[n] / 100;
    const earnedAmount = PARAMS.earned_income_amount[n];
    const phaseoutStart = is_joint
      ? PARAMS.phaseout_joint[n]
      : PARAMS.phaseout_single[n];

    // 32(a)(1): Credit base
    const creditBase = creditPct * Math.min(earned_income, earnedAmount);

    // 32(a)(2): Phaseout
    const incomeForPhaseout = Math.max(agi, earned_income);
    const excess = Math.max(0, incomeForPhaseout - phaseoutStart);
    const phaseout = phaseoutPct * excess;

    return Math.max(0, Math.round(creditBase - phaseout));
  })()"""

    gen.add_variable(
        "eitc",
        ["earned_income", "agi", "n_children", "is_joint"],
        eitc_formula,
        label="Earned Income Tax Credit",
        citation="26 USC 32",
    )

    return gen.generate()


if __name__ == "__main__":
    # Generate and print EITC calculator
    print(generate_eitc_calculator())
