"""Tests for Rust code generation from lowered RuleSpec IR."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from src.rulespec_compile.compile_model import LoweredInput, LoweredProgram
from src.rulespec_compile.expression_ir import (
    BinaryExpr,
    LiteralExpr,
    NameExpr,
    ReturnStmt,
)
from src.rulespec_compile.parser import parse_rulespec
from src.rulespec_compile.rust_generator import RustCodeGenerator


def _run_rust(
    code: str,
    inputs: dict[str, object],
    lowered_inputs: tuple[LoweredInput, ...],
) -> str:
    """Compile and run generated Rust, returning stdout."""
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc is required for Rust generator tests.")

    input_value_kinds = {
        name: compiled_input.value_kind
        for compiled_input in lowered_inputs
        for name in {
            compiled_input.public_name or compiled_input.name,
            compiled_input.name,
        }
    }
    uses_public_map = any(
        (compiled_input.public_name or compiled_input.name) != compiled_input.name
        for compiled_input in lowered_inputs
    )
    with tempfile.TemporaryDirectory(prefix="rulespec_compile_rust_test_") as tmp_dir:
        root = Path(tmp_dir)
        source = root / "main.rs"
        binary = root / "calculator"
        if uses_public_map:
            input_lines = [
                "    let mut public_inputs = BTreeMap::new();",
                *[
                    _format_rust_public_input_binding(name, value, input_value_kinds)
                    for name, value in inputs.items()
                ],
                "    let result = calculate_public(&public_inputs);",
            ]
        else:
            input_lines = [
                "    let result = calculate(CalculateInputs {",
                *[
                    _format_rust_input_binding(name, value, input_value_kinds)
                    for name, value in inputs.items()
                ],
                "        ..Default::default()",
                "    });",
            ]
        source.write_text(
            "\n".join(
                [
                    code,
                    "",
                    "fn main() {",
                    *input_lines,
                    "    for (name, value) in result.outputs.iter() {",
                    '        println!("{}={:?}", name, value);',
                    "    }",
                    "}",
                ]
            )
        )
        subprocess.run(
            _rustc_compile_command(rustc, source, binary),
            capture_output=True,
            text=True,
            check=True,
        )
        proc = subprocess.run(
            [str(binary)],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout


def _render_rust_literal(value: object, value_kind: str) -> str:
    """Render one Python test input as a Rust literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value_kind == "integer":
        if isinstance(value, (int, float)) and float(value).is_integer():
            return str(int(value))
        raise AssertionError(f"Expected integer Rust test input, got {value!r}.")
    if isinstance(value, (int, float)):
        rendered = repr(float(value))
        if "e" not in rendered and "." not in rendered:
            rendered += ".0"
        return rendered
    raise AssertionError(f"Unsupported Rust test input {value!r}.")


def _format_rust_input_binding(
    name: str,
    value: object,
    input_value_kinds: dict[str, str],
) -> str:
    """Render one Rust input struct field assignment for test execution."""
    return (
        "        "
        f"{name}: "
        f"{_render_rust_literal(value, input_value_kinds.get(name, 'number'))},"
    )


def _format_rust_public_input_binding(
    name: str,
    value: object,
    input_value_kinds: dict[str, str],
) -> str:
    """Render one Rust public-input map insertion for test execution."""
    kind = input_value_kinds.get(name, "number")
    literal = _render_rust_literal(value, kind)
    if kind == "boolean":
        rendered = f"RuleSpecValue::Bool({literal})"
    elif kind == "integer":
        rendered = f"RuleSpecValue::Integer({literal})"
    else:
        rendered = f"RuleSpecValue::Number({literal})"
    return f'    public_inputs.insert("{name}".to_string(), {rendered});'


def _rustc_compile_command(rustc: str, source: Path, binary: Path) -> list[str]:
    """Build a rustc compile command with a stable system linker when available."""
    command = [rustc, "--edition=2021", str(source), "-o", str(binary)]
    system_linker = Path("/usr/bin/cc")
    if system_linker.exists():
        command[1:1] = ["-C", f"linker={system_linker}"]
    return command


class TestRustCodeGenerator:
    """Tests for RustCodeGenerator and its lowered-program integration."""

    def test_init_defaults(self):
        """Generator initializes with sensible defaults."""
        gen = RustCodeGenerator()
        assert gen.module_name == "calculator"
        assert gen.include_provenance is True
        assert gen.parameters == {}
        assert gen.variables == []
        assert gen.inputs == {}

    def test_lowered_program_round_trips_and_generates_rust(self):
        """Lowered bundles can round-trip into executable Rust output."""
        rulespec = """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: Test
  versions:
  - effective_from: '2024-01-01'
    formula: '0.2'
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: |-
      taxable_income = wages - deduction
      return taxable_income * rate
"""
        lowered = parse_rulespec(rulespec).to_lowered_program(outputs=["tax"])
        round_trip = LoweredProgram.from_json(lowered.to_json())
        code = round_trip.to_rust_generator().generate()

        stdout = _run_rust(
            code,
            {"wages": 1000, "deduction": 100},
            round_trip.inputs,
        )

        assert "pub fn calculate" in code
        assert "tax=Number(180" in stdout

    def test_branching_formula_executes_in_rust(self):
        """Rust generation handles branch-assigned locals used later."""
        rulespec = """
format: rulespec/v1
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: |-
      if is_joint:
        rate = 0.1
      else:
        rate = 0.2
      return wages * rate
"""
        lowered = parse_rulespec(rulespec).to_lowered_program()
        code = lowered.to_rust_generator().generate()

        stdout = _run_rust(code, {"wages": 100, "is_joint": True}, lowered.inputs)

        assert "__local_rate" in code
        assert "tax=Number(10" in stdout

    def test_boolean_local_slots_are_typed_in_rust(self):
        """Boolean locals emit typed Option<bool> slots in Rust."""
        rulespec = """
format: rulespec/v1
rules:
- name: flag
  kind: derived
  entity: Person
  period: Year
  dtype: Bool
  versions:
  - effective_from: '2024-01-01'
    formula: |-
      eligible = wages <= 1000
      return eligible
"""
        lowered = parse_rulespec(rulespec).to_lowered_program()
        code = lowered.to_rust_generator().generate()

        stdout = _run_rust(code, {"wages": 500}, lowered.inputs)

        assert "Option<bool>" in code
        assert "flag=Bool(true)" in stdout

    def test_integer_outputs_remain_integer_at_rust_boundary(self):
        """Typed lowered outputs emit Integer values instead of Number."""
        rulespec = """
format: rulespec/v1
rules:
- name: count
  kind: derived
  entity: Person
  period: Year
  dtype: Integer
  versions:
  - effective_from: '2024-01-01'
    formula: return n_children + 1
"""
        lowered = parse_rulespec(rulespec).to_lowered_program()
        code = lowered.to_rust_generator().generate()

        stdout = _run_rust(code, {"n_children": 2}, lowered.inputs)

        assert "RuleSpecValue::Integer" in code
        assert "pub n_children: i64" in code
        assert "count=Integer(3)" in stdout

    def test_generated_citations_include_module_identity(self):
        """Rust citations keep the source module identity in generated code."""
        gen = RustCodeGenerator()
        gen.add_parameter("rate", {0: 1}, "26 USC 1", module_identity="shared")
        gen.add_input("wages", 0, "number")
        gen.add_variable(
            "tax",
            statements=(
                ReturnStmt(BinaryExpr(NameExpr("wages"), "*", LiteralExpr(1.0))),
            ),
            local_names=(),
            local_value_kinds={},
            parameter_dependencies=(),
            citation="26 USC 1(a)",
            module_identity="benefit_amount",
        )

        code = gen.generate()

        assert "pub module_identity: &'static str" in code
        assert 'module_identity: "shared"' in code
        assert 'module_identity: "benefit_amount"' in code

    def test_rust_generation_supports_public_input_map_names(self):
        """Rust emits a public map-based entrypoint for qualified input names."""
        gen = RustCodeGenerator()
        gen.add_input(
            "shared_income",
            0,
            "number",
            public_name="shared.rate.income",
        )
        gen.add_variable(
            "tax",
            statements=(
                ReturnStmt(
                    BinaryExpr(NameExpr("shared_income"), "*", LiteralExpr(2.0))
                ),
            ),
            local_names=(),
            local_value_kinds={},
            parameter_dependencies=(),
        )
        code = gen.generate()
        lowered_inputs = (
            LoweredInput(
                name="shared_income",
                default=0,
                value_kind="number",
                public_name="shared.rate.income",
            ),
        )

        stdout = _run_rust(
            code,
            {"shared.rate.income": 5},
            lowered_inputs,
        )

        assert "pub fn calculate_public" in code
        assert "tax=Number(10" in stdout

    def test_integer_scalar_parameter_reference_stays_exact_in_rust(self):
        """Direct integer parameter references emit exact i64 helpers and outputs."""
        rulespec = """
format: rulespec/v1
rules:
- name: bonus
  kind: parameter
  source: Test
  versions:
  - effective_from: '2024-01-01'
    formula: '2'
- name: count
  kind: derived
  entity: Person
  period: Year
  dtype: Integer
  versions:
  - effective_from: '2024-01-01'
    formula: return n_children + bonus
"""
        lowered = parse_rulespec(rulespec).to_lowered_program()
        code = lowered.to_rust_generator().generate()

        stdout = _run_rust(code, {"n_children": 2}, lowered.inputs)

        assert "fn __param_bonus(index: i64) -> i64" in code
        assert "count=Integer(4)" in stdout

    def test_integer_indexed_parameter_lookup_stays_exact_in_rust(self):
        """Indexed integer parameter lookups preserve exact integer kinds in Rust."""
        rulespec = """
format: rulespec/v1
rules:
- name: allowances
  kind: parameter
  source: external/allowances
- name: count
  kind: derived
  entity: Person
  period: Year
  dtype: Integer
  versions:
  - effective_from: '2024-01-01'
    formula: return allowances[n_children]
"""
        lowered = parse_rulespec(rulespec).to_lowered_program(
            rule_bindings={"allowances": [1, 2]}
        )
        code = lowered.to_rust_generator().generate()

        stdout = _run_rust(code, {"n_children": 1}, lowered.inputs)

        assert "fn __param_allowances(index: i64) -> i64" in code
        assert "count=Integer(2)" in stdout

    def test_rust_generation_rejects_string_formula_literals(self):
        """Rust generation fails loudly on unsupported string formulas."""
        rulespec = """
format: rulespec/v1
rules:
- name: name
  kind: derived
  entity: Person
  period: Year
  dtype: Text
  versions:
  - effective_from: '2024-01-01'
    formula: return "hello"
"""
        with pytest.raises(ValueError) as exc_info:
            parse_rulespec(rulespec).to_rust_generator().generate()
        message = str(exc_info.value)
        assert "Rust backend does not support string formula literals" in message
        assert "Python or JavaScript backend" in message
        assert "'hello'" in message
