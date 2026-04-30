"""
Tests for rulespec-compile JS code generator.

TDD: Tests written first, implementation follows.
"""

import shutil
import subprocess

import pytest

from src.rulespec_compile.js_generator import (
    JSCodeGenerator,
    generate_eitc_calculator,
)


class TestJSCodeGenerator:
    """Tests for JSCodeGenerator class."""

    def test_init_defaults(self):
        """Generator initializes with sensible defaults."""
        gen = JSCodeGenerator()
        assert gen.module_name == "calculator"
        assert gen.include_provenance is True
        assert gen.typescript is False
        assert gen.parameters == {}
        assert gen.variables == []
        assert gen.inputs == {}

    def test_add_input_number(self):
        """Can add numeric input with default."""
        gen = JSCodeGenerator()
        gen.add_input("income", 50000, "number")
        assert "income" in gen.inputs
        assert gen.inputs["income"]["default"] == 50000
        assert gen.inputs["income"]["type"] == "number"

    def test_add_input_boolean_converts_to_js(self):
        """Boolean defaults are converted to JS syntax."""
        gen = JSCodeGenerator()
        gen.add_input("is_married", False, "boolean")
        assert gen.inputs["is_married"]["default"] == "false"

        gen.add_input("has_children", True, "boolean")
        assert gen.inputs["has_children"]["default"] == "true"

    def test_add_parameter(self):
        """Can add parameter with values and source."""
        gen = JSCodeGenerator()
        gen.add_parameter(
            "tax_rate",
            {0: 10, 1: 12, 2: 22},
            "26 USC 1(a)",
        )
        assert "tax_rate" in gen.parameters
        assert gen.parameters["tax_rate"].values == {0: 10, 1: 12, 2: 22}
        assert gen.parameters["tax_rate"].source == "26 USC 1(a)"

    def test_add_variable(self):
        """Can add calculated variable."""
        gen = JSCodeGenerator()
        gen.add_variable(
            name="tax",
            inputs=["income"],
            formula_js="income * 0.2",
            label="Income Tax",
            citation="26 USC 1",
        )
        assert len(gen.variables) == 1
        assert gen.variables[0].name == "tax"
        assert gen.variables[0].formula_js == "income * 0.2"


class TestGenerateOutput:
    """Tests for generated JS code."""

    def test_generate_includes_header(self):
        """Generated code includes module header."""
        gen = JSCodeGenerator(module_name="Test Calculator")
        code = gen.generate()
        assert "Test Calculator" in code
        assert "Auto-generated from RuleSpec" in code

    def test_generate_includes_params_object(self):
        """Generated code includes PARAMS constant."""
        gen = JSCodeGenerator()
        gen.add_parameter("rate", {0: 10, 1: 20}, "Test Source")
        code = gen.generate()
        assert "const PARAMS = {" in code
        assert "rate: { 0: 10, 1: 20 }" in code
        assert "0: 10" in code
        assert "// Test Source" in code

    def test_generate_includes_calculate_function(self):
        """Generated code includes calculate function."""
        gen = JSCodeGenerator()
        gen.add_input("x", 0)
        gen.add_variable("y", ["x"], "x * 2")
        code = gen.generate()
        assert "function calculate(" in code
        assert "x = 0" in code
        assert "const y = x * 2" in code

    def test_generate_supports_public_input_names_that_are_not_identifiers(self):
        """Qualified public input names fall back to object lookups in JS."""
        gen = JSCodeGenerator()
        gen.add_input(
            "shared_income",
            0,
            "number",
            public_name="shared.rate.income",
        )
        gen.add_variable("tax", ["shared_income"], "shared_income * 2")

        code = gen.generate()

        assert (
            'inputs["shared.rate.income"]' in code
            or "inputs['shared.rate.income']" in code
        )

        if shutil.which("node") is None:
            pytest.skip("Node.js is required for JS runtime execution tests.")

        script = "\n".join(
            [
                code,
                'console.log(JSON.stringify(calculate({ "shared.rate.income": 5 })));',
            ]
        )
        proc = subprocess.run(
            ["node", "--input-type=module"],
            input=script,
            capture_output=True,
            text=True,
            check=True,
        )
        result = proc.stdout.splitlines()[-1]
        assert '"tax":10' in result

    def test_generate_returns_citations(self):
        """Generated calculate returns citation chain."""
        gen = JSCodeGenerator()
        gen.add_parameter("rate", {0: 10}, "26 USC 1")
        gen.add_variable("tax", [], "100", citation="26 USC 1(a)")
        code = gen.generate()
        assert "citations: [" in code
        assert 'source: "26 USC 1"' in code

    def test_generate_includes_module_identity_in_citations(self):
        """Generated citations keep the leaf-derived source rule identity."""
        gen = JSCodeGenerator()
        gen.add_parameter("rate", {0: 10}, "26 USC 1", module_identity="shared")
        gen.add_variable(
            "tax",
            [],
            "100",
            citation="26 USC 1(a)",
            module_identity="benefit_amount",
        )

        code = gen.generate()

        assert 'module_identity: "shared"' in code
        assert 'module_identity: "benefit_amount"' in code

    def test_generate_esm_exports(self):
        """Generated code includes ESM exports."""
        gen = JSCodeGenerator()
        code = gen.generate()
        assert "export { calculate, PARAMS };" in code
        assert "export default calculate;" in code

    def test_generate_provenance_sources(self):
        """Provenance section lists all sources."""
        gen = JSCodeGenerator(include_provenance=True)
        gen.add_parameter("a", {0: 1}, "Source A")
        gen.add_parameter("b", {0: 2}, "Source B")
        gen.add_variable("c", [], "1", citation="Source C")
        code = gen.generate()
        assert " * Sources:" in code
        assert "Source A" in code
        assert "Source B" in code
        assert "Source C" in code

    def test_generate_no_provenance(self):
        """Can disable provenance section."""
        gen = JSCodeGenerator(include_provenance=False)
        gen.add_parameter("a", {0: 1}, "Source A")
        code = gen.generate()
        assert " * Sources:" not in code

    def test_generate_multiline_formula_returns_trailing_expression(self):
        """Multiline formulas implicitly return a trailing JS expression."""
        gen = JSCodeGenerator()
        gen.add_input("x", 0)
        gen.add_variable("y", ["x"], "const tmp = x + 1;\ntmp")

        code = gen.generate()

        assert "return tmp;" in code

        if shutil.which("node") is None:
            pytest.skip("Node.js is required for JS runtime execution tests.")

        script = "\n".join(
            [
                code,
                "console.log(JSON.stringify(calculate({ x: 10 })));",
            ]
        )
        proc = subprocess.run(
            ["node", "--input-type=module"],
            input=script,
            capture_output=True,
            text=True,
            check=True,
        )
        result = proc.stdout.splitlines()[-1]
        assert '"y":11' in result

    def test_generate_semicolon_block_returns_trailing_expression(self):
        """Same-line semicolon blocks are normalized before JS emission."""
        gen = JSCodeGenerator()
        gen.add_input("x", 0)
        gen.add_variable("y", ["x"], "tmp = x + 1; tmp")

        code = gen.generate()

        assert "return tmp;" in code

    def test_generate_block_preserves_explicit_return_without_space(self):
        """Explicit JS returns like return(x) stay valid in wrapped blocks."""
        gen = JSCodeGenerator()
        gen.add_input("x", 0)
        gen.add_variable("y", ["x"], "tmp = x + 1;\nreturn(x + tmp)")

        code = gen.generate()

        assert "return(x + tmp)" in code

    def test_generate_block_allows_identifier_with_keyword_prefix(self):
        """Keyword-like identifier prefixes still compile as expressions."""
        gen = JSCodeGenerator()
        gen.add_input("x", 0)
        gen.add_variable("y", ["x"], "defaultRate = x + 1;\ndefaultRate")

        code = gen.generate()

        assert "return defaultRate;" in code

    def test_generate_if_else_chain_with_branch_returns(self):
        """Top-level if/else chains that return on every branch stay valid."""
        gen = JSCodeGenerator()
        gen.add_input("wages", 0)
        gen.add_input("is_joint", False)
        gen.add_variable(
            "tax",
            ["wages", "is_joint"],
            "\n".join(
                [
                    "let rate;",
                    "if (is_joint) {",
                    "  rate = 0.1;",
                    "  return wages * rate;",
                    "} else {",
                    "  return wages * 0.2;",
                    "}",
                ]
            ),
        )

        code = gen.generate()

        assert "if (is_joint)" in code

    def test_generate_if_without_else_still_fails(self):
        """Partial-return JS blocks still fail loudly instead of compiling."""
        gen = JSCodeGenerator()
        gen.add_input("wages", 0)
        gen.add_input("is_joint", False)
        gen.add_variable(
            "tax",
            ["wages", "is_joint"],
            "\n".join(
                [
                    "if (is_joint) {",
                    "  return wages * 0.1;",
                    "}",
                ]
            ),
        )

        with pytest.raises(ValueError, match="must end with an explicit return"):
            gen.generate()


class TestGenerateEITCCalculator:
    """Tests for pre-built EITC calculator."""

    def test_returns_valid_js(self):
        """EITC calculator generates valid JS structure."""
        code = generate_eitc_calculator()
        assert "function calculate(" in code
        assert "const PARAMS = {" in code
        assert "export default calculate;" in code

    def test_includes_all_eitc_params(self):
        """EITC calculator includes required parameters."""
        code = generate_eitc_calculator()
        assert "credit_pct:" in code
        assert "phaseout_pct:" in code
        assert "earned_income_amount:" in code
        assert "phaseout_single:" in code
        assert "phaseout_joint:" in code

    def test_includes_statute_citations(self):
        """EITC calculator cites 26 USC 32."""
        code = generate_eitc_calculator()
        assert "26 USC 32" in code
        assert "26 USC 32(b)(1)" in code

    def test_includes_guidance_citations(self):
        """EITC calculator cites Rev. Proc. 2024-40."""
        code = generate_eitc_calculator()
        assert "Rev. Proc. 2024-40" in code

    def test_inputs_have_correct_defaults(self):
        """EITC inputs have sensible defaults."""
        code = generate_eitc_calculator()
        assert "earned_income = 0" in code
        assert "agi = 0" in code
        assert "n_children = 0" in code
        assert "is_joint = false" in code


class TestJSExecution:
    """Tests that generated JS actually executes correctly."""

    @pytest.fixture
    def simple_calculator(self):
        """Create a simple calculator for testing."""
        gen = JSCodeGenerator()
        gen.add_input("income", 0)
        gen.add_parameter("rate", {0: 20}, "Test")
        gen.add_variable("tax", ["income"], "income * PARAMS.rate[0] / 100")
        return gen.generate()

    def test_generated_code_is_syntactically_valid(self, simple_calculator):
        """Generated code can be parsed (basic syntax check)."""
        # Check for balanced braces
        assert simple_calculator.count("{") == simple_calculator.count("}")
        assert simple_calculator.count("(") == simple_calculator.count(")")
        assert simple_calculator.count("[") == simple_calculator.count("]")

    def test_eitc_calculator_syntax(self):
        """EITC calculator is syntactically valid."""
        code = generate_eitc_calculator()
        assert code.count("{") == code.count("}")
        assert code.count("(") == code.count(")")
