"""Tests to cover remaining gaps in js_generator.py, snap.py, parser.py,
and python_generator.py.

Each test class targets specific uncovered lines.
"""

from unittest.mock import patch

import pytest

# ============================================================
# js_generator.py - missing lines: 128, 184-235, 315
# ============================================================


class TestJSGeneratorTypeScript:
    """Test TypeScript generation (lines 184-235)."""

    def test_typescript_generation(self):
        """TypeScript mode generates proper interfaces and typed function."""
        from src.rulespec_compile.js_generator import JSCodeGenerator

        gen = JSCodeGenerator(typescript=True)
        gen.add_input("income", 0, "number")
        gen.add_input("is_joint", False, "boolean")
        gen.add_parameter("rate", {0: 20}, "26 USC 1")
        gen.add_variable(
            "tax",
            ["income"],
            "income * PARAMS.rate[0] / 100",
            citation="26 USC 1(a)",
        )
        code = gen.generate()

        # Check TypeScript interface
        assert "interface CalculatorInputs {" in code
        assert "income?: number;" in code
        assert "is_joint?: boolean;" in code

        # Check result interface
        assert "interface CalculatorResult {" in code
        assert "tax: number;" in code
        assert "citations:" in code

        # Check function signature
        assert "function calculate(inputs: CalculatorInputs" in code
        assert "CalculatorResult {" in code

        # Check destructure
        assert "const {" in code

        # Check variable typing
        assert "const tax: number =" in code

        # Check return with citations
        assert 'source: "26 USC 1"' in code
        assert 'source: "26 USC 1(a)"' in code

    def test_typescript_with_no_citation(self):
        """TypeScript handles variables without citations."""
        from src.rulespec_compile.js_generator import JSCodeGenerator

        gen = JSCodeGenerator(typescript=True)
        gen.add_input("x", 0, "number")
        gen.add_variable("y", ["x"], "x * 2")
        code = gen.generate()
        assert "const y: number = x * 2;" in code

    def test_typescript_parameter_no_source(self):
        """TypeScript handles parameters without source."""
        from src.rulespec_compile.js_generator import JSCodeGenerator

        gen = JSCodeGenerator(typescript=True)
        gen.add_parameter("rate", {0: 10}, "")  # no source
        gen.add_input("x", 0, "number")
        gen.add_variable("y", ["x"], "x * 2")
        code = gen.generate()
        # The param without source should not appear in citations
        assert "rate:" in code


class TestJSGeneratorMainBlock:
    """Test __main__ block (line 315)."""

    def test_main_block_execution(self):
        """__main__ block generates EITC calculator to stdout."""
        import runpy

        with patch("builtins.print") as mock_print:
            runpy.run_module(
                "rulespec_compile.js_generator", run_name="__main__", alter_sys=True
            )
            output = mock_print.call_args[0][0]
            assert "function calculate(" in output


class TestJSGeneratorLine128:
    """Test line 128 - TypeScript branch in generate()."""

    def test_generate_dispatches_to_typescript(self):
        """When typescript=True, generate dispatches to TS."""
        from src.rulespec_compile.js_generator import JSCodeGenerator

        gen = JSCodeGenerator(typescript=True)
        gen.add_input("x", 0, "number")
        gen.add_variable("y", ["x"], "x + 1")
        code = gen.generate()
        assert "interface CalculatorInputs" in code


# ============================================================
# snap.py - missing lines: 99-103
# ============================================================


class TestSNAPEligibility:
    """Test calculate_snap_eligible function (lines 99-103)."""

    def test_eligible_below_limit(self):
        """Household below gross income limit is eligible."""
        from src.rulespec_compile.calculators.snap import calculate_snap_eligible

        result = calculate_snap_eligible(household_size=1, gross_income=500)
        assert result.eligible is True
        assert result.gross_income_limit == 1580

    def test_not_eligible_above_limit(self):
        """Household above gross income limit is not eligible."""
        from src.rulespec_compile.calculators.snap import calculate_snap_eligible

        result = calculate_snap_eligible(household_size=1, gross_income=2000)
        assert result.eligible is False

    def test_eligible_at_limit(self):
        """Household at exactly the limit is eligible."""
        from src.rulespec_compile.calculators.snap import calculate_snap_eligible

        result = calculate_snap_eligible(household_size=1, gross_income=1580)
        assert result.eligible is True

    def test_large_household_capped_at_8(self):
        """Household > 8 uses 8-person limits."""
        from src.rulespec_compile.calculators.snap import calculate_snap_eligible

        result8 = calculate_snap_eligible(household_size=8, gross_income=0)
        result10 = calculate_snap_eligible(household_size=10, gross_income=0)
        assert result8.gross_income_limit == result10.gross_income_limit

    def test_citations_included(self):
        """Eligibility result includes citations."""
        from src.rulespec_compile.calculators.snap import calculate_snap_eligible

        result = calculate_snap_eligible(household_size=1, gross_income=0)
        assert len(result.citations) == 2
        assert any("7 CFR 273.9(a)(1)" in c["source"] for c in result.citations)


# ============================================================
# parser.py - missing lines: 64, 171-173, 232, 297, 368, 416-417, 421
# ============================================================


class TestParserLine64:
    """Test VariableBlock.effective_formula returns empty string."""

    def test_effective_formula_empty(self):
        """effective_formula returns '' when no formula and no temporal code."""
        from src.rulespec_compile.parser import VariableBlock

        var = VariableBlock(name="test")
        assert var.effective_formula == ""

    def test_effective_formula_with_temporal_values_only(self):
        """Returns '' when temporal entries have values only."""
        from src.rulespec_compile.parser import TemporalEntry, VariableBlock

        var = VariableBlock(
            name="test",
            temporal=[
                TemporalEntry(from_date="2024-01-01", value=42.0),
            ],
        )
        assert var.effective_formula == ""


class TestParserInlineStatuteText:
    """Test inline triple-quoted statute text on a single line."""

    def test_inline_statute_text(self):
        """Triple-quoted text on a single line is parsed."""
        from src.rulespec_compile.parser import parse_rulespec

        rulespec = '"""This is inline statute text."""\n'
        result = parse_rulespec(rulespec)
        assert result.statute_text == "This is inline statute text."


class TestParserLine232:
    """Test unrecognized line in main parser loop (line 232: i += 1)."""

    def test_unrecognized_lines_skipped(self):
        """Lines that don't match any pattern are skipped."""
        from src.rulespec_compile.parser import parse_rulespec

        rulespec = """
this is not a valid block
still not a valid block

x:
  entity: Person
  period: Year
  dtype: Integer
  from 2024-01-01:
    return 0
"""
        result = parse_rulespec(rulespec)
        assert len(result.variables) == 1
        assert result.variables[0].name == "x"


class TestParserLine297:
    """Test unrecognized line in unified definition (line 297: i += 1)."""

    def test_unrecognized_attr_in_unified_def(self):
        """Unrecognized lines within a unified definition are skipped."""
        from src.rulespec_compile.parser import parse_rulespec

        rulespec = """
rate:
  source: "Test"
  ??? this is not a valid attr line ???
  from 2024-01-01: 0.30
"""
        result = parse_rulespec(rulespec)
        assert "rate" in result.parameters
        assert result.parameters["rate"].source == "Test"


class TestParserLine368:
    """Test empty/comment-only source block (line 368: continue on blank lines)."""

    def test_source_block_with_blank_lines(self):
        """Source block with blank/comment lines parses correctly."""
        from src.rulespec_compile.parser import parse_rulespec

        rulespec = """
source:
  # A comment

  citation: "Test Citation"
  accessed: 2025-01-01
tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return 0
"""
        result = parse_rulespec(rulespec)
        assert result.source.citation == "Test Citation"


class TestParserParameterEdgeCases:
    """Test parameter block edge cases."""

    def test_parameter_values_with_invalid_entries(self):
        """Malformed indexed parameter tables fail loudly."""
        from src.rulespec_compile.parser import ParserError, parse_rulespec

        rulespec = """
test_param:
  source: "Test"
  values:
    0: 10
    bad_key: 20
    1: not_a_number
    2: 30
"""
        with pytest.raises(ParserError, match="Invalid parameter values entry"):
            parse_rulespec(rulespec)

    def test_parameter_source_no_quotes(self):
        """Parameter source without quotes is parsed."""
        from src.rulespec_compile.parser import parse_rulespec

        rulespec = """
test_param:
  source: some/path/to/source
  values:
    0: 10
"""
        result = parse_rulespec(rulespec)
        assert result.parameters["test_param"].source == "some/path/to/source"

    def test_parameter_no_source(self):
        """Parameter block with no source still parses."""
        from src.rulespec_compile.parser import parse_rulespec

        rulespec = """
test_param:
  values:
    0: 10
"""
        result = parse_rulespec(rulespec)
        param = result.parameters["test_param"]
        assert param.source == ""
        assert param.values == {0: 10.0}


# ============================================================
# python_generator.py - missing lines: 148, 161, 163, 287
# ============================================================


class TestPythonGeneratorStringDefault:
    """Test string default values in Python generator (lines 148, 163)."""

    def test_string_default_with_type_hints(self):
        """String defaults are quoted in type-hinted function."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator(type_hints=True)
        gen.add_input("name", "world", "str")
        gen.add_variable("greeting", ["name"], 'f"Hello {name}"')
        code = gen.generate()
        assert 'name: str = "world"' in code

    def test_string_default_without_type_hints(self):
        """String defaults are quoted in non-type-hinted function."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator(type_hints=False)
        gen.add_input("name", "world", "str")
        gen.add_variable("greeting", ["name"], 'f"Hello {name}"')
        code = gen.generate()
        assert 'name="world"' in code


class TestPythonGeneratorBoolNoTypeHints:
    """Test bool defaults without type hints (line 161)."""

    def test_bool_default_without_type_hints(self):
        """Bool defaults work without type hints."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator(type_hints=False)
        gen.add_input("flag", True, "bool")
        gen.add_variable("result", ["flag"], "1 if flag else 0")
        code = gen.generate()
        assert "flag=True" in code


class TestPythonGeneratorMainBlock:
    """Test __main__ block (line 287)."""

    def test_main_block_execution(self):
        """__main__ block generates EITC calculator to stdout."""
        import runpy

        with patch("builtins.print") as mock_print:
            runpy.run_module(
                "rulespec_compile.python_generator", run_name="__main__", alter_sys=True
            )
            output = mock_print.call_args[0][0]
            assert "def calculate(" in output


# ============================================================
# cli.py - __main__ block (line 112)
# ============================================================


class TestCLIMainBlock:
    """Test cli.py __main__ block."""

    def test_cli_main_block(self):
        """__main__ block calls main()."""
        import runpy

        with patch("sys.argv", ["rulespec-compile", "eitc"]):
            with patch("builtins.print"):
                runpy.run_module(
                    "rulespec_compile.cli", run_name="__main__", alter_sys=True
                )


# ============================================================
# validation/cli.py - __main__ block (line 149)
# ============================================================


class TestValidationCLIMainBlock:
    """Test validation/cli.py __main__ block."""

    def test_validation_cli_main_block(self):
        """__main__ block calls main()."""
        import runpy
        from unittest.mock import MagicMock

        from src.rulespec_compile.validation.comparator import ComparisonResults

        mock_results = ComparisonResults(
            total_households=10,
            variables_compared=["eitc"],
            matches={"eitc": 10},
            mismatches={"eitc": []},
            match_rates={"eitc": 100.0},
            config=MagicMock(),
        )

        with patch("sys.argv", ["rulespec-validate", "--mode", "full"]):
            # Patch validate_full on the comparator module since that's where
            # the validation CLI imports it from
            with patch(
                "rulespec_compile.validation.comparator.validate_full",
                return_value=mock_results,
            ):
                runpy.run_module(
                    "rulespec_compile.validation.cli",
                    run_name="__main__",
                    alter_sys=True,
                )
