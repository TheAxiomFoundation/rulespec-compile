"""Tests for the RuleSpec parser."""

import subprocess
from pathlib import Path

import pytest

from src.rulespec_compile.parser import ParserError, RuleSpecFile, parse_rulespec


class TestSourceBlock:
    """Tests for top-level source metadata."""

    def test_parse_source_block(self):
        """A `source:` block is parsed into a SourceBlock."""
        result = parse_rulespec(
            """
source:
  lawarchive: us/statute/26/32/2025-01-01
  citation: "26 USC 32"
  accessed: 2025-12-12
"""
        )

        assert result.source is not None
        assert result.source.lawarchive == "us/statute/26/32/2025-01-01"
        assert result.source.citation == "26 USC 32"
        assert result.source.accessed == "2025-12-12"

    def test_source_block_optional(self):
        """Files do not need source metadata."""
        result = parse_rulespec(
            """
foo:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return 0
"""
        )

        assert result.source is None

    def test_source_block_requires_fields(self):
        """An empty source block fails loudly."""
        with pytest.raises(ParserError, match="source: block must contain"):
            parse_rulespec(
                """
source:
foo:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return 0
"""
            )


class TestRuleSpecV1:
    """Tests for the current structured RuleSpec v1 envelope."""

    def test_parse_rulespec_v1_rules(self):
        """RuleSpec v1 rules map into the compiler's rule model."""
        result = parse_rulespec(
            """
format: rulespec/v1
module:
  summary: |-
    Test source.
rules:
  - name: rate
    kind: parameter
    dtype: Rate
    source: Test Code
    source_url: https://example.test/rate
    versions:
      - effective_from: '2024-01-01'
        formula: '0.2'
  - name: tax
    kind: derived
    entity: Person
    period: Year
    dtype: Money
    unit: USD
    source: Test Code
    source_url: https://example.test/tax
    versions:
      - effective_from: '2024-01-01'
        formula: wages * rate
"""
        )

        assert result.statute_text == "Test source."
        assert result.parameters["rate"].temporal[0].value == 0.2
        assert result.parameters["rate"].reference == "https://example.test/rate"
        assert [variable.name for variable in result.variables] == ["tax"]
        assert result.variables[0].temporal[0].code == "wages * rate"
        assert result.variables[0].source_citation == "Test Code"

    def test_rulespec_v1_compiles_bare_expression_formula(self):
        """RuleSpec v1 formulas compile with the same implicit-return semantics."""
        lowered = parse_rulespec(
            """
format: rulespec/v1
rules:
  - name: rate
    kind: parameter
    dtype: Rate
    source: Test Code
    versions:
      - effective_from: '2024-01-01'
        formula: '0.2'
  - name: tax
    kind: derived
    entity: Person
    period: Year
    dtype: Money
    source: Test Code
    versions:
      - effective_from: '2024-01-01'
        formula: wages * rate
"""
        ).to_lowered_program(outputs=["tax"])

        assert lowered.outputs[0].name == "tax"
        assert lowered.inputs[0].external_name == "wages"


class TestParameterDefinitions:
    """Tests for RuleSpec parameter parsing."""

    def test_parse_scalar_parameter(self):
        """Parameters can use temporal scalar entries."""
        result = parse_rulespec(
            """
niit_rate:
  source: "26 USC 1411"
  from 2013-01-01: 0.038
"""
        )

        param = result.parameters["niit_rate"]
        assert param.source == "26 USC 1411"
        assert len(param.temporal) == 1
        assert param.temporal[0].from_date == "2013-01-01"
        assert param.temporal[0].value == 0.038

    def test_parse_multiple_temporal_values(self):
        """Parameters can have multiple temporal scalar entries."""
        result = parse_rulespec(
            """
threshold:
  source: "Rev. Proc. 2024-40"
  from 2024-01-01: 250000
  from 2023-01-01: 220000
  from 2022-01-01: 200000
"""
        )

        param = result.parameters["threshold"]
        assert [entry.value for entry in param.temporal] == [250000, 220000, 200000]
        assert param.values == {0: 250000, 1: 220000, 2: 200000}

    def test_parse_indexed_values_block(self):
        """Parameters can define indexed lookup tables with `values:`."""
        result = parse_rulespec(
            """
credit_pct:
  source: "26 USC 32(b)(1)"
  values:
    0: 7.65
    1: 34
    2: 40
    3: 45
"""
        )

        assert result.parameters["credit_pct"].values == {
            0: 7.65,
            1: 34.0,
            2: 40.0,
            3: 45.0,
        }

    def test_parameter_with_description(self):
        """Parameters keep description and unit metadata."""
        result = parse_rulespec(
            """
contribution_rate:
  description: "Household contribution as share of net income"
  unit: rate
  source: "USDA FNS"
  from 2024-01-01: 0.30
"""
        )

        param = result.parameters["contribution_rate"]
        assert param.description == "Household contribution as share of net income"
        assert param.unit == "rate"
        assert param.source == "USDA FNS"

    def test_invalid_values_block_fails_loudly(self):
        """Malformed indexed parameter tables are rejected."""
        with pytest.raises(ParserError, match="Invalid parameter values entry"):
            parse_rulespec(
                """
credit_pct:
  values:
    first: 7.65
"""
            )

    def test_rejects_mixed_values_and_temporal_entries(self):
        """Indexed tables and temporal entries cannot be mixed in one parameter."""
        with pytest.raises(ParserError, match="cannot mix a values block"):
            parse_rulespec(
                """
credit_pct:
  values:
    0: 7.65
  from 2024-01-01: 0.0765
"""
            )


class TestVariableDefinitions:
    """Tests for RuleSpec variable parsing."""

    def test_parse_variable_with_temporal_formula(self):
        """Variables can store a formula under a `from` entry."""
        result = parse_rulespec(
            """
niit:
  entity: TaxUnit
  period: Year
  dtype: Money
  from 2013-01-01:
    magi = agi + foreign_earned_income_exclusion
    threshold = 200000
    excess = max(0, magi - threshold)
    return min(net_investment_income, excess) * 0.038
"""
        )

        var = result.variables[0]
        assert var.name == "niit"
        assert var.entity == "TaxUnit"
        assert var.period == "Year"
        assert var.dtype == "Money"
        assert "0.038" in var.effective_formula
        assert "max(0, magi - threshold)" in var.effective_formula

    def test_variable_type_inference(self):
        """Entity/period/dtype distinguishes variables from parameters."""
        result = parse_rulespec(
            """
rate:
  from 2024-01-01: 0.038

tax:
  entity: TaxUnit
  period: Year
  dtype: Money
  from 2024-01-01:
    return income * 0.038
"""
        )

        assert "rate" in result.parameters
        assert [variable.name for variable in result.variables] == ["tax"]

    def test_variable_with_label(self):
        """Variables preserve label metadata."""
        result = parse_rulespec(
            """
eitc:
  entity: TaxUnit
  period: Year
  dtype: Money
  label: "Earned Income Tax Credit"
  from 2025-01-01:
    return max(0, earned_income * 0.34)
"""
        )

        assert result.variables[0].label == "Earned Income Tax Credit"

    def test_variable_with_default_metadata(self):
        """No-formula variable defaults are preserved for declared inputs."""
        result = parse_rulespec(
            """
is_us_citizen_national_or_resident:
  entity: Person
  period: Year
  dtype: Boolean
  default: false
"""
        )

        variable = result.variables[0]
        assert variable.default is False

    def test_scalar_computed_rule_without_entity_stays_variable(self):
        """Entity-less rules with code blocks stay computed rules, not parameters."""
        result = parse_rulespec(
            """
snap_self_employment_cost_exclusion:
  label: "SNAP self-employment cost exclusion"
  description: "Reduction for production costs"
  from 2008-10-01:
    min(
      snap_nonfarm_self_employment_production_costs,
      snap_nonfarm_self_employment_gross_income,
    ) + snap_farm_self_employment_production_costs
"""
        )

        assert "snap_self_employment_cost_exclusion" not in result.parameters
        assert [variable.name for variable in result.variables] == [
            "snap_self_employment_cost_exclusion"
        ]
        assert result.rule_decls[0].is_computed_rule is True

    def test_variable_with_multiple_temporal_formulas(self):
        """Variables can define multiple dated formulas."""
        result = parse_rulespec(
            """
credit:
  entity: TaxUnit
  period: Year
  dtype: Money
  from 2026-01-01:
    return income * 0.10
  from 2018-01-01:
    return income * 0.15
"""
        )

        var = result.variables[0]
        assert [entry.from_date for entry in var.temporal] == [
            "2026-01-01",
            "2018-01-01",
        ]
        assert "0.10" in var.effective_formula

    def test_variable_imports_block_parses_spec_style_imports(self):
        """Variables can declare per-rule imports with `path#symbol` syntax."""
        result = parse_rulespec(
            """
first_reduction:
  imports:
    - 26/62#adjusted_gross_income
    - 26/21/a/2#base_applicable_percentage as base_pct
  entity: TaxUnit
  period: Year
  dtype: Rate
  from 2002-01-01:
    max(0, adjusted_gross_income - base_pct)
"""
        )

        variable = result.variables[0]
        assert [spec.path for spec in variable.import_specs] == ["26/62", "26/21/a/2"]
        assert variable.import_specs[0].symbols[0].name == "adjusted_gross_income"
        assert variable.import_specs[1].symbols[0].alias == "base_pct"

    def test_variable_rejects_parameter_values_block(self):
        """Variables cannot use the parameter `values:` table syntax."""
        with pytest.raises(ParserError, match="cannot define a parameter values block"):
            parse_rulespec(
                """
tax:
  entity: Person
  period: Year
  dtype: Money
  values:
    0: 1
"""
            )


class TestImportsAndExports:
    """Tests for top-level import/export parsing."""

    def test_parse_rulespec_collects_top_level_imports(self, tmp_path):
        """Top-level import strings are preserved on the parsed file."""
        origin = tmp_path / "main.yaml"
        result = parse_rulespec(
            """
import "./shared.yaml"
import "../common/base.yaml"

tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return 1
""",
            origin=origin,
        )

        assert result.imports == ["./shared.yaml", "../common/base.yaml"]
        assert result.origin == origin.resolve()

    def test_parse_rulespec_collects_import_aliases(self, tmp_path):
        """Aliased imports are preserved structurally."""
        origin = tmp_path / "main.yaml"
        result = parse_rulespec(
            """
import "./shared.yaml" as shared
import "./base.yaml"
""",
            origin=origin,
        )

        assert result.imports == ["./shared.yaml", "./base.yaml"]
        assert result.import_specs[0].path == "./shared.yaml"
        assert result.import_specs[0].alias == "shared"
        assert result.import_specs[1].path == "./base.yaml"
        assert result.import_specs[1].alias is None

    def test_parse_rulespec_collects_top_level_spec_style_imports(self, tmp_path):
        """Top-level `imports:` blocks are preserved for live-stack RuleSpec files."""
        origin = tmp_path / "main.yaml"
        result = parse_rulespec(
            """
imports:
  - statute/crs/26-2-703/12#is_individual_responsibility_contract
  - regulation/9-CCR-2503-6/3.606.1/F#need_standard_for_assistance_unit
""",
            origin=origin,
        )

        assert result.imports == [
            "statute/crs/26-2-703/12",
            "regulation/9-CCR-2503-6/3.606.1/F",
        ]
        assert result.import_specs[0].symbols[0].name == (
            "is_individual_responsibility_contract"
        )
        assert result.import_specs[1].symbols[0].name == (
            "need_standard_for_assistance_unit"
        )

    def test_parse_selective_imports_exports_and_re_exports(self, tmp_path):
        """Selective imports, exports, and re-exports are preserved."""
        origin = tmp_path / "main.yaml"
        result = parse_rulespec(
            """
export tax as benefit_amount, taxable_income
export from "./shared.yaml" import upstream_benefit as benefit_amount_2
from "./shared.yaml" import rate, threshold as income_threshold
""",
            origin=origin,
        )

        assert result.imports == ["./shared.yaml"]
        assert result.exports == [
            "benefit_amount",
            "taxable_income",
            "benefit_amount_2",
        ]
        assert result.export_specs[0].name == "tax"
        assert result.export_specs[0].alias == "benefit_amount"
        assert result.export_specs[1].name == "taxable_income"
        assert result.export_specs[1].alias is None
        assert result.re_export_specs[0].path == "./shared.yaml"
        assert result.re_export_specs[0].symbols[0].name == "upstream_benefit"
        assert result.re_export_specs[0].symbols[0].alias == "benefit_amount_2"
        assert result.import_specs[0].path == "./shared.yaml"
        assert result.import_specs[0].symbols[0].name == "rate"
        assert result.import_specs[0].symbols[1].alias == "income_threshold"


class TestFormulaConversion:
    """Tests for parsed formulas compiling to JavaScript."""

    def test_converts_min_to_math_min(self):
        """min() is converted to Math.min()."""
        code = (
            parse_rulespec(
                """
x:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return min(a, b)
"""
            )
            .to_js_generator()
            .generate()
        )

        assert "Math.min(a, b)" in code

    def test_converts_max_to_math_max(self):
        """max() is converted to Math.max()."""
        code = (
            parse_rulespec(
                """
x:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return max(a, b)
"""
            )
            .to_js_generator()
            .generate()
        )

        assert "Math.max(a, b)" in code

    def test_converts_round_to_math_round(self):
        """round() is converted to Math.round()."""
        code = (
            parse_rulespec(
                """
x:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return round(income)
"""
            )
            .to_js_generator()
            .generate()
        )

        assert "Math.round(income)" in code

    def test_converts_parameter_references(self):
        """Parameter references compile to PARAMS lookups."""
        code = (
            parse_rulespec(
                """
rate:
  values:
    0: 20

x:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return rate[0] * income
"""
            )
            .to_js_generator()
            .generate()
        )

        assert "PARAMS.rate[0]" in code

    def test_nested_math_functions(self):
        """Nested helper calls render correctly."""
        code = (
            parse_rulespec(
                """
x:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return max(0, round(min(a, b)))
"""
            )
            .to_js_generator()
            .generate()
        )

        assert "Math.max(0, Math.round(Math.min(a, b)))" in code


class TestFullFiles:
    """Tests for complete RuleSpec files."""

    def test_parse_complete_file(self):
        """A full RuleSpec file with source, parameters, and variables parses."""
        result = parse_rulespec(
            """
# 26 USC 32 - Earned Income Tax Credit

source:
  lawarchive: us/statute/26/32/2025-01-01
  citation: "26 USC 32"
  accessed: 2025-12-12

credit_pct:
  source: statute/26/32/b/1/credit_pct

earned_income_amount:
  source: guidance/irs/rp-24-40/eitc/earned_income_amount

eitc:
  entity: TaxUnit
  period: Year
  dtype: Money
  label: "Earned Income Tax Credit"
  from 2025-01-01:
    earned_cap = earned_income_amount[n_children]
    credit_base = credit_pct[n_children] * min(earned_income, earned_cap)
    return max(0, credit_base - phaseout)
"""
        )

        assert isinstance(result, RuleSpecFile)
        assert result.source is not None
        assert result.source.citation == "26 USC 32"
        assert "credit_pct" in result.parameters
        assert [variable.name for variable in result.variables] == ["eitc"]

    def test_rulespec_file_to_js_generator(self):
        """Parsed RuleSpec files can generate JavaScript."""
        code = (
            parse_rulespec(
                """
source:
  citation: "Test"
  accessed: 2025-01-01

rate:
  source: test/path

tax:
  label: "Tax"
  entity: Person
  period: Year
  dtype: Money
  from 2025-01-01:
    return income * 0.2
"""
            )
            .to_js_generator()
            .generate()
        )

        assert "function calculate(" in code


class TestExampleFiles:
    """Tests for the shipped example `.yaml` files."""

    def _assert_valid_js(self, code: str):
        proc = subprocess.run(
            ["node", "--input-type=module", "--check"],
            input=code,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"JS syntax error: {proc.stderr}"

    def test_eitc_example_parses(self):
        """examples/eitc.yaml parses with indexed parameter tables."""
        eitc_path = Path(__file__).parent.parent / "examples" / "eitc.yaml"
        result = parse_rulespec(eitc_path.read_text())

        assert result.source is not None
        assert result.source.citation == "26 USC 32"
        assert len(result.parameters) == 5
        assert result.parameters["credit_pct"].values == {
            0: 7.65,
            1: 34.0,
            2: 40.0,
            3: 45.0,
        }
        assert [variable.name for variable in result.variables] == ["eitc"]

    def test_eitc_example_compiles_to_valid_js(self):
        """examples/eitc.yaml compiles to valid JS."""
        eitc_path = Path(__file__).parent.parent / "examples" / "eitc.yaml"
        self._assert_valid_js(
            parse_rulespec(eitc_path.read_text()).to_js_generator().generate()
        )

    def test_simple_tax_example_compiles(self):
        """examples/simple_tax.yaml compiles to valid JS."""
        simple_path = Path(__file__).parent.parent / "examples" / "simple_tax.yaml"
        self._assert_valid_js(
            parse_rulespec(simple_path.read_text()).to_js_generator().generate()
        )

    def test_ctc_example_parses(self):
        """examples/ctc.yaml parses with indexed parameter values."""
        ctc_path = Path(__file__).parent.parent / "examples" / "ctc.yaml"
        result = parse_rulespec(ctc_path.read_text())

        assert result.source is not None
        assert result.source.citation == "26 USC 24"
        assert len(result.parameters) == 7
        assert result.parameters["credit_per_child"].values == {0: 2200.0}
        assert [variable.name for variable in result.variables] == ["ctc", "actc"]

    def test_ctc_example_compiles_to_valid_js(self):
        """examples/ctc.yaml compiles to valid JS."""
        ctc_path = Path(__file__).parent.parent / "examples" / "ctc.yaml"
        self._assert_valid_js(
            parse_rulespec(ctc_path.read_text()).to_js_generator().generate()
        )

    def test_snap_example_parses(self):
        """examples/snap.yaml parses with indexed parameter tables."""
        snap_path = Path(__file__).parent.parent / "examples" / "snap.yaml"
        result = parse_rulespec(snap_path.read_text())

        assert result.source is not None
        assert result.source.citation == "7 USC 2017"
        assert len(result.parameters) == 5
        assert result.parameters["max_allotment"].values[1] == 292.0
        assert [variable.name for variable in result.variables] == [
            "snap_eligible",
            "snap_benefit",
        ]

    def test_snap_example_compiles_to_valid_js(self):
        """examples/snap.yaml compiles to valid JS."""
        snap_path = Path(__file__).parent.parent / "examples" / "snap.yaml"
        self._assert_valid_js(
            parse_rulespec(snap_path.read_text()).to_js_generator().generate()
        )


class TestStatuteText:
    """Tests for top-level statute text parsing."""

    def test_parse_statute_text(self):
        """Triple-quoted statute text is preserved."""
        result = parse_rulespec(
            '''
"""
In the case of an individual, there shall be imposed
a tax equal to 3.8 percent of the lesser of net investment
income or excess MAGI over the threshold amount.
"""
'''
        )

        assert result.statute_text is not None
        assert "3.8 percent" in result.statute_text
        assert "net investment" in result.statute_text

    def test_statute_text_optional(self):
        """Statute text is optional."""
        result = parse_rulespec(
            """
rate:
  from 2024-01-01: 0.30
"""
        )

        assert result.statute_text is None


class TestLegacySyntaxRejection:
    """Tests for removed legacy brace syntax."""

    def test_rejects_legacy_parameter_blocks(self):
        """Legacy brace syntax is rejected with a user-facing parser error."""
        with pytest.raises(ParserError, match="Legacy brace syntax"):
            parse_rulespec(
                """
parameter rate {
  source: "Test"
  values {
    0: 10
  }
}
"""
            )

    def test_rejects_legacy_variable_blocks(self):
        """Legacy variable braces are rejected instead of half-parsed."""
        with pytest.raises(ParserError, match="Legacy brace syntax"):
            parse_rulespec(
                """
variable tax {
  formula { return 0 }
}
"""
            )
