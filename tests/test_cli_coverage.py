"""Tests for cli.py to achieve 100% coverage.

Tests the main() CLI entry point by mocking sys.argv and verifying
all branches: compile (success, file-not-found, stdout), eitc (JS, Python,
to file, to stdout), no-command.
"""

import json
from unittest.mock import patch

import pytest

from src.rulespec_compile.cli import main
from src.rulespec_compile.harness import HarnessResult, HarnessSummary


class TestCLIMainCompile:
    """Test compile command branches."""

    def test_compile_file_not_found(self, tmp_path):
        """Compile with non-existent file prints error and exits 1."""
        fake_input = str(tmp_path / "nonexistent.yaml")
        with patch("sys.argv", ["rulespec-compile", "compile", fake_input]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_compile_to_stdout(self, tmp_path):
        """Compile with no -o prints JS to stdout."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: x
  kind: derived
  entity: Person
  period: Year
  dtype: Integer
  versions:
  - effective_from: '2024-01-01'
    formula: return 42
"""
        )
        with patch("sys.argv", ["rulespec-compile", "compile", str(input_file)]):
            # Should not raise, just prints to stdout
            with patch("builtins.print") as mock_print:
                main()
                # Should have printed the generated code
                output = mock_print.call_args_list[0][0][0]
                assert "calculate" in output

    def test_compile_to_file(self, tmp_path):
        """Compile with -o writes JS to file."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: x
  kind: derived
  entity: Person
  period: Year
  dtype: Integer
  versions:
  - effective_from: '2024-01-01'
    formula: return 42
"""
        )
        output_file = tmp_path / "output.js"
        with patch(
            "sys.argv",
            ["rulespec-compile", "compile", str(input_file), "-o", str(output_file)],
        ):
            main()
            assert output_file.exists()
            content = output_file.read_text()
            assert "calculate" in content

    def test_compile_python_to_stdout(self, tmp_path):
        """Compile --python prints Python to stdout."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
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
    formula: return wages * rate
"""
        )
        with patch(
            "sys.argv",
            ["rulespec-compile", "compile", str(input_file), "--python"],
        ):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "def calculate(" in output

    def test_compile_rust_to_stdout(self, tmp_path):
        """Compile --rust prints Rust to stdout."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
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
    formula: return wages * rate
"""
        )
        with patch(
            "sys.argv",
            ["rulespec-compile", "compile", str(input_file), "--rust"],
        ):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "pub fn calculate" in output

    def test_compile_effective_date_resolves_temporal_entries(self, tmp_path):
        """Compile can resolve temporal RuleSpec definitions with --effective-date."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: Test
  versions:
  - effective_from: '2024-01-01'
    formula: '0.2'
  - effective_from: '2025-01-01'
    formula: '0.25'
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--effective-date",
                "2025-06-01",
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "0.25" in output

    def test_compile_parameter_binding_supplies_source_only_parameter(self, tmp_path):
        """Compile can bind source-only parameters from the CLI."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: external/rate
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--binding",
                "rate=0.25",
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "0.25" in output

    def test_compile_binding_supplies_source_only_external_rule(self, tmp_path):
        """Compile accepts the rule-oriented --binding flag."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: external/rate
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--binding",
                "rate=0.35",
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "0.35" in output

    def test_compile_binding_file_supplies_source_only_rule(self, tmp_path):
        """Compile can bind source-only rules from a JSON bundle file."""
        input_file = tmp_path / "test.yaml"
        binding_file = tmp_path / "bindings.json"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: external/rate
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate
"""
        )
        binding_file.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "bindings": [{"symbol": "rate", "value": 0.4}],
                }
            )
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--binding-file",
                str(binding_file),
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "0.4" in output

    def test_compile_binding_file_supports_structured_rule_bundle(self, tmp_path):
        """Compile accepts the structured --binding-file bundle format."""
        input_file = tmp_path / "test.yaml"
        binding_file = tmp_path / "bindings.json"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: external/rate
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate
"""
        )
        binding_file.write_text(
            json.dumps(
                {
                    "bindings": [
                        {
                            "symbol": "rate",
                            "effective_date": "2025-01-01",
                            "value": 0.45,
                        }
                    ]
                }
            )
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--effective-date",
                "2025-01-01",
                "--binding-file",
                str(binding_file),
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "0.45" in output

    def test_compile_binding_file_supports_indexed_structured_yaml(self, tmp_path):
        """Compile can load indexed structured YAML through --binding-file."""
        base_amounts = tmp_path / "statutes" / "26" / "32" / "b" / "2" / "A"
        base_amounts.mkdir(parents=True)
        input_file = base_amounts / "base_amounts.yaml"
        binding_file = tmp_path / "eitc-2024.yaml"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: number_of_qualifying_children
  kind: input
  entity: TaxUnit
  period: Year
  dtype: Integer
  default: 0
- name: earned_income_amount
  kind: parameter
  source: external/rulespec-us
- name: phaseout_amount
  kind: parameter
  source: external/rulespec-us
- name: eitc_pair_total
  kind: derived
  entity: TaxUnit
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: |-
      earned = earned_income_amount[number_of_qualifying_children]
      phaseout = phaseout_amount[number_of_qualifying_children]
      return earned + phaseout
"""
        )
        binding_file.write_text(
            """
schema_version: 1
bindings:
- symbol: earned_income_amount
  source: Rev. Proc. 2023-34
  values:
    0: 10330.0
    1: 12390.0
- symbol: phaseout_amount
  source: Rev. Proc. 2023-34
  values:
    0: 16480.0
    1: 22720.0
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--effective-date",
                "2024-06-01",
                "--binding-file",
                str(binding_file),
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "12390.0" in output
                assert "22720.0" in output

    def test_compile_repeated_binding_files_merge(self, tmp_path):
        """Repeated binding files merge into one external-rule resolver."""
        input_file = tmp_path / "tax.yaml"
        scalar_bundle = tmp_path / "scalar.json"
        allowance_bundle = tmp_path / "allowance.yaml"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: external/rate
- name: allowance
  kind: parameter
  source: external/allowance
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate + allowance
"""
        )
        scalar_bundle.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "bindings": [{"symbol": "rate", "value": 0.2}],
                }
            )
        )
        allowance_bundle.write_text(
            """
schema_version: 1
bindings:
- symbol: allowance
  source: Allowance memo
  value: 5.0
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--effective-date",
                "2024-06-01",
                "--binding-file",
                str(scalar_bundle),
                "--binding-file",
                str(allowance_bundle),
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "0.2" in output
                assert "5.0" in output

    def test_compile_supports_qualified_parameter_binding_for_imported_param(
        self, tmp_path
    ):
        """CLI bindings can target imported source-only params by module identity."""
        (tmp_path / "shared.yaml").write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: external/rate
"""
        )
        input_file = tmp_path / "benefit_amount.yaml"
        input_file.write_text(
            """
format: rulespec/v1
imports:
- ./shared.yaml
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--binding",
                "shared.rate=0.25",
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "0.25" in output

    def test_compile_structured_binding_file_supplies_metadata(self, tmp_path):
        """Compile accepts structured rule-binding metadata."""
        input_file = tmp_path / "test.yaml"
        binding_file = tmp_path / "bindings.json"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: external/rate
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate
"""
        )
        binding_file.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "metadata": {"name": "TY2025 bundle"},
                    "bindings": [
                        {
                            "symbol": "rate",
                            "value": 0.4,
                            "source": "bundle://ty2025",
                        }
                    ],
                }
            )
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--binding-file",
                str(binding_file),
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "external/rate [bound from bundle://ty2025]" in output

    def test_compile_inline_binding_overrides_binding_file(self, tmp_path):
        """Inline rule bindings override file-backed values."""
        input_file = tmp_path / "test.yaml"
        binding_file = tmp_path / "bindings.json"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: external/rate
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate
"""
        )
        binding_file.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "bindings": [{"symbol": "rate", "value": 0.2}],
                }
            )
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--binding-file",
                str(binding_file),
                "--binding",
                "rate=0.33",
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "0.33" in output

    def test_compile_ambiguous_bare_parameter_binding_exits_1(self, tmp_path):
        """Ambiguous bare parameter bindings fail with a user-facing error."""
        (tmp_path / "left.yaml").write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: left-rate
"""
        )
        (tmp_path / "right.yaml").write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: right-rate
"""
        )
        input_file = tmp_path / "benefit_amount.yaml"
        input_file.write_text(
            """
format: rulespec/v1
imports:
- ./left.yaml
- ./right.yaml
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--binding",
                "rate=0.25",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_compile_supports_dotted_module_identity_parameter_binding(self, tmp_path):
        """Qualified parameter bindings accept dotted leaf identities."""
        (tmp_path / "shared.v1.yaml").write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: shared-rate
"""
        )
        input_file = tmp_path / "benefit_amount.yaml"
        input_file.write_text(
            """
format: rulespec/v1
imports:
- ./shared.v1.yaml
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--binding",
                "shared.v1.rate=0.25",
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert '"param": "shared_v1_rate"' in output
                assert '"module_identity": "shared.v1"' in output

    def test_compile_missing_source_only_parameter_binding_exits_1(self, tmp_path):
        """Referenced source-only parameters must be bound explicitly."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: external/rate
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate
"""
        )
        with patch("sys.argv", ["rulespec-compile", "compile", str(input_file)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_compile_if_else_formula_to_python(self, tmp_path):
        """Limited if/else formulas compile through the CLI."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
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
        )
        with patch(
            "sys.argv",
            ["rulespec-compile", "compile", str(input_file), "--python"],
        ):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "if is_joint:" in output
                assert "return wages * rate" in output

    def test_compile_select_output_prunes_return_shape(self, tmp_path):
        """CLI output selection returns only the requested variable."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: Test
  versions:
  - effective_from: '2024-01-01'
    formula: '0.1'
- name: taxable_income
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages - deduction
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return taxable_income * rate
- name: bonus
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * 0.5
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--select-output",
                "tax",
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                namespace = {}
                exec(mock_print.call_args_list[0][0][0], namespace)
                result = namespace["calculate"](wages=1000, deduction=100)
                assert result["tax"] == 90
                assert "taxable_income" not in result
                assert "bonus" not in result

    def test_compile_resolves_local_file_imports(self, tmp_path):
        """CLI compile loads local imported RuleSpec files before code generation."""
        shared = tmp_path / "shared.yaml"
        shared.write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: shared-rate
  versions:
  - effective_from: '2024-01-01'
    formula: '0.1'
- name: taxable_income
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages - deduction
"""
        )
        input_file = tmp_path / "main.yaml"
        input_file.write_text(
            """
format: rulespec/v1
imports:
- ./shared.yaml
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return taxable_income * rate
"""
        )
        with patch(
            "sys.argv",
            ["rulespec-compile", "compile", str(input_file), "--python"],
        ):
            with patch("builtins.print") as mock_print:
                main()
                namespace = {}
                exec(mock_print.call_args_list[0][0][0], namespace)
                result = namespace["calculate"](wages=1000, deduction=100)
                assert result["tax"] == 90

    def test_compile_resolves_aliased_imports(self, tmp_path):
        """CLI compile supports module-qualified references through import aliases."""
        (tmp_path / "left.yaml").write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: left-rate
  versions:
  - effective_from: '2024-01-01'
    formula: '0.1'
"""
        )
        (tmp_path / "right.yaml").write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: right-rate
  versions:
  - effective_from: '2024-01-01'
    formula: '0.2'
"""
        )
        input_file = tmp_path / "main.yaml"
        input_file.write_text(
            """
format: rulespec/v1
imports:
- path: ./left.yaml
  alias: left
- path: ./right.yaml
  alias: right
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * left.rate + wages * right.rate
"""
        )
        with patch(
            "sys.argv",
            ["rulespec-compile", "compile", str(input_file), "--python"],
        ):
            with patch("builtins.print") as mock_print:
                main()
                namespace = {}
                exec(mock_print.call_args_list[0][0][0], namespace)
                assert namespace["calculate"](wages=100)["tax"] == 30

    def test_compile_supports_selective_imports_from_explicit_exports(self, tmp_path):
        """CLI compile binds selected exported names without a whole-module import."""
        (tmp_path / "shared.yaml").write_text(
            """
format: rulespec/v1
exports:
- rate_public
- taxable_income
rules:
- name: rate_public
  kind: parameter
  source: shared-rate
  versions:
  - effective_from: '2024-01-01'
    formula: '0.1'
- name: hidden_rate
  kind: parameter
  source: hidden-rate
  versions:
  - effective_from: '2024-01-01'
    formula: '0.2'
- name: taxable_income
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages - deduction
"""
        )
        input_file = tmp_path / "main.yaml"
        input_file.write_text(
            """
format: rulespec/v1
imports:
- path: ./shared.yaml
  symbols:
  - name: rate_public
    alias: rate
  - taxable_income
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return taxable_income * rate
"""
        )
        with patch(
            "sys.argv",
            ["rulespec-compile", "compile", str(input_file), "--python"],
        ):
            with patch("builtins.print") as mock_print:
                main()
                namespace = {}
                exec(mock_print.call_args_list[0][0][0], namespace)
                assert namespace["calculate"](wages=1000, deduction=100)["tax"] == 90

    def test_compile_export_aliases_define_public_output_names(self, tmp_path):
        """CLI compile returns aliased public outputs instead of internal names."""
        input_file = tmp_path / "main.yaml"
        input_file.write_text(
            """
format: rulespec/v1
exports:
- name: tax
  alias: benefit_amount
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * 0.1
"""
        )
        with patch(
            "sys.argv",
            ["rulespec-compile", "compile", str(input_file), "--python"],
        ):
            with patch("builtins.print") as mock_print:
                main()
                namespace = {}
                exec(mock_print.call_args_list[0][0][0], namespace)
                result = namespace["calculate"](wages=100)
                assert result["benefit_amount"] == 10
                assert "tax" not in result

    def test_compile_select_output_uses_public_export_alias(self, tmp_path):
        """CLI selected outputs follow the exported public interface."""
        input_file = tmp_path / "main.yaml"
        input_file.write_text(
            """
format: rulespec/v1
exports:
- name: tax
  alias: benefit_amount
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * 0.1
- name: bonus
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * 0.5
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--select-output",
                "benefit_amount",
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                namespace = {}
                exec(mock_print.call_args_list[0][0][0], namespace)
                assert namespace["calculate"](wages=100) == {
                    "benefit_amount": 10,
                    "citations": [],
                }

    def test_compile_supports_module_re_exports(self, tmp_path):
        """CLI compile resolves re-exported symbols through intermediate modules."""
        (tmp_path / "base.yaml").write_text(
            """
format: rulespec/v1
exports:
- name: private_rate
  alias: rate
rules:
- name: private_rate
  kind: parameter
  source: base-rate
  versions:
  - effective_from: '2024-01-01'
    formula: '0.1'
"""
        )
        (tmp_path / "surface.yaml").write_text(
            """
format: rulespec/v1
re_exports:
- path: ./base.yaml
  symbols:
  - rate
"""
        )
        input_file = tmp_path / "main.yaml"
        input_file.write_text(
            """
format: rulespec/v1
imports:
- path: ./surface.yaml
  symbols:
  - rate
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate
"""
        )
        with patch(
            "sys.argv",
            ["rulespec-compile", "compile", str(input_file), "--python"],
        ):
            with patch("builtins.print") as mock_print:
                main()
                namespace = {}
                exec(mock_print.call_args_list[0][0][0], namespace)
                assert namespace["calculate"](wages=100)["tax"] == 10

    def test_compile_supports_module_roots_for_bare_imports(self, tmp_path):
        """CLI compile resolves bare imports through repeated --module-root flags."""
        shared = tmp_path / "lib" / "tax" / "shared.yaml"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_text(
            """
format: rulespec/v1
exports:
- name: private_rate
  alias: rate
rules:
- name: private_rate
  kind: parameter
  source: base-rate
  versions:
  - effective_from: '2024-01-01'
    formula: '0.1'
"""
        )
        input_file = tmp_path / "main.yaml"
        input_file.write_text(
            """
format: rulespec/v1
imports:
- path: tax/shared.yaml
  symbols:
  - rate
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--module-root",
                str(tmp_path / "lib"),
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                namespace = {}
                exec(mock_print.call_args_list[0][0][0], namespace)
                assert namespace["calculate"](wages=100)["tax"] == 10

    def test_compile_supports_cli_package_aliases(self, tmp_path):
        """CLI compile resolves package-prefixed imports through --package."""
        shared = tmp_path / "packages" / "tax" / "shared.yaml"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_text(
            """
format: rulespec/v1
exports:
- name: private_rate
  alias: rate
rules:
- name: private_rate
  kind: parameter
  source: base-rate
  versions:
  - effective_from: '2024-01-01'
    formula: '0.1'
"""
        )
        input_file = tmp_path / "main.yaml"
        input_file.write_text(
            """
format: rulespec/v1
imports:
- path: tax/shared.yaml
  symbols:
  - rate
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--python",
                "--package",
                f"tax={tmp_path / 'packages' / 'tax'}",
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                namespace = {}
                exec(mock_print.call_args_list[0][0][0], namespace)
                assert namespace["calculate"](wages=100)["tax"] == 10

    def test_compile_malformed_rulespec_toml_exits_1(self, tmp_path):
        """Malformed rulespec.toml config still surfaces as a normal CLI error."""
        (tmp_path / "rulespec.toml").write_text(
            '[module_resolution\nroots = ["./lib"]\n'
        )
        input_file = tmp_path / "main.yaml"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * 0.1
"""
        )
        with patch("sys.argv", ["rulespec-compile", "compile", str(input_file)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_compile_unknown_selected_output_exits_1(self, tmp_path):
        """Selecting a missing output variable surfaces a user-facing error."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * 0.1
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--select-output",
                "bonus",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_compile_unsupported_construct_exits_1(self, tmp_path):
        """Unsupported generic compilation surfaces a user-facing error."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
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
      while wages > 0:
        return wages
"""
        )
        with patch("sys.argv", ["rulespec-compile", "compile", str(input_file)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


class TestCLIMainLower:
    """Test lower command branches."""

    def test_lower_to_stdout_emits_pruned_lowered_json(self, tmp_path):
        """lower emits a serializable selected-output bundle to stdout."""
        input_file = tmp_path / "policy.yaml"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: Test
  versions:
  - effective_from: '2024-01-01'
    formula: '0.1'
- name: taxable_income
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages - deduction
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return taxable_income * rate
- name: bonus
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * 0.5
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "lower",
                str(input_file),
                "--select-output",
                "tax",
            ],
        ):
            with patch("builtins.print") as mock_print:
                main()
                payload = json.loads(mock_print.call_args_list[0][0][0])

        assert [output["name"] for output in payload["outputs"]] == ["tax"]
        assert [computation["name"] for computation in payload["computations"]] == [
            "taxable_income",
            "tax",
        ]

    def test_compile_missing_return_path_exits_1(self, tmp_path):
        """Control flow without a total return still fails loudly."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
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
      if wages > 0:
        return wages
"""
        )
        with patch("sys.argv", ["rulespec-compile", "compile", str(input_file)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_compile_unsupported_function_call_exits_1(self, tmp_path):
        """Unknown helper calls fail loudly instead of generating bad code."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return custom_credit(wages)
"""
        )
        with patch("sys.argv", ["rulespec-compile", "compile", str(input_file)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_compile_invalid_effective_date_exits_2(self, tmp_path):
        """Invalid effective dates are rejected by argparse."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return 1
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--effective-date",
                "2025-99-99",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_compile_invalid_rule_binding_exits_2(self, tmp_path):
        """Invalid rule binding syntax is rejected by argparse."""
        input_file = tmp_path / "test.yaml"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return 1
"""
        )
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--binding",
                "rate[bad]=x",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_compile_invalid_binding_file_exits_1(self, tmp_path):
        """Invalid binding files surface a user-facing error."""
        input_file = tmp_path / "test.yaml"
        binding_file = tmp_path / "bindings.json"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return 1
"""
        )
        binding_file.write_text("{bad json")
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--binding-file",
                str(binding_file),
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_compile_malformed_binding_file_exits_1(self, tmp_path):
        """Malformed binding files surface a user-facing error."""
        input_file = tmp_path / "test.yaml"
        binding_file = tmp_path / "bindings.json"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: external/rate
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate
"""
        )
        binding_file.write_text(json.dumps({"rate": {"schema_version": 1}}))
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--binding-file",
                str(binding_file),
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_compile_malformed_binding_list_file_exits_1(self, tmp_path):
        """Malformed list binding payloads surface a user-facing error."""
        input_file = tmp_path / "test.yaml"
        binding_file = tmp_path / "bindings.json"
        input_file.write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: external/rate
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * rate
"""
        )
        binding_file.write_text(json.dumps({"rate": [1, "x"]}))
        with patch(
            "sys.argv",
            [
                "rulespec-compile",
                "compile",
                str(input_file),
                "--binding-file",
                str(binding_file),
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


class TestCLIMainHarness:
    """Test harness command branches."""

    def test_harness_to_stdout(self):
        """harness outputs a human-readable scorecard."""
        with patch("sys.argv", ["rulespec-compile", "harness"]):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "Compiler harness score:" in output

    def test_harness_json_to_stdout(self):
        """harness --json outputs machine-readable summary JSON."""
        with patch(
            "sys.argv",
            ["rulespec-compile", "harness", "--json", "--case", "basic_straight_line"],
        ):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert '"score": "1/1"' in output

    def test_harness_include_external_flag_is_forwarded(self):
        """harness --include-external forwards the opt-in external flag."""
        summary = HarnessSummary(
            total=1,
            passed=1,
            failed=0,
            skipped=0,
            by_category={
                "policyengine": {"total": 1, "passed": 1, "failed": 0, "skipped": 0}
            },
            results=[
                HarnessResult(
                    case="policyengine_snap_example",
                    category="policyengine",
                    passed=True,
                    status="passed",
                    detail="Compiled SNAP stays within tolerance.",
                )
            ],
        )
        with patch(
            "src.rulespec_compile.cli.run_compiler_harness",
            return_value=summary,
        ) as mock_run:
            with patch(
                "sys.argv", ["rulespec-compile", "harness", "--include-external"]
            ):
                with patch("builtins.print"):
                    main()
        mock_run.assert_called_once_with(
            case_names=None,
            include_external=True,
            include_live=False,
        )

    def test_harness_include_live_flag_is_forwarded(self):
        """harness --include-live forwards the opt-in live flag."""
        summary = HarnessSummary(
            total=1,
            passed=0,
            failed=1,
            skipped=0,
            by_category={
                "live_stack": {"total": 1, "passed": 0, "failed": 1, "skipped": 0}
            },
            results=[
                HarnessResult(
                    case="live_rulespec_us_citation_identity",
                    category="live_stack",
                    passed=False,
                    status="failed",
                    detail="Contract gap.",
                )
            ],
        )
        with patch(
            "src.rulespec_compile.cli.run_compiler_harness",
            return_value=summary,
        ) as mock_run:
            with patch("sys.argv", ["rulespec-compile", "harness", "--include-live"]):
                with pytest.raises(SystemExit) as exc_info:
                    with patch("builtins.print"):
                        main()
                assert exc_info.value.code == 1
        mock_run.assert_called_once_with(
            case_names=None,
            include_external=False,
            include_live=True,
        )

    def test_harness_unknown_case_exits_1(self):
        """Unknown harness case names are rejected."""
        with patch(
            "sys.argv",
            ["rulespec-compile", "harness", "--case", "does_not_exist"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_harness_skipped_case_exits_1(self):
        """Skipped harness runs fail the CLI gate."""
        summary = HarnessSummary(
            total=1,
            passed=0,
            failed=0,
            skipped=1,
            by_category={"core": {"total": 1, "passed": 0, "failed": 0, "skipped": 1}},
            results=[
                HarnessResult(
                    case="basic_straight_line",
                    category="core",
                    passed=False,
                    status="skipped",
                    detail="Node.js is not available.",
                )
            ],
        )
        with patch(
            "src.rulespec_compile.cli.run_compiler_harness", return_value=summary
        ):
            with patch("sys.argv", ["rulespec-compile", "harness"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1


class TestCLIMainEitc:
    """Test eitc command branches."""

    def test_eitc_js_to_stdout(self):
        """eitc command outputs JS to stdout by default."""
        with patch("sys.argv", ["rulespec-compile", "eitc"]):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "function calculate(" in output

    def test_eitc_js_to_file(self, tmp_path):
        """eitc command writes JS to file with -o."""
        output_file = tmp_path / "eitc.js"
        with patch("sys.argv", ["rulespec-compile", "eitc", "-o", str(output_file)]):
            main()
            assert output_file.exists()
            content = output_file.read_text()
            assert "function calculate(" in content

    def test_eitc_python_to_stdout(self):
        """eitc --python outputs Python to stdout."""
        with patch("sys.argv", ["rulespec-compile", "eitc", "--python"]):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "def calculate(" in output

    def test_eitc_python_to_file(self, tmp_path):
        """eitc --python -o writes Python to file."""
        output_file = tmp_path / "eitc.py"
        with patch(
            "sys.argv",
            ["rulespec-compile", "eitc", "--python", "-o", str(output_file)],
        ):
            main()
            assert output_file.exists()
            content = output_file.read_text()
            assert "def calculate(" in content

    def test_eitc_custom_year(self):
        """eitc --year 2024 passes correct year."""
        with patch("sys.argv", ["rulespec-compile", "eitc", "--year", "2024"]):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args_list[0][0][0]
                assert "2024" in output


class TestCLIMainNoCommand:
    """Test no-command case."""

    def test_no_command_exits_1(self):
        """No command prints help and exits 1."""
        with patch("sys.argv", ["rulespec-compile"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
