"""Tests for multi-file RuleSpec program loading and compilation."""

import json
from pathlib import Path

import pytest

from src.rulespec_compile.compile_model import CompilationError
from src.rulespec_compile.program import load_rulespec_program
from src.rulespec_compile.rule_bindings import RuleBindingError


class TestRuleSpecProgram:
    """Tests for file-graph loading and compilation."""

    def test_working_families_example_graph_compiles_and_runs(self):
        """The shipped multi-file example compiles with qualified bindings."""
        entry = (
            Path(__file__).parent.parent
            / "examples"
            / "working_families"
            / "benefit_amount.yaml"
        )

        namespace = {}
        code = (
            load_rulespec_program(entry)
            .to_python_generator(
                rule_bindings={"phase_in_rate.rate": 0.25},
                outputs=["benefit_amount"],
            )
            .generate()
        )

        exec(code, namespace)

        result = namespace["calculate"](
            earned_income=4000,
            has_qualifying_child=True,
        )
        assert result["benefit_amount"] == 1000

        result = namespace["calculate"](
            earned_income=4000,
            has_qualifying_child=False,
        )
        assert result["benefit_amount"] == 0

    def test_working_families_example_lowering_preserves_module_identities(self):
        """The shipped multi-file example lowers with real file identities intact."""
        entry = (
            Path(__file__).parent.parent
            / "examples"
            / "working_families"
            / "benefit_amount.yaml"
        )

        lowered = load_rulespec_program(entry).to_lowered_program(
            rule_bindings={"phase_in_rate.rate": 0.25},
            outputs=["benefit_amount"],
        )

        assert [output.name for output in lowered.outputs] == ["benefit_amount"]
        assert {parameter.module_identity for parameter in lowered.parameters} == {
            "phase_in_cap",
            "phase_in_rate",
        }

    def test_load_rulespec_program_defaults_to_computed_outputs_only(self, tmp_path):
        """Default public outputs should not include free inputs from source files."""
        entry = tmp_path / "snap_child_support_deduction.yaml"
        entry.write_text(
            """
format: rulespec/v1
rules:
- name: snap_child_support_payments_made
  kind: input
  entity: SnapUnit
  period: Month
  dtype: Money
- name: snap_state_uses_child_support_deduction
  kind: input
  entity: SnapUnit
  period: Month
  dtype: Boolean
- name: snap_child_support_deduction
  kind: derived
  entity: SnapUnit
  period: Month
  dtype: Money
  versions:
  - effective_from: '2022-01-01'
    formula: |-
      if snap_state_uses_child_support_deduction:
        snap_child_support_payments_made
      else:
        0
"""
        )

        program = load_rulespec_program(entry)
        lowered = program.to_lowered_program()

        assert program.default_outputs == ["snap_child_support_deduction"]
        assert [output.name for output in lowered.outputs] == [
            "snap_child_support_deduction"
        ]
        assert {compiled_input.name for compiled_input in lowered.inputs} == {
            "snap_child_support_payments_made",
            "snap_state_uses_child_support_deduction",
        }

    def test_load_rulespec_program_supports_inline_rulespec_conditional_expressions(
        self, tmp_path
    ):
        """Inline `if cond: a else: b` RuleSpec expressions compile without rewrites."""
        entry = tmp_path / "snap_child_support_deduction.yaml"
        entry.write_text(
            """
format: rulespec/v1
rules:
- name: snap_child_support_payments_made
  kind: input
  entity: SnapUnit
  period: Month
  dtype: Money
- name: snap_state_uses_child_support_deduction
  kind: input
  entity: SnapUnit
  period: Month
  dtype: Boolean
- name: snap_child_support_deduction
  kind: derived
  entity: SnapUnit
  period: Month
  dtype: Money
  versions:
  - effective_from: '2022-01-01'
    formula: |-
      if snap_state_uses_child_support_deduction:
        snap_child_support_payments_made
      else:
        0
"""
        )

        namespace = {}
        code = load_rulespec_program(entry).to_python_generator().generate()

        exec(code, namespace)

        assert (
            namespace["calculate"](
                snap_child_support_payments_made=500,
                snap_state_uses_child_support_deduction=True,
            )["snap_child_support_deduction"]
            == 500
        )
        assert (
            namespace["calculate"](
                snap_child_support_payments_made=500,
                snap_state_uses_child_support_deduction=False,
            )["snap_child_support_deduction"]
            == 0
        )

    def test_load_rulespec_program_supports_chained_inline_rulespec_conditionals(
        self, tmp_path
    ):
        """Chained RuleSpec conditionals collapse into one expression."""
        entry = tmp_path / "need_standard.yaml"
        entry.write_text(
            """
format: rulespec/v1
rules:
- name: number_of_children_in_assistance_unit
  kind: input
  entity: TanfUnit
  period: Month
  dtype: Integer
- name: need_standard
  kind: derived
  entity: TanfUnit
  period: Month
  dtype: Money
  versions:
  - effective_from: '2026-04-02'
    formula: |-
      if number_of_children_in_assistance_unit == 0: 0 else:
      if number_of_children_in_assistance_unit == 1: 117 else:
      if number_of_children_in_assistance_unit == 2: 245 else:
      999
"""
        )

        namespace = {}
        code = load_rulespec_program(entry).to_python_generator().generate()

        exec(code, namespace)

        assert (
            namespace["calculate"](number_of_children_in_assistance_unit=0)[
                "need_standard"
            ]
            == 0
        )
        assert (
            namespace["calculate"](number_of_children_in_assistance_unit=2)[
                "need_standard"
            ]
            == 245
        )
        assert (
            namespace["calculate"](number_of_children_in_assistance_unit=3)[
                "need_standard"
            ]
            == 999
        )

    def test_load_rulespec_program_supports_multiline_expression_continuations(
        self, tmp_path
    ):
        """Bare expression lines can continue across multiple lines at one indent."""
        entry = tmp_path / "flag.yaml"
        entry.write_text(
            """
format: rulespec/v1
rules:
- name: a
  kind: input
  entity: Person
  period: Month
  dtype: Boolean
- name: b
  kind: input
  entity: Person
  period: Month
  dtype: Boolean
- name: flag
  kind: derived
  entity: Person
  period: Month
  dtype: Boolean
  versions:
  - effective_from: '2026-01-01'
    formula: |-
      a and
      b
"""
        )

        namespace = {}
        code = load_rulespec_program(entry).to_python_generator().generate()

        exec(code, namespace)

        assert namespace["calculate"](a=True, b=True)["flag"] is True
        assert namespace["calculate"](a=True, b=False)["flag"] is False

    def test_load_rulespec_program_supports_inline_if_elif_else_statements(
        self, tmp_path
    ):
        """Single-line `if` / `elif` / `else` branches normalize into real blocks."""
        entry = tmp_path / "phaseout_percentage.yaml"
        entry.write_text(
            """
format: rulespec/v1
rules:
- name: qualifying_child_count
  kind: input
  entity: TaxUnit
  period: Year
  dtype: Integer
- name: phaseout_pct_no_children
  kind: parameter
  versions:
  - effective_from: '2009-01-01'
    formula: '0.0765'
- name: phaseout_pct_1_child
  kind: parameter
  versions:
  - effective_from: '2009-01-01'
    formula: '0.1598'
- name: phaseout_pct_2_children
  kind: parameter
  versions:
  - effective_from: '2009-01-01'
    formula: '0.2106'
- name: phaseout_pct_3_plus_children
  kind: parameter
  versions:
  - effective_from: '2009-01-01'
    formula: '0.2106'
- name: phaseout_percentage
  kind: derived
  entity: TaxUnit
  period: Year
  dtype: Rate
  versions:
  - effective_from: '2009-01-01'
    formula: |-
      if qualifying_child_count >= 3: phaseout_pct_3_plus_children
      elif qualifying_child_count == 2: phaseout_pct_2_children
      elif qualifying_child_count == 1: phaseout_pct_1_child
      else: phaseout_pct_no_children
"""
        )

        namespace = {}
        code = load_rulespec_program(entry).to_python_generator().generate()

        exec(code, namespace)

        assert (
            namespace["calculate"](qualifying_child_count=0)["phaseout_percentage"]
            == 0.0765
        )
        assert (
            namespace["calculate"](qualifying_child_count=2)["phaseout_percentage"]
            == 0.2106
        )
        assert (
            namespace["calculate"](qualifying_child_count=4)["phaseout_percentage"]
            == 0.2106
        )

    def test_load_rulespec_program_compiles_cross_file_dependencies(self, tmp_path):
        """Entry files can compile imported helper variables and parameters."""
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
        entry = tmp_path / "main.yaml"
        entry.write_text(
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

        namespace = {}
        code = load_rulespec_program(entry).to_python_generator().generate()

        exec(code, namespace)

        result = namespace["calculate"](wages=1000, deduction=100)
        assert result["tax"] == 90
        assert "taxable_income" not in result

    def test_load_rulespec_program_resolves_spec_style_variable_imports(self, tmp_path):
        """Per-variable `imports:` blocks resolve through statutes-root paths."""
        statute_root = tmp_path / "statutes" / "26" / "62"
        statute_root.mkdir(parents=True)
        (statute_root / "62.yaml").write_text(
            """
format: rulespec/v1
rules:
- name: adjusted_gross_income
  kind: input
  entity: TaxUnit
  period: Year
  dtype: Money
"""
        )
        target_root = tmp_path / "statutes" / "26" / "21" / "a" / "2"
        target_root.mkdir(parents=True)
        entry = target_root / "A.yaml"
        entry.write_text(
            """
format: rulespec/v1
rules:
- name: first_reduction
  kind: derived
  entity: TaxUnit
  period: Year
  dtype: Rate
  imports:
  - path: 26/62
    symbols:
    - adjusted_gross_income
  versions:
  - effective_from: '2002-01-01'
    formula: adjusted_gross_income >= 15000
"""
        )

        lowered = load_rulespec_program(entry).to_lowered_program(
            outputs=["first_reduction"]
        )

        assert [compiled_input.name for compiled_input in lowered.inputs] == [
            "statutes_26_62_62_adjusted_gross_income"
        ]
        assert lowered.outputs[0].module_identity == "statutes/26/21/a/2/A"

    def test_load_rulespec_program_resolves_root_qualified_top_level_imports(
        self, tmp_path
    ):
        """Top-level `imports:` blocks can target root-qualified citation paths."""
        dependency_root = tmp_path / "statutes" / "crs" / "26-2-703"
        dependency_root.mkdir(parents=True)
        (dependency_root / "12.yaml").write_text(
            """
format: rulespec/v1
rules:
- name: contract_is_entered_into_by_participant_and_county_department
  kind: input
  entity: Person
  period: Month
  dtype: Boolean
- name: contract_is_pursuant_to_section_26_2_708
  kind: input
  entity: Person
  period: Month
  dtype: Boolean
- name: is_individual_responsibility_contract
  kind: derived
  entity: Person
  period: Month
  dtype: Boolean
  versions:
  - effective_from: '2026-04-03'
    formula: |-
      contract_is_entered_into_by_participant_and_county_department and
      contract_is_pursuant_to_section_26_2_708
"""
        )
        entry_root = tmp_path / "statutes" / "crs" / "26-2-711" / "1" / "a"
        entry_root.mkdir(parents=True)
        entry = entry_root / "I.yaml"
        entry.write_text(
            """
format: rulespec/v1
imports:
- path: statutes/crs/26-2-703/12
  symbols:
  - is_individual_responsibility_contract
rules:
- name: participant_fails_to_comply_with_terms_and_conditions_of_contract
  kind: input
  entity: Person
  period: Month
  dtype: Boolean
- name: good_cause_exists_as_determined_by_county
  kind: input
  entity: Person
  period: Month
  dtype: Boolean
- name: participant_is_sanctioned
  kind: derived
  entity: Person
  period: Month
  dtype: Boolean
  versions:
  - effective_from: '2026-04-03'
    formula: |-
      participant_fails_to_comply_with_terms_and_conditions_of_contract and
      is_individual_responsibility_contract and
      not good_cause_exists_as_determined_by_county
"""
        )

        lowered = load_rulespec_program(entry).to_lowered_program(
            outputs=["participant_is_sanctioned"]
        )

        assert {
            compiled_input.public_name or compiled_input.name
            for compiled_input in lowered.inputs
        } == {
            "participant_fails_to_comply_with_terms_and_conditions_of_contract",
            "good_cause_exists_as_determined_by_county",
            (
                "statutes/crs/26-2-703/12."
                "contract_is_entered_into_by_participant_and_county_department"
            ),
            "statutes/crs/26-2-703/12.contract_is_pursuant_to_section_26_2_708",
        }
        assert "statutes/crs/26-2-703/12.is_individual_responsibility_contract" not in {
            compiled_input.public_name or compiled_input.name
            for compiled_input in lowered.inputs
        }
        assert lowered.outputs[0].module_identity == "statutes/crs/26-2-711/1/a/I"

        namespace: dict[str, object] = {}
        code = (
            load_rulespec_program(entry)
            .to_python_generator(outputs=["participant_is_sanctioned"])
            .generate()
        )

        exec(code, namespace)

        result = namespace["calculate"](
            **{
                (
                    "statutes/crs/26-2-703/12."
                    "contract_is_entered_into_by_participant_and_county_department"
                ): True,
                (
                    "statutes/crs/26-2-703/12.contract_is_pursuant_to_section_26_2_708"
                ): True,
                (
                    "participant_fails_to_comply_with_terms_and_conditions_of_contract"
                ): True,
                "good_cause_exists_as_determined_by_county": False,
            }
        )

        assert result["participant_is_sanctioned"] is True

    def test_selected_outputs_prune_unreachable_imported_variables(self, tmp_path):
        """Graph pruning excludes unreachable imported variables before validation."""
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
- name: bonus
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
        entry = tmp_path / "main.yaml"
        entry.write_text(
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

        namespace = {}
        code = (
            load_rulespec_program(entry).to_python_generator(outputs=["tax"]).generate()
        )

        exec(code, namespace)

        assert namespace["calculate"](wages=1000, deduction=100)["tax"] == 90

    def test_load_rulespec_program_rejects_missing_import(self, tmp_path):
        """Missing imported files fail with a user-facing error."""
        entry = tmp_path / "main.yaml"
        entry.write_text("""
format: rulespec/v1
imports:
- ./missing.yaml
""")

        with pytest.raises(CompilationError, match="was not found"):
            load_rulespec_program(entry)

    def test_load_rulespec_program_rejects_non_rulespec_entry_file(self, tmp_path):
        """Entrypoints must use the .yaml extension."""
        entry = tmp_path / "main.txt"
        entry.write_text("""
format: rulespec/v1
rules:
- name: tax
  kind: input
  entity: Person
  period: Year
  dtype: Money
""")

        with pytest.raises(CompilationError, match="must use the \\.yaml extension"):
            load_rulespec_program(entry)

    def test_load_rulespec_program_rejects_non_rulespec_imports(self, tmp_path):
        """Imported files must also use the .yaml extension."""
        (tmp_path / "shared.txt").write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: shared-rate
  versions:
  - effective_from: '2024-01-01'
    formula: '0.1'
"""
        )
        entry = tmp_path / "main.yaml"
        entry.write_text("""
format: rulespec/v1
imports:
- ./shared.txt
""")

        with pytest.raises(CompilationError, match="must use the \\.yaml extension"):
            load_rulespec_program(entry)

    def test_load_rulespec_program_rejects_import_cycles(self, tmp_path):
        """Import cycles fail loudly instead of recursing forever."""
        (tmp_path / "a.yaml").write_text("""
format: rulespec/v1
imports:
- ./b.yaml
""")
        (tmp_path / "b.yaml").write_text("""
format: rulespec/v1
imports:
- ./a.yaml
""")

        with pytest.raises(CompilationError, match="Import cycle detected"):
            load_rulespec_program(tmp_path / "a.yaml")

    def test_load_rulespec_program_rejects_duplicate_symbols(self, tmp_path):
        """Plain imports still reject ambiguous duplicate symbol exposure."""
        (tmp_path / "left.yaml").write_text(
            """
format: rulespec/v1
rules:
- name: shared
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return 1
"""
        )
        (tmp_path / "right.yaml").write_text(
            """
format: rulespec/v1
rules:
- name: shared
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return 2
"""
        )
        (tmp_path / "main.yaml").write_text(
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
    formula: return shared
"""
        )

        with pytest.raises(CompilationError, match="Plain import scope collision"):
            load_rulespec_program(tmp_path / "main.yaml").to_compile_model()

    def test_load_rulespec_program_supports_aliased_duplicate_symbols(self, tmp_path):
        """Aliased imports can expose duplicate names without global collisions."""
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
        (tmp_path / "main.yaml").write_text(
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

        namespace = {}
        code = (
            load_rulespec_program(tmp_path / "main.yaml")
            .to_python_generator()
            .generate()
        )

        exec(code, namespace)

        assert namespace["calculate"](wages=100)["tax"] == 30

    def test_load_rulespec_program_rejects_duplicate_leaf_module_identities(
        self, tmp_path
    ):
        """Programs fail loudly when two files share the same subsection leaf."""
        left = tmp_path / "left"
        right = tmp_path / "right"
        left.mkdir()
        right.mkdir()
        (left / "shared.yaml").write_text(
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
        (right / "shared.yaml").write_text(
            """
format: rulespec/v1
rules:
- name: bonus
  kind: parameter
  source: right-bonus
  versions:
  - effective_from: '2024-01-01'
    formula: '2'
"""
        )
        entry = tmp_path / "benefit_amount.yaml"
        entry.write_text(
            """
format: rulespec/v1
imports:
- ./left/shared.yaml
- ./right/shared.yaml
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

        with pytest.raises(CompilationError) as exc_info:
            load_rulespec_program(entry).to_compile_model()
        message = str(exc_info.value)
        assert "Module identity collision" in message
        assert "'shared'" in message
        assert str(left / "shared.yaml") in message
        assert str(right / "shared.yaml") in message
        assert "rename" in message.lower()

    def test_load_rulespec_program_rejects_normalized_module_identity_collision(
        self, tmp_path
    ):
        """Fail loudly when two distinct identities normalize to the same prefix."""
        statute_dir = tmp_path / "statutes" / "us"
        statute_dir.mkdir(parents=True)
        (statute_dir / "bar-baz.yaml").write_text(
            """
format: rulespec/v1
rules:
- name: rate_a
  kind: parameter
  source: rate-a
  versions:
  - effective_from: '2024-01-01'
    formula: '0.1'
"""
        )
        (statute_dir / "bar_baz.yaml").write_text(
            """
format: rulespec/v1
rules:
- name: rate_b
  kind: parameter
  source: rate-b
  versions:
  - effective_from: '2024-01-01'
    formula: '0.2'
"""
        )
        entry = tmp_path / "statutes" / "us" / "entry.yaml"
        entry.write_text(
            """
format: rulespec/v1
imports:
- ./bar-baz.yaml
- ./bar_baz.yaml
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

        with pytest.raises(CompilationError) as exc_info:
            load_rulespec_program(entry).to_compile_model()
        message = str(exc_info.value)
        assert "Module identity collision after normalization" in message
        # Both distinct identities should be called out.
        assert "bar-baz" in message
        assert "bar_baz" in message
        # Both file paths should be included as well.
        assert str(statute_dir / "bar-baz.yaml") in message
        assert str(statute_dir / "bar_baz.yaml") in message
        # The internal symbol prefix that collided should be shown.
        assert "statutes_us_bar_baz" in message
        # The remediation suggestion should be present.
        assert "rename" in message.lower()

    def test_load_rulespec_program_lowered_bundle_preserves_module_identity(
        self, tmp_path
    ):
        """Lowered program metadata keeps leaf-derived rule identity per node."""
        (tmp_path / "shared.yaml").write_text(
            """
format: rulespec/v1
source:
  citation: 26 USC shared
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
  source: 26 USC shared
  versions:
  - effective_from: '2024-01-01'
    formula: return wages - deduction
"""
        )
        entry = tmp_path / "benefit_amount.yaml"
        entry.write_text(
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

        payload = json.loads(
            load_rulespec_program(entry).to_lowered_program().to_json()
        )

        assert [parameter["name"] for parameter in payload["parameters"]] == [
            "shared_rate"
        ]
        assert payload["parameters"][0]["module_identity"] == "shared"
        assert {
            computation["name"]: computation["module_identity"]
            for computation in payload["computations"]
        } == {
            "shared_taxable_income": "shared",
            "tax": "benefit_amount",
        }
        assert payload["outputs"] == [
            {
                "name": "tax",
                "variable_name": "tax",
                "value_kind": "number",
                "module_identity": "benefit_amount",
            }
        ]

    def test_load_rulespec_program_respects_explicit_exports_and_selective_imports(
        self, tmp_path
    ):
        """Selective imports can only bind symbols that a module exports."""
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
        (tmp_path / "main.yaml").write_text(
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

        namespace = {}
        code = (
            load_rulespec_program(tmp_path / "main.yaml")
            .to_python_generator()
            .generate()
        )

        exec(code, namespace)

        assert namespace["calculate"](wages=1000, deduction=100)["tax"] == 90

    def test_load_rulespec_program_resolves_qualified_rule_bindings(self, tmp_path):
        """Imported source-only parameters bind through module_identity.symbol."""
        (tmp_path / "shared.yaml").write_text(
            """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: external/rate
"""
        )
        entry = tmp_path / "benefit_amount.yaml"
        entry.write_text(
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

        namespace = {}
        code = (
            load_rulespec_program(entry)
            .to_python_generator(rule_bindings={"shared.rate": 0.25})
            .generate()
        )

        exec(code, namespace)

        assert namespace["calculate"](wages=100)["tax"] == 25

    def test_load_rulespec_program_rejects_ambiguous_bare_rule_bindings(self, tmp_path):
        """Bare source-only names fail when more than one module defines them."""
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
        entry = tmp_path / "benefit_amount.yaml"
        entry.write_text(
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
    formula: return wages
"""
        )

        with pytest.raises(
            RuleBindingError,
            match="Rule binding target 'rate' is ambiguous",
        ):
            load_rulespec_program(entry).to_compile_model(rule_bindings={"rate": 0.25})

    def test_load_rulespec_program_rejects_selective_import_of_hidden_symbol(
        self, tmp_path
    ):
        """Selective imports fail loudly when the target file does not export a name."""
        (tmp_path / "shared.yaml").write_text(
            """
format: rulespec/v1
exports:
- rate_public
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
"""
        )
        (tmp_path / "main.yaml").write_text(
            """
format: rulespec/v1
imports:
- path: ./shared.yaml
  symbols:
  - hidden_rate
rules:
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages * hidden_rate
"""
        )

        with pytest.raises(CompilationError, match="non-exported symbol 'hidden_rate'"):
            load_rulespec_program(tmp_path / "main.yaml").to_compile_model()

    def test_load_rulespec_program_supports_export_aliases_for_imports_and_outputs(
        self, tmp_path
    ):
        """Export aliases define both import names and public result keys."""
        (tmp_path / "shared.yaml").write_text(
            """
format: rulespec/v1
exports:
- name: private_rate
  alias: rate
rules:
- name: private_rate
  kind: parameter
  source: shared-rate
  versions:
  - effective_from: '2024-01-01'
    formula: '0.1'
"""
        )
        entry = tmp_path / "main.yaml"
        entry.write_text(
            """
format: rulespec/v1
imports:
- path: ./shared.yaml
  symbols:
  - rate
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
    formula: return wages * rate
"""
        )

        namespace = {}
        code = load_rulespec_program(entry).to_python_generator().generate()

        exec(code, namespace)

        result = namespace["calculate"](wages=100)
        assert result["benefit_amount"] == 10
        assert "tax" not in result

    def test_load_rulespec_program_select_output_uses_public_export_aliases(
        self, tmp_path
    ):
        """Selected outputs follow the public export surface, not internal names."""
        entry = tmp_path / "main.yaml"
        entry.write_text(
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

        namespace = {}
        code = (
            load_rulespec_program(entry)
            .to_python_generator(outputs=["benefit_amount"])
            .generate()
        )

        exec(code, namespace)

        result = namespace["calculate"](wages=100)
        assert result == {"benefit_amount": 10, "citations": []}

        with pytest.raises(
            CompilationError,
            match="Unknown exported output variable\\(s\\): tax",
        ):
            load_rulespec_program(entry).to_compile_model(outputs=["tax"])

    def test_load_rulespec_program_supports_module_re_exports(self, tmp_path):
        """Intermediate modules can re-export imported symbols without wrappers."""
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
        entry = tmp_path / "main.yaml"
        entry.write_text(
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

        namespace = {}
        code = load_rulespec_program(entry).to_python_generator().generate()

        exec(code, namespace)

        assert namespace["calculate"](wages=100)["tax"] == 10

    def test_load_rulespec_program_supports_entry_re_exported_public_outputs(
        self, tmp_path
    ):
        """Entry modules can publish imported outputs through re-exports."""
        (tmp_path / "upstream.yaml").write_text(
            """
format: rulespec/v1
exports:
- name: tax
  alias: upstream_benefit
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
        entry = tmp_path / "main.yaml"
        entry.write_text(
            """
format: rulespec/v1
re_exports:
- path: ./upstream.yaml
  symbols:
  - name: upstream_benefit
    alias: benefit_amount
"""
        )

        namespace = {}
        code = load_rulespec_program(entry).to_python_generator().generate()

        exec(code, namespace)

        assert namespace["calculate"](wages=100) == {
            "benefit_amount": 10,
            "citations": [],
        }

    def test_load_rulespec_program_re_exported_outputs_keep_upstream_module_identity(
        self, tmp_path
    ):
        """Public outputs exposed through re-exports preserve their source rule."""
        (tmp_path / "upstream.yaml").write_text(
            """
format: rulespec/v1
exports:
- name: tax
  alias: upstream_benefit
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
        entry = tmp_path / "benefit_amount.yaml"
        entry.write_text(
            """
format: rulespec/v1
re_exports:
- path: ./upstream.yaml
  symbols:
  - name: upstream_benefit
    alias: benefit_amount
"""
        )

        payload = json.loads(
            load_rulespec_program(entry).to_lowered_program().to_json()
        )

        assert payload["outputs"] == [
            {
                "name": "benefit_amount",
                "variable_name": "upstream_tax",
                "value_kind": "number",
                "module_identity": "upstream",
            }
        ]

    def test_load_rulespec_program_resolves_bare_imports_from_rulespec_toml(
        self, tmp_path
    ):
        """Program loading resolves bare imports via rulespec.toml roots."""
        (tmp_path / "rulespec.toml").write_text(
            """
[module_resolution]
roots = ["./lib"]
"""
        )
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
        entry = tmp_path / "main.yaml"
        entry.write_text(
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

        namespace = {}
        code = load_rulespec_program(entry).to_python_generator().generate()

        exec(code, namespace)

        assert namespace["calculate"](wages=100)["tax"] == 10

    def test_load_rulespec_program_resolves_package_alias_imports(self, tmp_path):
        """Program loading resolves package-prefixed imports through rulespec.toml."""
        (tmp_path / "rulespec.toml").write_text(
            """
[module_resolution.packages]
tax = "./packages/tax"
"""
        )
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
        entry = tmp_path / "main.yaml"
        entry.write_text(
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

        namespace = {}
        code = load_rulespec_program(entry).to_python_generator().generate()

        exec(code, namespace)

        assert namespace["calculate"](wages=100)["tax"] == 10
