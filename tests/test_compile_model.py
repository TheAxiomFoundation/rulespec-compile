"""Tests for the shared generic compile model."""

import json
from datetime import date

import pytest

from src.rulespec_compile.compile_model import CompilationError, LoweredProgram
from src.rulespec_compile.parser import parse_rulespec
from src.rulespec_compile.program import load_rulespec_program
from src.rulespec_compile.rule_bindings import RuleBindingError


class TestCompiledModule:
    """Tests for generic parsed-file compilation."""

    def test_extracts_inputs_from_formula_references(self):
        """Free references become explicit calculator inputs."""
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
      base_income = wages + tips
      return round(base_income * rate)
"""
        module = parse_rulespec(rulespec).to_compile_model()

        assert [compiled_input.name for compiled_input in module.inputs] == [
            "wages",
            "tips",
        ]
        assert [parameter.name for parameter in module.parameters] == ["rate"]
        assert [variable.name for variable in module.variables] == ["tax"]

    def test_no_formula_variables_become_declared_inputs(self):
        """No-formula rules participate as typed inputs instead of computations."""
        rulespec = """
format: rulespec/v1
rules:
- name: is_us_citizen_national_or_resident
  kind: input
  entity: Person
  period: Year
  dtype: Boolean
  default: false
- name: ctc_meets_citizenship_requirement
  kind: derived
  entity: Person
  period: Year
  dtype: Boolean
  versions:
  - effective_from: '1998-01-01'
    formula: is_us_citizen_national_or_resident
"""
        module = parse_rulespec(rulespec).to_compile_model()
        lowered = parse_rulespec(rulespec).to_lowered_program()

        assert [compiled_input.name for compiled_input in module.inputs] == [
            "is_us_citizen_national_or_resident"
        ]
        assert module.inputs[0].default is False
        assert [variable.name for variable in module.variables] == [
            "ctc_meets_citizenship_requirement"
        ]
        assert [compiled_input.name for compiled_input in lowered.inputs] == [
            "is_us_citizen_national_or_resident"
        ]
        assert [compiled_input.public_name for compiled_input in lowered.inputs] == [
            "is_us_citizen_national_or_resident"
        ]
        assert lowered.inputs[0].value_kind == "boolean"

    def test_imported_declared_inputs_lower_with_qualified_public_names(self, tmp_path):
        """Imported free inputs keep rule-identity public names in lowered bundles."""
        shared = tmp_path / "statutes" / "shared" / "rate.yaml"
        shared.parent.mkdir(parents=True)
        shared.write_text(
            """
format: rulespec/v1
rules:
- name: wages
  kind: input
  entity: Person
  period: Year
  dtype: Money
- name: taxable_amount
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: wages * 2
"""
        )
        entry = tmp_path / "statutes" / "shared" / "benefit.yaml"
        entry.write_text(
            """
format: rulespec/v1
imports:
- path: statutes/shared/rate
  symbols:
  - taxable_amount
rules:
- name: benefit
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: taxable_amount + base_amount
"""
        )

        lowered = load_rulespec_program(entry).to_lowered_program()

        assert {
            compiled_input.public_name or compiled_input.name
            for compiled_input in lowered.inputs
        } == {
            "statutes/shared/rate.wages",
            "base_amount",
        }

    def test_no_formula_variables_without_explicit_defaults_use_typed_fallbacks(self):
        """Declared inputs without defaults fall back from dtype, not heuristics."""
        rulespec = """
format: rulespec/v1
rules:
- name: full_time_student_months
  kind: input
  entity: Person
  period: Year
  dtype: Integer
- name: is_full_time_student
  kind: derived
  entity: Person
  period: Year
  dtype: Boolean
  versions:
  - effective_from: '2002-01-01'
    formula: full_time_student_months >= 5
"""
        module = parse_rulespec(rulespec).to_compile_model()

        assert [compiled_input.name for compiled_input in module.inputs] == [
            "full_time_student_months"
        ]
        assert module.inputs[0].default == 0
        assert module.inputs[0].python_type == "int"

    def test_scalar_external_rule_with_metadata_compiles_as_parameter(self):
        """Entity-less scalar rules with metadata stay external values, not inputs."""
        rulespec = """
format: rulespec/v1
rules:
- name: phase_in_rate
  kind: parameter
  dtype: Rate
  unit: rate
  source: 26 USC 32(b)(1)
  versions:
  - effective_from: '2024-01-01'
    formula: '0.25'
- name: benefit_amount
  kind: derived
  entity: TaxUnit
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: wages * phase_in_rate
"""
        module = parse_rulespec(rulespec).to_compile_model()

        assert [compiled_input.name for compiled_input in module.inputs] == ["wages"]
        assert [parameter.name for parameter in module.parameters] == ["phase_in_rate"]
        assert [variable.name for variable in module.variables] == ["benefit_amount"]

    def test_scalar_computed_rule_without_entity_compiles_as_output(self):
        """Entity-less computed live-style rules compile as computed outputs."""
        rulespec = """
format: rulespec/v1
rules:
- name: snap_self_employment_cost_exclusion
  kind: derived
  label: SNAP self-employment cost exclusion
  description: Reduction for production costs
  versions:
  - effective_from: '2008-10-01'
    formula: |-
      min(
        snap_nonfarm_self_employment_production_costs,
        snap_nonfarm_self_employment_gross_income,
      ) + snap_farm_self_employment_production_costs
"""
        lowered = parse_rulespec(rulespec).to_lowered_program()

        assert [compiled_input.name for compiled_input in lowered.inputs] == [
            "snap_nonfarm_self_employment_production_costs",
            "snap_nonfarm_self_employment_gross_income",
            "snap_farm_self_employment_production_costs",
        ]
        assert [output.name for output in lowered.outputs] == [
            "snap_self_employment_cost_exclusion"
        ]

    def test_lowered_program_round_trips_and_generates_python(self):
        """Compiled modules can emit and reload a lowered JSON bundle."""
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
        payload = json.loads(lowered.to_json())
        namespace = {}

        lowered_round_trip = LoweredProgram.from_json(lowered.to_json())
        code = lowered_round_trip.to_python_generator().generate()
        exec(code, namespace)

        assert [output["name"] for output in payload["outputs"]] == ["tax"]
        assert payload["computations"][0]["statements"][0]["kind"] == "assign"
        assert namespace["calculate"](wages=1000, deduction=100)["tax"] == 180

    def test_lowered_program_emits_typed_computations_and_outputs(self):
        """Lowered bundles expose explicit value kinds for computations/outputs."""
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
    formula: return wages <= 1000
"""
        payload = json.loads(parse_rulespec(rulespec).to_lowered_program().to_json())

        assert payload["computations"][0]["value_kind"] == "boolean"
        assert payload["computations"][0]["local_value_kinds"] == {}
        assert payload["outputs"][0]["value_kind"] == "boolean"

    def test_lowered_program_emits_module_identity_for_file_backed_rules(
        self, tmp_path
    ):
        """File-backed lowered bundles preserve the leaf rule identity."""
        origin = tmp_path / "benefit_amount.yaml"
        lowered = parse_rulespec(
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
""",
            origin=origin,
        ).to_lowered_program()
        payload = json.loads(lowered.to_json())

        assert payload["parameters"][0]["module_identity"] == "benefit_amount"
        assert payload["computations"][0]["module_identity"] == "benefit_amount"
        assert payload["outputs"][0]["module_identity"] == "benefit_amount"

    def test_lowered_program_emits_typed_parameters(self):
        """Lowered bundles expose explicit value kinds for resolved parameters."""
        rulespec = """
format: rulespec/v1
rules:
- name: allowance
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
    formula: return n_children + allowance
"""
        payload = json.loads(parse_rulespec(rulespec).to_lowered_program().to_json())

        assert payload["parameters"][0]["value_kind"] == "integer"
        assert payload["parameters"][0]["lookup_kind"] == "scalar"
        assert payload["parameters"][0]["index_value_kind"] is None

    def test_lowered_program_emits_indexed_parameter_contracts(self):
        """Lowered bundles expose explicit lookup contracts for indexed parameters."""
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
        payload = json.loads(
            parse_rulespec(rulespec)
            .to_lowered_program(rule_bindings={"allowances": [1, 2]})
            .to_json()
        )

        assert payload["parameters"][0]["lookup_kind"] == "indexed"
        assert payload["parameters"][0]["index_value_kind"] == "integer"

    def test_lowered_program_emits_typed_local_slots(self):
        """Lowered bundles expose stable local slot kinds inside computations."""
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
        payload = json.loads(parse_rulespec(rulespec).to_lowered_program().to_json())

        assert payload["computations"][0]["local_value_kinds"] == {
            "eligible": "boolean"
        }

    def test_lowered_program_requires_input_value_kinds(self):
        """Lowered JSON inputs must carry explicit value kinds."""
        with pytest.raises(
            CompilationError, match="missing required field 'value_kind'"
        ):
            LoweredProgram.from_json(
                json.dumps(
                    {
                        "inputs": [{"name": "n_children", "default": 0}],
                        "parameters": [],
                        "computations": [],
                        "outputs": [],
                    }
                )
            )

    def test_lowered_program_loads_explicit_output_module_identity(self):
        """Lowered JSON preserves explicit output module identity."""
        lowered = LoweredProgram.from_json(
            json.dumps(
                {
                    "inputs": [],
                    "parameters": [],
                    "computations": [
                        {
                            "name": "benefit_amount_tax",
                            "module_identity": "benefit_amount",
                            "statements": [
                                {
                                    "kind": "return",
                                    "expression": {"kind": "literal", "value": 10},
                                }
                            ],
                        }
                    ],
                    "outputs": [
                        {
                            "name": "benefit_amount",
                            "variable_name": "benefit_amount_tax",
                            "value_kind": "number",
                            "module_identity": "benefit_amount",
                        }
                    ],
                }
            )
        )

        assert lowered.outputs[0].module_identity == "benefit_amount"

    def test_lowered_program_rejects_input_without_value_kind(self):
        """Lowered JSON rejects inputs without explicit value_kind."""
        with pytest.raises(
            CompilationError, match="missing required field 'value_kind'"
        ):
            LoweredProgram.from_json(
                json.dumps(
                    {
                        "inputs": [{"name": "is_joint", "default": False}],
                        "parameters": [],
                        "computations": [],
                        "outputs": [],
                    }
                )
            )

    def test_lowered_program_uses_explicit_parameter_kinds(self):
        """Lowered JSON reuses explicit parameter kinds for local inference."""
        lowered = LoweredProgram.from_json(
            json.dumps(
                {
                    "inputs": [
                        {
                            "name": "index",
                            "default": 0,
                            "value_kind": "integer",
                        }
                    ],
                    "parameters": [
                        {
                            "name": "allowances",
                            "values": {"0": 1, "1": 2},
                            "value_kind": "integer",
                            "lookup_kind": "indexed",
                            "index_value_kind": "integer",
                        }
                    ],
                    "computations": [
                        {
                            "name": "count",
                            "statements": [
                                {
                                    "kind": "assign",
                                    "name": "chosen",
                                    "expression": {
                                        "kind": "subscript",
                                        "value": {
                                            "kind": "name",
                                            "name": "allowances",
                                        },
                                        "index": {
                                            "kind": "name",
                                            "name": "index",
                                        },
                                    },
                                },
                                {
                                    "kind": "return",
                                    "expression": {
                                        "kind": "name",
                                        "name": "chosen",
                                    },
                                },
                            ],
                            "local_names": ["chosen"],
                            "value_kind": "integer",
                        }
                    ],
                    "outputs": [
                        {
                            "name": "count",
                            "variable_name": "count",
                            "value_kind": "integer",
                        }
                    ],
                }
            )
        )

        assert lowered.parameters[0].value_kind == "integer"
        assert lowered.parameters[0].lookup_kind == "indexed"
        assert lowered.parameters[0].index_value_kind == "integer"
        assert lowered.computations[0].local_value_kinds == {"chosen": "integer"}

    def test_lowered_program_rejects_parameter_lookup_contract_mismatch(self):
        """Lowered bundles fail loudly when parameter contracts disagree with usage."""
        with pytest.raises(
            CompilationError,
            match="lookup_kind='scalar' but computations use it as 'indexed'",
        ):
            LoweredProgram.from_json(
                json.dumps(
                    {
                        "inputs": [
                            {
                                "name": "n_children",
                                "default": 0,
                                "value_kind": "integer",
                            }
                        ],
                        "parameters": [
                            {
                                "name": "allowances",
                                "values": {"0": 1, "1": 2},
                                "value_kind": "integer",
                                "lookup_kind": "scalar",
                                "index_value_kind": None,
                            }
                        ],
                        "computations": [
                            {
                                "name": "count",
                                "statements": [
                                    {
                                        "kind": "return",
                                        "expression": {
                                            "kind": "subscript",
                                            "value": {
                                                "kind": "name",
                                                "name": "allowances",
                                            },
                                            "index": {
                                                "kind": "name",
                                                "name": "n_children",
                                            },
                                        },
                                    }
                                ],
                                "local_names": [],
                                "value_kind": "integer",
                            }
                        ],
                        "outputs": [
                            {
                                "name": "count",
                                "variable_name": "count",
                                "value_kind": "integer",
                            }
                        ],
                    }
                )
            )

    def test_lowered_program_infers_local_kinds_from_loaded_inputs(self):
        """Lowered JSON reuses loaded input kinds when local kinds are absent."""
        lowered = LoweredProgram.from_json(
            json.dumps(
                {
                    "inputs": [
                        {
                            "name": "joint",
                            "default": False,
                            "value_kind": "boolean",
                        }
                    ],
                    "parameters": [],
                    "computations": [
                        {
                            "name": "flag",
                            "statements": [
                                {
                                    "kind": "assign",
                                    "name": "tmp",
                                    "expression": {"kind": "name", "name": "joint"},
                                },
                                {
                                    "kind": "return",
                                    "expression": {"kind": "name", "name": "tmp"},
                                },
                            ],
                            "local_names": ["tmp"],
                            "value_kind": "boolean",
                        }
                    ],
                    "outputs": [
                        {
                            "name": "flag",
                            "variable_name": "flag",
                            "value_kind": "boolean",
                        }
                    ],
                }
            )
        )

        assert lowered.computations[0].local_value_kinds == {"tmp": "boolean"}

    def test_lowered_program_rejects_incomplete_local_value_kind_map(self):
        """Typed lowered bundles must define a kind for every declared local."""
        with pytest.raises(
            CompilationError,
            match="missing local value kinds for: tmp",
        ):
            LoweredProgram.from_json(
                json.dumps(
                    {
                        "inputs": [],
                        "parameters": [],
                        "computations": [
                            {
                                "name": "flag",
                                "statements": [
                                    {
                                        "kind": "assign",
                                        "name": "tmp",
                                        "expression": {
                                            "kind": "literal",
                                            "value": True,
                                        },
                                    },
                                    {
                                        "kind": "return",
                                        "expression": {"kind": "name", "name": "tmp"},
                                    },
                                ],
                                "local_names": ["tmp"],
                                "local_value_kinds": {},
                                "value_kind": "boolean",
                            }
                        ],
                        "outputs": [
                            {
                                "name": "flag",
                                "variable_name": "flag",
                                "value_kind": "boolean",
                            }
                        ],
                    }
                )
            )

    def test_incompatible_branch_local_kinds_fail_loudly(self):
        """Locals assigned incompatible kinds across branches are unsupported."""
        rulespec = """
format: rulespec/v1
rules:
- name: result
  kind: derived
  entity: Person
  period: Year
  versions:
  - effective_from: '2024-01-01'
    formula: |-
      if is_toggle:
        tmp = is_joint
      else:
        tmp = 1
      return tmp
"""
        with pytest.raises(
            CompilationError,
            match="Local 'tmp' is assigned incompatible value kinds",
        ):
            parse_rulespec(rulespec).to_compile_model()

    def test_lowered_program_rejects_malformed_nested_json(self):
        """Malformed lowered bundles fail with CompilationError, not AttributeError."""
        with pytest.raises(
            CompilationError,
            match="Lowered statement must be an object",
        ):
            LoweredProgram.from_json(
                json.dumps(
                    {
                        "inputs": [],
                        "parameters": [],
                        "computations": [
                            {
                                "name": "tax",
                                "statements": ["not-an-object"],
                            }
                        ],
                        "outputs": [],
                    }
                )
            )

    def test_lowered_program_rejects_missing_nested_fields(self):
        """Malformed lowered bundles report missing nested fields explicitly."""
        with pytest.raises(
            CompilationError,
            match="Lowered assign statement is missing required field 'name'",
        ):
            LoweredProgram.from_json(
                json.dumps(
                    {
                        "inputs": [],
                        "parameters": [],
                        "computations": [
                            {
                                "name": "tax",
                                "statements": [
                                    {
                                        "kind": "assign",
                                        "expression": {
                                            "kind": "literal",
                                            "value": 1,
                                        },
                                    }
                                ],
                            }
                        ],
                        "outputs": [],
                    }
                )
            )

    def test_source_only_parameter_requires_binding(self):
        """Source-only referenced parameters fail without explicit bindings."""
        rulespec = """
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
        with pytest.raises(CompilationError, match="Supply a rule binding"):
            parse_rulespec(rulespec).to_compile_model()

    def test_unused_source_only_parameter_does_not_fail_compile(self):
        """Unused source-only parameters stay lazy until something references them."""
        rulespec = """
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
    formula: return wages * 0.1
"""
        module = parse_rulespec(rulespec).to_compile_model()

        assert [parameter.name for parameter in module.parameters] == []
        assert [variable.name for variable in module.variables] == ["tax"]

    def test_source_only_parameter_uses_explicit_binding(self):
        """Source-only referenced parameters compile with bound values."""
        rulespec = """
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
        namespace = {}
        code = (
            parse_rulespec(rulespec)
            .to_python_generator(rule_bindings={"rate": 0.25})
            .generate()
        )

        exec(code, namespace)

        assert namespace["calculate"](wages=100)["tax"] == 25

    def test_source_only_parameter_accepts_qualified_binding_name(self, tmp_path):
        """Single-file source-only params can bind through module_identity.symbol."""
        origin = tmp_path / "benefit_amount.yaml"
        rulespec = """
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
        namespace = {}
        code = (
            parse_rulespec(rulespec, origin=origin)
            .to_python_generator(rule_bindings={"benefit_amount.rate": 0.25})
            .generate()
        )

        exec(code, namespace)

        assert namespace["calculate"](wages=100)["tax"] == 25

    def test_source_only_parameter_accepts_effective_dated_rule_bindings(self):
        """Structured rule bindings resolve by compile effective date."""
        rulespec = """
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
        namespace = {}
        code = (
            parse_rulespec(rulespec)
            .to_python_generator(
                effective_date=date(2025, 1, 1),
                rule_bindings={
                    "bindings": [
                        {
                            "symbol": "rate",
                            "effective_date": "2024-01-01",
                            "value": 0.2,
                        },
                        {
                            "symbol": "rate",
                            "effective_date": "2025-01-01",
                            "value": 0.3,
                        },
                    ]
                },
            )
            .generate()
        )

        exec(code, namespace)

        assert namespace["calculate"](wages=100)["tax"] == 30

    def test_source_only_parameter_rejects_dated_rule_bindings_without_effective_date(
        self,
    ):
        """Date-specific rule bindings require an explicit compile date."""
        rulespec = """
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
        with pytest.raises(
            RuleBindingError,
            match="has only effective-dated bindings",
        ):
            parse_rulespec(rulespec).to_compile_model(
                rule_bindings={
                    "bindings": [
                        {
                            "symbol": "rate",
                            "effective_date": "2025-01-01",
                            "value": 0.3,
                        }
                    ]
                }
            )

    def test_scalar_parameter_reference_rejects_indexed_values(self):
        """Bare parameter references fail when the resolved parameter is indexed."""
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
    formula: return allowances
"""
        with pytest.raises(
            CompilationError,
            match="used as a scalar value but resolves to indexed entries",
        ):
            parse_rulespec(rulespec).to_compile_model(
                rule_bindings={"allowances": [1, 2]}
            )

    def test_parameter_cannot_be_used_as_scalar_and_indexed(self):
        """Parameters must not mix scalar and indexed access patterns."""
        rulespec = """
format: rulespec/v1
rules:
- name: rates
  kind: parameter
  source: external/rates
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: |-
      chosen = rates[n_children]
      return chosen + rates
"""
        with pytest.raises(
            CompilationError,
            match="used both as a scalar value and an indexed lookup",
        ):
            parse_rulespec(rulespec).to_compile_model(rule_bindings={"rates": [1, 2]})

    def test_structured_parameter_binding_carries_bundle_source(self):
        """Structured bindings surface bundle source metadata in generated output."""
        rulespec = """
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
        code = (
            parse_rulespec(rulespec)
            .to_python_generator(
                rule_bindings={"rate": {"value": 0.25, "source": "bundle://ty2025"}}
            )
            .generate()
        )

        assert "external/rate [bound from bundle://ty2025]" in code

    def test_python_compile_executes_multiline_formulas(self):
        """Parsed RuleSpec formulas compile to executable Python."""
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
      taxable_income = max(0, wages - 1000)
      applied_rate = is_joint ? rate / 2 : rate
      return round(taxable_income * applied_rate)
"""
        generator = parse_rulespec(rulespec).to_python_generator()
        code = generator.generate()
        namespace = {}

        exec(code, namespace)

        assert namespace["calculate"](wages=6000, is_joint=False)["tax"] == 1000
        assert namespace["calculate"](wages=6000, is_joint=True)["tax"] == 500

    def test_expression_ir_handles_boolean_ternary_and_indexed_parameters(self):
        """Generic compile supports the validated scalar expression subset."""
        rulespec = """
format: rulespec/v1
rules:
- name: rates
  kind: parameter
  source: external/rates
- name: threshold
  kind: parameter
  source: Test
  versions:
  - effective_from: '2024-01-01'
    formula: '100'
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: |-
      rate_index = has_bonus ? 1 : 0
      return !(wages <= threshold) && is_eligible ? round(wages * rates[rate_index]) : 0
"""
        js_code = (
            parse_rulespec(rulespec)
            .to_js_generator(rule_bindings={"rates": [0.1, 0.2]})
            .generate()
        )
        namespace = {}
        py_code = (
            parse_rulespec(rulespec)
            .to_python_generator(rule_bindings={"rates": [0.1, 0.2]})
            .generate()
        )

        exec(py_code, namespace)

        assert "PARAMS.rates[rate_index]" in js_code
        assert (
            namespace["calculate"](
                wages=200,
                has_bonus=True,
                is_eligible=True,
            )["tax"]
            == 40
        )
        assert (
            namespace["calculate"](
                wages=50,
                has_bonus=True,
                is_eligible=True,
            )["tax"]
            == 0
        )

    def test_simple_comparison_expression_compiles_to_python(self):
        """Direct comparison expressions stay scalar expressions in Python output."""
        rulespec = """
format: rulespec/v1
rules:
- name: threshold
  kind: parameter
  source: Test
  versions:
  - effective_from: '2024-01-01'
    formula: '1000'
- name: flag
  kind: derived
  entity: Person
  period: Year
  dtype: Bool
  versions:
  - effective_from: '2024-01-01'
    formula: return wages <= threshold
"""
        namespace = {}
        py_code = parse_rulespec(rulespec).to_python_generator().generate()

        exec(py_code, namespace)

        assert namespace["calculate"](wages=500)["flag"] is True
        assert namespace["calculate"](wages=1500)["flag"] is False

    def test_terminal_bare_expression_compiles_as_return(self):
        """A final bare expression compiles as an implicit return."""
        rulespec = """
format: rulespec/v1
rules:
- name: result
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: |-
      tmp = wages + 1
      tmp
"""
        namespace = {}
        py_code = parse_rulespec(rulespec).to_python_generator().generate()

        exec(py_code, namespace)

        assert namespace["calculate"](wages=10)["result"] == 11

    def test_supports_if_else_blocks_with_branch_local_assignments(self):
        """Generic compile supports limited if/else blocks with shared locals."""
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
        module = parse_rulespec(rulespec).to_compile_model()
        js_code = module.to_js_generator().generate()
        namespace = {}

        exec(module.to_python_generator().generate(), namespace)

        assert "let rate;" in js_code
        assert "if (is_joint) {" in js_code
        assert namespace["calculate"](wages=100, is_joint=True)["tax"] == 10
        assert namespace["calculate"](wages=100, is_joint=False)["tax"] == 20

    def test_select_outputs_prunes_to_reachable_subgraph(self):
        """Selected outputs keep only reachable variables, params, and inputs."""
        rulespec = """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: Test
  versions:
  - effective_from: '2024-01-01'
    formula: '0.1'
- name: bonus_rate
  kind: parameter
  source: Test
  versions:
  - effective_from: '2024-01-01'
    formula: '0.5'
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
    formula: return wages * bonus_rate
"""
        module = parse_rulespec(rulespec).to_compile_model(outputs=["tax"])

        assert module.outputs == ["tax"]
        assert [variable.name for variable in module.variables] == [
            "taxable_income",
            "tax",
        ]
        assert [parameter.name for parameter in module.parameters] == ["rate"]
        assert [compiled_input.name for compiled_input in module.inputs] == [
            "wages",
            "deduction",
        ]

    def test_select_outputs_ignores_unreachable_unsupported_variables(self):
        """Selected-output compile only validates the reachable variable subgraph."""
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
    formula: return wages * 0.1
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
        namespace = {}
        py_code = (
            parse_rulespec(rulespec).to_python_generator(outputs=["tax"]).generate()
        )

        exec(py_code, namespace)

        assert namespace["calculate"](wages=100)["tax"] == 10

    def test_selected_outputs_only_return_requested_values(self):
        """Selected outputs stay internal unless explicitly requested."""
        rulespec = """
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
"""
        namespace = {}
        code = parse_rulespec(rulespec).to_python_generator(outputs=["tax"]).generate()

        exec(code, namespace)

        result = namespace["calculate"](wages=1000, deduction=100)
        assert result["tax"] == 90
        assert "taxable_income" not in result

    def test_export_aliases_define_public_outputs_for_single_file_compile(self):
        """Single-file compile exposes exported output aliases in the result shape."""
        rulespec = """
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
        namespace = {}
        py_code = parse_rulespec(rulespec).to_python_generator().generate()

        exec(py_code, namespace)

        result = namespace["calculate"](wages=100)
        assert result["benefit_amount"] == 10
        assert "tax" not in result

    def test_export_aliases_restrict_selected_outputs_to_public_names(self):
        """Explicit exports make selected-output compile use the public interface."""
        rulespec = """
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
        with pytest.raises(
            CompilationError,
            match="Unknown exported output variable\\(s\\): tax",
        ):
            parse_rulespec(rulespec).to_compile_model(outputs=["tax"])

    def test_invalid_export_name_fails_even_when_other_exports_are_valid(self):
        """Mixed valid/invalid exports still fail loudly instead of dropping one."""
        rulespec = """
format: rulespec/v1
exports:
- tax
- missing_output
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
        with pytest.raises(
            CompilationError,
            match="File exports unknown symbol 'missing_output'",
        ):
            parse_rulespec(rulespec).to_compile_model()

    def test_unknown_selected_output_fails_loudly(self):
        """Selecting a missing output variable raises a user-facing error."""
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
    formula: return wages * 0.1
"""
        with pytest.raises(CompilationError, match="Unknown output variable"):
            parse_rulespec(rulespec).to_compile_model(outputs=["bonus"])

    def test_reorders_variables_by_dependency(self):
        """Variables can depend on earlier-or-later parsed variables."""
        rulespec = """
format: rulespec/v1
rules:
- name: rate
  kind: parameter
  source: Test
  versions:
  - effective_from: '2024-01-01'
    formula: '0.1'
- name: tax
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return taxable_income * rate
- name: taxable_income
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: return wages - deduction
"""
        code = parse_rulespec(rulespec).to_python_generator().generate()
        namespace = {}

        exec(code, namespace)

        result = namespace["calculate"](wages=10000, deduction=1500)
        assert result["taxable_income"] == 8500
        assert result["tax"] == 850

    def test_rejects_multiple_temporal_parameter_values(self):
        """Generic compile fails loudly on unsupported temporal parameters."""
        rulespec = """
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
        with pytest.raises(CompilationError, match="Pass an effective date"):
            parse_rulespec(rulespec).to_compile_model()

    def test_resolves_temporal_parameter_by_effective_date(self):
        """Temporal parameters resolve to the active entry for a compile date."""
        rulespec = """
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
        namespace = {}
        code = (
            parse_rulespec(rulespec)
            .to_python_generator(effective_date="2025-06-01")
            .generate()
        )

        exec(code, namespace)

        assert namespace["calculate"](wages=100)["tax"] == 25

    def test_resolves_temporal_variable_formula_by_effective_date(self):
        """Temporal variable formulas resolve against the compile date."""
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
    formula: return wages * 0.1
  - effective_from: '2025-01-01'
    formula: return wages * 0.2
"""
        namespace = {}
        code = (
            parse_rulespec(rulespec)
            .to_python_generator(effective_date="2024-06-01")
            .generate()
        )

        exec(code, namespace)

        assert namespace["calculate"](wages=100)["tax"] == 10

    def test_errors_when_effective_date_precedes_temporal_entries(self):
        """Temporal compile fails when no entry is active yet."""
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
    formula: return wages * 0.1
  - effective_from: '2025-01-01'
    formula: return wages * 0.2
"""
        with pytest.raises(CompilationError, match="has no temporal entry active"):
            parse_rulespec(rulespec).to_compile_model(effective_date="2023-01-01")

    def test_rejects_unsupported_function_calls(self):
        """Generic compile fails loudly on unknown helper calls."""
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
    formula: return custom_credit(wages)
"""
        with pytest.raises(
            CompilationError,
            match="calls unsupported function 'custom_credit'",
        ):
            parse_rulespec(rulespec).to_compile_model()

    def test_rejects_attribute_access(self):
        """Generic compile fails loudly on attribute access."""
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
    formula: return household.size
"""
        with pytest.raises(
            CompilationError,
            match="Attribute access is not supported",
        ):
            parse_rulespec(rulespec).to_compile_model()

    def test_rejects_missing_return_path(self):
        """Generic compile rejects formulas that do not return on all paths."""
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
      if wages > 0:
        return wages
"""
        with pytest.raises(
            CompilationError,
            match="does not return a value on all reachable paths",
        ):
            parse_rulespec(rulespec).to_compile_model()

    def test_rejects_unsupported_loop_statements(self):
        """Generic compile still fails loudly on unsupported loop control flow."""
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
      while wages > 0:
        return wages
"""
        with pytest.raises(
            CompilationError,
            match="unsupported statement 'while'",
        ):
            parse_rulespec(rulespec).to_compile_model()
