"""Tests for lowered-program batch execution."""

from datetime import date

import pandas as pd

from src.rulespec_compile.batch_executor import execute_lowered_program_batch
from src.rulespec_compile.parser import parse_rulespec


class TestBatchExecutor:
    """Test batch execution over lowered programs."""

    def test_executes_indexed_parameters_and_conditionals(self):
        rulespec = parse_rulespec(
            """
format: rulespec/v1
rules:
- name: multiplier
  kind: parameter
  source: Test
  values:
    0: 1.0
    1: 2.0
    2: 3.0
- name: result
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: |-
      factor = multiplier[min(n_children, 2)]
      return is_joint ? wages * factor : 0
"""
        )
        program = rulespec.to_lowered_program(effective_date=date(2024, 1, 1))

        result = execute_lowered_program_batch(
            program,
            pd.DataFrame(
                {
                    "wages": [100, 100, 100],
                    "n_children": [0, 1, 5],
                    "is_joint": [False, True, True],
                }
            ),
        )

        assert result.to_dict(orient="list") == {"result": [0, 200, 300]}

    def test_executes_if_statement_blocks_with_branch_locals(self):
        rulespec = parse_rulespec(
            """
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
      if is_joint:
        rate = 0.1
      else:
        rate = 0.2
      return wages * rate
"""
        )
        program = rulespec.to_lowered_program(effective_date=date(2024, 1, 1))

        result = execute_lowered_program_batch(
            program,
            pd.DataFrame({"wages": [100, 100], "is_joint": [True, False]}),
        )

        assert result.to_dict(orient="list") == {"result": [10, 20]}

    def test_skips_inactive_if_branch_evaluation(self):
        rulespec = parse_rulespec(
            """
format: rulespec/v1
rules:
- name: threshold
  kind: parameter
  source: Test
  values:
    0: 100.0
- name: result
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: |-
      if is_joint:
        return wages
      else:
        return threshold[n_children]
"""
        )
        program = rulespec.to_lowered_program(effective_date=date(2024, 1, 1))

        result = execute_lowered_program_batch(
            program,
            pd.DataFrame(
                {
                    "wages": [100, 200],
                    "is_joint": [True, True],
                    "n_children": [99, 999],
                }
            ),
        )

        assert result.to_dict(orient="list") == {"result": [100, 200]}

    def test_short_circuits_boolean_and_ternary_branches(self):
        rulespec = parse_rulespec(
            """
format: rulespec/v1
rules:
- name: threshold
  kind: parameter
  source: Test
  values:
    0: 100.0
- name: boolean_result
  kind: derived
  entity: Person
  period: Year
  dtype: Bool
  versions:
  - effective_from: '2024-01-01'
    formula: return is_joint and threshold[n_children] > 0
- name: money_result
  kind: derived
  entity: Person
  period: Year
  dtype: Money
  versions:
  - effective_from: '2024-01-01'
    formula: 'return is_joint ? threshold[n_children] : wages'
"""
        )
        program = rulespec.to_lowered_program(effective_date=date(2024, 1, 1))

        result = execute_lowered_program_batch(
            program,
            pd.DataFrame(
                {
                    "wages": [100, 200],
                    "is_joint": [False, False],
                    "n_children": [99, 999],
                }
            ),
        )

        assert result.to_dict(orient="list") == {
            "boolean_result": [False, False],
            "money_result": [100, 200],
        }
