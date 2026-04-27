"""Objective compiler harness for measuring generic compile progress."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .batch_executor import execute_lowered_program_batch
from .calculators import (
    calculate_actc,
    calculate_ctc,
    calculate_eitc,
    calculate_snap_benefit,
)
from .compile_model import CompilationError
from .parameter_bindings import ParameterBindingError
from .parser import parse_rulespec
from .program import load_rulespec_program
from .rule_bindings import load_rule_bindings_file, merge_rule_bindings
from .validation import ComparisonConfig, run_policyengine_household

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _REPO_ROOT.parent


@dataclass(frozen=True)
class HarnessCase:
    """One objective compiler case."""

    name: str
    category: str
    description: str
    rulespec: str | None = None
    supporting_files: dict[str, str] = field(default_factory=dict)
    entrypoint: str = "main.yaml"
    repo_entrypoint: str | None = None
    workspace_entrypoint: str | None = None
    repo_binding_files: tuple[str, ...] = ()
    workspace_binding_files: tuple[str, ...] = ()
    targets: tuple[str, ...] = ("js", "python", "rust")
    effective_date: str | None = None
    parameter_overrides: dict[str, Any] | None = None
    outputs: list[str] | None = None
    inputs: dict[str, Any] | None = None
    expected_input_names: list[str] | None = None
    forbidden_input_names: list[str] | None = None
    expected_output_module_identities: dict[str, str] | None = None
    expected_outputs: dict[str, Any] | None = None
    batch_inputs: dict[str, list[Any]] | None = None
    expected_batch_outputs: dict[str, list[Any]] | None = None
    output_tolerances: dict[str, float] | None = None
    oracle: str | None = None
    external: bool = False
    live: bool = False
    expected_error: str | None = None


@dataclass(frozen=True)
class HarnessResult:
    """One executed harness result."""

    case: str
    category: str
    passed: bool
    status: str
    detail: str


@dataclass(frozen=True)
class HarnessSummary:
    """Harness summary with case results and aggregate counts."""

    total: int
    passed: int
    failed: int
    skipped: int
    by_category: dict[str, dict[str, int]]
    results: list[HarnessResult] = field(default_factory=list)

    @property
    def score(self) -> str:
        """Human-readable score."""
        return f"{self.passed}/{self.total}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "score": self.score,
            "by_category": self.by_category,
            "results": [
                {
                    "case": result.case,
                    "category": result.category,
                    "passed": result.passed,
                    "status": result.status,
                    "detail": result.detail,
                }
                for result in self.results
            ],
        }


HARNESS_CASES: tuple[HarnessCase, ...] = (
    HarnessCase(
        name="basic_straight_line",
        category="core",
        description="Straight-line formulas compile for all supported targets.",
        rulespec="""
rate:
  source: "Test"
  from 2024-01-01: 0.2

tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return wages * rate
""",
        inputs={"wages": 100},
        expected_outputs={"tax": 20},
    ),
    HarnessCase(
        name="comparison_expression",
        category="core",
        description="Scalar comparison expressions execute correctly.",
        rulespec="""
threshold:
  source: "Test"
  from 2024-01-01: 1000

flag:
  entity: Person
  period: Year
  dtype: Bool
  from 2024-01-01:
    return wages <= threshold
""",
        inputs={"wages": 500},
        expected_outputs={"flag": True},
    ),
    HarnessCase(
        name="implicit_return_block",
        category="core",
        description="Terminal bare expressions are treated as implicit returns.",
        rulespec="""
result:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    tmp = wages + 1
    tmp
""",
        inputs={"wages": 10},
        expected_outputs={"result": 11},
    ),
    HarnessCase(
        name="temporal_resolution",
        category="temporal",
        description="Temporal formulas resolve with an effective date.",
        rulespec="""
tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return wages * 0.1
  from 2025-01-01:
    return wages * 0.2
""",
        effective_date="2025-06-01",
        inputs={"wages": 100},
        expected_outputs={"tax": 20},
    ),
    HarnessCase(
        name="source_only_parameter_binding",
        category="bindings",
        description="Source-only parameters compile with explicit bindings.",
        rulespec="""
rate:
  source: "external/rate"

tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return wages * rate
""",
        parameter_overrides={"rate": 0.25},
        inputs={"wages": 100},
        expected_outputs={"tax": 25},
    ),
    HarnessCase(
        name="branching_formula",
        category="control_flow",
        description="If/else formulas compile and execute correctly.",
        rulespec="""
tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    if is_joint:
      rate = 0.1
    else:
      rate = 0.2
    return wages * rate
""",
        inputs={"wages": 100, "is_joint": True},
        expected_outputs={"tax": 10},
    ),
    HarnessCase(
        name="branching_batch_execution",
        category="control_flow",
        description=(
            "Batch execution handles branch-local assignments and skips dead branches."
        ),
        rulespec="""
threshold:
  source: "Test"
  values:
    0: 100

tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    if is_joint:
      rate = 0.1
      return wages * rate
    else:
      return threshold[n_children]
""",
        inputs={"wages": 100, "is_joint": True, "n_children": 0},
        expected_outputs={"tax": 10},
        batch_inputs={
            "wages": [100, 200, 300],
            "is_joint": [True, False, True],
            "n_children": [999, 0, 12345],
        },
        expected_batch_outputs={"tax": [10, 100, 30]},
    ),
    HarnessCase(
        name="selected_output_pruning",
        category="subgraph",
        description="Selected outputs prune to the reachable variable graph.",
        rulespec="""
rate:
  source: "Test"
  from 2024-01-01: 0.1

taxable_income:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return wages - deduction

tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return taxable_income * rate

bonus:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return wages * 0.5
""",
        outputs=["tax"],
        inputs={"wages": 1000, "deduction": 100},
        expected_outputs={"tax": 90},
    ),
    HarnessCase(
        name="cross_file_import_pruning",
        category="graph",
        description="Imported helpers compile through the reachable cross-file graph.",
        rulespec="""
import "./shared.yaml"

tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return taxable_income * rate
""",
        supporting_files={
            "shared.yaml": """
rate:
  source: "shared-rate"
  from 2024-01-01: 0.1

taxable_income:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return wages - deduction

bonus:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    while wages > 0:
      return wages
"""
        },
        outputs=["tax"],
        inputs={"wages": 1000, "deduction": 100},
        expected_outputs={"tax": 90},
    ),
    HarnessCase(
        name="aliased_import_namespacing",
        category="graph",
        description="Import aliases allow duplicate symbol names across modules.",
        rulespec="""
import "./left.yaml" as left
import "./right.yaml" as right

tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return wages * left.rate + wages * right.rate
""",
        supporting_files={
            "left.yaml": """
rate:
  source: "left-rate"
  from 2024-01-01: 0.1
""",
            "right.yaml": """
rate:
  source: "right-rate"
  from 2024-01-01: 0.2
""",
        },
        inputs={"wages": 100},
        expected_outputs={"tax": 30},
    ),
    HarnessCase(
        name="selective_import_exports",
        category="graph",
        description="Selective imports respect explicit module exports.",
        rulespec="""
from "./shared.yaml" import rate_public as rate, taxable_income

tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return taxable_income * rate
""",
        supporting_files={
            "shared.yaml": """
export rate_public, taxable_income

rate_public:
  source: "shared-rate"
  from 2024-01-01: 0.1

hidden_rate:
  source: "hidden-rate"
  from 2024-01-01: 0.2

taxable_income:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return wages - deduction
""",
        },
        inputs={"wages": 1000, "deduction": 100},
        expected_outputs={"tax": 90},
    ),
    HarnessCase(
        name="export_alias_public_output",
        category="graph",
        description="Export aliases define public import names and result keys.",
        rulespec="""
from "./shared.yaml" import rate
export tax as benefit_amount

tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return wages * rate
""",
        supporting_files={
            "shared.yaml": """
export private_rate as rate

private_rate:
  source: "shared-rate"
  from 2024-01-01: 0.1
"""
        },
        inputs={"wages": 100},
        expected_outputs={"benefit_amount": 10},
    ),
    HarnessCase(
        name="module_re_export_surface",
        category="graph",
        description="Modules can re-export imported symbols into a new public surface.",
        rulespec="""
export from "./upstream.yaml" import upstream_benefit as benefit_amount
""",
        supporting_files={
            "upstream.yaml": """
export tax as upstream_benefit

tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return wages * 0.1
"""
        },
        inputs={"wages": 100},
        expected_outputs={"benefit_amount": 10},
    ),
    HarnessCase(
        name="module_root_manifest_import",
        category="graph",
        description="Bare imports resolve through rulespec.toml module roots.",
        rulespec="""
from "tax/shared.yaml" import rate

tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return wages * rate
""",
        supporting_files={
            "rulespec.toml": """
[module_resolution]
roots = ["./lib"]
""",
            "lib/tax/shared.yaml": """
export private_rate as rate

private_rate:
  source: "base-rate"
  from 2024-01-01: 0.1
""",
        },
        inputs={"wages": 100},
        expected_outputs={"tax": 10},
    ),
    HarnessCase(
        name="package_alias_manifest_import",
        category="graph",
        description="Workspace package aliases resolve stable bare import prefixes.",
        rulespec="""
from "tax/shared.yaml" import rate

tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return wages * rate
""",
        supporting_files={
            "rulespec.toml": """
[module_resolution.packages]
tax = "./packages/tax"
""",
            "packages/tax/shared.yaml": """
export private_rate as rate

private_rate:
  source: "base-rate"
  from 2024-01-01: 0.1
""",
        },
        inputs={"wages": 100},
        expected_outputs={"tax": 10},
    ),
    HarnessCase(
        name="unsupported_loop_fails",
        category="unsupported",
        description="Unsupported loops fail loudly instead of compiling.",
        rulespec="""
tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    while wages > 0:
      return wages
""",
        expected_error="unsupported statement 'while'",
    ),
    HarnessCase(
        name="oracle_eitc_example",
        category="oracle",
        description="Compiled eitc.yaml matches the Python reference implementation.",
        repo_entrypoint="examples/eitc.yaml",
        inputs={
            "earned_income": 15000,
            "agi": 15000,
            "n_children": 1,
            "is_joint": False,
        },
        outputs=["eitc"],
        oracle="eitc_reference",
    ),
    HarnessCase(
        name="oracle_ctc_example",
        category="oracle",
        description="Compiled ctc.yaml matches the Python reference implementation.",
        repo_entrypoint="examples/ctc.yaml",
        inputs={
            "n_qualifying_children": 2,
            "agi": 100000,
            "is_joint": True,
            "earned_income": 80000,
        },
        outputs=["ctc", "actc"],
        oracle="ctc_reference",
    ),
    HarnessCase(
        name="oracle_snap_example",
        category="oracle",
        description="Compiled snap.yaml matches the Python reference implementation.",
        repo_entrypoint="examples/snap.yaml",
        inputs={"household_size": 4, "gross_income": 2000},
        outputs=["snap_benefit"],
        oracle="snap_reference",
    ),
    HarnessCase(
        name="policyengine_snap_example",
        category="policyengine",
        description=(
            "Compiled snap.yaml stays within the PolicyEngine SNAP tolerance "
            "on a fixed household."
        ),
        repo_entrypoint="examples/snap.yaml",
        targets=("python",),
        inputs={"household_size": 4, "gross_income": 2000, "state_code": "CA"},
        outputs=["snap_benefit"],
        output_tolerances={"snap_benefit": ComparisonConfig().snap_tolerance},
        oracle="policyengine_snap_reference",
        external=True,
    ),
    HarnessCase(
        name="live_rulespec_us_override_yaml_binding_support",
        category="live_stack",
        description=(
            "Real rules-us override YAML artifacts should bind source-backed rules "
            "through citation-path identities."
        ),
        workspace_entrypoint="rulespec-compile/examples/statute/26/32/b/2/A/base_amounts.yaml",
        workspace_binding_files=("rules-us/irs/rev-proc-2023-34/eitc-2024.yaml",),
        effective_date="2024-06-01",
        outputs=["eitc_pair_total"],
        inputs={"number_of_qualifying_children": 1},
        expected_input_names=["number_of_qualifying_children"],
        expected_output_module_identities={
            "eitc_pair_total": "statute/26/32/b/2/A/base_amounts"
        },
        expected_outputs={"eitc_pair_total": 35110},
        live=True,
    ),
    HarnessCase(
        name="live_rulespec_us_input_variable_support",
        category="live_stack",
        description=(
            "Real rules-us files should treat defaulted no-formula variables "
            "as public inputs instead of unsupported computations."
        ),
        workspace_entrypoint="rules-us/statute/26/24/c/2.yaml",
        outputs=["ctc_meets_citizenship_requirement"],
        expected_input_names=["is_us_citizen_national_or_resident"],
        expected_output_module_identities={
            "ctc_meets_citizenship_requirement": "statute/26/24/c/2"
        },
        targets=(),
        live=True,
    ),
    HarnessCase(
        name="live_rulespec_us_citation_identity",
        category="live_stack",
        description=(
            "Real rules-us files should preserve statute citation paths as module "
            "identity in the lowered bundle."
        ),
        workspace_entrypoint="rules-us/statute/26/21/a/2/A.yaml",
        outputs=["first_reduction"],
        expected_output_module_identities={"first_reduction": "statute/26/21/a/2/A"},
        targets=(),
        live=True,
    ),
    HarnessCase(
        name="live_rulespec_us_import_graph_resolution",
        category="live_stack",
        description=(
            "Real rules-us imported computed rules should resolve through "
            "the file graph instead of surfacing as free lowered inputs."
        ),
        workspace_entrypoint="rules-us/statute/26/32/32.yaml",
        outputs=["earned_income_credit"],
        forbidden_input_names=[
            "eitc_amount",
            "eitc_denied_for_excess_investment_income",
            "meets_full_taxable_year_requirement",
        ],
        expected_output_module_identities={"earned_income_credit": "statute/26/32/32"},
        targets=(),
        live=True,
    ),
    HarnessCase(
        name="live_rulespec_us_scalar_computed_rule_support",
        category="live_stack",
        description=(
            "Real rules-us scalar computed rules without an entity should "
            "still lower as computed outputs."
        ),
        workspace_entrypoint="rules-us/statute/7/2014/d.yaml",
        outputs=["snap_self_employment_cost_exclusion"],
        expected_input_names=[
            "snap_nonfarm_self_employment_production_costs",
            "snap_nonfarm_self_employment_gross_income",
            "snap_farm_self_employment_production_costs",
        ],
        expected_output_module_identities={
            "snap_self_employment_cost_exclusion": "statute/7/2014/d"
        },
        targets=(),
        live=True,
    ),
    HarnessCase(
        name="live_rulespec_inline_conditional_support",
        category="live_stack",
        description=(
            "Real RuleSpec source files should compile inline `if cond: a else: b` "
            "expressions without requiring manual rewrites."
        ),
        workspace_entrypoint=(
            "rulespec/artifacts/eval-suites/"
            "us-snap-child-support-deduction-refresh1-ready-20260412/"
            "01-snap_child_support_deduction/openai-gpt-5.4/source/"
            "snap_child_support_deduction.yaml"
        ),
        outputs=["snap_child_support_deduction"],
        inputs={
            "snap_child_support_payments_made": 500,
            "snap_state_uses_child_support_deduction": True,
        },
        expected_input_names=[
            "snap_child_support_payments_made",
            "snap_state_uses_child_support_deduction",
        ],
        expected_outputs={"snap_child_support_deduction": 500},
        batch_inputs={
            "snap_child_support_payments_made": [500, 500],
            "snap_state_uses_child_support_deduction": [True, False],
        },
        expected_batch_outputs={"snap_child_support_deduction": [500, 0]},
        live=True,
    ),
    HarnessCase(
        name="live_rulespec_source_import_resolution",
        category="live_stack",
        description=(
            "Real RuleSpec source files should resolve citation imports through "
            "their adjacent artifact context instead of surfacing imported helpers "
            "as free inputs."
        ),
        workspace_entrypoint=(
            "rulespec/artifacts/eval-suites/"
            "us-snap-net-income-pre-shelter-refresh2-revalidated-ready-20260412/"
            "01-snap_net_income_pre_shelter/openai-gpt-5.4/source/"
            "SNAP-pre-shelter-net-income-under-7-USC-2014-e-6-A.yaml"
        ),
        outputs=["snap_net_income_pre_shelter"],
        forbidden_input_names=[
            "snap_gross_income",
            "snap_standard_deduction",
            "snap_earned_income_deduction",
            "snap_dependent_care_deduction",
            "snap_child_support_deduction",
            "snap_excess_medical_expense_deduction",
            "snap_other_applicable_deductions_before_shelter",
        ],
        targets=(),
        live=True,
    ),
    HarnessCase(
        name="live_rulespec_us_co_regulation_table_support",
        category="live_stack",
        description=(
            "Real state regulation files should compile multiline inline-condition "
            "tables without flattening them by hand."
        ),
        workspace_entrypoint="rules-us-co/regulation/9-CCR-2503-6/3.606.1/F.yaml",
        outputs=["need_standard_for_assistance_unit"],
        inputs={
            "number_of_children_in_assistance_unit": 2,
            "number_of_caretakers_in_assistance_unit": 1,
        },
        expected_input_names=[
            "number_of_children_in_assistance_unit",
            "number_of_caretakers_in_assistance_unit",
        ],
        expected_output_module_identities={
            "need_standard_for_assistance_unit": "regulation/9-CCR-2503-6/3.606.1/F"
        },
        expected_outputs={"need_standard_for_assistance_unit": 421},
        live=True,
    ),
    HarnessCase(
        name="live_rulespec_us_co_regulation_import_graph_resolution",
        category="live_stack",
        description=(
            "State regulation imports should surface imported free inputs through "
            "qualified rule-identity names instead of merged internal names."
        ),
        workspace_entrypoint="rules-us-co/regulation/9-CCR-2503-6/3.606.1/H.yaml",
        outputs=["meets_gross_income_need_standard_test"],
        inputs={
            (
                "regulation/9-CCR-2503-6/3.606.1/F."
                "number_of_children_in_assistance_unit"
            ): 2,
            (
                "regulation/9-CCR-2503-6/3.606.1/F."
                "number_of_caretakers_in_assistance_unit"
            ): 1,
            "countable_gross_earned_income_after_disregards": 200,
            "countable_unearned_income": 200,
        },
        expected_input_names=[
            "regulation/9-CCR-2503-6/3.606.1/F.number_of_children_in_assistance_unit",
            "regulation/9-CCR-2503-6/3.606.1/F.number_of_caretakers_in_assistance_unit",
            "countable_gross_earned_income_after_disregards",
            "countable_unearned_income",
        ],
        expected_output_module_identities={
            "meets_gross_income_need_standard_test": "regulation/9-CCR-2503-6/3.606.1/H"
        },
        expected_outputs={"meets_gross_income_need_standard_test": True},
        live=True,
    ),
    HarnessCase(
        name="live_rulespec_us_co_statute_import_graph_resolution",
        category="live_stack",
        description=(
            "State statute imports should surface upstream rule inputs through "
            "qualified rule-identity names."
        ),
        workspace_entrypoint="rules-us-co/statute/crs/26-2-711/1/a/I.yaml",
        outputs=[
            "participant_is_subject_to_sanction_for_noncompliance_with_individual_responsibility_contract"
        ],
        inputs={
            (
                "statute/crs/26-2-703/12."
                "contract_is_entered_into_by_participant_and_county_department"
            ): True,
            "statute/crs/26-2-703/12.contract_is_pursuant_to_section_26_2_708": True,
            "participant_fails_to_comply_with_terms_and_conditions_of_contract": True,
            "good_cause_exists_as_determined_by_county": False,
        },
        expected_input_names=[
            "statute/crs/26-2-703/12.contract_is_entered_into_by_participant_and_county_department",
            "statute/crs/26-2-703/12.contract_is_pursuant_to_section_26_2_708",
            "participant_fails_to_comply_with_terms_and_conditions_of_contract",
            "good_cause_exists_as_determined_by_county",
        ],
        expected_output_module_identities={
            (
                "participant_is_subject_to_sanction_for_noncompliance_with_"
                "individual_responsibility_contract"
            ): "statute/crs/26-2-711/1/a/I"
        },
        expected_outputs={
            (
                "participant_is_subject_to_sanction_for_noncompliance_with_"
                "individual_responsibility_contract"
            ): True
        },
        live=True,
    ),
)


def run_compiler_harness(
    case_names: list[str] | None = None,
    *,
    include_external: bool = False,
    include_live: bool = False,
) -> HarnessSummary:
    """Run the objective compiler harness."""
    selected_cases = _select_cases(
        case_names,
        include_external=include_external,
        include_live=include_live,
    )
    results = [_run_case(case) for case in selected_cases]
    by_category: dict[str, dict[str, int]] = {}
    for result in results:
        category_summary = by_category.setdefault(
            result.category,
            {"total": 0, "passed": 0, "failed": 0, "skipped": 0},
        )
        category_summary["total"] += 1
        category_summary[result.status] += 1

    passed = sum(1 for result in results if result.status == "passed")
    failed = sum(1 for result in results if result.status == "failed")
    skipped = sum(1 for result in results if result.status == "skipped")
    return HarnessSummary(
        total=len(results),
        passed=passed,
        failed=failed,
        skipped=skipped,
        by_category=by_category,
        results=results,
    )


def format_harness_summary(summary: HarnessSummary) -> str:
    """Format a harness summary for CLI output."""
    lines = [
        f"Compiler harness score: {summary.score}",
        f"Passed: {summary.passed}",
        f"Failed: {summary.failed}",
        f"Skipped: {summary.skipped}",
        "",
        "By category:",
    ]
    for category, counts in sorted(summary.by_category.items()):
        lines.append(
            f"- {category}: {counts['passed']}/{counts['total']} passed"
            + (f", {counts['failed']} failed" if counts["failed"] else "")
            + (f", {counts['skipped']} skipped" if counts["skipped"] else "")
        )

    failing = [result for result in summary.results if result.status != "passed"]
    if failing:
        lines.append("")
        lines.append("Non-passing cases:")
        for result in failing:
            lines.append(f"- {result.case} [{result.status}]: {result.detail}")

    return "\n".join(lines)


def format_harness_summary_json(summary: HarnessSummary) -> str:
    """Format a harness summary as JSON."""
    return json.dumps(summary.to_dict(), indent=2, sort_keys=True)


def _select_cases(
    case_names: list[str] | None,
    *,
    include_external: bool = False,
    include_live: bool = False,
) -> list[HarnessCase]:
    """Select a subset of harness cases by name."""
    if case_names is None:
        return [
            case
            for case in HARNESS_CASES
            if (include_external or not case.external)
            and (include_live or not case.live)
        ]

    available = {case.name: case for case in HARNESS_CASES}
    missing = [name for name in case_names if name not in available]
    if missing:
        names = ", ".join(missing)
        raise CompilationError(f"Unknown harness case(s): {names}.")
    return [available[name] for name in case_names]


def _run_case(case: HarnessCase) -> HarnessResult:
    """Run one harness case."""
    try:
        program = _load_case_program(case)
        expected_outputs = _resolve_expected_outputs(case)
        lowered_program = program.to_lowered_program(
            effective_date=case.effective_date,
            parameter_overrides=case.parameter_overrides,
            rule_bindings=_load_case_rule_bindings(case),
            outputs=case.outputs,
        )
        lowered_detail = _check_lowered_program(
            lowered_program,
            case=case,
            expected_outputs=expected_outputs,
        )
        if lowered_detail is not None:
            return HarnessResult(
                case=case.name,
                category=case.category,
                passed=False,
                status="failed",
                detail=lowered_detail,
            )
        if case.batch_inputs is not None:
            batch_detail = _check_batch_runtime(
                lowered_program,
                case.batch_inputs,
                case.expected_batch_outputs,
                output_tolerances=case.output_tolerances,
            )
            if batch_detail is not None:
                return HarnessResult(
                    case=case.name,
                    category=case.category,
                    passed=False,
                    status="failed",
                    detail=batch_detail,
                )
        runtime_inputs = None
        if case.inputs is not None:
            lowered_input_names = {
                lowered_input.external_name for lowered_input in lowered_program.inputs
            } | {lowered_input.name for lowered_input in lowered_program.inputs}
            runtime_inputs = {
                name: value
                for name, value in case.inputs.items()
                if name in lowered_input_names
            }
        generated: dict[str, str] = {}
        node_available = _has_node_runtime()
        rustc_available = _has_rustc_runtime()
        for target in case.targets:
            generated[target] = _compile_target(case, lowered_program, target)

        if case.expected_error is not None:
            return HarnessResult(
                case=case.name,
                category=case.category,
                passed=False,
                status="failed",
                detail="Expected compilation to fail, but it succeeded.",
            )

        if "js" in generated and node_available:
            js_check = _check_js_syntax(generated["js"])
            if js_check is not None:
                return HarnessResult(
                    case=case.name,
                    category=case.category,
                    passed=False,
                    status="failed",
                    detail=js_check,
                )
            if runtime_inputs is not None and expected_outputs is not None:
                runtime_detail = _check_js_runtime(
                    generated["js"],
                    runtime_inputs,
                    expected_outputs,
                    output_tolerances=case.output_tolerances,
                )
                if runtime_detail is not None:
                    return HarnessResult(
                        case=case.name,
                        category=case.category,
                        passed=False,
                        status="failed",
                        detail=runtime_detail,
                    )

        if (
            "python" in generated
            and runtime_inputs is not None
            and expected_outputs is not None
        ):
            runtime_detail = _check_python_runtime(
                generated["python"],
                runtime_inputs,
                expected_outputs,
                output_tolerances=case.output_tolerances,
            )
            if runtime_detail is not None:
                return HarnessResult(
                    case=case.name,
                    category=case.category,
                    passed=False,
                    status="failed",
                    detail=runtime_detail,
                )

        if (
            "rust" in generated
            and rustc_available
            and runtime_inputs is not None
            and expected_outputs is not None
        ):
            rust_input_kinds = {
                name: compiled_input.value_kind
                for compiled_input in lowered_program.inputs
                for name in {compiled_input.external_name, compiled_input.name}
            }
            runtime_detail = _check_rust_runtime(
                generated["rust"],
                runtime_inputs,
                rust_input_kinds,
                expected_outputs,
                output_tolerances=case.output_tolerances,
            )
            if runtime_detail is not None:
                return HarnessResult(
                    case=case.name,
                    category=case.category,
                    passed=False,
                    status="failed",
                    detail=runtime_detail,
                )

        if "js" in generated and not node_available:
            return HarnessResult(
                case=case.name,
                category=case.category,
                passed=False,
                status="skipped",
                detail=(
                    "Node.js is not available, so JavaScript validation was skipped "
                    "for this case after Python checks passed."
                ),
            )

        if "rust" in generated and not rustc_available:
            return HarnessResult(
                case=case.name,
                category=case.category,
                passed=False,
                status="skipped",
                detail=(
                    "rustc is not available, so Rust validation was skipped "
                    "for this case after JS/Python checks passed."
                ),
            )

        return HarnessResult(
            case=case.name,
            category=case.category,
            passed=True,
            status="passed",
            detail=case.description,
        )
    except ImportError as exc:
        if case.external or case.live:
            return HarnessResult(
                case=case.name,
                category=case.category,
                passed=False,
                status="skipped",
                detail=str(exc),
            )
        return HarnessResult(
            case=case.name,
            category=case.category,
            passed=False,
            status="failed",
            detail=str(exc),
        )
    except (CompilationError, ParameterBindingError) as exc:
        if case.expected_error and case.expected_error in str(exc):
            return HarnessResult(
                case=case.name,
                category=case.category,
                passed=True,
                status="passed",
                detail=f"Failed as expected: {exc}",
            )
        return HarnessResult(
            case=case.name,
            category=case.category,
            passed=False,
            status="failed",
            detail=str(exc),
        )


def _compile_target(case: HarnessCase, program, target: str) -> str:
    """Compile one case to one target."""
    if target == "js":
        return program.to_js_generator().generate()
    if target == "python":
        return program.to_python_generator().generate()
    if target == "rust":
        return program.to_rust_generator().generate()
    raise CompilationError(f"Unknown harness target '{target}'.")


def _load_case_program(case: HarnessCase):
    """Load a harness case as either one in-memory file or a file graph."""
    if case.repo_entrypoint is not None:
        return load_rulespec_program(_REPO_ROOT / case.repo_entrypoint)
    if case.workspace_entrypoint is not None:
        path = _WORKSPACE_ROOT / case.workspace_entrypoint
        if not path.exists():
            raise ImportError(f"Workspace harness case '{case.name}' requires {path}.")
        return load_rulespec_program(path)

    if not case.supporting_files:
        if case.rulespec is None:
            raise CompilationError(
                f"Harness case '{case.name}' does not define a RuleSpec entrypoint."
            )
        return parse_rulespec(case.rulespec)

    with tempfile.TemporaryDirectory(prefix="rulespec_compile_harness_") as tmp_dir:
        root = Path(tmp_dir)
        if case.rulespec is None:
            raise CompilationError(
                f"Harness case '{case.name}' does not define a RuleSpec entrypoint."
            )
        (root / case.entrypoint).write_text(case.rulespec.strip() + "\n")
        for relative_path, content in case.supporting_files.items():
            target = root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content.strip() + "\n")
        return load_rulespec_program(root / case.entrypoint)


def _load_case_rule_bindings(case: HarnessCase) -> Any:
    """Load any binding files associated with one harness case."""
    bundles = []
    for relative_path in case.repo_binding_files:
        bundles.append(load_rule_bindings_file(_REPO_ROOT / relative_path))
    for relative_path in case.workspace_binding_files:
        path = _WORKSPACE_ROOT / relative_path
        if not path.exists():
            raise ImportError(f"Workspace harness case '{case.name}' requires {path}.")
        bundles.append(load_rule_bindings_file(path))
    if not bundles:
        return {}
    return merge_rule_bindings(*bundles)


def _resolve_expected_outputs(case: HarnessCase) -> dict[str, Any] | None:
    """Resolve expected outputs from literals or a reference oracle."""
    if case.expected_outputs is not None:
        return case.expected_outputs
    if case.oracle is None:
        return None
    if case.inputs is None:
        raise CompilationError(
            f"Harness case '{case.name}' uses oracle '{case.oracle}' without inputs."
        )
    try:
        oracle = _ORACLE_FUNCTIONS[case.oracle]
    except KeyError as exc:
        raise CompilationError(
            f"Harness case '{case.name}' references unknown oracle '{case.oracle}'."
        ) from exc
    return oracle(case.inputs)


def _check_batch_runtime(
    program,
    batch_inputs: dict[str, list[Any]],
    expected_outputs: dict[str, list[Any]] | None,
    output_tolerances: dict[str, float] | None = None,
) -> str | None:
    """Execute one lowered program in batch mode and compare outputs."""
    if expected_outputs is None:
        raise CompilationError("Batch harness case is missing expected_batch_outputs.")
    result = execute_lowered_program_batch(program, pd.DataFrame(batch_inputs))
    actual = result.to_dict(orient="list")
    for name, expected_values in expected_outputs.items():
        actual_values = actual.get(name)
        if actual_values is None:
            return f"Expected batch output '{name}', but it was missing."
        if len(actual_values) != len(expected_values):
            return (
                f"Expected batch output {name} to have {len(expected_values)} rows, "
                f"got {len(actual_values)}."
            )
        tolerance = (output_tolerances or {}).get(name)
        for index, (actual_value, expected_value) in enumerate(
            zip(actual_values, expected_values, strict=True)
        ):
            if tolerance is None:
                if actual_value != expected_value:
                    return (
                        f"Expected batch output {name}[{index}]={expected_value!r}, "
                        f"got {actual_value!r}."
                    )
            elif not _within_tolerance(actual_value, expected_value, tolerance):
                return (
                    f"Expected batch output {name}[{index}] within ±{tolerance!r} "
                    f"of {expected_value!r}, got {actual_value!r}."
                )
    extra_names = set(actual) - set(expected_outputs)
    if extra_names:
        names = ", ".join(sorted(extra_names))
        return f"Expected only requested batch outputs, but got extra values: {names}."
    return None


def _check_lowered_program(
    program,
    *,
    case: HarnessCase,
    expected_outputs: dict[str, Any] | None,
) -> str | None:
    """Verify that the lowered bundle is serializable and internally consistent."""
    payload = json.loads(program.to_json())
    input_names = [
        compiled_input.get("public_name") or compiled_input["name"]
        for compiled_input in payload["inputs"]
    ]
    if case.expected_input_names is not None and set(input_names) != set(
        case.expected_input_names
    ):
        return (
            "Lowered bundle inputs did not match the expected public input surface: "
            f"{input_names}."
        )
    if case.forbidden_input_names:
        forbidden = sorted(set(input_names) & set(case.forbidden_input_names))
        if forbidden:
            return (
                "Lowered bundle exposed imported live symbols as free inputs: "
                f"{', '.join(forbidden)}."
            )

    output_names = [output["name"] for output in payload["outputs"]]
    if expected_outputs is not None and set(output_names) != set(expected_outputs):
        return (
            "Lowered bundle outputs did not match the expected public surface: "
            f"{output_names}."
        )
    if case.expected_output_module_identities:
        output_identities = {
            output["name"]: output.get("module_identity", "")
            for output in payload["outputs"]
        }
        for name, expected_identity in case.expected_output_module_identities.items():
            actual_identity = output_identities.get(name)
            if actual_identity != expected_identity:
                return (
                    f"Lowered bundle output '{name}' had module_identity "
                    f"{actual_identity!r}, expected {expected_identity!r}."
                )

    computation_names = {computation["name"] for computation in payload["computations"]}
    for output in payload["outputs"]:
        variable_name = output["variable_name"]
        if variable_name not in computation_names:
            return (
                "Lowered bundle output references unknown computation "
                f"'{variable_name}'."
            )
        if "value_kind" not in output:
            return f"Lowered bundle output '{output['name']}' is missing value_kind."
    for parameter in payload["parameters"]:
        if "value_kind" not in parameter:
            return (
                f"Lowered bundle parameter '{parameter['name']}' is missing value_kind."
            )
        if "lookup_kind" not in parameter:
            return (
                "Lowered bundle parameter "
                f"'{parameter['name']}' is missing lookup_kind."
            )
        if (
            parameter["lookup_kind"] == "indexed"
            and "index_value_kind" not in parameter
        ):
            return (
                "Lowered bundle parameter "
                f"'{parameter['name']}' is missing index_value_kind."
            )
    for computation in payload["computations"]:
        if "value_kind" not in computation:
            return (
                "Lowered bundle computation "
                f"'{computation['name']}' is missing value_kind."
            )
        if "local_value_kinds" not in computation:
            return (
                "Lowered bundle computation "
                f"'{computation['name']}' is missing local_value_kinds."
            )
        local_names = set(computation.get("local_names", []))
        local_value_kinds = set(computation["local_value_kinds"])
        if local_names != local_value_kinds:
            return (
                "Lowered bundle computation "
                f"'{computation['name']}' has incomplete local_value_kinds."
            )
    return None


def _oracle_eitc_reference(inputs: dict[str, Any]) -> dict[str, Any]:
    """Calculate expected EITC outputs from the Python reference oracle."""
    result = calculate_eitc(**inputs)
    return {"eitc": result.eitc}


def _oracle_ctc_reference(inputs: dict[str, Any]) -> dict[str, Any]:
    """Calculate expected CTC/ACTC outputs from the Python reference oracles."""
    ctc_result = calculate_ctc(
        n_qualifying_children=inputs["n_qualifying_children"],
        agi=inputs["agi"],
        is_joint=inputs["is_joint"],
    )
    actc_result = calculate_actc(
        n_qualifying_children=inputs["n_qualifying_children"],
        earned_income=inputs["earned_income"],
    )
    return {"ctc": ctc_result.ctc, "actc": actc_result.actc}


def _oracle_snap_reference(inputs: dict[str, Any]) -> dict[str, Any]:
    """Calculate expected SNAP outputs from the Python reference oracle."""
    result = calculate_snap_benefit(**inputs)
    return {"snap_benefit": result.benefit}


def _oracle_policyengine_snap_reference(inputs: dict[str, Any]) -> dict[str, Any]:
    """Calculate expected SNAP outputs from a PolicyEngine household oracle."""
    result = run_policyengine_household(
        {
            "gross_income": inputs["gross_income"],
            "household_size": inputs["household_size"],
            "state_code": inputs.get("state_code", "CA"),
        }
    )
    return {"snap_benefit": result["pe_snap"]}


_ORACLE_FUNCTIONS: dict[str, Any] = {
    "eitc_reference": _oracle_eitc_reference,
    "ctc_reference": _oracle_ctc_reference,
    "snap_reference": _oracle_snap_reference,
    "policyengine_snap_reference": _oracle_policyengine_snap_reference,
}


def _check_python_runtime(
    code: str,
    inputs: dict[str, Any],
    expected_outputs: dict[str, Any],
    output_tolerances: dict[str, float] | None = None,
) -> str | None:
    """Execute generated Python and compare expected outputs."""
    namespace: dict[str, Any] = {}
    exec(code, namespace)
    result = namespace["calculate"](**inputs)
    return _check_runtime_result(
        result,
        expected_outputs,
        target="Python",
        output_tolerances=output_tolerances,
    )


def _check_js_runtime(
    code: str,
    inputs: dict[str, Any],
    expected_outputs: dict[str, Any],
    output_tolerances: dict[str, float] | None = None,
) -> str | None:
    """Execute generated JavaScript and compare expected outputs."""
    node = shutil.which("node")
    if node is None:
        return None

    harness_code = "\n".join(
        [
            code,
            f"const __harnessInputs = {json.dumps(inputs, sort_keys=True)};",
            "const __harnessResult = calculate(__harnessInputs);",
            "console.log(JSON.stringify(__harnessResult));",
        ]
    )
    proc = subprocess.run(
        [node, "--input-type=module"],
        input=harness_code,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return proc.stderr.strip() or "Generated JavaScript failed at runtime."

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return "Generated JavaScript did not print a result."

    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError:
        return "Generated JavaScript returned non-JSON runtime output."

    return _check_runtime_result(
        result,
        expected_outputs,
        target="JS",
        output_tolerances=output_tolerances,
    )


def _check_runtime_result(
    result: dict[str, Any],
    expected_outputs: dict[str, Any],
    target: str,
    output_tolerances: dict[str, float] | None = None,
) -> str | None:
    """Compare runtime outputs against the expected public result shape."""
    for name, expected in expected_outputs.items():
        actual = result.get(name)
        tolerance = (output_tolerances or {}).get(name)
        if tolerance is None:
            if actual != expected:
                return f"Expected {target} output {name}={expected!r}, got {actual!r}."
            continue
        if not _within_tolerance(actual, expected, tolerance):
            return (
                f"Expected {target} output {name} within ±{tolerance!r} of "
                f"{expected!r}, got {actual!r}."
            )
    extra_names = set(result) - set(expected_outputs) - {"citations"}
    if extra_names:
        names = ", ".join(sorted(extra_names))
        return f"Expected only requested outputs, but got extra values: {names}."
    return None


def _check_rust_runtime(
    code: str,
    inputs: dict[str, Any],
    input_value_kinds: dict[str, str],
    expected_outputs: dict[str, Any],
    output_tolerances: dict[str, float] | None = None,
) -> str | None:
    """Compile and execute generated Rust and compare expected outputs."""
    rustc = shutil.which("rustc")
    if rustc is None:
        return None

    with tempfile.TemporaryDirectory(prefix="rulespec_compile_rust_") as tmp_dir:
        root = Path(tmp_dir)
        source = root / "main.rs"
        binary = root / "calculator"
        source.write_text(
            "\n".join(
                [
                    code,
                    "",
                    "fn main() {",
                    "    let mut public_inputs = BTreeMap::new();",
                    *[
                        _format_rust_public_input_binding(
                            name,
                            value,
                            input_value_kinds,
                        )
                        for name, value in inputs.items()
                    ],
                    "    let result = calculate_public(&public_inputs);",
                    "    for (name, value) in result.outputs.iter() {",
                    '        println!("{}={:?}", name, value);',
                    "    }",
                    "}",
                ]
            )
        )
        compile_proc = subprocess.run(
            _rustc_compile_command(rustc, source, binary),
            capture_output=True,
            text=True,
        )
        if compile_proc.returncode != 0:
            return compile_proc.stderr.strip() or "Generated Rust failed to compile."

        run_proc = subprocess.run(
            [str(binary)],
            capture_output=True,
            text=True,
        )
        if run_proc.returncode != 0:
            return run_proc.stderr.strip() or "Generated Rust failed at runtime."

    result: dict[str, Any] = {}
    for line in run_proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            name, raw_value = stripped.split("=", 1)
        except ValueError:
            return "Generated Rust returned malformed runtime output."
        parsed_value = _parse_rust_runtime_value(raw_value)
        if parsed_value is None:
            return f"Generated Rust returned unsupported runtime value {raw_value!r}."
        result[name] = parsed_value
    return _check_runtime_result(
        result,
        expected_outputs,
        target="Rust",
        output_tolerances=output_tolerances,
    )


def _within_tolerance(actual: Any, expected: Any, tolerance: float) -> bool:
    """Check whether one runtime value is within a numeric tolerance."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual == expected
    if not isinstance(actual, int | float) or not isinstance(expected, int | float):
        return actual == expected
    return abs(float(actual) - float(expected)) <= tolerance


def _check_js_syntax(code: str) -> str | None:
    """Check generated JS syntax when Node.js is available."""
    node = shutil.which("node")
    if node is None:
        return None
    proc = subprocess.run(
        [node, "--input-type=module", "--check"],
        input=code,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return proc.stderr.strip() or "Generated JavaScript has invalid syntax."
    return None


def _has_node_runtime() -> bool:
    """Return whether Node.js is available for JavaScript validation."""
    return shutil.which("node") is not None


def _has_rustc_runtime() -> bool:
    """Return whether rustc is available for Rust validation."""
    return shutil.which("rustc") is not None


def _render_rust_input_literal(value: Any, value_kind: str) -> str:
    """Render one harness input value to Rust syntax."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value_kind == "integer":
        if isinstance(value, bool):
            raise CompilationError(
                f"Rust harness does not support boolean integer input {value!r}."
            )
        if isinstance(value, (int, float)) and float(value).is_integer():
            return str(int(value))
        raise CompilationError(
            "Rust harness expected exact integer input for kind "
            f"'integer', got {value!r}."
        )
    if isinstance(value, (int, float)):
        rendered = repr(float(value))
        if "e" not in rendered and "." not in rendered:
            rendered += ".0"
        return rendered
    raise CompilationError(
        f"Rust harness does not support non-scalar input value {value!r}."
    )


def _format_rust_input_binding(
    name: str,
    value: Any,
    input_value_kinds: dict[str, str],
) -> str:
    """Render one Rust input struct field assignment for harness execution."""
    return (
        "        "
        f"{name}: "
        f"{_render_rust_input_literal(value, input_value_kinds.get(name, 'number'))},"
    )


def _format_rust_public_input_binding(
    name: str,
    value: Any,
    input_value_kinds: dict[str, str],
) -> str:
    """Render one Rust public-input map insertion for harness execution."""
    literal = _render_rust_input_literal(value, input_value_kinds.get(name, "number"))
    kind = input_value_kinds.get(name, "number")
    if kind == "boolean":
        rendered = f"RuleSpecValue::Bool({literal})"
    elif kind == "integer":
        rendered = f"RuleSpecValue::Integer({literal})"
    else:
        rendered = f"RuleSpecValue::Number({literal})"
    return f"    public_inputs.insert({json.dumps(name)}.to_string(), {rendered});"


def _parse_rust_runtime_value(raw_value: str) -> Any | None:
    """Parse one debug-printed Rust runtime value."""
    if raw_value == "Bool(true)":
        return True
    if raw_value == "Bool(false)":
        return False
    if raw_value.startswith("Integer(") and raw_value.endswith(")"):
        inner = raw_value[len("Integer(") : -1]
        try:
            return int(inner)
        except ValueError:
            return None
    if raw_value.startswith("Number(") and raw_value.endswith(")"):
        inner = raw_value[len("Number(") : -1]
        try:
            return float(inner)
        except ValueError:
            return None
    if raw_value.startswith('String("') and raw_value.endswith('")'):
        return raw_value[len('String("') : -2]
    return None


def _rustc_compile_command(rustc: str, source: Path, binary: Path) -> list[str]:
    """Build a rustc compile command with a stable system linker when available."""
    command = [rustc, "--edition=2021", str(source), "-o", str(binary)]
    system_linker = Path("/usr/bin/cc")
    if system_linker.exists():
        command[1:1] = ["-C", f"linker={system_linker}"]
    return command
