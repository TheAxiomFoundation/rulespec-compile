"""Tests for the objective compiler harness."""

import json
import shutil
from pathlib import Path

import pytest

from src.rulespec_compile.harness import (
    HARNESS_CASES,
    HarnessCase,
    _check_js_runtime,
    _run_case,
    format_harness_summary,
    format_harness_summary_json,
    run_compiler_harness,
)


class TestCompilerHarness:
    """Tests for harness execution and formatting."""

    def _assert_live_case_passes_or_skips(self, case_name: str):
        summary = run_compiler_harness(case_names=[case_name])

        assert summary.total == 1
        assert summary.failed == 0
        assert summary.results[0].case == case_name
        assert summary.results[0].status in {"passed", "skipped"}
        if summary.results[0].status == "skipped":
            assert "requires" in summary.results[0].detail
        return summary

    def test_run_compiler_harness_all_cases_pass(self):
        """The built-in harness runs green in the current repo state."""
        summary = run_compiler_harness()
        default_cases = [
            case for case in HARNESS_CASES if not case.external and not case.live
        ]

        assert summary.total == len(default_cases)
        assert summary.failed == 0
        assert summary.passed == len(default_cases)
        assert "control_flow" in summary.by_category
        assert "graph" in summary.by_category
        assert "live_stack" not in summary.by_category
        assert "oracle" in summary.by_category
        assert "policyengine" not in summary.by_category
        assert "subgraph" in summary.by_category

    def test_run_compiler_harness_single_case(self):
        """The harness can run a selected case by name."""
        summary = run_compiler_harness(case_names=["basic_straight_line"])

        assert summary.total == 1
        assert summary.passed == 1
        assert summary.results[0].case == "basic_straight_line"

    def test_run_compiler_harness_oracle_example_case(self):
        """Example-backed oracle cases compare compiled output to references."""
        summary = run_compiler_harness(case_names=["oracle_eitc_example"])

        assert summary.total == 1
        assert summary.passed == 1
        assert summary.results[0].case == "oracle_eitc_example"

    def test_run_compiler_harness_batch_branch_case(self):
        """Harness cases can validate lowered batch execution directly."""
        summary = run_compiler_harness(case_names=["branching_batch_execution"])

        assert summary.total == 1
        assert summary.passed == 1
        assert summary.results[0].case == "branching_batch_execution"

    def test_run_compiler_harness_external_policyengine_case(self, monkeypatch):
        """External oracle cases can compare compiled output to PolicyEngine."""
        monkeypatch.setattr(
            "src.rulespec_compile.harness.run_policyengine_household",
            lambda inputs: {"pe_snap": 410.0},
        )

        summary = run_compiler_harness(case_names=["policyengine_snap_example"])

        assert summary.total == 1
        assert summary.passed == 1
        assert summary.results[0].case == "policyengine_snap_example"

    def test_run_compiler_harness_external_case_skips_without_dependency(
        self, monkeypatch
    ):
        """External oracle cases skip cleanly when PolicyEngine is unavailable."""

        def _raise_import_error(_inputs):
            raise ImportError("policyengine-us required for validation")

        monkeypatch.setattr(
            "src.rulespec_compile.harness.run_policyengine_household",
            _raise_import_error,
        )

        summary = run_compiler_harness(case_names=["policyengine_snap_example"])

        assert summary.total == 1
        assert summary.skipped == 1
        assert summary.results[0].status == "skipped"

    def test_run_compiler_harness_include_live_adds_live_cases(self):
        """The opt-in live lane adds curated current-stack compatibility cases."""
        summary = run_compiler_harness(include_live=True)
        live_cases = [case for case in HARNESS_CASES if case.live]
        default_cases = [
            case for case in HARNESS_CASES if not case.external and not case.live
        ]

        assert summary.total == len(default_cases) + len(live_cases)
        assert summary.failed == 0
        assert "live_stack" in summary.by_category

    def test_run_compiler_harness_rulespec_us_v1_payroll_live_case(self):
        """The live lane compiles a current rules-us RuleSpec v1 payroll file."""
        self._assert_live_case_passes_or_skips("live_rulespec_us_v1_payroll_tax")

    def test_run_compiler_harness_rulespec_us_v1_credit_live_case(self):
        """The live lane compiles a current rules-us RuleSpec v1 credit file."""
        self._assert_live_case_passes_or_skips("live_rulespec_us_v1_credit_formula")

    def test_run_compiler_harness_rulespec_us_v1_if_live_case(self):
        """The live lane compiles a current RuleSpec v1 statement conditional."""
        self._assert_live_case_passes_or_skips("live_rulespec_us_v1_statement_if")

    def test_run_compiler_harness_rulespec_us_v1_nested_live_case(self):
        """The live lane preserves plural-path identity for nested files."""
        self._assert_live_case_passes_or_skips(
            "live_rulespec_us_v1_nested_path_identity"
        )

    def test_run_case_workspace_case_skips_when_repo_is_missing(self, monkeypatch):
        """Missing sibling live repos skip cleanly for opt-in workspace cases."""
        case = HarnessCase(
            name="missing_live_workspace",
            category="live_stack",
            description="Missing workspace repo skips.",
            workspace_entrypoint="definitely-missing-repo/example.yaml",
            targets=(),
            live=True,
        )
        monkeypatch.setattr(
            "src.rulespec_compile.harness._WORKSPACE_ROOT",
            Path("/definitely/missing"),
        )
        result = _run_case(case)

        assert result.status == "skipped"
        assert "requires" in result.detail

    def test_run_case_checks_expected_inputs_and_output_identity(self):
        """Lowered bundle checks can validate structural compatibility expectations."""
        case = HarnessCase(
            name="structural_lowering_contract",
            category="core",
            description=(
                "Structural lowered checks pass for expected inputs and identity."
            ),
            rulespec="""
tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    wages * 0.1
""",
            inputs={"wages": 100},
            expected_outputs={"tax": 10},
            expected_input_names=["wages"],
            expected_output_module_identities={"tax": "main"},
            supporting_files={"notes/placeholder.txt": "placeholder"},
            targets=(),
        )

        result = _run_case(case)

        assert result.status == "passed"

    def test_run_case_checks_output_identity_without_expected_inputs(self):
        """Structural checks can validate output identity without pinning inputs."""
        case = HarnessCase(
            name="identity_only_lowering_contract",
            category="core",
            description="Structural lowered checks can focus on output identity.",
            rulespec="""
tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    wages * 0.1
""",
            inputs={"wages": 100},
            expected_outputs={"tax": 10},
            expected_output_module_identities={"tax": "main"},
            supporting_files={"notes/placeholder.txt": "placeholder"},
            targets=(),
        )

        result = _run_case(case)

        assert result.status == "passed"

    def test_format_harness_summary_text_and_json(self):
        """Harness summaries can be rendered for CLI output."""
        summary = run_compiler_harness(case_names=["basic_straight_line"])

        text = format_harness_summary(summary)
        payload = json.loads(format_harness_summary_json(summary))

        assert "Compiler harness score: 1/1" in text
        assert payload["score"] == "1/1"
        assert payload["results"][0]["case"] == "basic_straight_line"

    def test_check_js_runtime_detects_wrong_output(self):
        """JS harness execution catches semantic output mismatches."""
        if shutil.which("node") is None:
            pytest.skip("Node.js is required for JS runtime harness checks.")

        code = """
function calculate({ wages = 0 }) {
  return {
    tax: wages * 0.1,
    citations: [],
  };
}

export { calculate };
export default calculate;
"""

        detail = _check_js_runtime(code, {"wages": 100}, {"tax": 99})

        assert detail == "Expected JS output tax=99, got 10."

    def test_run_compiler_harness_skips_js_cases_without_node(self, monkeypatch):
        """Harness reports JS-backed cases as skipped when Node.js is unavailable."""
        monkeypatch.setattr("src.rulespec_compile.harness.shutil.which", lambda _: None)

        summary = run_compiler_harness(case_names=["basic_straight_line"])

        assert summary.total == 1
        assert summary.passed == 0
        assert summary.skipped == 1
        assert summary.results[0].status == "skipped"
        assert "after Python checks passed" in summary.results[0].detail
