"""Tests for validation modules: comparator, cps_loader, runners, cli, __init__.

Uses unittest.mock to avoid needing PolicyEngine-US or real CPS data.
These are unit tests focused on coverage, not integration tests.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ============================================================
# validation/__init__.py
# ============================================================


class TestValidationInit:
    """Test that validation __init__ exports are accessible."""

    def test_imports_comparator(self):
        from src.rulespec_compile.validation import Comparator

        assert Comparator is not None

    def test_imports_comparison_config(self):
        from src.rulespec_compile.validation import ComparisonConfig

        assert ComparisonConfig is not None

    def test_imports_comparison_results(self):
        from src.rulespec_compile.validation import ComparisonResults

        assert ComparisonResults is not None

    def test_imports_cps_household(self):
        from src.rulespec_compile.validation import CPSHousehold

        assert CPSHousehold is not None

    def test_imports_load_cps_data(self):
        from src.rulespec_compile.validation import load_cps_data

        assert callable(load_cps_data)

    def test_imports_run_rulespec(self):
        from src.rulespec_compile.validation import run_rulespec

        assert callable(run_rulespec)

    def test_imports_run_policyengine(self):
        from src.rulespec_compile.validation import run_policyengine

        assert callable(run_policyengine)

    def test_imports_run_policyengine_household(self):
        from src.rulespec_compile.validation import run_policyengine_household

        assert callable(run_policyengine_household)


# ============================================================
# validation/comparator.py
# ============================================================


class TestComparisonConfig:
    """Test ComparisonConfig defaults and custom values."""

    def test_default_config(self):
        from src.rulespec_compile.validation.comparator import ComparisonConfig

        config = ComparisonConfig()
        assert config.eitc_tolerance == 1.0
        assert config.ctc_tolerance == 1.0
        assert config.actc_tolerance == 1.0
        assert config.snap_tolerance == 50.0
        assert config.id_col == "household_id"

    def test_custom_config(self):
        from src.rulespec_compile.validation.comparator import ComparisonConfig

        config = ComparisonConfig(eitc_tolerance=5.0, snap_tolerance=100.0)
        assert config.eitc_tolerance == 5.0
        assert config.snap_tolerance == 100.0


class TestMismatchRecord:
    """Test MismatchRecord dataclass."""

    def test_mismatch_record_defaults(self):
        from src.rulespec_compile.validation.comparator import MismatchRecord

        record = MismatchRecord(
            household_id=1,
            variable="eitc",
            rulespec_value=100,
            policyengine_value=200,
            difference=-100,
        )
        assert record.pct_difference is None
        assert record.state_code is None
        assert record.weight == 1.0

    def test_mismatch_record_custom(self):
        from src.rulespec_compile.validation.comparator import MismatchRecord

        record = MismatchRecord(
            household_id=1,
            variable="eitc",
            rulespec_value=100,
            policyengine_value=200,
            difference=-100,
            pct_difference=-50.0,
            state_code="CA",
            weight=2.5,
        )
        assert record.pct_difference == -50.0
        assert record.state_code == "CA"
        assert record.weight == 2.5


class TestComparator:
    """Test Comparator.compare and _compare_variable."""

    def _make_df(self, n=10, add_snap_data=False):
        """Create a sample DataFrame for comparison."""
        df = pd.DataFrame(
            {
                "household_id": range(n),
                "rulespec_eitc": [100] * n,
                "pe_eitc": [100] * n,
                "rulespec_ctc": [200] * n,
                "pe_ctc": [200] * n,
                "rulespec_actc": [50] * n,
                "pe_actc": [50] * n,
                "state_code": ["CA"] * n,
                "weight": [1.0] * n,
            }
        )
        if add_snap_data:
            spm_df = pd.DataFrame(
                {
                    "spm_unit_id": range(n),
                    "rulespec_snap": [300] * n,
                    "pe_snap": [300] * n,
                    "state_code": ["CA"] * n,
                    "weight": [1.0] * n,
                }
            )
            df.attrs["spm_snap_data"] = spm_df
        return df

    def test_compare_all_match(self):
        from src.rulespec_compile.validation.comparator import Comparator

        df = self._make_df(add_snap_data=True)
        comp = Comparator()
        results = comp.compare(df)
        assert results.matches["eitc"] == 10
        assert results.match_rates["eitc"] == 100.0
        assert len(results.mismatches["eitc"]) == 0

    def test_compare_with_mismatches(self):
        from src.rulespec_compile.validation.comparator import Comparator

        df = self._make_df()
        df.loc[0, "rulespec_eitc"] = 500  # Big mismatch
        comp = Comparator()
        results = comp.compare(df)
        assert results.matches["eitc"] == 9
        assert len(results.mismatches["eitc"]) == 1
        assert results.mismatches["eitc"][0].difference == 400

    def test_compare_with_nan_pe_values(self):
        """NaN PE values are excluded from comparison."""
        from src.rulespec_compile.validation.comparator import Comparator

        df = self._make_df()
        df.loc[0, "pe_eitc"] = np.nan
        comp = Comparator()
        results = comp.compare(df)
        # 9 valid, all match
        assert results.matches["eitc"] == 9

    def test_compare_all_nan(self):
        """All NaN PE values results in 0 matches."""
        from src.rulespec_compile.validation.comparator import Comparator

        df = self._make_df()
        df["pe_eitc"] = np.nan
        comp = Comparator()
        results = comp.compare(df)
        assert results.matches["eitc"] == 0

    def test_compare_missing_columns(self):
        """Missing columns are skipped gracefully."""
        from src.rulespec_compile.validation.comparator import Comparator

        df = pd.DataFrame(
            {
                "household_id": [1, 2],
                "rulespec_eitc": [100, 200],
                "pe_eitc": [100, 200],
            }
        )
        comp = Comparator()
        results = comp.compare(df)
        # CTC and ACTC columns missing, should be skipped
        assert "eitc" in results.matches
        assert results.matches["snap"] == 0  # No SPM data

    def test_compare_snap_from_spm(self):
        """SNAP comparison uses SPM data when available."""
        from src.rulespec_compile.validation.comparator import Comparator

        df = self._make_df(add_snap_data=True)
        comp = Comparator()
        results = comp.compare(df)
        assert results.matches["snap"] == 10
        assert results.match_rates["snap"] == 100.0

    def test_compare_snap_spm_mismatch(self):
        """SNAP mismatches are tracked from SPM data."""
        from src.rulespec_compile.validation.comparator import Comparator

        df = self._make_df(add_snap_data=True)
        spm_df = df.attrs["spm_snap_data"]
        spm_df.loc[0, "rulespec_snap"] = 1000  # Big mismatch
        comp = Comparator()
        results = comp.compare(df)
        assert len(results.mismatches["snap"]) == 1

    def test_compare_no_spm_data(self):
        """No SPM data results in 0 SNAP matches."""
        from src.rulespec_compile.validation.comparator import Comparator

        df = self._make_df(add_snap_data=False)
        comp = Comparator()
        results = comp.compare(df)
        assert results.matches["snap"] == 0
        assert results.match_rates["snap"] == 0

    def test_compare_reads_execution_modes_from_dataframe_attrs(self):
        """Comparison results preserve execution-mode provenance from the runner."""
        from src.rulespec_compile.validation.comparator import Comparator

        df = self._make_df(add_snap_data=False)
        df.attrs["rulespec_execution_mode"] = "compiled_example"
        df.attrs["policyengine_execution_mode"] = "policyengine_household"

        results = Comparator().compare(df)

        assert results.rulespec_execution_mode == "compiled_example"
        assert results.policyengine_execution_mode == "policyengine_household"

    def test_pct_difference_calculation(self):
        """Percentage difference is calculated correctly."""
        from src.rulespec_compile.validation.comparator import Comparator

        df = pd.DataFrame(
            {
                "household_id": [1],
                "rulespec_eitc": [150.0],
                "pe_eitc": [100.0],
                "state_code": ["CA"],
                "weight": [1.0],
            }
        )
        comp = Comparator()
        results = comp.compare(df)
        mismatch = results.mismatches["eitc"][0]
        assert mismatch.pct_difference == pytest.approx(50.0)

    def test_pct_difference_zero_pe(self):
        """Percentage difference is None when PE value is 0."""
        from src.rulespec_compile.validation.comparator import Comparator

        df = pd.DataFrame(
            {
                "household_id": [1],
                "rulespec_eitc": [100.0],
                "pe_eitc": [0.0],
                "state_code": ["CA"],
                "weight": [1.0],
            }
        )
        comp = Comparator()
        results = comp.compare(df)
        mismatch = results.mismatches["eitc"][0]
        assert mismatch.pct_difference is None

    def test_custom_config_tolerance(self):
        """Custom tolerance changes match behavior."""
        from src.rulespec_compile.validation.comparator import (
            Comparator,
            ComparisonConfig,
        )

        df = pd.DataFrame(
            {
                "household_id": [1],
                "rulespec_eitc": [101.0],
                "pe_eitc": [100.0],
                "state_code": ["CA"],
                "weight": [1.0],
            }
        )
        # Default tolerance (1.0) should match
        comp = Comparator()
        results = comp.compare(df)
        assert results.matches["eitc"] == 1

        # Tight tolerance should mismatch
        config = ComparisonConfig(eitc_tolerance=0.5)
        comp = Comparator(config)
        results = comp.compare(df)
        assert len(results.mismatches["eitc"]) == 1


class TestComparisonResults:
    """Test ComparisonResults summary, detailed_report, and save_report."""

    def _make_results(self, with_mismatches=False, with_data=False):
        from src.rulespec_compile.validation.comparator import (
            ComparisonConfig,
            ComparisonResults,
            MismatchRecord,
        )

        config = ComparisonConfig()
        mismatches_eitc = []
        if with_mismatches:
            for i in range(6):
                mismatches_eitc.append(
                    MismatchRecord(
                        household_id=i,
                        variable="eitc",
                        rulespec_value=100 + i * 10,
                        policyengine_value=200,
                        difference=-(100 - i * 10),
                        state_code="CA",
                    )
                )

        return ComparisonResults(
            total_households=100,
            variables_compared=["eitc", "ctc", "snap"],
            matches={"eitc": 94 if with_mismatches else 100, "ctc": 100, "snap": 0},
            mismatches={
                "eitc": mismatches_eitc,
                "ctc": [],
                "snap": [],
            },
            match_rates={
                "eitc": 94.0 if with_mismatches else 100.0,
                "ctc": 100.0,
                "snap": 0.0,
            },
            config=config,
            full_data=pd.DataFrame({"a": [1, 2]}) if with_data else None,
            rulespec_execution_mode="compiled_example",
            policyengine_execution_mode="policyengine_household",
        )

    def test_summary(self):
        results = self._make_results()
        summary = results.summary()
        assert summary["total_households"] == 100
        assert summary["rulespec_execution_mode"] == "compiled_example"
        assert summary["policyengine_execution_mode"] == "policyengine_household"
        assert "eitc" in summary["variables"]
        assert summary["variables"]["eitc"]["match_rate"] == 100.0

    def test_detailed_report_no_mismatches(self):
        results = self._make_results()
        report = results.detailed_report()
        assert "RuleSpec vs PolicyEngine-US Validation Report" in report
        assert "Total Households: 100" in report
        assert "RuleSpec execution mode: compiled_example" in report
        assert "PolicyEngine execution mode: policyengine_household" in report
        assert "EITC Comparison:" in report
        assert "Matches:" in report

    def test_detailed_report_with_mismatches(self):
        results = self._make_results(with_mismatches=True)
        report = results.detailed_report()
        assert "Worst mismatches:" in report
        # Should show up to 5 worst mismatches
        assert "HH" in report

    def test_detailed_report_skipped_variable(self):
        """Variables with 0 total compared show 'Skipped'."""
        results = self._make_results()
        report = results.detailed_report()
        assert "Skipped" in report  # snap has 0 matches and 0 mismatches

    def test_save_report(self, tmp_path):
        results = self._make_results(with_mismatches=True, with_data=True)
        results.save_report(tmp_path / "output")
        assert (tmp_path / "output" / "validation_report.txt").exists()
        assert (tmp_path / "output" / "validation_data.csv").exists()
        assert (tmp_path / "output" / "eitc_mismatches.csv").exists()

    def test_save_report_no_full_data(self, tmp_path):
        results = self._make_results(with_mismatches=False, with_data=False)
        results.save_report(tmp_path / "output2")
        assert (tmp_path / "output2" / "validation_report.txt").exists()
        # No data CSV or mismatch CSVs
        assert not (tmp_path / "output2" / "validation_data.csv").exists()

    def test_save_report_creates_dirs(self, tmp_path):
        results = self._make_results()
        nested = tmp_path / "deep" / "nested" / "dir"
        results.save_report(nested)
        assert (nested / "validation_report.txt").exists()


class TestValidateFunction:
    """Test the validate() convenience function."""

    def test_validate_sample(self):
        from src.rulespec_compile.validation.comparator import validate

        # Setup mock data
        mock_df = pd.DataFrame(
            {
                "household_id": [1, 2],
                "earned_income": [10000, 20000],
                "agi": [10000, 20000],
                "n_children": [1, 2],
                "is_joint": [False, True],
                "household_size": [2, 4],
                "gross_monthly_income": [833, 1667],
                "weight": [1.0, 1.0],
            }
        )

        results_df = mock_df.copy()
        results_df["rulespec_eitc"] = [100, 200]
        results_df["pe_eitc"] = [100, 200]
        results_df["rulespec_ctc"] = [100, 200]
        results_df["pe_ctc"] = [100, 200]
        results_df["rulespec_actc"] = [50, 100]
        results_df["pe_actc"] = [50, 100]
        results_df["state_code"] = ["CA", "NY"]
        results_df.attrs["rulespec_execution_mode"] = "compiled_example"
        results_df.attrs["policyengine_execution_mode"] = "policyengine_household"

        with (
            patch(
                "src.rulespec_compile.validation.cps_loader.load_cps_data",
                return_value=mock_df,
            ),
            patch(
                "src.rulespec_compile.validation.runners.run_both",
                return_value=results_df,
            ),
        ):
            results = validate(source="policyengine", year=2025, sample_size=2)
            assert results.total_households == 2
            assert results.rulespec_execution_mode == "compiled_example"

    def test_validate_with_output_dir(self, tmp_path):
        from src.rulespec_compile.validation.comparator import validate

        mock_df = pd.DataFrame(
            {
                "household_id": [1],
                "earned_income": [10000],
                "agi": [10000],
                "n_children": [1],
                "is_joint": [False],
                "household_size": [2],
                "gross_monthly_income": [833],
                "weight": [1.0],
            }
        )

        results_df = mock_df.copy()
        results_df["rulespec_eitc"] = [100]
        results_df["pe_eitc"] = [100]
        results_df["rulespec_ctc"] = [100]
        results_df["pe_ctc"] = [100]
        results_df["rulespec_actc"] = [50]
        results_df["pe_actc"] = [50]
        results_df["state_code"] = ["CA"]

        with (
            patch(
                "src.rulespec_compile.validation.cps_loader.load_cps_data",
                return_value=mock_df,
            ),
            patch(
                "src.rulespec_compile.validation.runners.run_both",
                return_value=results_df,
            ),
        ):
            output_dir = str(tmp_path / "results")
            validate(output_dir=output_dir)
            assert (tmp_path / "results" / "validation_report.txt").exists()


class TestValidateFullFunction:
    """Test the validate_full() convenience function."""

    def test_validate_full(self):
        from src.rulespec_compile.validation.comparator import validate_full

        results_df = pd.DataFrame(
            {
                "household_id": [1, 2],
                "rulespec_eitc": [100, 200],
                "pe_eitc": [100, 200],
                "rulespec_ctc": [100, 200],
                "pe_ctc": [100, 200],
                "rulespec_actc": [50, 100],
                "pe_actc": [50, 100],
                "state_code": ["CA", "NY"],
                "weight": [1.0, 1.0],
            }
        )
        results_df.attrs["rulespec_execution_mode"] = "compiled_batch"
        results_df.attrs["policyengine_execution_mode"] = "policyengine_microsim"

        with patch(
            "src.rulespec_compile.validation.runners.run_both_vectorized",
            return_value=results_df,
        ):
            results = validate_full(year=2025)
            assert results.total_households == 2
            assert results.rulespec_execution_mode == "compiled_batch"

    def test_validate_full_with_output_dir(self, tmp_path):
        from src.rulespec_compile.validation.comparator import validate_full

        results_df = pd.DataFrame(
            {
                "household_id": [1],
                "rulespec_eitc": [100],
                "pe_eitc": [100],
                "rulespec_ctc": [100],
                "pe_ctc": [100],
                "rulespec_actc": [50],
                "pe_actc": [50],
                "state_code": ["CA"],
                "weight": [1.0],
            }
        )

        with patch(
            "src.rulespec_compile.validation.runners.run_both_vectorized",
            return_value=results_df,
        ):
            output_dir = str(tmp_path / "full_results")
            validate_full(output_dir=output_dir)
            assert (tmp_path / "full_results" / "validation_report.txt").exists()


# ============================================================
# validation/cps_loader.py
# ============================================================


class TestCPSHousehold:
    """Test CPSHousehold dataclass."""

    def test_create_household(self):
        from src.rulespec_compile.validation.cps_loader import CPSHousehold

        hh = CPSHousehold(
            household_id=1,
            year=2025,
            state_code="CA",
            earned_income=50000,
            agi=55000,
            n_children=2,
            is_joint=True,
            household_size=4,
            gross_monthly_income=4583.33,
        )
        assert hh.household_id == 1
        assert hh.weight == 1.0  # default

    def test_custom_weight(self):
        from src.rulespec_compile.validation.cps_loader import CPSHousehold

        hh = CPSHousehold(
            household_id=1,
            year=2025,
            state_code="NY",
            earned_income=30000,
            agi=30000,
            n_children=1,
            is_joint=False,
            household_size=2,
            gross_monthly_income=2500,
            weight=3.5,
        )
        assert hh.weight == 3.5


class TestFipsToStateCode:
    """Test FIPS code to state code conversion."""

    def test_known_state(self):
        from src.rulespec_compile.validation.cps_loader import fips_to_state_code

        assert fips_to_state_code(6) == "CA"
        assert fips_to_state_code(36) == "NY"
        assert fips_to_state_code(48) == "TX"

    def test_unknown_fips(self):
        from src.rulespec_compile.validation.cps_loader import fips_to_state_code

        assert fips_to_state_code(999) == "US"

    def test_zero_fips(self):
        from src.rulespec_compile.validation.cps_loader import fips_to_state_code

        assert fips_to_state_code(0) == "US"


class TestLoadCPSFromPolicyEngine:
    """Test load_cps_from_policyengine with mocked PE."""

    def test_import_error_message(self):
        """Shows helpful error when policyengine-us not installed."""
        from src.rulespec_compile.validation.cps_loader import (
            load_cps_from_policyengine,
        )

        with patch.dict(sys.modules, {"policyengine_us": None}):
            with pytest.raises(ImportError, match="policyengine-us"):
                load_cps_from_policyengine()

    def test_loads_and_builds_records(self):
        """Loads data from PE Microsimulation and builds records."""
        # Create mock Microsimulation
        mock_sim = MagicMock()

        # Setup return values for calculate calls
        def calc_side_effect(var, year):
            values = {
                "tax_unit_id": pd.Series([1, 1, 2]),
                "spm_unit_id": pd.Series([10, 10, 20]),
                "tax_unit_earned_income": pd.Series([30000, 30000, 50000]),
                "adjusted_gross_income": pd.Series([35000, 35000, 55000]),
                "tax_unit_children": pd.Series([1, 1, 2]),
                "filing_status": pd.Series(["JOINT", "JOINT", "SINGLE"]),
                "spm_unit_gross_income": pd.Series([40000, 40000, 60000]),
                "state_code_str": pd.Series(["CA", "CA", "NY"]),
                "tax_unit_weight": pd.Series([1.5, 1.5, 2.0]),
            }
            return values.get(var, pd.Series([0, 0, 0]))

        mock_sim.calculate.side_effect = calc_side_effect

        # Patch the import and constructor
        mock_pe_mod = MagicMock()
        mock_pe_mod.Microsimulation.return_value = mock_sim

        with patch.dict(sys.modules, {"policyengine_us": mock_pe_mod}):
            # Need to reimport to pick up the mocked module
            import importlib

            import src.rulespec_compile.validation.cps_loader as cps_mod

            importlib.reload(cps_mod)
            df = cps_mod.load_cps_from_policyengine(year=2025)
            assert len(df) == 2
            assert "household_id" in df.columns
            assert "earned_income" in df.columns
            assert "is_joint" in df.columns

    def test_loads_with_sample_size(self):
        """Can subsample loaded data."""
        mock_sim = MagicMock()

        def calc_side_effect(var, year):
            n = 5
            values = {
                "tax_unit_id": pd.Series(list(range(n))),
                "spm_unit_id": pd.Series(list(range(n))),
                "tax_unit_earned_income": pd.Series([30000] * n),
                "adjusted_gross_income": pd.Series([35000] * n),
                "tax_unit_children": pd.Series([1] * n),
                "filing_status": pd.Series(["SINGLE"] * n),
                "spm_unit_gross_income": pd.Series([40000] * n),
                "state_code_str": pd.Series(["CA"] * n),
                "tax_unit_weight": pd.Series([1.0] * n),
            }
            return values.get(var, pd.Series([0] * n))

        mock_sim.calculate.side_effect = calc_side_effect

        mock_pe_mod = MagicMock()
        mock_pe_mod.Microsimulation.return_value = mock_sim

        with patch.dict(sys.modules, {"policyengine_us": mock_pe_mod}):
            import importlib

            import src.rulespec_compile.validation.cps_loader as cps_mod

            importlib.reload(cps_mod)
            df = cps_mod.load_cps_from_policyengine(year=2025, sample_size=2)
            assert len(df) == 2


class TestLoadCPSFromCSV:
    """Test load_cps_from_csv."""

    def test_loads_csv_file(self, tmp_path):
        from src.rulespec_compile.validation.cps_loader import load_cps_from_csv

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "taxsimid,state,mstat,depx,pwages,psemp,swages,ssemp,"
            "dividends,intrec,stcg,ltcg,pensions,gssi\n"
            "1,6,2,2,30000,0,20000,0,500,100,0,0,0,5000\n"
            "2,36,1,0,40000,5000,0,0,0,0,1000,2000,0,0\n"
        )
        df = load_cps_from_csv(str(csv_file))
        assert len(df) == 2
        assert df.loc[0, "state_code"] == "CA"
        assert df.loc[1, "state_code"] == "NY"
        assert df.loc[0, "is_joint"] == True  # noqa: E712
        assert df.loc[1, "is_joint"] == False  # noqa: E712
        assert df.loc[0, "n_children"] == 2
        assert df.loc[0, "household_size"] == 4  # 1 + 1 (joint) + 2 children

    def test_loads_csv_with_sample(self, tmp_path):
        from src.rulespec_compile.validation.cps_loader import load_cps_from_csv

        csv_file = tmp_path / "test.csv"
        lines = [
            "taxsimid,state,mstat,depx,pwages,psemp,swages,ssemp,"
            "dividends,intrec,stcg,ltcg,pensions,gssi"
        ]
        for i in range(10):
            lines.append(f"{i},6,1,0,{30000 + i * 1000},0,0,0,0,0,0,0,0,0")
        csv_file.write_text("\n".join(lines))
        df = load_cps_from_csv(str(csv_file), sample_size=3)
        assert len(df) == 3

    def test_csv_missing_columns_default_to_zero(self, tmp_path):
        """Missing optional columns default to 0."""
        from src.rulespec_compile.validation.cps_loader import load_cps_from_csv

        csv_file = tmp_path / "minimal.csv"
        csv_file.write_text("taxsimid,pwages\n1,30000\n")
        df = load_cps_from_csv(str(csv_file))
        assert len(df) == 1
        assert df.loc[0, "n_children"] == 0
        assert df.loc[0, "is_joint"] == False  # noqa: E712


class TestLoadCPSData:
    """Test load_cps_data dispatch function."""

    def test_unknown_source_raises(self):
        from src.rulespec_compile.validation.cps_loader import load_cps_data

        with pytest.raises(ValueError, match="Unknown source"):
            load_cps_data(source="unknown")

    def test_csv_without_path_raises(self):
        from src.rulespec_compile.validation.cps_loader import load_cps_data

        with pytest.raises(ValueError, match="csv_path required"):
            load_cps_data(source="csv")

    def test_csv_source_delegates(self, tmp_path):
        from src.rulespec_compile.validation.cps_loader import load_cps_data

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "taxsimid,state,mstat,depx,pwages,psemp,swages,ssemp,"
            "dividends,intrec,stcg,ltcg,pensions,gssi\n"
            "1,6,1,0,30000,0,0,0,0,0,0,0,0,0\n"
        )
        df = load_cps_data(source="csv", csv_path=str(csv_file))
        assert len(df) == 1

    @patch("src.rulespec_compile.validation.cps_loader.load_cps_from_policyengine")
    def test_policyengine_source_delegates(self, mock_load_pe):
        from src.rulespec_compile.validation.cps_loader import load_cps_data

        mock_load_pe.return_value = pd.DataFrame({"household_id": [1]})
        df = load_cps_data(source="policyengine")
        assert len(df) == 1
        mock_load_pe.assert_called_once()


class TestIterateHouseholds:
    """Test iterate_households generator."""

    def test_yields_cps_household_objects(self):
        from src.rulespec_compile.validation.cps_loader import iterate_households

        df = pd.DataFrame(
            {
                "household_id": [1, 2],
                "year": [2025, 2025],
                "state_code": ["CA", "NY"],
                "earned_income": [30000, 40000],
                "agi": [35000, 45000],
                "n_children": [1, 2],
                "is_joint": [False, True],
                "household_size": [2, 4],
                "gross_monthly_income": [2500, 3750],
                "weight": [1.0, 2.0],
            }
        )
        households = list(iterate_households(df))
        assert len(households) == 2
        assert households[0].household_id == 1
        assert households[0].state_code == "CA"
        assert households[1].is_joint is True


# ============================================================
# validation/runners.py
# ============================================================


class TestRunRuleSpec:
    """Test run_rulespec function."""

    @patch(
        "src.rulespec_compile.validation.runners._load_compiled_validation_calculators"
    )
    def test_run_rulespec_uses_compiled_examples(self, mock_load):
        """run_rulespec sources values from the compiled example calculators."""
        from src.rulespec_compile.validation.runners import (
            CompiledValidationCalculators,
            run_rulespec,
        )

        mock_load.return_value = CompiledValidationCalculators(
            eitc=lambda **kwargs: {"eitc": 111},
            ctc=lambda **kwargs: {"ctc": 222, "actc": 333},
            snap=lambda **kwargs: {"snap_benefit": 444},
        )
        df = pd.DataFrame(
            {
                "household_id": [1],
                "earned_income": [15000],
                "agi": [15000],
                "n_children": [1],
                "is_joint": [False],
                "household_size": [2],
                "gross_monthly_income": [1250],
            }
        )

        results = run_rulespec(df, show_progress=False)

        assert results.loc[0, "rulespec_eitc"] == 111
        assert results.loc[0, "rulespec_ctc"] == 222
        assert results.loc[0, "rulespec_actc"] == 333
        assert results.loc[0, "rulespec_snap"] == 444

    def test_run_rulespec_calculates_all(self):
        from src.rulespec_compile.validation.runners import run_rulespec

        df = pd.DataFrame(
            {
                "household_id": [1, 2],
                "earned_income": [15000, 0],
                "agi": [15000, 0],
                "n_children": [1, 0],
                "is_joint": [False, False],
                "household_size": [2, 1],
                "gross_monthly_income": [1250, 0],
            }
        )
        results = run_rulespec(df, show_progress=False)
        assert "rulespec_eitc" in results.columns
        assert "rulespec_ctc" in results.columns
        assert "rulespec_actc" in results.columns
        assert "rulespec_snap" in results.columns
        assert len(results) == 2
        assert results.attrs["rulespec_execution_mode"] == "compiled_example"

    def test_run_rulespec_matches_reference_examples(self):
        """Compiled sample validation calculators stay aligned with references."""
        from src.rulespec_compile.calculators import (
            calculate_actc,
            calculate_ctc,
            calculate_eitc,
            calculate_snap_benefit,
        )
        from src.rulespec_compile.validation.runners import run_rulespec

        df = pd.DataFrame(
            {
                "household_id": [1],
                "earned_income": [15000],
                "agi": [15000],
                "n_children": [1],
                "is_joint": [False],
                "household_size": [2],
                "gross_monthly_income": [1250],
            }
        )

        results = run_rulespec(df, show_progress=False)

        assert (
            results.loc[0, "rulespec_eitc"]
            == calculate_eitc(
                earned_income=15000,
                agi=15000,
                n_children=1,
                is_joint=False,
            ).eitc
        )
        assert (
            results.loc[0, "rulespec_ctc"]
            == calculate_ctc(
                n_qualifying_children=1,
                agi=15000,
                is_joint=False,
            ).ctc
        )
        assert (
            results.loc[0, "rulespec_actc"]
            == calculate_actc(
                n_qualifying_children=1,
                earned_income=15000,
            ).actc
        )
        assert (
            results.loc[0, "rulespec_snap"]
            == calculate_snap_benefit(
                household_size=2,
                gross_income=1250,
            ).benefit
        )

    def test_run_rulespec_with_progress(self):
        """run_rulespec with show_progress=True uses tqdm."""
        from src.rulespec_compile.validation.runners import run_rulespec

        df = pd.DataFrame(
            {
                "household_id": [1],
                "earned_income": [15000],
                "agi": [15000],
                "n_children": [1],
                "is_joint": [False],
                "household_size": [2],
                "gross_monthly_income": [1250],
            }
        )
        results = run_rulespec(df, show_progress=True)
        assert len(results) == 1


class TestRunPolicyEngine:
    """Test run_policyengine function with mocked PE."""

    def test_import_error(self):
        """Shows helpful error when policyengine-us not installed."""
        from src.rulespec_compile.validation.runners import run_policyengine

        with patch.dict(sys.modules, {"policyengine_us": None}):
            with pytest.raises(ImportError, match="policyengine-us"):
                run_policyengine(pd.DataFrame())

    @patch("src.rulespec_compile.validation.runners._run_single_pe_simulation")
    def test_run_policyengine_success(self, mock_sim):
        """Runs PE simulation for each household."""
        from src.rulespec_compile.validation.runners import run_policyengine

        mock_sim.return_value = {
            "pe_eitc": 500.0,
            "pe_ctc": 2200.0,
            "pe_actc": 1600.0,
            "pe_snap": 200.0,
        }

        # Need policyengine_us to be importable but we don't actually use it
        mock_pe = MagicMock()
        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            df = pd.DataFrame(
                {
                    "household_id": [1, 2],
                    "earned_income": [15000, 20000],
                    "agi": [15000, 20000],
                    "n_children": [1, 2],
                    "is_joint": [False, True],
                    "household_size": [2, 4],
                    "gross_monthly_income": [1250, 1667],
                }
            )
            results = run_policyengine(df, show_progress=False)
            assert len(results) == 2
            assert results.loc[0, "pe_eitc"] == 500.0
            assert (
                results.attrs["policyengine_execution_mode"] == "policyengine_household"
            )

    @patch("src.rulespec_compile.validation.runners._run_single_pe_simulation")
    def test_run_policyengine_handles_errors(self, mock_sim):
        """Continues with NaN when simulation fails."""
        from src.rulespec_compile.validation.runners import run_policyengine

        mock_sim.side_effect = Exception("Simulation failed")

        mock_pe = MagicMock()
        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            df = pd.DataFrame(
                {
                    "household_id": [1],
                    "earned_income": [15000],
                    "agi": [15000],
                    "n_children": [1],
                    "is_joint": [False],
                    "household_size": [2],
                    "gross_monthly_income": [1250],
                }
            )
            results = run_policyengine(df, show_progress=False)
            assert len(results) == 1
            assert np.isnan(results.loc[0, "pe_eitc"])
            assert "pe_error" in results.columns

    @patch("src.rulespec_compile.validation.runners._run_single_pe_simulation")
    def test_run_policyengine_with_progress(self, mock_sim):
        """run_policyengine with show_progress=True uses tqdm."""
        from src.rulespec_compile.validation.runners import run_policyengine

        mock_sim.return_value = {
            "pe_eitc": 500.0,
            "pe_ctc": 2200.0,
            "pe_actc": 1600.0,
            "pe_snap": 200.0,
        }

        mock_pe = MagicMock()
        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            df = pd.DataFrame(
                {
                    "household_id": [1],
                    "earned_income": [15000],
                    "agi": [15000],
                    "n_children": [1],
                    "is_joint": [False],
                    "household_size": [2],
                    "gross_monthly_income": [1250],
                }
            )
            results = run_policyengine(df, show_progress=True)
            assert len(results) == 1


class TestRunSinglePESimulation:
    """Test _run_single_pe_simulation with mocked PE."""

    def test_single_simulation(self):
        from src.rulespec_compile.validation.runners import _run_single_pe_simulation

        mock_sim_instance = MagicMock()
        mock_sim_instance.calculate.side_effect = lambda var, year: np.array([100.0])

        mock_simulation_cls = MagicMock(return_value=mock_sim_instance)
        mock_pe = MagicMock()
        mock_pe.Simulation = mock_simulation_cls

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            row = pd.Series(
                {
                    "earned_income": 15000,
                    "is_joint": False,
                    "n_children": 1,
                    "household_size": 2,
                    "state_code": "CA",
                }
            )
            result = _run_single_pe_simulation(row, 2025)
            assert "pe_eitc" in result
            assert "pe_snap" in result

    def test_simulation_with_joint(self):
        """Joint filer includes spouse in simulation."""
        from src.rulespec_compile.validation.runners import _run_single_pe_simulation

        mock_sim_instance = MagicMock()
        mock_sim_instance.calculate.side_effect = lambda var, year: np.array([200.0])

        mock_simulation_cls = MagicMock(return_value=mock_sim_instance)
        mock_pe = MagicMock()
        mock_pe.Simulation = mock_simulation_cls

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            row = pd.Series(
                {
                    "earned_income": 30000,
                    "is_joint": True,
                    "n_children": 2,
                    "household_size": 4,
                    "state_code": "NY",
                }
            )
            _run_single_pe_simulation(row, 2025)
            # Check that situation includes spouse
            call_args = mock_simulation_cls.call_args
            situation = call_args[1]["situation"]
            assert "spouse" in situation["people"]

    def test_simulation_with_extra_members(self):
        """Extra household members (beyond tax unit) added for SNAP."""
        from src.rulespec_compile.validation.runners import _run_single_pe_simulation

        mock_sim_instance = MagicMock()
        mock_sim_instance.calculate.side_effect = lambda var, year: np.array([100.0])

        mock_simulation_cls = MagicMock(return_value=mock_sim_instance)
        mock_pe = MagicMock()
        mock_pe.Simulation = mock_simulation_cls

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            row = pd.Series(
                {
                    "earned_income": 15000,
                    "is_joint": False,
                    "n_children": 0,
                    "household_size": 3,  # 1 adult + 2 extra
                    "state_code": "CA",
                }
            )
            _run_single_pe_simulation(row, 2025)
            call_args = mock_simulation_cls.call_args
            situation = call_args[1]["situation"]
            # Should have adult + 2 extra members
            assert "extra_0" in situation["people"]
            assert "extra_1" in situation["people"]


class TestRunPolicyEngineHousehold:
    """Test the single-household PolicyEngine bridge helper."""

    def test_household_helper_annualizes_monthly_snap_income(self):
        from src.rulespec_compile.validation.runners import run_policyengine_household

        mock_sim_instance = MagicMock()
        mock_sim_instance.calculate.side_effect = lambda var, year: np.array([100.0])

        mock_simulation_cls = MagicMock(return_value=mock_sim_instance)
        mock_pe = MagicMock()
        mock_pe.Simulation = mock_simulation_cls

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            result = run_policyengine_household(
                {
                    "gross_income": 2000,
                    "household_size": 4,
                    "state_code": "CA",
                }
            )

        assert "pe_snap" in result
        situation = mock_simulation_cls.call_args.kwargs["situation"]
        assert situation["people"]["adult"]["employment_income"][2025] == 24000
        assert situation["households"]["household"]["state_code"][2025] == "CA"


class TestRunBoth:
    """Test run_both function."""

    @patch("src.rulespec_compile.validation.runners.run_policyengine")
    @patch("src.rulespec_compile.validation.runners.run_rulespec")
    def test_run_both_merges(self, mock_rulespec, mock_pe):
        from src.rulespec_compile.validation.runners import run_both

        df = pd.DataFrame(
            {
                "household_id": [1, 2],
                "earned_income": [15000, 20000],
            }
        )
        mock_rulespec.return_value = pd.DataFrame(
            {
                "household_id": [1, 2],
                "rulespec_eitc": [100, 200],
            }
        )
        mock_rulespec.return_value.attrs["rulespec_execution_mode"] = "compiled_example"
        mock_pe.return_value = pd.DataFrame(
            {
                "household_id": [1, 2],
                "pe_eitc": [100, 200],
            }
        )
        mock_pe.return_value.attrs["policyengine_execution_mode"] = (
            "policyengine_household"
        )
        result = run_both(df)
        assert "rulespec_eitc" in result.columns
        assert "pe_eitc" in result.columns
        assert len(result) == 2
        assert result.attrs["rulespec_execution_mode"] == "compiled_example"
        assert result.attrs["policyengine_execution_mode"] == "policyengine_household"


class TestRunRuleSpecVectorized:
    """Test run_rulespec_vectorized function."""

    def test_vectorized_calculation(self):
        from src.rulespec_compile.validation.runners import run_rulespec_vectorized

        df = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "earned_income": [15000, 0, 50000],
                "agi": [15000, 0, 50000],
                "n_children": [1, 0, 3],
                "is_joint": [False, False, True],
                "household_size": [2, 1, 5],
                "gross_monthly_income": [1250, 0, 4167],
            }
        )
        results = run_rulespec_vectorized(df)
        assert "rulespec_eitc" in results.columns
        assert "rulespec_ctc" in results.columns
        assert "rulespec_actc" in results.columns
        assert "rulespec_snap" in results.columns
        assert len(results) == 3
        assert results.attrs["rulespec_execution_mode"] == "compiled_batch"
        # Zero income should have zero EITC
        assert results.loc[1, "rulespec_eitc"] == 0


class TestRunPEMicrosim:
    """Test run_policyengine_microsim with mocked PE."""

    def test_import_error(self):
        from src.rulespec_compile.validation.runners import run_policyengine_microsim

        with patch.dict(sys.modules, {"policyengine_us": None}):
            with pytest.raises(ImportError, match="policyengine-us"):
                run_policyengine_microsim()

    def test_microsim_runs(self):
        from src.rulespec_compile.validation.runners import run_policyengine_microsim

        mock_sim = MagicMock()

        def calc_side_effect(var, year):
            if var == "tax_unit_id":
                return pd.Series([1, 1, 2])
            elif var == "eitc":
                return pd.Series([500, 500, 300])
            elif var == "ctc":
                return pd.Series([2200, 2200, 0])
            elif var == "refundable_ctc":
                return pd.Series([1600, 1600, 0])
            elif var == "snap":
                return pd.Series([2400, 2400, 0])
            return pd.Series([0, 0, 0])

        mock_sim.calculate.side_effect = calc_side_effect

        mock_pe = MagicMock()
        mock_pe.Microsimulation.return_value = mock_sim

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            results = run_policyengine_microsim(year=2025)
            assert len(results) == 2
            assert "pe_eitc" in results.columns


class TestRunBothVectorized:
    """Test run_both_vectorized with mocked PE."""

    def test_import_error(self):
        from src.rulespec_compile.validation.runners import run_both_vectorized

        with patch.dict(sys.modules, {"policyengine_us": None}):
            with pytest.raises(ImportError, match="policyengine-us"):
                run_both_vectorized()

    def test_full_vectorized_pipeline(self):
        from src.rulespec_compile.validation.runners import run_both_vectorized

        mock_sim = MagicMock()

        def calc_side_effect(var, year):
            if var == "tax_unit_id":
                return pd.Series([1, 2])
            elif var == "spm_unit_id":
                return pd.Series([10, 20])
            elif var == "tax_unit_earned_income":
                return pd.Series([15000, 30000])
            elif var == "adjusted_gross_income":
                return pd.Series([15000, 30000])
            elif var == "tax_unit_children":
                return pd.Series([1, 2])
            elif var == "filing_status":
                return pd.Series(["SINGLE", "JOINT"])
            elif var == "tax_unit_size":
                return pd.Series([2, 4])
            elif var == "eitc":
                return pd.Series([500, 300])
            elif var == "ctc":
                return pd.Series([2200, 4400])
            elif var == "refundable_ctc":
                return pd.Series([1600, 1800])
            elif var == "snap":
                return pd.Series([2400, 1200])  # Annual
            elif var == "spm_unit_size":
                return pd.Series([2, 4])
            elif var == "spm_unit_net_income":
                return pd.Series([12000, 24000])
            return pd.Series([0, 0])

        mock_sim.calculate.side_effect = calc_side_effect

        mock_pe = MagicMock()
        mock_pe.Microsimulation.return_value = mock_sim

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            results = run_both_vectorized(year=2025)
            assert "rulespec_eitc" in results.columns
            assert "pe_eitc" in results.columns
            assert "spm_snap_data" in results.attrs
            spm_df = results.attrs["spm_snap_data"]
            assert "rulespec_snap" in spm_df.columns
            assert "pe_snap" in spm_df.columns


# ============================================================
# validation/cli.py
# ============================================================


class TestValidationCLI:
    """Test validation CLI."""

    @patch("src.rulespec_compile.validation.cli.validate_full")
    def test_full_mode(self, mock_validate_full):
        """Full mode calls validate_full."""
        from src.rulespec_compile.validation.cli import main
        from src.rulespec_compile.validation.comparator import ComparisonResults

        mock_validate_full.return_value = ComparisonResults(
            total_households=100,
            variables_compared=["eitc"],
            matches={"eitc": 100},
            mismatches={"eitc": []},
            match_rates={"eitc": 100.0},
            config=MagicMock(),
        )

        with patch("sys.argv", ["rulespec-validate", "--mode", "full"]):
            main()
            mock_validate_full.assert_called_once()

    @patch("src.rulespec_compile.validation.cli.validate")
    def test_sample_mode(self, mock_validate):
        """Sample mode calls validate."""
        from src.rulespec_compile.validation.cli import main
        from src.rulespec_compile.validation.comparator import ComparisonResults

        mock_validate.return_value = ComparisonResults(
            total_households=10,
            variables_compared=["eitc"],
            matches={"eitc": 10},
            mismatches={"eitc": []},
            match_rates={"eitc": 100.0},
            config=MagicMock(),
        )

        with patch(
            "sys.argv",
            ["rulespec-validate", "--mode", "sample", "--source", "policyengine"],
        ):
            main()
            mock_validate.assert_called_once()

    def test_sample_csv_without_path_errors(self):
        """Sample mode with csv source but no path errors."""
        from src.rulespec_compile.validation.cli import main

        with patch(
            "sys.argv",
            ["rulespec-validate", "--mode", "sample", "--source", "csv"],
        ):
            with pytest.raises(SystemExit):
                main()

    @patch("src.rulespec_compile.validation.cli.validate_full")
    def test_low_match_rate_exits_1(self, mock_validate_full):
        """Low match rate exits with code 1."""
        from src.rulespec_compile.validation.cli import main
        from src.rulespec_compile.validation.comparator import (
            ComparisonResults,
            MismatchRecord,
        )

        mismatches = [
            MismatchRecord(
                household_id=i,
                variable="eitc",
                rulespec_value=100,
                policyengine_value=500,
                difference=-400,
            )
            for i in range(80)
        ]

        mock_validate_full.return_value = ComparisonResults(
            total_households=100,
            variables_compared=["eitc"],
            matches={"eitc": 20},
            mismatches={"eitc": mismatches},
            match_rates={"eitc": 20.0},
            config=MagicMock(),
        )

        with patch("sys.argv", ["rulespec-validate", "--mode", "full"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("src.rulespec_compile.validation.cli.validate_full")
    def test_import_error_exits_1(self, mock_validate_full):
        """ImportError shows install instructions and exits 1."""
        from src.rulespec_compile.validation.cli import main

        mock_validate_full.side_effect = ImportError(
            "No module named 'policyengine_us'"
        )

        with patch("sys.argv", ["rulespec-validate", "--mode", "full"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("src.rulespec_compile.validation.cli.validate_full")
    def test_generic_error_exits_1(self, mock_validate_full):
        """Generic errors show traceback and exit 1."""
        from src.rulespec_compile.validation.cli import main

        mock_validate_full.side_effect = RuntimeError("Something went wrong")

        with patch("sys.argv", ["rulespec-validate", "--mode", "full"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("src.rulespec_compile.validation.cli.validate")
    def test_sample_csv_with_path(self, mock_validate):
        """Sample mode with csv source and path works."""
        from src.rulespec_compile.validation.cli import main
        from src.rulespec_compile.validation.comparator import ComparisonResults

        mock_validate.return_value = ComparisonResults(
            total_households=5,
            variables_compared=["eitc"],
            matches={"eitc": 5},
            mismatches={"eitc": []},
            match_rates={"eitc": 100.0},
            config=MagicMock(),
        )

        with patch(
            "sys.argv",
            [
                "rulespec-validate",
                "--mode",
                "sample",
                "--source",
                "csv",
                "--csv-path",
                "data.csv",
            ],
        ):
            main()
            mock_validate.assert_called_once()
            call_kwargs = mock_validate.call_args
            assert (
                call_kwargs[1]["csv_path"] == "data.csv"
                or call_kwargs.kwargs.get("csv_path") == "data.csv"
            )

    @patch("src.rulespec_compile.validation.cli.validate_full")
    def test_tolerance_args_passed(self, mock_validate_full):
        """Custom tolerance args are passed to config."""
        from src.rulespec_compile.validation.cli import main
        from src.rulespec_compile.validation.comparator import ComparisonResults

        mock_validate_full.return_value = ComparisonResults(
            total_households=100,
            variables_compared=["eitc"],
            matches={"eitc": 100},
            mismatches={"eitc": []},
            match_rates={"eitc": 100.0},
            config=MagicMock(),
        )

        with patch(
            "sys.argv",
            [
                "rulespec-validate",
                "--mode",
                "full",
                "--eitc-tolerance",
                "5.0",
                "--ctc-tolerance",
                "3.0",
                "--snap-tolerance",
                "100.0",
            ],
        ):
            main()
            call_kwargs = mock_validate_full.call_args
            config = call_kwargs[1].get("config") or call_kwargs.kwargs.get("config")
            assert config.eitc_tolerance == 5.0
            assert config.ctc_tolerance == 3.0
            assert config.snap_tolerance == 100.0

    @patch("src.rulespec_compile.validation.cli.validate_full")
    def test_no_valid_rates_doesnt_crash(self, mock_validate_full):
        """Empty valid_rates (all variables have 0 data) doesn't crash."""
        from src.rulespec_compile.validation.cli import main
        from src.rulespec_compile.validation.comparator import ComparisonResults

        mock_validate_full.return_value = ComparisonResults(
            total_households=0,
            variables_compared=["eitc"],
            matches={"eitc": 0},
            mismatches={"eitc": []},
            match_rates={"eitc": 0.0},
            config=MagicMock(),
        )

        with patch("sys.argv", ["rulespec-validate", "--mode", "full"]):
            # Should not crash - min_match_rate defaults to 100
            main()

    @patch("src.rulespec_compile.validation.cli.validate")
    def test_sample_mode_with_sample_size(self, mock_validate):
        """Sample size arg is passed through."""
        from src.rulespec_compile.validation.cli import main
        from src.rulespec_compile.validation.comparator import ComparisonResults

        mock_validate.return_value = ComparisonResults(
            total_households=50,
            variables_compared=["eitc"],
            matches={"eitc": 50},
            mismatches={"eitc": []},
            match_rates={"eitc": 100.0},
            config=MagicMock(),
        )

        with patch(
            "sys.argv",
            ["rulespec-validate", "--mode", "sample", "--sample-size", "50"],
        ):
            main()
            call_kwargs = mock_validate.call_args
            assert (
                call_kwargs[1].get("sample_size") == 50
                or call_kwargs.kwargs.get("sample_size") == 50
            )

    @patch("src.rulespec_compile.validation.cli.validate")
    def test_sample_mode_with_output_dir(self, mock_validate):
        """Output dir arg is passed through."""
        from src.rulespec_compile.validation.cli import main
        from src.rulespec_compile.validation.comparator import ComparisonResults

        mock_validate.return_value = ComparisonResults(
            total_households=10,
            variables_compared=["eitc"],
            matches={"eitc": 10},
            mismatches={"eitc": []},
            match_rates={"eitc": 100.0},
            config=MagicMock(),
        )

        with patch(
            "sys.argv",
            [
                "rulespec-validate",
                "--mode",
                "sample",
                "--output-dir",
                "/tmp/out",
            ],
        ):
            main()
