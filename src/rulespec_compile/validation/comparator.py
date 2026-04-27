"""
Comparator: Compare RuleSpec results against PolicyEngine-US.

Generates detailed comparison reports with tolerance-based matching,
following the policyengine-taxsim pattern.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ComparisonConfig:
    """Configuration for validation comparison."""

    # Tolerances (absolute, in dollars or benefit units)
    eitc_tolerance: float = 1.0  # $1 for EITC
    ctc_tolerance: float = 1.0  # $1 for CTC
    actc_tolerance: float = 1.0  # $1 for ACTC
    snap_tolerance: float = 50.0  # $50 for SNAP (more deductions we don't model)

    # Column mappings
    id_col: str = "household_id"


@dataclass
class MismatchRecord:
    """Record of a calculation mismatch."""

    household_id: int
    variable: str
    rulespec_value: float
    policyengine_value: float
    difference: float
    pct_difference: Optional[float] = None
    state_code: Optional[str] = None
    weight: float = 1.0


@dataclass
class ComparisonResults:
    """Results from comparing RuleSpec vs PolicyEngine."""

    total_households: int
    variables_compared: List[str]
    matches: Dict[str, int]
    mismatches: Dict[str, List[MismatchRecord]]
    match_rates: Dict[str, float]
    config: ComparisonConfig
    full_data: Optional[pd.DataFrame] = None
    rulespec_execution_mode: str = "unknown"
    policyengine_execution_mode: str = "unknown"

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "total_households": self.total_households,
            "rulespec_execution_mode": self.rulespec_execution_mode,
            "policyengine_execution_mode": self.policyengine_execution_mode,
            "variables": {
                var: {
                    "matches": self.matches[var],
                    "mismatches": len(self.mismatches[var]),
                    "match_rate": self.match_rates[var],
                    "tolerance": getattr(self.config, f"{var}_tolerance"),
                }
                for var in self.variables_compared
            },
        }

    def detailed_report(self) -> str:
        """Generate detailed text report."""
        lines = [
            "=" * 70,
            "RuleSpec vs PolicyEngine-US Validation Report",
            "=" * 70,
            f"Total Households: {self.total_households:,}",
            f"RuleSpec execution mode: {self.rulespec_execution_mode}",
            f"PolicyEngine execution mode: {self.policyengine_execution_mode}",
            "",
        ]

        for var in self.variables_compared:
            total_compared = self.matches[var] + len(self.mismatches[var])
            if total_compared == 0:
                lines.extend(
                    [
                        f"{var.upper()} Comparison:",
                        "-" * 40,
                        "  Skipped (no valid data for comparison)",
                        "",
                    ]
                )
                continue

            tol = getattr(self.config, f"{var}_tolerance")
            lines.extend(
                [
                    f"{var.upper()} Comparison:",
                    "-" * 40,
                    f"  Matches:     {self.matches[var]:,}"
                    f" ({self.match_rates[var]:.2f}%)",
                    f"  Mismatches:  {len(self.mismatches[var]):,}",
                    f"  Tolerance:   ±${tol:.0f}",
                    "",
                ]
            )

            # Show worst mismatches
            if self.mismatches[var]:
                worst = sorted(
                    self.mismatches[var],
                    key=lambda m: abs(m.difference),
                    reverse=True,
                )[:5]
                lines.append("  Worst mismatches:")
                for m in worst:
                    lines.append(
                        f"    HH {m.household_id}: rulespec=${m.rulespec_value:.0f}, "
                        f"PE=${m.policyengine_value:.0f}, diff=${m.difference:.0f}"
                    )
                lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def save_report(self, output_dir: Path):
        """Save comparison results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save summary report
        report_path = output_dir / "validation_report.txt"
        report_path.write_text(self.detailed_report())
        print(f"Saved report to: {report_path}")

        # Save full comparison data
        if self.full_data is not None:
            data_path = output_dir / "validation_data.csv"
            self.full_data.to_csv(data_path, index=False)
            print(f"Saved full data to: {data_path}")

        # Save mismatches per variable
        for var in self.variables_compared:
            if self.mismatches[var]:
                mismatch_df = pd.DataFrame(
                    [
                        {
                            "household_id": m.household_id,
                            "rulespec_value": m.rulespec_value,
                            "policyengine_value": m.policyengine_value,
                            "difference": m.difference,
                            "pct_difference": m.pct_difference,
                            "state_code": m.state_code,
                            "weight": m.weight,
                        }
                        for m in self.mismatches[var]
                    ]
                )
                mismatch_path = output_dir / f"{var}_mismatches.csv"
                mismatch_df.to_csv(mismatch_path, index=False)
                print(f"Saved {var} mismatches to: {mismatch_path}")


class Comparator:
    """Compare RuleSpec vs PolicyEngine results."""

    VARIABLES = [
        ("eitc", "rulespec_eitc", "pe_eitc"),
        ("ctc", "rulespec_ctc", "pe_ctc"),
        ("actc", "rulespec_actc", "pe_actc"),
        ("snap", "rulespec_snap", "pe_snap"),
    ]

    def __init__(self, config: Optional[ComparisonConfig] = None):
        self.config = config or ComparisonConfig()

    def compare(self, df: pd.DataFrame) -> ComparisonResults:
        """
        Compare RuleSpec and PolicyEngine results.

        Args:
            df: DataFrame with both RuleSpec and PE results
                (output from runners.run_both)
                May have attrs["spm_snap_data"] for SNAP comparison

        Returns:
            ComparisonResults with match statistics and mismatches
        """
        total = len(df)
        matches = {}
        mismatches = {}
        match_rates = {}

        # Compare tax unit level variables (EITC, CTC, ACTC)
        for var_name, rulespec_col, pe_col in self.VARIABLES:
            if var_name == "snap":
                continue  # Handle separately below

            if rulespec_col not in df.columns or pe_col not in df.columns:
                continue

            tolerance = getattr(self.config, f"{var_name}_tolerance")
            var_matches, var_mismatches = self._compare_variable(
                df, var_name, rulespec_col, pe_col, tolerance
            )
            matches[var_name] = var_matches
            mismatches[var_name] = var_mismatches
            valid_count = var_matches + len(var_mismatches)
            match_rates[var_name] = (
                (var_matches / valid_count * 100) if valid_count > 0 else 0
            )

        # Compare SNAP at SPM unit level if available
        spm_df = df.attrs.get("spm_snap_data")
        if spm_df is not None and len(spm_df) > 0:
            snap_matches, snap_mismatches = self._compare_variable(
                spm_df,
                "snap",
                "rulespec_snap",
                "pe_snap",
                self.config.snap_tolerance,
                id_col="spm_unit_id",  # SPM uses different ID column
            )
            matches["snap"] = snap_matches
            mismatches["snap"] = snap_mismatches
            snap_valid = snap_matches + len(snap_mismatches)
            match_rates["snap"] = (
                (snap_matches / snap_valid * 100) if snap_valid > 0 else 0
            )
        else:
            # No SPM data available
            matches["snap"] = 0
            mismatches["snap"] = []
            match_rates["snap"] = 0

        return ComparisonResults(
            total_households=total,
            rulespec_execution_mode=df.attrs.get("rulespec_execution_mode", "unknown"),
            policyengine_execution_mode=df.attrs.get(
                "policyengine_execution_mode",
                "unknown",
            ),
            variables_compared=list(matches.keys()),
            matches=matches,
            mismatches=mismatches,
            match_rates=match_rates,
            config=self.config,
            full_data=df,
        )

    def _compare_variable(
        self,
        df: pd.DataFrame,
        var_name: str,
        rulespec_col: str,
        pe_col: str,
        tolerance: float,
        id_col: Optional[str] = None,
    ) -> tuple:
        """Compare a single variable."""
        mismatches = []
        id_col = id_col or self.config.id_col

        # Handle NaN in PE results (errors during calculation or missing data)
        valid_mask = ~df[pe_col].isna()
        df_valid = df[valid_mask]

        # If no valid data, return 0 matches (will be excluded from stats)
        if len(df_valid) == 0:
            return 0, []

        # Check which are within tolerance
        is_match = np.isclose(
            df_valid[rulespec_col],
            df_valid[pe_col],
            atol=tolerance,
            equal_nan=True,
        )

        match_count = int(is_match.sum())

        # Collect mismatches
        mismatch_rows = df_valid[~is_match]
        for _, row in mismatch_rows.iterrows():
            rulespec_val = row[rulespec_col]
            pe_val = row[pe_col]
            diff = rulespec_val - pe_val

            # Calculate percentage difference (avoid div by zero)
            pct_diff = None
            if pe_val != 0:
                pct_diff = (diff / pe_val) * 100

            mismatches.append(
                MismatchRecord(
                    household_id=row[id_col],
                    variable=var_name,
                    rulespec_value=rulespec_val,
                    policyengine_value=pe_val,
                    difference=diff,
                    pct_difference=pct_diff,
                    state_code=row.get("state_code"),
                    weight=row.get("weight", 1.0),
                )
            )

        return match_count, mismatches


def validate(
    source: str = "policyengine",
    year: int = 2025,
    sample_size: Optional[int] = None,
    output_dir: Optional[str] = None,
    config: Optional[ComparisonConfig] = None,
    csv_path: Optional[str] = None,
) -> ComparisonResults:
    """
    Run sample validation pipeline (per-household, slower).

    Args:
        source: Data source ("policyengine" or "csv")
        year: Tax year
        sample_size: Optional sample size for faster validation
        output_dir: Directory to save results
        config: Comparison configuration
        csv_path: Path to CSV if source="csv"

    Returns:
        ComparisonResults
    """
    from .cps_loader import load_cps_data
    from .runners import run_both

    print(f"Loading CPS data from {source}...")
    df = load_cps_data(
        source=source,
        year=year,
        sample_size=sample_size,
        csv_path=csv_path,
    )
    print(f"Loaded {len(df):,} households")

    print("\nRunning calculators...")
    results_df = run_both(df, year=year)

    print("\nComparing results...")
    comparator = Comparator(config)
    results = comparator.compare(results_df)

    print("\n" + results.detailed_report())

    if output_dir:
        results.save_report(Path(output_dir))

    return results


def validate_full(
    year: int = 2025,
    output_dir: Optional[str] = None,
    config: Optional[ComparisonConfig] = None,
) -> ComparisonResults:
    """
    Run full CPS validation using vectorized operations (fast).

    This runs on the entire enhanced CPS dataset using PE Microsimulation,
    which is orders of magnitude faster than per-household simulations.

    Args:
        year: Tax year
        output_dir: Directory to save results
        config: Comparison configuration

    Returns:
        ComparisonResults
    """
    from .runners import run_both_vectorized

    results_df = run_both_vectorized(year=year)

    print("\nComparing results...")
    comparator = Comparator(config)
    results = comparator.compare(results_df)

    print("\n" + results.detailed_report())

    if output_dir:
        results.save_report(Path(output_dir))

    return results
