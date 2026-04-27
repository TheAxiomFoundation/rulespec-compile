"""
Runners for validation: execute calculators on CPS microdata.

Provides functions to run both RuleSpec calculators and PolicyEngine-US
on the same household data for comparison.

Two modes:
1. Vectorized (fast): Use PE Microsimulation on full CPS, vectorized RuleSpec
2. Individual (slow): Build individual situations for each household
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..batch_executor import execute_lowered_program_batch
from ..compile_model import LoweredProgram
from ..program import load_rulespec_program

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VALIDATION_EFFECTIVE_DATE = date(2025, 1, 1)


@dataclass(frozen=True)
class CompiledValidationCalculators:
    """Compiled calculator callables for the shipped validation examples."""

    eitc: Callable[..., dict[str, Any]]
    ctc: Callable[..., dict[str, Any]]
    snap: Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class LoweredValidationPrograms:
    """Lowered programs for the shipped validation examples."""

    eitc: LoweredProgram
    ctc: LoweredProgram
    snap: LoweredProgram


@lru_cache(maxsize=1)
def _load_lowered_validation_programs() -> LoweredValidationPrograms:
    """Lower the shipped validation examples once."""
    return LoweredValidationPrograms(
        eitc=_load_validation_program("eitc.yaml", outputs=["eitc"]),
        ctc=_load_validation_program("ctc.yaml", outputs=["ctc", "actc"]),
        snap=_load_validation_program("snap.yaml", outputs=["snap_benefit"]),
    )


@lru_cache(maxsize=1)
def _load_compiled_validation_calculators() -> CompiledValidationCalculators:
    """Compile the shipped validation examples to Python callables once."""
    programs = _load_lowered_validation_programs()
    return CompiledValidationCalculators(
        eitc=_lowered_program_to_python_callable(programs.eitc),
        ctc=_lowered_program_to_python_callable(programs.ctc),
        snap=_lowered_program_to_python_callable(programs.snap),
    )


def _load_validation_program(
    filename: str,
    *,
    outputs: list[str],
) -> LoweredProgram:
    """Lower one shipped validation example once."""
    return load_rulespec_program(_REPO_ROOT / "examples" / filename).to_lowered_program(
        effective_date=_VALIDATION_EFFECTIVE_DATE,
        outputs=outputs,
    )


def _lowered_program_to_python_callable(
    program: LoweredProgram,
) -> Callable[..., dict[str, Any]]:
    """Compile one lowered validation program to Python."""
    code = program.to_python_generator().generate()
    namespace: dict[str, Any] = {}
    exec(code, namespace)
    calculate = namespace.get("calculate")
    if not callable(calculate):
        raise RuntimeError("Compiled validation example did not define calculate().")
    return calculate


def run_rulespec(df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    """
    Run compiled RuleSpec examples on CPS household data.

    Args:
        df: DataFrame with household data (from load_cps_data)
        show_progress: Show progress bar

    Returns:
        DataFrame with household_id and calculated values
    """
    calculators = _load_compiled_validation_calculators()
    results = []
    iterator = (
        tqdm(df.iterrows(), total=len(df), desc="RuleSpec")
        if show_progress
        else df.iterrows()
    )

    for _, row in iterator:
        eitc_result = calculators.eitc(
            earned_income=row["earned_income"],
            agi=row["agi"],
            n_children=row["n_children"],
            is_joint=row["is_joint"],
        )
        ctc_result = calculators.ctc(
            n_qualifying_children=row["n_children"],
            agi=row["agi"],
            is_joint=row["is_joint"],
            earned_income=row["earned_income"],
        )
        snap_result = calculators.snap(
            household_size=row["household_size"],
            gross_income=row["gross_monthly_income"],
        )

        results.append(
            {
                "household_id": row["household_id"],
                "rulespec_eitc": eitc_result["eitc"],
                "rulespec_ctc": ctc_result["ctc"],
                "rulespec_actc": ctc_result["actc"],
                "rulespec_snap": snap_result["snap_benefit"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df.attrs["rulespec_execution_mode"] = "compiled_example"
    return results_df


def run_policyengine(
    df: pd.DataFrame,
    year: int = 2025,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Run PolicyEngine-US on CPS household data.

    Args:
        df: DataFrame with household data (from load_cps_data)
        year: Tax year
        show_progress: Show progress bar

    Returns:
        DataFrame with household_id and PolicyEngine calculated values
    """
    try:
        import policyengine_us  # noqa: F401
    except ImportError:
        raise ImportError(
            "policyengine-us required for validation. "
            "Install with: pip install policyengine-us"
        )

    results = []
    iterator = (
        tqdm(df.iterrows(), total=len(df), desc="PolicyEngine")
        if show_progress
        else df.iterrows()
    )

    for _, row in iterator:
        try:
            pe_values = _run_single_pe_simulation(row, year)
            pe_values["household_id"] = row["household_id"]
            results.append(pe_values)
        except Exception as e:
            # Log error but continue with other households
            results.append(
                {
                    "household_id": row["household_id"],
                    "pe_eitc": np.nan,
                    "pe_ctc": np.nan,
                    "pe_actc": np.nan,
                    "pe_snap": np.nan,
                    "pe_error": str(e),
                }
            )

    results_df = pd.DataFrame(results)
    results_df.attrs["policyengine_execution_mode"] = "policyengine_household"
    return results_df


def run_policyengine_household(
    inputs: Dict[str, Any],
    year: int = 2025,
) -> Dict[str, float]:
    """
    Run PolicyEngine-US for one normalized household input record.

    This is the narrow bridge used by the compiler harness for fixed
    example-oracle comparisons without going through the full CPS pipeline.
    """
    return _run_single_pe_simulation(pd.Series(inputs), year)


def _run_single_pe_simulation(row: pd.Series, year: int) -> Dict:
    """Run PolicyEngine simulation for a single household."""
    from policyengine_us import Simulation

    gross_monthly_income = float(
        row.get("gross_monthly_income", row.get("gross_income", 0)) or 0
    )
    earned_income = float(row.get("earned_income", 0) or 0)
    if earned_income == 0 and gross_monthly_income != 0:
        earned_income = gross_monthly_income * 12

    is_joint = bool(row.get("is_joint", False))
    n_children = int(row.get("n_children", row.get("n_qualifying_children", 0)) or 0)
    minimum_household_size = 1 + int(is_joint) + n_children
    household_size = int(row.get("household_size", minimum_household_size) or 0)
    household_size = max(household_size, minimum_household_size)
    state_code = str(row.get("state_code", "CA") or "CA")

    # Build people dict
    people = {
        "adult": {
            "age": {year: 30},
            "employment_income": {year: earned_income},
        }
    }

    members = ["adult"]

    # Add spouse if joint
    if is_joint:
        people["spouse"] = {
            "age": {year: 30},
            "employment_income": {year: 0},
        }
        members.append("spouse")

    # Add children
    for i in range(n_children):
        child_id = f"child_{i}"
        people[child_id] = {
            "age": {year: 5},
            "is_tax_unit_dependent": {year: True},
        }
        members.append(child_id)

    # Add additional household members for SNAP (beyond tax unit)
    extra_members = household_size - len(members)
    for i in range(extra_members):
        extra_id = f"extra_{i}"
        people[extra_id] = {
            "age": {year: 25},
        }
        members.append(extra_id)

    filing_status = "JOINT" if is_joint else "SINGLE"

    situation = {
        "people": people,
        "tax_units": {
            "tax_unit": {
                "members": members,
                "filing_status": {year: filing_status},
            }
        },
        "families": {"family": {"members": members}},
        "spm_units": {"spm_unit": {"members": members}},
        "households": {
            "household": {
                "members": members,
                "state_code": {year: state_code},
            }
        },
    }

    sim = Simulation(situation=situation)

    return {
        "pe_eitc": float(sim.calculate("eitc", year)[0]),
        "pe_ctc": float(sim.calculate("ctc", year)[0]),
        "pe_actc": float(sim.calculate("refundable_ctc", year)[0]),
        "pe_snap": float(sim.calculate("snap", year)[0]) / 12,  # Monthly
    }


def run_both(
    df: pd.DataFrame,
    year: int = 2025,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Run both RuleSpec and PolicyEngine on the same data.

    Args:
        df: DataFrame with household data
        year: Tax year
        show_progress: Show progress bars

    Returns:
        Merged DataFrame with both sets of results
    """
    rulespec_results = run_rulespec(df, show_progress=show_progress)
    pe_results = run_policyengine(df, year, show_progress)

    # Merge results
    merged = df.merge(rulespec_results, on="household_id")
    merged = merged.merge(pe_results, on="household_id")
    merged.attrs["rulespec_execution_mode"] = rulespec_results.attrs.get(
        "rulespec_execution_mode",
        "unknown",
    )
    merged.attrs["policyengine_execution_mode"] = pe_results.attrs.get(
        "policyengine_execution_mode",
        "unknown",
    )

    return merged


# =============================================================================
# VECTORIZED RUNNERS (Fast - for full CPS)
# =============================================================================


def run_rulespec_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run compiled RuleSpec examples over a full DataFrame batch.

    Uses the generic lowered-program batch executor for the current validated
    subset instead of handwritten vectorized formulas.
    """
    programs = _load_lowered_validation_programs()
    eitc_results = execute_lowered_program_batch(
        programs.eitc,
        df[["earned_income", "agi", "n_children", "is_joint"]],
    )
    ctc_results = execute_lowered_program_batch(
        programs.ctc,
        {
            "n_qualifying_children": df["n_children"].to_numpy(),
            "agi": df["agi"].to_numpy(),
            "is_joint": df["is_joint"].to_numpy(),
            "earned_income": df["earned_income"].to_numpy(),
        },
    )
    snap_results = execute_lowered_program_batch(
        programs.snap,
        {
            "household_size": df["household_size"].to_numpy(),
            "gross_income": df["gross_monthly_income"].to_numpy(),
        },
    )
    results_df = pd.DataFrame(
        {
            "household_id": df["household_id"].to_numpy(),
            "rulespec_eitc": eitc_results["eitc"].to_numpy(),
            "rulespec_ctc": ctc_results["ctc"].to_numpy(),
            "rulespec_actc": ctc_results["actc"].to_numpy(),
            "rulespec_snap": snap_results["snap_benefit"].to_numpy(),
        }
    )
    results_df.attrs["rulespec_execution_mode"] = "compiled_batch"
    return results_df


def run_policyengine_microsim(year: int = 2025) -> pd.DataFrame:
    """
    Run PolicyEngine on full enhanced CPS using Microsimulation (vectorized).

    This is MUCH faster than individual simulations - runs in ~30 seconds
    instead of ~22 hours.

    Returns DataFrame with tax_unit_id and PE calculated values.
    """
    try:
        from policyengine_us import Microsimulation
    except ImportError:
        raise ImportError(
            "policyengine-us required for validation. "
            "Install with: pip install policyengine-us"
        )

    print("Loading PolicyEngine Microsimulation (this may take a minute)...")
    sim = Microsimulation()  # Uses enhanced CPS by default

    print("Calculating EITC...")
    eitc = sim.calculate("eitc", year)

    print("Calculating CTC...")
    ctc = sim.calculate("ctc", year)

    print("Calculating refundable CTC...")
    actc = sim.calculate("refundable_ctc", year)

    print("Calculating SNAP...")
    snap = sim.calculate("snap", year) / 12  # Monthly

    # Get tax unit IDs
    tax_unit_id = sim.calculate("tax_unit_id", year)

    # Aggregate to tax unit level (sum over members)
    # For tax credits, we take the tax unit value (same for all members)
    unique_tu = np.unique(tax_unit_id)

    results = []
    for tu_id in unique_tu:
        mask = tax_unit_id == tu_id
        results.append(
            {
                "tax_unit_id": int(tu_id),
                "pe_eitc": float(eitc[mask].iloc[0]),
                "pe_ctc": float(ctc[mask].iloc[0]),
                "pe_actc": float(actc[mask].iloc[0]),
                "pe_snap": float(snap[mask].iloc[0]),
            }
        )

    return pd.DataFrame(results)


def run_both_vectorized(year: int = 2025) -> pd.DataFrame:
    """
    Run full CPS validation using vectorized operations.

    Extracts inputs and PE outputs from Microsimulation, then runs
    RuleSpec on the same inputs for comparison.

    Handles different entity levels:
    - Tax unit: EITC, CTC, ACTC
    - SPM unit: SNAP
    """
    try:
        from policyengine_us import Microsimulation
    except ImportError:
        raise ImportError(
            "policyengine-us required for validation. "
            "Install with: pip install policyengine-us"
        )

    print("Loading PolicyEngine Microsimulation...")
    sim = Microsimulation()

    # ==========================================================================
    # TAX UNIT LEVEL (EITC, CTC, ACTC)
    # ==========================================================================
    print("\n--- Tax Unit Level (EITC, CTC, ACTC) ---")
    print("Extracting tax unit data...")
    tax_unit_id = sim.calculate("tax_unit_id", year)
    unique_tu = np.unique(tax_unit_id)
    print(f"Found {len(unique_tu):,} tax units")

    print("Calculating PolicyEngine tax credit outputs...")
    pe_eitc = sim.calculate("eitc", year)
    pe_ctc = sim.calculate("ctc", year)
    pe_actc = sim.calculate("refundable_ctc", year)

    print("Extracting tax unit input variables...")
    earned_income = sim.calculate("tax_unit_earned_income", year)
    agi = sim.calculate("adjusted_gross_income", year)
    n_children = sim.calculate("tax_unit_children", year)
    filing_status = sim.calculate("filing_status", year)
    tax_unit_size = sim.calculate("tax_unit_size", year)

    # ==========================================================================
    # SPM UNIT LEVEL (SNAP)
    # ==========================================================================
    print("\n--- SPM Unit Level (SNAP) ---")
    print("Extracting SPM unit data...")
    spm_unit_id = sim.calculate("spm_unit_id", year)
    unique_spm = np.unique(spm_unit_id)
    print(f"Found {len(unique_spm):,} SPM units")

    print("Calculating PolicyEngine SNAP output...")
    pe_snap = sim.calculate("snap", year) / 12  # Monthly

    print("Extracting SPM unit input variables...")
    spm_unit_size = sim.calculate("spm_unit_size", year)
    # Use SPM unit net income as proxy for gross (PE models deductions)
    spm_unit_net_income = sim.calculate("spm_unit_net_income", year)

    # ==========================================================================
    # BUILD TAX UNIT RECORDS
    # ==========================================================================
    print("\nBuilding tax unit comparison dataset...")
    tu_records = []
    for tu_id in tqdm(unique_tu, desc="Tax units"):
        mask = tax_unit_id == tu_id
        idx = np.where(mask)[0][0]

        is_joint = filing_status.values[idx] == "JOINT"

        tu_records.append(
            {
                "household_id": int(tu_id),
                "earned_income": float(earned_income.values[idx]),
                "agi": float(agi.values[idx]),
                "n_children": int(n_children.values[idx]),
                "is_joint": is_joint,
                "household_size": int(tax_unit_size.values[idx]),
                "gross_monthly_income": float(agi.values[idx]) / 12,
                "pe_eitc": float(pe_eitc.values[idx]),
                "pe_ctc": float(pe_ctc.values[idx]),
                "pe_actc": float(pe_actc.values[idx]),
            }
        )

    tu_df = pd.DataFrame(tu_records)

    # ==========================================================================
    # BUILD SPM UNIT RECORDS FOR SNAP
    # ==========================================================================
    print("Building SPM unit comparison dataset...")
    spm_records = []
    for spm_id in tqdm(unique_spm, desc="SPM units"):
        mask = spm_unit_id == spm_id
        idx = np.where(mask)[0][0]

        # Get annual income, convert to monthly
        annual_income = float(spm_unit_net_income.values[idx])
        monthly_income = max(0, annual_income / 12)

        spm_records.append(
            {
                "spm_unit_id": int(spm_id),
                "household_size": int(spm_unit_size.values[idx]),
                "gross_monthly_income": monthly_income,
                "pe_snap": float(pe_snap.values[idx]),
            }
        )

    spm_df = pd.DataFrame(spm_records)

    # Run compiled RuleSpec SNAP on SPM units
    print("\nRunning RuleSpec SNAP (compiled batch)...")
    programs = _load_lowered_validation_programs()
    snap_results = execute_lowered_program_batch(
        programs.snap,
        {
            "household_size": spm_df["household_size"].to_numpy(),
            "gross_income": spm_df["gross_monthly_income"].to_numpy(),
        },
    )
    spm_df["rulespec_snap"] = snap_results["snap_benefit"].to_numpy()

    # ==========================================================================
    # RUN RuleSpec ON TAX UNITS
    # ==========================================================================
    print("Running RuleSpec tax credits (vectorized)...")
    rulespec_tu = run_rulespec_vectorized(tu_df)

    # Merge tax unit results
    tu_merged = tu_df.merge(rulespec_tu, on="household_id")

    # Add SNAP columns as NaN (different entity level)
    tu_merged["pe_snap"] = np.nan
    tu_merged["rulespec_snap"] = np.nan

    # ==========================================================================
    # COMBINE RESULTS
    # For comparison, we return tax unit data for EITC/CTC/ACTC
    # and add a separate SNAP comparison summary
    # ==========================================================================
    print(f"\nTax unit dataset: {len(tu_merged):,} units")
    print(f"SPM unit dataset: {len(spm_df):,} units")

    # Store SPM results as attribute for separate SNAP validation
    tu_merged.attrs["spm_snap_data"] = spm_df
    tu_merged.attrs["rulespec_execution_mode"] = "compiled_batch"
    tu_merged.attrs["policyengine_execution_mode"] = "policyengine_microsim"

    return tu_merged
