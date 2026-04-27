"""
CPS data loader for validation.

Loads enhanced CPS microdata and prepares it for running through calculators.
"""

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd


@dataclass
class CPSHousehold:
    """A single CPS household with attributes needed for tax/benefit calculations."""

    household_id: int
    year: int
    state_code: str

    # Tax unit attributes
    earned_income: float
    agi: float
    n_children: int
    is_joint: bool

    # SNAP attributes
    household_size: int
    gross_monthly_income: float

    # Weights
    weight: float = 1.0


def load_cps_from_policyengine(
    year: int = 2025,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load CPS microdata from PolicyEngine-US.

    Args:
        year: Tax year for calculations
        sample_size: If set, randomly sample this many households
        random_state: Random seed for reproducible sampling

    Returns:
        DataFrame with household-level data ready for validation
    """
    try:
        from policyengine_us import Microsimulation
    except ImportError:
        raise ImportError(
            "policyengine-us required for CPS validation. "
            "Install with: pip install policyengine-us"
        )

    # Create microsimulation with enhanced CPS (default)
    sim = Microsimulation()

    # Extract tax unit level variables
    tax_unit_id = sim.calculate("tax_unit_id", year)
    spm_unit_id = sim.calculate("spm_unit_id", year)

    # Get unique tax units
    unique_tax_units = np.unique(tax_unit_id)

    records = []
    for tu_id in unique_tax_units:
        mask = tax_unit_id == tu_id

        # Tax unit attributes
        earned_income = float(sim.calculate("tax_unit_earned_income", year)[mask].sum())
        agi = float(sim.calculate("adjusted_gross_income", year)[mask].sum())
        n_children = int(sim.calculate("tax_unit_children", year)[mask].max())
        filing_status = sim.calculate("filing_status", year)[mask].iloc[0]
        is_joint = filing_status == "JOINT"

        # Get SPM unit for this tax unit (for SNAP)
        spm_id = spm_unit_id[mask].iloc[0]
        spm_mask = spm_unit_id == spm_id
        household_size = int(spm_mask.sum())
        # Monthly income for SNAP
        gross_monthly_income = float(
            sim.calculate("spm_unit_gross_income", year)[spm_mask].sum() / 12
        )

        # State
        state = sim.calculate("state_code_str", year)[mask].iloc[0]

        # Weight
        weight = float(sim.calculate("tax_unit_weight", year)[mask].iloc[0])

        records.append(
            {
                "household_id": int(tu_id),
                "year": year,
                "state_code": state,
                "earned_income": earned_income,
                "agi": agi,
                "n_children": n_children,
                "is_joint": is_joint,
                "household_size": household_size,
                "gross_monthly_income": gross_monthly_income,
                "weight": weight,
            }
        )

    df = pd.DataFrame(records)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)

    return df.reset_index(drop=True)


def load_cps_from_csv(
    path: str,
    year: int = 2025,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load CPS data from a CSV file (e.g., from policyengine-taxsim).

    Args:
        path: Path to CSV file
        year: Tax year for calculations
        sample_size: If set, randomly sample this many households
        random_state: Random seed for reproducible sampling

    Returns:
        DataFrame with household-level data ready for validation
    """
    df = pd.read_csv(path)

    # Map TAXSIM format to our format
    records = []
    for _, row in df.iterrows():
        # Earned income = wages + self-employment
        earned_income = (
            row.get("pwages", 0)
            + row.get("psemp", 0)
            + row.get("swages", 0)
            + row.get("ssemp", 0)
        )

        # AGI approximation (more components could be added)
        agi = (
            earned_income
            + row.get("dividends", 0)
            + row.get("intrec", 0)
            + row.get("stcg", 0)
            + row.get("ltcg", 0)
            + row.get("pensions", 0)
            + row.get("gssi", 0)  # Social Security
        )

        # Number of dependents/children
        n_children = int(row.get("depx", 0))

        # Marital status (1=single, 2=married filing jointly)
        is_joint = row.get("mstat", 1) == 2

        # Household size (filer + spouse + dependents)
        household_size = 1 + (1 if is_joint else 0) + n_children

        # Monthly income for SNAP
        gross_monthly_income = agi / 12

        # State FIPS to code mapping
        state_fips = int(row.get("state", 0))
        state_code = fips_to_state_code(state_fips)

        records.append(
            {
                "household_id": int(row.get("taxsimid", 0)),
                "year": year,
                "state_code": state_code,
                "earned_income": earned_income,
                "agi": agi,
                "n_children": n_children,
                "is_joint": is_joint,
                "household_size": household_size,
                "gross_monthly_income": gross_monthly_income,
                "weight": 1.0,
            }
        )

    df = pd.DataFrame(records)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)

    return df.reset_index(drop=True)


def load_cps_data(
    source: str = "policyengine",
    year: int = 2025,
    sample_size: Optional[int] = None,
    random_state: int = 42,
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load CPS data from specified source.

    Args:
        source: "policyengine" or "csv"
        year: Tax year
        sample_size: Optional sample size
        random_state: Random seed
        csv_path: Path to CSV if source="csv"

    Returns:
        DataFrame ready for validation
    """
    if source == "policyengine":
        return load_cps_from_policyengine(year, sample_size, random_state)
    elif source == "csv":
        if csv_path is None:
            raise ValueError("csv_path required when source='csv'")
        return load_cps_from_csv(csv_path, year, sample_size, random_state)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'policyengine' or 'csv'")


def iterate_households(df: pd.DataFrame) -> Iterator[CPSHousehold]:
    """Iterate over DataFrame as CPSHousehold objects."""
    for _, row in df.iterrows():
        yield CPSHousehold(
            household_id=row["household_id"],
            year=row["year"],
            state_code=row["state_code"],
            earned_income=row["earned_income"],
            agi=row["agi"],
            n_children=row["n_children"],
            is_joint=row["is_joint"],
            household_size=row["household_size"],
            gross_monthly_income=row["gross_monthly_income"],
            weight=row["weight"],
        )


# State FIPS to abbreviation mapping
FIPS_TO_STATE = {
    0: "US",
    1: "AL",
    2: "AK",
    4: "AZ",
    5: "AR",
    6: "CA",
    8: "CO",
    9: "CT",
    10: "DE",
    11: "DC",
    12: "FL",
    13: "GA",
    15: "HI",
    16: "ID",
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    21: "KY",
    22: "LA",
    23: "ME",
    24: "MD",
    25: "MA",
    26: "MI",
    27: "MN",
    28: "MS",
    29: "MO",
    30: "MT",
    31: "NE",
    32: "NV",
    33: "NH",
    34: "NJ",
    35: "NM",
    36: "NY",
    37: "NC",
    38: "ND",
    39: "OH",
    40: "OK",
    41: "OR",
    42: "PA",
    44: "RI",
    45: "SC",
    46: "SD",
    47: "TN",
    48: "TX",
    49: "UT",
    50: "VT",
    51: "VA",
    53: "WA",
    54: "WV",
    55: "WI",
    56: "WY",
}


def fips_to_state_code(fips: int) -> str:
    """Convert FIPS code to state abbreviation."""
    return FIPS_TO_STATE.get(fips, "US")
