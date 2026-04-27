"""
EITC Calculator - Python reference implementation.

Source: 26 USC 32
Tax Year: 2025

This matches the logic in examples/eitc.yaml exactly.
"""

from dataclasses import dataclass

# Parameters from statute and guidance
EITC_PARAMS = {
    # 26 USC 32(b)(1) - statutory percentages
    "credit_pct": {0: 7.65, 1: 34.0, 2: 40.0, 3: 45.0},
    "phaseout_pct": {0: 7.65, 1: 15.98, 2: 21.06, 3: 21.06},
    # Rev. Proc. 2024-40 - inflation-adjusted thresholds for TY2025
    "earned_income_amount": {0: 8484, 1: 12729, 2: 17880, 3: 17880},
    "phaseout_single": {0: 10620, 1: 23350, 2: 23350, 3: 23350},
    "phaseout_joint": {0: 17730, 1: 30470, 2: 30470, 3: 30470},
}


@dataclass
class EITCResult:
    """EITC calculation result with citation chain."""

    eitc: int
    credit_base: float
    phaseout: float
    citations: list


def calculate_eitc(
    earned_income: float = 0,
    agi: float = 0,
    n_children: int = 0,
    is_joint: bool = False,
) -> EITCResult:
    """
    Calculate EITC per 26 USC 32.

    Args:
        earned_income: Earned income (wages, self-employment)
        agi: Adjusted Gross Income
        n_children: Number of qualifying children (0-3+)
        is_joint: True if married filing jointly

    Returns:
        EITCResult with credit amount and citation chain
    """
    # Cap children at 3 per 32(b)(1)
    n = min(n_children, 3)

    credit_pct = EITC_PARAMS["credit_pct"][n] / 100
    phaseout_pct = EITC_PARAMS["phaseout_pct"][n] / 100
    earned_amount = EITC_PARAMS["earned_income_amount"][n]
    phaseout_start = (
        EITC_PARAMS["phaseout_joint"][n]
        if is_joint
        else EITC_PARAMS["phaseout_single"][n]
    )

    # 32(a)(1): Credit percentage of earned income up to earned income amount
    credit_base = credit_pct * min(earned_income, earned_amount)

    # 32(a)(2): Phaseout based on greater of AGI or earned income
    income_for_phaseout = max(agi, earned_income)
    excess = max(0, income_for_phaseout - phaseout_start)
    phaseout = phaseout_pct * excess

    # Final credit (non-negative, rounded to nearest dollar)
    eitc = round(max(0, credit_base - phaseout))

    return EITCResult(
        eitc=eitc,
        credit_base=credit_base,
        phaseout=phaseout,
        citations=[
            {"param": "credit_pct", "source": "26 USC 32(b)(1)"},
            {"param": "phaseout_pct", "source": "26 USC 32(b)(1)"},
            {"param": "earned_income_amount", "source": "Rev. Proc. 2024-40"},
            {"param": "phaseout_single", "source": "Rev. Proc. 2024-40"},
            {"param": "phaseout_joint", "source": "Rev. Proc. 2024-40"},
            {"variable": "eitc", "source": "26 USC 32"},
        ],
    )
