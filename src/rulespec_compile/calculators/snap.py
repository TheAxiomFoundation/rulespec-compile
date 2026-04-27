"""
SNAP Calculator - Python reference implementation.

Source: 7 USC 2017, 7 CFR Part 273
Fiscal Year: 2025

This matches the logic in examples/snap.yaml exactly.
"""

from dataclasses import dataclass

# Parameters from statute and guidance
SNAP_PARAMS = {
    # USDA FNS FY2025 SNAP Allotments - 7 CFR 273.10(e)(2)(ii)
    "max_allotment": {
        1: 292,
        2: 536,
        3: 768,
        4: 975,
        5: 1158,
        6: 1390,
        7: 1536,
        8: 1756,
    },
    # 7 CFR 273.9(d)(1) - standard deduction by household size
    "standard_deduction": {
        1: 198,
        2: 198,
        3: 198,
        4: 208,
        5: 244,
        6: 279,
        7: 279,
        8: 279,
    },
    # 7 USC 2017(a) - 30% of net income
    "benefit_reduction_rate": 30,  # percent
    # 7 CFR 273.9(a)(1) - 130% FPL gross income limit (monthly)
    "gross_income_limit": {
        1: 1580,
        2: 2137,
        3: 2694,
        4: 3250,
        5: 3807,
        6: 4364,
        7: 4921,
        8: 5478,
    },
    # 7 CFR 273.9(a)(2) - 100% FPL net income limit (monthly)
    "net_income_limit": {
        1: 1215,
        2: 1644,
        3: 2072,
        4: 2500,
        5: 2929,
        6: 3357,
        7: 3786,
        8: 4214,
    },
    # Minimum benefit for 1-2 person households
    "min_benefit": 23,
}


@dataclass
class SNAPEligibilityResult:
    """SNAP eligibility check result."""

    eligible: bool
    gross_income_limit: float
    citations: list


@dataclass
class SNAPBenefitResult:
    """SNAP benefit calculation result."""

    benefit: int
    max_allotment: float
    net_income: float
    reduction: float
    citations: list


def calculate_snap_eligible(
    household_size: int = 1,
    gross_income: float = 0,
) -> SNAPEligibilityResult:
    """
    Check SNAP eligibility based on gross income test.

    Args:
        household_size: Number of people in household (1-8+)
        gross_income: Monthly gross income

    Returns:
        SNAPEligibilityResult with eligibility status
    """
    hh = min(household_size, 8)
    gross_limit = SNAP_PARAMS["gross_income_limit"][hh]
    eligible = gross_income <= gross_limit

    return SNAPEligibilityResult(
        eligible=eligible,
        gross_income_limit=gross_limit,
        citations=[
            {"param": "gross_income_limit", "source": "7 CFR 273.9(a)(1) - 130% FPL"},
            {"variable": "snap_eligible", "source": "7 USC 2017"},
        ],
    )


def calculate_snap_benefit(
    household_size: int = 1,
    gross_income: float = 0,
) -> SNAPBenefitResult:
    """
    Calculate SNAP monthly benefit per 7 USC 2017.

    Args:
        household_size: Number of people in household (1-8+)
        gross_income: Monthly gross income

    Returns:
        SNAPBenefitResult with benefit amount and citation chain
    """
    hh = min(household_size, 8)

    # Get max allotment for household size
    max_allotment = SNAP_PARAMS["max_allotment"][hh]

    # Calculate net income (gross - standard deduction)
    std_deduction = SNAP_PARAMS["standard_deduction"][hh]
    net_income = max(0, gross_income - std_deduction)

    # Check net income eligibility
    net_limit = SNAP_PARAMS["net_income_limit"][hh]
    is_eligible = net_income <= net_limit

    # Benefit = max allotment - 30% of net income
    reduction = net_income * SNAP_PARAMS["benefit_reduction_rate"] / 100
    benefit = max(0, max_allotment - reduction)

    # Minimum benefit for 1-2 person households
    min_benefit = SNAP_PARAMS["min_benefit"] if hh <= 2 else 0

    final_benefit = round(max(benefit, min_benefit)) if is_eligible else 0

    return SNAPBenefitResult(
        benefit=final_benefit,
        max_allotment=max_allotment,
        net_income=net_income,
        reduction=reduction,
        citations=[
            {"param": "max_allotment", "source": "USDA FNS FY2025 SNAP Allotments"},
            {"param": "standard_deduction", "source": "7 CFR 273.9(d)(1)"},
            {"param": "benefit_reduction_rate", "source": "7 USC 2017(a)"},
            {"param": "net_income_limit", "source": "7 CFR 273.9(a)(2) - 100% FPL"},
            {"variable": "snap_benefit", "source": "7 USC 2017"},
        ],
    )
