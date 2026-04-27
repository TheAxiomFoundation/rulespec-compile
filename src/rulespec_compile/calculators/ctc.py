"""
CTC Calculator - Python reference implementation.

Source: 26 USC 24
Tax Year: 2025

This matches the logic in examples/ctc.yaml exactly.
"""

import math
from dataclasses import dataclass

# Parameters from statute
CTC_PARAMS = {
    # 26 USC 24(a) - base credit (TY2025: $2,200)
    "credit_per_child": 2200,
    # 26 USC 24(d)(1)(B) - max refundable per child (TY2025: $1,900)
    "refundable_max": 1900,
    # 26 USC 24(d)(1)(A) - refundable rate
    "refundable_rate": 15,  # percent
    # 26 USC 24(d)(1)(A) - earned income threshold for refundable
    "refundable_threshold": 2500,
    # 26 USC 24(b)(2) - phaseout thresholds
    "phaseout_single": 200000,
    "phaseout_joint": 400000,
    # 26 USC 24(b)(1) - phaseout rate
    "phaseout_rate": 50,  # per $1,000
}


@dataclass
class CTCResult:
    """CTC calculation result with citation chain."""

    ctc: int
    base_credit: float
    phaseout_amount: float
    citations: list


@dataclass
class ACTCResult:
    """ACTC (refundable portion) calculation result."""

    actc: int
    earned_income_above_threshold: float
    citations: list


def calculate_ctc(
    n_qualifying_children: int = 0,
    agi: float = 0,
    is_joint: bool = False,
) -> CTCResult:
    """
    Calculate Child Tax Credit per 26 USC 24.

    Args:
        n_qualifying_children: Number of qualifying children under 17
        agi: Adjusted Gross Income
        is_joint: True if married filing jointly

    Returns:
        CTCResult with credit amount and citation chain
    """
    # 24(a): Base credit per qualifying child
    base_credit = n_qualifying_children * CTC_PARAMS["credit_per_child"]

    # 24(b): Phaseout for high earners
    phaseout_start = (
        CTC_PARAMS["phaseout_joint"] if is_joint else CTC_PARAMS["phaseout_single"]
    )
    excess = max(0, agi - phaseout_start)

    # Phaseout is $50 per $1,000 over threshold (rounded up)
    phaseout_amount = math.ceil(excess / 1000) * CTC_PARAMS["phaseout_rate"]

    # Credit after phaseout (can't go below zero)
    ctc = round(max(0, base_credit - phaseout_amount))

    return CTCResult(
        ctc=ctc,
        base_credit=base_credit,
        phaseout_amount=phaseout_amount,
        citations=[
            {"param": "credit_per_child", "source": "26 USC 24(a)"},
            {"param": "phaseout_single", "source": "26 USC 24(b)(2)"},
            {"param": "phaseout_joint", "source": "26 USC 24(b)(2)"},
            {"param": "phaseout_rate", "source": "26 USC 24(b)(1)"},
            {"variable": "ctc", "source": "26 USC 24"},
        ],
    )


def calculate_actc(
    n_qualifying_children: int = 0,
    earned_income: float = 0,
) -> ACTCResult:
    """
    Calculate Additional Child Tax Credit (refundable portion) per 26 USC 24(d).

    Args:
        n_qualifying_children: Number of qualifying children under 17
        earned_income: Earned income (wages, self-employment)

    Returns:
        ACTCResult with refundable credit amount
    """
    # 24(d): Refundable portion (ACTC)
    # 15% of earned income above $2,500, up to $1,700 per child
    earned_above_threshold = max(0, earned_income - CTC_PARAMS["refundable_threshold"])
    refundable_by_earnings = (
        earned_above_threshold * CTC_PARAMS["refundable_rate"] / 100
    )
    max_refundable = n_qualifying_children * CTC_PARAMS["refundable_max"]

    # ACTC is the lesser of: refundable calc or max refundable
    actc = round(min(refundable_by_earnings, max_refundable))

    return ACTCResult(
        actc=actc,
        earned_income_above_threshold=earned_above_threshold,
        citations=[
            {"param": "refundable_max", "source": "26 USC 24(d)(1)(B)"},
            {"param": "refundable_rate", "source": "26 USC 24(d)(1)(A)"},
            {"param": "refundable_threshold", "source": "26 USC 24(d)(1)(A)"},
            {"variable": "actc", "source": "26 USC 24(d)"},
        ],
    )
