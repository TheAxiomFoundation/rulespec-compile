"""
Python reference implementations of tax/benefit calculators.

These serve as the ground truth for validating:
1. Against external sources (IRS, USDA, etc.)
2. Against compiled outputs (JS, WASM, etc.)
"""

from .ctc import CTC_PARAMS, calculate_actc, calculate_ctc
from .eitc import EITC_PARAMS, calculate_eitc
from .snap import SNAP_PARAMS, calculate_snap_benefit, calculate_snap_eligible

__all__ = [
    "calculate_eitc",
    "EITC_PARAMS",
    "calculate_ctc",
    "calculate_actc",
    "CTC_PARAMS",
    "calculate_snap_benefit",
    "calculate_snap_eligible",
    "SNAP_PARAMS",
]
