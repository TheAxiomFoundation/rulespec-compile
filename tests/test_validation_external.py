"""
Unit tests for calculator edge cases and boundary conditions.

NOTE: Validation against PolicyEngine-US is done via the validation module
on CPS microdata, not via hand-built test cases. See:
    rulespec-validate --help
    python -m rulespec_compile.validation.cli --help
"""

from src.rulespec_compile.calculators import (
    calculate_actc,
    calculate_ctc,
    calculate_eitc,
    calculate_snap_benefit,
)


class TestEdgeCases:
    """Test edge cases and boundary conditions (unit tests, no external dependency)."""

    def test_eitc_zero_income(self):
        """EITC is 0 with no earned income."""
        result = calculate_eitc(earned_income=0, agi=0, n_children=2)
        assert result.eitc == 0

    def test_eitc_negative_not_possible(self):
        """EITC can never be negative."""
        result = calculate_eitc(earned_income=100000, agi=100000, n_children=0)
        assert result.eitc >= 0

    def test_eitc_four_children_same_as_three(self):
        """4+ children uses same parameters as 3 (capped)."""
        result3 = calculate_eitc(
            earned_income=20000, agi=20000, n_children=3, is_joint=True
        )
        result4 = calculate_eitc(
            earned_income=20000, agi=20000, n_children=4, is_joint=True
        )
        assert result3.eitc == result4.eitc

    def test_snap_large_household_uses_8(self):
        """Households > 8 use the 8-person values."""
        result8 = calculate_snap_benefit(household_size=8, gross_income=0)
        result10 = calculate_snap_benefit(household_size=10, gross_income=0)
        assert result10.benefit == result8.benefit

    def test_ctc_below_phaseout(self):
        """CTC at full value below phaseout threshold."""
        result = calculate_ctc(n_qualifying_children=2, agi=100000, is_joint=False)
        assert result.ctc == 4400  # $2,200 * 2 for TY2025

    def test_actc_below_threshold(self):
        """ACTC is 0 when earned income below $2,500."""
        result = calculate_actc(n_qualifying_children=2, earned_income=2000)
        assert result.actc == 0

    def test_ctc_phases_out(self):
        """CTC phases out for high earners."""
        # Single with $300k AGI should have reduced CTC
        result = calculate_ctc(n_qualifying_children=1, agi=300000, is_joint=False)
        # Base: $2,200, phaseout starts at $200k
        # Excess: $100k = 100 increments of $1k
        # Reduction: 100 * $50 = $5,000 (fully phased out)
        assert result.ctc == 0

    def test_snap_zero_income_gets_max(self):
        """Zero income household gets maximum allotment."""
        result = calculate_snap_benefit(household_size=1, gross_income=0)
        assert result.benefit == 292  # FY2025 max for 1-person household

    def test_eitc_joint_vs_single_phaseout(self):
        """Joint filers have higher phaseout thresholds."""
        income = 20000
        result_single = calculate_eitc(
            earned_income=income, agi=income, n_children=1, is_joint=False
        )
        result_joint = calculate_eitc(
            earned_income=income, agi=income, n_children=1, is_joint=True
        )
        # Joint should have higher credit due to higher phaseout start
        assert result_joint.eitc >= result_single.eitc
