"""
Validation tests: Compiled JS output vs Python reference implementations.

These tests ensure that compiled .yaml files produce the same results
as the Python reference implementations.

Test strategy:
1. Compile .yaml file to JS
2. Run both Python and JS with same inputs
3. Verify outputs match exactly
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from src.rulespec_compile.calculators import (
    calculate_actc,
    calculate_ctc,
    calculate_eitc,
    calculate_snap_benefit,
)
from src.rulespec_compile.parser import parse_rulespec


def run_js_calculator(js_code: str, inputs: dict) -> dict:
    """
    Execute JS calculator with given inputs and return result.

    Args:
        js_code: The compiled JS code
        inputs: Dict of input values

    Returns:
        Dict with calculation results
    """
    # Create a Node.js script that imports and runs the calculator
    inputs_json = json.dumps(inputs)
    runner = f"""
import {{ calculate }} from './calculator.mjs';
const result = calculate({inputs_json});
console.log(JSON.stringify(result));
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write calculator module
        calc_path = Path(tmpdir) / "calculator.mjs"
        calc_path.write_text(js_code)

        # Write runner script
        runner_path = Path(tmpdir) / "runner.mjs"
        runner_path.write_text(runner)

        # Execute with Node.js
        result = subprocess.run(
            ["node", str(runner_path)],
            capture_output=True,
            text=True,
            cwd=tmpdir,
        )

        if result.returncode != 0:
            raise RuntimeError(f"JS execution failed: {result.stderr}")

        return json.loads(result.stdout)


class TestEITCJSvsPython:
    """Validate compiled EITC JS matches Python implementation."""

    @pytest.fixture
    def eitc_js_code(self):
        """Load and compile EITC .yaml file."""
        eitc_path = Path(__file__).parent.parent / "examples" / "eitc.yaml"
        content = eitc_path.read_text()
        rulespec_file = parse_rulespec(content)
        gen = rulespec_file.to_js_generator()
        return gen.generate()

    # Test cases covering different scenarios
    CASES = [
        # (earned_income, agi, n_children, is_joint)
        (0, 0, 0, False),
        (10000, 10000, 0, False),
        (15000, 15000, 1, False),
        (20000, 20000, 2, False),
        (25000, 25000, 3, True),
        (50000, 50000, 2, True),
        (8260, 8260, 0, False),  # At earned income amount
        (17880, 17880, 3, False),  # At earned income amount for 3 kids
    ]

    @pytest.mark.parametrize("earned_income,agi,n_children,is_joint", CASES)
    def test_eitc_js_matches_python(
        self, eitc_js_code, earned_income, agi, n_children, is_joint
    ):
        """JS EITC calculation matches Python implementation."""
        # Python calculation
        py_result = calculate_eitc(
            earned_income=earned_income,
            agi=agi,
            n_children=n_children,
            is_joint=is_joint,
        )

        # JS calculation
        # Note: JS uses is_joint as number (0/1) due to type detection
        js_result = run_js_calculator(
            eitc_js_code,
            {
                "earned_income": earned_income,
                "agi": agi,
                "n_children": n_children,
                "is_joint": 1 if is_joint else 0,
            },
        )

        assert js_result["eitc"] == py_result.eitc, (
            f"EITC mismatch: JS={js_result['eitc']}, Python={py_result.eitc} "
            f"for inputs earned_income={earned_income}, agi={agi}, "
            f"n_children={n_children}, is_joint={is_joint}"
        )


class TestCTCJSvsPython:
    """Validate compiled CTC JS matches Python implementation."""

    @pytest.fixture
    def ctc_js_code(self):
        """Load and compile CTC .yaml file."""
        ctc_path = Path(__file__).parent.parent / "examples" / "ctc.yaml"
        content = ctc_path.read_text()
        rulespec_file = parse_rulespec(content)
        gen = rulespec_file.to_js_generator()
        return gen.generate()

    CASES = [
        # (n_qualifying_children, agi, is_joint, earned_income)
        (0, 50000, False, 30000),
        (1, 50000, False, 30000),
        (2, 100000, True, 80000),
        (3, 200000, False, 150000),
        (2, 450000, True, 300000),  # In phaseout
    ]

    @pytest.mark.parametrize("n_children,agi,is_joint,earned_income", CASES)
    def test_ctc_js_matches_python(
        self, ctc_js_code, n_children, agi, is_joint, earned_income
    ):
        """JS CTC calculation matches Python implementation."""
        # Python calculations
        py_ctc = calculate_ctc(
            n_qualifying_children=n_children,
            agi=agi,
            is_joint=is_joint,
        )
        py_actc = calculate_actc(
            n_qualifying_children=n_children,
            earned_income=earned_income,
        )

        # JS calculation
        js_result = run_js_calculator(
            ctc_js_code,
            {
                "n_qualifying_children": n_children,
                "agi": agi,
                "is_joint": 1 if is_joint else 0,
                "earned_income": earned_income,
            },
        )

        assert js_result["ctc"] == py_ctc.ctc, (
            f"CTC mismatch: JS={js_result['ctc']}, Python={py_ctc.ctc}"
        )
        assert js_result["actc"] == py_actc.actc, (
            f"ACTC mismatch: JS={js_result['actc']}, Python={py_actc.actc}"
        )


class TestSNAPJSvsPython:
    """Validate compiled SNAP JS matches Python implementation."""

    @pytest.fixture
    def snap_js_code(self):
        """Load and compile SNAP .yaml file."""
        snap_path = Path(__file__).parent.parent / "examples" / "snap.yaml"
        content = snap_path.read_text()
        rulespec_file = parse_rulespec(content)
        gen = rulespec_file.to_js_generator()
        return gen.generate()

    CASES = [
        # (household_size, gross_income)
        (1, 0),
        (1, 500),
        (2, 1000),
        (4, 2000),
        (4, 3000),
        (8, 0),
    ]

    @pytest.mark.parametrize("household_size,gross_income", CASES)
    def test_snap_js_matches_python(self, snap_js_code, household_size, gross_income):
        """JS SNAP calculation matches Python implementation."""
        # Python calculation
        py_result = calculate_snap_benefit(
            household_size=household_size,
            gross_income=gross_income,
        )

        # JS calculation
        js_result = run_js_calculator(
            snap_js_code,
            {
                "household_size": household_size,
                "gross_income": gross_income,
            },
        )

        assert js_result["snap_benefit"] == py_result.benefit, (
            f"SNAP mismatch: JS={js_result['snap_benefit']},"
            f" Python={py_result.benefit} "
            f"for household_size={household_size},"
            f" gross_income={gross_income}"
        )


class TestCitationChainPreserved:
    """Verify citation chain is preserved in JS output."""

    def test_eitc_citations_in_js(self):
        """EITC JS output includes all citations."""
        eitc_path = Path(__file__).parent.parent / "examples" / "eitc.yaml"
        content = eitc_path.read_text()
        rulespec_file = parse_rulespec(content)
        gen = rulespec_file.to_js_generator()
        js_code = gen.generate()

        js_result = run_js_calculator(js_code, {"earned_income": 10000})

        citations = js_result.get("citations", [])
        sources = [c.get("source", "") for c in citations]

        # Check key citations are present
        assert any("26 USC 32" in s for s in sources)
        assert any("Rev. Proc. 2024-40" in s for s in sources)
