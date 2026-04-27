"""Tests for CLI."""

import subprocess
import sys
from pathlib import Path


def run_cli(*args):
    """Run CLI and return output."""
    result = subprocess.run(
        [sys.executable, "-m", "rulespec_compile.cli", *args],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src"},
        cwd=Path(__file__).parent.parent,
    )
    return result


class TestCLI:
    """Tests for command-line interface."""

    def test_help(self):
        """--help shows usage."""
        result = run_cli("--help")
        assert result.returncode == 0
        assert "rulespec-compile" in result.stdout
        assert "eitc" in result.stdout

    def test_version(self):
        """--version shows version."""
        result = run_cli("--version")
        assert result.returncode == 0
        assert "0.2.0" in result.stdout

    def test_eitc_to_stdout(self):
        """eitc command outputs JS to stdout."""
        result = run_cli("eitc")
        assert result.returncode == 0
        assert "function calculate(" in result.stdout
        assert "EITC" in result.stdout

    def test_eitc_to_file(self, tmp_path):
        """eitc command can write to file."""
        output_file = tmp_path / "eitc.js"
        result = run_cli("eitc", "-o", str(output_file))
        assert result.returncode == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "function calculate(" in content

    def test_no_command_shows_help(self):
        """No command shows help and exits 1."""
        result = run_cli()
        assert result.returncode == 1
