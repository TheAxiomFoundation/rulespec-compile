"""
Command-line interface for rulespec-compile.

Usage:
    rulespec-compile compile input.yaml -o output.js
    rulespec-compile compile input.yaml --python -o output.py
    rulespec-compile compile input.yaml --rust -o output.rs
    rulespec-compile eitc -o eitc.js
"""

import argparse
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

from .compile_model import CompilationError
from .harness import (
    format_harness_summary,
    format_harness_summary_json,
    run_compiler_harness,
)
from .js_generator import generate_eitc_calculator
from .program import load_rulespec_program
from .python_generator import generate_eitc_calculator as generate_eitc_calculator_py
from .rule_bindings import (
    RuleBindingError,
    load_rule_bindings_file,
    merge_rule_bindings,
)


def _parse_effective_date(value: str) -> date:
    """Parse an ISO date for temporal compilation."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected YYYY-MM-DD."
        ) from exc


def _parse_rule_binding_name(name: str) -> str:
    """Validate a rule binding target name."""
    if re.fullmatch(r"[A-Za-z_]\w*", name):
        return name
    if "." not in name:
        raise argparse.ArgumentTypeError(f"Invalid rule binding target '{name}'.")

    module_identity, symbol = name.rsplit(".", 1)
    if not module_identity or not re.fullmatch(r"[A-Za-z_]\w*", symbol):
        raise argparse.ArgumentTypeError(f"Invalid rule binding target '{name}'.")
    return name


def _parse_rule_binding(value: str) -> tuple[str, int, float]:
    """Parse NAME=VALUE, module_identity.symbol=VALUE, or indexed variants."""
    try:
        lhs, rhs = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid rule binding '{value}'. Expected NAME=VALUE."
        ) from exc

    lhs = lhs.strip()
    rhs = rhs.strip()
    if not lhs:
        raise argparse.ArgumentTypeError(
            f"Invalid rule binding '{value}'. Missing rule name."
        )

    try:
        numeric_value = float(rhs)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid rule binding '{value}'. VALUE must be numeric."
        ) from exc

    indexed_match = re.fullmatch(r"(.+)\[(\d+)\]", lhs)
    if indexed_match:
        return (
            _parse_rule_binding_name(indexed_match.group(1)),
            int(indexed_match.group(2)),
            numeric_value,
        )

    try:
        binding_name = _parse_rule_binding_name(lhs)
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid rule binding '{value}'. Use NAME=VALUE, "
            "module_identity.symbol=VALUE, or indexed variants."
        ) from exc
    return binding_name, 0, numeric_value


def _parse_module_package(value: str) -> tuple[str, Path]:
    """Parse NAME=DIR workspace package bindings."""
    try:
        name, directory = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid package binding '{value}'. Expected NAME=DIR."
        ) from exc

    name = name.strip()
    directory = directory.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name):
        raise argparse.ArgumentTypeError(
            f"Invalid package binding '{value}'. NAME must be a valid package alias."
        )
    if not directory:
        raise argparse.ArgumentTypeError(
            f"Invalid package binding '{value}'. Missing package directory."
        )
    return name, Path(directory)


def _build_rule_bindings(
    bindings: list[tuple[str, int, float]] | None,
) -> dict[str, dict[int, float]]:
    """Aggregate repeated CLI rule bindings."""
    bindings_by_name: dict[str, dict[int, float]] = {}
    for name, index, value in bindings or []:
        bindings_by_name.setdefault(name, {})[index] = value
    return bindings_by_name


def _load_rule_binding_files(paths: list[Path] | None) -> Any:
    """Load and merge repeated binding-file arguments."""
    bundles = [load_rule_bindings_file(path) for path in paths or []]
    if not bundles:
        return {}
    return merge_rule_bindings(*bundles)


def _build_module_packages(
    bindings: list[tuple[str, Path]] | None,
) -> dict[str, Path]:
    """Aggregate repeated CLI package bindings."""
    packages: dict[str, Path] = {}
    for name, directory in bindings or []:
        resolved = directory.expanduser().resolve()
        existing = packages.get(name)
        if existing is not None and existing != resolved:
            raise CompilationError(
                f"CLI package alias '{name}' was provided more than once with "
                f"different directories: '{existing}' and '{resolved}'."
            )
        packages[name] = resolved
    return packages


def _add_program_compile_arguments(command_parser: argparse.ArgumentParser) -> None:
    """Add shared program-loading arguments for compile and lower commands."""
    command_parser.add_argument(
        "input",
        type=Path,
        help="Input .yaml file",
    )
    command_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )
    command_parser.add_argument(
        "--effective-date",
        type=_parse_effective_date,
        help="Resolve temporal RuleSpec definitions as of YYYY-MM-DD",
    )
    command_parser.add_argument(
        "--binding",
        action="append",
        type=_parse_rule_binding,
        help=(
            "Bind an external rule as NAME=VALUE, module_identity.symbol=VALUE, "
            "or indexed variants"
        ),
    )
    command_parser.add_argument(
        "--binding-file",
        action="append",
        type=Path,
        help="Load external rule bindings from JSON or YAML bundle files",
    )
    command_parser.add_argument(
        "--select-output",
        action="append",
        metavar="NAME",
        help="Compile only the reachable subgraph needed for this public output",
    )
    command_parser.add_argument(
        "--module-root",
        action="append",
        type=Path,
        metavar="DIR",
        help=(
            "Resolve bare imports from this workspace root in addition to rulespec.toml"
        ),
    )
    command_parser.add_argument(
        "--package",
        action="append",
        type=_parse_module_package,
        metavar="NAME=DIR",
        help="Bind imports starting with NAME/ to this workspace package directory",
    )


def _load_program_compile_inputs(args) -> tuple[Any, dict[str, Any]]:
    """Load the RuleSpec program and merged rule bindings for one CLI request."""
    rulespec_program = load_rulespec_program(
        args.input,
        module_roots=[root.expanduser().resolve() for root in args.module_root or []],
        module_packages=_build_module_packages(args.package),
    )
    file_rule_bindings = merge_rule_bindings(
        _load_rule_binding_files(args.binding_file),
    )
    rule_bindings = merge_rule_bindings(
        file_rule_bindings,
        _build_rule_bindings(args.binding),
    )
    return rulespec_program, rule_bindings


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="rulespec-compile",
        description=(
            "Compile RuleSpec .yaml files to standalone JavaScript, Python, or Rust"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compile command
    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile a .yaml file to JavaScript, Python, or Rust",
    )
    _add_program_compile_arguments(compile_parser)
    target_group = compile_parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--python",
        action="store_true",
        help="Generate Python code instead of JavaScript (default: JavaScript)",
    )
    target_group.add_argument(
        "--rust",
        action="store_true",
        help="Generate Rust code instead of JavaScript (default: JavaScript)",
    )
    lower_parser = subparsers.add_parser(
        "lower",
        help="Lower a .yaml file to a backend-neutral JSON bundle",
    )
    _add_program_compile_arguments(lower_parser)

    harness_parser = subparsers.add_parser(
        "harness",
        help="Run the objective compiler harness scorecard",
    )
    harness_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the harness summary as JSON",
    )
    harness_parser.add_argument(
        "--case",
        action="append",
        metavar="NAME",
        help="Run only the named harness case",
    )
    harness_parser.add_argument(
        "--include-external",
        action="store_true",
        help="Include opt-in external oracle cases such as PolicyEngine checks",
    )
    harness_parser.add_argument(
        "--include-live",
        action="store_true",
        help=("Include curated checks against sibling live-stack RuleSpec files"),
    )

    # EITC command (pre-built)
    eitc_parser = subparsers.add_parser(
        "eitc",
        help="Generate EITC calculator (26 USC 32)",
    )
    eitc_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )
    eitc_parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Tax year (default: 2025)",
    )
    eitc_parser.add_argument(
        "--python",
        action="store_true",
        help="Generate Python code instead of JavaScript (default: JavaScript)",
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.2.0",
    )

    args = parser.parse_args()

    if args.command in {"compile", "lower"}:
        if not args.input.exists():
            print(f"Error: {args.input} not found", file=sys.stderr)
            sys.exit(1)

        try:
            rulespec_program, rule_bindings = _load_program_compile_inputs(args)
            if args.command == "lower":
                code = rulespec_program.to_lowered_program(
                    effective_date=args.effective_date,
                    rule_bindings=rule_bindings,
                    outputs=args.select_output,
                ).to_json()
                lang = "Lowered JSON"
            elif args.rust:
                gen = rulespec_program.to_rust_generator(
                    effective_date=args.effective_date,
                    rule_bindings=rule_bindings,
                    outputs=args.select_output,
                )
                lang = "Rust"
                code = gen.generate()
            elif args.python:
                gen = rulespec_program.to_python_generator(
                    effective_date=args.effective_date,
                    rule_bindings=rule_bindings,
                    outputs=args.select_output,
                )
                lang = "Python"
                code = gen.generate()
            else:
                gen = rulespec_program.to_js_generator(
                    effective_date=args.effective_date,
                    rule_bindings=rule_bindings,
                    outputs=args.select_output,
                )
                lang = "JavaScript"
                code = gen.generate()
        except (CompilationError, RuleBindingError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if args.output:
            args.output.write_text(code)
            action = "Lowered" if args.command == "lower" else "Compiled"
            print(
                f"{action} {args.input} -> {args.output} ({lang})",
                file=sys.stderr,
            )
        else:
            print(code)

    elif args.command == "eitc":
        if hasattr(args, "python") and args.python:
            code = generate_eitc_calculator_py(tax_year=args.year)
            lang = "Python"
        else:
            code = generate_eitc_calculator(tax_year=args.year)
            lang = "JavaScript"

        if args.output:
            args.output.write_text(code)
            print(f"Generated {lang} EITC calculator -> {args.output}", file=sys.stderr)
        else:
            print(code)

    elif args.command == "harness":
        try:
            summary = run_compiler_harness(
                case_names=args.case,
                include_external=args.include_external,
                include_live=args.include_live,
            )
        except CompilationError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            print(format_harness_summary_json(summary))
        else:
            print(format_harness_summary(summary))

        if summary.failed or summary.skipped:
            sys.exit(1)

    elif args.command is None:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
