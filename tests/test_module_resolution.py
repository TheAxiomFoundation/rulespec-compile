"""Tests for rulespec.toml module-root resolution."""

import pytest

from src.rulespec_compile.module_resolution import (
    ModuleResolutionError,
    build_import_resolver,
    discover_module_resolution,
)


class TestModuleResolution:
    """Tests for bare-import resolution through configured module roots."""

    def test_discover_module_resolution_loads_roots_from_rulespec_toml(self, tmp_path):
        """Nearest rulespec.toml roots are resolved relative to the config file."""
        (tmp_path / "rulespec.toml").write_text(
            """
[module_resolution]
roots = ["./lib", "./shared"]
"""
        )
        entry = tmp_path / "policy" / "main.yaml"
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text("tax:\n  entity: Person\n  period: Year\n  dtype: Money\n")

        config = discover_module_resolution(entry)

        assert config.config_path == (tmp_path / "rulespec.toml").resolve()
        assert config.roots == (
            (tmp_path / "lib").resolve(),
            (tmp_path / "shared").resolve(),
        )

    def test_build_import_resolver_resolves_bare_imports_from_manifest(self, tmp_path):
        """Bare imports resolve through module roots declared in rulespec.toml."""
        (tmp_path / "rulespec.toml").write_text(
            """
[module_resolution]
roots = ["./lib"]
"""
        )
        target = tmp_path / "lib" / "tax" / "shared.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('rate:\n  source: "x"\n  from 2024-01-01: 0.1\n')
        entry = tmp_path / "main.yaml"
        entry.write_text('from "tax/shared.yaml" import rate\n')

        resolver = build_import_resolver(entry)

        assert resolver.resolve("tax/shared.yaml", entry) == target.resolve()

    def test_build_import_resolver_rejects_ambiguous_bare_imports(self, tmp_path):
        """The resolver fails loudly when multiple module roots match a bare import."""
        first = tmp_path / "first"
        second = tmp_path / "second"
        for root in (first, second):
            target = root / "tax" / "shared.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('rate:\n  source: "x"\n  from 2024-01-01: 0.1\n')
        entry = tmp_path / "main.yaml"
        entry.write_text('from "tax/shared.yaml" import rate\n')

        resolver = build_import_resolver(entry, module_roots=[first, second])

        with pytest.raises(ModuleResolutionError, match="is ambiguous"):
            resolver.resolve("tax/shared.yaml", entry)

    def test_discover_module_resolution_loads_package_aliases(self, tmp_path):
        """Package aliases are resolved relative to the nearest rulespec.toml."""
        (tmp_path / "rulespec.toml").write_text(
            """
[module_resolution.packages]
tax = "./packages/tax"
benefits = "./benefits"
"""
        )
        entry = tmp_path / "policy" / "main.yaml"
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text("tax:\n  entity: Person\n  period: Year\n  dtype: Money\n")

        config = discover_module_resolution(entry)

        assert config.packages == {
            "tax": (tmp_path / "packages" / "tax").resolve(),
            "benefits": (tmp_path / "benefits").resolve(),
        }

    def test_build_import_resolver_resolves_package_alias_imports(self, tmp_path):
        """Package aliases resolve imports without searching all module roots."""
        (tmp_path / "rulespec.toml").write_text(
            """
[module_resolution.packages]
tax = "./packages/tax"
"""
        )
        target = tmp_path / "packages" / "tax" / "shared.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('rate:\n  source: "x"\n  from 2024-01-01: 0.1\n')
        entry = tmp_path / "main.yaml"
        entry.write_text('from "tax/shared.yaml" import rate\n')

        resolver = build_import_resolver(entry)

        assert resolver.resolve("tax/shared.yaml", entry) == target.resolve()

    def test_build_import_resolver_rejects_conflicting_package_aliases(self, tmp_path):
        """Manifest and CLI package aliases must agree on directory targets."""
        (tmp_path / "rulespec.toml").write_text(
            """
[module_resolution.packages]
tax = "./packages/tax"
"""
        )
        entry = tmp_path / "main.yaml"
        entry.write_text('from "tax/shared.yaml" import rate\n')

        with pytest.raises(ModuleResolutionError, match="configured more than once"):
            build_import_resolver(
                entry,
                module_packages={"tax": tmp_path / "other-tax"},
            )
