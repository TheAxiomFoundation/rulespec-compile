"""Workspace module-root and package resolution for RuleSpec program imports."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import tomllib


class ModuleResolutionError(ValueError):
    """Raised when rulespec.toml or configured module roots are invalid."""


@dataclass(frozen=True)
class ModuleResolutionConfig:
    """Resolved module-root and package configuration for one RuleSpec program."""

    roots: tuple[Path, ...] = ()
    packages: dict[str, Path] = field(default_factory=dict)
    config_path: Path | None = None


@dataclass
class ImportResolver:
    """Resolve RuleSpec imports relative to files, package aliases, or module roots."""

    config: ModuleResolutionConfig
    _configured_import_cache: dict[str, Path] = field(default_factory=dict)

    def resolve(self, import_path: str, importer: Path) -> Path:
        """Resolve one import string for one importing file."""
        candidate = Path(import_path)
        if candidate.is_absolute():
            return candidate.resolve()
        if import_path.startswith("."):
            return (importer.parent / candidate).resolve()
        return self._resolve_configured_path(import_path)

    def _resolve_configured_path(self, import_path: str) -> Path:
        """Resolve a non-relative import through packages or module roots."""
        cached = self._configured_import_cache.get(import_path)
        if cached is not None:
            return cached
        resolved = self._resolve_package_path(import_path)
        if resolved is None:
            resolved = self._resolve_module_path(import_path)
        self._configured_import_cache[import_path] = resolved
        return resolved

    def _resolve_package_path(self, import_path: str) -> Path | None:
        """Resolve an import through an explicit package alias when configured."""
        package_name, separator, package_suffix = import_path.partition("/")
        package_root = self.config.packages.get(package_name)
        if package_root is None:
            return None
        if not separator or not package_suffix:
            raise ModuleResolutionError(
                f"Package import '{import_path}' must include a module path after "
                f"package '{package_name}'."
            )
        resolved = (package_root / package_suffix).resolve()
        if not resolved.is_file():
            raise ModuleResolutionError(
                f"Package import '{import_path}' was not found under package "
                f"'{package_name}' at '{package_root}'."
            )
        return resolved

    def _resolve_module_path(self, import_path: str) -> Path:
        """Resolve a bare import through configured module roots."""
        matches = [
            (root / import_path).resolve()
            for root in self.config.roots
            if (root / import_path).is_file()
        ]
        if not matches:
            configured_roots = ", ".join(str(root) for root in self.config.roots)
            raise ModuleResolutionError(
                f"Bare import '{import_path}' was not found in configured module "
                f"roots: {configured_roots or '(none)'}. Add a rulespec.toml "
                "module_resolution.roots entry or pass --module-root."
            )
        unique_matches = list(dict.fromkeys(matches))
        if len(unique_matches) > 1:
            joined = ", ".join(str(match) for match in unique_matches)
            raise ModuleResolutionError(
                f"Bare import '{import_path}' is ambiguous across module roots: "
                f"{joined}."
            )
        return unique_matches[0]


def build_import_resolver(
    entry_path: Path,
    module_roots: list[Path] | tuple[Path, ...] | None = None,
    module_packages: dict[str, Path] | None = None,
) -> ImportResolver:
    """Build the resolver for one entrypoint from rulespec.toml plus CLI overrides."""
    config = discover_module_resolution(entry_path)
    extra_roots = tuple(root.resolve() for root in module_roots or [])
    extra_packages = {
        name: package_path.resolve()
        for name, package_path in (module_packages or {}).items()
    }
    return ImportResolver(
        ModuleResolutionConfig(
            roots=_ordered_unique_paths((*config.roots, *extra_roots)),
            packages=_merge_package_paths(config.packages, extra_packages),
            config_path=config.config_path,
        )
    )


def discover_module_resolution(entry_path: Path) -> ModuleResolutionConfig:
    """Discover rulespec.toml module roots and package aliases from the tree."""
    start = entry_path.parent if entry_path.is_file() else entry_path
    for directory in (start, *start.parents):
        config_path = directory / "rulespec.toml"
        if not config_path.is_file():
            continue
        return _load_module_resolution_config(config_path)
    return ModuleResolutionConfig()


def _load_module_resolution_config(config_path: Path) -> ModuleResolutionConfig:
    """Load module roots and package aliases from one rulespec.toml file."""
    try:
        payload = tomllib.loads(config_path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ModuleResolutionError(
            f"Could not parse module config '{config_path}': {exc}."
        ) from exc
    except OSError as exc:
        raise ModuleResolutionError(
            f"Could not read module config '{config_path}': {exc}."
        ) from exc

    module_section = payload.get("module_resolution", {})
    if module_section == {}:
        return ModuleResolutionConfig(config_path=config_path)
    if not isinstance(module_section, dict):
        raise ModuleResolutionError(
            f"Config '{config_path}' has invalid [module_resolution] contents."
        )

    raw_roots = module_section.get("roots", [])
    if not isinstance(raw_roots, list):
        raise ModuleResolutionError(
            f"Config '{config_path}' must define module_resolution.roots as a list."
        )
    raw_packages = module_section.get("packages", {})
    if not isinstance(raw_packages, dict):
        raise ModuleResolutionError(
            f"Config '{config_path}' must define module_resolution.packages as a table."
        )

    roots: list[Path] = []
    for raw_root in raw_roots:
        if not isinstance(raw_root, str):
            raise ModuleResolutionError(
                f"Config '{config_path}' has non-string module root {raw_root!r}."
            )
        roots.append((config_path.parent / raw_root).resolve())
    packages = _normalize_package_paths(
        raw_packages,
        base_directory=config_path.parent,
        source=f"config '{config_path}'",
    )
    return ModuleResolutionConfig(
        roots=_ordered_unique_paths(tuple(roots)),
        packages=packages,
        config_path=config_path,
    )


def _ordered_unique_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Return paths in first-seen order with duplicates removed."""
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def _normalize_package_paths(
    packages: dict[str, str | Path],
    *,
    base_directory: Path,
    source: str,
) -> dict[str, Path]:
    """Normalize package-alias directories from config or CLI input."""
    normalized: dict[str, Path] = {}
    for name, raw_path in packages.items():
        if not isinstance(name, str) or not _is_valid_package_name(name):
            raise ModuleResolutionError(
                f"{source} defines invalid package alias {name!r}."
            )
        if not isinstance(raw_path, (str, Path)):
            raise ModuleResolutionError(
                f"{source} defines non-path package alias target {raw_path!r} for "
                f"'{name}'."
            )
        resolved = (base_directory / raw_path).resolve()
        normalized[name] = resolved
    return normalized


def _merge_package_paths(
    manifest_packages: dict[str, Path],
    cli_packages: dict[str, Path],
) -> dict[str, Path]:
    """Merge manifest and CLI package aliases, rejecting conflicting bindings."""
    merged = dict(manifest_packages)
    for name, package_path in cli_packages.items():
        existing = merged.get(name)
        if existing is not None and existing != package_path:
            raise ModuleResolutionError(
                f"Package alias '{name}' is configured more than once with "
                f"different directories: '{existing}' and '{package_path}'."
            )
        merged[name] = package_path
    return merged


def _is_valid_package_name(name: str) -> bool:
    """Return whether a package alias is safe to use in import prefixes."""
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name) is not None
