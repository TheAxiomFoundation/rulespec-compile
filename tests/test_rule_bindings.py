"""Tests for first-class external rule binding helpers."""

import json
from datetime import date

import pytest

from src.rulespec_compile.rule_bindings import (
    RuleBinding,
    RuleBindingError,
    RuleBindingTarget,
    load_rule_bindings_file,
    merge_rule_bindings,
)


class TestRuleBindings:
    """Tests for structured rule binding bundles and resolution."""

    def test_load_rule_binding_file_supports_structured_entries(self, tmp_path):
        """Structured binding files preserve identity, dates, and metadata."""
        path = tmp_path / "bindings.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "metadata": {"name": "TY2025"},
                    "bindings": [
                        {
                            "module_identity": "statutes/26/32/c/2/A",
                            "symbol": "phase_in_rate",
                            "effective_date": "2025-01-01",
                            "values": {"0": 0.34},
                            "source": "bundle://ty2025",
                        }
                    ],
                }
            )
        )

        bundle = load_rule_bindings_file(path)

        assert bundle.schema_version == 1
        assert bundle.metadata == {"name": "TY2025"}
        assert len(bundle.bindings) == 1
        assert bundle.bindings[0].target == RuleBindingTarget(
            module_identity="statutes/26/32/c/2/A",
            symbol="phase_in_rate",
        )
        assert bundle.bindings[0].effective_date == date(2025, 1, 1)
        assert bundle.bindings[0].binding == RuleBinding(
            values={0: 0.34},
            source="bundle://ty2025",
        )

    def test_merge_rule_bindings_later_source_wins_per_target_and_date(self):
        """Later sources override earlier values for the same dated target."""
        merged = merge_rule_bindings(
            {
                "bindings": [
                    {
                        "module_identity": "shared",
                        "symbol": "rate",
                        "effective_date": "2025-01-01",
                        "values": {"0": 0.2, "1": 0.25},
                    }
                ]
            },
            {
                "bindings": [
                    {
                        "module_identity": "shared",
                        "symbol": "rate",
                        "effective_date": "2025-01-01",
                        "values": {"1": 0.3},
                        "source": "bundle://override",
                    }
                ]
            },
        )

        binding = merged.to_resolver().resolve(
            module_identity="shared",
            symbol="rate",
            effective_date=date(2025, 1, 1),
        )

        assert binding == RuleBinding(
            values={0: 0.2, 1: 0.3},
            source="bundle://override",
        )

    def test_rule_resolver_requires_effective_date_for_dated_only_bindings(self):
        """Dated bindings fail loudly when compilation omits an effective date."""
        resolver = merge_rule_bindings(
            {
                "bindings": [
                    {
                        "module_identity": "shared",
                        "symbol": "rate",
                        "effective_date": "2025-01-01",
                        "value": 0.3,
                    }
                ]
            }
        ).to_resolver()

        with pytest.raises(RuleBindingError, match="has only effective-dated bindings"):
            resolver.resolve(module_identity="shared", symbol="rate")

    def test_load_rule_binding_file_rejects_removed_override_artifacts(self, tmp_path):
        """Removed override-artifact YAML fails with migration guidance."""
        path = tmp_path / "artifact.yaml"
        path.write_text(
            """
source:
  document: "Rev. Proc. 2023-34"
  section: "3.06"
  url: "https://www.irs.gov/example"
  effective_date: 2024-01-01

earned_income_amount:
  implements: statutes/26/32/j/1
  overrides: statutes/26/32/b/2/A/base_amounts#earned_income_amount
  indexed_by: num_qualifying_children
  values:
    0: 8260
    1: 12390
"""
        )

        with pytest.raises(RuleBindingError, match="removed override-artifact syntax"):
            load_rule_bindings_file(path)
