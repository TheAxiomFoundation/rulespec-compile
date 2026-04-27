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
                            "module_identity": "statute/26/32/c/2/A",
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
            module_identity="statute/26/32/c/2/A",
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

    def test_load_rule_binding_file_supports_rulespec_us_override_yaml(self, tmp_path):
        """RuleSpec override YAML loads into dated identity-aware bindings."""
        path = tmp_path / "eitc-2024.yaml"
        path.write_text(
            """
source:
  document: "Rev. Proc. 2023-34"
  section: "3.06"
  url: "https://www.irs.gov/example"
  effective_date: 2024-01-01

earned_income_amount:
  implements: statute/26/32/j/1
  overrides: statute/26/32/b/2/A/base_amounts#earned_income_amount
  indexed_by: num_qualifying_children
  values:
    0: 8260
    1: 12390
"""
        )

        bundle = load_rule_bindings_file(path)

        assert len(bundle.bindings) == 1
        entry = bundle.bindings[0]
        assert entry.target == RuleBindingTarget(
            module_identity="statute/26/32/b/2/A/base_amounts",
            symbol="earned_income_amount",
        )
        assert entry.effective_date == date(2024, 1, 1)
        assert entry.binding == RuleBinding(
            values={0: 8260.0, 1: 12390.0},
            source="Rev. Proc. 2023-34 § 3.06",
            reference="statute/26/32/j/1; https://www.irs.gov/example",
        )

    def test_load_rule_binding_file_supports_rulespec_override_artifact(self, tmp_path):
        """Override-style .yaml artifacts strip prose blocks before loading."""
        path = tmp_path / "artifact.yaml"
        path.write_text(
            '''
"""
Artifact prose block that is not part of the data payload.
"""

source:
  title: "Rev. Proc. 2023-34"
  effective_date: 2024-01-01

joint_return_adjustment:
  overrides: statute/26/32/b/2/B/base_joint_return_adjustment#joint_return_adjustment
  value: 6920
'''
        )

        bundle = load_rule_bindings_file(path)

        assert len(bundle.bindings) == 1
        entry = bundle.bindings[0]
        assert entry.target == RuleBindingTarget(
            module_identity="statute/26/32/b/2/B/base_joint_return_adjustment",
            symbol="joint_return_adjustment",
        )
        assert entry.binding == RuleBinding(
            values={0: 6920.0},
            source="Rev. Proc. 2023-34",
        )

    def test_load_rule_binding_file_rejects_non_integer_override_indices(
        self, tmp_path
    ):
        """Rate-table override artifacts fail until non-integer tables are supported."""
        path = tmp_path / "tax-brackets-2024.yaml"
        path.write_text(
            """
source:
  document: "Rev. Proc. 2023-34"

single:
  overrides: statute/26/1/brackets/base_thresholds#single
  indexed_by: rate
  values:
    0.10: 0
    0.12: 11600
"""
        )

        with pytest.raises(RuleBindingError, match="integer indices"):
            load_rule_bindings_file(path)
