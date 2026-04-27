"""Tests for parameter binding helpers."""

import json

import pytest

from src.rulespec_compile.parameter_bindings import (
    ParameterBinding,
    ParameterBindingError,
    ParameterBundle,
    load_parameter_overrides_file,
    merge_parameter_overrides,
    normalize_parameter_overrides,
)


class TestParameterBindings:
    """Tests for parameter binding helpers."""

    def test_load_parameter_file_supports_parameters_envelope(self, tmp_path):
        """Parameter files can wrap bindings in a top-level parameters object."""
        file_path = tmp_path / "bindings.json"
        file_path.write_text(json.dumps({"parameters": {"rate": [0.2, 0.25]}}))

        loaded = load_parameter_overrides_file(file_path)

        assert loaded == ParameterBundle(
            schema_version=1,
            parameters={"rate": ParameterBinding(values={0: 0.2, 1: 0.25})},
            metadata={},
        )

    def test_load_structured_parameter_bundle_preserves_metadata(self, tmp_path):
        """Structured parameter bundles preserve parameter metadata."""
        file_path = tmp_path / "bindings.json"
        file_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "metadata": {"name": "TY2025 bundle"},
                    "parameters": {
                        "rate": {
                            "values": {"0": 0.25},
                            "source": "bundle://ty2025",
                            "unit": "rate",
                        }
                    },
                }
            )
        )

        loaded = load_parameter_overrides_file(file_path)

        assert loaded.schema_version == 1
        assert loaded.metadata == {"name": "TY2025 bundle"}
        assert loaded.parameters["rate"] == ParameterBinding(
            values={0: 0.25},
            source="bundle://ty2025",
            unit="rate",
        )

    def test_load_parameter_file_allows_metadata_name_in_plain_map(self, tmp_path):
        """Compatibility maps can use reserved-looking names as parameter keys."""
        file_path = tmp_path / "bindings.json"
        file_path.write_text(json.dumps({"metadata": 1, "schema_version": 2}))

        loaded = load_parameter_overrides_file(file_path)

        assert loaded == ParameterBundle(
            parameters={
                "metadata": ParameterBinding(values={0: 1.0}),
                "schema_version": ParameterBinding(values={0: 2.0}),
            }
        )

    def test_load_parameter_file_allows_parameters_name_in_plain_map(self, tmp_path):
        """Indexed dict payloads can still bind a parameter named parameters."""
        file_path = tmp_path / "bindings.json"
        file_path.write_text(json.dumps({"parameters": {"0": 10, "1": 20}}))

        loaded = load_parameter_overrides_file(file_path)

        assert loaded == ParameterBundle(
            parameters={
                "parameters": ParameterBinding(values={0: 10.0, 1: 20.0}),
            }
        )

    def test_load_structured_bundle_with_numeric_parameter_name(self, tmp_path):
        """Explicit schema bundles stay structured even with numeric parameter names."""
        file_path = tmp_path / "bindings.json"
        file_path.write_text(
            json.dumps({"schema_version": 1, "metadata": {}, "parameters": {"0": 0.2}})
        )

        loaded = load_parameter_overrides_file(file_path)

        assert loaded == ParameterBundle(
            schema_version=1,
            parameters={"0": ParameterBinding(values={0: 0.2})},
            metadata={},
        )

    def test_load_parameter_file_allows_parameters_binding_object_name(self, tmp_path):
        """A parameter literally named parameters can still use binding metadata."""
        file_path = tmp_path / "bindings.json"
        file_path.write_text(
            json.dumps({"parameters": {"value": 10, "source": "bundle://test"}})
        )

        loaded = load_parameter_overrides_file(file_path)

        assert loaded == ParameterBundle(
            parameters={
                "parameters": ParameterBinding(
                    values={0: 10.0},
                    source="bundle://test",
                )
            }
        )

    def test_load_parameter_file_allows_reserved_key_collisions_in_plain_map(
        self, tmp_path
    ):
        """Reserved-looking keys can coexist in a plain compatibility map."""
        file_path = tmp_path / "bindings.json"
        file_path.write_text(json.dumps({"parameters": {"0": 1}, "metadata": 2}))

        loaded = load_parameter_overrides_file(file_path)

        assert loaded == ParameterBundle(
            parameters={
                "parameters": ParameterBinding(values={0: 1.0}),
                "metadata": ParameterBinding(values={0: 2.0}),
            }
        )

    def test_load_parameter_file_rejects_ambiguous_reserved_shape(self, tmp_path):
        """Ambiguous schema_version-plus-parameters shapes fail loudly."""
        file_path = tmp_path / "bindings.json"
        file_path.write_text(json.dumps({"schema_version": 1, "parameters": {"0": 1}}))

        with pytest.raises(ParameterBindingError, match="ambiguous"):
            load_parameter_overrides_file(file_path)

    def test_load_structured_bundle_allows_parameter_named_source(self, tmp_path):
        """Structured bundles still support parameter names that look reserved."""
        file_path = tmp_path / "bindings.json"
        file_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "metadata": {},
                    "parameters": {
                        "source": {"value": 0.2},
                    },
                }
            )
        )

        loaded = load_parameter_overrides_file(file_path)

        assert loaded == ParameterBundle(
            schema_version=1,
            parameters={"source": ParameterBinding(values={0: 0.2})},
            metadata={},
        )

    def test_load_structured_bundle_allows_parameter_named_value(self, tmp_path):
        """Structured bundles support a parameter literally named value."""
        file_path = tmp_path / "bindings.json"
        file_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "metadata": {},
                    "parameters": {
                        "value": {"value": 0.2},
                    },
                }
            )
        )

        loaded = load_parameter_overrides_file(file_path)

        assert loaded == ParameterBundle(
            schema_version=1,
            parameters={"value": ParameterBinding(values={0: 0.2})},
            metadata={},
        )

    def test_load_structured_bundle_allows_multiple_reserved_like_names(self, tmp_path):
        """Structured bundles support multiple reserved-looking parameter names."""
        file_path = tmp_path / "bindings.json"
        file_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "metadata": {},
                    "parameters": {
                        "value": {"value": 0.2},
                        "metadata": {"value": 0.3},
                    },
                }
            )
        )

        loaded = load_parameter_overrides_file(file_path)

        assert loaded == ParameterBundle(
            schema_version=1,
            parameters={
                "value": ParameterBinding(values={0: 0.2}),
                "metadata": ParameterBinding(values={0: 0.3}),
            },
            metadata={},
        )

    def test_merge_parameter_overrides_later_source_wins(self):
        """Later override sources replace earlier values index-by-index."""
        merged = merge_parameter_overrides(
            {"rate": [0.2, 0.25], "threshold": 100},
            {"rate": {1: 0.3}},
        )

        assert merged == {
            "rate": ParameterBinding(values={0: 0.2, 1: 0.3}),
            "threshold": ParameterBinding(values={0: 100.0}),
        }

    def test_normalize_parameter_overrides_rejects_invalid_type(self):
        """Unsupported parameter override payloads raise a binding error."""
        with pytest.raises(
            ParameterBindingError, match="Unsupported parameter override"
        ):
            normalize_parameter_overrides({"rate": "bad"})

    def test_normalize_parameter_overrides_accepts_qualified_binding_keys(self):
        """Qualified module_identity.symbol keys are valid parameter overrides."""
        normalized = normalize_parameter_overrides({"shared.rate": 0.25})

        assert normalized == {
            "shared.rate": ParameterBinding(values={0: 0.25}),
        }

    def test_load_parameter_bundle_rejects_unknown_schema_version(self, tmp_path):
        """Unknown parameter bundle schema versions fail loudly."""
        file_path = tmp_path / "bindings.json"
        file_path.write_text(json.dumps({"schema_version": 2, "parameters": {}}))

        with pytest.raises(ParameterBindingError, match="unsupported schema_version"):
            load_parameter_overrides_file(file_path)

    def test_load_parameter_bundle_rejects_non_numeric_nested_indices(self, tmp_path):
        """Malformed nested dict bindings fail with a binding error."""
        file_path = tmp_path / "bindings.json"
        file_path.write_text(json.dumps({"rate": {"schema_version": 1}}))

        with pytest.raises(
            ParameterBindingError,
            match="must use numeric indices and numeric values",
        ):
            load_parameter_overrides_file(file_path)

    def test_load_parameter_bundle_rejects_non_numeric_list_values(self, tmp_path):
        """Malformed list payloads fail with a binding error."""
        file_path = tmp_path / "bindings.json"
        file_path.write_text(json.dumps({"rate": [1, "x"]}))

        with pytest.raises(ParameterBindingError, match="must use numeric values"):
            load_parameter_overrides_file(file_path)

    def test_load_structured_parameter_bundle_rejects_non_numeric_list_values(
        self, tmp_path
    ):
        """Structured list payloads fail with a binding error."""
        file_path = tmp_path / "bindings.json"
        file_path.write_text(json.dumps({"rate": {"values": [1, "x"]}}))

        with pytest.raises(ParameterBindingError, match="must use numeric values"):
            load_parameter_overrides_file(file_path)
