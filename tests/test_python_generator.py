"""
Tests for Python code generation from RuleSpec.

Following TDD: write tests first, then implement.
"""


class TestPythonCodeGenerator:
    """Test PythonCodeGenerator initialization and setup."""

    def test_init_defaults(self):
        """Test generator initializes with sensible defaults."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator()
        assert gen.module_name == "calculator"
        assert gen.include_provenance is True
        assert gen.type_hints is True
        assert gen.parameters == {}
        assert gen.variables == []
        assert gen.inputs == {}

    def test_add_input_number(self):
        """Test adding numeric input."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator()
        gen.add_input("income", 0, "float")
        assert "income" in gen.inputs
        assert gen.inputs["income"]["default"] == 0
        assert gen.inputs["income"]["type"] == "float"

    def test_add_input_boolean(self):
        """Test adding boolean input."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator()
        gen.add_input("is_joint", False, "bool")
        assert gen.inputs["is_joint"]["default"] is False
        assert gen.inputs["is_joint"]["type"] == "bool"

    def test_add_parameter(self):
        """Test adding parameter with values."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator()
        gen.add_parameter("rate", {0: 10, 1: 20}, "26 USC 1")
        assert "rate" in gen.parameters
        assert gen.parameters["rate"].values == {0: 10, 1: 20}
        assert gen.parameters["rate"].source == "26 USC 1"

    def test_add_variable(self):
        """Test adding calculated variable."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator()
        gen.add_variable(
            "tax",
            ["income"],
            "income * PARAMS['rate'][0] / 100",
            label="Tax Amount",
            citation="26 USC 1(a)",
        )
        assert len(gen.variables) == 1
        assert gen.variables[0].name == "tax"
        assert gen.variables[0].formula_python == "income * PARAMS['rate'][0] / 100"


class TestGenerateOutput:
    """Test Python code generation output."""

    def test_generate_includes_header(self):
        """Generated code has docstring header."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator(module_name="Test Calculator")
        gen.add_input("x", 0)
        gen.add_variable("y", ["x"], "x * 2")
        code = gen.generate()

        assert '"""' in code
        assert "Test Calculator" in code
        assert "Auto-generated from RuleSpec" in code

    def test_generate_includes_params_dict(self):
        """Generated code includes PARAMS dictionary."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator()
        gen.add_parameter("credit_pct", {0: 7.65, 1: 34}, "26 USC 32(b)(1)")
        code = gen.generate()

        assert "PARAMS = {" in code
        assert '"credit_pct": {0: 7.65, 1: 34}' in code
        assert "26 USC 32(b)(1)" in code

    def test_generate_includes_calculate_function(self):
        """Generated code includes calculate function."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator()
        gen.add_input("income", 0, "float")
        gen.add_variable("tax", ["income"], "income * 0.2")
        code = gen.generate()

        assert "def calculate(" in code
        assert "income: float = 0" in code
        assert "return {" in code

    def test_generate_supports_public_input_names_that_are_not_identifiers(self):
        """Qualified public input names fall back to mapping-based Python inputs."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator()
        gen.add_input(
            "shared_income",
            0,
            "float",
            public_name="shared.rate.income",
        )
        gen.add_variable("tax", ["shared_income"], "shared_income * 2")
        code = gen.generate()
        namespace: dict[str, object] = {}

        exec(code, namespace)
        result = namespace["calculate"](**{"shared.rate.income": 5})

        assert "def calculate(inputs: dict[str, Any] | None = None" in code
        assert result["tax"] == 10

    def test_generate_returns_citations(self):
        """Generated code returns citation chain."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator()
        gen.add_parameter("rate", {0: 20}, "26 USC 1")
        gen.add_input("income", 0)
        gen.add_variable("tax", ["income"], "income * 0.2", citation="26 USC 1(a)")
        code = gen.generate()

        assert '"citations":' in code
        assert '"param": "rate"' in code or "'param': 'rate'" in code
        assert '"variable": "tax"' in code or "'variable': 'tax'" in code

    def test_generate_includes_module_identity_in_citations(self):
        """Generated Python citations keep the leaf-derived source rule identity."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator()
        gen.add_parameter("rate", {0: 20}, "26 USC 1", module_identity="shared")
        gen.add_input("income", 0)
        gen.add_variable(
            "tax",
            ["income"],
            "income * 0.2",
            citation="26 USC 1(a)",
            module_identity="benefit_amount",
        )
        code = gen.generate()

        assert '"module_identity": "shared"' in code
        assert '"module_identity": "benefit_amount"' in code

    def test_generate_no_provenance(self):
        """Can generate without provenance comments."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator(include_provenance=False)
        gen.add_parameter("rate", {0: 20}, "26 USC 1")
        code = gen.generate()

        assert "Sources:" not in code

    def test_generate_without_type_hints(self):
        """Can generate Python without type hints."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator(type_hints=False)
        gen.add_input("income", 0, "float")
        gen.add_variable("tax", ["income"], "income * 0.2")
        code = gen.generate()

        # Should not have type annotations
        assert ": float" not in code or "income: float = 0" not in code


class TestGenerateEITCCalculator:
    """Test pre-built EITC calculator generation."""

    def test_returns_valid_python(self):
        """Generated EITC calculator is valid Python."""
        from src.rulespec_compile.python_generator import generate_eitc_calculator

        code = generate_eitc_calculator()
        assert code
        # Should be valid Python syntax
        compile(code, "<string>", "exec")

    def test_includes_all_eitc_params(self):
        """EITC calculator includes all required parameters."""
        from src.rulespec_compile.python_generator import generate_eitc_calculator

        code = generate_eitc_calculator()
        assert "credit_pct" in code
        assert "phaseout_pct" in code
        assert "earned_income_amount" in code
        assert "phaseout_single" in code
        assert "phaseout_joint" in code

    def test_includes_statute_citations(self):
        """EITC calculator includes 26 USC 32 citations."""
        from src.rulespec_compile.python_generator import generate_eitc_calculator

        code = generate_eitc_calculator()
        assert "26 USC 32" in code
        assert "26 USC 32(b)(1)" in code

    def test_includes_guidance_citations(self):
        """EITC calculator includes IRS guidance citations."""
        from src.rulespec_compile.python_generator import generate_eitc_calculator

        code = generate_eitc_calculator()
        assert "Rev. Proc. 2024-40" in code


class TestPythonExecution:
    """Test that generated Python code executes correctly."""

    def test_generated_code_is_executable(self):
        """Generated code can be executed."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator()
        gen.add_input("x", 5, "int")
        gen.add_variable("y", ["x"], "x * 2")
        code = gen.generate()

        # Execute and test
        namespace = {}
        exec(code, namespace)
        assert "calculate" in namespace
        result = namespace["calculate"](x=10)
        assert result["y"] == 20

    def test_multiline_formula_returns_trailing_expression(self):
        """Multiline formulas implicitly return a trailing Python expression."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator()
        gen.add_input("x", 0, "int")
        gen.add_variable("y", ["x"], "tmp = x + 1\ntmp")
        namespace = {}

        exec(gen.generate(), namespace)

        assert namespace["calculate"](x=10)["y"] == 11

    def test_semicolon_block_returns_trailing_expression(self):
        """Same-line semicolon blocks are normalized before Python emission."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator()
        gen.add_input("x", 0, "int")
        gen.add_variable("y", ["x"], "tmp = x + 1; tmp")
        namespace = {}

        exec(gen.generate(), namespace)

        assert namespace["calculate"](x=10)["y"] == 11

    def test_block_preserves_explicit_return_without_space(self):
        """Explicit Python returns like return(x) stay valid in wrapped blocks."""
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        gen = PythonCodeGenerator()
        gen.add_input("x", 0, "int")
        gen.add_variable("y", ["x"], "tmp = x + 1\nreturn(x + tmp)")
        namespace = {}

        exec(gen.generate(), namespace)

        assert namespace["calculate"](x=10)["y"] == 21

    def test_eitc_calculator_execution(self):
        """EITC calculator executes and returns reasonable values."""
        from src.rulespec_compile.python_generator import generate_eitc_calculator

        code = generate_eitc_calculator()
        namespace = {}
        exec(code, namespace)

        # Test zero income
        result = namespace["calculate"](
            earned_income=0, agi=0, n_children=0, is_joint=False
        )
        assert result["eitc"] == 0

        # Test typical case
        result = namespace["calculate"](
            earned_income=15000,
            agi=15000,
            n_children=1,
            is_joint=False,
        )
        assert result["eitc"] > 0
        assert "citations" in result
        assert len(result["citations"]) > 0


class TestPythonVsJS:
    """Test that Python output matches JS output for same inputs."""

    def test_simple_calculation_matches_js(self):
        """Python and JS generators produce equivalent results."""
        from src.rulespec_compile.js_generator import JSCodeGenerator
        from src.rulespec_compile.python_generator import PythonCodeGenerator

        # Python version
        py_gen = PythonCodeGenerator()
        py_gen.add_input("x", 0, "float")
        py_gen.add_parameter("multiplier", {0: 2.5}, "Test")
        py_gen.add_variable("y", ["x"], "x * PARAMS['multiplier'][0]")
        py_code = py_gen.generate()

        # JS version (with adapted syntax)
        js_gen = JSCodeGenerator()
        js_gen.add_input("x", 0, "number")
        js_gen.add_parameter("multiplier", {0: 2.5}, "Test")
        js_gen.add_variable("y", ["x"], "x * PARAMS.multiplier[0]")
        js_gen.generate()

        # Execute Python
        py_namespace = {}
        exec(py_code, py_namespace)
        py_result = py_namespace["calculate"](x=10)

        # Both should produce y = 25
        assert py_result["y"] == 25
