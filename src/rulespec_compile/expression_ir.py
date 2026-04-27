"""Validated expression IR for the generic RuleSpec compile subset."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass


class ExpressionParseError(ValueError):
    """Raised when a RuleSpec expression cannot be compiled safely."""


SUPPORTED_FUNCTIONS = {"abs", "ceil", "floor", "max", "min", "round"}
_ASSIGNMENT_PATTERN = re.compile(r"([A-Za-z_]\w*)\s*=\s*(?![=])(.+)")


@dataclass(frozen=True)
class LiteralExpr:
    """A literal constant."""

    value: bool | int | float | str


@dataclass(frozen=True)
class NameExpr:
    """An identifier reference."""

    name: str


@dataclass(frozen=True)
class SubscriptExpr:
    """An indexed expression."""

    value: Expression
    index: Expression


@dataclass(frozen=True)
class CallExpr:
    """A supported builtin function call."""

    function: str
    arguments: tuple[Expression, ...]


@dataclass(frozen=True)
class UnaryExpr:
    """A unary operation."""

    operator: str
    operand: Expression


@dataclass(frozen=True)
class BinaryExpr:
    """A binary arithmetic operation."""

    left: Expression
    operator: str
    right: Expression


@dataclass(frozen=True)
class BoolExpr:
    """A boolean conjunction or disjunction."""

    operator: str
    values: tuple[Expression, ...]


@dataclass(frozen=True)
class CompareExpr:
    """A comparison chain."""

    left: Expression
    operators: tuple[str, ...]
    comparators: tuple[Expression, ...]


@dataclass(frozen=True)
class ConditionalExpr:
    """A ternary conditional expression."""

    condition: Expression
    if_true: Expression
    if_false: Expression


Expression = (
    LiteralExpr
    | NameExpr
    | SubscriptExpr
    | CallExpr
    | UnaryExpr
    | BinaryExpr
    | BoolExpr
    | CompareExpr
    | ConditionalExpr
)


@dataclass(frozen=True)
class AssignStmt:
    """A simple local assignment."""

    name: str
    expression: Expression


@dataclass(frozen=True)
class ReturnStmt:
    """A return statement."""

    expression: Expression


@dataclass(frozen=True)
class IfStmt:
    """A conditional statement."""

    condition: Expression
    body: tuple[Statement, ...]
    orelse: tuple[Statement, ...]


Statement = AssignStmt | ReturnStmt | IfStmt


def parse_expression(expression: str, variable_name: str) -> Expression:
    """Parse a RuleSpec expression into validated IR."""
    normalized = _normalize_expression(expression)
    try:
        parsed = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ExpressionParseError(
            f"Variable '{variable_name}' has an unsupported expression "
            f"'{expression}': {exc.msg}."
        ) from exc
    return _from_python_ast(parsed.body, expression, variable_name)


def collect_references(expression: Expression) -> list[str]:
    """Collect identifier references in encounter order."""
    names: list[str] = []

    def visit(node: Expression) -> None:
        if isinstance(node, LiteralExpr):
            return
        if isinstance(node, NameExpr):
            if node.name not in names:
                names.append(node.name)
            return
        if isinstance(node, SubscriptExpr):
            visit(node.value)
            visit(node.index)
            return
        if isinstance(node, CallExpr):
            for argument in node.arguments:
                visit(argument)
            return
        if isinstance(node, UnaryExpr):
            visit(node.operand)
            return
        if isinstance(node, BinaryExpr):
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, BoolExpr):
            for value in node.values:
                visit(value)
            return
        if isinstance(node, CompareExpr):
            visit(node.left)
            for comparator in node.comparators:
                visit(comparator)
            return
        if isinstance(node, ConditionalExpr):
            visit(node.condition)
            visit(node.if_true)
            visit(node.if_false)
            return
        raise AssertionError(f"Unhandled expression node: {type(node).__name__}")

    visit(expression)
    return names


def map_expression_names(
    expression: Expression,
    mapper,
) -> Expression:
    """Rewrite identifier references in an expression tree."""
    if isinstance(expression, LiteralExpr):
        return expression
    if isinstance(expression, NameExpr):
        return NameExpr(mapper(expression.name))
    if isinstance(expression, SubscriptExpr):
        return SubscriptExpr(
            value=map_expression_names(expression.value, mapper),
            index=map_expression_names(expression.index, mapper),
        )
    if isinstance(expression, CallExpr):
        return CallExpr(
            function=expression.function,
            arguments=tuple(
                map_expression_names(argument, mapper)
                for argument in expression.arguments
            ),
        )
    if isinstance(expression, UnaryExpr):
        return UnaryExpr(
            operator=expression.operator,
            operand=map_expression_names(expression.operand, mapper),
        )
    if isinstance(expression, BinaryExpr):
        return BinaryExpr(
            left=map_expression_names(expression.left, mapper),
            operator=expression.operator,
            right=map_expression_names(expression.right, mapper),
        )
    if isinstance(expression, BoolExpr):
        return BoolExpr(
            operator=expression.operator,
            values=tuple(
                map_expression_names(value, mapper) for value in expression.values
            ),
        )
    if isinstance(expression, CompareExpr):
        return CompareExpr(
            left=map_expression_names(expression.left, mapper),
            operators=expression.operators,
            comparators=tuple(
                map_expression_names(comparator, mapper)
                for comparator in expression.comparators
            ),
        )
    if isinstance(expression, ConditionalExpr):
        return ConditionalExpr(
            condition=map_expression_names(expression.condition, mapper),
            if_true=map_expression_names(expression.if_true, mapper),
            if_false=map_expression_names(expression.if_false, mapper),
        )
    raise AssertionError(f"Unhandled expression node: {type(expression).__name__}")


def map_statement_names(
    statements: tuple[Statement, ...],
    mapper,
) -> tuple[Statement, ...]:
    """Rewrite identifier references in a statement block."""

    def rewrite(statement: Statement) -> Statement:
        if isinstance(statement, AssignStmt):
            return AssignStmt(
                name=statement.name,
                expression=map_expression_names(statement.expression, mapper),
            )
        if isinstance(statement, ReturnStmt):
            return ReturnStmt(map_expression_names(statement.expression, mapper))
        if isinstance(statement, IfStmt):
            return IfStmt(
                condition=map_expression_names(statement.condition, mapper),
                body=tuple(rewrite(nested) for nested in statement.body),
                orelse=tuple(rewrite(nested) for nested in statement.orelse),
            )
        raise AssertionError(f"Unhandled statement node: {type(statement).__name__}")

    return tuple(rewrite(statement) for statement in statements)


def render_expression_js(expression: Expression, parameter_names: set[str]) -> str:
    """Render expression IR to JavaScript."""
    return _render(expression, parameter_names, target="js")


def render_expression_python(expression: Expression, parameter_names: set[str]) -> str:
    """Render expression IR to Python."""
    return _render(expression, parameter_names, target="python")


def parse_formula_statements(formula: str, variable_name: str) -> tuple[Statement, ...]:
    """Parse a RuleSpec formula block into validated statements."""
    normalized = _normalize_formula_block(formula)
    try:
        module = ast.parse(normalized, mode="exec")
    except SyntaxError as exc:
        raise ExpressionParseError(
            f"Variable '{variable_name}' has an unsupported formula statement "
            f"near line {exc.lineno}: {exc.msg}."
        ) from exc

    return _sequence_from_python_ast(module.body, normalized, variable_name)


def render_statement_block_js(
    statements: tuple[Statement, ...],
    parameter_names: set[str],
    indent: str = "",
) -> list[str]:
    """Render statement IR to JavaScript lines."""
    return _render_statement_block(statements, parameter_names, "js", indent)


def render_statement_block_python(
    statements: tuple[Statement, ...],
    parameter_names: set[str],
    indent: str = "",
) -> list[str]:
    """Render statement IR to Python lines."""
    return _render_statement_block(statements, parameter_names, "python", indent)


def collect_assigned_names(statements: tuple[Statement, ...]) -> list[str]:
    """Collect assigned local names in encounter order."""
    names: list[str] = []

    def visit(statement: Statement) -> None:
        if isinstance(statement, AssignStmt):
            if statement.name not in names:
                names.append(statement.name)
            return
        if isinstance(statement, ReturnStmt):
            return
        if isinstance(statement, IfStmt):
            for nested in statement.body:
                visit(nested)
            for nested in statement.orelse:
                visit(nested)
            return
        raise AssertionError(f"Unhandled statement node: {type(statement).__name__}")

    for statement in statements:
        visit(statement)
    return names


def formula_has_branching(statements: tuple[Statement, ...]) -> bool:
    """Return whether the formula contains any conditional control flow."""
    for statement in statements:
        if isinstance(statement, IfStmt):
            return True
    return False


def is_straight_line_formula(statements: tuple[Statement, ...]) -> bool:
    """Return whether the formula is assignment-only plus a final return."""
    return (
        bool(statements)
        and all(isinstance(statement, AssignStmt) for statement in statements[:-1])
        and isinstance(statements[-1], ReturnStmt)
    )


def _statement_from_python_ast(
    node: ast.stmt,
    normalized_formula: str,
    variable_name: str,
    is_terminal: bool = False,
) -> Statement:
    """Convert a validated Python AST statement into statement IR."""
    source = ast.get_source_segment(normalized_formula, node) or node.__class__.__name__

    if isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise ExpressionParseError(
                f"Variable '{variable_name}' uses unsupported assignment target in "
                f"'{source}'. Only simple local names are supported."
            )
        return AssignStmt(
            name=node.targets[0].id,
            expression=_from_python_ast(node.value, source, variable_name),
        )

    if isinstance(node, ast.Return):
        if node.value is None:
            raise ExpressionParseError(
                f"Variable '{variable_name}' has a bare return in '{source}'. "
                "Returns must include a value."
            )
        return ReturnStmt(_from_python_ast(node.value, source, variable_name))

    if isinstance(node, ast.Expr):
        if is_terminal:
            return ReturnStmt(_from_python_ast(node.value, source, variable_name))
        raise ExpressionParseError(
            f"Variable '{variable_name}' uses unsupported statement 'expr' in "
            f"'{source}'. Bare expressions are only supported as the final "
            "statement in a formula block."
        )

    if isinstance(node, ast.If):
        return IfStmt(
            condition=_from_python_ast(node.test, source, variable_name),
            body=_sequence_from_python_ast(
                node.body,
                normalized_formula,
                variable_name,
            ),
            orelse=_sequence_from_python_ast(
                node.orelse,
                normalized_formula,
                variable_name,
            ),
        )

    if isinstance(node, (ast.Import, ast.ImportFrom)):
        raise ExpressionParseError(
            f"Variable '{variable_name}' imports code in '{source}'. Imports are not "
            "supported by generic compilation."
        )

    kind = node.__class__.__name__.lower()
    raise ExpressionParseError(
        f"Variable '{variable_name}' uses unsupported statement '{kind}' in "
        f"'{source}'. Generic compilation currently supports assignments, "
        "if/elif/else, and return."
    )


def _sequence_from_python_ast(
    nodes: list[ast.stmt],
    normalized_formula: str,
    variable_name: str,
) -> tuple[Statement, ...]:
    """Convert a sequence of Python statements into statement IR."""
    return tuple(
        _statement_from_python_ast(
            node,
            normalized_formula,
            variable_name,
            is_terminal=index == len(nodes) - 1,
        )
        for index, node in enumerate(nodes)
    )


def _from_python_ast(node: ast.AST, original: str, variable_name: str) -> Expression:
    """Convert a validated Python AST node into expression IR."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (bool, int, float, str)):
            return LiteralExpr(node.value)
        raise ExpressionParseError(
            f"Variable '{variable_name}' uses unsupported literal syntax in "
            f"'{original}'."
        )

    if isinstance(node, ast.Name):
        return NameExpr(node.id)

    if isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Slice):
            raise ExpressionParseError(
                f"Variable '{variable_name}' uses slice syntax in '{original}'. "
                "Slices are not supported by generic compilation."
            )
        return SubscriptExpr(
            value=_from_python_ast(node.value, original, variable_name),
            index=_from_python_ast(node.slice, original, variable_name),
        )

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionParseError(
                f"Variable '{variable_name}' uses an unsupported call target in "
                f"'{original}'. Only builtin scalar helpers are supported."
            )
        if node.func.id not in SUPPORTED_FUNCTIONS:
            raise ExpressionParseError(
                f"Variable '{variable_name}' calls unsupported function "
                f"'{node.func.id}' in '{original}'. Supported functions are "
                "abs, ceil, floor, max, min, and round."
            )
        if node.keywords:
            raise ExpressionParseError(
                f"Variable '{variable_name}' uses keyword arguments in "
                f"'{original}'. Keyword arguments are not supported."
            )
        return CallExpr(
            function=node.func.id,
            arguments=tuple(
                _from_python_ast(argument, original, variable_name)
                for argument in node.args
            ),
        )

    if isinstance(node, ast.UnaryOp):
        operator = {
            ast.Not: "not",
            ast.UAdd: "+",
            ast.USub: "-",
        }.get(type(node.op))
        if operator is None:
            raise ExpressionParseError(
                f"Variable '{variable_name}' uses unsupported unary syntax in "
                f"'{original}'."
            )
        return UnaryExpr(
            operator=operator,
            operand=_from_python_ast(node.operand, original, variable_name),
        )

    if isinstance(node, ast.BinOp):
        operator = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
            ast.Pow: "**",
        }.get(type(node.op))
        if operator is None:
            raise ExpressionParseError(
                f"Variable '{variable_name}' uses unsupported arithmetic syntax in "
                f"'{original}'."
            )
        return BinaryExpr(
            left=_from_python_ast(node.left, original, variable_name),
            operator=operator,
            right=_from_python_ast(node.right, original, variable_name),
        )

    if isinstance(node, ast.BoolOp):
        operator = {ast.And: "and", ast.Or: "or"}.get(type(node.op))
        if operator is None:
            raise ExpressionParseError(
                f"Variable '{variable_name}' uses unsupported boolean syntax in "
                f"'{original}'."
            )
        return BoolExpr(
            operator=operator,
            values=tuple(
                _from_python_ast(value, original, variable_name)
                for value in node.values
            ),
        )

    if isinstance(node, ast.Compare):
        operators: list[str] = []
        for operator_node in node.ops:
            operator = {
                ast.Eq: "==",
                ast.NotEq: "!=",
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
            }.get(type(operator_node))
            if operator is None:
                raise ExpressionParseError(
                    f"Variable '{variable_name}' uses unsupported comparison syntax "
                    f"in '{original}'."
                )
            operators.append(operator)
        return CompareExpr(
            left=_from_python_ast(node.left, original, variable_name),
            operators=tuple(operators),
            comparators=tuple(
                _from_python_ast(comparator, original, variable_name)
                for comparator in node.comparators
            ),
        )

    if isinstance(node, ast.IfExp):
        return ConditionalExpr(
            condition=_from_python_ast(node.test, original, variable_name),
            if_true=_from_python_ast(node.body, original, variable_name),
            if_false=_from_python_ast(node.orelse, original, variable_name),
        )

    if isinstance(node, ast.Attribute):
        return NameExpr(_attribute_to_name(node, original, variable_name))

    raise ExpressionParseError(
        f"Variable '{variable_name}' uses unsupported expression syntax '{original}'."
    )


def _render(expression: Expression, parameter_names: set[str], target: str) -> str:
    """Render expression IR to the requested target language."""
    return _render_with_precedence(expression, parameter_names, target, parent_prec=0)


def _render_statement_block(
    statements: tuple[Statement, ...],
    parameter_names: set[str],
    target: str,
    indent: str,
) -> list[str]:
    """Render statements to target-specific source lines."""
    lines: list[str] = []
    for statement in statements:
        lines.extend(_render_statement(statement, parameter_names, target, indent))
    return lines


def _render_statement(
    statement: Statement,
    parameter_names: set[str],
    target: str,
    indent: str,
) -> list[str]:
    """Render one statement to target-specific source lines."""
    if isinstance(statement, AssignStmt):
        expression = _render(statement.expression, parameter_names, target)
        suffix = ";" if target == "js" else ""
        return [f"{indent}{statement.name} = {expression}{suffix}"]

    if isinstance(statement, ReturnStmt):
        expression = _render(statement.expression, parameter_names, target)
        suffix = ";" if target == "js" else ""
        return [f"{indent}return {expression}{suffix}"]

    if isinstance(statement, IfStmt):
        if target == "js":
            return _render_if_js(statement, parameter_names, indent)
        return _render_if_python(statement, parameter_names, indent)

    raise AssertionError(f"Unhandled statement node: {type(statement).__name__}")


def _render_if_js(
    statement: IfStmt,
    parameter_names: set[str],
    indent: str,
) -> list[str]:
    """Render a conditional statement to JavaScript."""
    condition = _render(statement.condition, parameter_names, "js")
    lines = [f"{indent}if ({condition}) {{"]
    lines.extend(
        _render_statement_block(
            statement.body,
            parameter_names,
            "js",
            indent + "  ",
        )
    )
    if statement.orelse:
        if len(statement.orelse) == 1 and isinstance(statement.orelse[0], IfStmt):
            nested = _render_if_js(statement.orelse[0], parameter_names, indent)
            first_nested = nested[0][len(indent) :]
            lines.append(f"{indent}}} else {first_nested}")
            lines.extend(nested[1:])
        else:
            lines.append(f"{indent}}} else {{")
            lines.extend(
                _render_statement_block(
                    statement.orelse,
                    parameter_names,
                    "js",
                    indent + "  ",
                )
            )
            lines.append(f"{indent}}}")
    else:
        lines.append(f"{indent}}}")
    return lines


def _render_if_python(
    statement: IfStmt,
    parameter_names: set[str],
    indent: str,
) -> list[str]:
    """Render a conditional statement to Python."""
    condition = _render(statement.condition, parameter_names, "python")
    lines = [f"{indent}if {condition}:"]
    lines.extend(
        _render_statement_block(
            statement.body,
            parameter_names,
            "python",
            indent + "    ",
        )
    )
    if statement.orelse:
        if len(statement.orelse) == 1 and isinstance(statement.orelse[0], IfStmt):
            nested = _render_if_python(statement.orelse[0], parameter_names, indent)
            first_nested = nested[0][len(indent) + len("if ") :]
            lines.append(f"{indent}elif {first_nested}")
            lines.extend(nested[1:])
        else:
            lines.append(f"{indent}else:")
            lines.extend(
                _render_statement_block(
                    statement.orelse,
                    parameter_names,
                    "python",
                    indent + "    ",
                )
            )
    return lines


def _render_with_precedence(
    expression: Expression,
    parameter_names: set[str],
    target: str,
    parent_prec: int,
) -> str:
    """Render an expression, wrapping only when required for precedence."""
    if isinstance(expression, LiteralExpr):
        return _render_literal(expression.value, target)

    if isinstance(expression, NameExpr):
        if expression.name in parameter_names:
            return _scalar_parameter_reference(expression.name, target)
        return expression.name

    if isinstance(expression, SubscriptExpr):
        if (
            isinstance(expression.value, NameExpr)
            and expression.value.name in parameter_names
        ):
            index = _render_with_precedence(
                expression.index, parameter_names, target, parent_prec=0
            )
            return _indexed_parameter_reference(expression.value.name, index, target)
        rendered = (
            f"{_render_with_precedence(expression.value, parameter_names, target, 90)}"
            f"[{_render_with_precedence(expression.index, parameter_names, target, 0)}]"
        )
        return _wrap(rendered, 90, parent_prec)

    if isinstance(expression, CallExpr):
        function_name = _render_function_name(expression.function, target)
        arguments = ", ".join(
            _render_with_precedence(argument, parameter_names, target, 0)
            for argument in expression.arguments
        )
        return _wrap(f"{function_name}({arguments})", 90, parent_prec)

    if isinstance(expression, UnaryExpr):
        operator = expression.operator
        operand_prec = 80 if operator in {"+", "-"} else 40
        operand = _render_with_precedence(
            expression.operand,
            parameter_names,
            target,
            parent_prec=operand_prec,
        )
        if operator == "not":
            rendered = f"{'!' if target == 'js' else 'not '}{operand}"
        else:
            rendered = f"{operator}{operand}"
        return _wrap(rendered, operand_prec, parent_prec)

    if isinstance(expression, BinaryExpr):
        precedence = {"+": 60, "-": 60, "*": 70, "/": 70, "%": 70, "**": 85}[
            expression.operator
        ]
        left_prec = precedence + 1 if expression.operator == "**" else precedence
        right_prec = precedence if expression.operator == "**" else precedence + 1
        left = _render_with_precedence(
            expression.left, parameter_names, target, left_prec
        )
        right = _render_with_precedence(
            expression.right, parameter_names, target, right_prec
        )
        rendered = f"{left} {expression.operator} {right}"
        return _wrap(rendered, precedence, parent_prec)

    if isinstance(expression, BoolExpr):
        precedence = 20 if expression.operator == "or" else 30
        operator = (
            " || "
            if target == "js" and expression.operator == "or"
            else (" && " if target == "js" else f" {expression.operator} ")
        )
        rendered = operator.join(
            _render_with_precedence(value, parameter_names, target, precedence + 1)
            for value in expression.values
        )
        return _wrap(rendered, precedence, parent_prec)

    if isinstance(expression, CompareExpr):
        precedence = 50
        parts = [
            _render_with_precedence(
                expression.left,
                parameter_names,
                target,
                precedence,
            )
        ]
        for operator, comparator in zip(
            expression.operators, expression.comparators, strict=True
        ):
            parts.append(operator)
            parts.append(
                _render_with_precedence(
                    comparator, parameter_names, target, precedence + 1
                )
            )
        return _wrap(" ".join(parts), precedence, parent_prec)

    if isinstance(expression, ConditionalExpr):
        precedence = 10
        condition = _render_with_precedence(
            expression.condition, parameter_names, target, 30
        )
        if_true = _render_with_precedence(
            expression.if_true, parameter_names, target, precedence
        )
        if_false = _render_with_precedence(
            expression.if_false, parameter_names, target, precedence + 1
        )
        if target == "js":
            rendered = f"{condition} ? {if_true} : {if_false}"
        else:
            rendered = f"{if_true} if {condition} else {if_false}"
        return _wrap(rendered, precedence, parent_prec)

    raise AssertionError(f"Unhandled expression node: {type(expression).__name__}")


def _render_literal(value: bool | int | float | str, target: str) -> str:
    """Render a literal to JS or Python syntax."""
    if isinstance(value, bool):
        if target == "js":
            return "true" if value else "false"
        return "True" if value else "False"
    if isinstance(value, str):
        return json.dumps(value) if target == "js" else repr(value)
    return str(value)


def _render_function_name(function: str, target: str) -> str:
    """Render a supported builtin helper."""
    if target == "js":
        return {
            "abs": "Math.abs",
            "ceil": "Math.ceil",
            "floor": "Math.floor",
            "max": "Math.max",
            "min": "Math.min",
            "round": "Math.round",
        }[function]
    return {"ceil": "math.ceil", "floor": "math.floor"}.get(function, function)


def _scalar_parameter_reference(name: str, target: str) -> str:
    """Render a scalar parameter reference."""
    if target == "js":
        return f"PARAMS.{name}[0]"
    return f'PARAMS["{name}"][0]'


def _indexed_parameter_reference(name: str, index: str, target: str) -> str:
    """Render an indexed parameter reference."""
    if target == "js":
        return f"PARAMS.{name}[{index}]"
    return f'PARAMS["{name}"][{index}]'


def _wrap(rendered: str, precedence: int, parent_prec: int) -> str:
    """Wrap an expression if its precedence is lower than its parent's."""
    if precedence < parent_prec:
        return f"({rendered})"
    return rendered


def _normalize_formula_block(formula: str) -> str:
    """Normalize a RuleSpec formula block into Python-like source for AST parsing."""
    normalized_lines: list[str] = []
    stripped_lines = [
        _strip_formula_comment(raw_line) for raw_line in formula.splitlines()
    ]
    expanded_lines = _expand_inline_if_statement_lines(stripped_lines)
    collapsed_lines = _collapse_inline_if_formula_lines(expanded_lines)
    for line in _collapse_expression_continuation_lines(collapsed_lines):
        if not line.strip():
            normalized_lines.append("")
            continue

        indent = line[: len(line) - len(line.lstrip())]
        stripped = line[len(indent) :]
        normalized_lines.append(indent + _normalize_statement_line(stripped))
    return "\n".join(normalized_lines)


def _normalize_statement_line(line: str) -> str:
    """Normalize a single RuleSpec statement line."""
    stripped = line.strip()
    if stripped.startswith("let "):
        stripped = stripped[len("let ") :].strip()

    if stripped.startswith("else if "):
        stripped = "elif " + stripped[len("else if ") :]

    if stripped == "else:":
        return stripped

    if stripped.startswith("return "):
        return f"return {_normalize_expression(stripped[len('return ') :])}"

    if stripped.endswith(":"):
        head, _, tail = stripped[:-1].partition(" ")
        if tail:
            return f"{head} {_normalize_expression(tail)}:"
        return stripped

    match = _ASSIGNMENT_PATTERN.fullmatch(stripped)
    if match:
        return f"{match.group(1)} = {_normalize_expression(match.group(2))}"

    return _normalize_expression(stripped)


def _normalize_expression(expression: str) -> str:
    """Convert RuleSpec expression syntax into parseable Python syntax."""
    stripped = expression.strip()
    return _rewrite_js_tokens(_convert_ternary(_convert_inline_if_expression(stripped)))


def _collapse_inline_if_formula_lines(lines: list[str]) -> list[str]:
    """Collapse chained inline RuleSpec conditionals into one expression line."""
    collapsed: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            collapsed.append(line)
            index += 1
            continue

        indent = line[: len(line) - len(line.lstrip())]
        if not stripped.startswith("if "):
            collapsed.append(line)
            index += 1
            continue

        parts = [stripped]
        next_index = index + 1
        while _inline_if_expects_continuation(parts[-1]):
            if next_index >= len(lines):
                break
            next_line = lines[next_index]
            if not next_line.strip():
                break
            next_indent = next_line[: len(next_line) - len(next_line.lstrip())]
            if next_indent != indent:
                break
            parts.append(next_line.strip())
            next_index += 1

        if len(parts) > 1:
            collapsed.append(indent + " ".join(parts))
            index = next_index
            continue

        collapsed.append(line)
        index += 1
    return collapsed


def _expand_inline_if_statement_lines(lines: list[str]) -> list[str]:
    """Expand one-line `if` / `elif` / `else` statements into block form."""
    expanded: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            expanded.append(line)
            continue

        indent = line[: len(line) - len(line.lstrip())]
        inline = _split_inline_if_statement(stripped)
        if inline is None:
            expanded.append(line)
            continue
        header, body = inline
        expanded.append(indent + header)
        expanded.append(indent + "    " + body)
    return expanded


def _collapse_expression_continuation_lines(lines: list[str]) -> list[str]:
    """Collapse same-indent expression continuations into one normalized line."""
    collapsed: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            collapsed.append(line)
            index += 1
            continue

        indent = line[: len(line) - len(line.lstrip())]
        if not _is_expression_continuation_candidate(stripped):
            collapsed.append(line)
            index += 1
            continue

        parts = [stripped]
        next_index = index + 1
        while next_index < len(lines):
            next_line = lines[next_index]
            next_stripped = next_line.strip()
            if not next_stripped:
                break
            next_indent = next_line[: len(next_line) - len(next_line.lstrip())]
            if next_indent != indent:
                break
            if not _is_expression_continuation_candidate(next_stripped):
                break
            parts.append(next_stripped)
            next_index += 1

        collapsed.append(indent + " ".join(parts))
        index = next_index
    return collapsed


def _inline_if_expects_continuation(expression: str) -> bool:
    """Return whether an inline RuleSpec conditional still needs its else branch."""
    return expression.strip().startswith("if ") and expression.rstrip().endswith(
        "else:"
    )


def _split_inline_if_statement(expression: str) -> tuple[str, str] | None:
    """Split a one-line statement-form conditional into header and body."""
    stripped = expression.strip()
    if _is_inline_if_expression(stripped):
        return None

    for keyword in ("if", "elif"):
        prefix = f"{keyword} "
        if stripped.startswith(prefix):
            colon_index = _find_top_level_colon(stripped, start_index=len(prefix))
            if colon_index == -1 or colon_index == len(stripped) - 1:
                return None
            return stripped[: colon_index + 1], stripped[colon_index + 1 :].strip()

    if stripped.startswith("else:") and stripped != "else:":
        return "else:", stripped[len("else:") :].strip()

    return None


def _is_inline_if_expression(expression: str) -> bool:
    """Return whether a line uses inline RuleSpec conditional expression syntax."""
    stripped = expression.strip()
    if not stripped.startswith("if "):
        return False
    condition_colon = _find_top_level_colon(stripped, start_index=len("if "))
    if condition_colon == -1:
        return False
    return (
        _find_top_level_else_marker(
            stripped,
            start_index=condition_colon + 1,
        )
        != -1
    )


def _convert_inline_if_expression(expression: str) -> str:
    """Convert `if cond: a else: b` RuleSpec syntax into Python ternary syntax."""
    stripped = expression.strip()
    if not _is_inline_if_expression(stripped):
        return expression

    condition_colon = _find_top_level_colon(stripped, start_index=len("if "))
    else_marker = _find_top_level_else_marker(
        stripped,
        start_index=condition_colon + 1,
    )

    condition = stripped[len("if ") : condition_colon].strip()
    if_true = stripped[condition_colon + 1 : else_marker].strip()
    if_false = stripped[else_marker + len("else:") :].strip()
    if not condition or not if_true or not if_false:
        raise ExpressionParseError(
            f"Malformed inline conditional expression: '{expression}'."
        )

    return (
        f"({_convert_inline_if_expression(if_true)} if "
        f"{_convert_inline_if_expression(condition)} else "
        f"{_convert_inline_if_expression(if_false)})"
    )


def _rewrite_js_tokens(expression: str) -> str:
    """Rewrite JS-style tokens outside string literals."""
    output: list[str] = []
    index = 0
    while index < len(expression):
        char = expression[index]
        if char in {"'", '"'}:
            literal, index = _consume_string_literal(expression, index)
            output.append(literal)
            continue
        if expression.startswith("!==", index):
            output.append("!=")
            index += 3
            continue
        if expression.startswith("===", index):
            output.append("==")
            index += 3
            continue
        if expression.startswith("&&", index):
            output.append(" and ")
            index += 2
            continue
        if expression.startswith("||", index):
            output.append(" or ")
            index += 2
            continue
        if char == "!" and not expression.startswith("!=", index):
            output.append(" not ")
            index += 1
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < len(expression) and (
                expression[index].isalnum() or expression[index] == "_"
            ):
                index += 1
            identifier = expression[start:index]
            if identifier == "true":
                output.append("True")
            elif identifier == "false":
                output.append("False")
            else:
                output.append(identifier)
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _consume_string_literal(expression: str, start: int) -> tuple[str, int]:
    """Consume a quoted string literal, preserving escapes."""
    quote = expression[start]
    index = start + 1
    while index < len(expression):
        char = expression[index]
        if char == "\\":
            index += 2
            continue
        if char == quote:
            index += 1
            return expression[start:index], index
        index += 1
    return expression[start:], len(expression)


def _convert_ternary(expression: str) -> str:
    """Convert JavaScript ternary syntax into Python conditional expressions."""
    question_index = _find_top_level_operator(expression, "?")
    if question_index == -1:
        return expression

    colon_index = _find_matching_colon(expression, question_index)
    if colon_index == -1:
        raise ExpressionParseError(f"Malformed ternary expression: '{expression}'.")

    condition = expression[:question_index].strip()
    if_true = expression[question_index + 1 : colon_index].strip()
    if_false = expression[colon_index + 1 :].strip()
    return (
        f"({_convert_ternary(if_true)} if {_convert_ternary(condition)} "
        f"else {_convert_ternary(if_false)})"
    )


def _find_top_level_operator(expression: str, operator: str) -> int:
    """Find an operator at the top expression level, ignoring strings."""
    depth = 0
    index = 0
    while index < len(expression):
        char = expression[index]
        if char in {"'", '"'}:
            _, index = _consume_string_literal(expression, index)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == operator and depth == 0:
            return index
        index += 1
    return -1


def _find_matching_colon(expression: str, question_index: int) -> int:
    """Find the ':' that matches a top-level ternary operator."""
    depth = 0
    nested_ternaries = 0
    index = question_index + 1
    while index < len(expression):
        char = expression[index]
        if char in {"'", '"'}:
            _, index = _consume_string_literal(expression, index)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char == "?":
            nested_ternaries += 1
        elif depth == 0 and char == ":":
            if nested_ternaries == 0:
                return index
            nested_ternaries -= 1
        index += 1
    return -1


def _find_top_level_colon(expression: str, start_index: int = 0) -> int:
    """Find a top-level ':' after the requested start index."""
    depth = 0
    index = start_index
    while index < len(expression):
        char = expression[index]
        if char in {"'", '"'}:
            _, index = _consume_string_literal(expression, index)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == ":" and depth == 0:
            return index
        index += 1
    return -1


def _find_top_level_else_marker(expression: str, start_index: int = 0) -> int:
    """Find a top-level `else:` marker in an inline RuleSpec conditional."""
    depth = 0
    index = start_index
    while index < len(expression):
        char = expression[index]
        if char in {"'", '"'}:
            _, index = _consume_string_literal(expression, index)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif (
            depth == 0
            and expression.startswith("else:", index)
            and (
                index == 0
                or not (expression[index - 1].isalnum() or expression[index - 1] == "_")
            )
        ):
            return index
        index += 1
    return -1


def _is_expression_continuation_candidate(line: str) -> bool:
    """Return whether a formula line is part of a bare expression sequence."""
    stripped = line.strip()
    assignment_candidate = stripped
    if assignment_candidate.startswith("let "):
        assignment_candidate = assignment_candidate[len("let ") :].strip()

    if stripped.startswith("return "):
        return False
    if stripped == "else:":
        return False
    if stripped.endswith(":") and not _is_inline_if_expression(stripped):
        return False
    return _ASSIGNMENT_PATTERN.fullmatch(assignment_candidate) is None


def _attribute_to_name(node: ast.Attribute, original: str, variable_name: str) -> str:
    """Flatten a simple dotted attribute chain into a qualified name."""
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        raise ExpressionParseError(
            f"Variable '{variable_name}' uses attribute access in '{original}'. "
            "Attribute access is not supported by generic compilation."
        )
    parts.append(current.id)
    return ".".join(reversed(parts))


def _strip_formula_comment(line: str) -> str:
    """Remove trailing RuleSpec comments from one formula line."""
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line
