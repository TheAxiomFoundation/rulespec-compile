"""Batch execution for lowered RuleSpec programs over NumPy/Pandas data."""

from __future__ import annotations

from functools import reduce
from typing import Any

import numpy as np
import pandas as pd

from .compile_model import CompilationError, LoweredParameter, LoweredProgram
from .expression_ir import (
    AssignStmt,
    BinaryExpr,
    BoolExpr,
    CallExpr,
    CompareExpr,
    ConditionalExpr,
    Expression,
    IfStmt,
    LiteralExpr,
    NameExpr,
    ReturnStmt,
    Statement,
    SubscriptExpr,
    UnaryExpr,
)


def execute_lowered_program_batch(
    program: LoweredProgram,
    inputs: pd.DataFrame | dict[str, Any],
) -> pd.DataFrame:
    """Execute a lowered program over a batch of rows."""
    normalized_inputs, batch_size = _normalize_batch_inputs(program, inputs)
    parameters = {parameter.name: parameter for parameter in program.parameters}
    environment: dict[str, np.ndarray] = dict(normalized_inputs)

    for computation in program.computations:
        scope: dict[str, np.ndarray] = dict(environment)
        result = _execute_statement_block(
            computation.statements,
            scope,
            parameters,
            batch_size,
            local_value_kinds=computation.local_value_kinds,
            return_value_kind=computation.value_kind,
        )
        environment[computation.name] = _coerce_value_kind(
            result,
            computation.value_kind,
            batch_size,
        )

    return pd.DataFrame(
        {
            output.name: _coerce_value_kind(
                environment[output.variable_name],
                output.value_kind,
                batch_size,
            )
            for output in program.outputs
        }
    )


def _normalize_batch_inputs(
    program: LoweredProgram,
    inputs: pd.DataFrame | dict[str, Any],
) -> tuple[dict[str, np.ndarray], int]:
    """Normalize batch inputs to arrays with one shared length."""
    raw_inputs = (
        inputs.to_dict(orient="list")
        if isinstance(inputs, pd.DataFrame)
        else dict(inputs)
    )
    lengths: set[int] = set()
    for value in raw_inputs.values():
        if np.isscalar(value):
            continue
        lengths.add(len(value))
    batch_size = lengths.pop() if lengths else 1
    if lengths:
        raise CompilationError("Batch inputs must all have the same length.")

    normalized: dict[str, np.ndarray] = {}
    for compiled_input in program.inputs:
        raw_value = raw_inputs.get(
            compiled_input.external_name,
            raw_inputs.get(compiled_input.name, compiled_input.default),
        )
        normalized[compiled_input.name] = _coerce_value_kind(
            raw_value,
            compiled_input.value_kind,
            batch_size,
        )
    return normalized, batch_size


def _execute_statement_block(
    statements: tuple[Statement, ...],
    environment: dict[str, np.ndarray],
    parameters: dict[str, LoweredParameter],
    batch_size: int,
    *,
    local_value_kinds: dict[str, str],
    return_value_kind: str,
) -> np.ndarray:
    """Execute one lowered statement block and return the computed value."""
    returned_mask = np.zeros(batch_size, dtype=bool)
    return_values = _allocate_value_kind_array(return_value_kind, batch_size)
    _execute_statement_sequence(
        statements,
        environment,
        parameters,
        batch_size,
        active_mask=np.ones(batch_size, dtype=bool),
        returned_mask=returned_mask,
        return_values=return_values,
        local_value_kinds=local_value_kinds,
        return_value_kind=return_value_kind,
    )
    if not returned_mask.all():
        raise CompilationError("Lowered computation did not return a value.")
    return return_values


def _execute_statement_sequence(
    statements: tuple[Statement, ...],
    environment: dict[str, np.ndarray],
    parameters: dict[str, LoweredParameter],
    batch_size: int,
    *,
    active_mask: np.ndarray,
    returned_mask: np.ndarray,
    return_values: np.ndarray,
    local_value_kinds: dict[str, str],
    return_value_kind: str,
) -> None:
    """Execute one statement sequence for the currently active rows."""
    current_mask = active_mask.copy()
    for statement in statements:
        if not current_mask.any():
            return

        if isinstance(statement, AssignStmt):
            local_kind = local_value_kinds.get(statement.name)
            if local_kind is None:
                raise CompilationError(
                    f"Batch execution is missing a local value kind for "
                    f"'{statement.name}'."
                )
            assigned = _coerce_value_kind(
                _evaluate_expression(
                    statement.expression,
                    environment,
                    parameters,
                    batch_size,
                    current_mask,
                ),
                local_kind,
                int(current_mask.sum()),
            )
            target = environment.get(statement.name)
            if target is None:
                target = _allocate_value_kind_array(local_kind, batch_size)
                environment[statement.name] = target
            target[current_mask] = assigned
            continue

        if isinstance(statement, ReturnStmt):
            return_values[current_mask] = _coerce_value_kind(
                _evaluate_expression(
                    statement.expression,
                    environment,
                    parameters,
                    batch_size,
                    current_mask,
                ),
                return_value_kind,
                int(current_mask.sum()),
            )
            returned_mask[current_mask] = True
            current_mask[:] = False
            continue

        if isinstance(statement, IfStmt):
            condition = _coerce_value_kind(
                _evaluate_expression(
                    statement.condition,
                    environment,
                    parameters,
                    batch_size,
                    current_mask,
                ),
                "boolean",
                int(current_mask.sum()),
            )
            body_mask = _compose_branch_mask(current_mask, condition)
            else_mask = _compose_branch_mask(current_mask, np.logical_not(condition))
            _execute_statement_sequence(
                statement.body,
                environment,
                parameters,
                batch_size,
                active_mask=body_mask,
                returned_mask=returned_mask,
                return_values=return_values,
                local_value_kinds=local_value_kinds,
                return_value_kind=return_value_kind,
            )
            if statement.orelse:
                _execute_statement_sequence(
                    statement.orelse,
                    environment,
                    parameters,
                    batch_size,
                    active_mask=else_mask,
                    returned_mask=returned_mask,
                    return_values=return_values,
                    local_value_kinds=local_value_kinds,
                    return_value_kind=return_value_kind,
                )
            current_mask &= np.logical_not(returned_mask)
            continue

        raise AssertionError(f"Unhandled statement node: {type(statement).__name__}")


def _evaluate_expression(
    expression: Expression,
    environment: dict[str, np.ndarray],
    parameters: dict[str, LoweredParameter],
    batch_size: int,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Evaluate one lowered expression over the active batch rows."""
    active_count = int(active_mask.sum())

    if isinstance(expression, LiteralExpr):
        return _broadcast_value(expression.value, active_count)

    if isinstance(expression, NameExpr):
        if expression.name in environment:
            return environment[expression.name][active_mask]
        if expression.name in parameters:
            parameter = parameters[expression.name]
            if parameter.lookup_kind != "scalar":
                raise CompilationError(
                    f"Batch execution encountered indexed parameter "
                    f"'{parameter.name}' without an index."
                )
            return _coerce_value_kind(
                parameter.values[0],
                parameter.value_kind,
                active_count,
            )
        raise CompilationError(
            f"Batch execution encountered unknown name '{expression.name}'."
        )

    if isinstance(expression, SubscriptExpr):
        if not isinstance(expression.value, NameExpr):
            raise CompilationError(
                "Batch execution only supports parameter indexing, not generic "
                "subscript expressions."
            )
        parameter_name = expression.value.name
        if parameter_name not in parameters:
            raise CompilationError(
                f"Batch execution encountered unknown indexed parameter "
                f"'{parameter_name}'."
            )
        index = _evaluate_expression(
            expression.index,
            environment,
            parameters,
            batch_size,
            active_mask,
        )
        return _lookup_parameter(parameters[parameter_name], index, active_count)

    if isinstance(expression, CallExpr):
        arguments = [
            _evaluate_expression(
                argument,
                environment,
                parameters,
                batch_size,
                active_mask,
            )
            for argument in expression.arguments
        ]
        return _evaluate_call(expression.function, arguments)

    if isinstance(expression, UnaryExpr):
        operand = _evaluate_expression(
            expression.operand,
            environment,
            parameters,
            batch_size,
            active_mask,
        )
        if expression.operator == "not":
            return np.logical_not(operand)
        if expression.operator == "+":
            return operand
        if expression.operator == "-":
            return -operand
        raise CompilationError(
            f"Batch execution does not support unary operator '{expression.operator}'."
        )

    if isinstance(expression, BinaryExpr):
        left = _evaluate_expression(
            expression.left,
            environment,
            parameters,
            batch_size,
            active_mask,
        )
        right = _evaluate_expression(
            expression.right,
            environment,
            parameters,
            batch_size,
            active_mask,
        )
        return _evaluate_binary(expression.operator, left, right)

    if isinstance(expression, BoolExpr):
        return _evaluate_boolean_expression(
            expression,
            environment,
            parameters,
            batch_size,
            active_mask,
        )

    if isinstance(expression, CompareExpr):
        left = _evaluate_expression(
            expression.left,
            environment,
            parameters,
            batch_size,
            active_mask,
        )
        comparisons = []
        current_left = left
        for operator, comparator in zip(
            expression.operators,
            expression.comparators,
            strict=True,
        ):
            current_right = _evaluate_expression(
                comparator,
                environment,
                parameters,
                batch_size,
                active_mask,
            )
            comparisons.append(_evaluate_compare(operator, current_left, current_right))
            current_left = current_right
        return reduce(np.logical_and, comparisons)

    if isinstance(expression, ConditionalExpr):
        condition = _coerce_value_kind(
            _evaluate_expression(
                expression.condition,
                environment,
                parameters,
                batch_size,
                active_mask,
            ),
            "boolean",
            active_count,
        )
        if condition.all():
            return _evaluate_expression(
                expression.if_true,
                environment,
                parameters,
                batch_size,
                active_mask,
            )
        if not condition.any():
            return _evaluate_expression(
                expression.if_false,
                environment,
                parameters,
                batch_size,
                active_mask,
            )
        true_values = _evaluate_expression(
            expression.if_true,
            environment,
            parameters,
            batch_size,
            _compose_branch_mask(active_mask, condition),
        )
        false_values = _evaluate_expression(
            expression.if_false,
            environment,
            parameters,
            batch_size,
            _compose_branch_mask(active_mask, np.logical_not(condition)),
        )
        return _merge_branch_values(condition, true_values, false_values)

    raise AssertionError(f"Unhandled expression node: {type(expression).__name__}")


def _evaluate_boolean_expression(
    expression: BoolExpr,
    environment: dict[str, np.ndarray],
    parameters: dict[str, LoweredParameter],
    batch_size: int,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Evaluate boolean expressions with per-row short-circuit semantics."""
    active_count = int(active_mask.sum())
    operands = iter(expression.values)
    result = _coerce_value_kind(
        _evaluate_expression(
            next(operands),
            environment,
            parameters,
            batch_size,
            active_mask,
        ),
        "boolean",
        active_count,
    )

    if expression.operator == "and":
        for operand in operands:
            pending = result.astype(bool)
            if not pending.any():
                break
            result[pending] = _coerce_value_kind(
                _evaluate_expression(
                    operand,
                    environment,
                    parameters,
                    batch_size,
                    _compose_branch_mask(active_mask, pending),
                ),
                "boolean",
                int(pending.sum()),
            )
        return result

    if expression.operator == "or":
        for operand in operands:
            pending = np.logical_not(result.astype(bool))
            if not pending.any():
                break
            result[pending] = _coerce_value_kind(
                _evaluate_expression(
                    operand,
                    environment,
                    parameters,
                    batch_size,
                    _compose_branch_mask(active_mask, pending),
                ),
                "boolean",
                int(pending.sum()),
            )
        return result

    raise CompilationError(
        f"Batch execution does not support boolean operator '{expression.operator}'."
    )


def _evaluate_call(function: str, arguments: list[np.ndarray]) -> np.ndarray:
    """Evaluate one supported call expression."""
    if function == "abs":
        return np.abs(arguments[0])
    if function == "ceil":
        return np.ceil(arguments[0])
    if function == "floor":
        return np.floor(arguments[0])
    if function == "round":
        return np.round(arguments[0])
    if function == "max":
        return reduce(np.maximum, arguments)
    if function == "min":
        return reduce(np.minimum, arguments)
    raise CompilationError(f"Batch execution does not support function '{function}'.")


def _evaluate_binary(operator: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Evaluate one binary expression."""
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/":
        return left / right
    if operator == "%":
        return left % right
    if operator == "**":
        return left**right
    raise CompilationError(f"Batch execution does not support operator '{operator}'.")


def _evaluate_compare(
    operator: str,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Evaluate one comparison expression."""
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    raise CompilationError(
        f"Batch execution does not support comparison operator '{operator}'."
    )


def _lookup_parameter(
    parameter: LoweredParameter,
    index: Any,
    batch_size: int,
) -> np.ndarray:
    """Resolve an indexed parameter over the active rows."""
    if parameter.lookup_kind != "indexed":
        raise CompilationError(
            f"Batch execution indexed scalar parameter '{parameter.name}'."
        )
    index_array = _coerce_value_kind(
        index,
        parameter.index_value_kind or "integer",
        batch_size,
    )
    unique_indexes = {int(value) for value in np.unique(index_array)}
    missing = unique_indexes.difference(parameter.values)
    if missing:
        names = ", ".join(str(value) for value in sorted(missing))
        raise CompilationError(
            f"Batch execution indexed parameter '{parameter.name}' with missing "
            f"keys: {names}."
        )
    max_index = max(parameter.values)
    lookup = np.zeros(max_index + 1, dtype=float)
    for key, value in parameter.values.items():
        lookup[key] = value
    return lookup[index_array]


def _coerce_value_kind(value: Any, value_kind: str, batch_size: int) -> np.ndarray:
    """Broadcast one scalar or vector input to the requested value kind."""
    array = _broadcast_value(value, batch_size)

    if value_kind == "boolean":
        return array.astype(bool)
    if value_kind == "integer":
        return np.rint(array).astype(int)
    if value_kind == "number":
        return array.astype(float)
    if value_kind == "string":
        return array.astype(str)
    raise CompilationError(f"Unknown batch value kind '{value_kind}'.")


def _broadcast_value(value: Any, batch_size: int) -> np.ndarray:
    """Broadcast one scalar or vector value to the requested length."""
    array = np.asarray(value)
    if array.ndim == 0:
        return np.full(batch_size, array.item())
    if len(array) != batch_size:
        raise CompilationError(
            f"Batch value had length {len(array)} but expected {batch_size} rows."
        )
    return array


def _allocate_value_kind_array(value_kind: str, batch_size: int) -> np.ndarray:
    """Allocate one empty batch slot for the given lowered value kind."""
    if value_kind == "boolean":
        return np.zeros(batch_size, dtype=bool)
    if value_kind == "integer":
        return np.zeros(batch_size, dtype=int)
    if value_kind == "number":
        return np.zeros(batch_size, dtype=float)
    if value_kind == "string":
        return np.full(batch_size, "", dtype=str)
    raise CompilationError(f"Unknown batch value kind '{value_kind}'.")


def _compose_branch_mask(
    active_mask: np.ndarray,
    branch_selection: np.ndarray,
) -> np.ndarray:
    """Lift one branch selection over active rows back to the full batch mask."""
    global_mask = np.zeros(len(active_mask), dtype=bool)
    global_mask[active_mask] = _coerce_value_kind(
        branch_selection,
        "boolean",
        int(active_mask.sum()),
    )
    return global_mask


def _merge_branch_values(
    condition: np.ndarray,
    if_true: Any,
    if_false: Any,
) -> np.ndarray:
    """Merge two already-masked branch results back into one active-row vector."""
    true_values = _broadcast_value(if_true, int(condition.sum()))
    false_values = _broadcast_value(if_false, int((~condition).sum()))
    dtype = _resolve_merge_dtype(true_values, false_values)
    merged = np.empty(len(condition), dtype=dtype)
    merged[condition] = true_values.astype(dtype, copy=False)
    merged[~condition] = false_values.astype(dtype, copy=False)
    return merged


def _resolve_merge_dtype(left: np.ndarray, right: np.ndarray) -> np.dtype:
    """Choose one safe dtype for merged branch results."""
    if left.size == 0 and right.size == 0:
        return np.dtype(float)
    if left.size == 0:
        return right.dtype
    if right.size == 0:
        return left.dtype

    left_family = _dtype_family(left.dtype)
    right_family = _dtype_family(right.dtype)
    if left_family != right_family:
        return np.dtype(object)

    try:
        return np.result_type(left.dtype, right.dtype)
    except TypeError:
        return np.dtype(object)


def _dtype_family(dtype: np.dtype) -> str:
    """Bucket NumPy dtypes into coarse families for branch merging."""
    if dtype.kind in {"b", "i", "u", "f", "c"}:
        return "numeric"
    if dtype.kind in {"U", "S"}:
        return "string"
    return "object"
