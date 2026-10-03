"""A ``where:`` predicate as a boolean array over the coordinates it masks.

The other half of what a declaration says: ``builder.py`` builds the thing,
this decides where it exists. A :class:`~mathspec.program.Predicate` in, one
``xr.DataArray`` of booleans out, and :func:`as_linopy_mask` puts it in the
shape linopy's ``mask=`` takes. Both lanes read the same node kinds, and
``relational/engine/predicates.py`` answers each with a polars
expression where this one answers with an array.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, replace
from functools import reduce
from typing import TYPE_CHECKING, Any, assert_never

import numpy as np
import xarray as xr
from mathspec import program

from specsolve.errors import DataError
from specsolve.messages import position_out_of_range_message, short_groups_message
from tests.linopy_lane import absence
from tests.linopy_lane.operators import _grouped, operator_at

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import linopy
    import pandas as pd


#: Where-comparison operators, evaluated element-wise on a DataArray.
_PREDICATE_OPS: dict[str, Callable[[Any, Any], Any]] = {
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '>': operator.gt,
    '<=': operator.le,
    '>=': operator.ge,
}


@dataclass(frozen=True)
class EvaluationContext:
    """Everything evaluating a plan needs beyond the node: the data, the axes, the model, the relations, the program.

    ``relations`` carries one array per value column, keyed by its relation's
    name and its own, over the dimensions the key names — what a predicate on a
    relation and a grouped operator both read instead of the parameter dataset.
    """

    dataset: xr.Dataset
    master_coords: Mapping[str, pd.Index]
    model: linopy.Model
    relations: Mapping[tuple[str, str], xr.DataArray]
    program: program.Program
    #: Whether *model* is solved and the plan is read at its solution — a
    #: variable is then its ``.solution`` and ``dual(c)`` the constraint's
    #: ``.dual`` — rather than built into it.
    solved: bool = False
    #: How a parameter reads where the data has no row. A coefficient is zero
    #: there, which is what the row multiplies by; the side of a where
    #: comparison is absent there, which is the false the language states for
    #: a comparison over a value that is not supplied.
    absent_parameter: Callable[[Any], Any] = absence.coefficient


def evaluate_where(mask: program.Mask | None, ctx: EvaluationContext) -> xr.DataArray:
    """Evaluate a lowered mask against a parameter dataset.

    Always a boolean DataArray. The no-mask case comes back 0-dimensional, so
    callers combine with ``&``/``|`` without case analysis.
    """
    if mask is None:
        return xr.DataArray(True)

    return _eval_node(mask.root, ctx)


def _eval_node(node: program.Predicate, ctx: EvaluationContext) -> xr.DataArray:
    """One predicate node as a boolean DataArray.

    Two absences read as exclusion rather than as an answer: a variable's
    masked-out coordinate is absent (:func:`absence.present`), and a
    comparison over NaN comes back false.

    **A null relation value is excluded explicitly rather than by ``fillna``.**
    A partial map arrives as an object array holding ``None``, and numpy
    answers ``None != 'north'`` with *True* rather than with null — so a ``!=``
    would keep exactly the labels that map nowhere.
    """
    dataset, master_coords = ctx.dataset, ctx.master_coords

    def evaluate(child: program.Predicate) -> xr.DataArray:
        return _eval_node(child, ctx)

    if isinstance(node, program.BooleanLiteral):
        return xr.DataArray(node.value)

    if isinstance(node, program.ParameterDefined):
        return _defined(dataset[node.name], ctx.program.parameters[node.name].dtype)

    if isinstance(node, program.VariableDefined):
        return absence.present(ctx.model, node.name)

    if isinstance(node, (program.ParameterComparison, program.DimensionComparison)):
        if isinstance(node, program.ParameterComparison):
            arr = dataset[node.name]
        else:
            arr = xr.DataArray(
                master_coords[node.name],
                coords={node.name: master_coords[node.name]},
                dims=[node.name],
            )

        result = _PREDICATE_OPS[node.op](arr, _as_the_axis_spells_it(arr, node.value))
        return result.fillna(False).astype(bool)

    if isinstance(node, program.ExpressionComparison):
        left, right = (_value(side, ctx) for side in (node.left, node.right))
        defined = left.notnull() & right.notnull()
        return (_PREDICATE_OPS[node.op](left, right) & defined).fillna(value=False).astype(bool)

    if isinstance(node, program.CountComparison):
        admitted = _along(evaluate(node.predicate.root), node.over, master_coords)
        return _PREDICATE_OPS[node.op](admitted.sum(node.over), node.value).astype(bool)

    if isinstance(node, program.TranslatedPredicate):
        operand = _along(evaluate(node.operand.root), node.along, master_coords)
        return operand.shift({node.along: node.offset}, fill_value=False)

    if isinstance(node, program.PulledBackPredicate):
        return _pulled_back(node, ctx)

    if isinstance(node, program.DimensionPosition):
        labels = master_coords[node.name]
        if node.partition is not None:
            by = node.partition.name
            (column,) = node.partition.group
            arr = _group_offsets(node, by, bound_relation(by, column, ctx.relations), np.asarray(labels))
            return (_PREDICATE_OPS[node.op](arr, 0) & arr.notnull()).fillna(value=False).astype(bool)
        at = node.position + len(labels) if node.position < 0 else node.position
        if not 0 <= at < len(labels):
            raise DataError(position_out_of_range_message(node.name, node.op, node.position, at, len(labels)))
        arr = xr.DataArray(np.arange(len(labels)), coords={node.name: labels}, dims=[node.name])
        return _PREDICATE_OPS[node.op](arr, at).astype(bool)

    if isinstance(node, program.RelationComparison):
        arr = bound_relation(node.name, node.column, ctx.relations)
        return (_PREDICATE_OPS[node.op](arr, node.value) & arr.notnull()).fillna(value=False).astype(bool)

    if isinstance(node, program.RelationPairComparison):
        left = bound_relation(node.name, node.column, ctx.relations)
        right = bound_relation(node.other, node.other_column, ctx.relations)
        defined = left.notnull() & right.notnull()
        return (_PREDICATE_OPS[node.op](left, right) & defined).fillna(value=False).astype(bool)

    if isinstance(node, program.RelationDefined):
        return _relation_has_a_row(node, ctx)

    if isinstance(node, program.Not):
        return ~evaluate(node.operand)

    if isinstance(node, program.And):
        return evaluate(node.left) & evaluate(node.right)

    if isinstance(node, program.Or):
        return evaluate(node.left) | evaluate(node.right)

    assert_never(node)


def _value(side: program.Expression, ctx: EvaluationContext) -> xr.DataArray:
    """One side of a comparison of expressions, as an array of its values.

    The builder's own walk answers, so a translation or a grouping under a
    ``where`` means what it means anywhere else. A parameter reads as absent
    rather than as the zero a coefficient takes, which is the false the
    language states for a comparison over a value that is not supplied.

    A side of numbers alone comes back as one, and is wrapped so both sides
    answer the same questions: ``0.5 <= p_max`` is a comparison of
    expressions like any other.
    """
    # in-function: the builder imports this module
    from tests.linopy_lane.builder import _eval

    value = _eval(side, replace(ctx, absent_parameter=lambda arr: arr))
    return value if isinstance(value, xr.DataArray) else xr.DataArray(value)


def _pulled_back(node: program.PulledBackPredicate, ctx: EvaluationContext) -> xr.DataArray:
    """Where the operand holds at the coarse coordinate the relation maps each fine one to.

    The expression ``at`` answers, so the predicate is read through the relation
    as an array is. A fine coordinate the relation has no row for reads the
    absence ``at`` leaves there, which is false in a mask.
    """
    # in-function: the builder imports this module
    from tests.linopy_lane.builder import _walked_arrays

    operand = _eval_node(node.operand.root, ctx)
    for dimension in node.direction.consumed_dims:
        operand = _along(operand, dimension, ctx.master_coords)
    read = operator_at(operand, _walked_arrays(node, ctx), into=node.direction.consumed_dims)
    return read.fillna(value=False).astype(bool)


def _along(arr: xr.DataArray, dimension: str, master_coords: Mapping[str, pd.Index]) -> xr.DataArray:
    """*arr* carrying *dimension*, broadcast over its labels where it does not.

    A predicate reading none of the dimension's own data answers the same at
    every coordinate of it, and counting or shifting along an axis the array
    has never seen would otherwise drop the reduction the node *is*.
    """
    return arr if dimension in arr.dims else arr.expand_dims({dimension: master_coords[dimension]})


def _defined(arr: xr.DataArray, dtype: str) -> xr.DataArray:
    """What a bare parameter name in a ``where`` asks: the declaration picks the reading.

    A ``bool`` is its own answer — a slot the data has no row for is false,
    the array having widened to float to hold the NaN — a ``str`` is defined
    wherever the data has a row, and a number has to be finite as well.
    """
    if dtype == 'bool':
        return arr.fillna(False).astype(bool)
    if dtype == 'str':
        return arr.notnull()
    return arr.notnull() & np.isfinite(arr)


def _group_offsets(node: program.DimensionPosition, by: str, groups: xr.DataArray, labels: np.ndarray) -> xr.DataArray:
    """Each coordinate's distance from the boundary of *its own* group.

    Zero marks the coordinate the position names, so every comparator reads the
    same as it does ungrouped. ``nan`` where the relation sends a coordinate
    nowhere: in no group, so no group's boundary.

    Raises:
        DataError: If any group is shorter than the position names, which would
            leave that group's rows unseeded and the model quietly unanchored.
    """
    partition = _grouped(node.name, labels, groups)
    needed = node.position + 1 if node.position >= 0 else -node.position
    short = sorted(str(g) for g, n in zip(partition.names, partition.counts, strict=True) if n < needed)
    if short:
        raise DataError(short_groups_message(node.name, by, node.op, node.position, short))
    target = node.position if node.position >= 0 else partition.size + node.position
    return partition.within.where(partition.grouped) - target


def _relation_has_a_row(node: program.RelationDefined, ctx: EvaluationContext) -> xr.DataArray:
    """Where the relation holds a row at the frame's coordinates.

    The arrays are padded to the key's whole product, so a key the table leaves
    out is null in every column; a row it does hold carries at least one value,
    a column left null in it being a value the model reads as absent.
    """
    columns = ctx.program.relations[node.name].values
    return reduce(operator.or_, (bound_relation(node.name, column, ctx.relations).notnull() for column in columns))


def unbound_relation_message(name: str) -> str:
    """A declared relation read with no attached map."""
    return f"relation '{name}' has no attached values. Pass it under key '{name}' as a table of the rows it holds."


def bound_relation(name: str, column: str, relations: Mapping[tuple[str, str], xr.DataArray]) -> xr.DataArray:
    """One value column of a map as an array over the dimensions its key names."""
    try:
        return relations[name, column]
    except KeyError:
        raise DataError(unbound_relation_message(name)) from None


def as_linopy_mask(mask: xr.DataArray) -> xr.DataArray | None:
    """Convert an evaluated where mask to linopy's ``mask=`` argument.

    linopy expects ``None`` for "no mask"; a 0-d True mask means exactly
    that. Everything else (including 0-d False) passes through.
    """
    if mask.ndim == 0 and bool(mask):
        return None
    return mask


def _as_the_axis_spells_it(arr: Any, value: Any) -> Any:
    """A where literal in the spelling the axis it is compared against uses.

    A quoted ISO date resolves to a ``datetime.date`` (the where rules), and a
    temporal axis arrives as ``datetime64`` — numpy compares the two by
    raising, so the axis decides, a literal carrying no dtype of its own.
    """
    if getattr(arr, 'dtype', None) is not None and arr.dtype.kind == 'M':
        return np.datetime64(value)
    return value
