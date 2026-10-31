"""Model builder: logical plan + data → linopy Model.

**One section per kind of translation**, in the order a build performs them:
the four declarations (``Variables``, ``Special-ordered sets``,
``Constraints``, ``Objectives``) each ending in the ``model.add_*`` call they
exist to make, then ``Plan evaluation`` for what an expression is worth.

Two questions a build asks are answered beside it: ``operators.py`` evaluates
a built-in once its operands are values, and ``where.py`` turns a predicate
into the boolean array a declaration is masked by. The positions an absent
value is spelled differently in are ``absence.py``. *Which* linopy call each
construct becomes is the table in ``docs/about/linopy.md``.
"""

from __future__ import annotations

import functools
import math
import operator
from typing import TYPE_CHECKING, Any, assert_never

import numpy as np
import xarray as xr
from mathspec import program

from specsolve._messages import null_bounds_message
from specsolve._sinks.capabilities import Capabilities, Capability, spelled
from specsolve.errors import DataError, SpecsolveError
from tests.linopy_lane import absence
from tests.linopy_lane._notes import note
from tests.linopy_lane.coverage import check_constant_side_covers, check_divisors_cover
from tests.linopy_lane.loader import OracleCannotBuildError
from tests.linopy_lane.operators import (
    operator_at,
    operator_grouped_sum,
    operator_shift,
    operator_sum,
    operator_sum_back,
)
from tests.linopy_lane.where import EvaluationContext, as_linopy_mask, bound_relation, evaluate_where

if TYPE_CHECKING:
    from collections.abc import Mapping

    import linopy
    import pandas as pd

_SIGN_MAP = {'==': '=', '<=': '<=', '>=': '>='}


def build_model(
    model: linopy.Model,
    program: program.Program,
    dataset: xr.Dataset,
    master_coords: dict[str, pd.Index],
    relations: dict[tuple[str, str], xr.DataArray],
    given: Mapping[str, xr.DataArray] | None = None,
) -> None:
    """Populate a linopy Model from a lowered program and loaded parameters.

    This mutates *model* in-place, adding variables, constraints and the
    objective as declared in *program*. Nothing is re-checked here: a program
    is trusted by construction, loading having decided every rule the
    language can decide without data. *given* is
    :func:`~tests.linopy_lane.loader.given_rows`' answer, for a bare name in a
    ``where``.
    """
    ctx = EvaluationContext(dataset, master_coords, model, relations, program, given=given or {})
    _build_variables(ctx)
    _build_sos(ctx)
    _build_constraints(ctx)
    _build_objective(ctx)


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------


def _build_variables(ctx: EvaluationContext) -> None:
    for name, vdef in ctx.program.variables.items():
        with note(f"while building variable '{name}'"):
            coords = {d: ctx.master_coords[d] for d in vdef.dims}
            mask = evaluate_where(vdef.where, ctx)

            _check_bounds_are_defined(name, vdef, ctx, mask)

            ctx.model.add_variables(
                lower=_bound(vdef.lower, ctx, -math.inf),
                upper=_bound(vdef.upper, ctx, math.inf),
                coords=coords,
                name=name,
                mask=as_linopy_mask(mask),
                binary=vdef.domain == 'binary',
                integer=vdef.domain == 'integer',
            )


def _check_bounds_are_defined(name: str, vdef: program.Variable, ctx: EvaluationContext, mask: Any) -> None:
    """Refuse a bound with no value at build, before the NaN reaches linopy's IO layer.

    Checked against the variable's own mask: a coordinate the variable does not
    occupy needs no bound, and each one counts once however many of its bounds
    are missing, as the relational lane counts its columns. A ``missing:
    absent`` bound is not asked: its gap is an open side (:func:`_bound`).
    """
    stated = [side for side in (vdef.lower, vdef.upper) if side is not None]
    owed = [ctx.dataset[p].isnull() for p in sorted(program.parameters_of(*stated)) if not ctx.absent(p)]
    if not owed:
        return
    frame = xr.DataArray(True).expand_dims({d: ctx.master_coords[d] for d in vdef.dims})
    missing = int((functools.reduce(operator.or_, owed) & frame & mask).sum())
    if missing:
        raise DataError(null_bounds_message(name, missing))


def _bound(bound: program.Expression | None, ctx: EvaluationContext, open_side: float) -> Any:
    """A bound as linopy takes it: the literal, the named parameter's array, or *open_side* where it is open.

    A gap is open under ``missing: absent``, and not filled otherwise:
    absence's zero is a coefficient and never a bound, so a gap survives to
    :func:`_check_bounds_are_defined`.
    """
    if bound is None:
        return open_side
    if isinstance(bound, program.Constant):
        return bound.value
    if isinstance(bound, program.ParameterReference):
        values = ctx.dataset[bound.name]
        return values.fillna(open_side) if ctx.absent(bound.name) else values
    msg = f'bounds accept a number or a parameter, and lowering builds nothing else — got {type(bound).__name__}'
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Special-ordered sets
# ---------------------------------------------------------------------------


def _build_sos(ctx: EvaluationContext) -> None:
    """Attach every ``sos:`` block to the variable it names."""
    for name, sos in ctx.program.sos.items():
        with note(f"while building sos '{name}'"):
            ctx.model.add_sos_constraints(
                ctx.model.variables[sos.variable],
                sos_type=sos.sos_type,
                sos_dim=sos.along,
            )


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


#: What this lane can build, in the sinks' vocabulary: it accepts the same
#: language, and cannot build a quadratic constraint —
#: ``linopy.Model.add_constraints`` refuses a ``QuadraticExpression`` outright
#: and no reformulation of it is exact.
CAPABILITIES = Capabilities(
    supports=frozenset({'integrality', 'sos', 'quadratic_objective', 'nonconvex_quadratic_objective'}),
)


def _declared(p: program.Program) -> frozenset[Capability]:
    """What *p* declares, in the sinks' vocabulary — the file's upper bound, since this lane has no built model to read."""
    footprint = p.footprint
    needed: set[Capability] = set()
    if footprint.domains - {'continuous'}:
        needed.add('integrality')
    if 'objective' in footprint.quadratic:
        needed.add('quadratic_objective')
    if 'constraint' in footprint.quadratic:
        needed.add('quadratic_constraint')
    if footprint.sos_types:
        needed.add('sos')
    return frozenset(needed)


def _refuse_what_the_lane_cannot_build(p: program.Program) -> None:
    """Refuse a construct the language accepts and this lane cannot build, before linopy is asked."""
    if missing := CAPABILITIES.missing(_declared(p)):
        raise OracleCannotBuildError(
            f'the linopy lane cannot build {spelled(missing)}, and no reformulation of it is exact. '
            f'The language accepts it and specsolve builds it, so this is a limit of the lane rather '
            f'than of the spec.'
        )


def _build_constraints(ctx: EvaluationContext) -> None:
    _refuse_what_the_lane_cannot_build(ctx.program)
    for name, row in ctx.program.constraints.items():
        with note(f"while building constraint '{name}'"):
            mask = evaluate_where(row.where, ctx)
            context = f"constraint '{name}'"

            check_divisors_cover(context, (row.lhs, row.rhs), ctx, mask, lambda node: _eval(node, ctx))
            check_constant_side_covers(context, row, ctx, mask)

            if mask is not None and not bool(np.asarray(mask).any()):
                continue
            lhs = _linear_where_squares_vanish(_eval(row.lhs, ctx), context)
            rhs = _linear_where_squares_vanish(_eval(row.rhs, ctx), context)
            if _term_free(lhs) and _term_free(rhs):
                continue

            term, other, sense = _sides(lhs, rhs, row.sense)
            if (gap := absence.holes(other)) is not None:
                mask, other = mask & ~gap, other.fillna(0.0)
                if not bool(np.asarray(mask).any()):
                    continue
            ctx.model.add_constraints(term, _SIGN_MAP[sense], other, name=name, mask=as_linopy_mask(mask))


def _linear_where_squares_vanish(side: Any, context: str) -> Any:
    """*side* as a linear expression where every square in it carries a zero coefficient.

    A file can write a square into a row its data never prices — a cost the
    assumptions hold at zero wherever the row stands — and linopy takes no
    quadratic row at all.

    Raises:
        OracleCannotBuildError: A square with a coefficient that is not zero.
    """
    from linopy.constants import FACTOR_DIM
    from linopy.expressions import LinearExpression, QuadraticExpression

    if not isinstance(side, QuadraticExpression):
        return side
    data = side.data
    squared = data.vars.isel({FACTOR_DIM: 1}) != -1
    if bool(((data.coeffs != 0) & squared).any()):
        raise OracleCannotBuildError(f'{context}: the linopy lane cannot build a quadratic constraint')
    linear = xr.Dataset(
        {
            'coeffs': data.coeffs.where(~squared, 0.0),
            'vars': data.vars.isel({FACTOR_DIM: 0}).where(~squared, -1),
            'const': data.const,
        }
    )
    return LinearExpression(linear, side.model)


#: What reading a comparison from its other side does to it.
_FLIPPED: dict[program.ConstraintSense, program.ConstraintSense] = {'==': '==', '<=': '>=', '>=': '<='}


def _sides(lhs: Any, rhs: Any, sense: program.ConstraintSense) -> tuple[Any, Any, program.ConstraintSense]:
    """The comparison with a term on the left, which is the only side linopy takes one on.

    Either side may carry the terms — ``cap >= p`` and ``p <= cap`` both build
    — and ``add_constraints`` accepts an expression as its ``lhs`` alone,
    answering anything else with a ``TypeError`` naming a linopy type. Reading
    the row from the other side reverses the comparison.

    Reached only once a side is known to carry a term, so the ``rhs`` returned
    where the ``lhs`` is term-free is the one that does.
    """
    if not _term_free(lhs):
        return lhs, rhs, sense
    return rhs, lhs, _FLIPPED[sense]


def _term_free(side: Any) -> bool:
    """Whether *side* has nowhere for a variable term to sit.

    A bare ``Variable`` is a term; a ``LinearExpression`` over an empty axis has
    a term dimension of length zero; anything else is data. Both sides
    term-free is a constraint the *data* emptied — a dimension with no members
    reduces away to a number — which the absence rules say is not a row. An
    expression naming no variable to begin with is refused at load.
    """
    if hasattr(side, 'to_linexpr'):
        return False
    return getattr(side, 'nterm', 0) == 0


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


def _build_objective(ctx: EvaluationContext) -> None:
    """Build the declared objective, if any, onto the model.

    An objective has no ``where``, so its divisor check runs with no row mask.
    The expression is scalar by the time it gets here, the language having
    refused one carrying dims, so it is evaluated like any other; linopy's
    ``*`` answers a product of two variables with a ``QuadraticExpression`` on
    its own.
    """
    odef = ctx.program.objective
    if odef is None:
        return
    with note('while building the objective'):
        check_divisors_cover('the objective', (odef.expression,), ctx, None, lambda node: _eval(node, ctx))

        expr = _eval(odef.expression, ctx)
        _refuse_an_objective_constant(expr)

        ctx.model.add_objective(expr, overwrite=True, sense=_LINOPY_SENSE[odef.sense])


#: The objective sense as linopy spells it.
_LINOPY_SENSE: dict[program.ObjectiveSense, str] = {'minimize': 'min', 'maximize': 'max'}


#: The one construct this lane accepts and cannot build, as the sentence a user reads.
OBJECTIVE_CONSTANT_IS_A_LANE_GAP = (
    "the objective carries a constant term, and this lane cannot build one: linopy's objective "
    'takes no constant. The relational lane builds it and returns the right number, so the spec '
    'is sayable and only this lane is short: run it with `specsolve.solve` / `specsolve.build`. '
    'Dropping the constant here is refused deliberately — it would answer a different model.'
)


def _refuse_an_objective_constant(expr: Any) -> None:
    """Refuse an objective this lane cannot build, before linopy is asked."""
    const = getattr(expr, 'const', None)
    if const is not None and bool(np.any(np.asarray(const) != 0)):
        raise OracleCannotBuildError(OBJECTIVE_CONSTANT_IS_A_LANE_GAP)


# ---------------------------------------------------------------------------
# Plan evaluation
# ---------------------------------------------------------------------------


def _eval(node: program.Expression, ctx: EvaluationContext) -> Any:
    """One plan node as a linopy term, an array, or a number.

    One node kind per branch: a variable is its linopy term, a parameter its
    filled array, arithmetic the Python operator linopy overloads, and an
    operator its function in ``operators.py``.
    """
    if isinstance(node, program.Constant):
        return node.value

    if isinstance(node, program.VariableReference):
        variable, declared = ctx.model.variables[node.name], ctx.program.variables[node.name].missing
        return absence.variable_value(variable, declared) if ctx.solved else absence.variable_term(variable, declared)

    if isinstance(node, program.Dual):
        assert ctx.solved, 'a dual reached a build — the language keeps one out of the math'
        return _dual(node.constraint, ctx)

    if isinstance(node, program.ParameterReference):
        values = ctx.dataset[node.name]
        return values if ctx.absent(node.name) else ctx.absent_parameter(values)

    if isinstance(node, program.Negate):
        return -_eval(node.operand, ctx)

    if isinstance(node, program.Add):
        return absence.combined(operator.add, _eval(node.left, ctx), _eval(node.right, ctx))

    if isinstance(node, program.Multiply):
        return absence.combined(operator.mul, _eval(node.left, ctx), _eval(node.right, ctx))

    if isinstance(node, program.Divide):
        divisor = _eval(node.divisor, ctx)
        divisor = absence.divisor(divisor) if ctx.solved else divisor
        return absence.combined(operator.truediv, _eval(node.numerator, ctx), divisor, neutral=1.0)

    if isinstance(node, program.Power):
        return _eval(node.base, ctx) ** _eval(node.exponent, ctx)

    if isinstance(node, program.Sum) and isinstance(node.operand, program.Join) and node.operand.columns.axes:
        return _grouped_sum(_eval(node.operand.operand, ctx), node.operand.columns, ctx)

    if isinstance(node, program.Sum):
        summed = _eval(node.operand, ctx)
        for axis in node.over:
            summed = operator_sum(summed, axis.dimension)
        return summed

    if isinstance(node, program.Join):
        columns = node.columns
        read = _walked_arrays(columns, columns.dropped, ctx)
        return operator_at(_eval(node.operand, ctx), read, into=columns.dropped_dims)

    if isinstance(node, program.Translate):
        return operator_shift(
            _eval(node.operand, ctx),
            over=node.along,
            offset=_amount(node.offset, ctx),
            wrap=node.wrap,
            fill=node.fill,
            by=_partition(node, ctx),
        )

    if isinstance(node, program.WindowSum):
        return operator_sum_back(
            _eval(node.operand, ctx),
            over=node.along,
            within=_amount(node.width, ctx),
            wrap=node.wrap,
            by=_partition(node, ctx),
        )

    if isinstance(node, program.Cases):
        return _cases(node, ctx)

    if isinstance(node, program.ExpressionReference):
        return _eval(node.body, ctx)

    assert_never(node)


def _dual(name: str, ctx: EvaluationContext) -> xr.DataArray:
    """``dual(name)`` at the solve — linopy's own ``.dual`` on the constraint.

    Refused on a model declaring integrality before linopy is asked: HiGHS
    hands a MIP back with a dual of zero on every row, and linopy stores it,
    so the number would be read rather than the absence the other lane
    reports.

    Raises:
        SpecsolveError: A variable declares integrality, so the duals are
            undefined; or the solver stored none.
    """
    discrete = sorted(n for n, v in ctx.program.variables.items() if v.domain != 'continuous')
    if discrete:
        raise SpecsolveError(
            f'named expression reads dual({name}), and duals are undefined for a mixed-integer model: '
            f'{", ".join(discrete)} declare integrality. Read it off a continuous model.'
        )
    try:
        return ctx.model.constraints[name].dual
    except AttributeError:
        raise SpecsolveError(
            f'named expression reads dual({name}), and this solve stored no duals — the solver returned none.'
        ) from None


def _in_region(value: Any, mask: xr.DataArray) -> Any:
    """*value* where the region holds, and a hard zero everywhere else.

    A **fill**, not a multiplication: inside the mask absence still stands, and
    outside it the value is a hard zero. A bare number has no absence to
    protect, so there the mask multiplies.
    """
    if hasattr(value, 'to_linexpr'):
        value = value.to_linexpr()
    if hasattr(value, 'where'):
        return value.where(mask, 0)
    return mask * value


def _cases(node: program.Cases, ctx: EvaluationContext) -> Any:
    """A value defined by region, as the regions added.

    The regions are disjoint and total — the language proved that before any
    data attached — so each one filled with zero outside itself and the lot
    added gives every coordinate exactly one region's value.
    """
    filled = (_in_region(_eval(region.value, ctx), evaluate_where(region.when, ctx)) for region in node.regions)
    return functools.reduce(functools.partial(absence.combined, operator.add), filled)


def _amount(amount: int | str, ctx: EvaluationContext) -> Any:
    """An offset or a width: the number, or the integer parameter naming it.

    Read through :func:`absence.coefficient` like any other parameter — a step
    nobody supplied is a step of nothing, which is what a zero offset means.
    """
    return absence.coefficient(ctx.dataset[amount]) if isinstance(amount, str) else amount


def _grouped_sum(operand: Any, columns: program.JoinColumns, ctx: EvaluationContext) -> Any:
    """``sum(x, over=a, by=relation[b])``: the join and the sum over the axes it opens, as one grouping.

    Where the call sums key columns away and groups by value columns, the
    relation is a map read at its key, and a ``groupby`` on the value arrays
    is the whole of it; every other shape is a membership sum
    (:func:`_member_sum`).
    """
    relation = columns.relation
    if relation.values and set(columns.dropped) <= set(relation.key) and set(columns.added) <= set(relation.values):
        return operator_grouped_sum(
            operand,
            _walked_arrays(columns, columns.added, ctx),
            into=columns.added_dims,
            joined=tuple(columns.dim(role) for role in columns.kept),
            labels=ctx.master_coords,
        )
    return _member_sum(operand, columns, ctx)


def _member_sum(operand: Any, columns: program.JoinColumns, ctx: EvaluationContext) -> Any:
    """A sum through the rows of a relation: every row carries its dropped columns' term to its added ones.

    The operand's dims become the relation's roles, so two roles over one
    dimension stay apart; the membership array multiplies in, the dropped
    roles sum out, and the added roles take their dimensions back.

    Raises:
        OracleCannotBuildError: The call joins on or groups by a value
            column while not reading the relation as a map at its key.
    """
    relation = columns.relation
    if valued := [role for role in (*columns.dropped, *columns.added) if role in relation.values]:
        raise OracleCannotBuildError(
            f"the linopy lane sums through the rows of '{columns.name}' over its key columns alone, and this "
            f'sum reads its value columns {valued} as well. The language accepts it and the relational lane '
            f'builds it, so this is a limit of the lane rather than of the spec.'
        )
    entering = {columns.dim(r): r for r in columns.joined if columns.dim(r) != r}
    leaving = {r: columns.dim(r) for r in columns.grouped if columns.dim(r) != r}
    summed = operand.rename(entering) * _membership(columns, ctx)
    for role in columns.dropped:
        summed = operator_sum(summed, role)
    return summed.rename(leaving)


def _membership(columns: program.JoinColumns, ctx: EvaluationContext) -> xr.DataArray:
    """One where the relation has a row and zero elsewhere, over its key's roles.

    A bare relation arrives as that array; a keyed one has a row wherever one
    of its value columns holds a value at the key.
    """
    relation = columns.relation
    if not relation.values:
        return bound_relation(columns.name, '', ctx.relations)
    held = functools.reduce(
        operator.or_, (bound_relation(columns.name, v, ctx.relations).notnull() for v in relation.values)
    )
    return held.rename({relation.dim(r): r for r in relation.key if relation.dim(r) != r}).astype(float)


def _walked_arrays(columns: program.JoinColumns, roles: tuple[str, ...], ctx: EvaluationContext) -> tuple[Any, ...]:
    """The relation's value columns *roles* as arrays over the dimensions its key names, in the order given."""
    return tuple(bound_relation(columns.name, role, ctx.relations) for role in roles)


def _partition(node: program.Translate | program.WindowSum, ctx: EvaluationContext) -> Any:
    """The relation a windowed operator may not reach across, as its values.

    **Named for the dimension its values are labels of**, not for itself: an
    amount declared over the group's own dim is read through this array by
    :func:`~tests.linopy_lane.operators._per_group`, which pairs the two by that
    name.
    """
    if node.partition is None:
        return None
    (column,) = node.partition.grouped
    array = bound_relation(node.partition.name, column, ctx.relations)
    return array.rename(node.partition.dim(column))
