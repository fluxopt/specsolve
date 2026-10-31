"""A parameter's dual: the rate of the optimal objective in each of its rows, read off one solve.

Every leaf but the varied one is held at the solve ([`Compiler`][]), so a rate
is a term's coefficient in the Lagrangian. A value on a ``where:`` threshold
switches columns or rows on or off as it rises; [`_crossings`][] decides each
from the same solve, never from another.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import polars as pl
from mathspec import program

from specsolve._collect import collected
from specsolve._engine import labels
from specsolve._engine.assembly import declares_quadratic
from specsolve._engine.compiler import Compiler, varied_label
from specsolve._engine.pieces import absence_restrictions
from specsolve._engine.predicates import masked
from specsolve._engine.readback import _as_strings
from specsolve._engine.scope import UNIT, Scope, join_on
from specsolve._messages import unknown_name_message
from specsolve._names import VALUE
from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from specsolve._engine.pieces import CompiledExpression, Piece

#: The varied leaf's label while a parameter dual adds up its rates.
_VARIED = '__varied label__'

_NO_RATES = pl.LazyFrame(schema={_VARIED: pl.Int64, VALUE: pl.Float64})

#: How far a value is raised to find the thresholds it sits on, relative to its size.
_RISE = 1e-9

#: A primal value or a dual this close to zero is zero: solvers place a nonbasic
#: column exactly on its bound, so this only absorbs their arithmetic.
_ZERO = 1e-9

#: A row this close to holding exactly has zero slack: the default feasibility
#: tolerance of every sink is 1e-6.
_FEASIBLE = 1e-6


def parameter_dual(compiler: Compiler, name: str) -> tuple[pl.DataFrame, list[str]]:
    """Parameter *name*'s rate at the solve *compiler* reads — ``(dims…, value)`` — and what the caller is warned of.

    The rate in each row is the one at which the optimal objective moves as
    that value rises: the Lagrangian's derivative, every other leaf held at the
    solve, plus what a rise switches on or off where the value sits on a
    ``where:`` threshold ([`_crossings`][]). It is ``±inf`` where the
    objective jumps. A row nothing reads has zero. Rows come in label order.

    Raises:
        KeyError: No parameter is called *name*.
        SpecsolveError: *name* stands where its rate is not one coefficient
            per row.
    """
    declared = compiler.scope.program.parameters
    if name not in declared:
        raise KeyError(unknown_name_message('parameter', name, declared))
    assert compiler.solution is not None and compiler.solution.dual is not None, (
        'a result refuses a parameter dual its solve left no duals for'
    )
    _refuse_a_nonlinear_read(compiler.scope.program, name)

    dims = declared[name].dims
    rows = compiler.scope.data.parameters[name].with_columns(varied_label(compiler.scope, dims).alias(_VARIED))
    table = rows.select(*dims, _VARIED)
    values = rows.select(_VARIED, pl.col(VALUE).cast(pl.Float64))
    crossed = _crossings(compiler, name, table, values)
    rates = _lagrangian_rates(replace(compiler, varied=name), name, values)
    rates += _bound_rates(compiler, name, table, crossed.settled)
    rates += crossed.rates
    summed = pl.concat(rates).group_by(_VARIED).agg(pl.col(VALUE).sum()) if rates else _NO_RATES
    out = (
        table.join(summed, on=_VARIED, how='left')
        .sort(_VARIED, descending=True)
        .select(*dims, pl.col(VALUE).fill_null(0.0))
        .pipe(collected)
    )
    return _as_strings(out, compiler.scope.data, dims), crossed.notes


def _refuse_a_nonlinear_read(model: program.Program, name: str) -> None:
    """Refuse a parameter the math reads other than linearly, where a rate is not one coefficient per row.

    Raises:
        SpecsolveError: *name* stands under a divisor, in a power, or on both
            sides of a product.
    """
    declarations: dict[str, tuple[program.Expression, ...]] = {
        f"constraint '{c}'": (d.lhs, d.rhs) for c, d in model.constraints.items()
    }
    if model.objective is not None:
        declarations['the objective'] = (model.objective.expression,)
    for where, expressions in declarations.items():
        for node in program.walk(*expressions):
            if isinstance(node, program.Divide) and name in program.parameters_of(node.divisor):
                how = 'divides by it'
            elif isinstance(node, program.Power) and name in program.parameters_of(node):
                how = 'raises it to a power'
            elif isinstance(node, program.Multiply) and name in (
                program.parameters_of(node.left) & program.parameters_of(node.right)
            ):
                how = 'multiplies it by itself'
            else:
                continue
            raise SpecsolveError(
                f"{where} {how}, so the rate of parameter '{name}' is not one number per row and "
                f'specsolve does not compute it. Declare the quantity the model reads, such as the '
                f"reciprocal of '{name}', as a parameter of its own, and ask the parameter dual of that."
            )


def _reads(name: str, *expressions: program.Expression) -> bool:
    """Whether a parameter or variable called *name* stands anywhere under *expressions*."""
    return name in program.parameters_of(*expressions) | program.variables_of(*expressions)


def _lagrangian_rates(compiler: Compiler, name: str, values: pl.LazyFrame) -> list[pl.LazyFrame]:
    """``(label, value)`` parts of the Lagrangian's rate in each label of the leaf *compiler* varies.

    The Lagrangian is the objective less each row's ``lhs - rhs`` times its
    dual, which is the sign [`Result.dual`][specsolve._result.Result.dual]
    gives a rate in the objective. Only *compiler*'s varied leaf compiles to a
    term, so a term's coefficient is its rate, and a quadratic term's rate in
    each label is its coefficient times the other label's value in *values*.
    Constraints are compiled as quadratic positions: the only term is the
    varied leaf, so a product of two is the square a quadratic row declares.
    """
    model = compiler.scope.program
    parts: list[pl.LazyFrame] = []
    if model.objective is not None and _reads(name, model.objective.expression):
        compiled = compiler.expression(model.objective.expression, 'the objective', quadratic=True)
        parts += _gradient(compiled, None, 1.0, values)
    for constraint, c in model.constraints.items():
        if not _reads(name, c.lhs, c.rhs):
            continue
        dual = compiler.expression(program.Dual(constraint, dims=c.dims), f"the dual of '{constraint}'").consts[0]
        for side, sign in ((c.lhs, -1.0), (c.rhs, 1.0)):
            compiled = compiler.expression(side, f"constraint '{constraint}'", quadratic=True)
            parts += _gradient(compiled, dual, sign, values)
    return parts


def _gradient(
    compiled: CompiledExpression, weight: Piece | None, sign: float, values: pl.LazyFrame
) -> list[pl.LazyFrame]:
    """*compiled*'s rate in each label of its terms, times *weight* where it lands and *sign*, as ``(label, value)``.

    *weight* is a row's dual, joined on the dims the term carries so that a
    term narrower than its row is owed at every row it lands on.
    """
    parts = []
    for piece in (*compiled.terms, *compiled.quads):
        frame = piece.frame
        rate = sign * pl.col('coeff')
        if weight is not None:
            frame = join_on(weight.frame, frame, piece.dims, 'inner')
            rate = rate * pl.col('cval')
        if piece.kind == 'term':
            parts.append(frame.select(pl.col('var_label').cast(pl.Int64).alias(_VARIED), rate.alias(VALUE)))
            continue
        for label, other in (('var_label', 'var_label_2'), ('var_label_2', 'var_label')):
            at = values.rename({_VARIED: other, VALUE: '__other value__'})
            paired = frame.with_columns(pl.col(other).cast(pl.Int64)).join(at, on=other, how='inner')
            parts.append(
                paired.select(
                    pl.col(label).cast(pl.Int64).alias(_VARIED), (rate * pl.col('__other value__')).alias(VALUE)
                )
            )
    return parts


def _toward(model: program.Program) -> float:
    """``1`` where the objective is minimised, ``-1`` where it is maximised: the sign that makes a rise a loss."""
    return -1.0 if model.objective is not None and model.objective.sense == 'maximize' else 1.0


def _primal(compiler: Compiler) -> pl.LazyFrame:
    """The solve's primal as ``(label, value)``."""
    assert compiler.solution is not None, 'a parameter dual reads a solve'
    primal = compiler.solution.primal
    return pl.LazyFrame({_VARIED: pl.int_range(primal.len(), dtype=pl.Int64, eager=True), VALUE: primal})


def _reduced_costs(compiler: Compiler, variable: str, primal: pl.LazyFrame) -> pl.LazyFrame:
    """``(label, value)``: the Lagrangian's rate in each column of *variable*, which is its reduced cost."""
    rates = _lagrangian_rates(replace(compiler, varied=variable), variable, primal)
    return pl.concat(rates).group_by(_VARIED).agg(pl.col(VALUE).sum()) if rates else _NO_RATES


def _bound_sides(v: program.Variable, name: str) -> set[str]:
    """The sides of *v* whose bound is parameter *name*."""
    return {
        side
        for side, bound in (('lower', v.lower), ('upper', v.upper))
        if isinstance(bound, program.ParameterReference) and bound.name == name
    }


def _bound_rates(
    compiler: Compiler, name: str, table: pl.LazyFrame, settled: Mapping[str, pl.LazyFrame]
) -> list[pl.LazyFrame]:
    """``(label, value)``: each column's reduced cost, owed to *name*'s row where *name* is the bound that binds.

    The reduced cost is the rate in the binding bound
    ([`Result.reduced_cost`][specsolve._result.Result.reduced_cost]), and
    its sign says which bound binds: under ``minimize`` a positive one holds a
    column up at its lower bound and a negative one down at its upper, and
    ``maximize`` flips both. It is the Lagrangian's rate in the column,
    computed by varying the variable, which leaves no handoff to hold. A
    column in *settled* is left out: a threshold decides it.
    """
    model = compiler.scope.program
    toward = _toward(model)
    primal = _primal(compiler)
    dims = model.parameters[name].dims
    parts = []
    for variable, v in model.variables.items():
        sides = _bound_sides(v, name)
        if not sides:
            continue
        reduced = _reduced_costs(compiler, variable, primal)
        held = compiler.scope.variables[variable]
        keys = v.dims or (UNIT,)
        columns = held.frame.select(*keys, pl.col('var_label').cast(pl.Int64).alias('__column__'))
        if variable in settled:
            columns = columns.join(settled[variable], on='__column__', how='anti')
        reduced_at = columns.join(reduced.rename({_VARIED: '__column__'}), on='__column__', how='left').with_columns(
            pl.col(VALUE).fill_null(0.0)
        )
        held_by = pl.lit(False)
        if 'lower' in sides:
            held_by = held_by | (pl.col(VALUE) * toward > 0)
        if 'upper' in sides:
            held_by = held_by | (pl.col(VALUE) * toward < 0)
        owed = reduced_at.filter(held_by)
        parts.append(join_on(owed, table, dims, 'inner').select(_VARIED, VALUE))
    return parts


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------


@dataclass
class _Crossed:
    """What a rise past the thresholds adds to a parameter dual."""

    #: ``(label, value)`` parts: a finite rate, or ``±inf`` where the objective jumps.
    rates: list[pl.LazyFrame] = field(default_factory=list)
    #: Per variable, the columns ``(__column__)`` a threshold decides, which the bound pass skips.
    settled: dict[str, pl.LazyFrame] = field(default_factory=dict)
    #: What the caller is warned of.
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Column:
    """One column a threshold decides: where the solve left it, and what it costs to move."""

    variable: str
    label: int
    #: The row of the parameter whose rise decides it.
    charged: int
    #: Where it sits, and its bounds at the current values.
    at: float
    lower: float
    upper: float
    #: Whether the column's own coordinate projects onto *charged*, so its bounds read that row.
    own: bool
    #: Whether each bound is the parameter's row *charged*.
    lower_moves: bool
    upper_moves: bool
    reduced_cost: float


@dataclass(frozen=True)
class _RowAt:
    """One switched-on row at the solve, its switched-on columns at zero."""

    #: The row's key values, in key order.
    coordinate: tuple[object, ...]
    #: Non-negative where the row holds.
    slack: float
    #: Each column's label to its coefficient.
    columns: dict[int, float]
    #: Each row of the varied parameter to the rate of ``lhs - rhs`` in it.
    growth: dict[int, float]


def _crossings(compiler: Compiler, name: str, table: pl.LazyFrame, values: pl.LazyFrame) -> _Crossed:
    """What a rise of each value of *name* switches on or off, decided from the solve alone.

    A value sits on a threshold where a rise of a relative [`_RISE`][]
    changes which columns a variable's ``where:`` builds, or which rows a
    constraint builds. Each switched coordinate is charged to the one row of
    *name* that its mask reads at that coordinate, so a mask that reads
    *name* through a shift, a sum, a lookup or a count is held as built, and
    so is a ``when:``. Then, against the current primal and duals:

    - a column switched off that the solve used, or a row switched off that
      has a dual, is a jump;
    - a row switched on that holds with slack changes nothing, and one that
      is violated is a jump;
    - a row switched on at zero slack on one column caps that column; on
      several, only a solve decides it, so it is held as built;
    - a column switched on, or capped by such a row, moves as far as its
      bounds and caps allow in the direction its reduced cost prices as a
      gain, and its rate is the reduced cost times that move; with no cap at
      zero distance that way, it is a jump.
    """
    scope = compiler.scope
    model = scope.program
    crossed = _Crossed()
    if model.parameters[name].dtype not in ('float', 'int'):
        return crossed
    risen = _risen(scope, name)
    crossed.notes += _held_as_built(scope, risen, name)
    variables = {
        n: v for n, v in model.variables.items() if v.where is not None and _local(v.where, name, v.dims, model)
    }
    masked_rows = {
        n for n, c in model.constraints.items() if c.where is not None and _local(c.where, name, c.dims, model)
    }
    if not variables and not masked_rows:
        return crossed

    assert compiler.solution is not None, 'a parameter dual reads a solve'
    toward = _toward(model)
    p_dims = model.parameters[name].dims
    labels_of = table.pipe(collected)
    next_label = compiler.solution.primal.len()
    primal = _primal(compiler).pipe(collected)

    grown = dict(scope.variables)
    after = dict(scope.variables)
    flipped: set[str] = set()
    switched_on: dict[str, pl.DataFrame] = {}
    for variable, v in variables.items():
        keys = list(v.dims or (UNIT,))
        held = scope.variables[variable]
        built = held.frame.select(*keys, pl.col('var_label').cast(pl.Int64)).pipe(collected)
        past = masked(risen, v.dims, v.where).select(*keys).pipe(collected)
        on = (
            past.join(built.select(keys), on=keys, how='anti')
            .with_row_index('var_label', offset=next_label)
            .with_columns(pl.col('var_label').cast(pl.Int64))
            .select(*keys, 'var_label')
        )
        off = built.join(past, on=keys, how='anti')
        next_label += on.height
        if on.height or off.height:
            flipped.add(variable)
        if on.height:
            switched_on[variable] = _projected(on, p_dims, labels_of)
            grown[variable] = labels.Labelled(pl.concat([built, on]).lazy(), held.start, built.height + on.height)
        kept = pl.concat([built.join(off.select(keys), on=keys, how='anti'), on])
        after[variable] = labels.Labelled(kept.lazy(), held.start, kept.height)
        if off.height:
            crossed.settled[variable] = off.select(pl.col('var_label').alias('__column__')).lazy()
            used = _projected(off, p_dims, labels_of).join(primal.rename({_VARIED: 'var_label'}), on='var_label')
            crossed.rates.append(_jump(used.filter(pl.col(VALUE).abs() > _ZERO), toward * math.inf))

    at = pl.concat([primal, *(pl.DataFrame({_VARIED: on['var_label'], VALUE: 0.0}) for on in switched_on.values())])
    priced = replace(compiler, scope=replace(scope, variables=grown))
    column_charge = {
        label: charged
        for on in switched_on.values()
        for label, charged in zip(on['var_label'], on[_VARIED], strict=True)
    }
    caps: list[tuple[int, int, str, float]] = []
    for constraint, c in model.constraints.items():
        if constraint not in masked_rows and not program.variables_of(c.lhs, c.rhs) & flipped:
            continue
        _rows_switched(
            compiler,
            priced,
            replace(risen, variables=after),
            constraint,
            c,
            name,
            labels_of,
            values,
            at,
            column_charge,
            caps,
            crossed,
        )

    columns = _decided_columns(compiler, priced, name, switched_on, caps, at, labels_of)
    for column in columns:
        crossed.rates.append(
            _column_rate(column, [cap for cap in caps if cap[:2] == (column.label, column.charged)], toward)
        )
    for variable in {column.variable for column in columns}:
        own = pl.LazyFrame(
            {'__column__': [c.label for c in columns if c.variable == variable and c.own]},
            schema={'__column__': pl.Int64},
        )
        crossed.settled[variable] = pl.concat([crossed.settled[variable], own]) if variable in crossed.settled else own
    return crossed


def _held_as_built(scope: Scope, risen: Scope, name: str) -> list[str]:
    """A note for each mask a rise of *name* flips somewhere that a parameter dual cannot charge to one row."""
    model = scope.program
    notes = []
    for kind, declarations in (('variable', model.variables), ('constraint', model.constraints)):
        for n, d in declarations.items():
            where = d.where
            if where is None or name not in where.names_read or _local(where, name, d.dims, model):
                continue
            if _flips(scope, risen, d.dims, where):
                notes.append(
                    f"{kind} '{n}' reads '{name}' in its where: through a shift, a sum, a lookup or a count, so "
                    f"the rows of '{name}' on that threshold are priced as the model was built"
                )
    expressions = [side for c in model.constraints.values() for side in (c.lhs, c.rhs)]
    if model.objective is not None:
        expressions.append(model.objective.expression)
    whens = [mask for _, regions in program.walk_regions(*expressions) for mask in regions if name in mask.names_read]
    if any(_flips(scope, risen, scope.in_declaration_order(mask.dims), mask) for mask in whens):
        notes.append(
            f"a when: reads '{name}', so the rows of '{name}' on its threshold are priced as the model was built"
        )
    return notes


def _flips(scope: Scope, risen: Scope, dims: tuple[str, ...], mask: program.Mask) -> bool:
    """Whether a rise to *risen* changes which coordinates of *dims* *mask* keeps."""
    keys = list(dims or (UNIT,))
    now = masked(scope, dims, mask).select(keys).pipe(collected)
    past = masked(risen, dims, mask).select(keys).pipe(collected)
    return now.height != past.height or bool(now.join(past, on=keys, how='anti').height)


def _local(mask: program.Mask, name: str, dims: tuple[str, ...], model: program.Program) -> bool:
    """Whether *mask* reads *name*, and only at the coordinate it filters, so a flip there is one row's.

    False where *name* has a dim the declaration lacks, since a flip could
    not be charged to one row.
    """
    if name not in mask.names_read or not set(model.parameters[name].dims) <= set(dims):
        return False
    shaping = (program.Sum, program.Join, program.Translate, program.WindowSum)
    for atom in mask.atoms:
        if isinstance(atom, program.ExpressionComparison):
            nodes = program.walk(atom.left, atom.right)
            if any(isinstance(node, shaping) and name in program.parameters_of(node) for node in nodes):
                return False
        elif isinstance(atom, program.CountComparison):
            if name in atom.condition.names_read:
                return False
        elif (
            isinstance(atom, (program.TranslatedCondition, program.JoinedCondition)) and name in atom.operand.names_read
        ):
            return False
    return True


def _risen(scope: Scope, name: str) -> Scope:
    """*scope* with every value of *name* raised by [`_RISE`][] of its size, past any threshold it sits on."""
    data = scope.data
    raised = data.parameters[name].with_columns(
        (pl.col(VALUE) + _RISE * pl.max_horizontal(pl.lit(1.0), pl.col(VALUE).abs())).alias(VALUE)
    )
    return replace(scope, data=replace(data, parameters={**data.parameters, name: raised}))


def _projected(frame: pl.DataFrame, p_dims: tuple[str, ...], labels_of: pl.DataFrame) -> pl.DataFrame:
    """*frame* with the label of the parameter row its coordinate projects onto, as [`_VARIED`][]."""
    return join_on(frame.lazy(), labels_of.lazy(), p_dims, 'inner').pipe(collected)


def _jump(charged: pl.DataFrame | pl.LazyFrame, rate: float) -> pl.LazyFrame:
    """``(label, value)``: *rate*, an infinity, at each row in *charged*."""
    return charged.lazy().select(_VARIED, pl.lit(rate, dtype=pl.Float64).alias(VALUE))


def _rows_switched(
    compiler: Compiler,
    priced: Compiler,
    risen: Scope,
    constraint: str,
    c: program.Constraint,
    name: str,
    labels_of: pl.DataFrame,
    values: pl.LazyFrame,
    at: pl.DataFrame,
    column_charge: Mapping[int, int],
    caps: list[tuple[int, int, str, float]],
    crossed: _Crossed,
) -> None:
    """Decide the rows of *constraint* a rise switches, adding jumps and notes to *crossed* and column caps to *caps*.

    A row is charged to the parameter row its coordinate projects onto, or,
    where the parameter has a dim the constraint lacks, to the one its
    single switched-on column is charged to.
    """
    assert compiler.solution is not None, 'a parameter dual reads a solve'
    toward = _toward(compiler.scope.program)
    keys = list(c.dims or (UNIT,))
    p_dims = compiler.scope.program.parameters[name].dims
    projects = set(p_dims) <= set(c.dims)
    built = compiler.solution.constraints[constraint].frame.select(*keys).pipe(collected)
    past = _row_coordinates(risen, constraint, c)
    on = past.join(built, on=keys, how='anti')
    off = built.join(past, on=keys, how='anti')
    if declares_quadratic(c) and (on.height or off.height):
        crossed.notes.append(
            f"constraint '{constraint}' is quadratic, so the rows a rise of '{name}' switches there are priced as built"
        )
        return

    if off.height:
        dual = compiler.expression(program.Dual(constraint, dims=c.dims), f"the dual of '{constraint}'").consts[0]
        priced_off = (
            join_on(off.lazy(), dual.frame, c.dims, 'inner').filter(pl.col('cval').abs() > _ZERO).pipe(collected)
        )
        if priced_off.height and projects:
            crossed.rates.append(_jump(_projected(priced_off, p_dims, labels_of), -toward * math.inf))
        elif priced_off.height:
            crossed.notes.append(f"rows of constraint '{constraint}' that a rise switches off are priced as built")
    if not on.height:
        return

    charged_rows = _projected(on, p_dims, labels_of) if projects else None
    unpriced: dict[str, list[dict[str, object]]] = {}
    for row in _rows_at(priced, constraint, c, name, on, values, at):
        coordinate = {d: row.coordinate[i] for i, d in enumerate(keys)}
        terms = row.columns
        if projects:
            assert charged_rows is not None, 'a projecting constraint charges by coordinate'
            matched = charged_rows.filter(*(pl.col(k) == row.coordinate[i] for i, k in enumerate(keys)))
            charged = matched[_VARIED].item() if matched.height else None
        else:
            charged = column_charge.get(next(iter(terms))) if len(terms) == 1 else None
        if row.slack > _FEASIBLE or not terms:
            continue
        if charged is None:
            unpriced.setdefault('cannot be charged to one row of the parameter', []).append(coordinate)
            continue
        if row.slack < -_FEASIBLE:
            crossed.rates.append(
                _jump(pl.DataFrame({_VARIED: [charged]}, schema={_VARIED: pl.Int64}), toward * math.inf)
            )
            continue
        if len(terms) != 1:
            unpriced.setdefault(
                'switch on with zero slack across several columns, which only a solve decides', []
            ).append(coordinate)
            continue
        ((column, coefficient),) = terms.items()
        growth = -row.growth.get(charged, 0.0) / coefficient
        positive = coefficient > 0
        if c.sense in ('<=', '=='):
            caps.append((column, charged, 'upper' if positive else 'lower', growth))
        if c.sense in ('>=', '=='):
            caps.append((column, charged, 'lower' if positive else 'upper', growth))
    for why, coordinates in unpriced.items():
        crossed.notes.append(
            f"{len(coordinates)} rows of constraint '{constraint}' that a rise of '{name}' switches on {why}, so "
            f'they are priced as built, the first at {coordinates[0]}'
        )


def _row_coordinates(scope: Scope, constraint: str, c: program.Constraint) -> pl.DataFrame:
    """The coordinates constraint *c* builds rows at under *scope*, as a build would."""
    build = Compiler(scope)
    quadratic = declares_quadratic(c)
    lhs = build.expression(c.lhs, f"constraint '{constraint}' lhs", quadratic=quadratic)
    rhs = build.expression(c.rhs, f"constraint '{constraint}' rhs", quadratic=quadratic)
    pieces = [*lhs.terms, *lhs.quads, *lhs.consts, *rhs.terms, *rhs.quads, *rhs.consts]
    keys = c.dims or (UNIT,)
    return labels.frame(scope, c.dims, c.where, '__row__', 0, absence_restrictions(pieces)).select(*keys)


def _rows_at(
    priced: Compiler,
    constraint: str,
    c: program.Constraint,
    name: str,
    rows: pl.DataFrame,
    values: pl.LazyFrame,
    at: pl.DataFrame,
) -> list[_RowAt]:
    """Each row in *rows* at the solve, switched-on columns at zero: its slack, columns and rate in *name*.

    Compiled as a build with *name* varied, so every variable and every row
    of *name* is a term, told apart by the sign of its label
    ([`varied_label`][specsolve._engine.compiler.varied_label]); a product
    of the two is a quadratic term.
    """
    build = replace(priced, solution=None, varied=name)
    keys = list(c.dims or (UNIT,))
    x = at.rename({_VARIED: '__x label__', VALUE: '__x__'}).lazy()
    theta = values.rename({_VARIED: '__p label__', VALUE: '__p__'})
    parts = []
    for side, sign in ((c.lhs, 1.0), (c.rhs, -1.0)):
        compiled = build.expression(side, f"constraint '{constraint}'", quadratic=True)
        for piece in (*compiled.terms, *compiled.quads, *compiled.consts):
            frame = join_on(rows.lazy(), piece.frame, piece.dims, 'inner')
            if piece.kind == 'const':
                parts.append(
                    frame.select(
                        *keys,
                        pl.lit('value').alias('kind'),
                        pl.lit(None, pl.Int64).alias('label'),
                        (sign * pl.col('cval')).alias('amount'),
                    )
                )
                continue
            if piece.kind == 'term':
                frame = frame.with_columns(pl.col('var_label').cast(pl.Int64).alias('__l__'))
                variable = frame.filter(pl.col('__l__') >= 0)
                parameter = frame.filter(pl.col('__l__') < 0).join(theta, left_on='__l__', right_on='__p label__')
                parts.append(
                    variable.select(
                        *keys,
                        pl.lit('column').alias('kind'),
                        pl.col('__l__').alias('label'),
                        (sign * pl.col('coeff')).alias('amount'),
                    )
                )
                parts.append(
                    parameter.select(
                        *keys,
                        pl.lit('value').alias('kind'),
                        pl.lit(None, pl.Int64).alias('label'),
                        (sign * pl.col('coeff') * pl.col('__p__')).alias('amount'),
                    )
                )
                parts.append(
                    parameter.select(
                        *keys,
                        pl.lit('growth').alias('kind'),
                        pl.col('__l__').alias('label'),
                        (sign * pl.col('coeff')).alias('amount'),
                    )
                )
                continue
            pair = (
                frame.with_columns(
                    pl.max_horizontal(pl.col('var_label').cast(pl.Int64), pl.col('var_label_2').cast(pl.Int64)).alias(
                        '__v__'
                    ),
                    pl.min_horizontal(pl.col('var_label').cast(pl.Int64), pl.col('var_label_2').cast(pl.Int64)).alias(
                        '__p label__'
                    ),
                )
                .join(theta, on='__p label__')
                .join(x, left_on='__v__', right_on='__x label__', how='left')
            )
            pair = pair.with_columns(pl.col('__x__').fill_null(0.0))
            parts.append(
                pair.select(
                    *keys,
                    pl.lit('column').alias('kind'),
                    pl.col('__v__').alias('label'),
                    (sign * pl.col('coeff') * pl.col('__p__')).alias('amount'),
                )
            )
            parts.append(
                pair.select(
                    *keys,
                    pl.lit('growth').alias('kind'),
                    pl.col('__p label__').alias('label'),
                    (sign * pl.col('coeff') * pl.col('__x__')).alias('amount'),
                )
            )
    if not parts:
        return []
    long = pl.concat(parts).pipe(collected)
    columns = long.filter(pl.col('kind') == 'column').group_by(*keys, 'label').agg(pl.col('amount').sum())
    with_x = columns.join(at.rename({_VARIED: 'label', VALUE: '__x__'}), on='label', how='left').with_columns(
        pl.col('__x__').fill_null(0.0)
    )
    value = (
        pl.concat(
            [
                long.filter(pl.col('kind') == 'value').select(*keys, 'amount'),
                with_x.select(*keys, (pl.col('amount') * pl.col('__x__')).alias('amount')),
            ]
        )
        .group_by(keys)
        .agg(pl.col('amount').sum().alias('g'))
    )
    growth = long.filter(pl.col('kind') == 'growth').group_by(*keys, 'label').agg(pl.col('amount').sum())
    out = []
    for row in rows.join(value, on=keys, how='left').with_columns(pl.col('g').fill_null(0.0)).iter_rows(named=True):
        here = [pl.col(k) == row[k] for k in keys]
        g = row['g']
        slack = -g if c.sense == '<=' else g if c.sense == '>=' else -abs(g)
        out.append(
            _RowAt(
                coordinate=tuple(row[k] for k in keys),
                slack=slack,
                columns=dict(columns.filter(*here).select('label', 'amount').iter_rows()),
                growth=dict(growth.filter(*here).select('label', 'amount').iter_rows()),
            )
        )
    return out


def _decided_columns(
    compiler: Compiler,
    priced: Compiler,
    name: str,
    switched_on: Mapping[str, pl.DataFrame],
    caps: list[tuple[int, int, str, float]],
    at: pl.DataFrame,
    labels_of: pl.DataFrame,
) -> list[_Column]:
    """Every column a threshold decides — switched on, or capped by a switched-on row — with what deciding it reads."""
    model = compiler.scope.program
    p_dims = model.parameters[name].dims
    wanted: dict[int, set[int]] = {}
    for on in switched_on.values():
        for label, charged in zip(on['var_label'], on[_VARIED], strict=True):
            wanted.setdefault(label, set()).add(charged)
    for column, charged, _, _ in caps:
        wanted.setdefault(column, set()).add(charged)
    if not wanted:
        return []
    out = []
    x = dict(at.iter_rows())
    for variable, v in model.variables.items():
        held = priced.scope.variables[variable]
        keys = list(v.dims or (UNIT,))
        frame = (
            held.frame.select(*keys, pl.col('var_label').cast(pl.Int64))
            .filter(pl.col('var_label').is_in(list(wanted)))
            .pipe(collected)
        )
        if not frame.height:
            continue
        alone = replace(compiler, scope=replace(compiler.scope, variables={**compiler.scope.variables, variable: held}))
        reduced = dict(_reduced_costs(alone, variable, at.lazy()).pipe(collected).iter_rows())
        bounded = _bounds_at(compiler.scope, frame, variable, v)
        own = (
            _projected(frame, p_dims, labels_of).select('var_label', pl.col(_VARIED).alias('__own__'))
            if set(p_dims) <= set(v.dims)
            else None
        )
        own_of = dict(own.iter_rows()) if own is not None else {}
        sides = _bound_sides(v, name)
        for row in bounded.iter_rows(named=True):
            label = row['var_label']
            for charged in sorted(wanted[label]):
                moves = own_of.get(label) == charged
                out.append(
                    _Column(
                        variable=variable,
                        label=label,
                        charged=charged,
                        at=x.get(label, 0.0),
                        own=moves,
                        lower=row['lb'],
                        upper=row['ub'],
                        lower_moves=moves and 'lower' in sides,
                        upper_moves=moves and 'upper' in sides,
                        reduced_cost=reduced.get(label, 0.0),
                    )
                )
    return out


def _bounds_at(scope: Scope, frame: pl.DataFrame, variable: str, v: program.Variable) -> pl.DataFrame:
    """``(var_label, lb, ub)`` for the columns in *frame* at the current values, an open side infinite."""
    out = frame.lazy()
    for side, bound, open_side in (('lb', v.lower, -math.inf), ('ub', v.upper, math.inf)):
        if bound is None:
            out = out.with_columns(pl.lit(open_side).alias(side))
        elif isinstance(bound, program.Constant):
            out = out.with_columns(pl.lit(float(bound.value)).alias(side))
        else:
            assert isinstance(bound, program.ParameterReference), 'a bound is a number or a parameter'
            out = scope.parameter_join(out, bound.name, v.dims, side, f"bound '{bound.name}' of '{variable}'")
            out = out.with_columns(pl.col(side).cast(pl.Float64).fill_null(open_side))
    return out.select('var_label', 'lb', 'ub').pipe(collected)


def _column_rate(column: _Column, caps: list[tuple[int, int, str, float]], toward: float) -> pl.LazyFrame:
    """``(label, value)``: what moving *column* adds to the rate of its charged row.

    The column moves the way its reduced cost prices as a gain, as fast as
    the tightest cap at zero distance that way allows, and is pushed by a cap
    that rises past where it sits; its bounds are caps too, moving at one
    where they are the charged row and still otherwise. A forced move
    against a cap that does not move, or a gain no cap at zero distance
    bounds, is a jump.
    """
    lower = [0.0 if not column.lower_moves else 1.0] if abs(column.at - column.lower) <= _ZERO else []
    upper = [0.0 if not column.upper_moves else 1.0] if abs(column.upper - column.at) <= _ZERO else []
    lower += [growth for *_, side, growth in caps if side == 'lower']
    upper += [growth for *_, side, growth in caps if side == 'upper']
    push = max(lower, default=-math.inf)
    room = min(upper, default=math.inf)
    rc = column.reduced_cost
    gains_up = rc * toward < -_ZERO
    gains_down = rc * toward > _ZERO
    if column.at < column.lower - _ZERO or column.at > column.upper + _ZERO or push > room:
        rate = toward * math.inf
    elif gains_up:
        rate = rc * room if room < math.inf else -toward * math.inf
    elif gains_down:
        rate = rc * push if push > -math.inf else -toward * math.inf
    else:
        rate = 0.0
    return pl.LazyFrame({_VARIED: [column.charged], VALUE: [rate]}, schema={_VARIED: pl.Int64, VALUE: pl.Float64})
