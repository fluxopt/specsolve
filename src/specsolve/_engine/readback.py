"""Reading a built model back: one constraint row, a solve as a [`Result`][], a named expression, and a start laid onto it."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from mathspec import program

from specsolve import _sinks as sinks
from specsolve._answer import WHOLE, Answer
from specsolve._collect import collected
from specsolve._engine import coverage, indices
from specsolve._engine.assembly import declares_quadratic
from specsolve._engine.attaching import attach
from specsolve._engine.queries import QueryBuilder, Solution
from specsolve._engine.scope import Scope
from specsolve._engine.terms import Term, presences_of
from specsolve._messages import coordinate_expr, reported_divisor_message, unknown_name_message
from specsolve._names import COEFF, COL, COL_L, COL_R, CONSTANT, ROW, VALUE
from specsolve._outputs import kinds_of
from specsolve._records import Record
from specsolve._result import ConstraintRow, Expressions, InfeasibleSubsystem, Result
from specsolve._sinks.solver import AT_LOWER, BASIC, BASIS, BASIS_STATUSES, Basis, seated
from specsolve._sinks.tables import SENSE_CODES
from specsolve.errors import SpecsolveError, SpecsolveWarning

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from specsolve._engine.assembly import BuiltModel
    from specsolve._engine.attaching import AttachedSources
    from specsolve._outputs import Output
    from specsolve._sinks.solver import InfeasibleSubsystemIndices, SolveAnswer
    from specsolve._sinks.tables import SinkTables
    from specsolve._status import SolveStatus

#: How an expression the caller writes becomes a program node. ``None``
#: wherever one stands for a build with no spec as written to compile it
#: against.
type Lower = Callable[[str | Mapping[str, object]], program.Expression]

#: Scratch columns. The spaces make them unrepresentable as declared names.
_SOLUTION = '__solution value__'
_EXPRESSION_ROW = '__expression row__'
_INDEX_ORDER = '__index order__'


def row(model: BuiltModel, name: str, coordinate: Mapping[str, object]) -> ConstraintRow:
    """One built constraint row, spelled back out. See [`row`][specsolve._api.Model.row].

    Raises:
        KeyError: No constraint of that name.
        SpecsolveError: A coordinate the entry cannot name, or one it built
            no row at.
    """
    if name not in model.constraints:
        raise KeyError(unknown_name_message('constraint', name, sorted(model.constraints)))

    at, ordered = _row_index(model, name, coordinate)
    starts = model.tables.row_starts
    coefficients = model.tables.matrix.slice(int(starts[at]), int(starts[at + 1] - starts[at]))
    stated = model.tables.rows.slice(at, 1)
    return ConstraintRow(
        name=name,
        coordinate=ordered,
        terms=_named_terms(model, coefficients),
        sense=str(stated.item(0, 'sense')),
        rhs=float(stated.item(0, 'rhs')),
    )


def infeasible_subsystem(model: BuiltModel, found: InfeasibleSubsystemIndices) -> InfeasibleSubsystem:
    """*found*'s solver indices as the entries and coordinates that built them.

    A row carries its sense and right-hand side, a bound its side and value,
    read off the sink tables the solver was given.
    """
    rows = pl.DataFrame({ROW: found.rows}).join(
        model.tables.rows.select(pl.col(ROW).cast(pl.Int64), pl.col('sense').cast(pl.String), 'rhs'), on=ROW
    )
    columns = model.tables.cols
    bounds = pl.concat(
        [
            pl.DataFrame({COL: at, VALUE: columns[column].gather(at)}).select(COL, pl.lit(side).alias('bound'), VALUE)
            for side, column, at in (('lower', 'lb', found.lower), ('upper', 'ub', found.upper))
        ]
    )
    return InfeasibleSubsystem(
        constraints=_members(model, model.constraints, model.program.constraints, rows, ROW, ('sense', 'rhs')),
        bounds=_members(model, model.variables, model.program.variables, bounds, COL, ('bound', VALUE)),
    )


def _members(
    model: BuiltModel,
    registry: Mapping[str, indices.Indexed],
    declared: Mapping[str, program.Variable | program.Constraint],
    found: pl.DataFrame,
    index: str,
    carried: Sequence[str],
) -> dict[str, pl.DataFrame]:
    """``(dims…, *carried)`` per entry owning a solver index in *found*, in entry and index order.

    An entry owns a contiguous run of solver indices, so its share is a range test.
    """
    found = found.with_columns(pl.col(index).cast(pl.Int64))
    out = {}
    for name, held in registry.items():
        mine = found.filter((pl.col(index) >= held.start) & (pl.col(index) < held.start + held.height))
        if not mine.height:
            continue
        dims = declared[name].dims
        table = (
            held.table.with_columns(pl.col(index).cast(pl.Int64))
            .join(mine.lazy(), on=index)
            .sort(index, *carried)
            .select(*dims, *carried)
        )
        out[name] = _as_strings(table, model.attached, dims).pipe(collected)
    return out


def _row_index(model: BuiltModel, name: str, coordinate: Mapping[str, object]) -> tuple[int, dict[str, object]]:
    """The global row index constraint *name* built at *coordinate*, and that coordinate in dimension order."""
    dims = model.program.constraints[name].dims
    if set(coordinate) != set(dims):
        raise SpecsolveError(
            f"constraint '{name}' is declared over {list(dims)}, and a row is read at all of them "
            f'— got {sorted(coordinate)}. A row is one coordinate: name every dimension once, and '
            'no dimension the constraint does not have.'
        )
    table = model.constraints[name].table
    schema = table.collect_schema()
    ordered = {d: coordinate[d] for d in dims}
    matches = [pl.col(d) == _label(name, d, v, schema[d]) for d, v in ordered.items()]
    found = table.filter(matches).pipe(collected) if matches else table.pipe(collected)
    if not found.height:
        raise SpecsolveError(
            f"constraint '{name}' built no row at {ordered}. Either a `where` left the "
            'coordinate out, every term it had was absent, or the labels are not ones the '
            'dimension holds — and which of those it is, is what diagnostics() reports as an '
            'omission.'
        )
    return int(found.item(0, ROW)), ordered


def _label(name: str, dim: str, value: object, dtype: pl.DataType) -> pl.Expr:
    """*value* as a literal of *dim*'s own type, or a refusal naming what it is not.

    The cast is the check: a wrong type and an unknown ``Enum`` label fail alike.
    """
    try:
        return pl.lit(pl.Series([value], dtype=dtype).item(0), dtype=dtype)
    except (pl.exceptions.PolarsError, TypeError, OverflowError) as refused:
        raise SpecsolveError(
            f"constraint '{name}' is declared over '{dim}', which holds {dtype}, and {value!r} is "
            f'not one of its labels. Read the row at a label the dimension has.'
        ) from refused


def _named_terms(model: BuiltModel, coefficients: pl.DataFrame) -> pl.DataFrame:
    """``(variable, coordinate, coefficient)`` for one row's coefficients, in the matrix's order.

    Each entry owns a contiguous run of column indices, so a term's variable
    is a range test and its place in that table is a positional take.
    ``coordinate`` is one string because one row's terms may span variables
    with different dimensions.
    """
    wanted = coefficients[COL].to_numpy()
    named = []
    for variable, held in model.variables.items():
        inside = wanted[(wanted >= held.start) & (wanted < held.start + held.height)]
        if not inside.size:
            continue
        dims = model.program.variables[variable].dims
        at = pl.Series('#position', inside - held.start, dtype=pl.UInt32)
        picked = held.table.select(pl.col(COL), *(pl.col(d) for d in dims)).select(pl.all().gather(at))
        schema = held.table.collect_schema()
        rendered = coordinate_expr({d: schema[d] for d in dims})
        named.append(
            picked.select(
                pl.col(COL).alias(COL),
                pl.lit(variable).alias('variable'),
                rendered.alias('coordinate'),
            ).pipe(collected)
        )
    owners = (
        pl.concat(named)
        if named
        else pl.DataFrame(schema={COL: pl.Int64, 'variable': pl.String, 'coordinate': pl.String})
    )
    return (
        coefficients.with_columns(pl.col(COL).cast(pl.Int64))
        .join(owners.with_columns(pl.col(COL).cast(pl.Int64)), on=COL, how='left', maintain_order='left')
        .select('variable', 'coordinate', pl.col(COEFF).alias('coefficient'))
    )


def laid_out(
    attached: AttachedSources, held: indices.Indexed, dims: tuple[str, ...], values: pl.Series
) -> pl.LazyFrame:
    """One entry's coordinates in index order, beside its share of *values*.

    Dimension columns leave as ``String`` ([`_as_strings`][]).
    """
    return _as_strings(held.valued(dims, values), attached, dims)


def reduced_costs(tables: SinkTables, primal: pl.Series, dual: pl.Series) -> pl.Series:
    """Each column's reduced cost: the objective's gradient less the rows' gradients weighted by *dual*.

    Computed here rather than asked of each solver, so it carries
    [`dual`][specsolve._result.Result.dual]'s one sign convention on
    every sink. The gradients are taken at *primal*, which only a quadratic
    term reads.
    """
    x = primal.to_numpy()
    y = dual.to_numpy()

    def summed(cols: pl.Series, weights: np.ndarray) -> np.ndarray:
        return np.bincount(cols.to_numpy(), weights=weights, minlength=tables.column_count)

    def index(column: pl.Series) -> np.ndarray:
        return column.to_numpy().astype(np.int64)

    obj, quad, matrix, qmatrix = tables.obj, tables.quad, tables.matrix, tables.qmatrix
    coefficient_rows = np.repeat(np.arange(tables.row_count), np.diff(tables.row_starts))
    reduced = np.zeros(tables.column_count, dtype=np.float64)
    reduced += summed(obj[COL], obj[COEFF].to_numpy())
    reduced -= summed(matrix[COL], matrix[COEFF].to_numpy() * y[coefficient_rows])
    for table, weights in (
        (quad, quad[COEFF].to_numpy()),
        (qmatrix, -qmatrix[COEFF].to_numpy() * y[index(qmatrix[ROW])]),
    ):
        reduced += summed(table[COL_L], weights * x[index(table[COL_R])])
        reduced += summed(table[COL_R], weights * x[index(table[COL_L])])
    return pl.Series(VALUE, reduced, dtype=pl.Float64)


def slacks(tables: SinkTables, activity: pl.Series) -> pl.Series:
    """Each row's distance to binding at *activity*: non-negative wherever the row holds.

    ``rhs - lhs`` for ``<=`` and ``lhs - rhs`` for ``>=``, so the value does
    not depend on which side a term is written on, and ``-|rhs - lhs|`` for
    ``==``, which holds only at zero.
    """
    rows = tables.dense_rows(np.inf)
    lhs = activity.to_numpy()
    gap = rows.rhs - lhs
    slack = np.where(
        rows.sense == SENSE_CODES['<='], gap, np.where(rows.sense == SENSE_CODES['>='], -gap, -np.abs(gap))
    )
    return pl.Series(VALUE, slack, dtype=pl.Float64)


def _as_strings[F: (pl.DataFrame, pl.LazyFrame)](table: F, attached: AttachedSources, dims: Sequence[str]) -> F:
    """*table* with those of *dims* that attaching encoded as ``Enum`` cast back to ``String``.

    A caller joins a dimension column against its own data, and polars refuses
    ``Enum`` against ``String``.
    """
    return table.with_columns(pl.col(d).cast(pl.String) for d in dims if attached.is_enum_encoded(d))


def reordered(
    attached: AttachedSources,
    registry: Mapping[str, indices.Indexed],
    declared: Mapping[str, program.Variable | program.Constraint],
    tables: Mapping[str, pl.DataFrame],
) -> pl.Series:
    """A saved solution's value tables back as the positional vector — the inverse of [`laid_out`][].

    A rebuild over the same spec and sources numbers the solver indices identically
    (docs/development/architecture.md, "The relational lane").

    Raises:
        SpecsolveError: An entry whose saved table misses a coordinate the
            rebuilt model holds — the table is not this model's answer.
    """
    in_start_order = sorted((held.start, name) for name, held in registry.items())
    aligned = [
        _aligned(attached, name, registry[name], declared[name].dims, tables.get(name)) for _, name in in_start_order
    ]
    return pl.concat(aligned) if aligned else pl.Series(_SOLUTION, [], dtype=pl.Float64)


def _aligned(
    attached: AttachedSources, name: str, held: indices.Indexed, dims: tuple[str, ...], stored: pl.DataFrame | None
) -> pl.Series:
    """One entry's saved values in its index order — its slice of the vector.

    An entry the rebuild leaves wholly absent has an empty slice, so a
    missing *stored* is no error there.
    """
    if held.height == 0:
        return pl.Series(_SOLUTION, [], dtype=pl.Float64)
    if stored is None:
        raise SpecsolveError(
            f"the saved answer holds no '{name}' table, but this model builds it, so it is not this model's "
            f'answer. Re-solve rather than read.'
        )
    if not dims:
        return stored[VALUE].rename(_SOLUTION)
    order = _as_strings(held.table.select(*dims).pipe(collected), attached, dims).with_row_index(_INDEX_ORDER)
    joined = order.join(stored, on=list(dims), how='left').sort(_INDEX_ORDER)
    if joined[VALUE].null_count():
        raise SpecsolveError(
            f"the saved answer's '{name}' table does not cover every coordinate this model builds, so it "
            f'is not an answer to this model. Re-solve rather than read.'
        )
    return joined[VALUE].rename(_SOLUTION)


def expressions(
    builder: QueryBuilder, named: Mapping[str, program.NamedExpression], lower: Lower | None
) -> Expressions:
    """The reads [`evaluate`][specsolve._result.Result.evaluate] is built from, over one builder.

    No query is built until a reader is called. Without *lower* there is no
    spec as written to compile against, so an undeclared expression has no reader.
    """

    def reader(name: str, expression: program.Expression) -> Callable[[], pl.DataFrame]:
        return lambda: expression_table(name, expression, builder)

    declared = {name: reader(name, e.expression) for name, e in named.items()}
    if lower is None:
        return Expressions(declared)

    def evaluate(written: str | Mapping[str, object]) -> pl.DataFrame:
        name = written if isinstance(written, str) and written in named else 'the expression'
        return expression_table(name, lower(written), builder)

    return Expressions(declared, evaluate)


def expression_table(name: str, expr: program.Expression, builder: QueryBuilder) -> pl.DataFrame:
    """Named expression *expr* evaluated at the solve *builder* holds — ``(dims…, value)``.

    It answers as a constraint over the same expression would: a coordinate a
    parameter does not cover contributes zero, and one where a variable is
    absent has no row, or a zero under ``missing: 0``. A quotient has no row
    where its divisor is absent or zero. A variable-free expression is one row.
    Dimensions come in the order the spec declares them, rows in index order.

    Raises:
        DataError: A divisor parameter with no row where the expression
            divides.
        SpecsolveError: The expression reads a dual and the solve left none.
    """
    context = f"named expression '{name}'"
    terms = builder.expression(expr, context, reported=True).constant
    dims = builder.scope.spanned(terms)
    carrier = indices.table(builder.scope, dims, None, _EXPRESSION_ROW, 0, presences_of(terms)).lazy()

    coverage.refuse_null_constants(
        [_holed(carrier, terms), *(p.table.with_columns(pl.col(CONSTANT).fill_null(0.0)) for p in terms)],
        program.parameters_of(*coverage.divisors_of(expr)),
        context,
        reported_divisor_message,
    )

    added = builder.summed_onto(terms, carrier, absent='zero')
    out = added.select(_EXPRESSION_ROW, *dims, pl.col(CONSTANT).alias(VALUE)).pipe(collected)
    ordered = indices.in_position_order(out, _EXPRESSION_ROW).drop(_EXPRESSION_ROW)
    return _as_strings(ordered, builder.scope.data, dims)


def _holed(carrier: pl.LazyFrame, terms: Sequence[Term]) -> pl.LazyFrame:
    """One ``CONSTANT`` per row of *carrier*, null where any of *terms* holds a null there, else zero.

    The refusal counts the coordinates the expression divides at, and two
    terms can hold a hole at one coordinate, as ``v / p - v / p`` does.
    """
    holed: pl.Expr = pl.lit(False)
    for i, p in enumerate(terms):
        column = f'__hole {i}__'
        if p.dims:
            hole = (
                p.table.filter(pl.col(CONSTANT).is_null())
                .select(*p.dims)
                .unique()
                .with_columns(pl.lit(True).alias(column))
            )
            carrier = carrier.join(hole, on=list(p.dims), how='left')
        else:
            carrier = carrier.join(p.table.select(pl.col(CONSTANT).is_null().any().alias(column)), how='cross')
        holed = holed | pl.col(column).fill_null(False)
    return carrier.select(pl.when(holed).then(None).otherwise(0.0).alias(CONSTANT))


def matched_basis(model: BuiltModel, columns: Mapping[str, pl.LazyFrame], rows: Mapping[str, pl.LazyFrame]) -> Basis:
    """Another answer's basis, *columns* and *rows* as its readers return them, laid onto this build by coordinate.

    A coordinate both builds hold keeps its status. A column only this build
    holds starts nonbasic at a bound, and a row only this build holds starts
    basic, which is how a row gained between two builds enters without moving
    the vertex. An entry whose dimensions changed is new. The result is then
    [`_counted`][] to as many basic statuses as rows, which is what a solver needs to
    take it, and [`seated`][specsolve._sinks.solver.seated]
    on this build's bounds.

    Raises:
        SpecsolveError: No status lands at a coordinate this build holds.
    """
    tables = model.tables
    status = pl.col(VALUE).cast(BASIS).to_physical()
    placed_columns = np.full(tables.column_count, AT_LOWER, dtype=np.int8)
    placed = _place(placed_columns, model, model.variables, model.program.variables, columns, status)
    placed_rows = np.full(tables.row_count, BASIC, dtype=np.int8)
    placed += _place(placed_rows, model, model.constraints, model.program.constraints, rows, status)
    _refuse_nothing_placed(placed)
    return seated(tables, *_counted(placed_columns, placed_rows))


def matched_values(model: BuiltModel, values: Mapping[str, pl.LazyFrame]) -> np.ndarray:
    """A value per column from *values*, ``(dims…, value)`` tables per variable, laid onto this build by coordinate.

    NaN where *values* gives none: a variable it does not name, a coordinate
    it lacks, an entry whose dimensions changed.

    Raises:
        SpecsolveError: No value lands at a coordinate this build holds.
    """
    out = np.full(model.tables.column_count, np.nan)
    _refuse_nothing_placed(
        _place(out, model, model.variables, model.program.variables, values, pl.col(VALUE).cast(pl.Float64))
    )
    return out


def _refuse_nothing_placed(placed: int) -> None:
    """Refuse a start that lands nowhere, which would start nothing yet read as a start taken."""
    if not placed:
        raise SpecsolveError(
            'start= gives no value at any coordinate this model holds, so it would start nothing. Name the '
            'entries as the spec declares them, and their coordinates as the readers return them.'
        )


def _place(
    into: np.ndarray,
    model: BuiltModel,
    held: Mapping[str, indices.Indexed],
    declared: Mapping[str, program.Variable] | Mapping[str, program.Constraint],
    previous: Mapping[str, pl.LazyFrame],
    value: pl.Expr,
) -> int:
    """*value* off each table of *previous*, written into *into* at the solver index of the same coordinate in *held*; how many landed.

    The join casts every dimension to a string on both sides, so a table typed
    otherwise than the read-back still matches: ``3`` and ``'3'`` are one
    label. A table whose columns are not its entry's dimensions and ``value``
    is skipped, as is a coordinate only one side holds.
    """
    positions = pl.Series(VALUE, np.arange(len(into), dtype=np.int64))
    placed = 0
    for name, indexed in held.items():
        dims = list(declared[name].dims)
        before = previous.get(name)
        if before is None or set(before.collect_schema().names()) != {*dims, VALUE}:
            continue
        here = laid_out(model.attached, indexed, tuple(dims), positions).select(
            *(pl.col(dim).cast(pl.String) for dim in dims), pl.col(VALUE).alias(_INDEX_ORDER)
        )
        given = before.select(*(pl.col(dim).cast(pl.String) for dim in dims), value)
        found = (here.join(given, on=dims, how='inner') if dims else here.join(given, how='cross')).pipe(collected)
        into[found[_INDEX_ORDER].to_numpy()] = found[VALUE].to_numpy()
        placed += found.height
    return placed


def _counted(columns: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """*columns* and *rows* with exactly as many basic statuses as rows, the count every solver's simplex needs.

    A basis carried across a rebuild has one too many for each nonbasic row
    the rebuild dropped, and one too few for each basic column it dropped.
    The surplus leaves from the last basic columns, which go to a bound; the
    shortfall is made up by the last nonbasic rows, whose slacks enter. Which
    ones is arbitrary: a basis the count makes singular, each solver repairs.
    """
    columns, rows = columns.copy(), rows.copy()
    surplus = int((columns == BASIC).sum() + (rows == BASIC).sum()) - len(rows)
    if surplus > 0:
        leaving = np.flatnonzero(columns == BASIC)[-surplus:]
        columns[leaving] = AT_LOWER
    elif surplus < 0:
        entering = np.flatnonzero(rows != BASIC)[surplus:]
        rows[entering] = BASIC
    return columns, rows


# ---------------------------------------------------------------------------
# a solve, read back as a result
# ---------------------------------------------------------------------------


def result(
    model: BuiltModel,
    answer: SolveAnswer,
    solver_name: str,
    lower: Lower | None,
    outputs: frozenset[Output] = frozenset(),
) -> Result:
    """*answer*'s vectors laid out against *model*, as a [`Result`][].

    Takes the answer rather than the solver, so that a benchmark can time
    reading one back without a solve; every vector must span this build.

    Args:
        model: The build *answer* answered.
        answer: What the solver returned.
        solver_name: The solver that returned it, for the reason it left no ray.
        lower: How an expression the caller writes becomes a program node, for
            [`evaluate`][specsolve._result.Result.evaluate].
        outputs: Which of [`OUTPUTS`][specsolve._outputs.OUTPUTS] the result
            carries, already checked.
    """
    assert answer.primal is not None or not answer.status.is_readable, (
        'a readable status must come with a primal vector'
    )
    assert (answer.activity is None) == (answer.primal is None), (
        'activity travels with the primal: every sink reads it whenever a solution exists, mixed-integer included'
    )
    primals, duals, rays = read_back(model, answer.primal, answer.dual, answer.dual_ray)
    status = answer.status
    no_duals = (
        None
        if answer.dual is not None
        else _no_duals_message(discrete(model), status.termination_condition, _quadratic_constraints(model))
    )
    return Result(
        Answer(
            Record.of(status.termination_condition, answer.objective, has_primal=status.is_readable, solved_at=None),
            tables={
                'primal': primals,
                'dual': duals,
                **{kind: _output(model, kind, answer) for kind in kinds_of(outputs)},
            },
            reasons={} if no_duals is None else {'dual': {WHOLE: no_duals}},
            wording=status.solver_wording,
        ),
        _expressions=_expressions(model, answer.primal, answer.dual, no_duals, lower),
        _dual_rays=rays,
        _no_dual_ray=None if answer.dual_ray is not None else _no_dual_ray_message(status, solver_name),
    )


def read_back(
    model: BuiltModel, primal: pl.Series | None, dual: pl.Series | None, dual_ray: pl.Series | None
) -> tuple[dict[str, pl.LazyFrame], ...]:
    """One solve's vectors as one table per entry: the primal, the duals and the ray.

    The tables reference *model*'s index tables, which a rebuild replaces
    rather than mutates. A ``None`` vector yields no tables rather than empty
    ones.
    """
    return (
        {} if primal is None else _per_variable(model, primal),
        {} if dual is None else _per_constraint(model, dual),
        {} if dual_ray is None else _per_constraint(model, dual_ray),
    )


def _output(model: BuiltModel, kind: str, answer: SolveAnswer) -> Mapping[str, pl.LazyFrame]:
    """The tables of *kind*, one of the [`OUTPUT_KINDS`][specsolve._outputs.OUTPUT_KINDS], computed only when asked for; empty where a vector it needs is absent."""
    match kind:
        case 'activity':
            return {} if answer.activity is None else _per_constraint(model, answer.activity)
        case 'reduced_cost':
            if answer.primal is None or answer.dual is None:
                return {}
            return _per_variable(model, reduced_costs(model.tables, answer.primal, answer.dual))
        case 'slack':
            return {} if answer.activity is None else _per_constraint(model, slacks(model.tables, answer.activity))
        case 'variable_basis':
            return {} if answer.basis is None else _per_variable(model, _statuses(answer.basis.columns))
        case 'constraint_basis':
            return {} if answer.basis is None else _per_constraint(model, _statuses(answer.basis.rows))
        case _:
            raise AssertionError(f'{kind!r} is none of the OUTPUT_KINDS, which kinds_of draws from')


def _statuses(codes: np.ndarray) -> pl.Series:
    """Basis status codes as the series a table is laid out from."""
    return pl.Series(VALUE, BASIS_STATUSES, dtype=BASIS).gather(codes)


def _per_constraint(model: BuiltModel, values: pl.Series) -> dict[str, pl.LazyFrame]:
    """A vector over the rows as one table per constraint."""
    return {
        name: laid_out(model.attached, model.constraints[name], c.dims, values)
        for name, c in model.program.constraints.items()
    }


def _per_variable(model: BuiltModel, values: pl.Series) -> dict[str, pl.LazyFrame]:
    """A vector over the columns as one table per variable."""
    return {
        name: laid_out(model.attached, model.variables[name], v.dims, values)
        for name, v in model.program.variables.items()
    }


def _expressions(
    model: BuiltModel, primal: pl.Series | None, dual: pl.Series | None, no_duals: str | None, lower: Lower | None
) -> Expressions:
    """What [`evaluate`][specsolve._result.Result.evaluate] reads through, at the solution *primal* and *dual* hold.

    The readers close over a snapshot the result owns, including a copy of
    the variable registry, so they keep answering after an update or
    ``close()``. A solve that left no primal has none.
    """
    if primal is None:
        return Expressions()
    solution = Solution(primal, dual, dict(model.constraints), no_duals)
    builder = QueryBuilder(Scope(model.program, model.attached, dict(model.variables)), solution)
    return expressions(builder, model.program.expressions, lower)


def evaluator(
    model: BuiltModel,
    primals: Mapping[str, pl.DataFrame],
    duals: Mapping[str, pl.DataFrame] | None,
    no_duals: str | None,
    lower: Lower,
) -> Callable[[str | Mapping[str, object]], pl.DataFrame]:
    """The reader of an expression the file never named, for a saved solution put back against *model*.

    *model* supplies the solver indices that put the saved values back in
    vector order.

    Args:
        model: A build of the spec and the sources the saved solution answered.
        primals: The saved ``(dims…, value)`` table per variable.
        duals: The same per constraint, or ``None`` where the solve left no duals.
        no_duals: Why there are no duals, or ``None`` when *duals* holds them.
        lower: How an expression the caller writes becomes a program node.
    """
    primal = reordered(model.attached, model.variables, model.program.variables, primals)
    dual = reordered(model.attached, model.constraints, model.program.constraints, duals) if duals is not None else None
    evaluate = _expressions(model, primal, dual, no_duals, lower).undeclared
    assert evaluate is not None, 'a reader given a way to lower an expression reads an undeclared one'
    return evaluate


def expressions_of(program: program.Program, sources: Mapping[str, pl.LazyFrame], lower: Lower | None) -> Expressions:
    """Attach *sources* and defer the reads [`specsolve.evaluate`][] values one expression through.

    Args:
        program: A compiled program with no variables — a calculation, so every
            expression has a value with no solver.
        sources: Tidied sources, as
            [`tidy_sources`][specsolve._sources.tidy_sources] produces.
        lower: How an expression the caller writes becomes a program node.

    Returns:
        The readers, each of which builds its query on call.
    """
    builder = QueryBuilder(Scope(program, attach(program, sources), {}))
    return expressions(builder, program.expressions, lower)


def discrete(model: BuiltModel) -> list[str]:
    """The variables *model* declared as anything but continuous."""
    return sorted(n for n, v in model.program.variables.items() if v.domain != 'continuous')


def _quadratic_constraints(model: BuiltModel) -> list[str]:
    """The constraints *model* declared as quadratic — a fact about the model, not the solve."""
    return sorted(n for n, c in model.program.constraints.items() if declares_quadratic(c))


def _no_dual_ray_message(status: SolveStatus, solver_name: str) -> str:
    """Why this answer carries no certificate of infeasibility.

    It takes the sink's name rather than asking the sink, whose session may
    already have been let go.
    """
    if status.termination_condition != 'infeasible':
        return (
            f'a dual ray certifies that a model has no solution, and this solve terminated '
            f'{status.termination_condition!r} rather than infeasible. Read dual() for the prices of a '
            f'model that does have one.'
        )
    ask = sinks.SOLVERS[solver_name].dual_ray_advice
    return f'the model is infeasible, and the {solver_name} sink returned no dual ray to certify it. ' + (
        f'{ask}.' if ask else 'A ray comes from the simplex, so a run that never reached it has none to give.'
    )


def _no_duals_message(discrete: Sequence[str], termination_condition: str, quadratic_rows: Sequence[str]) -> str:
    """The message for a solve that left values but no duals."""
    if quadratic_rows and not discrete:
        names = ', '.join(f"'{n}'" for n in quadratic_rows)
        return (
            f"a quadratic constraint prices only under gurobi's QCPDual, which is off by default: "
            f'{names} {"is" if len(quadratic_rows) == 1 else "are"} quadratic. Asking for those '
            f'prices makes the solver take the convex path, so a nonconvex row that solves without '
            f'them fails with them — which is why this is yours to ask for rather than ours to '
            f"assume. Re-solve with solver_options={{'QCPDual': 1}} if the model is convex."
        )
    if discrete:
        names = ', '.join(f"'{n}'" for n in discrete)
        return (
            f'duals are undefined for a mixed-integer model: {names} '
            f'{"is" if len(discrete) == 1 else "are"} not continuous. '
            f'Drop the integrality to price the LP relaxation instead.'
        )
    return (
        f'the solver returned no dual solution, though the solve terminated '
        f'{termination_condition!r}. Duals come from a simplex basis, which a '
        f'run stopped short of one does not have.'
    )


# ---------------------------------------------------------------------------
# a start, laid onto a build
# ---------------------------------------------------------------------------


def matched_start(
    model: BuiltModel, start: Result | Mapping[str, Mapping[str, pl.LazyFrame]], solver_name: str
) -> Basis | np.ndarray:
    """*start* laid onto *model*: a basis for an LP given one, else a value per column, NaN where none is given.

    *start* is an earlier answer, or a [`Start`][specsolve.types.Start]
    already read ([`read_start`][specsolve._start.read_start]). A
    mixed-integer model starts from values, so a basis given it is not used.
    An LP given both starts from the basis.

    Raises:
        SpecsolveError: A start that gives this model nothing it starts
            from or lands nowhere on this build, or values for an LP that
            *solver_name* cannot take.
    """
    given = start._start() if isinstance(start, Result) else start
    mixed_integer = bool(discrete(model))
    if not mixed_integer and (given.get('variable_basis') or given.get('constraint_basis')):
        return matched_basis(model, given.get('variable_basis', {}), given.get('constraint_basis', {}))
    if not given.get('primal'):
        raise SpecsolveError(_nothing_to_start_message(mixed_integer))
    values = matched_values(model, given['primal'])
    if not mixed_integer:
        _checked_lp_values(sinks.solver(solver_name), solver_name, values)
    return values


def _nothing_to_start_message(discrete: bool) -> str:
    """Why a start gives a model nothing it starts from: a basis alone for a mixed-integer model, or no table at all."""
    if discrete:
        return (
            'start= gives this mixed-integer model a basis alone, and a basis starts only an LP. Give values '
            "under 'primal' as well."
        )
    return "start= gives nothing to start from. Give a table under 'primal', 'variable_basis' or 'constraint_basis'."


def _checked_lp_values(sink: type[sinks.Solver], solver_name: str, values: np.ndarray) -> None:
    """Refuse a start of values for an LP that *sink* cannot take, and warn of one it takes where no gain is known.

    Raises:
        SpecsolveError: *sink* cannot take values that leave a column out.
    """
    partial = bool(np.isnan(values).any())
    match sink.lp_values['partial' if partial else 'complete']:
        case 'refused':
            raise SpecsolveError(
                f'{solver_name} cannot start an LP from values that leave a column out, and this start gives '
                f'no value at {int(np.isnan(values).sum())} of {len(values)}. Give a value at every '
                "coordinate of every variable, or start from an earlier answer solved with outputs={'basis'}."
            )
        case 'no_gain':
            warnings.warn(
                f'{solver_name} takes {"a partial" if partial else "a"} start of values for an LP, and no gain '
                f'from one is known: it measured no fewer iterations than a solve started cold. An earlier '
                f"answer solved with outputs={{'basis'}} starts an LP from its basis. If you see a start of "
                f'values speed up a solve, please report it at https://github.com/fluxopt/specsolve/issues.',
                SpecsolveWarning,
                stacklevel=5,
            )
