"""Reading a built model back: one constraint row, a solve's tables, a named expression."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from mathspec import program

from specsolve._collect import collected
from specsolve._engine import coverage, indices
from specsolve._engine.terms import Term, presences_of
from specsolve._messages import coordinate_expr, reported_divisor_message, unknown_name_message
from specsolve._names import COEFF, COL, COL_L, COL_R, CONSTANT, ROW, VALUE
from specsolve._result import ConstraintRow, Expressions, InfeasibleSubsystem
from specsolve._sinks.solver import AT_LOWER, BASIC, BASIS, seated
from specsolve._sinks.tables import SENSE_CODES
from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import numpy as np

    from specsolve._engine.assembly import BuiltModel
    from specsolve._engine.attaching import AttachedSources
    from specsolve._engine.queries import QueryBuilder
    from specsolve._sinks.solver import Basis, InfeasibleSubsystemIndices
    from specsolve._sinks.tables import SinkTables

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
    import numpy as np

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
    import numpy as np

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
    builder: QueryBuilder,
    named: Mapping[str, program.NamedExpression],
    lower: Callable[[str | Mapping[str, object]], program.Expression] | None,
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
    absent has no row, or a zero under ``missing: neutral``. A quotient has no row
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
    import numpy as np

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
    import numpy as np

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
    import numpy as np

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
    import numpy as np

    columns, rows = columns.copy(), rows.copy()
    surplus = int((columns == BASIC).sum() + (rows == BASIC).sum()) - len(rows)
    if surplus > 0:
        leaving = np.flatnonzero(columns == BASIC)[-surplus:]
        columns[leaving] = AT_LOWER
    elif surplus < 0:
        entering = np.flatnonzero(rows != BASIC)[surplus:]
        rows[entering] = BASIC
    return columns, rows
