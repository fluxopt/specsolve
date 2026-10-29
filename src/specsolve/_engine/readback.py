"""Reading a built model back: one constraint row, a solve's frames, a named expression."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import polars as pl
from mathspec import program

from specsolve._answer_layout import AT_LOWER, BASIC, BASIS
from specsolve._collect import collected
from specsolve._engine import coverage, labels
from specsolve._engine.compiler import varied_label
from specsolve._engine.pieces import absence_restrictions
from specsolve._engine.scope import UNIT, join_on
from specsolve._messages import coordinate_expr, unknown_name_message
from specsolve._names import VALUE
from specsolve._result import ConstraintRow, InfeasibleSubsystem
from specsolve._sinks.handoff import SENSE_CODES
from specsolve._sinks.solver import settled
from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import numpy as np

    from specsolve._engine.assembly import BuiltModel
    from specsolve._engine.attaching import AttachedSources
    from specsolve._engine.compiler import Compiler
    from specsolve._engine.pieces import CompiledExpression, Piece
    from specsolve._sinks.handoff import Handoff
    from specsolve._sinks.solver import Basis, InfeasibleSubsystemIndices

#: Scratch columns. The spaces make them unrepresentable as declared names.
_SOLUTION = '__solution value__'
_EXPRESSION_ROW = '__expression row__'
_LABEL_ORDER = '__label order__'


def row(model: BuiltModel, name: str, coordinate: Mapping[str, object]) -> ConstraintRow:
    """One built constraint row, spelled back out. See [`row`][specsolve._api.Model.row].

    Raises:
        KeyError: No constraint of that name.
        SpecsolveError: A coordinate the declaration cannot name, or one it built
            no row at.
    """
    if name not in model.constraints:
        raise KeyError(unknown_name_message('constraint', name, sorted(model.constraints)))

    at, ordered = _row_index(model, name, coordinate)
    starts = model.handoff.row_starts
    entries = model.handoff.matrix.slice(int(starts[at]), int(starts[at + 1] - starts[at]))
    stated = model.handoff.rows.slice(at, 1)
    return ConstraintRow(
        name=name,
        coordinate=ordered,
        terms=_named_terms(model, entries),
        sense=str(stated.item(0, 'sense')),
        rhs=float(stated.item(0, 'rhs')),
    )


def infeasible_subsystem(model: BuiltModel, found: InfeasibleSubsystemIndices) -> InfeasibleSubsystem:
    """*found*'s solver indices as the declarations and coordinates that built them.

    A row carries its sense and right-hand side, a bound its side and value,
    read off the hand-off the solver was given.
    """
    rows = pl.DataFrame({'row': found.rows}).join(
        model.handoff.rows.select(pl.col('row').cast(pl.Int64), pl.col('sense').cast(pl.String), 'rhs'), on='row'
    )
    columns = model.handoff.cols
    bounds = pl.concat(
        [
            pl.DataFrame({'var_label': indices, VALUE: columns[column].gather(indices)}).select(
                'var_label', pl.lit(side).alias('bound'), VALUE
            )
            for side, column, indices in (('lower', 'lb', found.lower), ('upper', 'ub', found.upper))
        ]
    )
    return InfeasibleSubsystem(
        constraints=_members(model, model.constraints, model.program.constraints, rows, 'row', ('sense', 'rhs')),
        bounds=_members(model, model.variables, model.program.variables, bounds, 'var_label', ('bound', VALUE)),
    )


def _members(
    model: BuiltModel,
    registry: Mapping[str, labels.Labelled],
    declared: Mapping[str, program.Variable | program.Constraint],
    found: pl.DataFrame,
    label: str,
    carried: Sequence[str],
) -> dict[str, pl.DataFrame]:
    """``(dims…, *carried)`` per declaration owning a label in *found*, in declaration and label order.

    A declaration owns a contiguous run of labels, so its share is a range test.
    """
    found = found.with_columns(pl.col(label).cast(pl.Int64))
    out = {}
    for name, held in registry.items():
        mine = found.filter((pl.col(label) >= held.start) & (pl.col(label) < held.start + held.height))
        if not mine.height:
            continue
        dims = declared[name].dims
        frame = (
            held.frame.with_columns(pl.col(label).cast(pl.Int64))
            .join(mine.lazy(), on=label)
            .sort(label, *carried)
            .select(*dims, *carried)
        )
        out[name] = _as_strings(frame, model.attached, dims).pipe(collected)
    return out


def _row_index(model: BuiltModel, name: str, coordinate: Mapping[str, object]) -> tuple[int, dict[str, object]]:
    """The global row index constraint *name* built at *coordinate*, and that coordinate in dim order."""
    dims = model.program.constraints[name].dims
    if set(coordinate) != set(dims):
        raise SpecsolveError(
            f"constraint '{name}' is declared over {list(dims)}, and a row is read at all of them "
            f'— got {sorted(coordinate)}. A row is one coordinate: name every dim once, and no dim '
            'the declaration does not have.'
        )
    frame = model.constraints[name].frame
    schema = frame.collect_schema()
    ordered = {d: coordinate[d] for d in dims}
    predicates = [pl.col(d) == _label(name, d, v, schema[d]) for d, v in ordered.items()]
    found = frame.filter(predicates).pipe(collected) if predicates else frame.pipe(collected)
    if not found.height:
        raise SpecsolveError(
            f"constraint '{name}' built no row at {ordered}. Either a `where` masked the "
            'coordinate out, every term it had was absent, or the labels are not ones the '
            'dimension holds — and which of those it is, is what diagnostics() reports as an '
            'omission.'
        )
    return int(found.item(0, 'row')), ordered


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


def _named_terms(model: BuiltModel, entries: pl.DataFrame) -> pl.DataFrame:
    """``(variable, coordinate, coefficient)`` for one row's matrix entries, in the entries' order.

    Each declaration owns a contiguous run of column indices, so a term's
    variable is a range test and its place in that frame is a positional take.
    ``coordinate`` is one string because one row's terms may span variables
    with different dims.
    """
    wanted = entries['col'].to_numpy()
    named = []
    for variable, held in model.variables.items():
        inside = wanted[(wanted >= held.start) & (wanted < held.start + held.height)]
        if not inside.size:
            continue
        dims = model.program.variables[variable].dims
        at = pl.Series('#position', inside - held.start, dtype=pl.UInt32)
        picked = held.frame.select(pl.col('var_label'), *(pl.col(d) for d in dims)).select(pl.all().gather(at))
        schema = held.frame.collect_schema()
        rendered = coordinate_expr({d: schema[d] for d in dims})
        named.append(
            picked.select(
                pl.col('var_label').alias('col'),
                pl.lit(variable).alias('variable'),
                rendered.alias('coordinate'),
            ).pipe(collected)
        )
    labelled = (
        pl.concat(named)
        if named
        else pl.DataFrame(schema={'col': pl.Int64, 'variable': pl.String, 'coordinate': pl.String})
    )
    return (
        entries.with_columns(pl.col('col').cast(pl.Int64))
        .join(labelled.with_columns(pl.col('col').cast(pl.Int64)), on='col', how='left', maintain_order='left')
        .select('variable', 'coordinate', pl.col('coeff').alias('coefficient'))
    )


def laid_out(
    attached: AttachedSources, held: labels.Labelled, dims: tuple[str, ...], values: pl.Series
) -> pl.LazyFrame:
    """One declaration's coordinates in label order, beside its share of *values*.

    Dim columns leave as ``String`` ([`_as_strings`][]).
    """
    return _as_strings(held.valued(dims, values), attached, dims)


def reduced_costs(handoff: Handoff, primal: pl.Series, dual: pl.Series) -> pl.Series:
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
        return np.bincount(cols.to_numpy(), weights=weights, minlength=handoff.column_count)

    def index(column: pl.Series) -> np.ndarray:
        return column.to_numpy().astype(np.int64)

    obj, quad, matrix, qmatrix = handoff.obj, handoff.quad, handoff.matrix, handoff.qmatrix
    entry_rows = np.repeat(np.arange(handoff.row_count), np.diff(handoff.row_starts))
    reduced = np.zeros(handoff.column_count, dtype=np.float64)
    reduced += summed(obj['col'], obj['coeff'].to_numpy())
    reduced -= summed(matrix['col'], matrix['coeff'].to_numpy() * y[entry_rows])
    for frame, weights in (
        (quad, quad['coeff'].to_numpy()),
        (qmatrix, -qmatrix['coeff'].to_numpy() * y[index(qmatrix['row'])]),
    ):
        reduced += summed(frame['col_l'], weights * x[index(frame['col_r'])])
        reduced += summed(frame['col_r'], weights * x[index(frame['col_l'])])
    return pl.Series(VALUE, reduced, dtype=pl.Float64)


def slacks(handoff: Handoff, activity: pl.Series) -> pl.Series:
    """Each row's distance to binding at *activity*: non-negative wherever the row holds.

    ``rhs - lhs`` for ``<=`` and ``lhs - rhs`` for ``>=``, so the value does
    not depend on which side a term is written on, and ``-|rhs - lhs|`` for
    ``==``, which holds only at zero.
    """
    import numpy as np

    rows = handoff.dense_rows(np.inf)
    lhs = activity.to_numpy()
    gap = rows.rhs - lhs
    slack = np.where(
        rows.sense == SENSE_CODES['<='], gap, np.where(rows.sense == SENSE_CODES['>='], -gap, -np.abs(gap))
    )
    return pl.Series(VALUE, slack, dtype=pl.Float64)


def _as_strings[F: (pl.DataFrame, pl.LazyFrame)](frame: F, attached: AttachedSources, dims: Sequence[str]) -> F:
    """*frame* with those of *dims* that attaching encoded as ``Enum`` cast back to ``String``.

    A caller joins a dim column against its own data, and polars refuses
    ``Enum`` against ``String``.
    """
    return frame.with_columns(pl.col(d).cast(pl.String) for d in dims if attached.is_enum_encoded(d))


def reordered(
    attached: AttachedSources,
    registry: Mapping[str, labels.Labelled],
    declared: Mapping[str, program.Variable | program.Constraint],
    frames: Mapping[str, pl.DataFrame],
) -> pl.Series:
    """A saved solution's value frames back as the positional vector — the inverse of [`laid_out`][].

    A rebuild over the same spec and sources numbers the labels identically
    (docs/development/architecture.md, "The relational lane").

    Raises:
        SpecsolveError: A declaration whose saved frame misses a coordinate the
            rebuilt model holds — the frame is not this model's answer.
    """
    in_start_order = sorted((held.start, name) for name, held in registry.items())
    pieces = [
        _aligned(attached, name, registry[name], declared[name].dims, frames.get(name)) for _, name in in_start_order
    ]
    return pl.concat(pieces) if pieces else pl.Series(_SOLUTION, [], dtype=pl.Float64)


def _aligned(
    attached: AttachedSources, name: str, held: labels.Labelled, dims: tuple[str, ...], stored: pl.DataFrame | None
) -> pl.Series:
    """One declaration's saved values in its label order — its slice of the vector.

    A declaration the rebuild masks away entirely has an empty slice, so a
    missing *stored* is no error there.
    """
    if held.height == 0:
        return pl.Series(_SOLUTION, [], dtype=pl.Float64)
    if stored is None:
        raise SpecsolveError(
            f"the saved answer holds no '{name}' frame, but this model builds it, so it is not this model's "
            f'answer. Re-solve rather than read.'
        )
    if not dims:
        return stored[VALUE].rename(_SOLUTION)
    order = _as_strings(held.frame.select(*dims).pipe(collected), attached, dims).with_row_index(_LABEL_ORDER)
    joined = order.join(stored, on=list(dims), how='left').sort(_LABEL_ORDER)
    if joined[VALUE].null_count():
        raise SpecsolveError(
            f"the saved answer's '{name}' frame does not cover every coordinate this model builds, so it "
            f'is not an answer to this model. Re-solve rather than read.'
        )
    return joined[VALUE].rename(_SOLUTION)


def readers(
    compiler: Compiler,
    named: Mapping[str, program.NamedExpression],
    lower: Callable[[str | Mapping[str, object]], program.Expression] | None,
) -> tuple[dict[str, Callable[[], pl.DataFrame]], Callable[[str | Mapping[str, object]], pl.DataFrame] | None]:
    """The reads [`evaluate`][specsolve._result.Result.evaluate] is built from, over one compiler.

    One deferred reader per declared name, and an ad-hoc evaluator. Nothing
    compiles until a reader is called. Without *lower* there is no spec as
    written to lower against, so the ad-hoc evaluator is ``None``.
    """

    def reader(name: str, expression: program.Expression) -> Callable[[], pl.DataFrame]:
        return lambda: expression_frame(name, expression, compiler)

    declared = {name: reader(name, e.expression) for name, e in named.items()}
    if lower is None:
        return declared, None

    def evaluate(written: str | Mapping[str, object]) -> pl.DataFrame:
        return expression_frame('the expression', lower(written), compiler)

    return declared, evaluate


def expression_frame(name: str, expr: program.Expression, compiler: Compiler) -> pl.DataFrame:
    """Named expression *expr* evaluated at the solve *compiler* holds — ``(dims…, value)``.

    It answers as a constraint over the same expression would: a coordinate a
    parameter does not cover contributes zero, and one where a variable is
    absent has no row, or a zero under ``missing: neutral``. A quotient has no row
    where its divisor is absent or zero. A variable-free expression is one row.
    Dims come in declaration order, rows in label order.

    Raises:
        DataError: A divisor parameter with no row where the expression
            divides.
        SpecsolveError: The expression reads a dual and the solve left none.
    """
    context = f"named expression '{name}'"
    compiled = compiler.expression(expr, context, reported=True)

    coverage.refuse_null_constants(
        [p.frame for p in compiled.consts],
        program.parameters_of(*coverage.divisors_of(expr)),
        context,
        _reported_divisor_message,
    )

    pieces = compiled.consts
    dims = compiler.scope.spanned(pieces)
    carrier = labels.frame(compiler.scope, dims, None, _EXPRESSION_ROW, 0, absence_restrictions(pieces)).lazy()
    added = compiler.summed_onto(pieces, carrier, absent='zero')
    out = added.select(_EXPRESSION_ROW, *dims, pl.col('cval').alias(VALUE)).pipe(collected)
    ordered = labels.in_position_order(out, _EXPRESSION_ROW).drop(_EXPRESSION_ROW)
    return _as_strings(ordered, compiler.scope.data, dims)


def _reported_divisor_message(name: str, missing: int) -> str:
    """The message for a divisor parameter a reported expression reads short of a row."""
    return (
        f"parameter '{name}' is used as a divisor but has no row at {missing} of the coordinates "
        f'the expression divides at. A missing row of a neutral parameter is not absence, so the '
        f'quotient is not dropped there, and there is no number to divide by.\n'
        f'  Supply the missing rows.\n'
        f'  Declare the parameter `missing: absent`, if the quotient has no value there.'
    )


def matched_basis(model: BuiltModel, columns: Mapping[str, pl.LazyFrame], rows: Mapping[str, pl.LazyFrame]) -> Basis:
    """Another answer's basis, *columns* and *rows* as its readers return them, laid onto this build by coordinate.

    A coordinate both builds hold keeps its status. A column only this build
    holds starts nonbasic at a bound, and a row only this build holds starts
    basic, which is how a row gained between two builds enters without moving
    the vertex. A declaration whose dims changed is new. The result is then
    [`_counted`][] to one basic entry per row, which is what a solver needs to
    take it, and [`settled`][specsolve._sinks.solver.settled]
    on this build's bounds.

    Raises:
        SpecsolveError: No status lands at a coordinate this build holds.
    """
    import numpy as np

    handoff = model.handoff
    status = pl.col(VALUE).cast(BASIS).to_physical()
    placed_columns = np.full(handoff.column_count, AT_LOWER, dtype=np.int8)
    placed = _place(placed_columns, model, model.variables, model.program.variables, columns, status)
    placed_rows = np.full(handoff.row_count, BASIC, dtype=np.int8)
    placed += _place(placed_rows, model, model.constraints, model.program.constraints, rows, status)
    _refuse_nothing_placed(placed)
    return settled(handoff, *_counted(placed_columns, placed_rows))


def matched_values(model: BuiltModel, values: Mapping[str, pl.LazyFrame]) -> np.ndarray:
    """A value per column from *values*, ``(dims…, value)`` frames per variable, laid onto this build by coordinate.

    NaN where *values* gives none: a variable it does not name, a coordinate
    it lacks, a declaration whose dims changed.

    Raises:
        SpecsolveError: No value lands at a coordinate this build holds.
    """
    import numpy as np

    out = np.full(model.handoff.column_count, np.nan)
    _refuse_nothing_placed(
        _place(out, model, model.variables, model.program.variables, values, pl.col(VALUE).cast(pl.Float64))
    )
    return out


def _refuse_nothing_placed(placed: int) -> None:
    """Refuse a start that lands nowhere, which would start nothing yet read as a start taken."""
    if not placed:
        raise SpecsolveError(
            'start= gives no value at any coordinate this model holds, so it would start nothing. Name the '
            'declarations as the spec declares them, and their coordinates as the readers return them.'
        )


def _place(
    into: np.ndarray,
    model: BuiltModel,
    held: Mapping[str, labels.Labelled],
    declared: Mapping[str, program.Variable] | Mapping[str, program.Constraint],
    previous: Mapping[str, pl.LazyFrame],
    value: pl.Expr,
) -> int:
    """*value* off each frame of *previous*, written into *into* at the label of the same coordinate in *held*; how many landed.

    The join casts every dim to a string on both sides, so a table typed
    otherwise than the read-back still matches: ``3`` and ``'3'`` are one
    label. A frame whose columns are not its declaration's dims and ``value``
    is skipped, as is a coordinate only one side holds.
    """
    import numpy as np

    positions = pl.Series(VALUE, np.arange(len(into), dtype=np.int64))
    placed = 0
    for name, labelled in held.items():
        dims = list(declared[name].dims)
        before = previous.get(name)
        if before is None or set(before.collect_schema().names()) != {*dims, VALUE}:
            continue
        here = laid_out(model.attached, labelled, tuple(dims), positions).select(
            *(pl.col(dim).cast(pl.String) for dim in dims), pl.col(VALUE).alias(_LABEL_ORDER)
        )
        given = before.select(*(pl.col(dim).cast(pl.String) for dim in dims), value)
        found = (here.join(given, on=dims, how='inner') if dims else here.join(given, how='cross')).pipe(collected)
        into[found[_LABEL_ORDER].to_numpy()] = found[VALUE].to_numpy()
        placed += found.height
    return placed


def _counted(columns: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """*columns* and *rows* with exactly one basic entry per row, the count every solver's simplex needs.

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


def parameter_dual(compiler: Compiler, name: str) -> pl.DataFrame:
    """Parameter *name*'s rate at the solve *compiler* reads — ``(dims…, value)``, one row per row of its table.

    The Lagrangian's derivative in each row, every other leaf held at the
    solve: the objective's rate, less each constraint row's rate times its
    dual, plus a column's reduced cost where *name* sets the bound that binds.
    Masks are not differentiated, so the model is held as built. Rows come
    in label order; a row nothing reads has zero.

    Raises:
        KeyError: No parameter is called *name*.
        SpecsolveError: *name* stands where its rate is not one coefficient
            per row.
    """
    declared = compiler.scope.program.parameters
    if name not in declared:
        raise KeyError(unknown_name_message('parameter', name, declared))
    solution = compiler.solution
    assert solution is not None and solution.dual is not None, (
        'a result refuses a parameter dual its solve left no duals for'
    )
    _refuse_a_nonlinear_read(compiler.scope.program, name)

    dims = declared[name].dims
    rows = compiler.scope.data.parameters[name].with_columns(varied_label(compiler.scope, dims).alias(_VARIED))
    table = rows.select(*dims, _VARIED)
    rates = _lagrangian_rates(
        replace(compiler, varied=name), name, rows.select(_VARIED, pl.col(VALUE).cast(pl.Float64))
    )
    rates += _bound_rates(compiler, name, table)
    summed = pl.concat(rates).group_by(_VARIED).agg(pl.col(VALUE).sum()) if rates else _NO_RATES
    out = (
        table.join(summed, on=_VARIED, how='left')
        .sort(_VARIED)
        .select(*dims, pl.col(VALUE).fill_null(0.0))
        .pipe(collected)
    )
    return _as_strings(out, compiler.scope.data, dims)


#: The varied leaf's label while a parameter dual adds up its rates.
_VARIED = '__varied label__'

_NO_RATES = pl.LazyFrame(schema={_VARIED: pl.Int64, VALUE: pl.Float64})


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


def _bound_rates(compiler: Compiler, name: str, table: pl.LazyFrame) -> list[pl.LazyFrame]:
    """``(label, value)``: each column's reduced cost, owed to *name*'s row where *name* is the bound that binds.

    The reduced cost is the rate in the binding bound
    ([`Result.reduced_cost`][specsolve._result.Result.reduced_cost]), and
    its sign says which bound binds: under ``minimize`` a positive one holds a
    column up at its lower bound and a negative one down at its upper, and
    ``maximize`` flips both. It is the Lagrangian's rate in the column,
    computed by varying the variable, which leaves no handoff to hold.
    """
    model = compiler.scope.program
    solution = compiler.solution
    assert solution is not None, 'a parameter dual reads a solve'
    toward = -1.0 if model.objective is not None and model.objective.sense == 'maximize' else 1.0
    primal = pl.LazyFrame(
        {_VARIED: pl.int_range(solution.primal.len(), dtype=pl.Int64, eager=True), VALUE: solution.primal}
    )
    dims = model.parameters[name].dims
    parts = []
    for variable, v in model.variables.items():
        sides = {
            side
            for side, bound in (('lower', v.lower), ('upper', v.upper))
            if isinstance(bound, program.ParameterReference) and bound.name == name
        }
        if not sides:
            continue
        rates = _lagrangian_rates(replace(compiler, varied=variable), variable, primal)
        reduced = pl.concat(rates).group_by(_VARIED).agg(pl.col(VALUE).sum()) if rates else _NO_RATES
        held = compiler.scope.variables[variable]
        keys = v.dims or (UNIT,)
        columns = held.frame.select(*keys, pl.col('var_label').cast(pl.Int64).alias('__column__'))
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
