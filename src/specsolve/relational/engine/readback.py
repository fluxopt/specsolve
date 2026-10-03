"""Reading a built model back: one constraint row, a solve's frames, a named expression."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from mathspec import program

from specsolve.errors import SpecsolveError, unknown_name_message
from specsolve.relational.collect import collect_engine
from specsolve.relational.engine import coverage, labels
from specsolve.relational.engine.pieces import absence_restrictions
from specsolve.relational.result import ConstraintRow

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from specsolve.relational.engine.assembly import BuiltModel
    from specsolve.relational.engine.attaching import AttachedSources
    from specsolve.relational.engine.compiler import Compiler

#: Scratch columns. The spaces make them unrepresentable as declared names.
_SOLUTION = '__solution value__'
_EXPRESSION_ROW = '__expression row__'
_LABEL_ORDER = '__label order__'


def row(model: BuiltModel, name: str, coordinate: Mapping[str, object]) -> ConstraintRow:
    """One built constraint row, spelled back out. See [`row`][specsolve.api.Model.row].

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
    found = frame.filter(predicates).collect() if predicates else frame.collect()
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
        rendered = pl.concat_str([pl.col(d).cast(pl.String) for d in dims], separator=', ') if dims else pl.lit('')
        named.append(
            picked.select(
                pl.col('var_label').alias('col'),
                pl.lit(variable).alias('variable'),
                rendered.alias('coordinate'),
            ).collect()
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

    The share is a column, not a concatenated frame, so a mismatched length
    raises instead of padding with nulls. Dim columns leave as ``String``
    ([`_as_strings`][]).
    """
    return _as_strings(held.frame.select(*dims).with_columns(held.share(values)), attached, dims)


def _as_strings[F: (pl.DataFrame, pl.LazyFrame)](frame: F, attached: AttachedSources, dims: Sequence[str]) -> F:
    """*frame* with those of *dims* that attaching encoded as ``Enum`` cast back to ``String``.

    A caller joins a dim column against its own data, and polars refuses
    ``Enum`` against ``String``.
    """
    return frame.with_columns(pl.col(d).cast(pl.String) for d in dims if attached.is_enum_encoded(d))


def reordered(
    attached: AttachedSources,
    registry: Mapping[str, labels.Labelled],
    declared: Mapping[str, program.VariableDeclaration | program.ConstraintDeclaration],
    frames: Mapping[str, pl.DataFrame],
) -> pl.Series:
    """A saved solution's value frames back as the positional vector — the inverse of [`laid_out`][].

    A rebuild over the same spec and sources numbers the labels identically
    (docs/about/architecture.md, "The relational lane").

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
        return stored['value'].rename(_SOLUTION)
    order = _as_strings(held.frame.select(*dims).collect(), attached, dims).with_row_index(_LABEL_ORDER)
    joined = order.join(stored, on=list(dims), how='left').sort(_LABEL_ORDER)
    if joined['value'].null_count():
        raise SpecsolveError(
            f"the saved answer's '{name}' frame does not cover every coordinate this model builds, so it "
            f'is not an answer to this model. Re-solve rather than read.'
        )
    return joined['value'].rename(_SOLUTION)


def readers(
    compiler: Compiler,
    named: Mapping[str, program.ExpressionDeclaration],
    lower: Callable[[str | Mapping[str, object]], program.Expression] | None,
) -> tuple[dict[str, Callable[[], pl.DataFrame]], Callable[[str | Mapping[str, object]], pl.DataFrame] | None]:
    """The reads [`evaluate`][specsolve.relational.result.Result.evaluate] is built from, over one compiler.

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
    absent has no row, or a zero under ``absence: zero``. A quotient has no row
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
    out = added.select(_EXPRESSION_ROW, *dims, pl.col('cval').alias('value')).collect(engine=collect_engine())
    ordered = labels.in_position_order(out, _EXPRESSION_ROW).drop(_EXPRESSION_ROW)
    return _as_strings(ordered, compiler.scope.data, dims)


def _reported_divisor_message(name: str, missing: int) -> str:
    """The message for a divisor parameter a reported expression reads short of a row."""
    return (
        f"parameter '{name}' is used as a divisor but has no row at {missing} of the coordinates "
        f'the expression divides at. A missing parameter row is not absence, so the quotient '
        f'is not dropped there, and there is no number to divide by.\n'
        f'  Supply the missing rows.\n'
        f'  Give the value 0 at a coordinate the quotient should skip: a quotient by zero has no value.'
    )
