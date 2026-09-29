"""Reading a built model back: one constraint row, a solve's frames, a named expression."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from mathspec import program

from specsolve.errors import SpecsolveError, reported_divisor_message, unknown_name_message
from specsolve.relational.collect import polars_engine
from specsolve.relational.engines.polars import coverage, labels
from specsolve.relational.engines.polars.fragments import absence_restrictions
from specsolve.relational.result import ConstraintRow

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from specsolve.relational.engines.polars.assembly import BuiltModel
    from specsolve.relational.engines.polars.attaching import AttachedSources
    from specsolve.relational.engines.polars.compiler import PolarsCompiler

#: Scratch columns of the expression reader. The spaces make them
#: unrepresentable as declared names.
SOLUTION = '__solution value__'
_EXPRESSION_ROW = '__expression row__'
_LABEL_ORDER = '__label order__'


def row(model: BuiltModel, name: str, coordinate: Mapping[str, object]) -> ConstraintRow:
    """One built constraint row, spelled back out. See [`row`][specsolve.api.Model.row].

    Three positional takes against frames the build already keeps, and no scan
    of the matrix: the constraint's own coordinate frame carries the global row
    index, ``row_starts`` says where that row's entries lie, and each
    variable's frame carries the global column index its terms point at.

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
    """The global row index constraint *name* built at *coordinate*, and that coordinate in dim order.

    The coordinate has to name **every** dim of the declaration: a partial
    one matches a set of rows.

    Raises:
        SpecsolveError: The coordinate names dims the declaration does not,
            holds a label the dimension cannot, or matches no row the build
            produced — a row masked out by ``where`` or dropped for having no
            terms.
    """
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

    The cast **is** the check: a string against an integer dim and a stranger
    against an ``Enum`` are one failure.
    """
    try:
        return pl.lit(pl.Series([value], dtype=dtype).item(0), dtype=dtype)
    except (pl.exceptions.PolarsError, TypeError, OverflowError) as refused:
        raise SpecsolveError(
            f"constraint '{name}' is declared over '{dim}', which holds {dtype}, and {value!r} is "
            f'not one of its labels. Read the row at a label the dimension has.'
        ) from refused


def _named_terms(model: BuiltModel, entries: pl.DataFrame) -> pl.DataFrame:
    """``(variable, coordinate, coefficient)`` for one row's matrix entries.

    Each declaration owns a contiguous, dense run of column indices, so which
    variable a term belongs to is a range test, and a term's place in its own
    declaration's frame is its label minus that block's start — a positional
    take, not a search.

    ``coordinate`` is rendered rather than spread across dim columns because
    one row's terms may come from variables with *different* dims. It carries
    the labels alone, in the declaration's dim order — linopy's ``p[1, wind]``
    bracket. The terms leave in the order the entries arrived in, which is the
    solver's own column order.
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

    The order was never lost: [`labels.frame`][] hands back a
    label-ascending frame, and the solver's vector is positional in the same
    index. The share is attached as a column rather than concatenated as a
    frame, so a mismatched length raises instead of padding with nulls.

    **Dim columns leave in ``String``**, where the build holds them as
    ``pl.Enum``: a returned frame is something a caller joins against their
    own data, and polars refuses ``Enum`` against ``String``.
    """
    labelled = held.frame.select(*dims).with_columns(held.share(values))
    return labelled.with_columns(pl.col(d).cast(pl.String) for d in string_dims(attached, dims))


def string_dims(attached: AttachedSources, dims: Sequence[str]) -> list[str]:
    """Those of *dims* attaching encoded as ``Enum`` — its string ones."""
    return [d for d in dims if attached.is_enum_encoded(d)]


def reordered(
    attached: AttachedSources,
    registry: Mapping[str, labels.Labelled],
    declared: Mapping[str, program.VariableDeclaration | program.ConstraintDeclaration],
    frames: Mapping[str, pl.DataFrame],
) -> pl.Series:
    """A saved solution's value frames back as the positional vector — the inverse of [`laid_out`][].

    Each declaration's ``(dims…, value)`` is aligned to its
    [`Labelled`][labels.Labelled] frame's label order and the values concatenated in
    ``start`` order, rebuilding the vector a solver returned. A rebuild of the
    model over the same spec and sources numbers the labels identically
    (docs/about/architecture.md, "The relational lane").

    Args:
        attached: The rebuilt model's sources, for which dims it enum-encoded.
        registry: The rebuilt model's ``variables`` or ``constraints``.
        declared: The program's ``variables`` or ``constraints``, for the dims.
        frames: The saved ``(dims…, value)`` frame per name, as read back.

    Raises:
        SpecsolveError: A declaration whose saved frame misses a coordinate the
            rebuilt model holds — the frame is not this model's answer.
    """
    in_start_order = sorted((held.start, name) for name, held in registry.items())
    pieces = [
        _aligned(attached, name, registry[name], declared[name].dims, frames.get(name)) for _, name in in_start_order
    ]
    return pl.concat(pieces) if pieces else pl.Series(SOLUTION, [], dtype=pl.Float64)


def _aligned(
    attached: AttachedSources, name: str, held: labels.Labelled, dims: tuple[str, ...], stored: pl.DataFrame | None
) -> pl.Series:
    """One declaration's saved values in its label order — its slice of the vector.

    The values are joined onto the rebuilt label frame on the dims, the string
    ones cast as [`laid_out`][] casts them. A declaration the rebuild masks
    away entirely holds no label, so its slice is empty and a missing *stored*
    is no error; a missing one the rebuild does build raises.
    """
    if held.height == 0:
        return pl.Series(SOLUTION, [], dtype=pl.Float64)
    if stored is None:
        raise SpecsolveError(
            f"the saved answer holds no '{name}' frame, but this model builds it, so it is not this model's "
            f'answer. Re-solve rather than read.'
        )
    if not dims:
        return stored['value'].rename(SOLUTION)
    order = (
        held.frame.select(*dims)
        .collect()
        .with_columns(pl.col(d).cast(pl.String) for d in string_dims(attached, dims))
        .with_row_index(_LABEL_ORDER)
    )
    joined = order.join(stored, on=list(dims), how='left').sort(_LABEL_ORDER)
    if joined['value'].null_count():
        raise SpecsolveError(
            f"the saved answer's '{name}' frame does not cover every coordinate this model builds, so it "
            f'is not an answer to this model. Re-solve rather than read.'
        )
    return joined['value'].rename(SOLUTION)


def readers(
    compiler: PolarsCompiler,
    named: Mapping[str, program.ExpressionDeclaration],
    lower: Callable[[str | Mapping[str, object]], program.Expression] | None,
) -> tuple[dict[str, Callable[[], pl.DataFrame]], Callable[[str | Mapping[str, object]], pl.DataFrame] | None]:
    """The reads [`evaluate`][specsolve.relational.result.Result.evaluate] is built from, over one compiler.

    Every producer of them — a live solve, a rebuilt archive, and the
    variable-free arithmetic path — comes through here, so the read a declared
    name gets and the read an ad-hoc expression gets are defined once and
    differ only in the compiler. Nothing is compiled until a reader is called.

    Args:
        compiler: The compiler each read compiles through — carrying a solution,
            or none for pure arithmetic.
        named: The declared named expressions, by name.
        lower: How an expression written the way ``expressions:`` writes one
            becomes a plan node in the spec's namespace, or ``None`` where there
            is no spec as written to lower against — then ad-hoc evaluation is
            unavailable and the second element is ``None``.

    Returns:
        One deferred reader per declared name, and the ad-hoc evaluator (or
        ``None``).
    """

    def reader(name: str, expression: program.Expression) -> Callable[[], pl.DataFrame]:
        return lambda: expression_frame(name, expression, compiler)

    declared = {name: reader(name, e.expression) for name, e in named.items()}
    if lower is None:
        return declared, None

    def evaluate(written: str | Mapping[str, object]) -> pl.DataFrame:
        return expression_frame('the expression', lower(written), compiler)

    return declared, evaluate


def expression_frame(name: str, expr: program.Expression, compiler: PolarsCompiler) -> pl.DataFrame:
    """Named expression *expr* evaluated at the solve *compiler* holds — ``(dims…, value)``.

    Every leaf is a number after a solve: the compiler reads a variable as its
    primal and ``dual(c)`` as the constraint's row duals, so the expression is
    const fragments at whatever degree the file wrote it — a product of two
    variables, a variable under a power, a division by one — each added up
    per coordinate over the expression's coordinate product the way a
    constraint's right-hand side is.

    The frame answers the way a constraint over the same expression would: a
    coordinate a parameter does not cover contributes zero, a coordinate where
    a variable is absent has no row — or holds a zero, under ``absence:
    zero`` — and a variable-free expression is one row of ``value``. A
    quotient has no row where its divisor is absent or zero. Dims come back in declaration order and rows in label order
    over those dims.

    Raises:
        DataError: A divisor parameter with no row where the expression
            divides — checked before any sum can read the null as zero.
        SpecsolveError: The expression reads a dual and the solve left none.
    """
    context = f"named expression '{name}'"
    compiled = compiler.expression(expr, context, reported=True)

    coverage.refuse_null_constants(
        [p.frame for p in compiled.consts],
        program.parameters_of(*coverage.divisors_of(expr)),
        context,
        reported_divisor_message,
    )

    fragments = compiled.consts
    dims = compiler.scope.spanned(fragments)
    carrier = labels.frame(compiler.scope, dims, None, _EXPRESSION_ROW, 0, absence_restrictions(fragments)).lazy()
    added = compiler.added(fragments, carrier, absent='zero')
    out = added.select(_EXPRESSION_ROW, *dims, pl.col('cval').alias('value')).collect(engine=polars_engine())
    ordered = labels.in_position_order(out, _EXPRESSION_ROW).drop(_EXPRESSION_ROW)
    return ordered.with_columns(pl.col(d).cast(pl.String) for d in string_dims(compiler.scope.data, dims))
