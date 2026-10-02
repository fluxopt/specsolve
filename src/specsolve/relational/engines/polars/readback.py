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
from specsolve.relational.sinks.handoff import Declared, Run, SetRun
from specsolve.relational.sinks.writers.base import Names

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from specsolve.relational.engines.polars.assembly import BuiltModel
    from specsolve.relational.engines.polars.attaching import AttachedSources
    from specsolve.relational.engines.polars.compiler import PolarsCompiler

#: Scratch columns. The spaces make them unrepresentable as declared names.
SOLUTION = '__solution value__'
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


#: What a label keeps in a file name. LP reads nothing else back, and
#: ``(``, ``)`` and ``,`` are the name's own.
_UNSAFE_IN_A_NAME = r"""[^\w!"#$%&'.;?@`{|}~]"""


def file_names(model: BuiltModel) -> Names:
    """Every column and row named by its declaration at its coordinate, ``p(2030,wind)``.

    A label keeps the characters both LP and MPS readers take back and writes
    each other one as ``_``, so ``2030-01-01`` becomes ``2030_01_01``. A
    declaration with no dims is ``p()``: a bare name can be a keyword, and
    ``end`` or ``st`` ends a section.

    Raises:
        SpecsolveError: Two coordinates of one declaration write as one name,
            which a reader would take for one column or refuse as a duplicate row.
    """
    return Names(
        columns=_declared_names(model.variables, model.program.variables, 'variable'),
        rows=_declared_names(model.constraints, model.program.constraints, 'constraint'),
    )


def _declared_names(
    owned: Mapping[str, labels.Labelled],
    declarations: Mapping[str, program.VariableDeclaration | program.ConstraintDeclaration],
    kind: str,
) -> pl.Series:
    """One kind's names in solver order: each declaration's label frame, spelled and concatenated.

    The build keeps its declarations in the order it numbered them, and each
    frame arrives in label order, so the concatenation is in solver order.
    """
    names = []
    for name, held in owned.items():
        dims = declarations[name].dims
        cleaned = [pl.col(d).cast(pl.String).str.replace_all(_UNSAFE_IN_A_NAME, '_') for d in dims]
        if cleaned:
            spelled = pl.concat_str(pl.lit(f'{name}('), pl.concat_str(cleaned, separator=','), pl.lit(')'))
            frame = held.frame.select(*dims, spelled.alias('#name')).collect(engine=polars_engine())
        else:
            frame = pl.DataFrame({'#name': [f'{name}()'] * held.height}, schema={'#name': pl.String})
        _refuse_a_shared_name(frame, name, dims, kind)
        names.append(frame.get_column('#name'))
    return pl.concat(names) if names else pl.Series('#name', [], dtype=pl.String)


def _refuse_a_shared_name(frame: pl.DataFrame, name: str, dims: tuple[str, ...], kind: str) -> None:
    """Refuse a declaration two of whose coordinates clean to one name, naming both."""
    shared = frame.filter(pl.col('#name').is_duplicated())
    if not shared.height:
        return
    first = shared.filter(pl.col('#name') == shared.item(0, '#name')).head(2)
    a, b = (tuple(first.row(i, named=True)[d] for d in dims) for i in range(2))
    raise SpecsolveError(
        f"{kind} '{name}' writes the coordinates {a} and {b} as one name, {shared.item(0, '#name')!r}. "
        'A name keeps letters, digits and !"#$%&\'.;?@`{|}~ from a label and writes any other character '
        'as _. Relabel one of the two, or write the file without names.'
    )


def declared(model: BuiltModel) -> Declared:
    """Which declaration, at which coordinate, owns each column, row and set.

    Each declaration's frame arrives in label order, and the build keeps them
    in the order it took them, which is the order of their runs.
    """
    variables = [_run(name, owned, model.program.variables[name].dims) for name, owned in model.variables.items()]
    constraints = [_run(name, owned, model.program.constraints[name].dims) for name, owned in model.constraints.items()]
    sets = [
        SetRun(name, s.variable, model.program.variables[s.variable].dims.index(s.along), s.sos_type)
        for name, s in model.program.sos.items()
    ]
    return Declared(variables, constraints, sets)


def _run(name: str, owned: labels.Labelled, dims: tuple[str, ...]) -> Run:
    """One declaration's run, its coordinates the label frame's dim columns."""
    return Run(name, owned.start, owned.height, dims, owned.frame.select(dims).collect(engine=polars_engine()))


def laid_out(
    attached: AttachedSources, held: labels.Labelled, dims: tuple[str, ...], values: pl.Series
) -> pl.LazyFrame:
    """One declaration's coordinates in label order, beside its share of *values*.

    The share is a column, not a concatenated frame, so a mismatched length
    raises instead of padding with nulls. Dim columns leave as ``String``,
    because a caller joins them against their own data and polars refuses
    ``Enum`` against ``String``.
    """
    labelled = held.frame.select(*dims).with_columns(held.share(values))
    return labelled.with_columns(pl.col(d).cast(pl.String) for d in string_dims(attached, dims))


def string_dims(attached: AttachedSources, dims: Sequence[str]) -> list[str]:
    """Those of *dims* attaching encoded as ``Enum``."""
    return [d for d in dims if attached.is_enum_encoded(d)]


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
    return pl.concat(pieces) if pieces else pl.Series(SOLUTION, [], dtype=pl.Float64)


def _aligned(
    attached: AttachedSources, name: str, held: labels.Labelled, dims: tuple[str, ...], stored: pl.DataFrame | None
) -> pl.Series:
    """One declaration's saved values in its label order — its slice of the vector.

    A declaration the rebuild masks away entirely has an empty slice, so a
    missing *stored* is no error there.
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


def expression_frame(name: str, expr: program.Expression, compiler: PolarsCompiler) -> pl.DataFrame:
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
        reported_divisor_message,
    )

    fragments = compiled.consts
    dims = compiler.scope.spanned(fragments)
    carrier = labels.frame(compiler.scope, dims, None, _EXPRESSION_ROW, 0, absence_restrictions(fragments)).lazy()
    added = compiler.added(fragments, carrier, absent='zero')
    out = added.select(_EXPRESSION_ROW, *dims, pl.col('cval').alias('value')).collect(engine=polars_engine())
    ordered = labels.in_position_order(out, _EXPRESSION_ROW).drop(_EXPRESSION_ROW)
    return ordered.with_columns(pl.col(d).cast(pl.String) for d in string_dims(compiler.scope.data, dims))
