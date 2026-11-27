"""One build: every entry into rows of the model tables a sink drains.

[`build`][] is the whole of it. The variable and constraint index tables fill
during a build, because an entry built later reads what earlier ones produced.
Each entry plans its queries alone, and a shared pass runs one kind of query
for every entry at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ParamSpec, TypeVar, get_args

import numpy as np
import polars as pl
from mathspec import program

from specsolve import _sinks as sinks
from specsolve._answer.result import Diagnostics
from specsolve._collect import collected, collected_all, in_memory
from specsolve._engine import coverage, indices
from specsolve._engine.queries import QueryBuilder
from specsolve._engine.scope import Scope, join_on
from specsolve._engine.specialize import is_empty, specialized
from specsolve._engine.terms import Term, presences_of
from specsolve._names import COEFF, COL, COL_2, COL_L, COL_R, CONSTANT, ROW
from specsolve._sinks.tables import SENSE

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mathspec.program import ObjectiveSense
    from polars._typing import MaintainOrderJoin, PolarsDataType

    from specsolve._attached import AttachedSources
    from specsolve._clock import Lap, Step


#: The most columns or rows a model may reach for its build to collect in memory.
IN_MEMORY_REACH = 250_000

_Built = TypeVar('_Built')
_Args = ParamSpec('_Args')

#: One constraint's ``rows``, its share of the matrix, and its quadratic share.
_ConstraintShare = tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None]

#: One variable with a column, as [`Assembly._bounded`][] reads it: its name, its domain and the query of its bounds.
_Bounds = tuple[str, program.VariableDomain, pl.LazyFrame]


@dataclass(frozen=True)
class _Planned:
    """One constraint between its compile and the shared passes that read it."""

    name: str
    c: program.Constraint
    linear: list[tuple[Term, float]]
    quadratic: list[tuple[Term, float]]
    constant: list[tuple[Term, float]]
    #: Whether a variable's presence narrows its rows, so the rows it removed are counted as not built.
    narrowed: bool
    #: The query of its rows, numbered from 0, and what their table becomes.
    query: pl.LazyFrame
    finish: Callable[[pl.DataFrame], pl.DataFrame]
    #: How many rows its ``where`` admits before a presence narrows them; ``None`` without both.
    admitted: pl.LazyFrame | None


#: The tables a sink reads, as schemas.
_COLS = ('lb', 'ub', 'vtype')
_OBJ = (COL, COEFF)
_QUAD = (COL_L, COL_R, COEFF)
_ROWS = (ROW, 'sense', 'rhs')
_MATRIX = (ROW, COL, COEFF)
_QMATRIX = (ROW, COL_L, COL_R, COEFF)
_SOS = ('set', 'type', COL, 'weight')

#: The dtype of each of those columns. ``COL``, ``set`` and ``weight`` are the
#: solver's ``Int32``; a row index stays ``Int64``, a position in the full
#: coordinate product, which can pass 2^31.
_DTYPES = {
    COL: pl.Int32, ROW: pl.Int64,
    'lb': pl.Float64, 'ub': pl.Float64, 'rhs': pl.Float64, COEFF: pl.Float64,
    'sense': SENSE, 'vtype': pl.Enum(get_args(program.VariableDomain)),
    'set': pl.Int32, 'type': pl.UInt8, 'weight': pl.Int32,
    COL_L: pl.Int32, COL_R: pl.Int32,
}  # fmt: skip


@dataclass
class Measured:
    """What one build measured about itself, and a rebuild replaces wholesale.

    It outlives ``close()``, so it holds counts and small tables, never a read
    of the model. Most of it is written as measured, so a build that raises
    still reports what it got to.
    """

    #: ``name -> (coordinates, rows)`` for each parameter attached short of the
    #: coordinates its dimensions reach.
    sparse: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: ``name -> rows not built``, because every term they had vanished.
    omitted: dict[str, int] = field(default_factory=dict)
    #: ``name -> (smallest, largest)`` coefficient magnitude, per constraint.
    coefficients: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: ``name -> (smallest, largest)`` bound magnitude, per variable.
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: ``name -> (smallest, largest)`` right-hand-side magnitude, per
    #: constraint, over the rows that survived.
    rhs: dict[str, tuple[float, float]] = field(default_factory=dict)
    objective_range: tuple[float, float] | None = None
    columns: int = 0
    rows: int = 0
    nonzeros: int = 0

    def diagnostics(self) -> Diagnostics:
        """What the build did that the answer does not show, as the tables a caller reads."""
        return Diagnostics(
            omissions=_per_name('constraint', self.omitted, rows_not_built=pl.UInt32),
            coefficient_range=_per_name('constraint', self.coefficients, smallest=pl.Float64, largest=pl.Float64),
            bound_range=_per_name('variable', self.bounds, smallest=pl.Float64, largest=pl.Float64),
            rhs_range=_per_name('constraint', self.rhs, smallest=pl.Float64, largest=pl.Float64),
            sparse_parameters=_per_name(
                'parameter',
                {name: (reach, rows, reach - rows) for name, (reach, rows) in self.sparse.items()},
                coordinates=pl.UInt64,
                rows=pl.UInt64,
                missing=pl.UInt64,
            ),
            objective_range=self.objective_range,
        )


def _per_name(kind: str, measured: Mapping[str, object], **columns: PolarsDataType) -> pl.DataFrame:
    """One [`Diagnostics`][specsolve._answer.result.Diagnostics] table: a row per name in *measured*, in build order.

    *kind* names the first column, and the remaining *columns* carry each
    value in order — a scalar for one column, a tuple for several.
    """
    names = list(measured)
    values = [v if isinstance(v, tuple) else (v,) for v in measured.values()]
    schema = {kind: pl.String, **columns}
    return pl.DataFrame(
        {kind: names, **{column: [v[i] for v in values] for i, column in enumerate(columns)}}, schema=schema
    )


@dataclass(frozen=True)
class BuiltModel:
    """One build's product: the tables a sink drains, and what reads it back."""

    program: program.Program
    attached: AttachedSources
    #: One [`Indexed`][specsolve._engine.indices.Indexed] per
    #: entry, one map per index space: a variable and a constraint may share a name.
    variables: dict[str, indices.Indexed]
    constraints: dict[str, indices.Indexed]
    tables: sinks.SinkTables


def build(program: program.Program, attached: AttachedSources, measured: Measured, lap: Lap) -> BuiltModel:
    """Build every entry of *program* over *attached* into the model tables.

    The program a build reads is [`specialized`][specsolve._engine.specialize.specialized]
    to what the data settled, and the model it returns holds that one. A build
    that raises returns no model rather than half of one.

    Args:
        program: The compiled program.
        attached: Its data, as [`attach`][specsolve._engine.attaching.attach] returns it.
        measured: Where the build writes what it measures about itself, as it
            measures it, so a build that raises still reports what it got to.
        lap: Ends a lap of ``build`` at each step, one per entry.
    """
    measured.sparse = short_parameters(program, attached)
    return Assembly(specialized(program, attached), attached, measured, lap).run()


class Assembly:
    """One build in progress: the mutable half, discarded once [`run`][] has frozen it."""

    def __init__(
        self,
        program: program.Program,
        attached: AttachedSources,
        measured: Measured,
        lap: Lap,
    ) -> None:
        """Start a build of *program*, [`specialized`][specsolve._engine.specialize.specialized] to *attached*."""
        self.program = program
        self.lap = lap
        self.attached = attached
        self.measured = measured
        self.variables: dict[str, indices.Indexed] = {}
        self.constraints: dict[str, indices.Indexed] = {}
        self.scope = Scope(program, attached, self.variables)
        self.builder = QueryBuilder(self.scope, builds=True)
        self.n_cols = 0
        self.n_rows = 0
        self.small = False
        #: How many special-ordered sets have been numbered, contiguous ``0..n-1`` across the model.
        self.n_sets = 0
        self.quad: pl.DataFrame | None = None
        self.obj_const = 0.0
        self.obj_sense: ObjectiveSense | None = None
        self._empty_tables = {}

    def run(self) -> BuiltModel:
        """Build every entry, then freeze what they produced.

        Each entry plans its queries, and a shared pass runs every entry's
        queries of one kind together: the variables' ``bounds``, then the
        constraints' ``indices`` and their ``terms``. A constraint numbers its
        rows from 0 while it builds, and [`_placed`][] moves each onto the
        model's rows once every constraint knows how many it kept.

        Quadratic constraints build last, so their rows are a contiguous tail
        a sink takes as a slice; the sort is stable, so file order survives in
        each half. The matrix and ``rows`` leave in ``(row, col)`` order, as
        ``SinkTables`` promises its sinks. A build whose [`_reach`][] is at most
        [`IN_MEMORY_REACH`][] collects [`in_memory`][].
        """
        self.small = self._reach() <= IN_MEMORY_REACH
        with in_memory(self.small):
            self.lap('build', 'setup', None)
            cols = self._variables()
            sets = [
                self._timed('set', name, self._build_sos, s, self.program.variables[s.variable])
                for name, s in self.program.sos.items()
            ]
            ordered = sorted(self.program.constraints.items(), key=lambda item: declares_quadratic(item[1]))
            shares = self._constraints(ordered)
            o = self.program.objective
            objective = None if o is None else self._timed('objective', None, self._build_objective, o)

        built = self._placed([name for name, _ in ordered], shares)
        stacked = indices.in_position_order(_stack([m for _, m, _ in built if m is not None], _MATRIX), ROW)
        matrix_starts = _row_starts(stacked, self.n_rows)
        matrix = stacked.select(COL, COEFF).rechunk()
        self.lap('build', 'matrix', None)
        self.measured.columns = self.n_cols
        self.measured.rows = self.n_rows
        self.measured.nonzeros = matrix.height
        tables = sinks.SinkTables(
            cols=_stack(cols, _COLS),
            obj=_stack([objective] if objective is not None else [], _OBJ),
            quad=_stack([] if self.quad is None else [self.quad], _QUAD),
            qmatrix=indices.in_position_order(_stack([q for _, _, q in built if q is not None], _QMATRIX), ROW),
            rows=indices.in_position_order(_stack([r for r, _, _ in built], _ROWS), ROW),
            matrix=matrix,
            sos=_stack(sets, _SOS),
            row_starts=matrix_starts,
            column_count=self.n_cols,
            row_count=self.n_rows,
            objective_sense=self.obj_sense,
            objective_constant=self.obj_const,
        )
        model = BuiltModel(self.program, self.attached, self.variables, self.constraints, tables)
        self.lap('build', 'tables', None)
        return model

    def _timed(
        self, step: Step, entry: str | None, build: Callable[_Args, _Built], *args: _Args.args, **kwargs: _Args.kwargs
    ) -> _Built:
        """*build* on *args*, then a lap of the build as *step* of *entry*, even if it raises.

        What an entry does alone is its own, so its seconds do not move with
        the order entries build in; the queries it plans for a shared pass are
        that pass's.
        """
        try:
            return build(*args, **kwargs)
        finally:
            self.lap('build', step, entry)

    def _variables(self) -> list[pl.DataFrame]:
        """Every variable indexed in its own row, then their bounds in the ``bounds`` pass: each one's share of ``cols``.

        What an entry planned is let go when this returns, so no term of it
        outlives the pass that read it.
        """
        bounds = [
            self._timed('variable', name, self._index_variable, name, v) for name, v in self.program.variables.items()
        ]
        cols = self._bounded([b for b in bounds if b is not None])
        self.lap('build', 'bounds', None)
        return cols

    def _constraints(self, ordered: list[tuple[str, program.Constraint]]) -> dict[str, _ConstraintShare]:
        """Every constraint compiled in its own row, then the ``indices`` and ``terms`` passes: each one's shares by name.

        As in [`_variables`][], a constraint's compiled terms are let go when
        this returns, before the stacking holds every share twice.
        """
        planned = [self._timed('constraint', name, self._planned_constraint, name, c) for name, c in ordered]
        live = self._indexed([c for c in planned if c is not None])
        self.lap('build', 'indices', None)
        shares = self._terms(live)
        self.lap('build', 'terms', None)
        return shares

    def _batches[T](self, entries: list[T]) -> list[list[T]]:
        """*entries* in the batches a shared pass reads: one batch within [`IN_MEMORY_REACH`][], one entry per batch above it.

        A batch's queries are collected together, which pays polars' fixed
        cost of a query once rather than per query; in a port model that was
        about 15% of a build (#2022). Above the reach each query is large, and a
        batch is read, checked and let go before the next is read, so a pass
        holds no more at once than one entry built alone does.
        """
        if self.small:
            return [entries]
        return [[entry] for entry in entries]

    def _together(self, queries: list[pl.LazyFrame]) -> list[pl.DataFrame]:
        """Every query of one batch: in one pass within [`IN_MEMORY_REACH`][], one after another above it.

        Above the reach one pass would run an entry's queries side by side, each
        holding its working set at once.
        """
        if self.small:
            return collected_all(queries)
        return [collected(query) for query in queries]

    def _reach(self) -> int:
        """The columns or the rows the model would have with no ``where``, whichever is more: known before the build.

        A constraint over dimensions no variable spans has more rows than there are
        columns, so the rows count too.
        """
        sizes = {d: held.size for d, held in self.attached.dimensions.items()}
        return max(
            sum(math.prod(sizes[d] for d in declared.dims) for declared in entries.values())
            for entries in (self.program.variables, self.program.constraints)
        )

    def _matrix_share(
        self, tables: list[pl.LazyFrame], name: str, *expressions: program.Expression
    ) -> tuple[pl.LazyFrame, Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.Series]]]:
        """One constraint's share, as the query of its terms and what their table becomes: in ``(row, col)`` order, repeated cells summed.

        The finish returns the share, and the rows that had any term, read
        before the prune: a row whose every coefficient is zero is not a row
        with no terms.
        """

        def share(stacked: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
            coverage.refuse_null_coefficients(stacked, name, *expressions)
            collapsed, dropped = _collapsed(stacked, (ROW, COL), ordered=True)
            return collapsed, stacked.get_column(ROW).unique() if dropped else _ordered_rows(collapsed)

        return pl.concat(tables), share

    # ------------------------------------------------------------------
    # entries
    # ------------------------------------------------------------------

    def _index_variable(self, name: str, v: program.Variable) -> _Bounds | None:
        """One variable's indexed table, and the query of its bounds for [`_bounded`][]; ``None`` for one without a column."""
        start = self.n_cols
        if is_empty(v, self.attached):
            indexed = self._empty_table(v.dims, COL)
        else:
            indexed = indices.table(self.scope, v.dims, v.where, COL, start)
        self.n_cols = start + indexed.height
        self.variables[name] = indices.Indexed(indexed.lazy(), start, indexed.height)
        if not indexed.height:
            return None

        lb, ub = pl.col('lb').cast(pl.Float64), pl.col('ub').cast(pl.Float64)
        if v.domain != 'continuous':
            lb, ub = _whole(lb, lb.ceil()), _whole(ub, ub.floor())
        with in_memory():
            query = self.builder.bounds(indexed.lazy(), name, v).select(COL, lb.alias('lb'), ub.alias('ub'))
        return name, v.domain, query

    def _bounded(self, bounds: list[_Bounds]) -> list[pl.DataFrame]:
        """Every variable's share of ``cols``, its bounds read in [`_batches`][].

        A share leaves in index order: ``cols`` carries no ``COL``, so a row's
        position is its solver column index. The bounds are collected in memory:
        they are parameters looked up onto column indices already in memory, and polars
        2.0's streaming engine runs that ordered join 4.6 to 5.5 times slower
        than its in-memory one on the nodal benchmark at sizes m and l (#1857).
        """
        cols = []
        for batch in self._batches(bounds):
            with in_memory():
                read = self._together([query for _, _, query in batch])
            for (name, domain, _), table in zip(batch, read, strict=True):
                bounded = indices.in_position_order(table, COL)
                cols.append(bounded.select('lb', 'ub', pl.lit(domain, dtype=_DTYPES['vtype']).alias('vtype')))
                if (spread := _magnitude_range(bounded, 'lb', 'ub')) is not None:
                    self.measured.bounds[name] = spread
        return cols

    def _build_sos(self, s: program.Sos, v: program.Variable) -> pl.DataFrame:
        """One entry's sets as ``(set, type, col, weight)``, over *v*.

        A set and a weight are the two halves of a coordinate's row-major
        position, split at the ``along`` dimension; a variable with a condition
        renumbers the sets contiguously. The stream leaves grouped by set and
        ascending in weight, so a sink reads a set's edges off the neighbouring
        row.
        """
        held = self.variables[s.variable]
        sizes = {d: held.size for d, held in self.scope.data.dimensions.items()}
        stride = math.prod(sizes[d] for d in v.dims[v.dims.index(s.along) + 1 :])
        span = sizes[s.along] * stride

        if v.where is None:
            table = pl.select(pl.int_range(held.height, dtype=pl.Int64).alias('#position')).lazy()
            place, col = pl.col('#position'), (pl.col('#position') + held.start)
        else:
            table = held.table
            place, col = self.scope.row_major(v.dims, self.scope.position_of), pl.col(COL)
        placed = table.select(
            ((place // span) * stride + place % stride).alias('#set position'),
            ((place // stride) % sizes[s.along] + 1).cast(_DTYPES['weight']).alias('weight'),
            col.cast(_DTYPES[COL]).alias(COL),
        ).pipe(collected)

        position = pl.col('#set position')
        grouped = placed if placed.get_column('#set position').is_sorted() else placed.sort('#set position', 'weight')
        contiguous = position if v.where is None else (position != position.shift(1)).fill_null(True).cum_sum() - 1
        built = grouped.select(
            (contiguous + self.n_sets).cast(_DTYPES['set']).alias('set'),
            pl.lit(s.sos_type, dtype=_DTYPES['type']).alias('type'),
            COL,
            'weight',
        )
        if built.height:
            self.n_sets = built.item(-1, 'set') + 1
        return built

    def _planned_constraint(self, name: str, c: program.Constraint) -> _Planned | None:
        """One constraint's terms compiled, and the query of its rows for [`_indexed`][]; ``None`` for one without a row.

        Its rows are numbered from 0. One without a ``where`` is indexed only
        inside the presences that narrow it, so a sparse absence never meets the
        full product of its dimensions.
        """
        if is_empty(c, self.attached):
            self._without_a_row(name, self._empty_table(c.dims, ROW))
            return None
        admits = declares_quadratic(c)
        lhs = self.builder.expression(c.lhs, f"constraint '{name}' lhs", admits_quadratic=admits)
        rhs = self.builder.expression(c.rhs, f"constraint '{name}' rhs", admits_quadratic=admits)
        linear = [(p, 1.0) for p in lhs.linear] + [(p, -1.0) for p in rhs.linear]
        quadratic = [(p, 1.0) for p in lhs.quadratic] + [(p, -1.0) for p in rhs.quadratic]
        constant = [(p, 1.0) for p in rhs.constant] + [(p, -1.0) for p in lhs.constant]
        presences = presences_of([p for p, _ in (*linear, *quadratic, *constant)])
        query, finish = indices.planned(self.scope, c.dims, c.where, ROW, 0, presences)
        admitted = None if c.where is None or not presences else indices.counted(self.scope, c.dims, c.where)
        return _Planned(name, c, linear, quadratic, constant, bool(presences), query, finish, admitted)

    def _indexed(self, planned: list[_Planned]) -> list[_Planned]:
        """Every constraint's rows, read in [`_batches`][]; the constraints that hold a row.

        A presence narrows the rows the ``where`` admits, and the rows it
        removed are counted as not built.
        """
        live = []
        for batch in self._batches(planned):
            read = self._together([p.query for p in batch] + [p.admitted for p in batch if p.admitted is not None])
            admitted = iter(read[len(batch) :])
            for p, table in zip(batch, read, strict=False):
                indexed = p.finish(table)
                if p.narrowed:
                    declared = (
                        math.prod(self.attached.dimensions[d].size for d in p.c.dims)
                        if p.admitted is None
                        else next(admitted).item()
                    )
                    if declared > indexed.height:
                        self.measured.omitted[p.name] = self.measured.omitted.get(p.name, 0) + declared - indexed.height
                if not indexed.height:
                    self._without_a_row(p.name, indexed)
                    continue
                self.constraints[p.name] = indices.Indexed(indexed.lazy(), 0, indexed.height)
                live.append(p)
        return live

    def _terms(self, live: list[_Planned]) -> dict[str, _ConstraintShare]:
        """Every constraint's constant side and matrix shares, read in [`_batches`][] and checked in build order.

        The [`coverage`][] checks run on each table before an aggregate can
        hide a hole. A row is built where either matrix has a term.
        """
        shares: dict[str, _ConstraintShare] = {}
        for batch in self._batches(live):
            queries: dict[tuple[str, str], pl.LazyFrame] = {}
            finishes: dict[tuple[str, str], Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.Series]]] = {}
            sides: dict[str, dict[str, frozenset[str]]] = {}
            for p in batch:
                table = self.constraints[p.name].table
                subject = f"constraint '{p.name}'"
                if program.parameters_of(*coverage.divisors_of(p.c.lhs, p.c.rhs)):
                    narrowed = coverage.narrowed_to_rows(table, [t for t, _ in p.constant])
                    queries[p.name, 'undefined'] = coverage.undefined_constants(narrowed)
                queries[p.name, 'side'], sides[p.name] = coverage.constant_side_planned(
                    self.scope, table, p.constant, p.c
                )
                if p.linear:
                    queries[p.name, 'matrix'], finishes[p.name, 'matrix'] = self._matrix_share(
                        self._linear_tables(p, table), subject, p.c.lhs, p.c.rhs
                    )
                if p.quadratic:
                    queries[p.name, 'quadratic'], finishes[p.name, 'quadratic'] = self._quadratic_share(
                        table, p.quadratic, p.name, p.c
                    )
            read = dict(zip(queries, self._together(list(queries.values())), strict=True))
            shares |= {p.name: self._shares(p, read, finishes, sides[p.name]) for p in batch}
        return shares

    def _linear_tables(self, p: _Planned, table: pl.LazyFrame) -> list[pl.LazyFrame]:
        """Each linear term of *p* placed on its rows as ``(row, col, coeff)``, by its own coordinate where it can."""
        carried_order: MaintainOrderJoin | None = 'left_right' if len(p.linear) == 1 else None
        dense = p.c.where is None and not p.narrowed
        tables = []
        for term, sign in p.linear:
            if dense and set(term.dims) == set(p.c.dims):
                placed = term.table.with_columns(self._dense_row(p.c.dims).alias(ROW))
            else:
                placed = join_on(table, term.table, term.dims, 'inner', maintain_order=carried_order)
            tables.append(
                placed.select(
                    ROW,
                    pl.col(COL).cast(_DTYPES[COL]).alias(COL),
                    (sign * pl.col(COEFF)).cast(pl.Float64).alias(COEFF),
                )
            )
        return tables

    def _shares(
        self,
        p: _Planned,
        read: dict[tuple[str, str], pl.DataFrame],
        finishes: Mapping[tuple[str, str], Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.Series]]],
        gaps: Mapping[str, frozenset[str]],
    ) -> _ConstraintShare:
        """One constraint's ``rows``, matrix share and quadratic share, from the tables [`_terms`][] read for it.

        Each table is let go once its share is made, so the raw terms of every
        constraint are not held to the end of the pass.
        """
        subject = f"constraint '{p.name}'"
        if (undefined := read.pop((p.name, 'undefined'), None)) is not None:
            divisors = program.parameters_of(*coverage.divisors_of(p.c.lhs, p.c.rhs))
            coverage.refuse_undefined_constants(undefined, divisors, subject)
        rows = coverage.covered(read.pop((p.name, 'side')), gaps, subject)
        none = pl.Series(ROW, [], dtype=_DTYPES[ROW])
        if not p.linear and not p.quadratic:
            rows, _, _, _ = self._drop_termless_rows(p.name, rows, _stack([], _MATRIX), None, none)
            return rows, None, None
        matrix, term_rows = (
            finishes[p.name, 'matrix'](read.pop((p.name, 'matrix'))) if p.linear else (_stack([], _MATRIX), none)
        )
        qmatrix = None
        if p.quadratic:
            qmatrix, quadratic_rows = finishes[p.name, 'quadratic'](read.pop((p.name, 'quadratic')))
            term_rows = pl.concat([term_rows, quadratic_rows]).unique()
        rows, matrix, qmatrix, _ = self._drop_termless_rows(p.name, rows, matrix, qmatrix, term_rows)
        spread = _magnitude_range(matrix, COEFF)
        if spread is not None:
            self.measured.coefficients[p.name] = spread
        if (sides := _magnitude_range(rows, 'rhs')) is not None:
            self.measured.rhs[p.name] = sides
        return rows, matrix, qmatrix

    def _placed(self, names: list[str], shares: dict[str, _ConstraintShare]) -> list[_ConstraintShare]:
        """Every constraint's shares, in build order, moved from its own rows onto the model's.

        A constraint's rows follow the ones every constraint before it kept, and
        a constraint keeps its rows only once its terms are read, so the move
        waits for the last one. Each share leaves *shares* as it moves, so its
        rows are not held twice.
        """
        placed = []
        for name in names:
            held = self.constraints[name]
            offset = self.n_rows
            self.constraints[name] = indices.Indexed(_moved(held.table, offset), offset, held.height)
            self.n_rows = offset + held.height
            if name not in shares:
                continue
            rows, matrix, qmatrix = shares.pop(name)
            placed.append(
                (
                    _moved(rows, offset),
                    None if matrix is None else _moved(matrix, offset),
                    None if qmatrix is None else _moved(qmatrix, offset),
                )
            )
        return placed

    def _empty_table(self, dims: tuple[str, ...], index: str) -> pl.DataFrame:
        """What an empty entry gets in place of its indexed table: its columns, no rows.

        One per dimensions and index column, shared, since building it is most
        of what such an entry costs.
        """
        if (dims, index) not in self._empty_tables:
            self._empty_tables[dims, index] = indices.nothing(self.scope, dims, index)
        return self._empty_tables[dims, index]

    def _without_a_row(self, name: str, indexed: pl.DataFrame) -> None:
        """A constraint that holds no row, with no query built for its terms.

        Its terms could only narrow its rows further, so querying them would
        change nothing but the cost: each query planned costs a fixed
        time that an entry without a row never earns back.
        """
        self.constraints[name] = indices.Indexed(indexed.lazy(), 0, 0)

    def _dense_row(self, dims: tuple[str, ...]) -> pl.Expr:
        """The row a coordinate lands on in an entry indexed over the whole product of *dims*, numbered from 0.

        Such an entry numbers its rows row-major over the positions
        ([`indices.table`][specsolve._engine.indices.table]), so a term
        computes its row from its own coordinate rather than joining the entry's rows.
        [`tidy_sources`][specsolve._sources.tidy_sources] admits no label outside its dimension, so every coordinate a
        term carries is a row of the entry, as the join would find.
        """
        return self.scope.row_major(dims, self.scope.position_of).cast(pl.Int64)

    def _quadratic_share(
        self, table: pl.LazyFrame, quadratic: list[tuple[Term, float]], name: str, c: program.Constraint
    ) -> tuple[pl.LazyFrame, Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.Series]]]:
        """One constraint's quadratic coefficients, as their query and what their table becomes: ``(row, col_l, col_r, coeff)``.

        Pairs are ordered by column index for [`_objective_quadratic`][]'s
        reason. The finish returns the share and the rows that had any
        quadratic term, read before the prune as [`_matrix_share`][] reads its
        own.
        """
        tables = [
            join_on(table, p.table, p.dims, 'inner').select(
                ROW,
                *_ordered_pair(),
                (sign * pl.col(COEFF)).cast(pl.Float64).alias(COEFF),
            )
            for p, sign in quadratic
        ]

        def share(stacked: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
            coverage.refuse_null_coefficients(stacked, f"constraint '{name}'", c.lhs, c.rhs)
            collapsed, dropped = _collapsed(stacked, (ROW, COL_L, COL_R), ordered=True)
            return collapsed, (stacked if dropped else collapsed).get_column(ROW).unique()

        return pl.concat(tables), share

    def _drop_termless_rows(
        self,
        name: str,
        rows: pl.DataFrame,
        matrix: pl.DataFrame,
        qmatrix: pl.DataFrame | None,
        kept: pl.Series,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None, int]:
        """Rows that kept no variable term are not built, and the entry's rows close up from 0.

        *kept* is read before zeros are pruned, so a row whose every term is
        zero stays: ``0 >= 10`` is infeasible. Survivors renumber, because the
        dual read-back reads an entry's rows by position. Every row either
        share holds is in *kept*, so each of its coefficients moves with its row.
        """
        if kept.len() == rows.height:
            return rows, matrix, qmatrix, rows.height

        surviving = rows.filter(pl.col(ROW).is_in(kept)).sort(ROW)
        renumber = surviving.select(ROW).with_row_index('__new__')
        self.measured.omitted[name] = self.measured.omitted.get(name, 0) + rows.height - surviving.height
        remap = dict(zip(renumber.get_column(ROW), renumber.get_column('__new__'), strict=True))
        rows = surviving.with_columns(pl.col(ROW).replace_strict(remap))
        matrix = matrix.with_columns(pl.col(ROW).replace_strict(remap))
        if qmatrix is not None:
            qmatrix = qmatrix.with_columns(pl.col(ROW).replace_strict(remap))
        kept_table = (
            self.constraints[name].table.filter(pl.col(ROW).is_in(kept)).with_columns(pl.col(ROW).replace_strict(remap))
        )
        self.constraints[name] = indices.Indexed(kept_table, 0, surviving.height)
        return rows, matrix, qmatrix, surviving.height

    def _build_objective(self, o: program.Objective | None) -> pl.DataFrame | None:
        """The objective as ``(col, coeff)``, or ``None`` if it has no terms.

        With no objective declared the sense stays ``min`` and the constant
        ``0``. Rows landing on one column are summed here, because the sink tables'
        scatter keeps only the last write.
        """
        if o is None:
            return None
        terms = self.builder.expression(o.expression, 'objective', admits_quadratic=True)
        coverage.refuse_null_constants(
            [p.table for p in terms.constant], program.parameters_of(*coverage.divisors_of(o.expression)), 'objective'
        )
        for p in terms.constant:
            assert not p.dims, (
                f'objective constant term has dimensions {list(p.dims)} — the language refuses a '
                f'variable-free term of an objective that carries any'
            )
            self.obj_const += p.table.select(pl.col(CONSTANT).sum()).pipe(collected).item() or 0.0
        self.obj_sense = o.sense
        self.quad = self._objective_quadratic(terms.quadratic, o.expression)
        if not terms.linear:
            return None
        tables = [p.table.select(pl.col(COL).cast(_DTYPES[COL]).alias(COL), pl.col(COEFF)) for p in terms.linear]
        stacked = pl.concat(tables).pipe(collected)
        coverage.refuse_null_coefficients(stacked, 'objective', o.expression)
        objective, _ = _collapsed(stacked, (COL,), ordered=False, space=self.n_cols)
        self.measured.objective_range = _magnitude_range(objective, COEFF)
        return objective

    def _objective_quadratic(self, quadratic: tuple[Term, ...], expression: program.Expression) -> pl.DataFrame | None:
        """The objective's quadratic part as ``(col_l, col_r, coeff)``, or ``None``.

        One row per unordered pair, at the coefficient the file wrote, not
        halved; each sink converts. It leaves sorted, because
        [`structure`][specsolve._sinks.tables.SinkTables.structure] hashes it.
        """
        if not quadratic:
            return None
        tables = [p.table.select(*_ordered_pair(), pl.col(COEFF)) for p in quadratic]
        stacked = pl.concat(tables).pipe(collected)
        coverage.refuse_null_coefficients(stacked, 'objective', expression)
        quad, _ = _collapsed(stacked, (COL_L, COL_R), ordered=True)
        return quad


def short_parameters(program: program.Program, attached: AttachedSources) -> dict[str, tuple[int, int]]:
    """Which parameters arrived short, and by how much: ``name -> (reach, rows)``.

    [`tidy_sources`][specsolve._sources.tidy_sources] refuses duplicates and strangers, so a parameter's height is the
    number of coordinates it covers.
    """
    short: dict[str, tuple[int, int]] = {}
    for name, p in program.parameters.items():
        if not p.dims:
            continue
        reach = math.prod(attached.dimensions[d].size for d in p.dims)
        rows = attached.parameter_rows[name]
        if rows < reach:
            short[name] = (reach, rows)
    return short


def declares_quadratic(c: program.Constraint) -> bool:
    """Whether constraint *c* multiplies two variable-carrying operands, either side."""
    return program.is_quadratic(c.lhs) or program.is_quadratic(c.rhs)


def _ordered_pair() -> tuple[pl.Expr, pl.Expr]:
    """A quadratic pair canonicalised by column index, so ``x·y`` and ``y·x`` land in one row."""
    return (
        pl.min_horizontal(COL, COL_2).cast(_DTYPES[COL_L]).alias(COL_L),
        pl.max_horizontal(COL, COL_2).cast(_DTYPES[COL_R]).alias(COL_R),
    )


def _whole(bound: pl.Expr, inward: pl.Expr) -> pl.Expr:
    """An integer column's *bound* as a whole number: the nearest one within 1e-6, else *inward*, its rounding into the range.

    HiGHS answers a fractional bound on an integer column as optimal with no
    feasible values, so every sink is handed whole bounds. The 1e-6 is HiGHS's
    default ``mip_feasibility_tolerance``: a bound that noise put just past a
    whole number keeps that number rather than moving a whole step.
    """
    nearest = bound.round()
    return pl.when((bound - nearest).abs() <= 1e-6).then(nearest).otherwise(inward)


def _magnitude_range(table: pl.DataFrame, *columns: str) -> tuple[float, float] | None:
    """The smallest and largest magnitude across *columns*, or ``None`` where none has one.

    Zero and infinity are dropped, as in the ``Bound`` and ``RHS`` lines a solver prints.
    """
    sides: list[pl.Expr] = []
    for i, column in enumerate(columns):
        value = pl.col(column)
        finite, up, down = value.is_finite(), value > 0, value < 0
        sides += [
            value.filter(finite & up).min().alias(f'#low+{i}'),
            value.filter(finite & up).max().alias(f'#high+{i}'),
            value.filter(finite & down).max().alias(f'#low-{i}'),
            value.filter(finite & down).min().alias(f'#high-{i}'),
        ]
    answered = table.select(sides).row(0, named=True)
    lows = [abs(bound) for name, bound in answered.items() if name.startswith('#low') and bound is not None]
    highs = [abs(bound) for name, bound in answered.items() if name.startswith('#high') and bound is not None]
    return (min(lows), max(highs)) if lows else None


def _collapsed(
    stacked: pl.DataFrame, keys: tuple[str, ...], *, ordered: bool, space: int | None = None
) -> tuple[pl.DataFrame, bool]:
    """*stacked* with repeated *keys* summed and zeros dropped, in key order where *ordered* — and whether a zero went.

    The sort and the aggregate run only when a linear probe says they would
    change something. The stack is rechunked first, because ``shift(1)``
    crosses chunk boundaries that ``is_sorted`` does not. A cancelling pair
    becomes a zero only at the aggregate, so the prune runs again there.
    Unordered, a repeat is probed over *space*, the contiguous index count.
    """
    pruned = _pruned(stacked.rechunk())
    dropped = pruned.height != stacked.height
    stacked = pruned
    if ordered:
        tied = pl.all_horizontal([pl.col(k) == pl.col(k).shift(1) for k in keys])
        probes = stacked.select(_in_key_order(keys).all().alias('#ordered'), tied.any().alias('#repeated'))
        in_order, repeated = probes.row(0)
        if not in_order:
            stacked = _sorted(stacked, keys)
            repeated = stacked.select(tied.any()).item()
    else:
        assert space is not None and len(keys) == 1, 'an unordered probe counts one contiguous integer key'
        repeated = _repeats_an_index(stacked.get_column(keys[0]), space)
    if not repeated:
        return stacked, dropped
    aggregated = stacked.lazy().group_by(*keys).agg(pl.col(COEFF).sum())
    if ordered:
        aggregated = aggregated.sort(*keys)
    summed = aggregated.pipe(collected)
    share = _pruned(summed)
    return share, dropped or share.height != summed.height


def _sorted(stacked: pl.DataFrame, keys: tuple[str, ...]) -> pl.DataFrame:
    """*stacked* sorted by *keys*, on one integer that orders as they do where one can hold them all.

    The keys are solver indices, never negative, so each is a digit in a base one past
    its largest value.
    """
    largest = stacked.select(pl.col(k).max() for k in keys).row(0)
    if math.prod(value + 1 for value in largest) >= 2**63:
        return stacked.sort(*keys)
    position: pl.Expr = pl.lit(0, dtype=pl.Int64)
    for key, value in zip(keys, largest, strict=True):
        position = position * (value + 1) + pl.col(key).cast(pl.Int64)
    return stacked.sort(position)


def _in_key_order(keys: tuple[str, ...]) -> pl.Expr:
    """Whether each row is at or after the one before it, lexicographically over *keys*."""
    head, *rest = keys
    now, before = pl.col(head), pl.col(head).shift(1)
    if not rest:
        return now >= before
    return (now > before) | ((now == before) & _in_key_order(tuple(rest)))


def _ordered_rows(matrix: pl.DataFrame) -> pl.Series:
    """The distinct ``ROW`` indices of a matrix already ordered by ``ROW``.

    ``set_sorted`` is an assertion, not a check: an unordered column silently
    loses rows, so call it only on what [`_collapsed`][] returned ordered.
    """
    return matrix.get_column(ROW).set_sorted().unique()


def _repeats_an_index(keys: pl.Series, count: int) -> bool:
    """Whether any of *keys* occurs twice, over a contiguous ``0..count-1`` space.

    *count* must exceed every index; a stale one raises out of bounds rather
    than answering wrong.
    """
    seen = np.zeros(count, dtype=bool)
    seen[keys.to_numpy()] = True
    return int(np.count_nonzero(seen)) != keys.len()


def _pruned(matrix: pl.DataFrame) -> pl.DataFrame:
    """*matrix* without its exactly-zero coefficients — unchanged, and not rechunked, when it has none.

    A pruned share no longer says which rows had terms, so
    [`Assembly._matrix_share`][] reads that set before pruning.
    """
    if not matrix.select(pl.col(COEFF).eq(0).any()).item():
        return matrix
    return matrix.filter(pl.col(COEFF) != 0).rechunk()


def _row_starts(ordered: pl.DataFrame, row_count: int) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
    """Each row's first coefficient in the row-ordered *ordered* — CSR's own index.

    *ordered* must ascend in ``ROW``: a row in two runs silently gets wrong spans.
    """
    runs = ordered[ROW].rle()
    starts = np.zeros(row_count + 1, dtype=np.int64)
    starts[runs.struct.field('value').to_numpy() + 1] = runs.struct.field('len').to_numpy()
    return np.cumsum(starts, out=starts)


def _stack(tables: list[pl.DataFrame], columns: tuple[str, ...]) -> pl.DataFrame:
    """Concatenate *tables*, or an empty table of *columns* when there are none."""
    if tables:
        return pl.concat(tables)
    return pl.DataFrame(schema={name: _DTYPES[name] for name in columns})


def _moved[F: (pl.DataFrame, pl.LazyFrame)](table: F, offset: int) -> F:
    """*table* with its ``ROW`` *offset* rows on: an entry's rows, numbered from 0 while it built, among the model's."""
    return table if not offset else table.with_columns(pl.col(ROW) + offset)
