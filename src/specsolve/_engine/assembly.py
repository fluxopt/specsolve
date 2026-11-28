"""One build: every entry into rows of the model tables a sink drains.

[`build`][] is the whole of it. The variable and constraint index tables fill
during a build, because an entry built later reads what earlier ones produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ParamSpec, TypeVar, get_args

import numpy as np
import polars as pl
from mathspec import program

from specsolve import _sinks as sinks
from specsolve._answer.result import Diagnostics
from specsolve._collect import collected, in_memory
from specsolve._engine import coverage, indices
from specsolve._engine.queries import QueryBuilder
from specsolve._engine.scope import Scope, join_on
from specsolve._engine.specialize import is_empty, specialized, specialized_on
from specsolve._engine.terms import Term, presences_of
from specsolve._names import COEFF, COL, COL_2, COL_L, COL_R, CONSTANT, ROW
from specsolve._sinks.tables import SENSE

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

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
        #: How many special-ordered sets have been numbered, contiguous ``0..n-1`` across the model.
        self.n_sets = 0
        self.quad: pl.DataFrame | None = None
        self.obj_const = 0.0
        self.obj_sense: ObjectiveSense | None = None
        self._empty_tables = {}

    def run(self) -> BuiltModel:
        """Build every entry, then freeze what they produced.

        Quadratic constraints build last, so their rows are a contiguous tail
        a sink takes as a slice; the sort is stable, so file order survives in
        each half. The matrix and ``rows`` leave in ``(row, col)`` order, as
        ``SinkTables`` promises its sinks. A build whose [`_reach`][] is at most
        [`IN_MEMORY_REACH`][] collects [`in_memory`][].
        """
        with in_memory(self._reach() <= IN_MEMORY_REACH):
            self.lap('build', 'specialize', None)
            cols = [
                self._timed('variable', name, self._build_variable, name, v)
                for name, v in self.program.variables.items()
            ]
            sets = [
                self._timed('sos', name, self._build_sos, s, self.program.variables[s.variable])
                for name, s in self.program.sos.items()
            ]
            ordered = sorted(self.program.constraints.items(), key=lambda item: declares_quadratic(item[1]))
            built = [self._timed('constraint', name, self._build_constraint, name, c) for name, c in ordered]
            o = self.program.objective
            objective = None if o is None else self._timed('objective', None, self._build_objective, o)

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
        self.lap('build', 'sink_tables', None)
        return model

    def _timed(
        self, step: Step, entry: str | None, build: Callable[_Args, _Built], *args: _Args.args, **kwargs: _Args.kwargs
    ) -> _Built:
        """*build* on *args*, then a lap of the build as *step* of *entry*, even if it raises.

        Each entry collects its own share before it returns, so the seconds are
        its own and do not move with the order entries build in.
        """
        try:
            return build(*args, **kwargs)
        finally:
            self.lap('build', step, entry)

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
    ) -> tuple[pl.DataFrame, pl.Series]:
        """One constraint's share: in ``(row, col)`` order, repeated cells summed.

        Returns:
            The share, and the rows that had any term, read before the prune: a
            row whose every coefficient is zero is not a row with no terms.
        """
        stacked = pl.concat(tables).pipe(collected)
        coverage.refuse_null_coefficients(stacked, name, *expressions)
        share, dropped = _collapsed(stacked, (ROW, COL), ordered=True)
        return share, stacked.get_column(ROW).unique() if dropped else _ordered_rows(share)

    # ------------------------------------------------------------------
    # entries
    # ------------------------------------------------------------------

    def _build_variable(self, name: str, v: program.Variable) -> pl.DataFrame:
        """One variable's indexed table, and its share of ``cols``.

        The share leaves in index order: ``cols`` carries no ``COL``, so a row's
        position is its solver column index. The bounds are collected in memory:
        they are parameters looked up onto column indices already in memory, and polars
        2.0's streaming engine runs that ordered join 4.6 to 5.5 times slower
        than its in-memory one on the nodal benchmark at sizes m and l (#1857).
        """
        start = self.n_cols
        if is_empty(v, self.attached):
            indexed = self._empty_table(v.dims, COL)
        else:
            indexed = indices.table(self.scope, v.dims, v.where, COL, start)
        self.n_cols = start + indexed.height
        self.variables[name] = indices.Indexed(indexed.lazy(), start, indexed.height)
        if not indexed.height:
            return _stack([], _COLS)
        return self._columns(name, v, indexed.lazy())

    def _columns(self, name: str, v: program.Variable, table: pl.LazyFrame) -> pl.DataFrame:
        """The share of ``cols`` for the variable indexed by *table*, its bounds read off the data attached now."""
        lb, ub = pl.col('lb').cast(pl.Float64), pl.col('ub').cast(pl.Float64)
        if v.domain != 'continuous':
            lb, ub = _whole(lb, lb.ceil()), _whole(ub, ub.floor())
        with in_memory():
            bounds = self.builder.bounds(table, name, v).select(COL, lb.alias('lb'), ub.alias('ub'))
            bounded = indices.in_position_order(bounds.pipe(collected), COL)
        cols = bounded.select('lb', 'ub', pl.lit(v.domain, dtype=_DTYPES['vtype']).alias('vtype'))

        self.measured.bounds.pop(name, None)
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

    def _build_constraint(self, name: str, c: program.Constraint) -> _ConstraintShare:
        """One constraint as its ``rows``, its share of the matrix, and its quadratic share.

        The [`coverage`][] checks run in its table's order, before an aggregate
        can hide a hole. A row is built where either matrix has a term, and the
        entry's index narrows to the rows that survived. A constraint whose
        ``where`` admits no row has no query built. One without a ``where`` is
        indexed only once the presences that narrow it are known, so a sparse
        absence never meets the full product of its dimensions.
        """
        start = self.n_rows
        if is_empty(c, self.attached):
            return self._without_a_row(name, self._empty_table(c.dims, ROW), start)
        admitted = None if c.where is None else indices.table(self.scope, c.dims, c.where, ROW, start)
        if admitted is not None and not admitted.height:
            return self._without_a_row(name, admitted, start)
        linear, quadratic, constant = self._terms(name, c)
        presences = presences_of([p for p, _ in (*linear, *quadratic, *constant)])
        if admitted is None or presences:
            indexed = indices.table(self.scope, c.dims, c.where, ROW, start, presences)
        else:
            indexed = admitted
        if presences:
            declared = (
                math.prod(self.attached.dimensions[d].size for d in c.dims) if admitted is None else admitted.height
            )
            if declared > indexed.height:
                self.measured.omitted[name] = self.measured.omitted.get(name, 0) + declared - indexed.height
        self.n_rows = start + indexed.height
        table = indexed.lazy()
        self.constraints[name] = indices.Indexed(table, start, indexed.height)

        rows = self._constant_side(name, c, table, constant)

        if not linear and not quadratic:
            none = pl.Series(ROW, [], dtype=_DTYPES[ROW])
            rows, _, _, self.n_rows = self._drop_termless_rows(name, rows, _stack([], _MATRIX), None, none, start)
            return rows, None, None

        matrix, term_rows = self._linear_share(name, c, table, start, linear, dense=c.where is None and not presences)
        qmatrix, quadratic_rows = self._quadratic_share(table, quadratic, name, c)
        if qmatrix is not None:
            term_rows = pl.concat([term_rows, quadratic_rows]).unique()
        rows, matrix, qmatrix, self.n_rows = self._drop_termless_rows(name, rows, matrix, qmatrix, term_rows, start)
        spread = _magnitude_range(matrix, COEFF)
        if spread is not None:
            self.measured.coefficients[name] = spread
        if (sides := _magnitude_range(rows, 'rhs')) is not None:
            self.measured.rhs[name] = sides
        return rows, matrix, qmatrix

    def _empty_table(self, dims: tuple[str, ...], index: str) -> pl.DataFrame:
        """What an empty entry gets in place of its indexed table: its columns, no rows.

        One per dimensions and index column, shared, since building it is most
        of what such an entry costs.
        """
        if (dims, index) not in self._empty_tables:
            self._empty_tables[dims, index] = indices.nothing(self.scope, dims, index)
        return self._empty_tables[dims, index]

    def _without_a_row(
        self, name: str, indexed: pl.DataFrame, start: int
    ) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None]:
        """A constraint that holds no row, with no query built for its terms.

        Its terms could only narrow its rows further, so querying them would
        change nothing but the cost: each query planned costs a fixed
        time that an entry without a row never earns back.
        """
        self.constraints[name] = indices.Indexed(indexed.lazy(), start, 0)
        return _stack([], _ROWS), None, None

    def _terms(
        self, name: str, c: program.Constraint
    ) -> tuple[list[tuple[Term, float]], list[tuple[Term, float]], list[tuple[Term, float]]]:
        """*c* as its linear, quadratic and constant terms, each signed for ``lhs - rhs``.

        A linear or a quadratic term keeps the side it was written on; a constant term moves to the right.
        """
        admits = declares_quadratic(c)
        lhs = self.builder.expression(c.lhs, f"constraint '{name}' lhs", admits_quadratic=admits)
        rhs = self.builder.expression(c.rhs, f"constraint '{name}' rhs", admits_quadratic=admits)
        linear = [(p, 1.0) for p in lhs.linear] + [(p, -1.0) for p in rhs.linear]
        quadratic = [(p, 1.0) for p in lhs.quadratic] + [(p, -1.0) for p in rhs.quadratic]
        constant = [(p, 1.0) for p in rhs.constant] + [(p, -1.0) for p in lhs.constant]
        return linear, quadratic, constant

    def _linear_share(
        self,
        name: str,
        c: program.Constraint,
        table: pl.LazyFrame,
        start: int,
        linear: list[tuple[Term, float]],
        *,
        dense: bool,
    ) -> tuple[pl.DataFrame, pl.Series]:
        """*c*'s *linear* terms placed on the rows of *table*, numbered from *start*, as [`_matrix_share`][] returns them.

        *dense* says *table* is the whole product of *c*'s dimensions, so a term
        over all of them computes its row ([`_dense_row`][]) instead of joining.
        """
        if not linear:
            return _stack([], _MATRIX), pl.Series(ROW, [], dtype=_DTYPES[ROW])
        carried_order: MaintainOrderJoin | None = 'left_right' if len(linear) == 1 else None
        tables = []
        for p, sign in linear:
            if dense and set(p.dims) == set(c.dims):
                placed = p.table.with_columns(self._dense_row(c.dims, start).alias(ROW))
            else:
                placed = join_on(table, p.table, p.dims, 'inner', maintain_order=carried_order)
            tables.append(
                placed.select(
                    ROW,
                    pl.col(COL).cast(_DTYPES[COL]).alias(COL),
                    (sign * pl.col(COEFF)).cast(pl.Float64).alias(COEFF),
                )
            )
        return self._matrix_share(tables, f"constraint '{name}'", c.lhs, c.rhs)

    def _constant_side(
        self, name: str, c: program.Constraint, table: pl.LazyFrame, constant: list[tuple[Term, float]]
    ) -> pl.DataFrame:
        """The rows of *table* as ``(ROW, sense, rhs)``, every constant term checked where the rows ask for it."""
        subject = f"constraint '{name}'"
        coverage.refuse_null_constants(
            coverage.narrowed_to_rows(table, [p for p, _ in constant]),
            program.parameters_of(*coverage.divisors_of(c.lhs, c.rhs)),
            subject,
        )
        return coverage.constant_side(self.scope, table, constant, c, subject)

    def _dense_row(self, dims: tuple[str, ...], start: int) -> pl.Expr:
        """The row a coordinate lands on in an entry indexed over the whole product of *dims* from *start*.

        Such an entry numbers its rows row-major over the positions
        ([`indices.table`][specsolve._engine.indices.table]), so a term
        computes its row from its own coordinate rather than joining the entry's rows.
        [`tidy_sources`][specsolve._sources.tidy_sources] admits no label outside its dimension, so every coordinate a
        term carries is a row of the entry, as the join would find.
        """
        return pl.lit(start, dtype=pl.Int64) + self.scope.row_major(dims, self.scope.position_of)

    def _quadratic_share(
        self, table: pl.LazyFrame, quadratic: list[tuple[Term, float]], name: str, c: program.Constraint
    ) -> tuple[pl.DataFrame | None, pl.Series]:
        """One constraint's quadratic coefficients as ``(row, col_l, col_r, coeff)``, in that order.

        Pairs are ordered by column index for [`_objective_quadratic`][]'s
        reason.

        Returns:
            The share, ``None`` without a quadratic term, and the rows that had
            any quadratic term, read before the prune as
            [`_matrix_share`][] reads its own.
        """
        if not quadratic:
            return None, pl.Series(ROW, [], dtype=_DTYPES[ROW])
        tables = [
            join_on(table, p.table, p.dims, 'inner').select(
                ROW,
                *_ordered_pair(),
                (sign * pl.col(COEFF)).cast(pl.Float64).alias(COEFF),
            )
            for p, sign in quadratic
        ]
        stacked = pl.concat(tables).pipe(collected)
        coverage.refuse_null_coefficients(stacked, f"constraint '{name}'", c.lhs, c.rhs)
        share, dropped = _collapsed(stacked, (ROW, COL_L, COL_R), ordered=True)
        return share, (stacked if dropped else share).get_column(ROW).unique()

    def _drop_termless_rows(
        self,
        name: str,
        rows: pl.DataFrame,
        matrix: pl.DataFrame,
        qmatrix: pl.DataFrame | None,
        kept: pl.Series,
        start: int,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None, int]:
        """Rows that kept no variable term are not built, and the entry's rows close up.

        *kept* is read before zeros are pruned, so a row whose every term is
        zero stays: ``0 >= 10`` is infeasible. Survivors renumber from *start*,
        because the dual read-back reads an entry's rows by position. Every row either
        share holds is in *kept*, so each of its coefficients moves with its row.
        """
        if kept.len() == rows.height:
            return rows, matrix, qmatrix, start + rows.height

        surviving = rows.filter(pl.col(ROW).is_in(kept)).sort(ROW)
        renumber = surviving.select(ROW).with_row_index('__new__', offset=start)
        self.measured.omitted[name] = self.measured.omitted.get(name, 0) + rows.height - surviving.height
        remap = dict(zip(renumber.get_column(ROW), renumber.get_column('__new__'), strict=True))
        rows = surviving.with_columns(pl.col(ROW).replace_strict(remap))
        matrix = matrix.with_columns(pl.col(ROW).replace_strict(remap))
        if qmatrix is not None:
            qmatrix = qmatrix.with_columns(pl.col(ROW).replace_strict(remap))
        kept_table = (
            self.constraints[name].table.filter(pl.col(ROW).is_in(kept)).with_columns(pl.col(ROW).replace_strict(remap))
        )
        self.constraints[name] = indices.Indexed(kept_table, start, surviving.height)
        return rows, matrix, qmatrix, start + surviving.height

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


def numbers_only(
    compiled: program.Program,
    built: BuiltModel,
    measured: Measured,
    attached: AttachedSources,
    changed: Collection[str],
) -> bool:
    """Whether *changed* changes only numbers [`spliced_model`][] can read again, so *built* keeps its indices.

    Every name must be a parameter that no build reads for which columns, rows
    or coefficients exist, nor for a quadratic row
    ([`_structural_parameters`][]), nor one *compiled* is
    [`specialized_on`][], attached now at exactly the coordinates *built*
    attached it at. Both are asked of *compiled*, the program as written:
    *built* keeps the program the data specialized, which no longer names a
    condition the old data settled. A constraint it feeds must have built
    every row it indexed: its checks ran on rows the stored table no longer
    holds once a row is dropped.
    """
    plan = built.program
    structural = _structural_parameters(compiled) | specialized_on(compiled)
    if not set(changed) <= set(plan.parameters) or set(changed) & structural:
        return False
    if any(name in measured.omitted for name in _entries_reading(plan.constraints, changed, _constraint_parameters)):
        return False
    return all(_same_coordinates(built.attached, attached, name, plan.parameters[name].dims) for name in changed)


def spliced_model(
    built: BuiltModel,
    attached: AttachedSources,
    changed: Collection[str],
    measured: Measured,
    lap: Lap,
) -> BuiltModel:
    """*built* with every bound, constant side, linear coefficient and objective that reads *changed* read again.

    The tables a full build of the same data gives, where [`numbers_only`][]
    holds: the indices, the quadratic rows and the sets are *built*'s, and each
    recomputed share is spliced over the run of indices its entry owns. An
    entry that owns no index has no share to read, so it is not queried. A
    constraint's coefficients are read again only where *changed* scales a
    variable in it; it built every row it indexed, so one with no ``where``
    spans the whole product of its dimensions. *measured* is *built*'s,
    updated for what was read. *lap* logs the steps a full build logs, for
    the entries read again: ``setup``, one ``variable`` or ``constraint`` row
    per entry, the ``objective``, then ``tables``. No ``matrix`` row: a
    constraint's coefficients are spliced in within its own row.

    Raises:
        DataError: A bound, constant side, coefficient or objective the new
            data leaves without a value, as a full build would.
    """
    plan, tables = built.program, built.tables
    assembly = Assembly(plan, attached, measured, lap)
    assembly.variables.update(built.variables)
    assembly.constraints.update(built.constraints)
    assembly.n_cols, assembly.n_rows = tables.column_count, tables.row_count
    cols, rows, matrix, starts = tables.cols, tables.rows, tables.matrix, tables.row_starts
    obj, quad, constant = tables.obj, tables.quad, tables.objective_constant

    def variable(name: str) -> None:
        nonlocal cols
        held = built.variables[name]
        share = assembly._columns(name, plan.variables[name], held.table)
        cols = _spliced(cols, held.start, held.start + held.height, share)

    def constraint(name: str) -> None:
        nonlocal rows, matrix, starts
        held, c = built.constraints[name], plan.constraints[name]
        terms, _, consts = assembly._terms(name, c)
        share = indices.in_position_order(assembly._constant_side(name, c, held.table, consts), ROW)
        measured.rhs.pop(name, None)
        if (sides := _magnitude_range(share, 'rhs')) is not None:
            measured.rhs[name] = sides
        at = rows.get_column(ROW)
        lo, hi = at.search_sorted(held.start, 'left'), at.search_sorted(held.start + held.height, 'left')
        rows = _spliced(rows, lo, hi, share)
        if _coefficient_parameters(c) & set(changed):
            linear, _ = assembly._linear_share(name, c, held.table, held.start, terms, dense=c.where is None)
            matrix, starts = _respliced(matrix, starts, held.start, held.start + held.height, linear)
            measured.coefficients.pop(name, None)
            if (spread := _magnitude_range(linear, COEFF)) is not None:
                measured.coefficients[name] = spread

    def objective(o: program.Objective) -> None:
        nonlocal obj, quad, constant
        measured.objective_range = None
        terms = assembly._build_objective(o)
        obj = _stack([terms] if terms is not None else [], _OBJ)
        quad = _stack([] if assembly.quad is None else [assembly.quad], _QUAD)
        constant = assembly.obj_const

    with in_memory(assembly._reach() <= IN_MEMORY_REACH):
        lap('build', 'splice', None)
        for name in _entries_reading(plan.variables, changed, _bound_parameters):
            if built.variables[name].height:
                assembly._timed('variable', name, variable, name)
        for name in _entries_reading(plan.constraints, changed, _constraint_parameters):
            if built.constraints[name].height:
                assembly._timed('constraint', name, constraint, name)
        o = plan.objective
        if o is not None and program.parameters_of(o.expression) & set(changed):
            assembly._timed('objective', None, objective, o)

    measured.nonzeros = matrix.height
    measured.bounds = _in_build_order(measured.bounds, built.variables)
    measured.coefficients = _in_build_order(measured.coefficients, built.constraints)
    measured.rhs = _in_build_order(measured.rhs, built.constraints)
    refreshed = replace(
        tables,
        cols=cols,
        rows=rows,
        matrix=matrix,
        row_starts=starts,
        obj=obj,
        quad=quad,
        objective_constant=constant,
    )
    model = BuiltModel(plan, attached, built.variables, built.constraints, refreshed)
    lap('build', 'sink_tables', None)
    return model


def _structural_parameters(plan: program.Program) -> frozenset[str]:
    """The parameters a build reads for which columns, rows or coefficients exist, or for a quadratic row.

    Those a ``where`` reads, or a ``cases`` region's condition, or a shift or a
    window as its amount, decide which coordinates exist. A coefficient of a
    quadratic constraint is part of a row a sink takes whole. Every other use
    is a bound, a constant side, a linear coefficient or the objective, which
    [`spliced_model`][] reads again.
    """
    names: set[str] = set()
    for entry in (*plan.variables.values(), *plan.constraints.values()):
        if entry.where is not None:
            names |= entry.where.names_read
    for c in plan.constraints.values():
        if declares_quadratic(c):
            names |= _coefficient_parameters(c)
        for node in program.walk(c.lhs, c.rhs):
            if isinstance(node, program.Cases):
                names.update(*(region.when.names_read for region in node.regions))
            elif isinstance(node, (program.Translate, program.WindowSum)):
                amount = node.offset if isinstance(node, program.Translate) else node.width
                if isinstance(amount, str):
                    names.add(amount)
    return frozenset(names)


def _coefficient_parameters(c: program.Constraint) -> frozenset[str]:
    """The parameters that scale a variable in *c*: a factor of a product that carries one, or a divisor of one."""
    names: set[str] = set()
    for node in program.walk(c.lhs, c.rhs):
        if isinstance(node, program.Multiply):
            for side, other in ((node.left, node.right), (node.right, node.left)):
                if program.carries_variable(other):
                    names |= program.parameters_of(side)
        elif isinstance(node, program.Divide) and program.carries_variable(node.numerator):
            names |= program.parameters_of(node.divisor)
    return frozenset(names)


def _bound_parameters(v: program.Variable) -> frozenset[str]:
    """The parameters *v*'s bounds read."""
    return program.parameters_of(*(side for side in (v.lower, v.upper) if side is not None))


def _constraint_parameters(c: program.Constraint) -> frozenset[str]:
    """The parameters either side of *c* reads."""
    return program.parameters_of(c.lhs, c.rhs)


def _entries_reading[D](
    entries: Mapping[str, D], changed: Collection[str], reads: Callable[[D], frozenset[str]]
) -> list[str]:
    """The names of *entries* that *reads* says read a name in *changed*, in the order they are written."""
    return [name for name, d in entries.items() if reads(d) & set(changed)]


def _same_coordinates(before: AttachedSources, now: AttachedSources, name: str, dims: tuple[str, ...]) -> bool:
    """Whether parameter *name* is attached at the coordinates it was, a filled one at the coordinates it was filled to.

    [`tidy_sources`][specsolve._sources.tidy_sources] refuses a repeated
    coordinate, so equal heights and nothing new is the same set.
    """
    if before.parameter_rows[name] != now.parameter_rows[name]:
        return False
    if not dims:
        return True
    new = now.parameters[name].select(*dims).join(before.parameters[name].select(*dims), on=list(dims), how='anti')
    return new.select(pl.len()).pipe(collected).item() == 0


def _in_build_order(
    ranges: dict[str, tuple[float, float]], built: Mapping[str, object]
) -> dict[str, tuple[float, float]]:
    """*ranges* keyed as *built* is ordered: a range read again goes back to its entry's place in the diagnostics."""
    return {name: ranges[name] for name in built if name in ranges}


def _spliced(table: pl.DataFrame, lo: int, hi: int, share: pl.DataFrame) -> pl.DataFrame:
    """*table* with its rows ``[lo, hi)`` replaced by *share*."""
    return pl.concat([table.slice(0, lo), share, table.slice(hi)])


def _respliced(
    matrix: pl.DataFrame, starts: np.ndarray[tuple[int, ...], np.dtype[np.int64]], lo: int, hi: int, share: pl.DataFrame
) -> tuple[pl.DataFrame, np.ndarray[tuple[int, ...], np.dtype[np.int64]]]:
    """The CSR *matrix* and its *starts* with rows ``[lo, hi)`` owning *share* instead.

    *share* is ``(ROW, COL, COEFF)`` in ``(ROW, COL)`` order, every row in
    ``[lo, hi)``: a coefficient that falls to zero leaves the matrix, so the
    rows after *hi* move by the difference.
    """
    first, last = int(starts[lo]), int(starts[hi])
    spliced = pl.concat([matrix.slice(0, first), share.select(COL, COEFF), matrix.slice(last)]).rechunk()
    owned = _row_starts(share.with_columns(pl.col(ROW) - lo), hi - lo) + first
    return spliced, np.concatenate([starts[:lo], owned, starts[hi + 1 :] + (int(owned[-1]) - last)])


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
