"""One build: every entry into rows of the model tables a sink drains.

The variable and constraint index tables fill during a build, because an
entry built later reads what earlier ones produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, get_args

import numpy as np
import polars as pl
from mathspec import program

from specsolve import _sinks as sinks
from specsolve._collect import collected, in_memory
from specsolve._engine import coverage, indices
from specsolve._engine.queries import QueryBuilder
from specsolve._engine.scope import Scope, join_on
from specsolve._engine.specialize import rowless
from specsolve._engine.terms import Term, presences_of
from specsolve._messages import null_bounds_message
from specsolve._names import COEFF, COL, COL_2, COL_L, COL_R, CONSTANT, ROW
from specsolve._sinks.handoff import SENSE
from specsolve.errors import DataError

if TYPE_CHECKING:
    from mathspec.program import ObjectiveSense
    from polars._typing import MaintainOrderJoin

    from specsolve._engine.attaching import AttachedSources


#: The most columns or rows a model may reach for its build to collect in memory.
IN_MEMORY_REACH = 250_000

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


@dataclass(frozen=True)
class BuiltModel:
    """One build's product: the handoff a sink drains, and what reads it back."""

    program: program.Program
    attached: AttachedSources
    #: One [`Indexed`][specsolve._engine.indices.Indexed] per
    #: entry, one map per index space: a variable and a constraint may share a name.
    variables: dict[str, indices.Indexed]
    constraints: dict[str, indices.Indexed]
    handoff: sinks.Handoff


class Assembly:
    """One build in progress: the mutable half, discarded once [`run`][] has frozen it."""

    def __init__(self, program: program.Program, attached: AttachedSources, measured: Measured) -> None:
        """Start a build of *program*, [`specialized`][specsolve._engine.specialize.specialized] to *attached*."""
        self.program = program
        self.attached = attached
        self.measured = measured
        self.variables: dict[str, indices.Indexed] = {}
        self.constraints: dict[str, indices.Indexed] = {}
        self.scope = Scope(program, attached, self.variables)
        self.builder = QueryBuilder(self.scope)
        self.n_cols = 0
        self.n_rows = 0
        #: How many special-ordered sets have been numbered, contiguous ``0..n-1`` across the model.
        self.n_sets = 0
        self.quad: pl.DataFrame | None = None
        self.obj_const = 0.0
        self.obj_sense: ObjectiveSense | None = None

    def run(self) -> BuiltModel:
        """Build every entry, then freeze what they produced.

        Quadratic constraints build last, so their rows are a contiguous tail
        a sink takes as a slice; the sort is stable, so file order survives in
        each half. The matrix and ``rows`` leave in ``(row, col)`` order, as
        ``Handoff`` promises its sinks. A build whose [`_reach`][] is at most
        [`IN_MEMORY_REACH`][] collects [`in_memory`][].
        """
        with in_memory(self._reach() <= IN_MEMORY_REACH):
            cols = [self._build_variable(name, v) for name, v in self.program.variables.items()]
            sets = [self._build_sos(s, self.program.variables[s.variable]) for s in self.program.sos.values()]
            ordered = sorted(self.program.constraints.items(), key=lambda item: declares_quadratic(item[1]))
            built = [self._build_constraint(name, c) for name, c in ordered]
            objective = self._build_objective(self.program.objective)

        stacked = indices.in_position_order(_stack([m for _, m, _ in built if m is not None], _MATRIX), ROW)
        matrix_starts = _row_starts(stacked, self.n_rows)
        matrix = stacked.select(COL, COEFF).rechunk()

        self.measured.columns = self.n_cols
        self.measured.rows = self.n_rows
        self.measured.nonzeros = matrix.height
        handoff = sinks.Handoff(
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
        return BuiltModel(self.program, self.attached, self.variables, self.constraints, handoff)

    def _reach(self) -> int:
        """The columns or the rows the model would have with no ``where``, whichever is more: known before the build.

        A constraint over dimensions no variable spans has more rows than there are
        columns, so the rows count too.
        """
        cardinality = self.attached.cardinality
        return max(
            sum(math.prod(cardinality[d] for d in declared.dims) for declared in entries.values())
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
        if rowless(v, self.attached):
            indexed = indices.nothing(self.scope, v.dims, COL)
        else:
            indexed = indices.table(self.scope, v.dims, v.where, COL, start)
        self.n_cols = start + indexed.height
        self.variables[name] = indices.Indexed(indexed.lazy(), start, indexed.height)
        if not indexed.height:
            return _stack([], _COLS)

        lb, ub = pl.col('lb').cast(pl.Float64), pl.col('ub').cast(pl.Float64)
        if v.domain != 'continuous':
            lb, ub = _whole(lb, lb.ceil()), _whole(ub, ub.floor())
        with in_memory():
            bounds = self.builder.bounds(indexed.lazy(), name, v).select(COL, lb.alias('lb'), ub.alias('ub'))
            bounded = indices.in_position_order(bounds.pipe(collected), COL)
        cols = bounded.select('lb', 'ub', pl.lit(v.domain, dtype=_DTYPES['vtype']).alias('vtype'))

        if bounded.get_column('lb').null_count() or bounded.get_column('ub').null_count():
            bad = cols.filter(pl.col('lb').is_null() | pl.col('ub').is_null()).height
            raise DataError(null_bounds_message(name, bad))

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
        cardinality = self.scope.data.cardinality
        stride = math.prod(cardinality[d] for d in v.dims[v.dims.index(s.along) + 1 :])
        span = cardinality[s.along] * stride

        if v.where is None:
            table = pl.select(pl.int_range(held.height, dtype=pl.Int64).alias('#position')).lazy()
            place, col = pl.col('#position'), (pl.col('#position') + held.start)
        else:
            table = held.table
            place, col = self.scope.row_major(v.dims, self.scope.position_of), pl.col(COL)
        placed = table.select(
            ((place // span) * stride + place % stride).alias('#set position'),
            ((place // stride) % cardinality[s.along] + 1).cast(_DTYPES['weight']).alias('weight'),
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

    def _build_constraint(
        self, name: str, c: program.Constraint
    ) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None]:
        """One constraint as its ``rows``, its share of the matrix, and its quadratic share.

        The [`coverage`][] checks run in its table's order, before an aggregate
        can hide a hole. A row is built where either matrix has a term, and the
        entry's index narrows to the rows that survived. A constraint whose
        ``where`` admits no row has no query built. One without a ``where`` is
        indexed only once the presences that narrow it are known, so a sparse
        absence never meets the full product of its dimensions.
        """
        start = self.n_rows
        if rowless(c, self.attached):
            return self._without_a_row(name, indices.nothing(self.scope, c.dims, ROW), start)
        admitted = None if c.where is None else indices.table(self.scope, c.dims, c.where, ROW, start)
        if admitted is not None and not admitted.height:
            return self._without_a_row(name, admitted, start)
        admits = declares_quadratic(c)
        lhs = self.builder.expression(c.lhs, f"constraint '{name}' lhs", admits_quadratic=admits)
        rhs = self.builder.expression(c.rhs, f"constraint '{name}' rhs", admits_quadratic=admits)
        linear = [(p, 1.0) for p in lhs.linear] + [(p, -1.0) for p in rhs.linear]
        quadratic = [(p, 1.0) for p in lhs.quadratic] + [(p, -1.0) for p in rhs.quadratic]
        constant = [(p, 1.0) for p in rhs.constant] + [(p, -1.0) for p in lhs.constant]
        presences = presences_of([p for p, _ in (*linear, *quadratic, *constant)])
        if admitted is None or presences:
            indexed = indices.table(self.scope, c.dims, c.where, ROW, start, presences)
        else:
            indexed = admitted
        if presences:
            declared = math.prod(self.attached.cardinality[d] for d in c.dims) if admitted is None else admitted.height
            if declared > indexed.height:
                self.measured.omitted[name] = self.measured.omitted.get(name, 0) + declared - indexed.height
        self.n_rows = start + indexed.height
        table = indexed.lazy()
        self.constraints[name] = indices.Indexed(table, start, indexed.height)

        subject = f"constraint '{name}'"
        coverage.refuse_null_constants(
            coverage.narrowed_to_rows(table, [p for p, _ in constant]),
            program.parameters_of(*coverage.divisors_of(c.lhs, c.rhs)),
            subject,
        )
        rows = coverage.constant_side(self.scope, table, constant, c, subject)

        if not linear and not quadratic:
            none = pl.Series(ROW, [], dtype=_DTYPES[ROW])
            rows, _, _, self.n_rows = self._drop_termless_rows(name, rows, _stack([], _MATRIX), None, none, start)
            return rows, None, None

        tables = []
        carried_order: MaintainOrderJoin | None = 'left_right' if len(linear) == 1 else None
        dense = c.where is None and not presences
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
        matrix, term_rows = (
            self._matrix_share(tables, subject, c.lhs, c.rhs)
            if tables
            else (_stack([], _MATRIX), pl.Series(ROW, [], dtype=_DTYPES[ROW]))
        )
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
        ``0``. Rows landing on one column are summed here, because the handoff
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
        [`structure`][specsolve._sinks.handoff.Handoff.structure] hashes it.
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
        reach = math.prod(attached.cardinality[d] for d in p.dims)
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
