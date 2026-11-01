"""One build: every declaration into rows of the model frames a sink drains.

The variable and constraint label frames fill during a build, because a
declaration built later reads what earlier ones produced.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, get_args

import numpy as np
import polars as pl
from mathspec import program

from specsolve import _sinks as sinks
from specsolve._collect import collected, collected_all, in_memory
from specsolve._engine import coverage, labels
from specsolve._engine.compiler import Compiler
from specsolve._engine.scope import Scope, join_on
from specsolve._engine.terms import Presence, Term, absence_restrictions
from specsolve._messages import null_bounds_message
from specsolve._sinks.handoff import SENSE
from specsolve.errors import DataError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mathspec.program import ObjectiveSense
    from polars._typing import MaintainOrderJoin

    from specsolve._engine.attaching import AttachedSources


#: The most columns or rows a model may reach for its build to collect in memory.
IN_MEMORY_REACH = 250_000

#: The frames a sink reads, as schemas.
_COLS = ('lb', 'ub', 'vtype')
_OBJ = ('col', 'coeff')
_QUAD = ('col_l', 'col_r', 'coeff')
_ROWS = ('row', 'sense', 'rhs')
_MATRIX = ('row', 'col', 'coeff')
_QMATRIX = ('row', 'col_l', 'col_r', 'coeff')
_SOS = ('set', 'type', 'col', 'weight')

#: The dtype of each of those columns. ``col``, ``set`` and ``weight`` are the
#: solver's ``Int32``; a label stays ``Int64``, a position in the pre-mask
#: coordinate product, which can pass 2^31.
_DTYPES = {
    'col': pl.Int32, 'row': pl.Int64,
    'lb': pl.Float64, 'ub': pl.Float64, 'rhs': pl.Float64, 'coeff': pl.Float64,
    'sense': SENSE, 'vtype': pl.Enum(get_args(program.VariableDomain)),
    'set': pl.Int32, 'type': pl.UInt8, 'weight': pl.Int32,
    'col_l': pl.Int32, 'col_r': pl.Int32,
}  # fmt: skip


@dataclass
class Measured:
    """What one build measured about itself, and a rebuild replaces wholesale.

    It outlives ``close()``, so it holds counts and small frames, never a read
    of the model. Most of it is written as measured, so a build that raises
    still reports what it got to.
    """

    #: ``name -> (coordinates, rows)`` for each parameter attached short of the
    #: coordinates its dims reach.
    sparse: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: ``name -> rows not built``, because every term they had vanished.
    omitted: dict[str, int] = field(default_factory=dict)
    #: ``name -> (smallest, largest)`` coefficient magnitude, per constraint block.
    coefficients: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: ``name -> (smallest, largest)`` bound magnitude, per variable block.
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: ``name -> (smallest, largest)`` right-hand-side magnitude, per constraint
    #: block, over the rows that survived.
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
    #: One [`Labelled`][specsolve._engine.labels.Labelled] per
    #: declaration, one map per label space: a variable and a constraint may share a name.
    variables: dict[str, labels.Labelled]
    constraints: dict[str, labels.Labelled]
    handoff: sinks.Handoff


@dataclass(frozen=True)
class _Constraint:
    """One constraint compiled, with everything moved to its left: linear, quadratic and constant terms, each with its sign.

    *restrictions* are the presences its rows must lie in, read off its terms.
    """

    name: str
    declaration: program.Constraint
    linear: list[tuple[Term, float]]
    quadratic: list[tuple[Term, float]]
    constant: list[tuple[Term, float]]
    restrictions: list[Presence]

    @classmethod
    def compiled(cls, compiler: Compiler, name: str, c: program.Constraint) -> _Constraint:
        """*c*'s two sides compiled, the right-hand side negated."""
        admits = declares_quadratic(c)
        lhs = compiler.expression(c.lhs, f"constraint '{name}' lhs", admits_quadratic=admits)
        rhs = compiler.expression(c.rhs, f"constraint '{name}' rhs", admits_quadratic=admits)
        linear = [(p, 1.0) for p in lhs.linear] + [(p, -1.0) for p in rhs.linear]
        quadratic = [(p, 1.0) for p in lhs.quadratic] + [(p, -1.0) for p in rhs.quadratic]
        constant = [(p, 1.0) for p in rhs.constant] + [(p, -1.0) for p in lhs.constant]
        restrictions = absence_restrictions([p for p, _ in (*linear, *quadratic, *constant)])
        return cls(name, c, linear, quadratic, constant, restrictions)

    @property
    def subject(self) -> str:
        return f"constraint '{self.name}'"

    @property
    def domain(self) -> labels.Domain:
        return labels.Domain(self.declaration.dims, self.declaration.where, self.restrictions)


class Assembly:
    """One build in progress: the mutable half, discarded once [`run`][] has frozen it."""

    def __init__(self, program: program.Program, attached: AttachedSources, measured: Measured) -> None:
        self.program = program
        self.attached = attached
        self.measured = measured
        self.variables: dict[str, labels.Labelled] = {}
        self.constraints: dict[str, labels.Labelled] = {}
        self.scope = Scope(program, attached, self.variables)
        self.compiler = Compiler(self.scope)
        self.n_cols = 0
        #: How many special-ordered sets have been numbered, dense ``0..n-1`` across the model.
        self.n_sets = 0
        self.quad: pl.DataFrame | None = None
        self.obj_const = 0.0
        self.obj_sense: ObjectiveSense | None = None

    def run(self) -> BuiltModel:
        """Build every declaration, then freeze what they produced.

        Quadratic constraints build last, so their rows are a contiguous tail
        a sink takes as a slice; the sort is stable, so file order survives in
        each half. Each constraint builds on rows of its own from 0, and its
        block is moved onto the model's rows here, once every block knows which
        of its rows it kept. The matrix and ``rows`` leave in ``(row, col)``
        order, as ``Handoff`` promises its sinks. A build whose [`_reach`][] is
        at most [`IN_MEMORY_REACH`][] collects [`in_memory`][].
        """
        with in_memory(self._reach() <= IN_MEMORY_REACH):
            cols = self._build_variables()
            sets = [self._build_sos(s, self.program.variables[s.variable]) for s in self.program.sos.values()]
            ordered = sorted(self.program.constraints.items(), key=lambda item: declares_quadratic(item[1]))
            built = self._build_constraints(ordered)
            objective = self._build_objective(self.program.objective)

        offsets = list(itertools.accumulate((rows.height for _, rows, _, _ in built), initial=0))
        row_count = offsets[-1]
        for (name, _), (frame, rows, _, _), offset in zip(ordered, built, offsets, strict=False):
            self.constraints[name] = labels.Labelled(_moved(frame, 'row', offset).lazy(), offset, rows.height)
        stacked = _on_model_rows([m for _, _, m, _ in built], offsets, _MATRIX)
        matrix_starts = _row_starts(stacked, row_count)
        matrix = stacked.select('col', 'coeff').rechunk()

        self.measured.columns = self.n_cols
        self.measured.rows = row_count
        self.measured.nonzeros = matrix.height
        handoff = sinks.Handoff(
            cols=_stack(cols, _COLS),
            obj=_stack([objective] if objective is not None else [], _OBJ),
            quad=_stack([] if self.quad is None else [self.quad], _QUAD),
            qmatrix=_on_model_rows([q for _, _, _, q in built], offsets, _QMATRIX),
            rows=_on_model_rows([r for _, r, _, _ in built], offsets, _ROWS),
            matrix=matrix,
            sos=_stack(sets, _SOS),
            row_starts=matrix_starts,
            column_count=self.n_cols,
            row_count=row_count,
            objective_sense=self.obj_sense,
            objective_constant=self.obj_const,
        )
        return BuiltModel(self.program, self.attached, self.variables, self.constraints, handoff)

    def _reach(self) -> int:
        """The columns or the rows the model would have with no ``where``, whichever is more: known before the build.

        A constraint over dims no variable spans has more rows than there are
        columns, so the rows count too.
        """
        cardinality = self.attached.cardinality
        return max(
            sum(math.prod(cardinality[d] for d in declared.dims) for declared in declarations.values())
            for declarations in (self.program.variables, self.program.constraints)
        )

    def _matrix_share(
        self, stacked: pl.DataFrame, name: str, *expressions: program.Expression
    ) -> tuple[pl.DataFrame, pl.Series]:
        """One constraint's share: in ``(row, col)`` order, repeated cells summed.

        Returns:
            The share, and the rows that had any term, read before the prune: a
            row whose every coefficient is zero is not a row with no terms.
        """
        coverage.refuse_null_coefficients(stacked, name, *expressions)
        share, dropped = _collapsed(stacked, ('row', 'col'), ordered=True)
        return share, stacked.get_column('row').unique() if dropped else _ordered_rows(share)

    # ------------------------------------------------------------------
    # declarations
    # ------------------------------------------------------------------

    def _build_variables(self) -> list[pl.DataFrame]:
        """Every variable's labelled frame, and its share of ``cols``, in declaration order.

        The labels of every variable are read in one pass and their bounds in
        another ([`labels.frames`][specsolve._engine.labels.frames]).
        A share leaves in label order: ``cols`` carries no ``col``, so a row's
        position is its solver column index. The bounds are collected in
        memory: they are parameters looked up onto labels already in memory,
        and polars 2.0's streaming engine runs that ordered join 4.6 to 5.5
        times slower than its in-memory one on the nodal benchmark at sizes m
        and l (#1857).
        """
        declared = self.program.variables
        domains = [labels.Domain(v.dims, v.where) for v in declared.values()]
        for name, labelled in zip(declared, labels.frames(self.scope, domains, 'var_label'), strict=True):
            self.variables[name] = labels.Labelled(
                _moved(labelled, 'var_label', self.n_cols).lazy(), self.n_cols, labelled.height
            )
            self.n_cols += labelled.height

        with in_memory():
            bounds = collected_all(
                self.compiler.bounds(self.variables[name].frame, name, v).select('var_label', *_bounds(v))
                for name, v in declared.items()
            )
        return [
            self._columns(name, v, labels.in_position_order(bounded, 'var_label'))
            for (name, v), bounded in zip(declared.items(), bounds, strict=True)
        ]

    def _columns(self, name: str, v: program.Variable, bounded: pl.DataFrame) -> pl.DataFrame:
        """One variable's share of ``cols``, from its bounds in label order.

        Raises:
            DataError: A bound with no value at a column.
        """
        cols = bounded.select('lb', 'ub', pl.lit(v.domain, dtype=_DTYPES['vtype']).alias('vtype'))
        if bounded.get_column('lb').null_count() or bounded.get_column('ub').null_count():
            bad = cols.filter(pl.col('lb').is_null() | pl.col('ub').is_null()).height
            raise DataError(null_bounds_message(name, bad))

        if (spread := _magnitude_range(bounded, 'lb', 'ub')) is not None:
            self.measured.bounds[name] = spread
        return cols

    def _build_sos(self, s: program.Sos, v: program.Variable) -> pl.DataFrame:
        """One declaration's sets as ``(set, type, col, weight)``, over *v*.

        A set and a weight are the two halves of a coordinate's row-major
        position, split at the ``along`` dim; a masked variable renumbers the
        sets densely. The stream leaves grouped by set and ascending in weight,
        so a sink reads a set's edges off the neighbouring row.
        """
        held = self.variables[s.variable]
        cardinality = self.scope.data.cardinality
        stride = math.prod(cardinality[d] for d in v.dims[v.dims.index(s.along) + 1 :])
        span = cardinality[s.along] * stride

        if v.where is None:
            frame = pl.select(pl.int_range(held.height, dtype=pl.Int64).alias('#position')).lazy()
            place, col = pl.col('#position'), (pl.col('#position') + held.start)
        else:
            frame = held.frame
            place, col = self.scope.row_major(v.dims, self.scope.ordinal_of), pl.col('var_label')
        placed = frame.select(
            ((place // span) * stride + place % stride).alias('#set position'),
            ((place // stride) % cardinality[s.along] + 1).cast(_DTYPES['weight']).alias('weight'),
            col.cast(_DTYPES['col']).alias('col'),
        ).pipe(collected)

        position = pl.col('#set position')
        grouped = placed if placed.get_column('#set position').is_sorted() else placed.sort('#set position', 'weight')
        dense = position if v.where is None else (position != position.shift(1)).fill_null(True).cum_sum() - 1
        built = grouped.select(
            (dense + self.n_sets).cast(_DTYPES['set']).alias('set'),
            pl.lit(s.sos_type, dtype=_DTYPES['type']).alias('type'),
            'col',
            'weight',
        )
        if built.height:
            self.n_sets = built.item(-1, 'set') + 1
        return built

    def _build_constraints(
        self, ordered: list[tuple[str, program.Constraint]]
    ) -> list[tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame | None]]:
        """Each constraint as [`_block`][] gives it, in *ordered*'s order.

        Every constraint's labels are read in one pass, its constant side in a
        second and its linear terms in a third
        ([`labels.frames`][specsolve._engine.labels.frames]). The
        [`coverage`][] checks run in each table's order, before an aggregate
        can hide a hole. Each constraint's raw terms are let go as soon as its
        block has summed them: held through every block, they raised the peak
        memory of the network benchmark at 10M variables by about a fifth (#2022).
        """
        constraints = [_Constraint.compiled(self.compiler, name, c) for name, c in ordered]
        frames = labels.frames(self.scope, [con.domain for con in constraints], 'row')
        for con, frame in zip(constraints, frames, strict=True):
            c, constant = con.declaration, [p for p, _ in con.constant]
            if con.restrictions and (declared := labels.declared_height(self.scope, c.dims, c.where)) > frame.height:
                self.measured.omitted[con.name] = self.measured.omitted.get(con.name, 0) + declared - frame.height
            divisors = program.parameters_of(*coverage.divisors_of(c.lhs, c.rhs))
            coverage.refuse_null_constants(coverage.narrowed_to_rows(frame.lazy(), constant), divisors, con.subject)
        sides = collected_all(
            coverage.constant_side(self.scope, frame.lazy(), con.constant, con.declaration)
            for con, frame in zip(constraints, frames, strict=True)
        )
        linear = collected_all(self._linear(con, frame) for con, frame in zip(constraints, frames, strict=True))
        return [
            self._block(con, frame, coverage.refuse_uncovered(side, con.constant, con.subject), linear.pop(0))
            for con, frame, side in zip(constraints, frames, sides, strict=True)
        ]

    def _linear(self, con: _Constraint, frame: pl.DataFrame) -> pl.LazyFrame:
        """*con*'s linear terms as ``(row, col, coeff)``, on its own rows; no rows where it has none."""
        if not con.linear:
            return pl.LazyFrame(schema={name: _DTYPES[name] for name in _MATRIX})
        c = con.declaration
        carried_order: MaintainOrderJoin | None = 'left_right' if len(con.linear) == 1 else None
        dense = c.where is None and not con.restrictions
        frames = []
        for p, sign in con.linear:
            if dense and set(p.dims) == set(c.dims):
                placed = p.frame.with_columns(self._dense_row(c.dims).alias('row'))
            else:
                placed = join_on(frame.lazy(), p.frame, p.dims, 'inner', maintain_order=carried_order)
            frames.append(
                placed.select(
                    'row',
                    pl.col('var_label').cast(_DTYPES['col']).alias('col'),
                    (sign * pl.col('coeff')).cast(pl.Float64).alias('coeff'),
                )
            )
        return pl.concat(frames)

    def _block(
        self, con: _Constraint, frame: pl.DataFrame, rows: pl.DataFrame, stacked: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
        """One constraint as its labelled block, its ``rows``, its share of the matrix and its quadratic share.

        All four are on the block's own rows, numbered from 0, which
        [`run`][] moves onto the model's. A row is built where either matrix
        has a term, and the labelled block narrows to the rows that survived.
        """
        c = con.declaration
        matrix, term_rows = self._matrix_share(stacked, con.subject, c.lhs, c.rhs)
        qmatrix = self._quadratic_share(frame, con.quadratic, con.name, c)
        if qmatrix is not None:
            term_rows = pl.concat([term_rows, qmatrix.get_column('row').unique()]).unique()
        frame, rows, matrix, qmatrix = self._drop_termless_rows(con.name, frame, rows, matrix, qmatrix, term_rows)
        if (spread := _magnitude_range(matrix, 'coeff')) is not None:
            self.measured.coefficients[con.name] = spread
        if (sides := _magnitude_range(rows, 'rhs')) is not None:
            self.measured.rhs[con.name] = sides
        return frame, rows, matrix, qmatrix

    def _dense_row(self, dims: tuple[str, ...]) -> pl.Expr:
        """The row a coordinate lands on in a block built over the whole product of *dims*.

        Such a block numbers its rows row-major over the declared ordinals
        ([`labels.frames`][specsolve._engine.labels.frames]), so a term
        computes its row from its own coordinate rather than joining the block.
        The door admits no label outside its dimension, so every coordinate a
        term carries is a row of the block, as the join would find.
        """
        return self.scope.row_major(dims, self.scope.ordinal_of)

    def _quadratic_share(
        self, frame: pl.DataFrame, quadratic: list[tuple[Term, float]], name: str, c: program.Constraint
    ) -> pl.DataFrame | None:
        """One constraint's quadratic entries as ``(row, col_l, col_r, coeff)``, in that order.

        Pairs are ordered by column index for [`_objective_quadratic`][]'s
        reason.
        """
        if not quadratic:
            return None
        frames = [
            join_on(frame.lazy(), p.frame, p.dims, 'inner').select(
                'row',
                *_ordered_pair(),
                (sign * pl.col('coeff')).cast(pl.Float64).alias('coeff'),
            )
            for p, sign in quadratic
        ]
        stacked = pl.concat(frames).pipe(collected)
        coverage.refuse_null_coefficients(stacked, f"constraint '{name}'", c.lhs, c.rhs)
        share, _ = _collapsed(stacked, ('row', 'col_l', 'col_r'), ordered=True)
        return share

    def _drop_termless_rows(
        self,
        name: str,
        frame: pl.DataFrame,
        rows: pl.DataFrame,
        matrix: pl.DataFrame,
        qmatrix: pl.DataFrame | None,
        kept: pl.Series,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
        """Rows that kept no variable term are not built, and the block closes up: the labelled block, ``rows`` and both shares.

        *kept* is read before zeros are pruned, so a row whose every term is
        zero stays: ``0 >= 10`` is infeasible. Survivors renumber from 0,
        because the dual read-back reads a block by position. Every row a share
        holds is in *kept*, so each of its entries moves with its row.
        """
        if kept.len() == rows.height:
            return frame, rows, matrix, qmatrix

        surviving = rows.filter(pl.col('row').is_in(kept)).sort('row')
        renumber = surviving.select('row').with_row_index('__new__')
        self.measured.omitted[name] = self.measured.omitted.get(name, 0) + rows.height - surviving.height
        remap = dict(zip(renumber.get_column('row'), renumber.get_column('__new__'), strict=True))
        renumbered = pl.col('row').replace_strict(remap, return_dtype=_DTYPES['row'])
        rows = surviving.with_columns(renumbered)
        matrix = matrix.with_columns(renumbered)
        if qmatrix is not None:
            qmatrix = qmatrix.with_columns(renumbered)
        frame = frame.filter(pl.col('row').is_in(kept)).with_columns(renumbered)
        return frame, rows, matrix, qmatrix

    def _build_objective(self, o: program.Objective | None) -> pl.DataFrame | None:
        """The objective as ``(col, coeff)``, or ``None`` if it has no terms.

        With no objective declared the sense stays ``min`` and the constant
        ``0``. Rows landing on one column are summed here, because the hand-off
        scatter keeps only the last write.
        """
        if o is None:
            return None
        comp = self.compiler.expression(o.expression, 'objective', admits_quadratic=True)
        coverage.refuse_null_constants(
            [p.frame for p in comp.constant], program.parameters_of(*coverage.divisors_of(o.expression)), 'objective'
        )
        for p in comp.constant:
            assert not p.dims, (
                f'objective constant term has dims {list(p.dims)} — the language refuses a '
                f'variable-free term of an objective that carries any'
            )
            self.obj_const += p.frame.select(pl.col('cval').sum()).pipe(collected).item() or 0.0
        self.obj_sense = o.sense
        self.quad = self._objective_quadratic(comp.quadratic, o.expression)
        if not comp.linear:
            return None
        frames = [
            p.frame.select(pl.col('var_label').cast(_DTYPES['col']).alias('col'), pl.col('coeff')) for p in comp.linear
        ]
        stacked = pl.concat(frames).pipe(collected)
        coverage.refuse_null_coefficients(stacked, 'objective', o.expression)
        objective, _ = _collapsed(stacked, ('col',), ordered=False, space=self.n_cols)
        self.measured.objective_range = _magnitude_range(objective, 'coeff')
        return objective

    def _objective_quadratic(self, quadratic: tuple[Term, ...], expression: program.Expression) -> pl.DataFrame | None:
        """The objective's quadratic part as ``(col_l, col_r, coeff)``, or ``None``.

        One row per unordered pair, at the coefficient the file wrote, not
        halved; each sink converts. It leaves sorted, because
        [`structure`][specsolve._sinks.handoff.Handoff.structure] hashes it.
        """
        if not quadratic:
            return None
        frames = [p.frame.select(*_ordered_pair(), pl.col('coeff')) for p in quadratic]
        stacked = pl.concat(frames).pipe(collected)
        coverage.refuse_null_coefficients(stacked, 'objective', expression)
        quad, _ = _collapsed(stacked, ('col_l', 'col_r'), ordered=True)
        return quad


def short_parameters(program: program.Program, attached: AttachedSources) -> dict[str, tuple[int, int]]:
    """Which parameters arrived short, and by how much: ``name -> (reach, rows)``.

    The door refuses duplicates and strangers, so a parameter's height is the
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
        pl.min_horizontal('var_label', 'var_label_2').cast(_DTYPES['col_l']).alias('col_l'),
        pl.max_horizontal('var_label', 'var_label_2').cast(_DTYPES['col_r']).alias('col_r'),
    )


def _bounds(v: program.Variable) -> tuple[pl.Expr, pl.Expr]:
    """*v*'s ``lb`` and ``ub`` as floats, each a whole number where *v* is not continuous ([`_whole`][])."""
    lb, ub = pl.col('lb').cast(pl.Float64), pl.col('ub').cast(pl.Float64)
    if v.domain != 'continuous':
        lb, ub = _whole(lb, lb.ceil()), _whole(ub, ub.floor())
    return lb.alias('lb'), ub.alias('ub')


def _whole(bound: pl.Expr, inward: pl.Expr) -> pl.Expr:
    """An integer column's *bound* as a whole number: the nearest one within 1e-6, else *inward*, its rounding into the range.

    HiGHS answers a fractional bound on an integer column as optimal with no
    feasible values, so every sink is handed whole bounds. The 1e-6 is HiGHS's
    default ``mip_feasibility_tolerance``: a bound that noise put just past a
    whole number keeps that number rather than moving a whole step.
    """
    nearest = bound.round()
    return pl.when((bound - nearest).abs() <= 1e-6).then(nearest).otherwise(inward)


def _magnitude_range(frame: pl.DataFrame, *columns: str) -> tuple[float, float] | None:
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
    answered = frame.select(sides).row(0, named=True)
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
    Unordered, a repeat is probed over *space*, the dense label count.
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
        assert space is not None and len(keys) == 1, 'an unordered probe counts one dense integer key'
        repeated = _repeats_a_label(stacked.get_column(keys[0]), space)
    if not repeated:
        return stacked, dropped
    aggregated = stacked.lazy().group_by(*keys).agg(pl.col('coeff').sum())
    if ordered:
        aggregated = aggregated.sort(*keys)
    summed = aggregated.pipe(collected)
    share = _pruned(summed)
    return share, dropped or share.height != summed.height


def _sorted(stacked: pl.DataFrame, keys: tuple[str, ...]) -> pl.DataFrame:
    """*stacked* sorted by *keys*, on one integer that orders as they do where one can hold them all.

    The keys are labels, never negative, so each is a digit in a base one past
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
    """The distinct ``row`` labels of a matrix already ordered by ``row``.

    ``set_sorted`` is an assertion, not a check: an unordered column silently
    loses rows, so call it only on what [`_collapsed`][] returned ordered.
    """
    return matrix.get_column('row').set_sorted().unique()


def _repeats_a_label(labels: pl.Series, count: int) -> bool:
    """Whether any of *labels* occurs twice, over a dense ``0..count-1`` space.

    *count* must exceed every label; a stale one raises out of bounds rather
    than answering wrong.
    """
    seen = np.zeros(count, dtype=bool)
    seen[labels.to_numpy()] = True
    return int(np.count_nonzero(seen)) != labels.len()


def _pruned(matrix: pl.DataFrame) -> pl.DataFrame:
    """*matrix* without its exactly-zero coefficients — unchanged, and not rechunked, when it has none.

    A pruned share no longer says which rows had terms, so
    [`Assembly._matrix_share`][] reads that set before pruning.
    """
    if not matrix.select(pl.col('coeff').eq(0).any()).item():
        return matrix
    return matrix.filter(pl.col('coeff') != 0).rechunk()


def _row_starts(ordered: pl.DataFrame, row_count: int) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
    """Each row's first entry in the row-ordered *ordered* — CSR's own index.

    *ordered* must ascend in ``row``: a row in two runs silently gets wrong spans.
    """
    runs = ordered['row'].rle()
    starts = np.zeros(row_count + 1, dtype=np.int64)
    starts[runs.struct.field('value').to_numpy() + 1] = runs.struct.field('len').to_numpy()
    return np.cumsum(starts, out=starts)


def _on_model_rows(blocks: list[pl.DataFrame | None], offsets: Sequence[int], columns: tuple[str, ...]) -> pl.DataFrame:
    """*blocks*, each on its constraint's own rows, stacked onto the model's in row order: block *i* moves on by ``offsets[i]``.

    A constraint without such a share is ``None``.
    """
    moved = [_moved(block, 'row', offset) for block, offset in zip(blocks, offsets, strict=False) if block is not None]
    return labels.in_position_order(_stack(moved, columns), 'row')


def _moved(frame: pl.DataFrame, column: str, offset: int) -> pl.DataFrame:
    """*frame* with *offset* added to *column*, keeping its sorted flag.

    Added to the Series, because a polars query costs a fixed time a small
    frame never earns back.
    """
    if not offset:
        return frame
    given = frame.get_column(column)
    moved = given + offset
    if given.flags['SORTED_ASC']:
        moved = moved.set_sorted()
    return pl.DataFrame([moved if c.name == column else c for c in frame.get_columns()])


def _stack(frames: list[pl.DataFrame], columns: tuple[str, ...]) -> pl.DataFrame:
    """Concatenate *frames*, or an empty frame of *columns* when there are none."""
    if frames:
        return pl.concat(frames)
    return pl.DataFrame(schema={name: _DTYPES[name] for name in columns})
