"""What every sink reads, and nothing more.

Four frames plus the scalars a writer needs to size its batching, and the
projections more than one sink needs — the dense column and row vectors, the
matrix a block at a time. Those belong to the contract rather than to either
solver, so two sinks cannot disagree about the model they loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, get_args

import polars as pl
from mathspec import program

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import numpy as np

#: The arrays a sink hands a solver. ``numpy.typing.NDArray`` leaves the shape
#: parameter ``Any``, and nothing here reads a rank, so these name the dtype
#: and say only that the shape is a tuple of ints.
type Floats = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type Ints = np.ndarray[tuple[int, ...], np.dtype[np.int64]]
type Bools = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]


@dataclass(frozen=True)
class ColumnVectors:
    """The per-column vectors a solver sink is handed.

    Three float vectors and an integrality mask, each as long as the model
    has columns.
    """

    lb: Floats
    ub: Floats
    cost: Floats
    integral: Bools


@dataclass(frozen=True)
class RowVectors:
    """A sense code and a right-hand side, each as long as the model has rows."""

    sense: np.ndarray[tuple[int, ...], np.dtype[np.uint8]]
    rhs: Floats


@dataclass(frozen=True)
class MatrixBlock:
    """One chunk of rows ``[lo, hi)`` and the matrix entries those rows own.

    ``starts`` is the offset of each row's entries within the chunk — what a
    solver's matrix API asks for. A row with no entries takes the next row's
    offset, and so occupies no span.
    """

    lo: int
    hi: int
    entries: pl.DataFrame
    starts: Ints

    @property
    def height(self) -> int:
        """How many rows the chunk spans — entries or not."""
        return self.hi - self.lo


#: ``sense`` as a number. The vocabulary is the language's; the order is
#: arbitrary and shared, since a solver indexes its own spelling with these.
SENSE_CODES: Mapping[program.ConstraintSense, int] = {
    sense: code for code, sense in enumerate(get_args(program.ConstraintSense))
}

#: The dtype the ``rows`` frame holds a comparison in. Built from
#: [`SENSE_CODES`][] so a category's index *is* its code.
SENSE = pl.Enum(list(SENSE_CODES))


@dataclass(frozen=True)
class Handoff:
    r"""The built model, as a sink sees it.

    ``cols`` (lb, ub, vtype), ``obj`` (col, coeff), ``rows`` (row, sense, rhs)
    and ``matrix`` in CSR: ``(col, coeff)`` in row-major order, with
    ``row_starts[r] : row_starts[r + 1]`` the half-open span row ``r`` owns.
    The objective constant lives outside the frames, having no column to
    attach to.

    ``quad`` is the objective's quadratic part, in ``(col_l, col_r)`` order,
    one row per **unordered pair** of columns: the objective contains ``coeff · x[col_l] · x[col_r]``, whole.
    Three sinks spell that three ways — a Hessian is :math:`\frac12 x^\top Q x`,
    the LP section is divided by two, Gurobi takes :math:`x^\top Q x` — so what
    arrives here is the algebra and the conversion belongs to whoever loads it.
    Empty for every affine model, which is nearly all of them.

    ``qmatrix`` is the same form for the *rows* that carry one:
    ``(row, col_l, col_r, coeff)`` in that order. The rows it names are a
    **contiguous tail** of the label space — quadratic is a property of a
    declaration, so the engine builds those last — which lets a sink holding
    linear and quadratic rows in different objects read its answer back as two
    runs rather than a scatter, beginning at [`linear_row_count`][].

    ``sos`` is the fifth stream and the one that lands unevenly: ``(set, type,
    col, weight)`` in ``(set, weight)`` order, one row per member, empty for
    the models that declare none. It is the only frame a sink may be unable to
    ingest — SOS is a *sink capability*, not a property of the model — so a
    solver without the concept states so and refuses a model carrying one.

    ``cols``, ``rows`` and ``matrix`` all arrive in the solver's own order —
    ``cols`` by column, the other two by row — which is what lets every dense
    vector be read positionally rather than keyed.

    ``col`` and ``row`` are dense ``0..n-1``, so they *are* the solver's own
    indices and no sink builds a mapping. **``cols`` carries no ``col`` and
    ``matrix`` no ``row``**: a ``cols`` row's position is its index and a
    matrix entry's row is where it sits between two starts, which is what a
    solver's matrix API takes. [`matrix_block`][] spells them back out for the
    one consumer that renders them. ``obj`` keeps its ``col``, being genuinely
    sparse, and **carries no order contract at all**: its rows arrive in
    whatever order collapsing them produced, which differs between two builds
    of one model. Every consumer reads it scattered over the column index
    ([`dense_columns`][]), so nothing downstream can tell — and anything new
    that reads it must scatter too rather than read it in place.
    """

    cols: pl.DataFrame
    obj: pl.DataFrame
    quad: pl.DataFrame
    qmatrix: pl.DataFrame
    rows: pl.DataFrame
    matrix: pl.DataFrame
    sos: pl.DataFrame
    row_starts: Ints
    column_count: int
    row_count: int
    #: ``None`` where the file declares no objective — a feasibility problem,
    #: which asks whether the constraints can be met and has no direction to
    #: be optimised in. A sink whose format needs a keyword anyway picks one
    #: at its own edge over an empty objective, where every direction agrees.
    objective_sense: program.ObjectiveSense | None
    objective_constant: float

    def _spans(self, budget: int | None) -> Iterator[tuple[int, int]]:
        """The row ranges a block reader walks — one rule, for both of them.

        Width is the average row, since a reader pays in nonzeros: 100k rows is
        900k entries in one model and 10M in another.

        ``budget=None`` is one span.
        """
        linear = self.linear_row_count
        if budget is None:
            return iter([(0, linear)])
        return ranges(linear, budget, self.matrix.height / max(1, self.row_count))

    def _span(self, lo: int, hi: int) -> pl.DataFrame:
        """The matrix entries rows ``[lo, hi)`` own — the CSR arithmetic, once.

        Both block readers slice through here, so how a span is located, and
        the half-open ``hi`` bound, cannot drift between them.
        """
        first = int(self.row_starts[lo])
        return self.matrix.slice(first, int(self.row_starts[hi]) - first)

    @cached_property
    def linear_row_count(self) -> int:
        """How many rows a sink may load as ordinary linear constraints.

        Where the quadratic tail starts, or the whole row count for a model
        with none.
        """
        return int(self.qmatrix['row'][0]) if self.qmatrix.height else self.row_count

    def dense_columns(self, infinity: float) -> ColumnVectors:
        """The column vectors over the solver's index, ready to hand over.

        *infinity* is the solver's own spelling of an absent bound, so the
        vectors come back ready to hand over unedited. ``cols`` arrives one row
        per column in ``col`` order, so its three vectors are the frame's own;
        only ``cost`` is scattered, ``obj`` being sparse, and a variable in no
        objective term costs zero. Every vector returned is freshly produced —
        nothing aliases the built model.
        """
        prepared = self.cols.select(
            _finite(pl.col('lb'), infinity).alias('lb'),
            _finite(pl.col('ub'), infinity).alias('ub'),
            (pl.col('vtype') != 'continuous').alias('integral'),
        )
        cost = _scattered(self.column_count, self.obj['col'].to_numpy(), self.obj['coeff'].to_numpy(), 0.0)
        return ColumnVectors(
            lb=prepared['lb'].to_numpy(),
            ub=prepared['ub'].to_numpy(),
            cost=cost,
            integral=prepared['integral'].to_numpy(),
        )

    def dense_rows(self, infinity: float) -> RowVectors:
        """The row vectors over the solver's row index, ready to hand over.

        The row half of [`dense_columns`][], so a chunk of rows is a slice
        rather than a search. It stops at the sense, a [`SENSE_CODES`][]
        byte, because that is where the solvers part — HiGHS wants
        ``lower``/``upper``, the others a comparison and right-hand side. A
        row with no entry gets a comparison nothing can fail (``>=`` against
        ``-infinity``) rather than the ``== 0`` that would be an equality the
        model never stated.

        ``rows`` leaves the build in row order, so a frame holding a row per
        label is both vectors already; the scatter is for one that falls short.
        """
        sided = self.rows.select(
            'row',
            pl.col('sense').to_physical().cast(pl.UInt8).alias('op'),
            'rhs',
        )
        if sided.height == self.row_count:
            return RowVectors(sense=sided['op'].to_numpy(), rhs=sided['rhs'].to_numpy())
        at = sided['row'].to_numpy()
        return RowVectors(
            sense=_scattered(self.row_count, at, sided['op'].to_numpy(), SENSE_CODES['>=']),
            rhs=_scattered(self.row_count, at, sided['rhs'].to_numpy(), -infinity),
        )

    @cached_property
    def structure(self) -> bytes:
        """A digest of everything a re-solve may **not** change.

        The question a loaded solver asks of a rebuilt model: may I keep what
        I hold and take the new numbers by value? Bounds, costs and right-hand
        sides go in that way; the counts, the matrix, each row's comparison,
        each column's type and every SOS member do not, so a model whose
        digest moved has to be loaded again.

        **A quadratic *constraint* is structure whole** — coefficients and
        right-hand side — where the quadratic *objective* contributes only its
        pattern. A model whose quadratic row moved at all is loaded again.

        **The quadratic objective contributes its pattern and not its values.**
        A pair that appeared or moved is a model to load again; a coefficient
        that merely changed is pushed.

        **A set is structure even though nothing about it is a coefficient.**

        Every vector read has an order contract — the label-ordered columns,
        the row-ordered matrix and rows — so two builds of one model agree.
        """
        import hashlib

        import numpy as np

        digest = hashlib.blake2b(digest_size=16)
        digest.update(f'{self.column_count} {self.row_count} {self.objective_sense}'.encode())
        for vector in (
            self.cols['vtype'].to_physical().to_numpy(),
            self.quad['col_l'].to_numpy(),
            self.quad['col_r'].to_numpy(),
            self.qmatrix['row'].to_numpy(),
            self.qmatrix['col_l'].to_numpy(),
            self.qmatrix['col_r'].to_numpy(),
            self.qmatrix['coeff'].to_numpy(),
            self.rows.filter(pl.col('row') >= self.linear_row_count)['rhs'].to_numpy(),
            self.rows['sense'].to_physical().to_numpy(),
            self.matrix['col'].to_numpy(),
            self.matrix['coeff'].to_numpy(),
            self.row_starts,
            *(self.sos[column].to_numpy() for column in self.sos.columns),
        ):
            digest.update(np.ascontiguousarray(vector).data)
        return digest.digest()

    @cached_property
    def contents(self) -> str:
        """A digest of the built model **whole** — the numbers included.

        [`structure`][]'s counterpart, and the one question a saved answer
        asks of a model rebuilt later: is this the model I answered? So it
        covers what ``structure`` leaves out on purpose — the bounds, the costs
        and the right-hand sides a re-solve may push — because a pushed number
        is a different answer even where it is the same matrix.

        Over the built model rather than over the sources, so two source
        mappings that a build cannot tell apart agree here, whatever shape or
        encoding they arrived in. Order is the build's own: a source whose rows
        moved builds a different label order and so digests differently.

        **The objective is read through the dense cost vector**, not off
        ``obj``: that frame is genuinely sparse and carries no order contract,
        so two builds of one model lay its rows out differently and hashing
        them in place would call one model two.

        Read on first ask and kept, the frames being immutable.
        """
        import hashlib

        import numpy as np

        digest = hashlib.blake2b(digest_size=16)
        digest.update(self.structure)
        digest.update(f'{self.objective_constant}'.encode())
        for vector in (
            self.cols['lb'].to_numpy(),
            self.cols['ub'].to_numpy(),
            _scattered(self.column_count, self.obj['col'].to_numpy(), self.obj['coeff'].to_numpy(), 0.0),
            self.quad['coeff'].to_numpy(),
            self.rows['row'].to_numpy(),
            self.rows['rhs'].to_numpy(),
        ):
            digest.update(np.ascontiguousarray(vector).data)
        return digest.hexdigest()

    def sets(self) -> Iterator[tuple[int, pl.Series, pl.Series]]:
        """Each special-ordered set: its type, member columns, and weights.

        In declared ``(set, weight)`` order; the type is read off the first
        member, every member of a set carrying the same one. Nothing here is
        pushed on an update: a set is structure, so a model whose members moved
        is one [`structure`][] has already sent back to be loaded again.
        """
        for members in self.sos.partition_by('set', maintain_order=True):
            yield members.item(0, 'type'), members.get_column('col'), members.get_column('weight')

    def quadratic_blocks(self) -> Iterator[tuple[int, pl.DataFrame]]:
        """Each quadratic row and the ``(col_l, col_r, coeff)`` entries it owns.

        One row at a time, unlike the linear matrix: every API that takes a
        quadratic constraint takes one per call. They are the contiguous tail
        beginning at [`linear_row_count`][], so they arrive ascending and a
        sink's read-back stays two runs.
        """
        for (row,), entries in self.qmatrix.group_by('row', maintain_order=True):
            yield int(row), entries.select('col_l', 'col_r', 'coeff')

    def row_blocks(self, budget: int | None) -> Iterator[MatrixBlock]:
        """Each chunk of rows with the matrix entries it owns — every sink's reader.

        A chunk is a ``slice``: ``row_starts`` already says where every row's
        entries sit, so nothing is sorted and nothing is searched. A consumer
        that needs the ``row`` labels spelled back out asks
        [`matrix_block`][] with the chunk's own range, so its spans and
        entries cannot disagree.
        """
        for lo, hi in self._spans(budget):
            yield MatrixBlock(lo, hi, self._span(lo, hi), self.row_starts[lo:hi] - self.row_starts[lo])

    def matrix_block(self, lo: int, hi: int) -> pl.DataFrame:
        """Rows ``[lo, hi)`` of the matrix with their ``row`` labels spelled out.

        The adjoint of what CSR compressed — ``np.repeat`` walks the start
        offsets back into one label per entry.
        """
        import numpy as np

        labels = np.repeat(np.arange(lo, hi, dtype=np.int64), np.diff(self.row_starts[lo : hi + 1]))
        return self._span(lo, hi).with_columns(pl.Series('row', labels))


@dataclass(frozen=True)
class Run:
    """One declaration's contiguous run of columns or rows.

    Attributes:
        name: The declaration.
        start: Its first column or row.
        height: How many columns or rows it owns.
        dims: Its dims, in declaration order.
        coordinates: One row of labels per column or row, in solver order,
            one column per dim; no columns for a declaration without dims.
    """

    name: str
    start: int
    height: int
    dims: tuple[str, ...]
    coordinates: pl.DataFrame


@dataclass(frozen=True)
class SetRun:
    """One ``sos:`` declaration: the sets over one variable's columns.

    Attributes:
        name: The declaration.
        variable: The variable its sets order, which carries no other set.
        along: The position of the ordering dim among that variable's dims.
        sos_type: 1 or 2.
    """

    name: str
    variable: str
    along: int
    sos_type: program.SosType


@dataclass(frozen=True)
class Declared:
    """Which declaration, at which coordinate, owns each column, row and set, in solver order.

    What a sink reads beside the [`Handoff`][] when it names what it hands
    over, which the handoff's dense indices do not.
    """

    variables: Sequence[Run]
    constraints: Sequence[Run]
    sets: Sequence[SetRun]


def spelled_senses(spelling: Mapping[str, str]) -> np.ndarray[tuple[int, ...], np.dtype[np.str_]]:
    """[`SENSE_CODES`][] as one solver's spellings, indexed by code.

    A sense added to [`SENSE_CODES`][] and not to *spelling* raises instead.
    """
    import numpy as np

    out = np.empty(len(SENSE_CODES), dtype='<U1')
    for sense, code in SENSE_CODES.items():
        out[code] = spelling[sense]
    return out


def solver_vector(values: Any) -> pl.Series:  # pyrefly: ignore[explicit-any] — a solver hands back its own array type
    """One quantity a solver produced, in its own index — every sink's read-back.

    A series rather than a ``(label, value)`` frame: the read-back takes a
    declaration's share by slicing.
    """
    import numpy as np

    return pl.Series('value', np.asarray(values, dtype=np.float64))


def _finite(value: pl.Expr, infinity: float) -> pl.Expr:
    """*value* with each infinity as the finite sentinel the asking solver reads as one.

    Both substitutions in one expression. A ``NaN`` never arrives: the door
    refuses one in a parameter and the schema refuses one written in the file.
    """
    return (
        pl.when(value == float('inf'))
        .then(pl.lit(infinity))
        .when(value == float('-inf'))
        .then(pl.lit(-infinity))
        .otherwise(value)
    )


def _scattered(count: int, at: Ints, values: Any, absent: Any) -> Any:  # pyrefly: ignore[explicit-any] — the dtype is the model's
    """*values* written at the label each one belongs to, *absent* elsewhere."""
    import numpy as np

    dense = np.full(count, absent, dtype=values.dtype)
    dense[at] = values
    return dense


def ranges(total: int, budget: int, width: float) -> Iterator[tuple[int, int]]:
    """Half-open ``[lo, hi)`` ranges covering ``[0, total)``, each holding about ``budget`` elements.

    One unit costs ``width`` of them, and every caller states it — a row is
    its average nonzeros, a column is one. A ``width`` below 1 is read as 1.
    Empty input yields nothing rather than one empty range.
    """
    per_chunk = max(1, int(budget // max(1.0, width)))
    for lo in range(0, total, per_chunk):
        yield lo, min(lo + per_chunk, total)
