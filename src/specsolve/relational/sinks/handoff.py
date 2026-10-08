"""What every sink reads, and nothing more.

The frames, the scalars a writer needs to size its batching, and the
projections more than one sink needs — the dense column and row vectors, the
matrix a block at a time — so two sinks cannot disagree about the model they
loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, get_args

import polars as pl
from mathspec import program

if TYPE_CHECKING:
    import hashlib
    from collections.abc import Iterator, Mapping

    import numpy as np

#: The arrays a sink hands a solver. ``numpy.typing.NDArray`` leaves the shape
#: parameter ``Any``, and nothing here reads a rank, so these name the dtype
#: and say only that the shape is a tuple of ints.
type Floats = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type Ints = np.ndarray[tuple[int, ...], np.dtype[np.int64]]
type Bools = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]


@dataclass(frozen=True)
class ColumnVectors:
    """The per-column vectors a solver sink is handed, each as long as the model has columns."""

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
    one row per **unordered pair** of columns: the objective contains
    ``coeff · x[col_l] · x[col_r]``, whole. Three sinks spell that three ways —
    a Hessian is :math:`\frac12 x^\top Q x`, the LP section is divided by two,
    Gurobi takes :math:`x^\top Q x` — so the conversion belongs to whoever
    loads it. Empty for every affine model.

    ``qmatrix`` is the same form for the *rows* that carry one:
    ``(row, col_l, col_r, coeff)``. Its rows are a **contiguous tail** of the
    label space, beginning at [`linear_row_count`][] — the engine builds
    quadratic declarations last — so a sink holding linear and quadratic rows
    in different objects reads its answer back as two runs, not a scatter.

    ``sos`` is ``(set, type, col, weight)`` in ``(set, weight)`` order, one row
    per member. It is the only frame a sink may be unable to ingest — SOS is a
    *sink capability* — so a solver without the concept refuses a model
    carrying one.

    ``cols``, ``rows`` and ``matrix`` arrive in the solver's own order —
    ``cols`` by column, the other two by row — and ``col`` and ``row`` are
    dense ``0..n-1``, so they *are* the solver's indices: every dense vector is
    read positionally and no sink builds a mapping. **``cols`` carries no
    ``col`` and ``matrix`` no ``row``**: a ``cols`` row's position is its
    index, and a matrix entry's row is where it sits between two starts;
    [`matrix_block`][] spells them back out. ``obj`` keeps its ``col``, being
    sparse, and **carries no order contract**: its row order differs between
    two builds of one model, so anything that reads it must scatter it over the
    column index, as [`dense_columns`][] does.
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
        """The row ranges both block readers walk; ``budget=None`` is one span.

        Width is the average row, since a reader pays in nonzeros: 100k rows is
        900k entries in one model and 10M in another.
        """
        linear = self.linear_row_count
        if budget is None:
            return iter([(0, linear)])
        return ranges(linear, budget, self.matrix.height / max(1, self.row_count))

    def _span(self, lo: int, hi: int) -> pl.DataFrame:
        """The matrix entries rows ``[lo, hi)`` own.

        Both block readers slice through here, so the CSR arithmetic and the
        half-open ``hi`` bound cannot drift between them.
        """
        first = int(self.row_starts[lo])
        return self.matrix.slice(first, int(self.row_starts[hi]) - first)

    @cached_property
    def linear_row_count(self) -> int:
        """How many rows a sink may load as linear constraints: where the quadratic tail starts, or every row."""
        return int(self.qmatrix['row'][0]) if self.qmatrix.height else self.row_count

    def dense_columns(self, infinity: float) -> ColumnVectors:
        """The column vectors over the solver's index, ready to hand over unedited.

        *infinity* is the solver's own spelling of an absent bound. A variable
        in no objective term costs zero. Every vector returned is freshly
        produced — nothing aliases the built model.
        """
        prepared = self.cols.select(
            _finite(pl.col('lb'), infinity).alias('lb'),
            _finite(pl.col('ub'), infinity).alias('ub'),
            (pl.col('vtype') != 'continuous').alias('integral'),
        )
        return ColumnVectors(
            lb=prepared['lb'].to_numpy(),
            ub=prepared['ub'].to_numpy(),
            cost=self._dense_cost(),
            integral=prepared['integral'].to_numpy(),
        )

    def _dense_cost(self) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """The objective's linear coefficient per column, zero for a column in no term; ``obj`` is sparse and unordered."""
        return _scattered(self.column_count, self.obj['col'].to_numpy(), self.obj['coeff'].to_numpy(), 0.0)

    def dense_rows(self, infinity: float) -> RowVectors:
        """The row half of [`dense_columns`][], so a chunk of rows is a slice rather than a search.

        It stops at the sense, a [`SENSE_CODES`][] byte, because that is where
        the solvers part — HiGHS wants ``lower``/``upper``, the others a
        comparison and right-hand side. A row with no entry gets a comparison
        nothing can fail (``>=`` against ``-infinity``) rather than the
        ``== 0`` that would be an equality the model never stated.
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
        sides go in that way. The counts, the matrix, each row's comparison,
        each column's type, every SOS member, the quadratic objective's
        pattern and each quadratic *constraint* whole — coefficients and
        right-hand side — do not, so a model whose digest moved is loaded
        again. A quadratic objective coefficient that merely changed is pushed.

        Every vector read has an order contract, so two builds of one model
        agree.
        """
        return _digest(
            f'{self.column_count} {self.row_count} {self.objective_sense}'.encode(),
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
        ).digest()

    @cached_property
    def contents(self) -> str:
        """A digest of the built model **whole** — the numbers included.

        Whether two builds made one model, to the last bit. So it covers what
        [`structure`][] leaves out — the bounds, costs and right-hand sides a
        re-solve may push. A saved answer is checked against the data instead
        ([`digest_of_data`][specsolve.relational.answer_layout.digest_of_data]),
        which another machine reads the same.

        Over the built model rather than the sources, so two source mappings a
        build cannot tell apart agree here. A source whose rows moved builds a
        different label order and so digests differently. The objective is
        read through the dense cost vector, since ``obj`` carries no order
        contract and hashed in place would call one model two.
        """
        return _digest(
            self.structure + f'{self.objective_constant}'.encode(),
            self.cols['lb'].to_numpy(),
            self.cols['ub'].to_numpy(),
            self._dense_cost(),
            self.quad['coeff'].to_numpy(),
            self.rows['row'].to_numpy(),
            self.rows['rhs'].to_numpy(),
        ).hexdigest()

    def sets(self) -> Iterator[tuple[int, pl.Series, pl.Series]]:
        """Each special-ordered set: its type, member columns, and weights, in ``(set, weight)`` order.

        The type is read off the first member, every member carrying the same
        one. Nothing here is pushed on an update: a set is [`structure`][].
        """
        for members in self.sos.partition_by('set', maintain_order=True):
            yield members.item(0, 'type'), members.get_column('col'), members.get_column('weight')

    def quadratic_blocks(self) -> Iterator[tuple[int, pl.DataFrame]]:
        """Each quadratic row, ascending, and the ``(col_l, col_r, coeff)`` entries it owns.

        One row at a time, since every API that takes a quadratic constraint
        takes one per call.
        """
        for (row,), entries in self.qmatrix.group_by('row', maintain_order=True):
            yield int(row), entries.select('col_l', 'col_r', 'coeff')

    def row_blocks(self, budget: int | None) -> Iterator[MatrixBlock]:
        """Each chunk of rows with the matrix entries it owns — every sink's reader.

        A chunk is a ``slice`` by ``row_starts``, so nothing is sorted or
        searched. A consumer that needs the ``row`` labels asks
        [`matrix_block`][] with the chunk's own range, so its spans and entries
        cannot disagree.
        """
        for lo, hi in self._spans(budget):
            yield MatrixBlock(lo, hi, self._span(lo, hi), self.row_starts[lo:hi] - self.row_starts[lo])

    def matrix_block(self, lo: int, hi: int) -> pl.DataFrame:
        """Rows ``[lo, hi)`` of the matrix with their ``row`` labels spelled back out of ``row_starts``."""
        import numpy as np

        labels = np.repeat(np.arange(lo, hi, dtype=np.int64), np.diff(self.row_starts[lo : hi + 1]))
        return self._span(lo, hi).with_columns(pl.Series('row', labels))


def _digest(head: bytes, *vectors: np.ndarray) -> hashlib.blake2b:
    """A 16-byte blake2b over *head* and then each of *vectors*' bytes, in the order given."""
    import hashlib

    import numpy as np

    digest = hashlib.blake2b(head, digest_size=16)
    for vector in vectors:
        digest.update(np.ascontiguousarray(vector).data)
    return digest


def _finite(value: pl.Expr, infinity: float) -> pl.Expr:
    """*value* with each infinity as the finite sentinel the asking solver reads as one.

    A ``NaN`` never arrives: the door refuses one in a parameter and the schema
    one written in the file.
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
