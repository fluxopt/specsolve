"""What every solver sink is: a loaded model that takes an update's numbers and re-solves warm.

This module imports no solver.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import polars as pl

from specsolve.errors import SpecsolveError
from specsolve.relational.answer_layout import AT_LOWER, AT_UPPER, BASIC, BASIS_STATUSES, FIXED, SUPERBASIC
from specsolve.relational.names import VALUE
from specsolve.relational.sinks.handoff import SENSE_CODES

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence, Sized

    import numpy as np

    from specsolve.relational.sinks.capabilities import Capabilities
    from specsolve.relational.sinks.handoff import Handoff, Ints
    from specsolve.relational.status import SolveStatus


#: The status of a nonbasic row, by its sense.
_ROW_BOUND = {'<=': AT_UPPER, '>=': AT_LOWER, '==': FIXED}


@dataclass(frozen=True)
class Basis:
    """Where the solve ended: a [`BASIS_STATUSES`][] code per column and per row, in label order."""

    columns: np.ndarray[tuple[int], np.dtype[np.int8]]
    rows: np.ndarray[tuple[int], np.dtype[np.int8]]


@dataclass(frozen=True)
class SolveAnswer:
    """What a solve concluded, and the vectors it left.

    ``primal`` and ``activity`` (each row's left-hand side) are ``None`` where
    the solve left nothing worth reading. ``dual`` is also ``None`` for a
    mixed-integer model and for a run stopped short of a simplex basis.
    """

    status: SolveStatus
    objective: float
    primal: pl.Series | None
    dual: pl.Series | None
    activity: pl.Series | None
    #: A weight per row certifying that the constraints cannot all hold, in
    #: the sign convention of [`Solver.dual_ray`][], or ``None``.
    dual_ray: pl.Series | None = None
    #: The basis the solve ended on, read only where [`Solver.run`][] was asked
    #: for it, and ``None`` wherever the solve ended at no vertex.
    basis: Basis | None = None

    @classmethod
    def unreadable(cls, status: SolveStatus, dual_ray: pl.Series | None = None) -> SolveAnswer:
        """The answer for a solve that left nothing worth reading: a NaN objective and no vector but the ray."""
        return cls(status, float('nan'), None, None, None, dual_ray)


@dataclass(frozen=True)
class InfeasibleSubsystemIndices:
    """An irreducible infeasible subsystem, in the solver's own row and column indices."""

    rows: Ints
    #: The columns whose lower bound is in it.
    lower: Ints
    #: The columns whose upper bound is in it.
    upper: Ints


class Solver(ABC):
    """One solver, holding one model. Subclassed once per member of ``SOLVERS``.

    A driver gets one from [`loaded`][specsolve.relational.sinks.solvers.loaded],
    then runs and closes it::

        solver = solvers.loaded(held, name, handoff, options)
        solver.run(handoff)  # …repeatedly
        solver.close()
    """

    def __init__(
        self,
        handoff: Handoff,
        batch_rows: int | None = None,
        solver_options: Mapping[str, Any] | None = None,
    ) -> None:
        #: The options the model was loaded with.
        self._options = dict(solver_options or {})
        self._load(handoff, batch_rows)
        #: The loaded frames, until [`structure`][] replaces them with their digest.
        self._handoff: Handoff | None = handoff
        #: The digest, or ``None`` until [`structure`][] is first asked. Read through it.
        self._structure: bytes | None = None
        #: The matrix's coefficients as loaded, in entry order, for [`keeps`][].
        self._coefficients = handoff.matrix['coeff'].to_numpy()
        #: The loaded model's spans, read by [`_spans`][].
        self._columns = handoff.column_count
        self._rows = handoff.row_count

    #: The packages this member imports lazily, all needed for it to run.
    requires: ClassVar[tuple[str, ...]]

    #: What this member can ingest
    #: ([`refusal`][specsolve.relational.sinks.refusal] acts on it).
    capabilities: ClassVar[Capabilities]

    #: What to tell a caller when [`is_available`][] says no.
    unavailable_message: ClassVar[str]

    #: What a start of values does for an LP on this member, when it gives
    #: every column a value and when it leaves some out: ``used``,
    #: ``no_gain`` where the member takes them and no gain from them is known,
    #: so the solve warns, or ``refused`` where it cannot take them.
    lp_values: ClassVar[Mapping[Literal['complete', 'partial'], Literal['used', 'no_gain', 'refused']]]

    #: Option names, casefolded, whose value an answer records: the ones that
    #: change what a solve returns. Any other option is recorded by name
    #: alone, so a credential passed as an option never reaches an archive.
    recorded_options: ClassVar[frozenset[str]]

    def structure(self) -> bytes:
        """The loaded model's digest, read off its frames once, after which the frames are let go."""
        if self._structure is None:
            assert self._handoff is not None, 'a solver holds the handoff it loaded until its digest replaces it'
            self._structure = self._handoff.structure
            self._handoff = None
        return self._structure

    def keeps(self, handoff: Handoff, solver_options: Mapping[str, Any] | None) -> bool:
        """Whether this held solver may keep its load and take *handoff* by value.

        The same options and [`structure`][], and every matrix coefficient
        within 1e-12 relative of the one loaded, which stays: a coefficient
        summed over rows in another order moves in its last bits.
        """
        import numpy as np

        return (
            self._options == dict(solver_options or {})
            and self.structure() == handoff.structure
            and np.allclose(handoff.matrix['coeff'].to_numpy(), self._coefficients, rtol=1e-12, atol=0.0)
        )

    @classmethod
    def imported(cls) -> Any:
        """Every package in [`requires`][], imported. Returns the first, the member's own library.

        Raises:
            ModuleNotFoundError: With [`unavailable_message`][], if any is missing.
        """
        try:
            modules = [__import__(package) for package in cls.requires]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(cls.unavailable_message) from exc
        return modules[0]

    @classmethod
    def is_available(cls) -> bool:
        """Whether this environment can run this solver, probed without importing it or raising.

        Probed at the top-level name: ``find_spec`` on a dotted one imports the parent.
        """
        return all(importlib.util.find_spec(package.partition('.')[0]) is not None for package in cls.requires)

    @abstractmethod
    def _load(self, handoff: Handoff, batch_rows: int | None) -> None:
        """Hand *handoff* to the solver and hold whatever reads it back. Called by ``__init__``."""

    @abstractmethod
    def push(self, handoff: Handoff) -> None:
        """*handoff*'s bounds, costs and right-hand sides onto the loaded model, as whole vectors.

        Called only after *handoff*'s digest matched the loaded one.
        """

    def warm(self, basis: Basis) -> None:
        """Start the next [`run`][] of this LP from *basis* instead of from scratch.

        Raises:
            SpecsolveError: A basis that does not span the loaded model.
        """
        self._spans('basis', basis.columns, self._columns, 'columns')
        self._spans('basis', basis.rows, self._rows, 'rows')
        self._warm(basis)

    @abstractmethod
    def _warm(self, basis: Basis) -> None:
        """Set *basis* on the loaded model in the solver's own statuses. [`warm`][] has checked its spans."""

    def start(self, values: np.ndarray[tuple[int], np.dtype[np.float64]]) -> None:
        """Start the next [`run`][] from *values*, one per column and NaN where none is given.

        A mixed-integer model takes them as a starting incumbent, completing
        what is missing and repairing what is infeasible as far as it can; an
        LP as [`lp_values`][] says.

        Raises:
            SpecsolveError: Values that do not span the loaded model.
        """
        self._spans('start', values, self._columns, 'columns')
        self._start(values)

    @abstractmethod
    def _start(self, values: np.ndarray[tuple[int], np.dtype[np.float64]]) -> None:
        """Hand the values *values* gives to the solver to start from. [`start`][] has checked their span."""

    def _spans(self, what: str, values: Sized, expected: int, axis: str) -> None:
        """Refuse a *what* whose *values* do not span the loaded model's *expected* *axis*."""
        if len(values) != expected:
            raise SpecsolveError(
                f'this {what} carries {len(values)} entries for a model with {expected} {axis}. It is '
                f'positional, so one laid out for a differently shaped model would start the solve from a '
                f'state about a different one. This is an engine bug rather than a problem with the model '
                f'— please report it.'
            )

    def run(self, handoff: Handoff, *, basis: bool = False) -> SolveAnswer:
        """Solve what is loaded and read it back, with the [`Basis`][] it ended on where *basis* asks.

        Raises:
            SpecsolveError: A solver vector that does not span the model.
        """
        answer = self._run(handoff)
        if basis and answer.primal is not None:
            answer = replace(answer, basis=self._settled(handoff))
        self._check_span('primal', answer.primal, handoff.column_count)
        self._check_span('dual', answer.dual, handoff.row_count)
        self._check_span('activity', answer.activity, handoff.row_count)
        self._check_span('dual ray', answer.dual_ray, handoff.row_count)
        return answer

    def _settled(self, handoff: Handoff) -> Basis | None:
        """[`_basis`][], [`settled`][] against the model it was read from, after its spans are checked."""
        read = self._basis()
        if read is None:
            return None
        columns, rows = read
        self._check_span('column basis', columns, handoff.column_count)
        self._check_span('row basis', rows, handoff.row_count)
        return settled(handoff, columns, rows)

    @abstractmethod
    def _basis(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The basis the last [`run`][] ended on as [`BASIS_STATUSES`][] codes, columns then rows.

        Called only after a run that left a primal. ``None`` where it ended at
        no vertex: a mixed-integer model, an interior-point run without
        crossover. A nonbasic row may carry any nonbasic code, and a column at
        equal bounds either bound's: [`_settled`][] reads both off the model.
        """

    def _check_span(self, quantity: str, values: Sized | None, expected: int) -> None:
        """Refuse a solver vector that does not span the model. ``None`` passes."""
        if values is not None and len(values) != expected:
            raise SpecsolveError(
                f'{type(self).__name__} returned {len(values)} {quantity} values for a model with '
                f'{expected}. Reading a solution back is positional, so a vector that does not span '
                f'the model describes a different one. This is an engine bug rather than a problem '
                f'with the model — please report it.'
            )

    @abstractmethod
    def _run(self, handoff: Handoff) -> SolveAnswer:
        """Solve what is loaded and read it back.

        *handoff* is read only for the objective's constant. An infeasible solve
        returns what [`dual_ray`][] gives.
        """

    def _unreadable(self, status: SolveStatus) -> SolveAnswer:
        """The answer for a solve that left nothing worth reading, carrying the [`dual_ray`][] where it was infeasible."""
        return SolveAnswer.unreadable(status, self.dual_ray() if status.termination_condition == 'infeasible' else None)

    @abstractmethod
    def dual_ray(self) -> pl.Series | None:
        """A weight per row certifying that this infeasible model has no solution.

        Called only after an infeasible solve. A row's weight carries the sign
        the row is written with, HiGHS's and Xpress's convention. A member whose
        solver signs the other way negates what it reads.

        Returns:
            The weights in row order, or ``None`` where this solver produced
            none.
        """

    @abstractmethod
    def infeasible_subsystem(self) -> InfeasibleSubsystemIndices | None:
        """The rows and bounds that cannot hold together, read only after an infeasible solve.

        Integrality, and any constraint that is neither a row nor a bound, is
        left out.

        Returns:
            The subsystem, or ``None`` where the solver did not prove one.
        """

    @abstractmethod
    def forget(self) -> None:
        """Make the next run begin as if the loaded model had never been solved.

        A member with nothing to discard implements this as a no-op.
        """

    @property
    @abstractmethod
    def handle(self) -> Any:
        """The native object the load handed back, or ``None`` once closed.

        Owned by this holder: [`close`][] releases it, not the caller.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the loaded model, and anything outside this process with it.

        Idempotent. Afterwards [`handle`][] is ``None``. A holder dropped
        without closing releases the same: a member whose library does not do
        that on collection registers a finalizer over the objects, innermost
        first, rather than over itself.
        """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


def spelled_senses(spelling: Mapping[str, str]) -> np.ndarray[tuple[int, ...], np.dtype[np.str_]]:
    """[`SENSE_CODES`][] as one solver's spellings, indexed by code.

    A sense added to [`SENSE_CODES`][] and not to *spelling* raises instead.
    """
    import numpy as np

    out = np.empty(len(SENSE_CODES), dtype='<U1')
    for sense, code in SENSE_CODES.items():
        out[code] = spelling[sense]
    return out


def settled(handoff: Handoff, columns: np.ndarray, rows: np.ndarray) -> Basis:
    """*columns* and *rows*, [`BASIS_STATUSES`][] codes, with each nonbasic one at a bound *handoff* has.

    A member reads only basic or not, and which bound a column sits at, and a
    basis carried from another model may name a bound this one lacks. So a
    column keeps its side where that bound is finite, else takes the other,
    else is ``superbasic`` (free, at zero), and is ``fixed`` where its bounds
    are equal. A row's bound follows from its sense, since each row has one.
    """
    import numpy as np

    cols = handoff.dense_columns(np.inf)
    lower, upper = np.isfinite(cols.lb), np.isfinite(cols.ub)
    side = np.where(upper & ((columns == AT_UPPER) | ~lower), AT_UPPER, np.where(lower, AT_LOWER, SUPERBASIC))
    side = np.where(cols.lb == cols.ub, FIXED, side)
    columns = np.where(np.isin(columns, (AT_LOWER, AT_UPPER, FIXED)), side, columns)
    bound = np.asarray([_ROW_BOUND[sense] for sense in SENSE_CODES], dtype=np.int8)
    rows = np.where(np.isin(rows, (BASIC, SUPERBASIC)), rows, bound[handoff.dense_rows(np.inf).sense])
    return Basis(columns.astype(np.int8), rows.astype(np.int8))


def basis_codes(native: Any, codes: Sequence[int]) -> np.ndarray[tuple[int], np.dtype[np.int8]]:  # pyrefly: ignore[explicit-any] — a solver hands back its own array type
    """A solver's own basis statuses as [`BASIS_STATUSES`][] codes: *codes* indexed by each status."""
    import numpy as np

    return np.asarray(codes, dtype=np.int8)[np.asarray(native, dtype=np.int64)]


def solver_codes(codes: np.ndarray, native: Mapping[int, int]) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    """[`BASIS_STATUSES`][] *codes* as a solver's own statuses: *native* maps every code to the solver's status.

    Raises:
        KeyError: A *native* that leaves a code out.
    """
    import numpy as np

    return np.asarray([native[code] for code in range(len(BASIS_STATUSES))], dtype=np.int64)[codes]


def solver_vector(values: Any) -> pl.Series:  # pyrefly: ignore[explicit-any] — a solver hands back its own array type
    """One quantity a solver produced, in its own index — every sink's read-back.

    A series rather than a ``(label, value)`` frame: the read-back takes a
    declaration's share by slicing.
    """
    return pl.Series(VALUE, values, dtype=pl.Float64)
