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
from specsolve.relational.sinks.handoff import SENSE_CODES

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence, Sized

    import numpy as np

    from specsolve.relational.sinks.capabilities import Capabilities
    from specsolve.relational.sinks.handoff import Handoff
    from specsolve.relational.status import SolveStatus


@dataclass(frozen=True)
class WarmStart:
    """What one solve leaves for a later session: a basis, or an incumbent.

    Read with [`Solver.warm_start`][], applied with [`Solver.warm`][]. The
    statuses are the reading solver's own encoding, so only that solver takes
    it back.
    """

    #: The member of ``SOLVERS`` that read it.
    solver: str
    #: Basis status per column in label order, or ``None`` where no basis was left.
    column_statuses: Any | None
    #: Basis status per row in label order; filled exactly when
    #: [`column_statuses`][] is.
    row_statuses: Any | None
    #: Primal value per column in label order (a mixed-integer incumbent), or
    #: ``None`` where the basis carries the start.
    column_values: Any | None

    def basis(self) -> tuple[Any, Any] | None:
        """Both status vectors, or ``None`` where the incumbent carries the start."""
        if self.column_statuses is not None and self.row_statuses is not None:
            return self.column_statuses, self.row_statuses
        return None


#: A basis status in the one vocabulary every member reads its solver's into,
#: each at the index that is its code. A row's bound is its right-hand side,
#: so a binding ``<=`` row is ``at_upper``, a binding ``>=`` row ``at_lower``,
#: and a nonbasic ``==`` row, like a nonbasic variable whose bounds are equal,
#: ``fixed``. ``superbasic`` is nonbasic between its bounds.
BASIS_STATUSES = ('basic', 'at_lower', 'at_upper', 'fixed', 'superbasic')
BASIC, AT_LOWER, AT_UPPER, FIXED, SUPERBASIC = range(len(BASIS_STATUSES))

#: The status of a nonbasic row, by its sense.
_ROW_BOUND = {'<=': AT_UPPER, '>=': AT_LOWER, '==': FIXED}

#: [`BASIS_STATUSES`][] as the dtype a basis is read back in.
BASIS = pl.Enum(BASIS_STATUSES)


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
        #: The loaded model's spans, read by [`_takes`][].
        self._columns = handoff.column_count
        self._rows = handoff.row_count

    #: The packages this member imports lazily, all needed for it to run.
    requires: ClassVar[tuple[str, ...]]

    #: What this member can ingest
    #: ([`refusal`][specsolve.relational.sinks.refusal] acts on it).
    capabilities: ClassVar[Capabilities]

    #: What to tell a caller when [`is_available`][] says no.
    unavailable_message: ClassVar[str]

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
        """Whether this held solver may keep its load and take *handoff* by value."""
        return self._options == dict(solver_options or {}) and self.structure() == handoff.structure

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

    @abstractmethod
    def warm_start(self) -> WarmStart | None:
        """What the loaded model holds to warm a later session.

        The basis after an LP solve, the incumbent after a mixed-integer one,
        or ``None`` where it holds neither, as before any solve.
        """

    def warm(self, ws: WarmStart) -> None:
        """Start the next [`run`][] from *ws* instead of from scratch.

        The caller vouches that *ws* was read from a model with this one's
        label set.

        Raises:
            SpecsolveError: A warm start read from another solver, or whose
                vectors do not span the loaded model.
        """
        self._takes(ws)
        self._warm(ws)

    def _takes(self, ws: WarmStart) -> None:
        """Refuse a warm start from another solver, or whose vectors do not span the loaded model."""
        mine = type(self).__name__.lower()
        if ws.solver != mine:
            raise SpecsolveError(
                f'this warm start was read from {ws.solver!r} and cannot warm a {mine!r} session: '
                f"basis statuses and incumbents are the reading solver's own encoding, so applied "
                f'elsewhere they would start the solve from a state that means something else. '
                f'Read a warm start from the solver that will take it back.'
            )
        spans = (
            ('column statuses', ws.column_statuses, self._columns, 'columns'),
            ('row statuses', ws.row_statuses, self._rows, 'rows'),
            ('column values', ws.column_values, self._columns, 'columns'),
        )
        for quantity, values, expected, axis in spans:
            if values is not None and len(values) != expected:
                raise SpecsolveError(
                    f'this warm start carries {len(values)} {quantity} for a model with {expected} '
                    f'{axis}. A basis and an incumbent are positional, so one read from a '
                    f'differently shaped model would start the solve from a state about a '
                    f'different one — carry a warm start only across builds whose label set '
                    f'is unchanged.'
                )

    @abstractmethod
    def _warm(self, ws: WarmStart) -> None:
        """Apply *ws* onto the loaded model. [`warm`][] has already checked its solver and spans."""

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
        if answer.basis is not None:
            self._check_span('column basis', answer.basis.columns, handoff.column_count)
            self._check_span('row basis', answer.basis.rows, handoff.row_count)
        return answer

    def _settled(self, handoff: Handoff) -> Basis | None:
        """[`_basis`][] with each nonbasic status read against the model's bounds.

        A member reads only basic or not, and which bound a column sits at: a
        row's bound follows from its sense, since each row has one, and a
        column at bounds that are equal is ``fixed`` whichever one the solver
        names.
        """
        import numpy as np

        read = self._basis()
        if read is None:
            return None
        columns, rows = read
        cols = handoff.dense_columns(np.inf)
        columns = np.where(np.isin(columns, (AT_LOWER, AT_UPPER)) & (cols.lb == cols.ub), FIXED, columns)
        bound = np.asarray([_ROW_BOUND[sense] for sense in SENSE_CODES], dtype=np.int8)
        nonbasic = ~np.isin(rows, (BASIC, SUPERBASIC))
        rows = np.where(nonbasic, bound[handoff.dense_rows(np.inf).sense], rows)
        return Basis(columns.astype(np.int8), rows.astype(np.int8))

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


def basis_codes(native: Any, codes: Sequence[int]) -> np.ndarray[tuple[int], np.dtype[np.int8]]:  # pyrefly: ignore[explicit-any] — a solver hands back its own array type
    """A solver's own basis statuses as [`BASIS_STATUSES`][] codes: *codes* indexed by each status."""
    import numpy as np

    return np.asarray(codes, dtype=np.int8)[np.asarray(native, dtype=np.int64)]


def solver_vector(values: Any) -> pl.Series:  # pyrefly: ignore[explicit-any] — a solver hands back its own array type
    """One quantity a solver produced, in its own index — every sink's read-back.

    A series rather than a ``(label, value)`` frame: the read-back takes a
    declaration's share by slicing.
    """
    import numpy as np

    return pl.Series('value', np.asarray(values, dtype=np.float64))
