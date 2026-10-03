"""What every solver sink is: a loaded model that takes an update's numbers and re-solves warm.

This module imports no solver.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

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

    A subclass owns the hand-off: loading, pushing values, running, releasing.
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
        """What the loaded model holds to warm a later session, if anything.

        Returns:
            The basis after an LP solve, the incumbent after a mixed-integer
            one, and ``None`` where the model holds neither, as before any
            solve.
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

    def run(self, handoff: Handoff) -> SolveAnswer:
        """Solve what is loaded and read it back.

        Raises:
            SpecsolveError: A solver vector that does not span the model.
        """
        answer = self._run(handoff)
        self._check_span('primal', answer.primal, handoff.column_count)
        self._check_span('dual', answer.dual, handoff.row_count)
        self._check_span('activity', answer.activity, handoff.row_count)
        self._check_span('dual ray', answer.dual_ray, handoff.row_count)
        return answer

    def _check_span(self, quantity: str, values: pl.Series | None, expected: int) -> None:
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

    def dual_ray(self) -> pl.Series | None:
        """A weight per row certifying that this infeasible model has no solution.

        Called only after an infeasible solve. A row's weight carries the sign
        the row is written with, HiGHS's and Xpress's convention. A member whose
        solver signs the other way negates what it reads.

        Returns:
            The weights in row order, or ``None`` where this solver produced
            none.
        """
        return None

    @abstractmethod
    def forget(self) -> None:
        """Discard the work the last solve did, keeping the model loaded.

        The next run begins as if the model had never been solved. A member
        with nothing to discard implements this as a no-op.
        """

    @property
    @abstractmethod
    def handle(self) -> Any:
        """The native object the load handed back, or ``None`` once closed.

        Owned by this holder: the caller does not release it, [`close`][]
        does.
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
