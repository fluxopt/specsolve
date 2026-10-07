"""What a caller reads back — a solve's [`Result`][], a build's [`Diagnostics`][].

A [`Result`][] holds one finished frame per declaration, its values already
laid out over the build's coordinates.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import datetime  # noqa: TC003  — a Record annotation this module writes
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from specsolve.errors import NoSolutionError, SpecsolveError
from specsolve.messages import no_model_behind_this_answer_message, unknown_name_message
from specsolve.relational.answer_layout import (
    ACTIVITY,
    NO_PROVENANCE,
    RECORD_FILE,
    RECORD_SCHEMA,
    Metrics,
    Provenance,
    Record,
    checked_kind,
    clear_the_answer,
    write_format,
    write_reasons,
    write_whole,
)
from specsolve.relational.collect import collect_engine

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pandas as pd
    import polars as pl
    import xarray as xr

    from specsolve.relational.status import SolveStatus


#: How much of the session a solve keeps, as a request to
#: [`specsolve.api.Model.solve`][] and as the report in
#: [`Result.kept`][]. The solver and the work it did can only be dropped in
#: that order, so the fourth combination does not exist.
Keep = Literal['nothing', 'solver', 'progress']

#: What each word keeps, in the order of how much that is.
KEEPS: Mapping[Keep, str] = {
    'nothing': 'the model is handed to a fresh solver, which has nothing to begin from',
    'solver': 'the solver already holding the model is reused, and the work the last solve did is discarded',
    'progress': 'the solver is reused and carries on from where the last solve got to',
}


def unknown_keep_message(keep: object) -> str:
    """The message for a *keep* outside the three."""
    options = '\n'.join(f'  {name}: {what}' for name, what in KEEPS.items())
    return f'unknown keep {keep!r}. A solve may keep:\n{options}'


#: What the bridges out say when the environment cannot serve them, ``{module}``
#: being pandas or xarray.
_NEEDS_THE_EXTRA = (
    '{module} is not installed, and specsolve does not install it, so this build cannot bridge out to it: '
    'pip install {module}. A result needs nothing added to be read as it stands — primal() and dual() return '
    'polars frames, and save() writes one file per declaration.'
)


def tidy_to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    """A tidy polars frame as pandas, column by column.

    Raises:
        ModuleNotFoundError: On an install without pandas, naming the extra.
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_NEEDS_THE_EXTRA.format(module='pandas')) from exc

    return pd.DataFrame({column: frame[column].to_numpy() for column in frame.columns})


def tidy_to_dataarray(frame: pd.DataFrame, name: str) -> xr.DataArray:
    """A tidy frame as an ``xarray.DataArray``, labelled by its non-``value`` columns.

    A scalar declaration has none and comes back 0-dimensional.

    Raises:
        ModuleNotFoundError: On an install without xarray, naming the extra.
    """
    if importlib.util.find_spec('xarray') is None:
        raise ModuleNotFoundError(_NEEDS_THE_EXTRA.format(module='xarray'))

    dims = [column for column in frame.columns if column != 'value']
    if not dims:
        return frame['value'].to_xarray().rename(name)
    return frame.set_index(dims).to_xarray()['value'].rename(name)


def tidy_to_dataset(names: Sequence[str], one: Callable[[str], xr.DataArray]) -> xr.Dataset:
    """*names* as one dataset, each array built by *one*."""
    first, *rest = names
    dataset = one(first).to_dataset(name=first)
    for name in rest:
        dataset[name] = one(name)
    return dataset


def _number(value: float, *, sign: bool = False) -> str:
    """*value* as the shortest string that reads back as itself, without a trailing ``.0``."""
    text = repr(float(value))
    text = text.removesuffix('.0')
    return f'+{text}' if sign and not text.startswith('-') else text


def _bracket(labels: str) -> str:
    """``[1, wind]``, or nothing at all for a declaration over no dims."""
    return f'[{labels}]' if labels else ''


def _named_at(name: str, coordinate: Mapping[str, object]) -> str:
    """``balance[snapshot=1, g=gas]`` — a declaration at one coordinate, in its dim order."""
    return name + _bracket(', '.join(f'{dim}={label}' for dim, label in coordinate.items()))


@dataclass(frozen=True)
class ConstraintRow:
    """One built constraint row, spelled back out — what [`row`][specsolve.api.Model.row] returns.

    The row at one coordinate as the model built it, read off the built model,
    so it needs no solve. ``where`` masking, absent variables and coefficients
    the data made exactly zero have already removed their terms, so it can be
    shorter than the file suggests.

    Printed, it is one line of math in linopy's format; a row wider than
    [`display_terms`][] prints each variable's term count and coefficient
    span instead. [`terms`][] is the same content as a frame.

    Attributes:
        name: The constraint this row belongs to.
        coordinate: Where in that declaration it sits.
        terms: ``(variable, coordinate, coefficient)``, one row per term, in
            the solver's own column order. ``coordinate`` is the term's labels
            in its variable's dim order, as one string.
        sense: ``<=``, ``>=`` or ``==``.
        rhs: What the left-hand side is compared against.
    """

    name: str
    coordinate: Mapping[str, object]
    terms: pl.DataFrame
    sense: str
    rhs: float

    #: How many terms a line spells out before it summarises instead.
    display_terms = 12

    def __str__(self) -> str:
        """The row as one line: ``balance[snapshot=1]: +1 p[…] +50 p[…] >= 60``."""
        return f'{_named_at(self.name, self.coordinate)}: {self._body()} {self.sense} {_number(self.rhs)}'

    #: The line, not the field-by-field dataclass dump.
    __repr__ = __str__

    def _body(self) -> str:
        """The terms, spelled out or summarised."""
        if not self.terms.height:
            return '(no terms)'
        if self.terms.height <= self.display_terms:
            return ' '.join(
                f'{_number(coefficient, sign=True)} {variable}{_bracket(coordinate)}'
                for variable, coordinate, coefficient in self.terms.iter_rows()
            )
        return f'{self.terms.height} terms — {self._per_declaration()}'

    def _per_declaration(self) -> str:
        """``p: 300 (|coef| 1…50)`` per variable, in the row's own term order."""
        import polars as pl

        grouped = self.terms.group_by('variable', maintain_order=True).agg(
            pl.len().alias('terms'),
            pl.col('coefficient').abs().min().alias('low'),
            pl.col('coefficient').abs().max().alias('high'),
        )
        return ', '.join(
            f'{variable}: {terms} (|coef| {_number(low)})'
            if low == high
            else f'{variable}: {terms} (|coef| {_number(low)}…{_number(high)})'
            for variable, terms, low, high in grouped.iter_rows()
        )


@dataclass(frozen=True)
class InfeasibleSubsystem:
    """The rows and bounds that cannot hold together — what [`infeasible_subsystem`][specsolve.api.Model.infeasible_subsystem] returns.

    Irreducible: drop any one member and the rest can hold. A model can have
    several, and each solver may find a different one. Printed, it is one line
    per member, constraints first.

    Attributes:
        constraints: ``(dims…, sense, rhs)`` per constraint with a row in it,
            in declaration order, its rows in label order.
        bounds: ``(dims…, bound, value)`` per variable with a bound in it,
            in declaration order. ``bound`` is ``lower`` or ``upper``.
    """

    constraints: Mapping[str, pl.DataFrame]
    bounds: Mapping[str, pl.DataFrame]

    def __str__(self) -> str:
        """``balance[snapshot=1] == 200``, then ``p[snapshot=1, tech=gas] <= 100 (upper bound)``."""
        lines = [
            f'{_named_at(name, {d: member[d] for d in frame.columns[:-2]})} {member["sense"]} {_number(member["rhs"])}'
            for name, frame in self.constraints.items()
            for member in frame.iter_rows(named=True)
        ]
        lines += [
            f'{_named_at(name, {d: member[d] for d in frame.columns[:-2]})} '
            f'{_BOUND_SENSE[member["bound"]]} {_number(member["value"])} ({member["bound"]} bound)'
            for name, frame in self.bounds.items()
            for member in frame.iter_rows(named=True)
        ]
        return '\n'.join(lines)

    #: The lines, not the field-by-field dataclass dump.
    __repr__ = __str__


#: How each side of a variable's bound reads as a comparison.
_BOUND_SENSE = {'lower': '>=', 'upper': '<='}


@dataclass(frozen=True)
class Diagnostics:
    """What a build and its solves did that the answer does not show.

    Advisory, all of it: no answer depends on any field.
    """

    #: The shape the build produced: columns, rows, and matrix entries.
    columns: int
    rows: int
    nonzeros: int

    #: ``(constraint, rows_not_built)`` — every declared row that did not reach
    #: the solver: one emptied of all its terms, and one a propagated absence
    #: deleted. Empty where every declared row was built. A recurrence's first
    #: coordinate counts, so a ``shift`` against the horizon's edge reports
    #: here, as the boundary rather than a fault.
    omissions: pl.DataFrame

    #: ``(parameter, coordinates, rows, missing)`` — one row per parameter whose
    #: source is short of the coordinates its dims reach, in declaration order.
    #: Empty where every one is complete. An entry is a report, not a fault:
    #: absence is how a model masks. A parameter over no dims is never here.
    sparse_parameters: pl.DataFrame

    #: ``(constraint, smallest, largest)`` — the coefficient magnitudes each
    #: constraint block put in the matrix, one row per block that kept a term,
    #: in build order. ``largest / smallest`` over the frame is the
    #: conditioning to compare against the solver's.
    coefficient_range: pl.DataFrame

    #: ``(variable, smallest, largest)`` — the bound magnitudes each variable
    #: block put on its columns, one row per block that declared a finite one.
    #: Zero and infinity are excluded. A model can be clean on
    #: [`coefficient_range`][] and still have bounds the solver asks to have
    #: scaled. A large ``largest`` is usually a big number standing in for
    #: "uncapped", and wants no upper bound at all rather than a rounder one.
    bound_range: pl.DataFrame

    #: ``(constraint, smallest, largest)`` — the same for each block's
    #: right-hand sides, over the rows that survived.
    rhs_range: pl.DataFrame

    #: The same pair for the objective's coefficients, or ``None`` where the
    #: spec declares no objective and where every term of one cancelled.
    objective_range: tuple[float, float] | None

    #: How many times this model has been solved, and how many of those solves
    #: loaded the solver from scratch instead of pushing values onto one that
    #: already held it. ``loads == solves`` on an iterating driver means the
    #: model masks on a parameter that varies, unless the driver asked for
    #: ``keep='nothing'``. ``loads`` ticks on exactly the solves that report
    #: [`Result.kept`][] of ``nothing``.
    solves: int
    loads: int

    #: Cumulative wall-clock seconds per phase, keyed by the phase's name:
    #: ``attach`` (the caller's sources onto the plan), ``build`` (declarations
    #: into the model frames), ``handoff`` (the built model into a solver),
    #: ``solve`` (the solver's own run), ``write`` (the built model to a
    #: file). A phase that never ran has no key; one that ran again holds the
    #: sum.
    seconds: Mapping[str, float]

    def metrics(self) -> Metrics:
        """The sizes, counters and clocks as one value — the row ``archive=`` records beside the answer.

        [`Metrics`][specsolve.relational.answer_layout.Metrics] says what each
        field means. A phase this build never entered reads zero, and
        ``specsolve_run`` is null.
        """
        clocks = self.seconds
        return Metrics(
            columns=self.columns,
            rows=self.rows,
            nonzeros=self.nonzeros,
            solves=self.solves,
            loads=self.loads,
            attach_seconds=clocks.get('attach', 0.0),
            build_seconds=clocks.get('build', 0.0),
            handoff_seconds=clocks.get('handoff', 0.0),
            solve_seconds=clocks.get('solve', 0.0),
            write_seconds=clocks.get('write', 0.0),
        )


def _named(frames: Mapping[str, pl.LazyFrame], name: str, kind: str) -> pl.LazyFrame:
    try:
        return frames[name]
    except KeyError:
        raise KeyError(unknown_name_message(kind, name, frames)) from None


def evaluated(
    declared: Mapping[str, Callable[[], pl.DataFrame]],
    evaluator: Callable[[str | Mapping[str, object]], pl.DataFrame] | None,
    expression: str | Mapping[str, object],
) -> pl.DataFrame:
    """*expression* valued: by the declared reader that holds it, else lowered by *evaluator*.

    A declared name is served by its own reader first, so it answers off an
    archive that retains no model.

    Raises:
        SpecsolveError: *expression* is not a declared name and *evaluator* is
            ``None``.
    """
    if isinstance(expression, str) and expression in declared:
        return declared[expression]()
    if evaluator is None:
        raise SpecsolveError(no_model_behind_this_answer_message())
    return evaluator(expression)


@dataclass
class Result:
    """What a solve returned — the outcome, and access to any values.

    Returned whatever the solve concluded: test [`has_primal`][] before
    reading values, or catch [`NoSolutionError`][specsolve.errors.NoSolutionError].
    A result owns its values, so it outlives an update, another solve or
    ``model.close()``. It keeps the label frames of the build it answered
    alive until [`close`][], which is optional.
    """

    _status: SolveStatus
    _objective: float
    #: One ``(dims…, value)`` frame per declaration, lazy and in label order.
    #: ``None`` after [`close`][], which is what "closed" means; an empty
    #: mapping is a solve that left nothing.
    _primals: Mapping[str, pl.LazyFrame] | None
    _duals: Mapping[str, pl.LazyFrame] | None
    #: The constraints' left-hand sides at the solution, laid out exactly as
    #: [`_duals`][], and present whenever the primals are.
    _activities: Mapping[str, pl.LazyFrame] | None
    #: How much of the session this solve kept, read off what actually ran.
    _kept: Keep
    #: One deferred reader per declared named expression, and the ad-hoc
    #: evaluator. ``_evaluate`` is ``None`` where there is no spec as written:
    #: a build off an already-lowered ``Program``, or an answer read off disk.
    _expressions: Mapping[str, Callable[[], pl.DataFrame]] | None = None
    _evaluate: Callable[[str | Mapping[str, object]], pl.DataFrame] | None = None
    #: Why there are no duals, when a solve that left values still has none.
    #: ``None`` whenever [`_duals`][] holds them.
    _no_duals: str | None = None
    #: One ``(dims…, value)`` frame per constraint, laid out exactly as
    #: [`_duals`][], carrying the certificate an infeasible solve left.
    #: Empty on every solve that was not infeasible.
    _dual_rays: Mapping[str, pl.LazyFrame] | None = None
    #: Why there is no certificate — the status, or the solver setting that
    #: would have produced one. ``None`` whenever [`_dual_rays`][] holds it.
    _no_dual_ray: str | None = None
    #: [`digest_of`][specsolve.relational.answer_layout.digest_of] the spec this
    #: answered, or ``None`` for a solve run off a lowered program.
    _spec_digest: str | None = None
    #: When the solver returned, in UTC.
    _solved_at: datetime | None = None
    #: The built model's digest, or the callable that computes it on the first
    #: ask. ``None`` for an answer written before the column, or built by hand.
    #: Read through [`model_digest`][].
    _model_digest: str | Callable[[], str] | None = None
    #: The archive this answer was read back out of, as its record names it.
    #: ``None`` for a live solve: the name is stamped when an archive is
    #: written, not when the solver returns.
    _run: str | None = None
    #: What produced this answer. Empty for one built by hand.
    _provenance: Provenance = NO_PROVENANCE

    def model_digest(self) -> str | None:
        """Which model this answered — the document and the data it was attached to.

        [`spec_digest`][] names the document alone, so two scenarios of one
        spec share that and differ here. Computed on the first ask and kept.
        """
        if callable(self._model_digest):
            self._model_digest = self._model_digest()
        return self._model_digest

    @property
    def status(self) -> str:
        """Coarse outcome: ``ok`` / ``warning`` / ``error`` / ``aborted`` / ``unknown``."""
        return self._status.status

    @property
    def termination_condition(self) -> str:
        """What the solver said — ``optimal``, ``infeasible``, ``time_limit`` and so on."""
        return self._status.termination_condition

    @property
    def is_ok(self) -> bool:
        """The linopy rollup: not an error, an abort or a refusal."""
        return self._status.is_ok

    @property
    def has_primal(self) -> bool:
        """Whether there are values to read — what the accessors gate on.

        Narrower than [`is_ok`][]: a run stopped at a time limit before any
        incumbent is ``ok`` with nothing to read.
        """
        return self._status.is_readable

    @property
    def objective(self) -> float:
        """The objective value, or ``nan`` when there is no solution."""
        return self._objective

    @property
    def spec_digest(self) -> str | None:
        """Which spec this answered — a digest of the file, not its name.

        Two answers with one digest answered the same document, perhaps over
        other data. ``None`` where the solve ran off a lowered program.
        """
        return self._spec_digest

    @property
    def solved_at(self) -> datetime | None:
        """When the solver returned, in UTC — ``None`` where the solve carried no clock."""
        return self._solved_at

    @property
    def provenance(self) -> Provenance:
        """The solver, its options and the package versions that produced this answer.

        Read back as it was written, so an answer loaded from an archive names
        the environment that solved it rather than the one reading it.
        """
        return self._provenance

    @property
    def record(self) -> Record:
        """How this solve terminated, as the one row [`save`][] writes for it.

        A sweep keeps the same row per slice in [`record`][specsolve.sweep.Sweep.record].
        ``objective`` is ``None`` rather than ``nan`` where there are no
        values. Asking computes [`model_digest`][], as a save does.
        """
        return Record.of(
            self.termination_condition,
            self.objective,
            has_primal=self.has_primal,
            spec_digest=self._spec_digest,
            solved_at=self._solved_at,
            model_digest=self.model_digest(),
            provenance=self._provenance,
        )._replace(specsolve_run=self._run)

    @property
    def kept(self) -> Keep:
        """How much of the session this solve kept: ``solver``, ``progress`` or ``nothing``.

        What happened, not what was asked: a first solve or a structure that
        moved keeps ``nothing`` whatever ``keep=`` requested, so a driver that
        asked for ``progress`` and reads ``nothing`` is being told its labels
        moved. Advisory, like [`Diagnostics`][]: no answer depends on it.
        """
        return self._kept

    def _unclosed(self, what: str) -> Mapping[str, pl.LazyFrame]:
        """The primals, or why nothing here can be read: this result was closed."""
        if self._primals is None:
            raise SpecsolveError(
                f'cannot read {what}: this result was closed, and closing releases its values and its '
                f'hold on the coordinates they lay out over. Frames already read stay valid — they are '
                f'their own data — so read what you need before close(), or drop the `with` and close '
                f'when you are done.'
            )
        return self._primals

    def _readable(self, frames: Mapping[str, pl.LazyFrame] | None, what: str) -> Mapping[str, pl.LazyFrame]:
        """*frames*, or why they cannot be read — closed first, then the status."""
        self._unclosed(what)
        if not self._status.is_readable:
            wording = f' ({self._status.solver_wording})' if self._status.solver_wording else ''
            raise NoSolutionError(
                f'cannot read {what}: the solve terminated {self.termination_condition!r}'
                f'{wording}, so there are no values to read. Test '
                f'`has_primal` first. This raises rather than returning, because the solver '
                f'hands back a full-length vector of zeros either way and it is '
                f'indistinguishable from an answer.'
            )
        assert frames is not None, 'close() releases the primal, dual and activity frames together'
        return frames

    def primal(self, name: str) -> pl.DataFrame:
        """The tidy solution of variable *name* — ``(dims…, value)``.

        Rows come back in label order, row-major over the variable's coordinate
        product, so two reads and two runs agree.

        Raises:
            NoSolutionError: The solve left no values to read.
            SpecsolveError: This result was closed.
            KeyError: No variable is called *name*.
        """
        frames = self._readable(self._primals, f"the primal of '{name}'")
        return _named(frames, name, 'variable').collect(engine=collect_engine())

    def dual(self, name: str) -> pl.DataFrame:
        """Shadow prices of constraint *name* — ``(dims…, value)``, [`primal`][]'s shape and order.

        Each is the rate at which the optimal objective rises as the row's
        right side rises: of ``lhs <= rhs``, the rate in ``d`` of
        ``lhs <= rhs + d``. That is mathspec's ``dual(c)`` for every
        comparator, sense and sink, so which side a term is written on decides
        the sign.

        Duals exist only where a solver ran here, not for a model written to a
        file and solved elsewhere. Reduced costs and slacks are not read.

        Raises:
            NoSolutionError: The solve left no values at all.
            SpecsolveError: This result was closed, or it left primals but no
                duals — an integer variable makes them undefined, and so does
                an ``sos:`` set that ``Spec.expand()`` wrote out as binaries.
                ``gurobi`` and ``xpress`` branch on a set itself and keep
                them.
            KeyError: No constraint is called *name*.
        """
        frames = self._readable(self._duals, f"the dual of '{name}'")
        if self._no_duals is not None:
            raise SpecsolveError(self._no_duals)
        return _named(frames, name, 'constraint').collect(engine=collect_engine())

    def dual_ray(self, name: str) -> pl.DataFrame:
        """Constraint *name*'s share of the certificate that this model has no solution — ``(dims…, value)``.

        The only reader that answers on an infeasible solve, where
        [`primal`][], [`dual`][] and [`activity`][] all raise. Weight every row
        by its value here and add them, and the combined row demands more than
        the columns can deliver inside their bounds: the proof that nothing
        satisfies all of them at once, and what a Benders feasibility cut is
        built from.

        [`dual`][]'s shape and order, and the row's own sign on every sink.
        Where every column is held only by a lower bound of zero, the proof is
        ``Σ weight * right-hand side > 0``.

        ``highs`` always produces a certificate; ``gurobi`` needs
        ``{'InfUnbdInfo': 1}`` and ``xpress`` needs ``{'presolve': 0}`` in
        *solver_options*, set before the solve. A ray is live only: [`save`][]
        writes none, and no sweep spills one.

        Raises:
            SpecsolveError: This result was closed; the solve was not
                infeasible; or the sink produced no ray, in which case the
                message names the solver option that would have.
            KeyError: No constraint is called *name*.
        """
        self._unclosed(f"the dual ray of '{name}'")
        if self._no_dual_ray is not None:
            raise SpecsolveError(self._no_dual_ray)
        assert self._dual_rays is not None, 'a ray is released with the primals, which _unclosed just checked'
        return _named(self._dual_rays, name, 'constraint').collect(engine=collect_engine())

    def activity(self, name: str) -> pl.DataFrame:
        """The left-hand side of constraint *name* at the solution — ``(dims…, value)``, [`dual`][]'s shape and order.

        The solver's own number, not a recomputation, and readable whenever
        there is a solution, a mixed-integer one included.

        Raises:
            NoSolutionError: The solve left no values to read.
            SpecsolveError: This result was closed.
            KeyError: No constraint is called *name*.
        """
        frames = self._readable(self._activities, f"the activity of '{name}'")
        return _named(frames, name, 'constraint').collect(engine=collect_engine())

    def evaluate(self, expression: str | Mapping[str, object]) -> pl.DataFrame:
        """The value of *expression* at this solution — ``(dims…, value)``, [`primal`][]'s shape and order.

        *expression* is what one ``expressions:`` entry takes: a name the file
        declares, an expression string, or the mapping carrying ``cases:``
        with ``dims:`` and ``otherwise:``. The value is aggregated to the
        expression's own dims, in declaration order.

        Anything but a declared name lowers the spec as written, which costs
        what ``check`` costs. [`save`][] writes, and a sweep spills, only the
        expressions declared under ``expressions:``.

        Raises:
            NoSolutionError: The solve left no values to read.
            SpecsolveError: This result was closed; an undeclared expression
                on a model built from a lowered ``Program`` or read back off
                disk, which has nothing to lower it against; an archive whose
                sources build another model than the one this answered; or a
                divisor with no value.
            LanguageError: A construct outside the language, or a name the
                spec does not declare — a new parameter is a build, not a
                read.
        """
        self._readable(self._primals, 'an expression')
        return evaluated(self._expressions or {}, self._evaluate, expression)

    def _frame(self, name: str, kind: str) -> pl.DataFrame:
        """*name* through the reader *kind* names — the dispatch every bridge shares."""
        reader = {'primal': self.primal, 'dual': self.dual, 'expression': self.evaluate}[checked_kind(kind)]
        return reader(name)

    def _names(self, kind: str) -> tuple[str, ...]:
        """Every name of *kind* this result can read — what a bridge takes by default.

        Raises:
            NoSolutionError: The solve left no values to read.
            SpecsolveError: This result was closed, or *kind* is ``dual`` and
                the duals are undefined.
        """
        if checked_kind(kind) == 'primal':
            return tuple(self._readable(self._primals, 'the solution'))
        if kind == 'dual':
            frames = self._readable(self._duals, 'the duals')
            if self._no_duals is not None:
                raise SpecsolveError(self._no_duals)
            return tuple(frames)
        self._readable(self._primals, 'the expressions')
        return tuple(self._expressions or {})

    def to_pandas(self, name: str, kind: str = 'primal') -> pd.DataFrame:
        """One name's values as a tidy `pandas.DataFrame`.

        *name* is read through the reader *kind* names: ``primal``, ``dual``
        or ``expression``. Needs pandas, which specsolve does not install; the
        xarray bridges need xarray too.
        """
        return tidy_to_pandas(self._frame(name, kind))

    def to_dataarray(self, name: str, kind: str = 'primal') -> xr.DataArray:
        """One name's values as a labelled `xarray.DataArray`, [`to_pandas`][]'s arguments.

        Dense over the name's dims: a masked coordinate comes back NaN.
        """
        return tidy_to_dataarray(self.to_pandas(name, kind), name)

    def to_dataset(self, *names: str, kind: str = 'primal') -> xr.Dataset:
        """The named values of one *kind* as one `xarray.Dataset`; every name of that kind where none is given.

        Each arrives dense over its own dims, all at once — on a large model
        name the few you need.
        """
        return tidy_to_dataset(names or self._names(kind), lambda name: self.to_dataarray(name, kind))

    def save(self, directory: str | Path) -> Path:
        """Every kind this solve answered with, one file per name, into *directory*.

        ``record.parquet`` holds the
        [`Record`][specsolve.relational.answer_layout.Record], with a null
        rather than ``nan`` objective where none was reached. Beside it are
        ``primal/<name>.parquet`` per variable, ``dual/<name>.parquet`` per
        constraint where the duals are defined, ``activity/<name>.parquet``
        per constraint, and ``expression/<name>.parquet`` per named expression
        this data can evaluate. ``reasons.parquet`` holds
        ``(kind, name, reason)`` for whatever is deliberately left out — one
        row per failed expression, one with an empty *name* for the duals —
        and is absent when nothing is. A solve that left no values writes the
        record alone. The same model and data write the same bytes.

        ``format.json`` stamps the layout and the specsolve that wrote it:
        ``{"layout": 3, "specsolve": "…"}``. Every reader refuses another
        layout with a [`LayoutError`][specsolve.errors.LayoutError] that says
        to solve the model again and save it.

        Whatever a previous save left in *directory* is removed first; files
        outside the layout are left alone.

        Returns:
            The directory.

        Raises:
            SpecsolveError: This result was closed.
        """
        import polars as pl

        primals = self._unclosed('the solution')
        out = Path(directory)
        clear_the_answer(out)
        write_format(out)
        record = self.record._replace(specsolve_run=None)
        write_whole(pl.DataFrame([record._asdict()], schema_overrides=RECORD_SCHEMA), out / RECORD_FILE)
        if not self._status.is_readable:
            return out
        for name, frame in primals.items():
            write_whole(frame, out / 'primal' / f'{name}.parquet')
        for name, frame in (self._duals or {}).items():
            write_whole(frame, out / 'dual' / f'{name}.parquet')
        for name, frame in (self._activities or {}).items():
            write_whole(frame, out / ACTIVITY / f'{name}.parquet')
        no_expressions: dict[str, str] = {}
        for name, reader in (self._expressions or {}).items():
            try:
                frame = reader()
            except SpecsolveError as absent:
                no_expressions[name] = str(absent)
                continue
            write_whole(frame, out / 'expression' / f'{name}.parquet')
        write_reasons(out, self._no_duals, {'expression': no_expressions})
        return out

    def close(self) -> None:
        """Release this result's frames, and its hold on the build's label frames, early. Optional.

        Frames already read stay valid. The model and the solver are the
        [`Model`][specsolve.api.Model]'s to close.
        """
        self._primals = self._duals = self._activities = self._expressions = None
        self._dual_rays = None
        self._evaluate = None

    def __enter__(self) -> Result:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False
