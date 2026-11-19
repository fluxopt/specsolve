"""What a caller reads back — a solve's [`Result`][], a build's [`Diagnostics`][].

A [`Result`][] holds one finished frame per declaration, its values already
laid out over the build's coordinates. [`Result.save`][] writes one to disk;
[`load_result`][] reads it back, and [`scan_result`][] leaves it there.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import datetime  # noqa: TC003  — a Record annotation this module writes
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from specsolve import _parquet as parquet
from specsolve._answer_layout import (
    BASES,
    NO_BASIS,
    NO_PROVENANCE,
    OUTPUT_KINDS,
    PRICED,
    RECORD_FILE,
    RECORD_SCHEMA,
    Metrics,
    Provenance,
    Record,
    asked_for,
    check_format,
    checked_kind,
    clear_the_answer,
    kinds_of,
    not_requested_message,
    read_outputs,
    read_reasons,
    read_spec,
    saved_frames,
    write_format,
    write_reasons,
    write_spec,
    write_whole,
)
from specsolve._collect import collected
from specsolve._messages import coordinate_text, no_model_behind_this_answer_message, unknown_name_message
from specsolve._names import VALUE
from specsolve.errors import LayoutError, NoSolutionError, SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pandas as pd
    import polars as pl
    import xarray as xr
    from mathspec.spec import Spec

    from specsolve._answer_layout import FrameWriter
    from specsolve._status import SolveStatus


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

    dims = [column for column in frame.columns if column != VALUE]
    if not dims:
        return frame[VALUE].to_xarray().rename(name)
    return frame.set_index(dims).to_xarray()[VALUE].rename(name)


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


def _bracket(coordinate: str) -> str:
    """``[t=1, g=wind]``, or nothing at all for a declaration over no dims."""
    return f'[{coordinate}]' if coordinate else ''


def _named_at(name: str, coordinate: Mapping[str, object]) -> str:
    """``balance[snapshot=1, g=gas]`` — a declaration at one coordinate, in its dim order."""
    return name + _bracket(coordinate_text(coordinate))


@dataclass(frozen=True)
class ConstraintRow:
    """One built constraint row, spelled back out — what [`row`][specsolve._api.Model.row] returns.

    The row at one coordinate as the model built it, read off the built model,
    so it needs no solve. ``where`` masking, absent variables and coefficients
    the data made exactly zero have already removed their terms, so it can be
    shorter than the file suggests.

    Printed, it is one line of math, each term at its named coordinate; a row
    wider than [`display_terms`][] prints each variable's term count and
    coefficient span instead. [`terms`][] is the same content as a frame.

    Attributes:
        name: The constraint this row belongs to.
        coordinate: Where in that declaration it sits.
        terms: ``(variable, coordinate, coefficient)``, one row per term, in
            the solver's own column order. ``coordinate`` is the term's
            coordinate as one string, ``t=1, g=gas``, in its variable's dim
            order.
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
    """The rows and bounds that cannot hold together — what [`infeasible_subsystem`][specsolve._api.Model.infeasible_subsystem] returns.

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
    #: model masks on a parameter that varies.
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

        [`Metrics`][specsolve._answer_layout.Metrics] says what each
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
    #: ``{kind: {name: frame}}``: one ``(dims…, value)`` frame per declaration,
    #: lazy and in label order, under ``primal``, ``dual`` and each of the
    #: [`OUTPUT_KINDS`][specsolve._answer_layout.OUTPUT_KINDS] the solve asked
    #: for. Its keys are what this answer carries; a kind of which the solve
    #: left nothing maps to nothing. ``None`` after [`close`][], which is what
    #: "closed" means.
    _frames: Mapping[str, Mapping[str, pl.LazyFrame]] | None
    #: One deferred reader per declared named expression, and the ad-hoc
    #: evaluator. ``_evaluate`` is ``None`` where there is no spec as written:
    #: a build off an already-lowered ``Program``, or an answer read off disk.
    _expressions: Mapping[str, Callable[[], pl.DataFrame]] | None = None
    _evaluate: Callable[[str | Mapping[str, object]], pl.DataFrame] | None = None
    #: Why there are no duals, when a solve that left values still has none.
    #: ``None`` whenever [`_frames`][] holds them.
    _no_duals: str | None = None
    #: One ``(dims…, value)`` frame per constraint, laid out exactly as the
    #: duals, carrying the certificate an infeasible solve left.
    #: Empty on every solve that was not infeasible.
    _dual_rays: Mapping[str, pl.LazyFrame] | None = None
    #: Why there is no certificate — the status, or the solver setting that
    #: would have produced one. ``None`` whenever [`_dual_rays`][] holds it.
    _no_dual_ray: str | None = None
    #: The spec this answered, or ``None`` for a solve run off a lowered program.
    _spec: Spec | None = None
    #: When the solver returned, in UTC.
    _solved_at: datetime | None = None
    #: The archive this answer was read back out of, as its record names it.
    #: ``None`` for a live solve: the name is stamped when an archive is
    #: written, not when the solver returns.
    _run: str | None = None
    #: What produced this answer. Empty for one built by hand.
    _provenance: Provenance = NO_PROVENANCE

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
    def spec(self) -> Spec | None:
        """The spec this answered, as written; ``None`` where the solve ran off a lowered program.

        Two answers whose specs compare equal answered the same document,
        perhaps over other data.
        """
        return self._spec

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

        A sweep keeps the same row per slice in [`record`][specsolve._sweep.Sweep.record].
        ``objective`` is ``None`` rather than ``nan`` where there are no
        values.
        """
        return Record.of(
            self.termination_condition,
            self.objective,
            has_primal=self.has_primal,
            solved_at=self._solved_at,
            provenance=self._provenance,
        )._replace(specsolve_run=self._run)

    def _unclosed(self, what: str) -> Mapping[str, Mapping[str, pl.LazyFrame]]:
        """Every frame held, by kind, or why nothing here can be read: this result was closed."""
        if self._frames is None:
            raise SpecsolveError(
                f'cannot read {what}: this result was closed, and closing releases its values and its '
                f'hold on the coordinates they lay out over. Frames already read stay valid — they are '
                f'their own data — so read what you need before close(), or drop the `with` and close '
                f'when you are done.'
            )
        return self._frames

    def _readable(self, kind: str, what: str) -> Mapping[str, pl.LazyFrame]:
        """The frames of *kind*, one this answer carries, or why they cannot be read — closed first, then the status."""
        frames = self._unclosed(what)[kind]
        if not self._status.is_readable:
            wording = f' ({self._status.solver_wording})' if self._status.solver_wording else ''
            raise NoSolutionError(
                f'cannot read {what}: the solve terminated {self.termination_condition!r}'
                f'{wording}, so there are no values to read. Test '
                f'`has_primal` first. This raises rather than returning, because the solver '
                f'hands back a full-length vector of zeros either way and it is '
                f'indistinguishable from an answer.'
            )
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
        frames = self._readable('primal', f"the primal of '{name}'")
        return _named(frames, name, 'variable').pipe(collected)

    def dual(self, name: str) -> pl.DataFrame:
        """Shadow prices of constraint *name* — ``(dims…, value)``, [`primal`][]'s shape and order.

        Each is the rate at which the optimal objective rises as the row's
        right side rises: of ``lhs <= rhs``, the rate in ``d`` of
        ``lhs <= rhs + d``. That is mathspec's ``dual(c)`` for every
        comparator, sense and sink, so which side a term is written on decides
        the sign.

        Duals exist only where a solver ran here, not for a model written to a
        file and solved elsewhere.

        Raises:
            NoSolutionError: The solve left no values at all.
            SpecsolveError: This result was closed, or it left primals but no
                duals — an integer variable makes them undefined, and so does
                an ``sos:`` set that ``Spec.expand()`` wrote out as binaries.
                ``gurobi`` and ``xpress`` branch on a set itself and keep
                them.
            KeyError: No constraint is called *name*.
        """
        frames = self._readable('dual', f"the dual of '{name}'")
        if self._no_duals is not None:
            raise SpecsolveError(self._no_duals)
        return _named(frames, name, 'constraint').pipe(collected)

    def reduced_cost(self, name: str) -> pl.DataFrame:
        """Reduced costs of variable *name* — ``(dims…, value)``, [`primal`][]'s shape and order.

        Each is the rate at which the optimal objective rises as the
        variable's binding bound rises, in the sign convention of [`dual`][]
        for every sense and sink: of ``x >= l``, the rate in ``d`` of
        ``x >= l + d``. A variable strictly between its bounds has zero.

        It is computed from the duals as the objective's gradient less each
        row's gradient times that row's dual, so it exists wherever
        [`dual`][] does and nowhere else. Carried only where the solve was
        asked for it with ``outputs={'reduced_cost'}``.

        Raises:
            NoSolutionError: The solve left no values at all.
            SpecsolveError: This result was closed; the solve was not asked
                for reduced costs; or it left primals but no duals, as
                [`dual`][] raises.
            KeyError: No variable is called *name*.
        """
        return self._output('reduced_cost', name)

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
        return _named(self._dual_rays, name, 'constraint').pipe(collected)

    def activity(self, name: str) -> pl.DataFrame:
        """The left-hand side of constraint *name* at the solution — ``(dims…, value)``, [`dual`][]'s shape and order.

        The solver's own number, not a recomputation, and readable whenever
        there is a solution, a mixed-integer one included. Carried only where
        the solve was asked for it with ``outputs={'activity'}``.

        Raises:
            NoSolutionError: The solve left no values to read.
            SpecsolveError: This result was closed, or the solve was not asked
                for its activity.
            KeyError: No constraint is called *name*.
        """
        return self._output('activity', name)

    def slack(self, name: str) -> pl.DataFrame:
        """How far constraint *name* is from binding at the solution — ``(dims…, value)``, [`dual`][]'s shape and order.

        Non-negative wherever the row holds, whichever side a term is written
        on: ``rhs - lhs`` for ``<=``, ``lhs - rhs`` for ``>=``, and
        ``-|rhs - lhs|`` for ``==``, which holds only at zero. A binding row
        reads zero, and a violated one, such as at an incumbent within the
        solver's tolerance, reads below zero by how much. Computed from
        [`activity`][], so readable wherever it is, and carried only where
        the solve was asked for it with ``outputs={'slack'}``.

        Raises:
            NoSolutionError: The solve left no values to read.
            SpecsolveError: This result was closed, or the solve was not asked
                for its slack.
            KeyError: No constraint is called *name*.
        """
        return self._output('slack', name)

    def variable_basis(self, name: str) -> pl.DataFrame:
        """The basis status of each column of variable *name* where the solve ended — ``(dims…, value)``, [`primal`][]'s shape and order.

        ``value`` is one of ``basic``, ``at_lower``, ``at_upper``, ``fixed``
        (nonbasic at bounds that are equal) and ``superbasic`` (nonbasic
        between its bounds), in that order as an ``Enum``, the same words for
        every sink. Carried only where the solve was asked for it with
        ``outputs={'basis'}``, and only an LP that ended on a basis has one.

        Raises:
            NoSolutionError: The solve left no values to read.
            SpecsolveError: This result was closed, the solve was not asked for
                its basis, or it ended on none.
            KeyError: No variable is called *name*.
        """
        return self._output('variable_basis', name)

    def constraint_basis(self, name: str) -> pl.DataFrame:
        """The basis status of each row of constraint *name* where the solve ended — ``(dims…, value)``, [`dual`][]'s shape and order.

        In [`variable_basis`][]'s words, with the right-hand side as the
        row's bound: a binding ``<=`` row is ``at_upper``, a binding ``>=``
        row ``at_lower``, a binding ``==`` row ``fixed``, and a row that is
        not binding ``basic``. Carried only where the solve was asked for it
        with ``outputs={'basis'}``, and only an LP that ended on a basis has
        one.

        Raises:
            NoSolutionError: The solve left no values to read.
            SpecsolveError: This result was closed, the solve was not asked for
                its basis, or it ended on none.
            KeyError: No constraint is called *name*.
        """
        return self._output('constraint_basis', name)

    def _start(self) -> dict[str, Mapping[str, pl.LazyFrame]]:
        """What this answer gives a solve to start from, keyed as [`Start`][specsolve.types.Start] is: its primal, and its basis where it carries one.

        Raises:
            NoSolutionError: The solve left no values to read.
            SpecsolveError: This result was closed.
        """
        given = {'primal': self._readable('primal', 'the values to start from')}
        for kind in BASES:
            if frames := self._unclosed('the basis to start from').get(kind):
                given[kind] = frames
        return given

    def _output(self, kind: str, name: str) -> pl.DataFrame:
        """*name*'s frame of *kind*, one of the [`OUTPUT_KINDS`][specsolve._answer_layout.OUTPUT_KINDS]: what each output's reader returns."""
        return _named(self._carried(kind, name), name, OUTPUT_KINDS[kind].per).pipe(collected)

    def _carried(self, kind: str, name: str) -> Mapping[str, pl.LazyFrame]:
        """The frames of *kind*, one of the [`OUTPUT_KINDS`][specsolve._answer_layout.OUTPUT_KINDS], or why they cannot be read.

        Closed first, then not asked for, then the status, then, for a kind
        that exists only where the duals do, the duals' reason, and for a basis,
        [`NO_BASIS`][specsolve._answer_layout.NO_BASIS] where the
        solve ended on none.
        """
        what = f"the {kind.replace('_', ' ')} of '{name}'"
        if kind not in self._unclosed(what):
            raise SpecsolveError(not_requested_message(kind, name))
        frames = self._readable(kind, what)
        if kind in PRICED and self._no_duals is not None:
            raise SpecsolveError(self._no_duals)
        if kind in BASES and not frames:
            raise SpecsolveError(NO_BASIS)
        return frames

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
        self._readable('primal', 'an expression')
        return evaluated(self._expressions or {}, self._evaluate, expression)

    def _frame(self, name: str, kind: str) -> pl.DataFrame:
        """*name* through the reader *kind* names — the dispatch every bridge shares."""
        kind = checked_kind(kind)
        if kind in OUTPUT_KINDS:
            return self._output(kind, name)
        return {'primal': self.primal, 'dual': self.dual, 'expression': self.evaluate}[kind](name)

    def _names(self, kind: str) -> tuple[str, ...]:
        """Every name of *kind* this result can read — what a bridge takes by default.

        Raises:
            NoSolutionError: The solve left no values to read.
            SpecsolveError: This result was closed, *kind* is ``dual`` or
                ``reduced_cost`` and the duals are undefined, or *kind* is an
                output the solve was not asked for.
        """
        if checked_kind(kind) == 'primal':
            return tuple(self._readable('primal', 'the solution'))
        if kind == 'dual':
            frames = self._readable('dual', 'the duals')
            if self._no_duals is not None:
                raise SpecsolveError(self._no_duals)
            return tuple(frames)
        if kind in OUTPUT_KINDS:
            return tuple(self._carried(kind, 'anything'))
        self._readable('primal', 'the expressions')
        return tuple(self._expressions or {})

    def to_pandas(self, name: str, kind: str = 'primal') -> pd.DataFrame:
        """One name's values as a tidy `pandas.DataFrame`.

        *name* is read through the reader *kind* names: ``primal``, ``dual``,
        ``expression``, or the reader of an
        [`Output`][specsolve.types.Output] the solve was asked for. Needs pandas, which specsolve does not install; the xarray
        bridges need xarray too.
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
        [`Record`][specsolve._answer_layout.Record], with a null
        rather than ``nan`` objective where none was reached, and
        ``spec.yaml`` the [`spec`][] it answered. Beside them are
        ``primal/<name>.parquet`` per variable, ``dual/<name>.parquet`` per
        constraint where the duals are defined, ``expression/<name>.parquet``
        per named expression this data can evaluate, and
        ``<output>/<name>.parquet`` for each [`Output`][specsolve.types.Output]
        the solve was asked for, such as ``activity/`` per constraint.
        ``reasons.parquet`` holds ``(kind, name, reason)`` for whatever is
        deliberately left out — one row per failed expression, one with an
        empty *name* for the duals — and is absent when nothing is. A solve that left no values writes the
        record and the spec alone. The same model and data write the same bytes.

        ``format.json`` stamps the layout, the specsolve that wrote it and the
        outputs the answer carries:
        ``{"layout": 4, "specsolve": "…", "outputs": ["activity"]}``. Every reader refuses another
        layout with a [`LayoutError`][specsolve.errors.LayoutError] that says
        to solve the model again and save it.

        Whatever a previous save left in *directory* is removed first; files
        outside the layout are left alone.

        Returns:
            The directory.

        Raises:
            SpecsolveError: This result was closed, or this polars has no
                parquet.
        """
        parquet.require('Result.save')
        return self._save(Path(directory), write_whole)

    def _save(self, out: Path, write: FrameWriter) -> Path:
        """[`save`][], each table written with *write*: an archive passes the writer that adds its run."""
        import polars as pl

        carried = self._unclosed('the solution')
        clear_the_answer(out)
        write_format(out, asked_for(kind for kind in carried if kind in OUTPUT_KINDS))
        write_spec(out, self._spec)
        record = self.record._replace(specsolve_run=None)
        write(pl.DataFrame([record._asdict()], schema_overrides=RECORD_SCHEMA), out / RECORD_FILE)
        if not self._status.is_readable:
            return out
        held, no_expressions = self._all_frames()
        for kind, frames in held.items():
            for name, frame in frames.items():
                write(frame, out / kind / f'{name}.parquet')
        write_reasons(out, self._no_duals, {'expression': no_expressions}, write)
        return out

    def _all_frames(self) -> tuple[dict[str, Mapping[str, pl.LazyFrame]], dict[str, str]]:
        """``{kind: {name: frame}}`` for every kind this answer carries, and why each expression left out is.

        What [`save`][] writes and a sweep keeps of a slice. The declared
        expressions are evaluated now, under ``expression``, and one this data
        cannot evaluate has no frame and a reason instead.

        Raises:
            SpecsolveError: This result was closed.
        """
        expressions: dict[str, pl.LazyFrame] = {}
        no_expressions: dict[str, str] = {}
        for name, reader in (self._expressions or {}).items():
            try:
                expressions[name] = reader().lazy()
            except SpecsolveError as absent:
                no_expressions[name] = str(absent)
        return {**self._unclosed('the solution'), 'expression': expressions}, no_expressions

    def close(self) -> None:
        """Release this result's frames, and its hold on the build's label frames, early. Optional.

        Frames already read stay valid. The model and the solver are the
        [`Model`][specsolve._api.Model]'s to close.
        """
        self._frames = self._expressions = self._dual_rays = None
        self._evaluate = None

    def __enter__(self) -> Result:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


def _absent(reason: str) -> Callable[[], pl.DataFrame]:
    """A named expression's reader that raises *reason* at the read."""

    def read() -> pl.DataFrame:
        raise SpecsolveError(reason)

    return read


def load_result(directory: str | Path) -> Result:
    """Read back an answer [`Result.save`][] wrote — a solve, off disk.

    Every reader answers what it answered in the session that solved: the
    values, the duals, each named expression, the outputs the solve was asked
    for, and the reason behind anything the solve could not produce. No build
    or solver is needed.

    The solver's verbatim wording behind a refusal is not recorded — the
    termination condition is. A solve that reached no objective reads back as
    ``nan``.

    Args:
        directory: Where [`save`][specsolve._result.Result.save] wrote
            it. One inside an archive is
            [`load_archive`][specsolve._archive.load_archive]'s to find.

    Returns:
        The result, read into memory, so it owes *directory* nothing.
        [`scan_result`][] is the same answer left on disk.

    Raises:
        LayoutError: A directory holding no ``record.parquet``, or one whose
            layout has moved since it was written.
    """
    return _answer_under(Path(directory), whole=True)


def scan_result(directory: str | Path) -> Result:
    """The answer under *directory*, read as its readers are called rather than now.

    The same answer as [`load_result`][], but each frame is a
    `polars.scan_parquet` of its file, so an answer larger than memory is
    readable a name at a time, and unread names cost nothing.

    **The files have to outlive the result**: a name read after the
    directory is gone raises where the scan is collected, and a file
    rewritten underneath it comes back changed. *directory* is as
    [`load_result`][] takes it.

    Raises:
        LayoutError: As [`load_result`][] raises it.
    """
    return _answer_under(Path(directory), whole=False)


def _answer_under(out: Path, *, whole: bool) -> Result:
    """The saved answer under *out*, its frames read into memory now where *whole*, else scanned."""
    parquet.require('reading a saved answer')
    record_file = out / RECORD_FILE
    if not record_file.is_file():
        raise LayoutError(
            f'{str(out)!r} holds no {RECORD_FILE!r}, so it is not an answer save() wrote. Every one '
            f'carries that record whether or not the solve produced values.'
        )
    check_format(out)
    record = Record(**parquet.read(record_file).row(0, named=True))
    status = record.solve_status
    objective = float('nan') if record.objective is None else record.objective
    no_duals, absent = read_reasons(out)
    expressions: dict[str, Callable[[], pl.DataFrame]] = {
        name: (lambda frame=frame: frame.pipe(collected))
        for name, frame in saved_frames(out / 'expression', whole=whole).items()
    }
    expressions.update({name: _absent(why) for name, why in absent.get('expression', {}).items()})
    return Result(
        status,
        objective,
        {kind: saved_frames(out / kind, whole=whole) for kind in ('primal', 'dual', *kinds_of(read_outputs(out)))},
        expressions,
        _no_duals=no_duals,
        _spec=read_spec(out),
        _solved_at=record.solved_at,
        _run=record.specsolve_run,
        _provenance=record.provenance,
    )
