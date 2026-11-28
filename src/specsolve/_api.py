"""The runner: attach data to a YAML spec and execute it.

Five verbs take a spec: ``check``, ``build`` (YAML + sources → a [`Model`][]),
``solve``, ``write``, and ``evaluate`` for a spec with no variables.

Example::

    import specsolve as sps

    result = sps.solve(
        'spec.yaml',
        {'p_max': 'p_max.parquet', 'load': 'load.parquet', 'snapshot': range(8760)},
    )
    result.objective
    result.primal('p')  # tidy polars.DataFrame (coordinates..., value)
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from mathspec import advice

from specsolve import _parquet as parquet
from specsolve import _sinks as sinks
from specsolve._answer.outputs import checked_outputs
from specsolve._answer.records import METRICS_SCHEMA, Metrics, Provenance, installed_version
from specsolve._answer.result import Result
from specsolve._archive_layout import carried_sources, check_the_target, write_archive
from specsolve._clock import Clock, untimed
from specsolve._encoding import encoded
from specsolve._engine import readback
from specsolve._engine.assembly import Measured, numbers_only, spliced_model
from specsolve._engine.assembly import build as build_model
from specsolve._engine.attaching import attach
from specsolve._inputs import Label, Source, SpecLike, as_program, as_spec, expression_node
from specsolve._sinks.solver import Basis
from specsolve._sources import refuse_unknown_sources, tidy_sources
from specsolve._start import read_start, refuse_unknown_start_word
from specsolve.errors import SpecsolveError, SpecsolveWarning

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

    from mathspec.program import Expression, Program

    from specsolve._answer.outputs import Output
    from specsolve._answer.result import ConstraintRow, Diagnostics, InfeasibleSubsystem
    from specsolve._answer.status import SolveStatus
    from specsolve._attached import AttachedSources
    from specsolve._clock import Lap
    from specsolve._engine.assembly import BuiltModel
    from specsolve._start import Start

__all__ = ['build', 'check', 'evaluate', 'solve', 'tidy', 'write']


def check(spec: SpecLike) -> Program:
    """Parse, validate and compile a spec; attach no data.

    Every other verb reads the spec through this, so what this refuses they
    refuse too. Whether a sink takes the model is a fact about the build, and
    [`Model.check_sink`][specsolve.types.Model.check_sink] answers it with no solve.

    Args:
        spec: A YAML path, a mapping, or a ``Spec`` — what ``mathspec.to_spec``
            takes. A compiled ``Program`` is not taken. A ``piecewise:`` block
            is written out first: ``to_spec(spec).expand('piecewise')`` keeps
            every ``sos:`` set for a sink that takes one, and
            ``to_spec(spec).expand()`` writes the sets out as binaries every
            sink takes.

    Returns:
        The compiled program; typeset it or read its entries through
        `mathspec`. No verb takes it back; keep the
        ``Spec`` for that.

    Raises:
        LanguageError: A construct outside the language, a
            ``piecewise:`` block still to be written out, or a fragment that
            reads a name under ``given:`` — ``mathspec.merge`` composes it.
        SpecsolveError: Two entries whose names differ only by case, or a
            name that starts with ``specsolve_`` in any letter case, which is reserved.
        ValueError: A schema or expression that does not parse.

    Warns:
        SpecsolveWarning: Advice short of an error — a declared dimension nothing
            uses as an axis, a variable the objective drives to infinity with
            nothing to stop it. Issued here and nowhere else.
    """
    program = as_program(spec)
    for note in advice(program):
        warnings.warn(str(note), SpecsolveWarning, stacklevel=2)
    return program


def tidy(spec: SpecLike, sources: Mapping[str, Source]) -> dict[str, pl.DataFrame]:
    """The tables a solve of *spec* attaches from *sources*, one per name the spec declares.

    Every source is read and checked as [`build`][] does. The tables are what
    an archive holds under ``sources/``, less its ``specsolve_run`` column, so
    a returned table goes back in as a source unchanged.

    Args:
        spec: As [`check`][] takes it.
        sources: As [`build`][] takes them.

    Returns:
        Each dimension as ``(dim, specsolve_position)``: its labels once each,
        and the ``Int64`` position from 0 that ``shift`` counts. The labels
        of a ``str`` dimension are ``String``, whatever encoding they arrived
        in. Each parameter as ``(dims…, value)`` and each relation as the
        columns it declares. Every other column is in the type it arrived in,
        a datetime in microseconds.

    Raises:
        LanguageError: A construct outside the language.
        DataError: As [`build`][] refuses the sources.
    """
    return attach_sources(as_program(spec), sources)[0]


def attach_sources(
    program: Program,
    sources: Mapping[str, Source],
    lap: Lap = untimed,
    last: tuple[Mapping[str, pl.DataFrame], AttachedSources] | None = None,
    changed: Collection[str] = (),
) -> tuple[dict[str, pl.DataFrame], AttachedSources]:
    """*sources* read, held to what *program* declares and attached to it: the one way a caller's data reaches a build.

    Args:
        program: The compiled spec.
        sources: The caller's sources, as [`build`][] takes them.
        lap: Ends a lap at each step of the ``tidy`` and ``attach`` phases, as
            [`timings`][specsolve._api.Model.timings] names them.
        last: What an earlier call returned, before the sources named
            *changed* were replaced. A parameter that replacement leaves alone
            ([`_unchanged`][]) is taken from it as it is: neither read nor
            checked again, and with no lap. Dimensions, relations and the
            assumptions are read and checked every time.
        changed: The names whose sources were replaced since *last*.

    Returns:
        The tables as they were read, which [`tidy`][] returns and an archive
        holds, and the same data as a build reads it.

    Raises:
        DataError: As [`build`][] refuses the sources.
    """
    read, held = last if last is not None else (None, None)
    kept = () if last is None else _unchanged(program, changed)
    tables = tidy_sources(program, sources, lap, read, kept)
    return tables, attach(program, encoded(program, tables, lap, held, kept), lap)


def _unchanged(program: Program, changed: Collection[str]) -> frozenset[str]:
    """The parameters of *program* that a replacement of the sources named *changed* leaves as they were checked.

    A parameter's checks read its own source and the labels of each of its
    dimensions, so it is unchanged only where none of them is in *changed*.
    """
    return frozenset(
        name for name, p in program.parameters.items() if name not in changed and not set(p.dims) & set(changed)
    )


def _refuse_a_decision(program: Program) -> None:
    """Refuse variables, constraints or an objective: [`evaluate`][] is arithmetic, not a solve."""
    declared = []
    if program.variables:
        declared.append(f'variables ({", ".join(sorted(program.variables))})')
    if program.constraints:
        declared.append(f'constraints ({", ".join(sorted(program.constraints))})')
    if program.objective is not None:
        declared.append('an objective')
    if not declared:
        return
    raise SpecsolveError(
        f'evaluate takes a spec with no variables — dimensions, parameters, relations and expressions, '
        f'evaluated as arithmetic. This one declares {", ".join(declared)}, which makes it a problem to solve: a '
        f'variable has no value until a solver picks one. Solve it with sps.solve(spec, sources), or '
        f'drop the decision to evaluate the arithmetic that remains.'
    )


def evaluate(spec: SpecLike, sources: Mapping[str, Source], expression: str | Mapping[str, object]) -> pl.DataFrame:
    """The value of *expression* over a spec with no variables — arithmetic, no solver.

    A spec with only dimensions, parameters, relations and ``expressions:``
    is a calculation: each expression reads only the attached data. This
    attaches *sources* and values one expression, the way
    [`evaluate`][specsolve._answer.result.Result.evaluate] does at a
    solution. The spec is read as [`solve`][] reads it; one that declares
    variables belongs there.

    Args:
        spec: As [`check`][] takes it.
        sources: As [`build`][] takes them.
        expression: What one ``expressions:`` entry takes — a name the spec
            declares, an expression string, or the mapping carrying ``cases:``
            with ``dims:`` and ``otherwise:``.

    Returns:
        The value, ``(dims…, value)`` over the expression's own dimensions. Only
        this expression's query is built.

    Raises:
        LanguageError: A construct outside the language, or a name
            the spec does not declare.
        SpecsolveError: A spec that declares variables, constraints or an
            objective.
        DataError: A source that is missing, unreadable, or the wrong shape,
            or a divisor with no value where the expression divides.
    """
    spec = as_spec(spec)
    program = as_program(spec)
    _refuse_a_decision(program)
    _, attached = attach_sources(program, sources)
    return readback.expressions_of(program, attached, lambda written: expression_node(spec, written)).value(expression)


class Model:
    """A spec with your data attached to it — what [`build`][] returns.

    ``check`` → ``Program`` (the math) → ``build`` → ``Model`` (the math with
    your data) → ``solve`` → ``Result`` (one answer).

    One build feeds any number of [`solve`][] and [`write`][] calls;
    [`update`][] puts new numbers on it without re-reading the YAML, and
    [`diagnostics`][] says what it did. Nothing has to be released;
    [`close`][] hands a large model back early.
    """

    def __init__(self, spec: SpecLike, sources: Mapping[str, Source]) -> None:
        #: Every lap of this model's life, and the solves each of its runs took.
        self._clock = Clock()
        self._spec = as_spec(spec)
        self._program = as_program(self._spec)
        self._clock.lap('load')
        self._sources = dict(sources)
        #: What the last build read, as [`tidy_sources`][] gave it.
        self._tidied: dict[str, pl.DataFrame] = {}
        #: The build, or ``None`` where there is not one — closed, released by
        #: an update that raised, or never run.
        self._built: BuiltModel | None = None
        #: What the last build measured about itself. Outlives ``_built``.
        self._measured = Measured()
        #: The solver holding this model, kept between solves and across rebuilds.
        self._solver: sinks.Solver | None = None
        #: What the last solve concluded, or ``None`` once a load, a build or
        #: [`close`][] has changed what the solver holds.
        self._solved: SolveStatus | None = None
        self._fill()

    def _lower(self, written: str | Mapping[str, object]) -> Expression:
        """One unnamed expression as a program node, for a result reading a quantity the file never named.

        Held here because nothing under ``_engine/`` may see the model as
        written (docs/development/architecture.md, hard rule 2).
        """
        return expression_node(self._spec, written)

    def _standing(self, doing: str) -> BuiltModel:
        """The built model, or why there is not one *doing* what the caller asked."""
        if self._built is None:
            raise SpecsolveError(
                f'there is no built model {doing}: it was closed, or an update raised and released '
                f'it rather than leaving half of one behind. Build it again — update() with data it can '
                f'attach, or build() from the start.'
            )
        return self._built

    def _fill(self, changed: Collection[str] = ()) -> None:
        """Build from what is attached now; a failure closes the model rather than leaving it stale.

        What the build read is kept for an archive: a one-shot iterator is
        spent by then, and a path may have been rewritten. A parameter that
        the sources named *changed* leave alone is taken as the standing build
        read it ([`attach_sources`][]); with no standing build, as after a
        failure, every source is read again. A change of numbers alone
        ([`numbers_only`][specsolve._engine.assembly.numbers_only]) reads
        again only the bounds, constant sides and objective it feeds
        ([`spliced_model`][specsolve._engine.assembly.spliced_model]);
        anything else builds in full. The held solver
        reads its [`structure`][specsolve._sinks.solver.Solver.structure]
        before the previous build is released, since that read lets go of its
        tables.
        """
        lap = self._clock.lap
        held, measured = self._built, self._measured
        last = None if held is None else (self._tidied, held.attached)
        try:
            self._tidied, attached = attach_sources(self._program, self._sources, lap, last, changed)
            if self._solver is not None:
                self._solver.structure()
                lap('handoff', 'digest')
            self._built = None
            self._solved = None
            if held is not None and changed and numbers_only(self._program, held, measured, attached, changed):
                self._built = spliced_model(held, attached, changed, measured, lap)
                return
            self._measured = Measured()
            self._built = build_model(self._program, attached, self._measured, lap)
        except BaseException:
            self.close()
            raise

    def update(self, sources: Mapping[str, Source]) -> Model:
        """Put new numbers on the same model, in place.

        ::

            model.update({'cap_hat': capacity}).solve()

        ``model.update(x)`` answers what ``build(spec, sources | x)`` answers,
        whatever changed. Data that changes which coordinates a condition admits
        renumbers solver indices, so the model is rebuilt and solved cold rather
        than pushed onto a loaded solver;
        [`solver_instances`][specsolve._answer.records.Metrics.solver_instances] says which ran.

        Results taken before the update keep reading their own tables, and
        keep their build's index tables alive until dropped or until
        [`close`][specsolve._answer.result.Result.close] is called. A sweep, a
        rolling horizon or a myopic pathway is
        [`solve_over`][specsolve._sweep.solve_over.solve_over], which runs this loop.

        A parameter whose own source and whose dimensions' labels *sources*
        leaves alone is not read or checked again: a file rewritten in place at
        a path it was built from is read once its name is passed again.

        Args:
            sources: Only what changed; the rest keeps what [`build`][]
                attached. A dimension's labels count too, which is how a
                coordinate set grows.

        Returns:
            This object.

        Raises:
            DataError: A name the spec does not declare, refused before
                anything changes. Data the build refuses releases the model, as
                a build that raises does.
        """
        self._clock.start()
        refuse_unknown_sources(self._program, sources)
        self._sources.update(sources)
        self._fill(sources.keys())
        return self

    def solve(
        self,
        solver_name: str = 'highs',
        *,
        solver_options: Mapping[str, object] | None = None,
        also_record: Sequence[str] | None = None,
        archive: str | Path | None = None,
        outputs: Iterable[Output] = (),
        start: Result | Start | Literal['previous'] | None = 'previous',
    ) -> Result:
        """Hand the built model to a solver and solve it.

        A solver that can stay loaded is kept between calls, so an updated
        model pushes only its numbers; [`diagnostics`][] counts the solves
        that loaded it again.

        Args:
            solver_name: ``highs``, which ships with the package; ``gurobi``,
                which needs the ``[gurobi]`` extra; or ``xpress``, which needs
                the ``[xpress]`` extra.
            solver_options: Forwarded to the solver verbatim, in its own
                vocabulary, so a time limit is ``time_limit``, ``TimeLimit`` or
                ``timelimit``. Gurobi's are applied when its environment is
                created, so ``ComputeServer``, ``TokenServer`` and
                ``WLSAccessID`` reach it too. The result's
                [`provenance`][specsolve._answer.result.Result.provenance]
                records the value of an option that changes the answer, such
                as a time limit or a gap, and the name alone of any other.
            also_record: Option names whose value the result records as
                well, beside the ones the solver records already, in any
                letter case. Name no credential here: an archive goes to
                shared storage.
            archive: Where to write the spec, the data attached to it **now**,
                and this answer, so that
                [`read_archive`][specsolve._archive.read_archive] gives all
                three back and the model solves again from the file alone. A
                ``.zip`` suffix packs it into one file; anything else is a
                directory. What the build and its solves spent goes in beside
                the answer, as
                [`Metrics`][specsolve._answer.records.Metrics]. Each
                source goes in as the table [`tidy`][] returns, with
                ``specsolve_run`` added, and members are stored uncompressed.
                The answer returned carries the archived
                [`sources`][specsolve._answer.result.Result.sources]: the paths
                under ``sources/`` for a directory, the tables for a zip.
            outputs: Each [`Output`][specsolve.types.Output] the answer
                carries beside the primal, the duals and the declared
                expressions: ``activity`` for each constraint's left-hand side,
                ``reduced_cost`` for each variable's reduced cost, ``slack``
                for each constraint's distance to binding, and ``basis`` for
                the basis status each variable and constraint ended on, read
                with ``variable_basis`` and ``constraint_basis``. The result and its archive
                carry these and nothing else, and the reader of one not asked
                for refuses.
            start: What the solve begins from; it changes how the solver gets
                to the optimum, never which one. ``'previous'``, the default,
                carries on from the last solve while the solver stays loaded,
                and begins from nothing after an update that loads it again.
                ``None`` begins from nothing. An earlier answer, live or read
                from an archive, or a
                [`Start`][specsolve.types.Start] of tables keyed by reader, is
                matched by coordinate, so one from another build of the spec,
                with rows or columns gained or lost, starts it too. An LP starts from a basis where one is given, which an
                answer carries when solved with ``outputs={'basis'}``: a
                coordinate it leaves out starts at a bound if it is a
                variable's, and not binding if it is a constraint's. Otherwise
                an LP, and a mixed-integer model always, starts from values,
                which the solver completes and repairs. Where a solver takes
                values for an LP and no gain from them is known, the solve
                warns.

        Returns:
            The solution, holding this model.

        Raises:
            SpecsolveError: A solver name nothing serves, one this environment
                cannot run, a bare string as *also_record* or *outputs*, a
                name in *outputs* that is not an output, a *start* word other
                than ``'previous'``, or a *start* this
                model or solver cannot start from: a key that names no reader, a table naming no entry or
                lacking its dimensions, a basis status outside the five words, a
                basis alone for a mixed-integer model, a start that lands on
                no coordinate, or values for an LP that the solver cannot
                take.
            LayoutError: An *archive* directory that already holds something,
                refused before the solve.
        """
        self._clock.start()
        out = None if archive is None else Path(archive)
        if out is not None:
            parquet.require('archive=')
            check_the_target(out)
        if isinstance(also_record, str):
            raise SpecsolveError(
                f'also_record={also_record!r} is one string, which would name each of its letters. '
                f'Pass a list: also_record=[{also_record!r}].'
            )
        asked = checked_outputs(outputs)
        refuse_unknown_start_word(start)
        read = start if start is None or isinstance(start, (str, Result)) else self._read_start(start)
        held, reloaded = self._handoff(solver_name, solver_options, start=read)
        self._clock.count_a_solve(reloaded=reloaded)
        model = self._standing('to hand over')
        vectors = held.run(model.tables, basis='basis' in asked)
        self._clock.lap('solve')
        self._solved = vectors.status
        solved = readback.result(model, vectors, solver_name, self._lower, asked)
        answered = self._with_record_fields(solved, solver_name, solver_options, also_record or ())
        self._clock.lap('readback')
        answered = answered._taking(self._share())
        if out is not None:
            answered = self._archive(out, answered)
        self._clock.next_run()
        return answered

    def _handoff(
        self,
        solver_name: str,
        solver_options: Mapping[str, object] | None,
        *,
        start: Result | Mapping[str, Mapping[str, pl.LazyFrame]] | Literal['previous'] | None = 'previous',
    ) -> tuple[sinks.Solver, bool]:
        """The handoff phase whole: [`solve`][] up to the run, lapped as ``handoff``.

        *start* is matched, the sink checked, the solver loaded or the new
        numbers pushed onto the one held, as [`loaded`][specsolve._sinks.loaded]
        decides, and the start given. A solver kept forgets its last run unless
        *start* is ``'previous'``. Counts no solve, so a benchmark that times
        this alone leaves the counters true.

        Returns:
            The solver, and whether it was loaded again.

        Raises:
            SpecsolveError: A *start* this model or this solver cannot start
                from, or a sink that cannot take the built model.
        """
        lap = self._clock.lap
        model = self._standing('to hand over')
        matched = None if start is None or isinstance(start, str) else readback.matched_start(model, start, solver_name)
        lap('handoff', 'match')
        self.check_sink(solver_name)
        self._solved = None
        lap('handoff', 'capability')
        held = self._solver
        self._solver = sinks.loaded(held, solver_name, model.tables, solver_options)
        reloaded = self._solver is not held
        lap('handoff', 'solver_instance' if reloaded else 'push')
        if not reloaded and start != 'previous':
            self._solver.forget()
        if isinstance(matched, Basis):
            self._solver.start_from_basis(matched)
        elif matched is not None:
            self._solver.start_from_values(matched)
        lap('handoff', 'start')
        return self._solver, reloaded

    def _share(self) -> Metrics:
        """What the model took since its last answer was handed back: the answer about to be handed back's row."""
        return self._metrics(run=self._clock.run)

    def _metrics(self, run: int | None = None) -> Metrics:
        """The last build's sizes with the clock's counters and clocks, over the whole life or over *run*."""
        measured = self._measured
        return self._clock.metrics(measured.columns, measured.rows, measured.nonzeros, run)

    def _with_record_fields(
        self,
        answer: Result,
        solver_name: str,
        solver_options: Mapping[str, object] | None,
        also_record: Sequence[str],
    ) -> Result:
        """*answer* as [`solve`][] hands it back: the vectors laid out, with what only this model knows of the solve.

        That is the record fields for when the solve ended and its provenance,
        and the spec an archive writes beside them; the read-back fills the
        status and the objective. Every step between the read-back and the
        caller lives here, so the read benchmark, which builds an answer
        without a solver, pays what a solve pays.
        """
        provenance = _provenance(solver_name, solver_options, also_record)
        record = answer._answer.record._replace(solved_at=datetime.now(UTC), **provenance._asdict())
        return replace(answer, _spec=self._spec, _answer=replace(answer._answer, record=record))

    def _read_start(self, start: Start) -> dict[str, dict[str, pl.LazyFrame]]:
        """*start*'s tables read against this build, as [`read_start`][] reads them.

        Raises:
            SpecsolveError: As [`read_start`][] raises.
        """
        return read_start(start, self._program, self._tidied, self._standing('to start').attached.dimensions)

    def _archive(self, out: Path, answered: Result) -> Result:
        """Write this model, what is attached to it now, and *answered* to *out*; *answered* with the row it recorded.

        The metrics row is the last member written, so its ``archive_seconds``
        covers every other one; writing and landing it, and handing *answered*
        the sources the archive carries, are in its run of [`timings`][] and in
        [`metrics`][], not in the row.
        """
        recorded = self._share()

        def metrics() -> pl.DataFrame:
            nonlocal recorded
            self._clock.lap('archive')
            recorded = self._share()
            return pl.DataFrame([recorded._asdict()], schema_overrides=METRICS_SCHEMA)

        tables = self._tidied
        write_archive(out, self._spec, tables, answer=answered._save, metrics=metrics)
        landed = replace(answered._taking(recorded), _sources=carried_sources(out, tables))
        self._clock.lap('archive')
        return landed

    def write(self, path: str | Path) -> None:
        """Stream the built model to *path*, in the format its suffix names.

        Raises:
            ValueError: A suffix nothing writes.
            SpecsolveError: A construct the format has no section for, as
                [`check_sink`][] refuses it.
        """
        self._clock.start()
        path = Path(path)
        suffix = path.suffix.lower()
        chosen = sinks.writer(suffix)
        self.check_sink(suffix)
        chosen.write(self._standing('to hand over').tables, path)
        self._clock.lap('write')

    def check_sink(self, sink: str) -> None:
        """Refuse the built model where *sink* cannot take it; no solve, no file.

        ::

            sps.build('dispatch.yaml', sources).check_sink('highs')

        The answer is read off the built model, not the file: a square the
        data prices at zero, an integer variable with no built column or a set
        with no members asks for nothing. [`solve`][] and [`write`][] refuse
        exactly what this refuses, with the same message.

        Args:
            sink: A solver name (``highs``, ``gurobi``, ``xpress``) or an
                output suffix (``.lp``, ``.mps``).

        Raises:
            SpecsolveError: A construct the sink cannot take, naming it and the
                sinks that do; or a name belonging to no sink.
        """
        if (refused := sinks.refusal(self._standing('to hand over').tables, sink)) is not None:
            raise SpecsolveError(refused)

    def row(self, name: str, /, **coordinate: Label) -> ConstraintRow:
        """One built constraint row at one coordinate — its terms, sense and right-hand side.

        The verb for *this row is wrong and I do not know why*. It reads the
        **built** model and needs no solve: a term whose variable was absent
        is missing, and a row a ``where`` leaves absent is not there at all. A
        column has no reader: a variable's bounds are in the spec, and its
        coefficients are this read transposed.

        Args:
            name: A declared constraint. Positional, so a dimension may be
                called ``name``.
            coordinate: One label per dimension of that constraint, all of them.

        Returns:
            The terms as ``(variable, coordinate, coefficient)``, beside the
            comparison and the right-hand side.

        Raises:
            KeyError: No constraint is called *name*.
            SpecsolveError: The coordinate names the wrong dimensions, holds a label
                its dimension cannot hold, matches no row the build produced,
                or the model has been closed.

        Example:
            >>> print(model.row('balance', snapshot=1))  # doctest: +SKIP
            balance[snapshot=1]: +1 p[snapshot=1, tech=wind] +50 p[snapshot=1, tech=gas] >= 60
        """
        return readback.row(self._standing(f"to read '{name}' out of"), name, coordinate)

    def infeasible_subsystem(self) -> InfeasibleSubsystem:
        """The rows and bounds that make the last solve infeasible, by entry and coordinate.

        The verb for *this model has no solution and I do not know why*. It
        asks the solver that ran the last solve for an IIS (irreducible
        infeasible subsystem). The search runs only on this call, since it can
        cost more than the solve, and under the solve's ``solver_options``, so
        a solver's IIS settings and time limit go there. Read a member row's
        terms with [`row`][].

        Every sink finds one for a linear model. For a discrete model, every
        sink leaves out the integrality that makes the rows and bounds
        conflict, and ``highs`` can find none where integrality causes the
        conflict.

        Raises:
            SpecsolveError: The model has not been solved since it was built,
                updated or closed; the last solve was not infeasible; or the
                solver found no subsystem, or stopped at a limit before it was
                irreducible.

        Example:
            >>> model.solve().termination_condition  # doctest: +SKIP
            'infeasible'
            >>> print(model.infeasible_subsystem())  # doctest: +SKIP
            balance[snapshot=1] == 200
            p[snapshot=1, tech=gas] <= 100 (upper bound)
            p[snapshot=1, tech=wind] <= 50 (upper bound)
        """
        if self._solved is None:
            raise SpecsolveError(
                'there is no solve to explain: this model has not been solved since it was built, '
                'updated or closed. Solve it, and ask infeasible_subsystem() while the last solve is the infeasible one.'
            )
        if self._solved.termination_condition != 'infeasible':
            raise SpecsolveError(
                f'an infeasible subsystem explains why a model has no solution, and the last solve '
                f'terminated {self._solved.termination_condition!r} rather than infeasible.'
            )
        assert self._solver is not None, 'a solve leaves a solver, and whatever drops it drops the status too'
        model = self._standing('to hand over')
        found = self._solver.infeasible_subsystem()
        if found is None:
            raise SpecsolveError(_no_infeasible_subsystem_message(self._solver.name, readback.discrete(model)))
        return readback.infeasible_subsystem(model, found)

    def _evaluator(
        self,
        primals: Mapping[str, pl.DataFrame],
        duals: Mapping[str, pl.DataFrame] | None,
        no_duals: str | None,
    ) -> Callable[[str | Mapping[str, object]], pl.DataFrame]:
        """An ad-hoc expression reader over a *saved* solution, put back against this build.

        What an archive and a sweep hand [`evaluate`][specsolve._answer.result.Result.evaluate]
        for a quantity the file never named: the saved tables are laid back in
        this build's index order, and the reader is the one a live solve gives.
        A build, never a solve.

        Args:
            primals: The saved ``(dims…, value)`` table per variable.
            duals: The same per constraint, or ``None`` where the solve left none.
            no_duals: Why there are no duals, raised at a dual read.
        """
        return readback.evaluator(self._standing('to hand over'), primals, duals, no_duals, self._lower)

    def metrics(self) -> Metrics:
        """This model's size, counters and clocks over its whole life, taken now.

        ``solves`` counts every solve, and each clock sums every lap of its
        phase in [`timings`][]. A result's
        [`metrics`][specsolve.types.Result.metrics] is its run's, frozen when
        it was handed back, so an archive's landing, which comes after, is in
        the run here and not in the result. Answerable after [`close`][], and
        after a build that raised, which leaves the sizes at zero.
        """
        return self._metrics()

    def timings(self) -> pl.DataFrame:
        """Every lap of this model's life, one row each: ``(run, phase, step, entry, seconds)``.

        Each lap begins where the one before it ended, so the rows of a phase
        add up to its ``*_seconds`` in [`metrics`][], and the rows of a run to
        what the model took between handing back two results. ``run`` counts
        the results handed back before the lap: ``0`` up to the first.
        ``step`` says what a lap did inside a phase that has steps, and
        ``entry`` names the entry it built. The build's steps are
        ``specialize``, the program rewritten with what the data settled;
        ``variable``, ``sos`` and ``constraint``, one row per entry; the
        ``objective``, which has no name; ``matrix``, every constraint's share
        stacked into one matrix in row order; and ``sink_tables``, the rest of
        what a solver is handed. An entry's seconds are its own: they hold when the
        entries build in another order. An update that gives only new numbers
        logs ``splice``, the entries it reads again and ``sink_tables``. The tidy's steps are ``read``, every
        path read at once, and ``shape``, one row per dimension, relation and
        parameter, each held to its columns and their types, the first
        parameter's also holding every parameter's shaping, and a last row
        with no entry for the refusal that needs every source read. The
        attach's steps are ``encode``, one row per relation and parameter,
        its labels turned into its dimensions' own type; ``content``, one row
        per relation and parameter, checked to hold only its dimensions'
        labels, no coordinate twice, and a row wherever it refuses to lack
        one; ``fill``, the missing rows a divisor reads written in;
        ``settle``, each condition the data decides read once; and
        ``assumptions``, the spec's assumptions checked. An update logs a
        ``shape``, ``encode`` and ``content`` row only for a parameter it
        reads again. The handoff's steps are ``digest``, the held solver's fingerprint of its
        model, read before a rebuild;
        ``match``, the ``start=`` matched to this build; ``capability``, the
        solver checked to take the model; ``push``, the held solver kept and the new
        numbers pushed onto it, or ``solver_instance``, a new solver instance
        given the model, the held one closed, as
        [`solver_instances`][specsolve._answer.records.Metrics.solver_instances]
        counts them; and ``start``, the start given to the solver, or its last
        solve forgotten. The same steps for every
        solver. Elsewhere both are null.
        """
        return self._clock.timings()

    def diagnostics(self) -> Diagnostics:
        """What the build looks like that the answer does not show: what it left out and how its numbers spread.

        Answerable after [`close`][], and after a build that raised. A raise
        leaves the sizes at zero, since they are taken once a model is whole,
        and everything measured before it stands.
        """
        return self._measured.diagnostics()

    def close(self) -> None:
        """Release the built model, and any solver still holding it. A [`Result`][] keeps its own tables."""
        if self._solver is not None:
            self._solver.close()
            self._solver = None
        self._built = None
        self._solved = None

    def __enter__(self) -> Model:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


def build(spec: SpecLike, sources: Mapping[str, Source]) -> Model:
    """Attach *sources* to *spec* and build it — the model with your data on it.

    Args:
        spec: As [`check`][] takes it.
        sources: Parameter names to parquet paths or in-memory tables, and
            dimension names to their labels — a table of labels, a parquet path,
            or a bare sequence — wherever the YAML declares none. The shapes a
            value may take, and what attaching refuses, are
            [the data contract](https://specsolve.readthedocs.io/en/latest/reference/data/).
            Each is read once and kept without a copy, so it must not change
            while the model or its answers are in use; change data through
            [`Model.update`][].

    Returns:
        The built model.

    Raises:
        LanguageError: A construct outside the language.
        DataError: A source that is missing, unreadable, or the wrong shape.
    """
    return Model(spec, sources)


def solve(
    spec: SpecLike,
    sources: Mapping[str, Source],
    solver_name: str = 'highs',
    *,
    solver_options: Mapping[str, object] | None = None,
    also_record: Sequence[str] | None = None,
    archive: str | Path | None = None,
    outputs: Iterable[Output] = (),
    start: Result | Start | None = None,
) -> Result:
    """Build *spec* and solve it in one call.

    To solve the same spec again with new numbers, use [`build`][] and
    [`Model.update`][].

    Args:
        spec: As [`check`][] takes it.
        sources: As [`build`][] takes them.
        solver_name: As [`Model.solve`][] takes it.
        solver_options: As [`Model.solve`][] takes them.
        also_record: As [`Model.solve`][] takes them.
        archive: As [`Model.solve`][] takes it — a ``.zip``, or a directory.
        outputs: As [`Model.solve`][] takes them.
        start: As [`Model.solve`][] takes it.

    Returns:
        The solution. It owns its tables; the model and the solver are
        released before this returns.

    Raises:
        SpecsolveError: A solver name nothing serves, or *outputs* that
            [`Model.solve`][] refuses — both checked before the build — or as
            [`Model.solve`][] raises.
    """
    sinks.solver(solver_name)
    checked_outputs(outputs)
    model = build(spec, sources)
    try:
        return model.solve(
            solver_name,
            solver_options=solver_options,
            also_record=also_record,
            archive=archive,
            outputs=outputs,
            start=start,
        )
    finally:
        model.close()


def write(
    spec: SpecLike,
    sources: Mapping[str, Source],
    out: str | Path,
) -> Path:
    """Build *spec* and stream it to *out*, in the format its suffix names.

    Args:
        spec: As [`check`][] takes it.
        sources: As [`build`][] takes them.
        out: Where to write; ``.lp`` and ``.mps`` ship, and name a model's
            columns and rows the same way.

    Returns:
        The path written.

    Raises:
        ValueError: A suffix nothing writes — checked before the build.
        SpecsolveError: A construct the format has no section for, read off the
            built model, naming the sinks that take it.
    """
    out = Path(out)
    sinks.writer(out.suffix.lower())
    with build(spec, sources) as model:
        model.write(out)
    return out


def _json_value(value: object) -> object:
    """*value* as strict JSON holds it, which is all a BI tool or polars' ``str.json_decode`` reads.

    A numpy scalar becomes the Python number it holds. A non-finite float
    becomes the string ``"inf"``, ``"-inf"`` or ``"nan"``, because JSON has no
    such number and ``time_limit=inf`` is HiGHS's own default.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _provenance(
    solver_name: str, solver_options: Mapping[str, object] | None, also_record: Sequence[str] = ()
) -> Provenance:
    """What a solve on *solver_name* with *solver_options* records about itself.

    The options are written as one JSON object, because a column of structs is
    one that several BI tools cannot read. Only an option on the solver's
    ``recorded_options`` keeps its value: an archive goes to storage other
    people read, and a list of what to hide would leak whatever it missed.
    """
    served = sinks.solver(solver_name)
    recorded = served.recorded_options | {name.casefold() for name in also_record}
    options = {
        name: _json_value(value) if name.casefold() in recorded else '<not recorded>'
        for name, value in (solver_options or {}).items()
    }
    return Provenance(
        solver_name,
        installed_version(served.requires[0]),
        json.dumps(options, sort_keys=True, default=str, allow_nan=False),
        installed_version('specsolve'),
        installed_version('mathspec'),
    )


def _no_infeasible_subsystem_message(solver_name: str, discrete: Sequence[str]) -> str:
    """Why an infeasible solve's solver found no subsystem."""
    if discrete and solver_name == 'highs':
        return (
            f'HiGHS found no subsystem, and this model declares integer variables '
            f'({", ".join(discrete)}), so integrality may be what conflicts, which HiGHS can search '
            f'without. Solve with gurobi or xpress, which search with it.'
        )
    return (
        f'the model is infeasible, and the {solver_name} sink found no subsystem that explains it. The '
        f"search runs under the solve's own solver_options, so a time limit there can stop it short."
    )
