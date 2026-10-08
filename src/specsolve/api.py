"""The runner: attach data to a YAML spec and execute it.

Five verbs take a spec: ``check``, ``build`` (YAML + sources → a [`Model`][]),
``solve``, ``write``, and ``evaluate`` for a spec with no variables.
``load_result`` reads back an answer [`Result.save`][] wrote, and
``scan_result`` leaves it on disk.

Example::

    import specsolve as sps

    result = sps.solve(
        'spec.yaml',
        {'p_max': 'p_max.parquet', 'load': 'load.parquet', 'snapshot': range(8760)},
    )
    result.objective
    result.primal('p')  # tidy polars.DataFrame (coords..., value)
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

from specsolve.archive_layout import beside, check_the_target, write_archive
from specsolve.errors import (
    LayoutError,
    SpecsolveError,
    SpecsolveWarning,
)
from specsolve.inputs import Buildable, Label, Source, declared, lower, lowered
from specsolve.relational.answer_layout import (
    METRICS_FILE,
    METRICS_SCHEMA,
    RECORD_FILE,
    Provenance,
    Record,
    check_format,
    checked_outputs,
    digest_of,
    installed,
    kinds_of,
    read_outputs,
    read_reasons,
    saved_frames,
    write_whole,
)
from specsolve.relational.collect import collected
from specsolve.relational.engine.engine import Engine, expression_readers
from specsolve.relational.result import Result, evaluated
from specsolve.relational.sinks import solver, writer
from specsolve.sources import numbered, read_start, refuse_unknown_sources, tidy_sources

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from mathspec.program import Expression, Program

    from specsolve.relational.answer_layout import Output
    from specsolve.relational.result import ConstraintRow, Diagnostics, Start

__all__ = ['build', 'check', 'evaluate', 'load_result', 'scan_result', 'solve', 'tidy', 'write']


def check(spec: Buildable) -> Program:
    """Parse, validate and lower a spec; attach no data.

    Every other verb reads the spec through this, so what this refuses they
    refuse too. Whether a sink takes the model is a fact about the build, and
    [`Model.check`][specsolve.types.Model.check] answers it with no solve.

    Args:
        spec: A YAML path, a mapping, or a ``Spec`` — what ``mathspec.to_spec``
            takes. A lowered ``Program`` is not taken. A ``piecewise:`` block
            is written out first: ``to_spec(spec).expand('piecewise')`` keeps
            every ``sos:`` set for a sink that takes one, and
            ``to_spec(spec).expand()`` writes the sets out as binaries every
            sink takes.

    Returns:
        The lowered program, for reading the plan; typeset it or read its
        declarations through `mathspec`. No verb takes it back; keep the
        ``Spec`` for that.

    Raises:
        LanguageError: A construct outside the streaming language, a
            ``piecewise:`` block still to be written out, or a fragment that
            reads a name under ``given:`` — ``mathspec.merge`` composes it.
        SpecsolveError: Two declarations whose names differ only by case, or a
            name that starts with ``specsolve_`` in any letter case, which is reserved.
        ValueError: A schema or expression that does not parse.

    Warns:
        SpecsolveWarning: Advice short of an error — a declared dimension nothing
            uses as an axis, a variable the objective drives to infinity with
            nothing to stop it. Issued here and nowhere else.
    """
    program = lowered(spec)
    for note in advice(program):
        warnings.warn(str(note), SpecsolveWarning, stacklevel=2)
    return program


def tidy(spec: Buildable, sources: Mapping[str, Source]) -> dict[str, pl.DataFrame]:
    """The tables a solve of *spec* attaches from *sources*, one per name the spec declares.

    Every source is read and checked as [`build`][] does. The tables are what
    an archive holds under ``sources/``, less its ``specsolve_run`` column, so
    a returned table goes back in as a source unchanged.

    Args:
        spec: As [`check`][] takes it.
        sources: As [`build`][] takes them.

    Returns:
        Each dimension as ``(dim, specsolve_position)``: its labels once each,
        and the ``Int64`` position from 0 that ``shift`` counts. Each
        parameter as ``(dims…, value)``. Each relation as the columns it
        declares.

    Raises:
        LanguageError: A construct outside the streaming language.
        DataError: As [`build`][] refuses the sources.
    """
    program = lowered(spec)
    return {name: table.pipe(collected) for name, table in numbered(program, tidy_sources(program, sources)).items()}


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


def evaluate(spec: Buildable, sources: Mapping[str, Source], expression: str | Mapping[str, object]) -> pl.DataFrame:
    """The value of *expression* over a spec with no variables — arithmetic, no solver.

    A spec with only dimensions, parameters, relations and ``expressions:``
    is a calculation: each expression reads only the attached data. This
    attaches *sources* and values one expression, the way
    [`evaluate`][specsolve.relational.result.Result.evaluate] does at a
    solution. The spec is read as [`solve`][] reads it; one that declares
    variables belongs there.

    Args:
        spec: As [`check`][] takes it.
        sources: As [`build`][] takes them.
        expression: What one ``expressions:`` entry takes — a name the spec
            declares, an expression string, or the mapping carrying ``cases:``
            with ``dims:`` and ``otherwise:``.

    Returns:
        The value, ``(dims…, value)`` over the expression's own dims. Only
        this expression is compiled.

    Raises:
        LanguageError: A construct outside the streaming language, or a name
            the spec does not declare.
        SpecsolveError: A spec that declares variables, constraints or an
            objective.
        DataError: A source that is missing, unreadable, or the wrong shape,
            or a divisor with no value where the expression divides.
    """
    document = declared(spec)
    program = lowered(document)
    _refuse_a_decision(program)
    readers, evaluator = expression_readers(
        program, tidy_sources(program, sources), lambda written: lower(document, written)
    )
    return evaluated(readers, evaluator, expression)


class Model:
    """A spec with your data attached to it — what [`build`][] returns.

    ``check`` → ``Program`` (the math) → ``build`` → ``Model`` (the math with
    your data) → ``solve`` → ``Result`` (one answer).

    One build feeds any number of [`solve`][] and [`write`][] calls;
    [`update`][] puts new numbers on it without re-reading the YAML, and
    [`diagnostics`][] says what it did. Nothing has to be released;
    [`close`][] hands a large model back early.
    """

    def __init__(self, spec: Buildable, sources: Mapping[str, Source]) -> None:
        self._spec = declared(spec)
        self._program = lowered(self._spec)
        #: The document's digest; the data's is [`_model_digest`][].
        self._spec_digest = digest_of(self._spec)
        self._sources = dict(sources)
        #: What the last build read, as [`tidy_sources`][] gave it.
        self._tidied: dict[str, pl.LazyFrame] = {}
        self._engine = Engine()
        self._fill()

    def _lower(self, written: str | Mapping[str, object]) -> Expression:
        """One unnamed expression as a plan node, for a result reading a quantity the file never named.

        Held here because the engine may not see the model as written
        (docs/about/architecture.md, hard rule 2).
        """
        return lower(self._spec, written)

    def _fill(self) -> None:
        """Build from what is attached now; a failure closes the model rather than leaving it stale.

        What the build read is kept for an archive: a one-shot iterator is
        spent by then, and a path may have been rewritten.
        """
        try:
            self._tidied = tidy_sources(self._program, self._sources)
            self._engine.build(self._program, self._tidied)
        except BaseException:
            self._engine.close()
            raise

    def update(self, sources: Mapping[str, Source]) -> Model:
        """Put new numbers on the same model, in place.

        ::

            model.update({'cap_hat': capacity}).solve()

        ``model.update(x)`` answers what ``build(spec, sources | x)`` answers,
        whatever changed. Data that moves a mask renumbers labels, so the model
        is rebuilt and solved cold rather than pushed onto a loaded solver;
        [`loads`][specsolve.relational.result.Diagnostics.loads] says which ran.

        Results taken before the update keep reading their own frames, and
        keep their build's label frames alive until dropped or until
        [`close`][specsolve.relational.result.Result.close] is called. A sweep, a
        rolling horizon or a myopic pathway is
        [`solve_over`][specsolve.strategy.solve_over], which runs this loop.

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
        refuse_unknown_sources(self._program, sources)
        self._sources.update(sources)
        self._fill()
        return self

    def solve(
        self,
        solver_name: str = 'highs',
        *,
        solver_options: Mapping[str, object] | None = None,
        record_options: Sequence[str] | None = None,
        archive: str | Path | None = None,
        outputs: Iterable[Output] = (),
        start: Result | Start | None = None,
    ) -> Result:
        """Hand the built model to a solver and solve it.

        A solver that can stay loaded is kept between calls, so an updated
        model pushes only its numbers; [`diagnostics`][] counts the solves
        that loaded it again. A solve begins from nothing unless *start* says
        otherwise.

        Args:
            solver_name: ``highs``, which ships with the package; ``gurobi``,
                which needs the ``[gurobi]`` extra; or ``xpress``, which needs
                the ``[xpress]`` extra.
            solver_options: Forwarded to the solver verbatim, in its own
                vocabulary, so a time limit is ``time_limit``, ``TimeLimit`` or
                ``timelimit``. Gurobi's are applied when its environment is
                created, so ``ComputeServer``, ``TokenServer`` and
                ``WLSAccessID`` reach it too. The result's
                [`provenance`][specsolve.relational.result.Result.provenance]
                records the value of an option that changes the answer, such
                as a time limit or a gap, and the name alone of any other.
            record_options: More option names whose value the result
                records, in any letter case. Name no credential here: an
                archive goes to shared storage.
            archive: Where to write the spec, the data attached to it **now**,
                and this answer, so that
                [`load_archive`][specsolve.archive.load_archive] gives all
                three back and the model solves again from the file alone. A
                ``.zip`` suffix packs it into one file; anything else is a
                directory. What the build and its solves spent goes in beside
                the answer, as
                [`Metrics`][specsolve.relational.answer_layout.Metrics]. Each
                source goes in as the table [`tidy`][] returns, with
                ``specsolve_run`` added, and members are stored uncompressed.
            outputs: Each [`Output`][specsolve.types.Output] the answer
                carries beside the primal, the duals and the declared
                expressions: ``activity`` for each constraint's left-hand side,
                ``reduced_cost`` for each variable's reduced cost, ``slack``
                for each constraint's distance to binding, and ``basis`` for
                the basis status each variable and constraint ended on, read
                with ``variable_basis`` and ``constraint_basis``. The result, its save and its archive
                carry these and nothing else, and the reader of one not asked
                for refuses.
            start: What to start the solve from: an earlier answer, live,
                loaded with [`load_result`][] or from an archive, or a
                [`Start`][specsolve.types.Start] of tables keyed by reader. It
                is matched by coordinate, so one from another build of the
                spec, with rows or columns gained or lost, starts it too, and
                it changes how the solver gets to the optimum, never which
                one. An LP starts from a basis where one is given, which an
                answer carries when solved with ``outputs={'basis'}``: a
                coordinate it leaves out starts at a bound if it is a
                variable's, and not binding if it is a constraint's. Otherwise
                an LP, and a mixed-integer model always, starts from values,
                which the solver completes and repairs. Where a solver takes
                values for an LP and no gain from them is known, the solve
                warns. This model's last answer is not matched while its
                solver stays loaded: the solver carries on, the cheap way to
                step a model through updates.

        Returns:
            The solution, holding this model.

        Raises:
            SpecsolveError: A solver name nothing serves, one this environment
                cannot run, a bare string as *record_options* or *outputs*, a
                name in *outputs* that is not an output, or a *start* this
                model or solver cannot start from: a key that names no reader, a table naming no declaration or
                lacking its dims, a basis status outside the five words, a
                basis alone for a mixed-integer model, a start that lands on
                no coordinate, or values for an LP that the solver cannot
                take.
            LayoutError: An *archive* directory that already holds something,
                refused before the solve.
        """
        out = None if archive is None else Path(archive)
        if out is not None:
            check_the_target(out)
        if isinstance(record_options, str):
            raise SpecsolveError(
                f'record_options={record_options!r} is one string, which would name each of its letters. '
                f'Pass a list: record_options=[{record_options!r}].'
            )
        asked = checked_outputs(outputs)
        answered = replace(
            self._engine.solve(
                solver_name,
                solver_options=solver_options,
                lower=self._lower,
                outputs=asked,
                start=start if start is None or isinstance(start, Result) else self._read_start(start),
            ),
            _spec_digest=self._spec_digest,
            _solved_at=datetime.now(UTC),
            _provenance=_provenance(solver_name, solver_options, record_options or ()),
        )
        if out is not None:
            self._archive(out, answered)
        return answered

    def _read_start(self, start: Start) -> dict[str, dict[str, pl.LazyFrame]]:
        """*start*'s tables read against this build, as [`read_start`][specsolve.sources.read_start] reads them.

        Raises:
            SpecsolveError: A word, which only [`solve_over`][specsolve.strategy.solve_over]
                takes, or as [`read_start`][specsolve.sources.read_start] raises.
        """
        if isinstance(start, str):
            raise SpecsolveError(
                f'start={start!r} is a word only solve_over takes: a solve has no slice before it. Pass an earlier '
                'answer or a Start.'
            )
        return read_start(start, self._program, self._tidied)

    def _archive(self, out: Path, answered: Result) -> None:
        """Write this model, what is attached to it now, and *answered* to *out*.

        The metrics row is written here, not by [`Result.save`][]: it spans the
        model's life, not one solve.
        """
        with beside(out) as scratch:
            answer = answered.save(scratch)
            taken = self._engine.diagnostics().metrics()
            write_whole(pl.DataFrame([taken._asdict()], schema_overrides=METRICS_SCHEMA), answer / METRICS_FILE)
            write_archive(out, self._spec, numbered(self._program, self._tidied), axis=None, answer=answer)

    def write(self, path: str | Path) -> None:
        """Stream the built model to *path*, in the format its suffix names.

        Raises:
            ValueError: A suffix nothing writes.
            SpecsolveError: A construct the format has no section for, as
                [`check`][specsolve.types.Model.check] refuses it.
        """
        self._engine.write(path)

    def check(self, sink: str) -> None:
        """Refuse the built model where *sink* cannot take it; no solve, no file.

        ::

            sps.build('dispatch.yaml', sources).check('highs')

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
        self._engine.check(sink)

    def row(self, name: str, /, **coordinate: Label) -> ConstraintRow:
        """One built constraint row at one coordinate — its terms, sense and right-hand side.

        The verb for *this row is wrong and I do not know why*. It reads the
        **built** model and needs no solve: a term whose variable was absent
        is missing, and a row a ``where`` masked out is not there at all. A
        column has no reader: a variable's bounds are in the spec, and its
        coefficients are this read transposed.

        Args:
            name: A declared constraint. Positional, so a dimension may be
                called ``name``.
            coordinate: One label per dim of that constraint, all of them.

        Returns:
            The terms as ``(variable, coordinate, coefficient)``, beside the
            comparison and the right-hand side.

        Raises:
            KeyError: No constraint is called *name*.
            SpecsolveError: The coordinate names the wrong dims, holds a label
                its dimension cannot hold, matches no row the build produced,
                or the model has been closed.

        Example:
            >>> print(model.row('balance', snapshot=1))  # doctest: +SKIP
            balance[snapshot=1]: +1 p[snapshot=1, tech=wind] +50 p[snapshot=1, tech=gas] >= 60
        """
        return self._engine.row(name, coordinate)

    def _evaluator(
        self,
        primals: Mapping[str, pl.DataFrame],
        duals: Mapping[str, pl.DataFrame] | None,
        no_duals: str | None,
    ) -> Callable[[str | Mapping[str, object]], pl.DataFrame]:
        """An ad-hoc expression reader over a *saved* solution, put back against this build.

        What an archive and a sweep hand [`evaluate`][specsolve.relational.result.Result.evaluate]
        for a quantity the file never named: the saved frames are laid back in
        this build's label order, and the reader is the one a live solve gives.
        A build, never a solve.

        Args:
            primals: The saved ``(dims…, value)`` frame per variable.
            duals: The same per constraint, or ``None`` where the solve left none.
            no_duals: Why there are no duals, raised at a dual read.
        """
        evaluate = self._engine.reconstruct(primals, duals, no_duals, self._lower)
        assert evaluate is not None, 'a model built from a spec as written lowers an ad-hoc expression'
        return evaluate

    def _model_digest(self) -> str:
        """Which model this build *is* — the document and the data attached to it now.

        Over the built tables, so the same program over a differently ordered
        dimension digests differently.
        """
        return self._engine.contents()

    def diagnostics(self) -> Diagnostics:
        """What this build and its solves did that the answer does not show.

        Answerable after [`close`][], and after a build that raised. A raise
        leaves the sizes at zero, since they are taken once a model is whole,
        and everything measured before it stands.
        """
        return self._engine.diagnostics()

    def close(self) -> None:
        """Release the built model, and any solver still holding it."""
        self._engine.close()

    def __enter__(self) -> Model:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


def build(spec: Buildable, sources: Mapping[str, Source]) -> Model:
    """Attach *sources* to *spec* and build it — the model with your data on it.

    Args:
        spec: As [`check`][] takes it.
        sources: Parameter names to parquet paths or in-memory tables, and
            dimension names to their labels — an index table, a parquet path,
            or a bare sequence — wherever the YAML declares none. The shapes a
            value may take, and what attaching refuses, are
            [the data contract](https://specsolve.readthedocs.io/en/latest/reference/data/).

    Returns:
        The built model.

    Raises:
        LanguageError: A construct outside the streaming language.
        DataError: A source that is missing, unreadable, or the wrong shape.
    """
    return Model(spec, sources)


def solve(
    spec: Buildable,
    sources: Mapping[str, Source],
    solver_name: str = 'highs',
    *,
    solver_options: Mapping[str, object] | None = None,
    record_options: Sequence[str] | None = None,
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
        record_options: As [`Model.solve`][] takes them.
        archive: As [`Model.solve`][] takes it — a ``.zip``, or a directory.
        outputs: As [`Model.solve`][] takes them.
        start: As [`Model.solve`][] takes it.

    Returns:
        The solution. It owns its frames; the model and the solver are
        released before this returns.

    Raises:
        SpecsolveError: A solver name nothing serves, or *outputs* that
            [`Model.solve`][] refuses — both checked before the build — or as
            [`Model.solve`][] raises.
    """
    solver(solver_name)
    checked_outputs(outputs)
    model = build(spec, sources)
    try:
        return model.solve(
            solver_name,
            solver_options=solver_options,
            record_options=record_options,
            archive=archive,
            outputs=outputs,
            start=start,
        )
    finally:
        model.close()


def write(
    spec: Buildable,
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
    writer(out.suffix.lower())
    with build(spec, sources) as model:
        model.write(out)
    return out


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
        directory: Where [`save`][specsolve.relational.result.Result.save] wrote
            it. One inside an archive is
            [`load_archive`][specsolve.archive.load_archive]'s to find.

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
    record_file = out / RECORD_FILE
    if not record_file.is_file():
        raise LayoutError(
            f'{str(out)!r} holds no {RECORD_FILE!r}, so it is not an answer save() wrote. Every one '
            f'carries that record whether or not the solve produced values.'
        )
    check_format(out)
    record = Record(**pl.read_parquet(record_file).row(0, named=True))
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
        saved_frames(out / 'primal', whole=whole),
        saved_frames(out / 'dual', whole=whole),
        {kind: saved_frames(out / kind, whole=whole) for kind in kinds_of(read_outputs(out))},
        expressions,
        _no_duals=no_duals,
        _spec_digest=record.spec_digest,
        _solved_at=record.solved_at,
        _model_digest=record.model_digest,
        _run=record.specsolve_run,
        _provenance=record.provenance,
    )


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
    solver_name: str, solver_options: Mapping[str, object] | None, record_options: Sequence[str] = ()
) -> Provenance:
    """What a solve on *solver_name* with *solver_options* records about itself.

    The options are written as one JSON object, because a column of structs is
    one that several BI tools cannot read. Only an option on the solver's
    ``recorded_options`` keeps its value: an archive goes to storage other
    people read, and a list of what to hide would leak whatever it missed.
    """
    served = solver(solver_name)
    recorded = served.recorded_options | {name.casefold() for name in record_options}
    options = {
        name: _json_value(value) if name.casefold() in recorded else '<not recorded>'
        for name, value in (solver_options or {}).items()
    }
    return Provenance(
        solver_name,
        installed(served.requires[0]),
        json.dumps(options, sort_keys=True, default=str, allow_nan=False),
        installed('specsolve'),
        installed('mathspec'),
    )
