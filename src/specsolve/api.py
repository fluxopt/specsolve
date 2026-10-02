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

import warnings
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from mathspec import advice

from specsolve import expressions
from specsolve.errors import (
    DataError,
    LayoutError,
    SpecsolveError,
    SpecsolveWarning,
    another_model_behind_this_answer_message,
)
from specsolve.lanes import Buildable, Label, Source, declared, lowered
from specsolve.layout import beside, check_the_target, write_archive
from specsolve.relational.engines.polars.engine import PolarsEngine, expression_readers
from specsolve.relational.parquet import (
    METRICS_FILE,
    METRICS_SCHEMA,
    RECORD_FILE,
    Record,
    check_format,
    digest_of,
    read_reasons,
    write_whole,
)
from specsolve.relational.result import Result, evaluated
from specsolve.relational.sinks import solver, writer
from specsolve.sources import attachable, tidy_sources, unknown_source_keys_message

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mathspec.program import Expression, Program

    from specsolve.relational.result import ConstraintRow, Diagnostics, Keep

__all__ = ['build', 'check', 'evaluate', 'load_result', 'scan_result', 'solve', 'write']


def check(spec: Buildable) -> Program:
    """Parse, validate and lower a spec; attach no data.

    The CI verb: with no data and no solver, a spec repository validates every
    commit. Every other verb reads the spec through the same door, so what this
    refuses they refuse too.

    Whether a sink takes the model is not asked here: that is a fact about
    the model a build produces, and [`Model.check`][specsolve.Model.check]
    answers it with no solve.

    Args:
        spec: A YAML path, a mapping, or a ``Spec`` — what ``mathspec.to_spec``
            takes, so a framework that emits declarations passes the mapping
            and writes no file. A ``Spec`` is not read again. A lowered
            ``Program`` is not taken. A ``piecewise:`` block is written out
            first: ``to_spec(spec).expand('piecewise')`` keeps every ``sos:``
            set for a sink that takes one, and ``to_spec(spec).expand()``
            writes the sets out too, as binaries every sink takes.

    Returns:
        The lowered program: what a build reads rows off, for reading the plan.
        No verb takes it back; keep the ``Spec`` for that. It is the
        language's own type — typeset it, or read its declarations, through
        `mathspec`.

    Raises:
        LanguageError: A construct outside the streaming language, or a
            ``piecewise:`` block still to be written out.
        SpecsolveError: Two declarations whose names differ only by case.
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


def _refuse_a_decision(program: Program) -> None:
    """Refuse a spec that declares a decision — [`evaluate`][] is arithmetic, not a solve.

    Raises:
        SpecsolveError: The program declares variables, constraints or an
            objective.
    """
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

    A spec that declares no variables is a calculation, not an optimisation:
    dimensions, parameters, relations and ``expressions:``. Each expression reads
    only the attached data, so it has a value with no solve and no chosen point.
    This attaches *sources* and values one expression, the way
    [`evaluate`][specsolve.relational.result.Result.evaluate] does at a solution. The
    language it is read through — what loads, what is refused, how a construct
    prints and lowers — is the one a spec that solves is read through; only the
    variables are absent.

    A spec that declares variables is a problem to solve, and belongs to [`solve`][]: an
    expression over a decision has no value until the decision is made.

    Args:
        spec: As [`check`][] takes it — a YAML path, a mapping, or a ``Spec``.
        sources: As [`build`][] takes them: parameter names to tables or
            parquet paths, and dimension names to their labels.
        expression: What one ``expressions:`` entry takes — a name the spec
            declares, an expression string, or the mapping carrying ``cases:``
            with ``dims:`` and ``otherwise:``.

    Returns:
        The value, ``(dims…, value)`` over the expression's own dims. Only
        this expression is compiled: a declared one nothing asks for costs
        nothing.

    Raises:
        LanguageError: A construct outside the streaming language, or a name
            the spec does not declare.
        SpecsolveError: A spec that declares variables, constraints or an
            objective — a problem to solve, not a calculation to evaluate.
        DataError: A source that is missing, unreadable, or the wrong shape,
            or a divisor with no value where the expression divides.
    """
    document = declared(spec)
    program = lowered(document)
    _refuse_a_decision(program)
    readers, evaluator = expression_readers(
        program, tidy_sources(program, sources), lambda written: expressions.lower(document, written)
    )
    return evaluated(readers, evaluator, expression)


class Model:
    """A spec with your data attached to it — what [`build`][] returns.

    Three nouns, each arrow adding one thing: a ``Program`` is the math,
    a ``Model`` is the math with your data, a ``Result`` is one answer:
    ``check`` → ``Program`` → ``build`` → ``Model`` → ``solve`` → ``Result``.

    One build feeds any number of sinks — [`solve`][] and [`write`][] on
    the same object — [`update`][] puts new numbers on it without re-reading
    the YAML or re-lowering the plan, and [`diagnostics`][] says what it did.
    Nothing has to be released; [`close`][] hands a large model back early.
    """

    def __init__(self, spec: Buildable, sources: Mapping[str, Source]) -> None:
        self._spec = declared(spec)
        self._program = lowered(self._spec)
        #: The document's digest; the data's is [`_model_digest`][].
        self._spec_digest = digest_of(self._spec.to_yaml())
        self._sources = dict(sources)
        self._engine = PolarsEngine()
        self._fill()

    def _lower(self, written: str | Mapping[str, object]) -> Expression:
        """One unnamed expression as a plan node, for a result reading a quantity the file never named.

        Held here because the engine may not see the model as written
        (docs/about/architecture.md, hard rule 2).
        """
        return expressions.lower(self._spec, written)

    def _fill(self) -> None:
        """Build from what is attached now; a failure closes the model rather than leaving it stale."""
        try:
            self._engine.build(self._program, tidy_sources(self._program, self._sources))
        except BaseException:
            self._engine.close()
            raise

    def update(self, sources: Mapping[str, Source]) -> Model:
        """Put new numbers on the same model, in place.

        ::

            model.update({'cap_hat': capacity}).solve()

        Any new data is accepted: ``model.update(x)`` answers what
        ``build(spec, sources | x)`` answers, whatever changed. Data that moves
        a mask renumbers labels, so the model is rebuilt and solved cold
        instead of pushed onto a loaded solver, and
        [`loads`][specsolve.relational.result.Diagnostics.loads] says which ran.

        Results taken before the update keep reading: each owns the frames it
        reads, and an update builds new ones rather than touching those. A
        retained result keeps its build's label frames alive until it is
        dropped or [`close`][specsolve.relational.result.Result.close] is called.

        A loop whose next numbers depend on the last answer is this; a sweep, a
        rolling horizon or a myopic pathway is [`solve_over`][specsolve.strategy.solve_over],
        which runs the loop.

        Args:
            sources: Only what changed; the rest keeps what [`build`][]
                attached. A dimension's labels as well as a parameter, which is
                how a coordinate set grows.

        Returns:
            This object.

        Raises:
            DataError: A name the spec does not declare, since an update that
                named nothing would solve the old numbers again. An update that
                raises releases the model, as a build that raises does.
        """
        _refuse_unknown(sources, attachable(self._program))
        self._sources.update(sources)
        self._fill()
        return self

    def solve(
        self,
        solver_name: str = 'highs',
        *,
        solver_options: Mapping[str, object] | None = None,
        keep: Keep = 'solver',
        archive: str | Path | None = None,
    ) -> Result:
        """Hand the built model to a solver and solve it.

        A solver that can stay loaded is kept between calls, so an updated
        model skips the hand-off and only its numbers are pushed. Whether the
        *work* that solver did is kept too is *keep*, off by default. How much
        this solve actually kept is its
        [`kept`][specsolve.relational.result.Result.kept].

        Args:
            solver_name: ``highs``, which ships with the package; ``gurobi``,
                which needs the ``[gurobi]`` extra; or ``xpress``, which needs
                the ``[xpress]`` extra. The caller chooses: nothing in the spec
                names a solver.
            solver_options: Forwarded to the solver verbatim, in its own
                vocabulary, so a time limit is ``time_limit``, ``TimeLimit`` or
                ``timelimit``. Gurobi's are applied when its environment is
                created, so ``ComputeServer``, ``TokenServer`` and
                ``WLSAccessID`` reach it too.
            keep: How much of the session this solve may keep: ``solver``,
                ``progress`` or ``nothing``. ``solver``, the
                default, reuses the solver holding the model and discards the
                work it did; ``progress`` keeps that work too, which is what
                an iterating driver moving one step at a time wants;
                ``nothing`` keeps neither, which is what timing a build or
                comparing against a cold baseline needs and what no solver
                option can promise. A preference: a model whose structure
                moved is loaded again whatever was asked.
            archive: Where to write the whole thing — the spec, the data
                attached to it **now**, and this answer — so that
                [`load_archive`][specsolve.archive.load_archive] gives all three back and
                the model solves again from the file alone. A ``.zip`` suffix
                packs it into one file and anything else is a directory. What
                the build and its solves have spent goes in beside the answer,
                as [`Metrics`][specsolve.relational.parquet.Metrics]. The
                sources go in through the door [`build`][] reads them through:
                a parquet path is copied as its own bytes, anything else is
                written as the table it stands for, and members are stored
                uncompressed.

        Returns:
            The solution, holding this model.

        Raises:
            SpecsolveError: A solver name nothing serves, one this environment
                cannot run, or a *keep* other than those three.
            LayoutError: An *archive* directory that already holds something,
                refused before the solve.
        """
        out = None if archive is None else Path(archive)
        if out is not None:
            check_the_target(out)
        answered = replace(
            self._engine.solve(
                solver_name,
                solver_options=solver_options,
                keep=keep,
                lower=self._lower,
            ),
            _spec_digest=self._spec_digest,
            _solved_at=datetime.now(UTC),
        )
        if out is not None:
            self._archive(out, answered)
        return answered

    def _archive(self, out: Path, answered: Result) -> None:
        """Write this model, what is attached to it now, and *answered* to *out*.

        The metrics row is written here, not by [`Result.save`][]: it spans the
        model's life, not one solve.
        """
        with beside(out) as scratch:
            answer = answered.save(scratch)
            taken = self._engine.diagnostics().metrics()
            write_whole(pl.DataFrame([taken._asdict()], schema_overrides=METRICS_SCHEMA), answer / METRICS_FILE)
            tables = tidy_sources(self._program, self._sources)
            write_archive(out, self._spec, self._sources, tables=tables, axis=None, answer=answer)

    def write(self, path: str | Path) -> None:
        """Stream the built model to *path*, in the format its suffix names.

        Raises:
            ValueError: A suffix nothing writes.
            SpecsolveError: A construct the format has no section for, as
                [`check`][specsolve.Model.check] refuses it.
        """
        self._engine.write(path)

    def check(self, sink: str) -> None:
        """Refuse the built model where *sink* cannot take it; no solve, no file.

        ::

            with sps.build('dispatch.yaml', sources) as model:
                model.check('highs')

        The answer is read off the model this build produced, not off the
        file: a square the data prices at zero, an integer variable no column
        is built for or a set with no members asks for nothing. [`solve`][]
        and [`write`][] refuse exactly what this refuses, with the same
        message, so a CI job that builds every example and checks it pays for
        no solve.

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

        The verb for *this row is wrong and I do not know why*. ``to_latex``
        and its siblings render the spec as math before any data, and
        [`dual`][specsolve.relational.result.Result.dual] gives a row's number
        without its terms; this gives the row the build actually produced, at
        the coordinate you name.

        Reads the **built** model and needs no solve, so it answers on a model
        that never reached a solver — and it is the built row, so a term whose
        variable was absent is missing from it and a row a ``where`` masked out
        is not there at all. It shows what the model says rather than what the
        file appears to say. A column has no reader: a variable's bounds are in
        the spec, and its coefficients are this read transposed.

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
            balance[snapshot=1]: +1 p[1, wind] +50 p[1, gas] >= 60
        """
        return self._engine.row(name, coordinate)

    def evaluator(
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
            duals: The same per constraint, or ``None`` where the solve left no
                duals.
            no_duals: Why there are no duals, raised at a dual read; ``None``
                when *duals* holds them.
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

        Answerable after [`close`][], and after a build that raised: every
        field is a count, a clock or a small frame the engine keeps, not a read
        of the model it releases. A raise leaves the sizes at zero — they are
        taken once a model is whole — and everything measured before it stands.
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


def _refuse_unknown(given: Mapping[str, object], declared: Mapping[str, object]) -> None:
    """Refuse an update naming anything *declared* does not hold."""
    if unknown := set(given) - set(declared):
        raise DataError(unknown_source_keys_message(unknown, declared))


def build(spec: Buildable, sources: Mapping[str, Source]) -> Model:
    """Attach *sources* to *spec* and build it — the model with your data on it.

    Args:
        spec: As [`check`][] takes it.
        sources: Parameter names to parquet paths or in-memory tables, and
            dimension names to their labels — an index table, a parquet path,
            or a bare sequence — wherever the YAML declares none. The whole
            of the build's input: the shapes a value may take, and what
            attaching refuses, are
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
    archive: str | Path | None = None,
) -> Result:
    """Build *spec* and solve it in one call.

    The one-shot spelling: a caller who will solve the same spec again with
    new numbers wants [`build`][] and [`Model.update`][].

    There is no ``keep`` here — this builds the model it solves, so the solve
    is the first of that model's life and
    [`kept`][specsolve.relational.result.Result.kept] is always ``nothing``.
    Choosing what to keep is [`Model.solve`][].

    Args:
        spec: As [`check`][] takes it.
        sources: As [`build`][] takes them.
        solver_name: As [`Model.solve`][] takes it.
        solver_options: As [`Model.solve`][] takes them.
        archive: Where to write the spec, its data and this answer, as
            [`Model.solve`][] takes it — a ``.zip``, or a directory.

    Returns:
        The solution. It owns its frames; the model and the solver are
        released before this returns. ``result.close()`` drops its own hold
        early.

    Raises:
        SpecsolveError: A solver name nothing serves — checked before the build.
    """
    solver(solver_name)
    model = build(spec, sources)
    try:
        return model.solve(solver_name, solver_options=solver_options, archive=archive)
    finally:
        model.close()


def write(
    spec: Buildable,
    sources: Mapping[str, Source],
    out: str | Path,
) -> Path:
    """Build *spec* and stream it to a file, in the format *out*'s suffix names.

    Args:
        spec: As [`check`][] takes it.
        sources: As [`build`][] takes them.
        out: Where to write; ``.lp`` and ``.mps`` are what ship. The two
            describe one model and name its columns and rows the same way.

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


def _whole(file: Path) -> pl.LazyFrame:
    """*file* read into memory now, as the `polars.LazyFrame` a saved frame is held as."""
    return pl.read_parquet(file).lazy()


#: How a saved frame is read: [`_whole`][] now, `polars.scan_parquet` at the
#: first collect.
type Reading = Callable[[Path], pl.LazyFrame]


def _saved_frames(under: Path, read: Reading) -> dict[str, pl.LazyFrame]:
    """Every ``<name>.parquet`` under *under*, keyed by name; empty where it does not exist."""
    if not under.is_dir():
        return {}
    return {file.stem: read(file) for file in sorted(under.glob('*.parquet'))}


def _absent(reason: str) -> Callable[[], pl.DataFrame]:
    """A named expression's reader that raises *reason* at the read."""

    def read() -> pl.DataFrame:
        raise SpecsolveError(reason)

    return read


def load_result(directory: str | Path) -> Result:
    """Read back an answer [`Result.save`][] wrote — a solve, off disk.

    Every reader answers what it answered in the session that solved: the
    values, the duals and activities, each named expression, and the reason
    behind anything the solve could not produce. None of it needs the build
    that made it or the solver that filled it.

    Two things do not come back: [`kept`][specsolve.relational.result.Result.kept]
    reads ``nothing``, this result holding no solver, and the solver's verbatim
    wording behind a refusal is not recorded — the termination condition is. A
    solve that reached no objective wrote null and reads back as ``nan``.

    Args:
        directory: Where [`save`][specsolve.relational.result.Result.save] wrote
            it. One that came out of an archive is
            [`load_archive`][specsolve.archive.load_archive]'s to find.

    Returns:
        The result, read whole: the frames are in memory when this returns, so
        it owes *directory* nothing. [`scan_result`][] is the same answer left
        on disk.

    Raises:
        LayoutError: A directory holding no ``record.parquet``, or one whose
            layout has moved since it was written.
    """
    return _answer_under(Path(directory), _whole)


def scan_result(directory: str | Path) -> Result:
    """The answer under *directory*, read as its readers are called rather than now.

    [`load_result`][]'s other half, and the same value: every reader answers
    what that one's does. What differs is when the bytes move — each frame is
    a `polars.scan_parquet` of the file it lies in, so an answer far
    larger than memory is readable a name at a time, and one whose names go
    unread costs nothing to open.

    The files stay where they are, so **they have to outlive the result**: a
    name read after the directory is gone raises where the scan is collected,
    and a file rewritten underneath it comes back changed.

    Args:
        directory: As [`load_result`][] takes it.

    Raises:
        LayoutError: As [`load_result`][] raises it.
    """
    return _answer_under(Path(directory), pl.scan_parquet)


def _answer_under(out: Path, read: Reading) -> Result:
    """The saved answer under *out*, its frames read *read*'s way."""
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
    if not status.is_readable:
        return Result(
            status,
            objective,
            {},
            {},
            {},
            'nothing',
            _spec_digest=record.spec_digest,
            _solved_at=record.solved_at,
            _model_digest=record.model_digest,
            _run=record.run,
        )

    no_duals, no_expressions = read_reasons(out)
    expressions: dict[str, Callable[[], pl.DataFrame]] = {
        name: (lambda frame=frame: frame.collect()) for name, frame in _saved_frames(out / 'expression', read).items()
    }
    expressions.update({name: _absent(why) for name, why in no_expressions.items()})
    return Result(
        status,
        objective,
        _saved_frames(out / 'primal', read),
        _saved_frames(out / 'dual', read),
        _saved_frames(out / 'activity', read),
        'nothing',
        expressions,
        _no_duals=no_duals,
        _spec_digest=record.spec_digest,
        _solved_at=record.solved_at,
        _model_digest=record.model_digest,
        _run=record.run,
    )


def _refuse_another_model(answer: Result, model: Model) -> None:
    """Refuse a saved answer against a model built from other data than the one it answered.

    The spec is compared where the pair is read; the data needs a build, so it
    is compared here. An answer carrying no digest is taken as given.

    Raises:
        SpecsolveError: Sources that build a model other than the answered one.
    """
    answered = answer.model_digest()
    if answered is not None and answered != (rebuilt := model._model_digest()):
        raise SpecsolveError(another_model_behind_this_answer_message(answered, rebuilt))


def attach_readers(answer: Result, spec: Buildable, sources: Mapping[str, Source]) -> Result:
    """*answer* with an undeclared expression readable through [`evaluate`][specsolve.relational.result.Result.evaluate].

    *spec* is rebuilt over *sources* at the first undeclared read, never
    solved, and cached. A rebuild from other data than the solve ran on is
    refused. *answer* comes back unchanged where the solve left no values.

    Args:
        answer: A saved solve, as [`load_result`][] or [`scan_result`][]
            read it back.
        spec: The model the answer solved, as [`build`][] takes it.
        sources: What it was solved with, as [`build`][] takes them.
    """
    if not answer._primals:
        return answer
    frames = answer._primals
    dual_frames = answer._duals
    no_duals = answer._no_duals
    built: list[Callable[[str | Mapping[str, object]], pl.DataFrame]] = []

    def evaluate(written: str | Mapping[str, object]) -> pl.DataFrame:
        if not built:
            primals = {name: frame.collect() for name, frame in frames.items()}
            duals = (
                {name: frame.collect() for name, frame in dual_frames.items()}
                if no_duals is None and dual_frames
                else None
            )
            model = build(spec, sources)
            _refuse_another_model(answer, model)
            built.append(model.evaluator(primals, duals, no_duals))
        return built[0](written)

    return replace(answer, _evaluate=evaluate)
