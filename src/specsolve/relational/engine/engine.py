"""Polars engine: build the model frames, hand them to a sink, read the answer back.

The engine owns the lifecycle — a build, its solver, the counters and clocks
[`Engine.diagnostics`][] reports — and none of the three questions it
asks on the way: what the data is ([`attaching`][specsolve.relational.engine.attaching]),
what each declaration contributes ([`assembly`][specsolve.relational.engine.assembly]),
how a row or a solve reads back ([`readback`][specsolve.relational.engine.readback]).
The lane is described in docs/about/architecture.md.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import polars as pl

from specsolve.errors import SpecsolveError
from specsolve.relational import sinks
from specsolve.relational.collect import sized
from specsolve.relational.engine import readback
from specsolve.relational.engine.assembly import (
    Assembly,
    BuiltModel,
    Measured,
    declares_quadratic,
    short_parameters,
)
from specsolve.relational.engine.attaching import attach
from specsolve.relational.engine.compiler import Compiler, Solution
from specsolve.relational.engine.scope import Scope
from specsolve.relational.result import KEEPS, ConstraintRow, Diagnostics, Keep, Result, unknown_keep_message

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from mathspec import program
    from polars._typing import PolarsDataType

    from specsolve.relational.sinks.solvers.base import SolveAnswer
    from specsolve.relational.status import SolveStatus


def _no_built_model(doing: str) -> str:
    """The message for a call made with no built model."""
    return (
        f'there is no built model {doing}: it was closed, or an update raised and released '
        f'it rather than leaving half of one behind. Build it again — update() with data it can '
        f'attach, or build() from the start.'
    )


class Engine:
    """Build a ``Program`` into polars frames, then sink it."""

    def __init__(self) -> None:
        #: The build, or ``None`` where there is not one — closed, released by
        #: an update that raised, or never run.
        self._built: BuiltModel | None = None
        #: What the last build measured about itself. Outlives ``_built``.
        self._measured = Measured()
        #: The solver holding this model, kept between solves and across rebuilds.
        self._solver: sinks.Solver | None = None
        #: How many solves this model has been through, and how many loaded the solver from scratch.
        self._solves = 0
        self._loads = 0
        #: Wall seconds each phase has spent, cumulative across rebuilds.
        self._seconds: dict[str, float] = {}

    @property
    def _model(self) -> BuiltModel:
        """The built model, or why there is not one."""
        if self._built is None:
            raise SpecsolveError(_no_built_model('to hand over'))
        return self._built

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(self, program: program.Program, sources: Mapping[str, pl.LazyFrame]) -> None:
        """Attach *sources*, then build every declaration into the model frames.

        A second call rebuilds over the same object. The held solver reads its
        [`structure`][specsolve.relational.sinks.solvers.base.Solver.structure]
        before the previous build is released, since that read lets go of these
        frames. A build that raises leaves no model rather than half of one.
        """
        if self._solver is not None:
            self._solver.structure()
        self._built = None
        self._measured = Measured()
        with _clocked(self._seconds, 'attach'):
            attached = attach(program, sources)
        self._measured.sparse = short_parameters(program, attached)
        assembly = Assembly(program, attached, self._measured)
        with _clocked(self._seconds, 'build'), sized(_largest(program, attached.cardinality)):
            self._built = assembly.run()

    # ------------------------------------------------------------------
    # sinks — see relational/sinks/; the engine only supplies the frames
    # ------------------------------------------------------------------

    def row(self, name: str, coordinate: Mapping[str, object]) -> ConstraintRow:
        """One built constraint row, spelled back out. See [`row`][specsolve.api.Model.row]."""
        if self._built is None:
            raise SpecsolveError(_no_built_model(f"to read '{name}' out of"))
        return readback.row(self._built, name, coordinate)

    def write(self, path: str | Path) -> None:
        """Stream the built model to *path*, in the format its suffix names.

        Raises:
            ValueError: A suffix nothing writes.
            SpecsolveError: A construct this format cannot spell.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        chosen = sinks.writer(suffix)
        self.check(suffix)
        with _clocked(self._seconds, 'write'):
            chosen.write(self._model.handoff, path)

    def check(self, sink: str) -> None:
        """Refuse the built model where the sink called *sink* cannot take it.

        Read off the hand-off, so a square the data prices at zero or an
        integer variable with no column built asks for nothing. What
        [`solve`][] and [`write`][] refuse, this refuses, with the same
        message.

        Raises:
            SpecsolveError: A construct the sink cannot take, naming it and the
                sinks that do; or a name belonging to no sink.
        """
        if (refused := sinks.refusal(self._model.handoff, sink)) is not None:
            raise SpecsolveError(refused)

    def _hand_off(
        self, solver_name: str, solver_options: Mapping[str, object] | None, keep: Keep
    ) -> tuple[sinks.Solver, Keep]:
        """[`solve`][] up to the run, which is where a benchmark of an update stops the clock.

        The held solver keeps the model where
        [`loaded`][specsolve.relational.sinks.solvers.loaded] allows and is
        loaded again where not; what comes back beside it is what it kept,
        ``nothing`` after a load. Counts toward neither ``solves`` nor
        ``loads``: [`solve`][] counts, so timing this alone leaves them true.
        """
        if keep not in KEEPS:
            raise SpecsolveError(unknown_keep_message(keep))
        self.check(solver_name)
        with _clocked(self._seconds, 'handoff'):
            if keep == 'nothing' and self._solver is not None:
                self._solver.close()
                self._solver = None
            held = self._solver
            self._solver = sinks.loaded(held, solver_name, self._model.handoff, solver_options)
            kept: Keep = keep if self._solver is held else 'nothing'
            if kept == 'solver':
                self._solver.forget()
        return self._solver, kept

    def solve(
        self,
        solver_name: str = 'highs',
        *,
        solver_options: Mapping[str, object] | None = None,
        keep: Keep = 'solver',
        lower: Callable[[str | Mapping[str, object]], program.Expression] | None = None,
    ) -> Result:
        """Hand the built model to a solver and solve it.

        The solver stays loaded where
        [`loaded`][specsolve.relational.sinks.solvers.loaded] allows. A construct
        the solver cannot ingest is refused before the load, read off the built
        model rather than the file.

        Args:
            solver_name: One of [`SOLVERS`][specsolve.relational.sinks.SOLVERS].
            solver_options: Forwarded to the solver verbatim, in its own
                vocabulary (``{'time_limit': 60, 'mip_rel_gap': 0.01}``).
            keep: How much of the session this solve may keep — one of
                [`KEEPS`][specsolve.relational.result.KEEPS]. A preference:
                [`kept`][specsolve.relational.result.Result.kept] reports what
                happened.
            lower: How an expression the caller writes becomes a plan node,
                for [`evaluate`][specsolve.relational.result.Result.evaluate], or
                ``None`` for a build from an already-lowered ``Program``.

        Returns:
            The solution, holding this engine and the build it answered.

        Raises:
            SpecsolveError: A *keep* outside
                [`KEEPS`][specsolve.relational.result.KEEPS].
        """
        solver, kept = self._hand_off(solver_name, solver_options, keep)
        handoff = self._model.handoff
        self._solves += 1
        if kept == 'nothing':
            self._loads += 1
        with _clocked(self._seconds, 'solve'):
            answer = solver.run(handoff)
        return self._answered(answer, solver_name, kept, lower)

    def _answered(
        self,
        answer: SolveAnswer,
        solver_name: str,
        kept: Keep,
        lower: Callable[[str | Mapping[str, object]], program.Expression] | None,
    ) -> Result:
        """[`solve`][] after the run: *answer*'s vectors laid out against this build, as a [`Result`][].

        Takes the answer rather than the solver, so that a benchmark can time
        reading one back without a solve; every vector must span this build.
        """
        handoff = self._model.handoff
        assert answer.primal is not None or not answer.status.is_readable, (
            'a readable status must come with a primal vector'
        )
        assert (answer.activity is None) == (answer.primal is None), (
            'activity travels with the primal: every sink reads it whenever a solution exists, mixed-integer included'
        )
        primals, duals, activities, rays = self._read_back(answer.primal, answer.dual, answer.activity, answer.dual_ray)
        no_duals = (
            None
            if answer.dual is not None
            else _no_duals_message(
                self._discrete(),
                answer.status.termination_condition,
                quadratic_rows=self._quadratic_constraints(),
            )
        )
        expressions, evaluate = self._readers(answer.primal, answer.dual, no_duals, lower)
        return Result(
            _status=answer.status,
            _objective=answer.objective,
            _primals=primals,
            _duals=duals,
            _activities=activities,
            _kept=kept,
            _expressions=expressions,
            _evaluate=evaluate,
            _no_duals=no_duals,
            _dual_rays=rays,
            _no_dual_ray=None if answer.dual_ray is not None else _no_dual_ray_message(answer.status, solver_name),
            _model_digest=lambda: handoff.contents,
            _coordinates=_largest(self._model.program, self._model.attached.cardinality),
        )

    def contents(self) -> str:
        """This build's digest — what a saved answer is checked against.

        Raises:
            SpecsolveError: Asked of an engine holding no built model.
        """
        if self._built is None:
            raise SpecsolveError(_no_built_model('to digest'))
        return self._model.handoff.contents

    def diagnostics(self) -> Diagnostics:
        """What this build and its solves did that the answer does not show; answerable after [`close`][]."""
        measured = self._measured
        return Diagnostics(
            columns=measured.columns,
            rows=measured.rows,
            nonzeros=measured.nonzeros,
            omissions=_per_name('constraint', measured.omitted, rows_not_built=pl.UInt32),
            coefficient_range=_per_name('constraint', measured.coefficients, smallest=pl.Float64, largest=pl.Float64),
            bound_range=_per_name('variable', measured.bounds, smallest=pl.Float64, largest=pl.Float64),
            rhs_range=_per_name('constraint', measured.rhs, smallest=pl.Float64, largest=pl.Float64),
            sparse_parameters=_per_name(
                'parameter',
                {name: (reach, rows, reach - rows) for name, (reach, rows) in measured.sparse.items()},
                coordinates=pl.UInt64,
                rows=pl.UInt64,
                missing=pl.UInt64,
            ),
            objective_range=measured.objective_range,
            solves=self._solves,
            loads=self._loads,
            seconds=dict(self._seconds),
        )

    def _read_back(
        self,
        primal: pl.Series | None,
        dual: pl.Series | None,
        activity: pl.Series | None,
        dual_ray: pl.Series | None,
    ) -> tuple[dict[str, pl.LazyFrame], ...]:
        """One solve's answer as one frame per declaration — a [`Result`][]'s own.

        The frames reference this build's label frames, which [`build`][]
        replaces rather than mutates. A ``None`` vector yields no frames rather
        than empty ones.
        """
        model = self._model
        program = model.program

        def rows(values: pl.Series | None) -> dict[str, pl.LazyFrame]:
            if values is None:
                return {}
            return {
                name: readback.laid_out(model.attached, model.constraints[name], c.dims, values)
                for name, c in program.constraints.items()
            }

        return (
            {
                name: readback.laid_out(model.attached, model.variables[name], v.dims, primal)
                for name, v in program.variables.items()
            }
            if primal is not None
            else {},
            rows(dual),
            rows(activity),
            rows(dual_ray),
        )

    def _readers(
        self,
        primal: pl.Series | None,
        dual: pl.Series | None,
        no_duals: str | None,
        lower: Callable[[str | Mapping[str, object]], program.Expression] | None,
    ) -> tuple[
        dict[str, Callable[[], pl.DataFrame]],
        Callable[[str | Mapping[str, object]], pl.DataFrame] | None,
    ]:
        """What [`evaluate`][specsolve.relational.result.Result.evaluate] reads through: a reader per declared name, and the ad-hoc evaluator.

        Both close over a snapshot the result owns, including a copy of the
        variable registry, so they keep answering after an update or ``close()``.
        """
        if primal is None:
            return {}, None
        model = self._model
        solution = Solution(primal, dual, dict(model.constraints), no_duals)
        compiler = Compiler(Scope(model.program, model.attached, dict(model.variables)), solution)
        return readback.readers(compiler, model.program.expressions, lower)

    def reconstruct(
        self,
        primals: Mapping[str, pl.DataFrame],
        duals: Mapping[str, pl.DataFrame] | None,
        no_duals: str | None,
        lower: Callable[[str | Mapping[str, object]], program.Expression] | None,
    ) -> Callable[[str | Mapping[str, object]], pl.DataFrame] | None:
        """The ad-hoc evaluator for a saved solution, over this rebuilt model.

        This build supplies the labels that put the saved values back in vector
        order. It comes back where *lower* is given.

        Args:
            primals: The saved ``(dims…, value)`` frame per variable.
            duals: The same per constraint, or ``None`` where the solve left no duals.
            no_duals: Why there are no duals, or ``None`` when *duals* holds them.
            lower: How an expression the caller writes becomes a plan node.
        """
        model = self._model
        primal = readback.reordered(model.attached, model.variables, model.program.variables, primals)
        dual = (
            readback.reordered(model.attached, model.constraints, model.program.constraints, duals)
            if duals is not None
            else None
        )
        _, evaluate = self._readers(primal, dual, no_duals, lower)
        return evaluate

    def _discrete(self) -> list[str]:
        """The variables this model declared as anything but continuous."""
        return sorted(n for n, v in self._model.program.variables.items() if v.domain != 'continuous')

    def _quadratic_constraints(self) -> list[str]:
        """The constraints this model declared as quadratic — a fact about the model, not the solve."""
        return sorted(n for n, c in self._model.program.constraints.items() if declares_quadratic(c))

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Drop the built model and close a loaded solver. A [`Result`][] keeps its own frames."""
        if self._solver is not None:
            self._solver.close()
            self._solver = None
        self._built = None

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


def _per_name(kind: str, measured: Mapping[str, object], **columns: PolarsDataType) -> pl.DataFrame:
    """One [`Diagnostics`][specsolve.relational.result.Diagnostics] frame: a row per name in *measured*, in build order.

    *kind* names the first column, and the remaining *columns* carry each
    value in order — a scalar for one column, a tuple for several.
    """
    names = list(measured)
    values = [v if isinstance(v, tuple) else (v,) for v in measured.values()]
    schema = {kind: pl.String, **columns}
    return pl.DataFrame(
        {kind: names, **{column: [v[i] for v in values] for i, column in enumerate(columns)}}, schema=schema
    )


def expression_readers(
    program: program.Program,
    sources: Mapping[str, pl.LazyFrame],
    lower: Callable[[str | Mapping[str, object]], program.Expression] | None,
) -> tuple[dict[str, Callable[[], pl.DataFrame]], Callable[[str | Mapping[str, object]], pl.DataFrame] | None]:
    """Attach *sources* and defer the reads [`specsolve.evaluate`][] values one expression through.

    Args:
        program: A lowered program with no variables — a calculation, so every
            expression has a value with no solver.
        sources: Tidied sources, as
            [`tidy_sources`][specsolve.sources.tidy_sources] produces.
        lower: How an ad-hoc expression becomes a plan node in the model's
            namespace, or ``None`` where ad-hoc evaluation is not offered.

    Returns:
        One deferred reader per declared named expression, and the ad-hoc
        evaluator; either compiles on call.
    """
    compiler = Compiler(Scope(program, attach(program, sources), {}))
    return readback.readers(compiler, program.expressions, lower)


def _no_dual_ray_message(status: SolveStatus, solver_name: str) -> str:
    """Why this answer carries no certificate of infeasibility.

    It takes the sink's name rather than asking the sink, whose session the
    engine may already have let go.
    """
    if status.termination_condition != 'infeasible':
        return (
            f'a dual ray certifies that a model has no solution, and this solve terminated '
            f'{status.termination_condition!r} rather than infeasible. Read dual() for the prices of a '
            f'model that does have one.'
        )
    asks = {
        'gurobi': "re-solve with solver_options={'InfUnbdInfo': 1}, which gurobi needs set before the "
        'solve to compute the certificate at all',
        'xpress': "re-solve with solver_options={'presolve': 0}: xpress has a ray only where the simplex "
        'found the infeasibility, and none where presolve found it first',
    }
    ask = asks.get(solver_name)
    return f'the model is infeasible, and the {solver_name} sink returned no dual ray to certify it. ' + (
        f'{ask}.' if ask else 'A ray comes from the simplex, so a run that never reached it has none to give.'
    )


def _no_duals_message(
    discrete: Sequence[str],
    termination_condition: str,
    quadratic_rows: Sequence[str],
) -> str:
    """The message for a solve that left values but no duals."""
    if quadratic_rows and not discrete:
        names = ', '.join(f"'{n}'" for n in quadratic_rows)
        return (
            f"a quadratic constraint prices only under gurobi's QCPDual, which is off by default: "
            f'{names} {"is" if len(quadratic_rows) == 1 else "are"} quadratic. Asking for those '
            f'prices makes the solver take the convex path, so a nonconvex row that solves without '
            f'them fails with them — which is why this is yours to ask for rather than ours to '
            f"assume. Re-solve with solver_options={{'QCPDual': 1}} if the model is convex."
        )
    if discrete:
        names = ', '.join(f"'{n}'" for n in discrete)
        return (
            f'duals are undefined for a mixed-integer model: {names} '
            f'{"is" if len(discrete) == 1 else "are"} not continuous. '
            f'Drop the integrality to price the LP relaxation instead.'
        )
    return (
        f'the solver returned no dual solution, though the solve terminated '
        f'{termination_condition!r}. Duals come from a simplex basis, which a '
        f'run stopped short of one does not have.'
    )


@contextmanager
def _clocked(seconds: dict[str, float], phase: str) -> Iterator[None]:
    """Add the block's wall time onto ``seconds[phase]``, on failure too."""
    started = perf_counter()
    try:
        yield
    finally:
        seconds[phase] = seconds.get(phase, 0.0) + perf_counter() - started


def _largest(program: program.Program, cardinality: Mapping[str, int]) -> int:
    """The coordinate count of the largest variable or constraint, unmasked."""
    declarations = (*program.variables.values(), *program.constraints.values())
    return max((math.prod(cardinality[d] for d in declared.dims) for declared in declarations), default=1)
