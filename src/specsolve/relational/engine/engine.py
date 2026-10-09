"""Polars engine: build the model frames, hand them to a sink, read the answer back.

The engine owns the lifecycle — a build, its solver, the counters and clocks
[`Engine.diagnostics`][] reports — and none of the three questions it
asks on the way: what the data is ([`attaching`][specsolve.relational.engine.attaching]),
what each declaration contributes ([`assembly`][specsolve.relational.engine.assembly]),
how a row or a solve reads back ([`readback`][specsolve.relational.engine.readback]).
The lane is described in docs/about/architecture.md.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import polars as pl

from specsolve.errors import SpecsolveError, SpecsolveWarning
from specsolve.relational import sinks
from specsolve.relational.answer_layout import BASIS, BASIS_STATUSES, kinds_of
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
from specsolve.relational.names import VALUE
from specsolve.relational.result import ConstraintRow, Diagnostics, Result
from specsolve.relational.sinks.solvers.base import Basis

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    import numpy as np
    from mathspec import program
    from polars._typing import PolarsDataType

    from specsolve.relational.answer_layout import Output
    from specsolve.relational.result import InfeasibleSubsystem
    from specsolve.relational.sinks.solvers.base import SolveAnswer
    from specsolve.relational.status import SolveStatus


def _statuses(codes: np.ndarray) -> pl.Series:
    """Basis status codes as the series a frame is laid out from."""
    return pl.Series(VALUE, BASIS_STATUSES, dtype=BASIS).gather(codes)


def _nothing_to_start_message(discrete: bool) -> str:
    """Why a start gives a model nothing it starts from: a basis alone for a mixed-integer model, or no table at all."""
    if discrete:
        return (
            'start= gives this mixed-integer model a basis alone, and a basis starts only an LP. Give values '
            "under 'primal' as well."
        )
    return "start= gives nothing to start from. Give a table under 'primal', 'variable_basis' or 'constraint_basis'."


def _checked_lp_values(sink: type[sinks.Solver], solver_name: str, values: np.ndarray) -> None:
    """Refuse a start of values for an LP that *sink* cannot take, and warn of one it takes where no gain is known.

    Raises:
        SpecsolveError: *sink* cannot take values that leave a column out.
    """
    import numpy as np

    partial = bool(np.isnan(values).any())
    match sink.lp_values['partial' if partial else 'complete']:
        case 'refused':
            raise SpecsolveError(
                f'{solver_name} cannot start an LP from values that leave a column out, and this start gives '
                f'no value at {int(np.isnan(values).sum())} of {len(values)}. Give a value at every '
                "coordinate of every variable, or start from an earlier answer solved with outputs={'basis'}."
            )
        case 'no_gain':
            warnings.warn(
                f'{solver_name} takes {"a partial" if partial else "a"} start of values for an LP, and no gain '
                f'from one is known: it measured no fewer iterations than a solve started cold. An earlier '
                f"answer solved with outputs={{'basis'}} starts an LP from its basis. If you see a start of "
                f'values speed up a solve, please report it at https://github.com/fluxopt/specsolve/issues.',
                SpecsolveWarning,
                stacklevel=5,
            )


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
        #: What the last solve concluded, or ``None`` once a load, a build or
        #: [`close`][] has changed what the solver holds.
        self._solved: SolveStatus | None = None
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
        self._solved = None
        self._measured = Measured()
        with _clocked(self._seconds, 'attach'):
            attached = attach(program, sources)
        self._measured.sparse = short_parameters(program, attached)
        assembly = Assembly(program, attached, self._measured)
        with _clocked(self._seconds, 'build'):
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
        self, solver_name: str, solver_options: Mapping[str, object] | None, *, carry_on: bool = False
    ) -> tuple[sinks.Solver, bool]:
        """[`solve`][] up to the run, which is where a benchmark of an update stops the clock.

        Returns the solver and whether it was loaded again, which
        [`loaded`][specsolve.relational.sinks.solvers.loaded] decides. A solver
        kept forgets its last run unless *carry_on*. Counts toward neither
        ``solves`` nor ``loads``, so timing this alone leaves them true.
        """
        self.check(solver_name)
        self._solved = None
        with _clocked(self._seconds, 'handoff'):
            held = self._solver
            self._solver = sinks.loaded(held, solver_name, self._model.handoff, solver_options)
            reloaded = self._solver is not held
            if not reloaded and not carry_on:
                self._solver.forget()
        return self._solver, reloaded

    def solve(
        self,
        solver_name: str = 'highs',
        *,
        solver_options: Mapping[str, object] | None = None,
        lower: Callable[[str | Mapping[str, object]], program.Expression] | None = None,
        outputs: frozenset[Output] = frozenset(),
        start: Result | Mapping[str, Mapping[str, pl.LazyFrame]] | Literal['previous'] | None = 'previous',
        last: bool = False,
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
            lower: How an expression the caller writes becomes a plan node,
                for [`evaluate`][specsolve.relational.result.Result.evaluate], or
                ``None`` for a build from an already-lowered ``Program``.
            outputs: Which of
                [`OUTPUTS`][specsolve.relational.answer_layout.OUTPUTS] the
                result carries, already checked.
            start: ``'previous'``, to carry on from the last solve while its
                solver stays loaded and begin from nothing otherwise; ``None``,
                to begin from nothing; or an earlier answer or a
                [`Start`][specsolve.types.Start] already read
                ([`read_start`][specsolve.sources.read_start]), matched by
                coordinate: an LP from a basis where one is given
                ([`matched_basis`][specsolve.relational.engine.readback.matched_basis]),
                and otherwise, and a mixed-integer model always, from values
                ([`matched_values`][specsolve.relational.engine.readback.matched_values]).
            last: Whether the caller closes this engine after this solve. The
                build then lets go of what the answer does not read before the
                solver runs ([`_let_go`][]), so the solver's run does not share
                the process with a second copy of the model.

        Returns:
            The solution, holding this engine and the build it answered.

        Raises:
            SpecsolveError: A *start* this model or this solver cannot start
                from, refused before the solver loads.
        """
        carry_on = start == 'previous'
        matched = None if start is None or isinstance(start, str) else self._matched_start(start, solver_name)
        solver, reloaded = self._hand_off(solver_name, solver_options, carry_on=carry_on)
        if isinstance(matched, Basis):
            solver.warm(matched)
        elif matched is not None:
            solver.start(matched)
        if last:
            self._let_go(solver, outputs)
        handoff = self._model.handoff
        self._solves += 1
        if reloaded:
            self._loads += 1
        with _clocked(self._seconds, 'solve'):
            answer = solver.run(handoff, basis='basis' in outputs)
        self._solved = answer.status
        return self._answered(answer, solver_name, lower, outputs)

    def _let_go(self, solver: sinks.Solver, outputs: frozenset[Output]) -> None:
        """Drop the frames the answer of a last solve does not read, from the build and from *solver*.

        The solver has its own copy of the model by now, so the build's
        matrix, objective, columns and sets are a second copy. ``rows`` and
        the quadratic objective stay, since the run reads them. The columns
        stay where a basis is asked for, and the matrix and objective where a
        reduced cost is.
        """
        solver.release()
        handoff = self._model.handoff
        priced, based = 'reduced_cost' in outputs, 'basis' in outputs
        kept = replace(
            handoff,
            cols=handoff.cols if based else handoff.cols.clear(),
            obj=handoff.obj if priced else handoff.obj.clear(),
            qmatrix=handoff.qmatrix if priced else handoff.qmatrix.clear(),
            matrix=handoff.matrix if priced else handoff.matrix.clear(),
            sos=handoff.sos.clear(),
        )
        self._built = replace(self._model, handoff=kept)

    def _matched_start(
        self, start: Result | Mapping[str, Mapping[str, pl.LazyFrame]], solver_name: str
    ) -> Basis | np.ndarray:
        """*start* laid onto this build: a basis for an LP given one, else a value per column, NaN where none is given.

        A mixed-integer model starts from values, so a basis given it is not
        used. An LP given both starts from the basis.

        Raises:
            SpecsolveError: A start that gives this model nothing it starts
                from or lands nowhere on this build, or values for an LP that
                *solver_name* cannot take.
        """
        model = self._model
        given = start._start() if isinstance(start, Result) else start
        discrete = bool(self._discrete())
        if not discrete and (given.get('variable_basis') or given.get('constraint_basis')):
            return readback.matched_basis(model, given.get('variable_basis', {}), given.get('constraint_basis', {}))
        if not given.get('primal'):
            raise SpecsolveError(_nothing_to_start_message(discrete))
        values = readback.matched_values(model, given['primal'])
        if not discrete:
            _checked_lp_values(sinks.solver(solver_name), solver_name, values)
        return values

    def _answered(
        self,
        answer: SolveAnswer,
        solver_name: str,
        lower: Callable[[str | Mapping[str, object]], program.Expression] | None,
        outputs: frozenset[Output] = frozenset(),
    ) -> Result:
        """[`solve`][] after the run: *answer*'s vectors laid out against this build, as a [`Result`][].

        Takes the answer rather than the solver, so that a benchmark can time
        reading one back without a solve; every vector must span this build.
        """
        assert answer.primal is not None or not answer.status.is_readable, (
            'a readable status must come with a primal vector'
        )
        assert (answer.activity is None) == (answer.primal is None), (
            'activity travels with the primal: every sink reads it whenever a solution exists, mixed-integer included'
        )
        primals, duals, rays = self._read_back(answer.primal, answer.dual, answer.dual_ray)
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
            _outputs={kind: self._output(kind, answer) for kind in kinds_of(outputs)},
            _expressions=expressions,
            _evaluate=evaluate,
            _no_duals=no_duals,
            _dual_rays=rays,
            _no_dual_ray=None if answer.dual_ray is not None else _no_dual_ray_message(answer.status, solver_name),
        )

    def infeasible_subsystem(self) -> InfeasibleSubsystem:
        """The infeasible subsystem of the last solve, by declaration and coordinate. See [`infeasible_subsystem`][specsolve.api.Model.infeasible_subsystem].

        Raises:
            SpecsolveError: No solve since the last build, update or close; a
                last solve that was not infeasible; or no subsystem found.
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
        found = self._solver.infeasible_subsystem()
        if found is None:
            raise SpecsolveError(
                _no_infeasible_subsystem_message(type(self._solver).__name__.lower(), self._discrete())
            )
        return readback.infeasible_subsystem(self._model, found)

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
        dual_ray: pl.Series | None,
    ) -> tuple[dict[str, pl.LazyFrame], ...]:
        """One solve's answer as one frame per declaration — a [`Result`][]'s own.

        The frames reference this build's label frames, which [`build`][]
        replaces rather than mutates. A ``None`` vector yields no frames rather
        than empty ones.
        """
        return (
            {} if primal is None else self._per_variable(primal),
            {} if dual is None else self._per_constraint(dual),
            {} if dual_ray is None else self._per_constraint(dual_ray),
        )

    def _output(self, kind: str, answer: SolveAnswer) -> Mapping[str, pl.LazyFrame]:
        """The frames of *kind*, one of the [`OUTPUT_KINDS`][specsolve.relational.answer_layout.OUTPUT_KINDS], computed only when asked for; empty where a vector it needs is absent."""
        match kind:
            case 'activity':
                return {} if answer.activity is None else self._per_constraint(answer.activity)
            case 'reduced_cost':
                if answer.primal is None or answer.dual is None:
                    return {}
                return self._per_variable(readback.reduced_costs(self._model.handoff, answer.primal, answer.dual))
            case 'slack':
                return (
                    {}
                    if answer.activity is None
                    else self._per_constraint(readback.slacks(self._model.handoff, answer.activity))
                )
            case 'variable_basis':
                return {} if answer.basis is None else self._per_variable(_statuses(answer.basis.columns))
            case 'constraint_basis':
                return {} if answer.basis is None else self._per_constraint(_statuses(answer.basis.rows))
            case _:
                raise AssertionError(f'{kind!r} is none of the OUTPUT_KINDS, which kinds_of draws from')

    def _per_constraint(self, values: pl.Series) -> dict[str, pl.LazyFrame]:
        """A vector over the rows as one frame per constraint, as [`_read_back`][] lays out a dual."""
        model = self._model
        return {
            name: readback.laid_out(model.attached, model.constraints[name], c.dims, values)
            for name, c in model.program.constraints.items()
        }

    def _per_variable(self, values: pl.Series) -> dict[str, pl.LazyFrame]:
        """A vector over the columns as one frame per variable, as [`_read_back`][] lays out a primal."""
        model = self._model
        return {
            name: readback.laid_out(model.attached, model.variables[name], v.dims, values)
            for name, v in model.program.variables.items()
        }

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
        self._solved = None

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
