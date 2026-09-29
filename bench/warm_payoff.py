"""Does carrying a basis across a genuine rebuild pay? A Benders master, swept.

    pixi run -e bench python -m bench.warm_payoff m
    pixi run -e bench python -m bench.warm_payoff s m l --steps 200 --wall

A capacity-expansion Benders (#382) whose master is sized from data and solved
three ways at every rebuild: cold, from the previous basis spliced per
declaration, and from that basis truncated to the new height. It is not an arm
and writes no results file; no ``src/`` code carries a basis across a rebuild.

The primary number is simplex iterations, which are deterministic. Wall time is
behind ``--wall`` and prints the load averages beside itself.

Rows are numbered per declaration, so a row gained by ``optimality_cut`` shifts
every row of ``feasibility_cut``: :func:`spliced` re-indexes for that, and
:func:`prefixed` does not.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import highspy
import numpy as np
import polars as pl
from mathspec import to_spec

import specsolve as sps
from bench.cases import Shape, _seed
from specsolve.relational.sinks.solvers import SOLVERS
from specsolve.relational.sinks.solvers.base import WarmStart

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from specsolve.relational.engines.polars.engine import PolarsEngine

MODELS = Path(__file__).resolve().parent / 'expansion'

#: The status of a spliced row the previous basis did not have: a fresh slack is basic.
BASIC = int(highspy.HighsBasisStatus.kBasic)

#: Generators per rung — the axis swept. The master is one column per
#: generator plus ``theta``, and one row per cut it has accumulated.
SIZES: Mapping[str, int] = {'xs': 6, 's': 100, 'm': 1_000, 'l': 10_000}

#: Snapshots in the dispatch subproblem, fixed across the ladder.
SNAPSHOTS = 24

#: The relative gap at which the decomposition stops.
TOLERANCE = 1e-6


@dataclass(frozen=True)
class Step:
    """One master rebuild, solved three ways."""

    columns: int
    rows: int
    nonzeros: int
    cold_iterations: int
    warm_iterations: int
    naive_iterations: int
    cold_objective: float
    warm_objective: float
    cold_seconds: float
    warm_seconds: float


@dataclass(frozen=True)
class Run:
    """A whole decomposition at one size."""

    generators: int
    snapshots: int
    steps: tuple[Step, ...]
    converged: bool
    lower: float
    upper: float

    @property
    def cold_iterations(self) -> int:
        return sum(s.cold_iterations for s in self.steps)

    @property
    def warm_iterations(self) -> int:
        return sum(s.warm_iterations for s in self.steps)

    @property
    def naive_iterations(self) -> int:
        return sum(s.naive_iterations for s in self.steps)

    @property
    def nonzeros(self) -> int:
        """Coefficients the run emitted into the master, over every rebuild."""
        return sum(s.nonzeros for s in self.steps)

    @property
    def cold_seconds(self) -> float:
        return sum(s.cold_seconds for s in self.steps)

    @property
    def warm_seconds(self) -> float:
        return sum(s.warm_seconds for s in self.steps)


def instance(n_gen: int, n_snap: int) -> dict[str, pl.DataFrame]:
    """Seeded data for a capacity expansion with a real build-or-run trade.

    Marginal cost and capital cost are anti-correlated, so the answer is a
    portfolio. Each snapshot's load is half of what the full portfolio could
    serve, so the problem is feasible while the subproblem is infeasible at zero
    capacity, and both cut families grow.
    """
    rng = _seed(Shape('warm_payoff', {'generator': n_gen, 'snapshot': n_snap}, n_gen * n_snap))
    gens = [f'g{i:05d}' for i in range(n_gen)]

    cap_max = rng.uniform(50.0, 150.0, n_gen)
    cost = rng.uniform(10.0, 100.0, n_gen)
    invest = (110.0 - cost) * rng.uniform(0.8, 1.2, n_gen)
    avail = rng.uniform(0.2, 1.0, (n_snap, n_gen))

    return {
        'generator': pl.DataFrame({'generator': gens}),
        'snapshot': pl.DataFrame({'snapshot': np.arange(n_snap)}),
        'invest': pl.DataFrame({'generator': gens, 'value': invest}),
        'cap_max': pl.DataFrame({'generator': gens, 'value': cap_max}),
        'cost': pl.DataFrame({'generator': gens, 'value': cost}),
        'load': pl.DataFrame({'snapshot': np.arange(n_snap), 'value': 0.5 * (avail @ cap_max)}),
        'avail': pl.DataFrame(
            {
                'snapshot': np.repeat(np.arange(n_snap), n_gen),
                'generator': gens * n_snap,
                'value': avail.ravel(),
            }
        ),
    }


def spliced(
    previous: WarmStart,
    was: Mapping[str, Any],
    now: Mapping[str, Any],
    order: Sequence[str],
    n_rows: int,
) -> WarmStart:
    """*previous* re-indexed onto a model whose declarations changed height.

    *was* and *now* are the engine's ``name -> Labelled`` maps before and after
    the rebuild, and *order* the declaration order they were numbered in. Each
    declaration keeps the leading rows it still has and the rest start
    :data:`BASIC`; a declaration that appeared or vanished is skipped, which
    leaves its rows basic too. Columns cross unchanged.
    """
    assert previous.row_statuses is not None, 'a spliced start carries a basis; an incumbent has no rows to splice'
    rows = np.full(n_rows, BASIC, dtype=np.int8)
    for name in order:
        old, new = was.get(name), now.get(name)
        if old is None or new is None:
            continue
        keep = min(old.height, new.height)
        rows[new.start : new.start + keep] = previous.row_statuses[old.start : old.start + keep]
    return WarmStart(
        solver='highs',
        column_statuses=previous.column_statuses,
        row_statuses=rows,
        column_values=None,
    )


def prefixed(previous: WarmStart, n_rows: int) -> WarmStart:
    """*previous* truncated or padded to *n_rows*, ignoring declaration order.

    Right only for a model whose growth is all in the last declaration.
    """
    assert previous.row_statuses is not None, 'a prefix carry needs a basis; an incumbent has no rows to truncate'
    rows = np.full(n_rows, BASIC, dtype=np.int8)
    keep = min(len(previous.row_statuses), n_rows)
    rows[:keep] = previous.row_statuses[:keep]
    return WarmStart(
        solver='highs',
        column_statuses=previous.column_statuses,
        row_statuses=rows,
        column_values=None,
    )


def _solved(tables: Any, start: WarmStart | None) -> tuple[Any, int, float, WarmStart | None]:
    """A fresh HiGHS session on *tables*, optionally started from *start*.

    The iteration count is read off the private handle; no public surface
    reports it.
    """
    solver = SOLVERS['highs'](tables)
    if start is not None:
        solver.warm(start)
    began = time.perf_counter()
    answer = solver.run(tables)
    seconds = time.perf_counter() - began
    iterations = int(solver._handle.getInfo().simplex_iteration_count)
    carried = solver.warm_start()
    solver.close()
    return answer, iterations, seconds, carried


def _slope_at(solution: sps.Result, avail: pl.DataFrame, capacity: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """The subproblem's subgradient in capacity, and its value at *capacity*.

    The subgradient is the capacity row's dual, weighted by availability and
    summed over snapshots.
    """
    slope = (
        solution.dual('capacity')
        .join(avail, on=['snapshot', 'generator'], suffix='_avail')
        .with_columns((pl.col('value') * pl.col('value_avail')).alias('term'))
        .group_by('generator')
        .agg(pl.col('term').sum().alias('slope'))
    )
    here = slope.join(capacity, on='generator').select((pl.col('slope') * pl.col('value')).sum()).item()
    return slope, here


def _appended(tables: dict[str, pl.DataFrame], family: str, constant: float, slope: pl.DataFrame) -> None:
    """One more cut in *family*, in place — the master's rows are its data."""
    index = tables[f'{family}_const'].height
    tables[f'{family}_const'] = pl.concat(
        [tables[f'{family}_const'], pl.DataFrame({family: [index], 'value': [constant]})]
    )
    tables[f'{family}_slope'] = pl.concat(
        [
            tables[f'{family}_slope'],
            slope.select(pl.lit(index, dtype=pl.Int64).alias(family), 'generator', pl.col('slope').alias('value')),
        ]
    )


def _empty_cuts() -> dict[str, pl.DataFrame]:
    return {
        'cut_const': pl.DataFrame(schema={'cut': pl.Int64, 'value': pl.Float64}),
        'cut_slope': pl.DataFrame(schema={'cut': pl.Int64, 'generator': pl.String, 'value': pl.Float64}),
        'fcut_const': pl.DataFrame(schema={'fcut': pl.Int64, 'value': pl.Float64}),
        'fcut_slope': pl.DataFrame(schema={'fcut': pl.Int64, 'generator': pl.String, 'value': pl.Float64}),
    }


def _blocks(engine: PolarsEngine) -> tuple[dict[str, Any], list[str]]:
    """The engine's row blocks and the order they were numbered in."""
    return dict(engine._model.constraints), list(engine._model.program.constraints)


def sweep(n_gen: int, n_snap: int = SNAPSHOTS, steps: int = 200) -> Run:
    """Run the decomposition once, solving every master rebuild three ways.

    The cold answer drives the loop, so all three see the same masters. Each
    carrying arm chains its own basis, and the warm objective is asserted equal
    to the cold one every step.
    """
    data = instance(n_gen, n_snap)
    gens = data['invest']['generator'].to_list()
    dispatch = {name: data[name] for name in ('generator', 'snapshot', 'cost', 'load', 'avail')}

    def slice_for(spec: Any, **extra: Any) -> dict[str, Any]:
        """The part of *dispatch* this spec declares — `feasibility` reads no cost."""
        known = to_spec(spec)
        names = {**known.parameters, **known.dimensions, **known.relations}
        return {name: frame for name, frame in {**dispatch, **extra}.items() if name in names}

    cuts = _empty_cuts()
    capacity = pl.DataFrame({'generator': gens, 'value': [0.0] * n_gen})
    upper, lower = float('inf'), float('-inf')
    carried: WarmStart | None = None
    naive: WarmStart | None = None
    was: dict[str, Any] = {}
    taken: list[Step] = []
    converged = False

    with (
        sps.build(MODELS / 'sub.yaml', slice_for(MODELS / 'sub.yaml', cap_hat=capacity)) as sub_model,
        sps.build(MODELS / 'feasibility.yaml', slice_for(MODELS / 'feasibility.yaml', cap_hat=capacity)) as short_model,
        sps.build(
            MODELS / 'master.yaml',
            {'invest': data['invest'], 'cap_max': data['cap_max'], **cuts, 'generator': gens, 'cut': [], 'fcut': []},
        ) as master,
    ):
        for _ in range(steps):
            sub = sub_model.update({'cap_hat': capacity}).solve()
            if sub.has_primal:
                slope, here = _slope_at(sub, data['avail'], capacity)
                spent = capacity.join(data['invest'], on='generator', suffix='_rate')
                upper = min(upper, spent.select((pl.col('value') * pl.col('value_rate')).sum()).item() + sub.objective)
                _appended(cuts, 'cut', sub.objective - here, slope)
            else:
                short = short_model.update({'cap_hat': capacity}).solve()
                slope, here = _slope_at(short, data['avail'], capacity)
                _appended(cuts, 'fcut', here - short.objective, slope)

            master.update(
                {
                    **cuts,
                    'cut': cuts['cut_const']['cut'].to_list(),
                    'fcut': cuts['fcut_const']['fcut'].to_list(),
                }
            )
            engine = master._engine
            built = engine._model.handoff
            now, order = _blocks(engine)

            cold, cold_iterations, cold_seconds, _ = _solved(built, None)
            start = None if carried is None else spliced(carried, was, now, order, built.row_count)
            warm, warm_iterations, warm_seconds, carried = _solved(built, start)
            crude = None if naive is None else prefixed(naive, built.row_count)
            _, naive_iterations, _, naive = _solved(built, crude)
            assert abs(warm.objective - cold.objective) <= 1e-6 * max(abs(cold.objective), 1.0), (
                f'a carried basis moved the answer: cold {cold.objective!r}, warm {warm.objective!r} '
                f'at {built.row_count} rows — a warm start may move the route and never the optimum'
            )

            taken.append(
                Step(
                    columns=built.column_count,
                    rows=built.row_count,
                    nonzeros=built.matrix.height,
                    cold_iterations=cold_iterations,
                    warm_iterations=warm_iterations,
                    naive_iterations=naive_iterations,
                    cold_objective=cold.objective,
                    warm_objective=warm.objective,
                    cold_seconds=cold_seconds,
                    warm_seconds=warm_seconds,
                )
            )

            was = now
            lower = cold.objective
            assert cold.primal is not None, 'the master is bounded and feasible at every capacity it proposes'
            capacity = pl.DataFrame(
                {'generator': gens, 'value': engine._model.variables['cap'].share(cold.primal).to_list()}
            )
            if upper < float('inf') and upper - lower <= TOLERANCE * abs(upper):
                converged = True
                break

    return Run(n_gen, n_snap, tuple(taken), converged, lower, upper)


def _report(run: Run, wall: bool) -> None:
    last = run.steps[-1]
    verdict = 'converged' if run.converged else 'stopped at the step budget'
    print(f'\n{run.generators} generators x {run.snapshots} snapshots — {verdict} after {len(run.steps)} rebuilds')
    print(f'  master: {last.columns} columns, {last.rows} rows, {last.nonzeros} nonzeros at the last rebuild')
    print(f'  bounds: lower {run.lower:.6g}, upper {run.upper:.6g}')

    print('\n  step   rows   cold iters   spliced   prefix')
    for i, s in enumerate(run.steps):
        print(f'  {i:4}  {s.rows:5}   {s.cold_iterations:10}   {s.warm_iterations:7}   {s.naive_iterations:6}')
    saved = 1 - run.warm_iterations / run.cold_iterations if run.cold_iterations else 0.0
    print(
        f'\n  total simplex iterations: cold {run.cold_iterations}, spliced {run.warm_iterations} '
        f'({saved:.1%} saved), naive prefix {run.naive_iterations}'
    )
    print(f'  coefficients the rebuilds emitted, whatever the solve started from: {run.nonzeros}')

    if wall:
        one, five, fifteen = os.getloadavg()
        print(f'  master solve wall seconds: cold {run.cold_seconds:.3f}, warm {run.warm_seconds:.3f}')
        print(f'  (load averages {one:.2f} {five:.2f} {fifteen:.2f} — only meaningful near zero)')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m bench.warm_payoff', description=__doc__)
    parser.add_argument('sizes', nargs='+', choices=sorted(SIZES), help='generators per rung')
    parser.add_argument('--snapshots', type=int, default=SNAPSHOTS, help='snapshots in the dispatch subproblem')
    parser.add_argument('--steps', type=int, default=200, help='cap on Benders iterations')
    parser.add_argument('--wall', action='store_true', help='also print master solve wall seconds and the load average')
    args = parser.parse_args(argv)

    for size in args.sizes:
        _report(sweep(SIZES[size], args.snapshots, args.steps), args.wall)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
