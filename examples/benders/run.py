"""Benders decomposition on specsolve, checked against the model it decomposes.

    pixi run python examples/benders/run.py

It shows what the language can express and that the answer is right; specsolve
ships no decomposition driver (https://github.com/fluxopt/specsolve/issues/596).

Three files:

- ``monolith.yaml``  the problem in one plan — the answer everything else must reach
- ``master.yaml``    capacity, plus a placeholder for what operating it will cost
- ``sub.yaml``       dispatch at a capacity someone else chose; infeasible if it is too small

Both cut families come out of the *same* subproblem. An optimality cut is read
off its prices, and a feasibility cut off ``dual_ray`` — the certificate an
infeasible solve leaves, which says which combination of rows cannot hold.

The master's cuts are **data**. It declares ``cut`` and ``fcut`` with members
from data and never changes; an iteration appends rows to their parameter
tables. No YAML is written at runtime.

Each spec is parsed and built once, above the loop, and ``update`` puts the
next iteration's numbers on the model that is already there. The subproblem's
``cap_hat`` reaches its rows as a right-hand side, so HiGHS keeps the model it
holds and re-solves from the last basis; the master grows a row a step and is
loaded again, as ``diagnostics().loads`` reports.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from mathspec import to_spec

import specsolve as sps

HERE = Path(__file__).parent
SNAPSHOTS = [0, 1, 2, 3]
GENERATORS = ['wind', 'gas']

SUB = to_spec(HERE / 'sub.yaml')
MASTER = to_spec(HERE / 'master.yaml')

SOURCES = {
    'snapshot': pl.DataFrame({'snapshot': SNAPSHOTS}),
    'generator': pl.DataFrame({'generator': GENERATORS}),
    'invest': pl.DataFrame({'generator': GENERATORS, 'value': [90.0, 30.0]}),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 25.0]}),
    'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [40.0, 80.0, 55.0, 95.0]}),
    'avail': pl.DataFrame(
        {
            'snapshot': [s for s in SNAPSHOTS for _ in GENERATORS],
            'generator': GENERATORS * len(SNAPSHOTS),
            'value': [0.9, 1.0, 0.2, 1.0, 0.6, 1.0, 0.1, 1.0],
        }
    ),
}


def slice_for(spec, **extra):
    """The part of ``SOURCES`` *spec* declares, plus what this call adds.

    Attaching refuses a name a spec does not declare, so a driver over several
    models says which slice it means.
    """
    known = {**spec.parameters, **spec.dimensions}
    return {name: frame for name, frame in {**SOURCES, **extra}.items() if name in known}


EMPTY = {
    'cut_const': pl.DataFrame(schema={'cut': pl.Int64, 'value': pl.Float64}),
    'cut_slope': pl.DataFrame(schema={'cut': pl.Int64, 'generator': pl.String, 'value': pl.Float64}),
    'fcut_const': pl.DataFrame(schema={'fcut': pl.Int64, 'value': pl.Float64}),
    'fcut_slope': pl.DataFrame(schema={'fcut': pl.Int64, 'generator': pl.String, 'value': pl.Float64}),
}


def slope_at(solution: sps.types.Result, capacity: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """How the subproblem's value moves with capacity, and its value there.

    The capacity constraint's shadow price is that derivative, weighted by
    availability and summed over snapshots. Reading it needs nothing but
    ``dual`` and a join against the model's own ``avail`` table.
    """
    slope = (
        solution.dual('capacity')
        .join(SOURCES['avail'], on=['snapshot', 'generator'], suffix='_avail')
        .with_columns((pl.col('value') * pl.col('value_avail')).alias('term'))
        .group_by('generator')
        .agg(pl.col('term').sum().alias('slope'))
    )
    here = slope.join(capacity, on='generator').select((pl.col('slope') * pl.col('value')).sum()).item()
    return slope, here


def cut_from_ray(solution: sps.types.Result) -> tuple[pl.DataFrame, float]:
    """The feasibility cut carried by a certificate that this capacity cannot be dispatched.

    ``dual_ray`` weights the subproblem's rows so that together they
    contradict: weight each capacity row by its u and each balance row by its
    v, and ``Σ u·cap_hat·avail + Σ v·load`` comes out positive, which is the
    proof. Every term is linear in capacity, so asking the master for a
    capacity where that same combination is *not* positive is one row —
    slope ``Σ_s u·avail`` per generator, against ``minus Σ_s v·load``.
    """
    slope = (
        solution.dual_ray('capacity')
        .join(SOURCES['avail'], on=['snapshot', 'generator'], suffix='_avail')
        .with_columns((pl.col('value') * pl.col('value_avail')).alias('term'))
        .group_by('generator')
        .agg(pl.col('term').sum().alias('slope'))
    )
    against = (
        solution.dual_ray('balance')
        .join(SOURCES['load'], on='snapshot', suffix='_load')
        .select((pl.col('value') * pl.col('value_load')).sum())
        .item()
    )
    return slope, -against


def appended(tables: dict[str, pl.DataFrame], family: str, constant: float, slope: pl.DataFrame) -> None:
    """One more cut in *family*, in place."""
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


def main() -> None:
    """The decomposition loop, against the same problem solved in one plan.

    An infeasible subproblem has no duals to read, so the cut comes from its
    ``dual_ray`` instead.

    The gap is only checked once some capacity has proved dispatchable: every
    feasibility cut leaves the upper bound at infinity.
    """
    with sps.solve(HERE / 'monolith.yaml', SOURCES) as whole:
        truth = whole.objective
    print(f'the whole problem, in one plan: {truth:.2f}\n')

    tables = dict(EMPTY)
    capacity = pl.DataFrame({'generator': GENERATORS, 'value': [0.0] * len(GENERATORS)})
    upper = float('inf')
    empty = {'generator': GENERATORS, 'cut': [], 'fcut': []}

    with (
        sps.build(SUB, slice_for(SUB, cap_hat=capacity)) as sub_model,
        sps.build(MASTER, {'invest': SOURCES['invest'], **tables, **empty}) as master,
    ):
        for step in range(25):
            sub = sub_model.update({'cap_hat': capacity}).solve()
            dispatchable = sub.has_primal
            if dispatchable:
                slope, here = slope_at(sub, capacity)
                spent = capacity.join(SOURCES['invest'], on='generator', suffix='_rate')
                upper = min(upper, spent.select((pl.col('value') * pl.col('value_rate')).sum()).item() + sub.objective)
                appended(tables, 'cut', sub.objective - here, slope)
            else:
                slope, against = cut_from_ray(sub)
                appended(tables, 'fcut', against, slope)

            coordinates = {'cut': tables['cut_const']['cut'].to_list(), 'fcut': tables['fcut_const']['fcut'].to_list()}
            answer = master.update({**tables, **coordinates}).solve()
            lower = answer.objective
            capacity = answer.primal('cap').select('generator', 'value')

            kind = 'optimality' if dispatchable else 'feasibility'
            bound = f'{upper:.2f}' if upper < float('inf') else 'none yet'
            print(f'  step {step}  {kind:11}  lower {lower:8.2f}   upper {bound}')
            if upper < float('inf') and upper - lower <= 1e-6 * abs(upper):
                break

        print(f'\ndecomposed: {upper:.2f} in {step + 1} steps')
        print(f'monolithic: {truth:.2f}')
        print(f'difference: {abs(upper - truth):.1e}')
        print(f'cuts: {tables["cut_const"].height} optimality, {tables["fcut_const"].height} feasibility')
        for name, model in (('the subproblem', sub_model), ('the master', master)):
            seen = model.diagnostics()
            print(f'{name} loaded the solver {seen.loads} time(s) in {seen.solves} solves')


if __name__ == '__main__':
    main()
