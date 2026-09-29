#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_stochastic``: PyPSA's own two-stage stochastic network. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_stochastic.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**One dimension separates the two stages.** ``n.set_scenarios`` turns every
operational variable into a scenario-indexed one and leaves the nominal
variables alone: ``Generator-p`` comes out with dims ``(scenario, name,
snapshot)`` and ``Generator-p_nom`` with dims ``(name,)``. Capacity is chosen
once and lives through all three futures; dispatch is chosen after the load is
known. The port states it by giving ``p`` a ``scenario`` in its ``dims`` and
``p_nom`` none.

**Every objective term is weighted, capital cost included.** ``define_objective``
splits into ``capex_terms`` and ``opex_terms`` and runs both through
``_expected``, which selects each scenario and multiplies by its weight
(``optimize.py:361``). The weights sum to one, so the capital term is unchanged
by the round trip, and the port writes it once.

**What the expectation is worth** is printed by :func:`what_the_mean_would_build`:
the same network with every scenario replaced by the probability-weighted mean
load builds 141 MW and no peaking plant at all, and that fleet cannot serve the
severe future's 210 MW in any dispatch. The stochastic answer is not the answer
to the averaged question.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_stochastic.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    The port's ``probability`` table is PyPSA's ``scenario_weightings``, passed
    to ``set_scenarios``; the port's ``load`` over ``(scenario, snapshot)`` is a
    ``p_set`` frame whose columns carry the scenario level PyPSA gives every
    time-varying input once the network is stochastic. Both generators are
    extendable with no ``p_nom_max``: the fleet is what the model chooses, and a
    ceiling nothing reaches would be a parameter the port carries for nothing.
    """
    n = pypsa.Network()
    n.set_snapshots(list(tables['snapshot']['snapshot']))
    n.set_scenarios(tables['probability'].set_index('scenario')['value'])

    n.add('Bus', 'hub')
    generators: pd.DataFrame = tables['generator'].set_index('generator')
    n.add(
        'Generator',
        generators.index,
        bus='hub',
        p_nom_extendable=True,
        capital_cost=tables['capex'].set_index('generator')['value'],
        marginal_cost=tables['opex'].set_index('generator')['value'],
    )

    n.add('Load', 'l', bus='hub')
    load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='scenario', values='value')
    n.loads_t.p_set = pd.DataFrame(
        {(s, 'l'): load[s].to_numpy() for s in tables['probability']['scenario']}, index=n.snapshots
    )
    return n


def balance_duals(n: pypsa.Network) -> dict[str, list]:
    """The dual of the nodal balance per scenario and snapshot, keyed the way the port keys it.

    PyPSA's row index is ``(scenario, bus, snapshot)`` and the port's is
    ``(scenario, snapshot)``, there being one bus. The prices carry the scenario
    weight, because the objective terms they price do: ``mild`` at 6.0 is
    ``0.6 * 10``, not 10.
    """
    dual = n.model.constraints['Bus-nodal_balance'].dual.to_series()
    return {
        'scenario': [str(s) for s, _, _ in dual.index],
        'snapshot': [int(t) for _, _, t in dual.index],
        'value': [float(v) for v in dual.to_numpy()],
    }


def what_the_mean_would_build(tables: dict[str, pd.DataFrame]) -> None:
    """Solve the same network against the mean load, and try that fleet on the futures.

    The expected-value model collapses three futures into their
    probability-weighted average. Its fleet is infeasible in the severe future.
    """
    load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='scenario', values='value')
    weights: pd.Series = tables['probability'].set_index('scenario')['value']
    mean = (load * weights).sum(axis=1)

    averaged = tables | {
        'load': pd.DataFrame(
            {
                'scenario': ['mean'] * len(mean),
                'snapshot': list(mean.index),
                'value': list(mean.to_numpy()),
            }
        ),
        'probability': pd.DataFrame({'scenario': ['mean'], 'value': [1.0]}),
    }
    n = build(averaged)
    n.optimize(solver_name='highs')
    built = n.generators.p_nom_opt.xs('mean', level='scenario')

    print('\nwhat the mean load would have built')
    print(f'  mean load       {[float(v) for v in mean]}')
    print(f'  fleet           {built.to_dict()}, total {float(built.sum())}')
    print(f'  cost of that model {float(n.objective)!r}')
    print(f'  the severe future asks {float(tables["load"]["value"].max())} of it, so it has no dispatch at all')


def main() -> float:
    tables = load_tables()
    n = build(tables)
    status, condition = n.optimize(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(f'objective_constant {float(n.objective_constant)!r}')
    print(f'duals {json.dumps({"power_balance": balance_duals(n)})}')
    print('\nwhich dimensions each stage spans')
    for name in ('Generator-p_nom', 'Generator-p'):
        print(f'  {name:16} {n.model.variables[name].dims}')
    print('\ncapacity, chosen once')
    print(n.generators.p_nom_opt.xs(str(tables['probability']['scenario'][0]), level='scenario'))
    print('\ndispatch, chosen per scenario')
    print(n.generators_t.p)
    what_the_mean_would_build(tables)
    return float(n.objective)


if __name__ == '__main__':
    main()
