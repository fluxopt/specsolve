#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_cvar``: PyPSA's own CVaR risk preference. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_cvar.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**The risk preference is three variables and two rows.**
``set_risk_preference(alpha, omega)`` turns on ``define_cvar_variables``
(``variables.py:291``): ``CVaR-a`` over the scenarios, and the scalars
``CVaR-theta`` and ``CVaR``. ``define_objective`` then adds ``a_s >= OPEX_s -
theta`` per scenario and ``theta + 1/(1-alpha) * sum_s p_s a_s <= CVaR``, and
minimises ``CAPEX + (1-omega) * E[OPEX] + omega * CVaR``
(``optimize.py:377-419``). It is Rockafellar-Uryasev, and the epigraph is what
makes it linear.

**The tail is the operating cost only.** Capital cost sits outside the CVaR term
in PyPSA's objective, so risk aversion prices what a future *costs to run*, not
what it cost to build. Worth knowing before reading ``omega`` as a general
weight on regret.

**What the risk preference is worth** is printed by
:func:`what_risk_neutral_builds`: the same instance with no risk preference
builds a different fleet and reaches a different objective.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_cvar.json'


def load_tables() -> dict[str, pd.DataFrame | float]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame | float]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    The scenario half is ``pypsa_stochastic``'s: ``probability`` is
    ``scenario_weightings`` and ``load`` over ``(scenario, snapshot)`` is a
    ``p_set`` frame with a scenario level. On top of it, ``alpha`` and ``omega``
    — the two scalars the port declares — are exactly ``set_risk_preference``'s
    arguments, which is where the three auxiliary variables and their two rows
    come from.
    """
    n = pypsa.Network()
    n.set_snapshots(list(tables['snapshot']['snapshot']))
    n.set_scenarios(tables['probability'].set_index('scenario')['value'])
    n.set_risk_preference(alpha=float(tables['alpha']), omega=float(tables['omega']))

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
    ``(scenario, snapshot)``, there being one bus. Under a risk preference these
    prices are no longer the scenario weight times a marginal cost: a future
    outside the tail is priced by ``(1-omega) * p_s`` of its costs and one inside
    it by that plus ``omega * p_s / (1-alpha)``, which is why the severe column
    is the large one and none of them is round.
    """
    dual = n.model.constraints['Bus-nodal_balance'].dual.to_series()
    return {
        'scenario': [str(s) for s, _, _ in dual.index],
        'snapshot': [int(t) for _, _, t in dual.index],
        'value': [float(v) for v in dual.to_numpy()],
    }


def what_risk_neutral_builds(tables: dict[str, pd.DataFrame | float]) -> None:
    """Solve the same instance with the risk preference switched off.

    ``omega = 0`` is the risk-neutral objective exactly — ``(1-omega) E[OPEX] +
    omega CVaR`` collapses to the expectation.
    """
    neutral = build(tables | {'omega': 0.0})
    neutral.optimize(solver_name='highs')
    built = neutral.generators.p_nom_opt.xs('mild', level='scenario')
    print('\nwhat the risk-neutral model builds (omega = 0)')
    print(f'  fleet    {built.to_dict()}, total {float(built.sum())}')
    print(f'  cost     {float(neutral.objective)!r}')


def main() -> float:
    tables = load_tables()
    n = build(tables)
    status, condition = n.optimize(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    m = n.model
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(f'objective_constant {float(n.objective_constant)!r}')
    print(f'duals {json.dumps({"power_balance": balance_duals(n)})}')
    print('\nthe three auxiliary variables')
    print(f'  theta (VaR)  {float(m.variables["CVaR-theta"].solution)!r}')
    print(f'  CVaR         {float(m.variables["CVaR"].solution)!r}')
    print(f'  a            {[float(v) for v in m.variables["CVaR-a"].solution.to_numpy()]}')
    print('\nthe fleet, and what each future costs to run')
    print(n.generators.p_nom_opt.xs('mild', level='scenario'))
    opex = (
        (n.generators_t.p * n.generators.marginal_cost.xs('mild', level='scenario'))
        .T.groupby(level=0)
        .sum()
        .sum(axis=1)
    )
    print(opex)
    what_risk_neutral_builds(tables)
    return float(n.objective)


if __name__ == '__main__':
    main()
