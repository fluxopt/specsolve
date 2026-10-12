"""The model the tutorials grow, one spec per step, solved on the data they share.

The first three tutorials solve ``examples/dispatch.yaml``. On-off generators
(``examples/tutorial/commitment.yaml``, which the fixing how-to prices) and a
battery (``examples/tutorial/storage.yaml``, which the last tutorial solves in
windows) each have to change the answer, or the page that uses them has
nothing to show. The data is
``examples/tutorial/data/``: four snapshots, and three days of them where a
rolling horizon needs room for windows.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

import specsolve as sps
from tests.conftest import EXAMPLES_DIR

TUTORIAL = EXAMPLES_DIR / 'tutorial'
DATA = TUTORIAL / 'data'

#: The columns of ``generators.csv`` each spec declares as a parameter.
COLUMNS = {
    'dispatch': ['p_max', 'cost'],
    'commitment': ['p_max', 'cost', 'p_min', 'no_load_cost'],
    'storage': ['p_max', 'cost', 'p_min', 'no_load_cost'],
}

SPECS = {
    'dispatch': EXAMPLES_DIR / 'dispatch.yaml',
    'commitment': TUTORIAL / 'commitment.yaml',
    'storage': TUTORIAL / 'storage.yaml',
}


def sources(step: str, load: pl.DataFrame) -> dict[str, Any]:
    """The tutorial data as one spec reads it: a table per parameter, and the labels of each dimension."""
    generators = pl.read_csv(DATA / 'generators.csv')
    tables: dict[str, Any] = {column: generators.select('generator', value=column) for column in COLUMNS[step]}
    tables |= {'generator': generators['generator'], 'load': load}
    if step == 'storage':
        tables['soc_initial'] = 0.0
    return tables


def one_day() -> pl.DataFrame:
    return pl.read_csv(DATA / 'load.csv')


def three_days() -> pl.DataFrame:
    """``load.csv`` three times over, on an ``hour`` column, which the window renumbers into ``snapshot``."""
    day = one_day()['value'].to_list()
    return pl.DataFrame({'hour': range(3 * len(day)), 'value': 3 * day})


@pytest.mark.parametrize(
    ('step', 'objective'),
    [
        pytest.param('dispatch', 10500.0, id='dispatch'),
        pytest.param('commitment', 14000.0, id='commitment'),
        pytest.param('storage', 10850.0, id='storage'),
    ],
)
def test_each_step_solves_to_the_objective_its_page_quotes(step: str, objective: float) -> None:
    load = one_day()
    result = sps.solve(SPECS[step], sources(step, load) | {'snapshot': load['snapshot']})
    assert result.objective == pytest.approx(objective), f'the {step} step no longer solves to {objective}'


def test_commitment_curtails_wind_to_keep_gas_at_its_minimum() -> None:
    """Gas cannot run below 50, so where the load leaves it less than that, wind gives way."""
    load = one_day()
    result = sps.solve(SPECS['commitment'], sources('commitment', load) | {'snapshot': load['snapshot']})
    wind = result.primal('p').filter(pl.col('generator') == 'wind')['value'].to_list()
    assert wind == pytest.approx([60.0, 70.0, 80.0, 40.0]), (
        'wind is curtailed in snapshots 1 and 3, where gas runs at its minimum of 50'
    )


def test_the_battery_lets_gas_switch_off() -> None:
    """The battery stores the wind that commitment curtailed, and covers snapshot 3 without gas."""
    load = one_day()
    result = sps.solve(SPECS['storage'], sources('storage', load) | {'snapshot': load['snapshot']})
    gas = result.primal('on').filter(pl.col('generator') == 'gas')['value'].to_list()
    assert gas == pytest.approx([0.0, 1.0, 1.0, 0.0]), 'gas is off in snapshot 3, where it was on without the battery'


def schedule_cost(sweep: sps.types.Sweep) -> float:
    """What the schedule the windows keep costs, without the lookahead each window also solved."""
    generators = pl.read_csv(DATA / 'generators.csv')
    output = sweep.primal('p').join(generators, on='generator')
    on = sweep.primal('on').join(generators, on='generator')
    return float((output['value'] * output['cost']).sum() + (on['value'] * on['no_load_cost']).sum())


def test_a_rolling_horizon_pays_for_what_its_windows_cannot_see() -> None:
    """Three-hour windows over a four-hour day: lookahead lets a window keep charge for the next one."""
    load = three_days()
    full = sps.solve(SPECS['storage'], sources('storage', load.rename({'hour': 'snapshot'})) | {'snapshot': range(12)})
    costs = []
    for lookahead in (0, 1, 3):
        sweep = sps.solve_over(
            SPECS['storage'],
            sources('storage', load),
            sps.EachWindow('hour', steps=3, lookahead=lookahead, into='snapshot'),
            carry={'soc_initial': 'soc'},
        )
        assert sweep.primal('soc')['hour'].to_list() == list(range(12)), 'the windows cover every hour exactly once'
        costs.append(schedule_cost(sweep))
    assert costs[0] > costs[1] > costs[2], f'each hour of lookahead has to buy something: {costs}'
    assert costs[2] == pytest.approx(full.objective), 'three hours of lookahead see as far as full foresight'
