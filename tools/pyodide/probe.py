"""specsolve in Pyodide: a build and solve from polars frames, then a sweep over an axis read from JSON, each checked against its closed form.

Run by ``run.mjs`` inside Pyodide, never by pytest: the polars there has no
streaming engine and no parquet, so the sources are frames and nothing is
written.
"""

from __future__ import annotations

import json

import polars as pl

import specsolve as sps
from specsolve.relational.collect import collect_engine

GENERATORS = ['wind', 'gas']
LOAD = [80.0, 90.0, 100.0, 70.0, 60.0, 120.0]
WIND, GAS_COST = 50.0, 40.0
SPEC = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
    'parameters': {'cap': {'dims': ['g']}, 'cost': {'dims': ['g']}, 'load': {'dims': ['t']}},
    'variables': {'p': {'dims': ['g', 't'], 'bounds': {'lower': 0}}},
    'constraints': {
        'limit': {'dims': ['g', 't'], 'expression': 'p <= cap'},
        'balance': {'dims': ['t'], 'expression': 'sum(p, over=g) == load'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}


def main() -> str:
    sources = {
        'g': GENERATORS,
        't': list(range(len(LOAD))),
        'cap': pl.DataFrame({'g': GENERATORS, 'value': [WIND, 200.0]}),
        'cost': pl.DataFrame({'g': GENERATORS, 'value': [0.0, GAS_COST]}),
        'load': pl.DataFrame({'t': range(len(LOAD)), 'value': LOAD}),
    }
    assert collect_engine() == 'in-memory', "Pyodide's polars refuses the streaming engine, and the probe saw it"

    result = sps.solve(SPEC, sources)
    assert result.termination_condition == 'optimal', result.termination_condition
    expected = GAS_COST * sum(load - WIND for load in LOAD)
    assert abs(result.objective - expected) < 1e-6, (
        f'gas covers the load above the wind: {result.objective} != {expected}'
    )

    flat = {'low': 60.0, 'high': 120.0}
    scenarios = pl.DataFrame(
        {
            'scenario': [name for name in flat for _ in LOAD],
            't': list(range(len(LOAD))) * len(flat),
            'value': [load for load in flat.values() for _ in LOAD],
        }
    )
    axis = json.loads(json.dumps(sps.EachCoordinate('scenario').to_dict()))
    runs = sps.solve_over(SPEC, {**sources, 'load': scenarios}, axis)
    objectives = dict(runs.record.select('scenario', 'objective').iter_rows())
    assert objectives == {name: GAS_COST * (load - WIND) * len(LOAD) for name, load in flat.items()}, (
        f'one solve per scenario, gas covering the flat load above the wind: {objectives}'
    )
    return f'polars {pl.__version__}, specsolve {sps.__version__}: a solve and a sweep of {len(objectives)}'
