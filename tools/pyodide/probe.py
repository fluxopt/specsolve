"""specsolve in Pyodide, before pyarrow is loaded and after, each answer checked against its closed form.

Run by ``run.mjs`` inside Pyodide, never by pytest. The polars there has no
streaming engine and no parquet. Before pyarrow: a solve and a sweep from
frames, the sweep over an axis read from JSON, and an archive refused by name.
After: the sweep archived as a zip and read back, and solved again from it.
"""

from __future__ import annotations

import json

import polars as pl

import specsolve as sps
from specsolve._relational.collect import collect_engine

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


#: Each scenario's flat load, and the cost the gas pays to cover what the wind does not.
FLAT = {'low': 60.0, 'high': 120.0}
EXPECTED = {name: GAS_COST * (load - WIND) * len(LOAD) for name, load in FLAT.items()}


def _sources() -> dict[str, object]:
    return {
        'g': GENERATORS,
        't': list(range(len(LOAD))),
        'cap': pl.DataFrame({'g': GENERATORS, 'value': [WIND, 200.0]}),
        'cost': pl.DataFrame({'g': GENERATORS, 'value': [0.0, GAS_COST]}),
        'load': pl.DataFrame({'t': range(len(LOAD)), 'value': LOAD}),
    }


def _by_scenario() -> dict[str, object]:
    scenarios = pl.DataFrame(
        {
            'scenario': [name for name in FLAT for _ in LOAD],
            't': list(range(len(LOAD))) * len(FLAT),
            'value': [load for load in FLAT.values() for _ in LOAD],
        }
    )
    return {**_sources(), 'load': scenarios}


def _objectives(sweep: sps.types.Sweep) -> dict[str, float]:
    return dict(sweep.record.select('scenario', 'objective').iter_rows())


def without_pyarrow() -> str:
    sources = _sources()
    assert collect_engine() == 'in-memory', "Pyodide's polars refuses the streaming engine, and the probe saw it"

    result = sps.solve(SPEC, sources)
    assert result.termination_condition == 'optimal', result.termination_condition
    expected = GAS_COST * sum(load - WIND for load in LOAD)
    assert abs(result.objective - expected) < 1e-6, (
        f'gas covers the load above the wind: {result.objective} != {expected}'
    )

    axis = json.loads(json.dumps(sps.EachCoordinate('scenario').to_dict()))
    objectives = _objectives(sps.solve_over(SPEC, _by_scenario(), axis))
    assert objectives == EXPECTED, f'one solve per scenario, gas covering the flat load above the wind: {objectives}'
    try:
        sps.solve(SPEC, sources, archive='/tmp/refused')
    except sps.errors.SpecsolveError as refused:
        assert 'this polars was built without it' in str(refused), f'the refusal names parquet: {refused}'
    else:
        raise AssertionError("Pyodide's polars has no parquet, and archive= went ahead regardless")
    return (
        f'polars {pl.__version__}, specsolve {sps.__version__}: a solve, a sweep of {len(objectives)}, '
        f'and an archive refused for want of parquet'
    )


def with_pyarrow() -> str:
    archived = sps.solve_over(SPEC, _by_scenario(), sps.EachCoordinate('scenario'), archive='/tmp/study.zip')
    assert _objectives(archived) == EXPECTED, 'the archived sweep answers as the one in memory did'
    study = sps.load_archive('/tmp/study.zip', '/tmp/study')
    assert _objectives(study.sweep) == EXPECTED, 'the archive reads back through pyarrow'
    assert study.sweep.primal('p').equals(archived.primal('p')), 'every value it wrote'
    again = sps.solve_over(study.spec, study.sources, study.axis)
    assert _objectives(again) == EXPECTED, 'and solves again from what it holds'
    return 'a sweep archived as a zip through pyarrow, read back and solved again from it'
