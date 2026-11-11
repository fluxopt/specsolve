"""The fuzzer in ``tests/fuzz/`` still draws models the language accepts, and every case it found still fails.

The campaign itself is ``pixi run python -m tests.fuzz``; this module keeps the
generator honest and holds each finding, reduced, as a strict ``xfail``. A
case is checked by the same oracles a campaign runs, so the day its fix lands
it XPASSes, the marker comes off, and the case stays as the regression.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Any

import polars as pl
import pytest
from mathspec import to_spec
from mathspec.errors import MathSpecError

from tests.fuzz.check import check
from tests.fuzz.generate import Case, generate

#: Of this many seeds, at least ``ACCEPTED`` must load: below it the campaign checks the refusal path, not the engine.
SEEDS = 200
ACCEPTED = 0.95

T = {'t': {'dtype': 'int', 'ordered': True}}
G = {'g': {'dtype': 'str'}}


@pytest.fixture(autouse=True)
def _quiet_linopy() -> None:
    logging.getLogger('linopy').setLevel(logging.ERROR)


def test_the_generator_stays_inside_the_language() -> None:
    accepted = 0
    for seed in range(SEEDS):
        try:
            to_spec(generate(seed).spec)
            accepted += 1
        except MathSpecError:
            pass
    assert accepted >= ACCEPTED * SEEDS, f'only {accepted} of {SEEDS} generated models load'


def _drawn(seed: int, hash_seed: str) -> str:
    """Seed *seed*'s spec and data, drawn in a fresh interpreter under *hash_seed*."""
    script = (
        'import json, sys; from tests.fuzz.generate import generate; c = generate(int(sys.argv[1])); '
        'print(json.dumps([c.spec, {k: v.to_dicts() if hasattr(v, "to_dicts") else v for k, v in c.sources.items()}, c.reads], default=str))'
    )
    env = {**os.environ, 'PYTHONHASHSEED': hash_seed}
    return subprocess.run(
        [sys.executable, '-c', script, str(seed)], env=env, capture_output=True, text=True, check=True
    ).stdout


@pytest.mark.parametrize('seed', [11, 135, 349])
def test_a_seed_is_the_whole_reproduction(seed: int) -> None:
    """A campaign's baseline names seeds, so a seed must draw one case in every process, whatever the hash seed."""
    assert _drawn(seed, '1') == _drawn(seed, '2'), f'seed {seed} draws a different case under another hash seed'


@pytest.mark.parametrize('seed', range(4))
def test_every_oracle_runs_to_a_verdict(seed: int) -> None:
    """The oracles return findings rather than raising, whatever the seed draws."""
    outcome = check(generate(seed))
    assert outcome.status in {'rejected', 'refused', 'checked'}, f'an oracle ended the case as {outcome.status!r}'


def _model(dimensions: dict[str, Any], constraint: str, dims: list[str], **sections: Any) -> dict[str, Any]:
    """A spec with one constraint and an objective that maximises every variable it declares."""
    variables = sections.pop('variables')
    return {
        'dimensions': dimensions,
        'variables': variables,
        **sections,
        'constraints': {'c': {'dims': dims, 'expression': constraint}},
        'objective': {
            'sense': 'maximize',
            'expression': ' + '.join(f'sum({v})' if variables[v]['dims'] else v for v in variables),
        },
    }


def _bounded(*dims: str) -> dict[str, Any]:
    return {'dims': list(dims), 'bounds': {'lower': 0, 'upper': 1}}


def _case(spec: dict[str, Any], sources: dict[str, Any], *reads: str, linear: bool = True) -> Case:
    return Case(seed=0, spec=spec, sources=sources, reads=list(reads), linear=linear)


def _found(reason: str) -> Any:
    return pytest.mark.xfail(reason=reason, raises=AssertionError, strict=True)


REDUCED = [
    pytest.param(
        _case(
            _model(
                {**T, **G},
                'shift(shift(v, along=t, offset=1), along=t, offset=1, edge=0) <= v',
                ['g', 't'],
                variables={'v': _bounded('g', 't')},
            ),
            {'t': [1, 2], 'g': ['a']},
        ),
        id='nested shift over two dimensions',
    ),
    pytest.param(
        _case(
            _model(
                {**T, **G},
                'shift(x + y, along=t, offset=1) <= 1',
                ['g', 't'],
                variables={'x': _bounded('t', 'g'), 'y': _bounded('g')},
            ),
            {'t': [1, 2, 3], 'g': ['a']},
        ),
        id='shift over a term without the dimension',
    ),
    pytest.param(
        _case(
            _model(
                {**T, **G},
                'sum_back(x + y, along=t, window=2) <= 1',
                ['g', 't'],
                variables={'x': _bounded('t', 'g'), 'y': _bounded('g')},
            ),
            {'t': [1, 2, 3], 'g': ['a']},
        ),
        id='sum_back over a term without the dimension',
    ),
    *(
        pytest.param(
            _case(
                _model(
                    {**T, **G},
                    f'{operator}(x + y, along=t, {amount}) <= 1',
                    ['g', 't'],
                    variables={'x': _bounded('t', 'g'), 'y': {**_bounded('g'), 'where': "g != 'b'", **missing}},
                ),
                {'t': [1, 2, 3], 'g': ['a', 'b']},
            ),
            id=f'{operator} over a {label} term without the dimension',
        )
        for operator, amount in [('shift', 'offset=1, edge=0'), ('sum_back', 'window=2')]
        for label, missing in [('masked', {}), ('neutral', {'missing': 'neutral'})]
    ),
    pytest.param(
        _case(
            _model(
                {**T, **G},
                'sum(x + y, over=t) <= 2',
                ['g'],
                variables={'x': {**_bounded('t', 'g'), 'where': 't != 2'}, 'y': _bounded('g')},
            ),
            {'t': [1, 2, 3], 'g': ['a', 'b']},
        ),
        id='sum over a term without the dimension beside a masked term',
    ),
    pytest.param(
        _case(
            _model(
                {**T, **G},
                'sum_back(x + y + q, along=t, window=w) <= 4',
                ['g', 't'],
                parameters={'w': {'dims': ['g'], 'dtype': 'int'}, 'q': {'dims': ['t']}},
                variables={'x': _bounded('t', 'g'), 'y': {**_bounded('t'), 'where': 't != 2'}},
            ),
            {
                't': [1, 2, 3],
                'g': ['a', 'b'],
                'w': pl.DataFrame({'g': ['a', 'b'], 'value': [1, 2]}),
                'q': pl.DataFrame({'t': [1, 2, 3], 'value': [1.0, 0.5, 0.25]}),
            },
        ),
        id='a parameter window over a masked term without its dimension',
    ),
    pytest.param(
        _case(
            _model(
                {'h': {'dtype': 'str'}, 'k': {'dtype': 'str'}, **G},
                'at(p + y, by=r[h]) <= 0.5',
                ['g', 'k'],
                relations={'r': {'key': 'k', 'values': 'h'}},
                parameters={'p': {'dims': ['h', 'g']}},
                variables={'y': _bounded('g')},
            ),
            {
                'h': ['h1'],
                'k': ['k1', 'k2'],
                'g': ['g1'],
                'p': pl.DataFrame({'h': ['h1'], 'g': ['g1'], 'value': [-0.5]}),
                'r': pl.DataFrame({'k': ['k1', 'k2'], 'h': ['h1', 'h1']}),
            },
        ),
        id='at over a term without the column',
    ),
    pytest.param(
        _case(
            _model(
                {'k': {'dtype': 'int'}},
                'v <= p',
                ['k'],
                parameters={'p': {'dims': ['k'], 'missing': 2.5}},
                variables={'v': {'dims': ['k'], 'bounds': {'lower': 0, 'upper': 5}}},
            ),
            {'k': [1, 2], 'p': pl.DataFrame({'k': [1], 'value': [7]})},
            'p',
        ),
        id='a fractional missing value under an integer column',
    ),
    pytest.param(
        _case(
            _model(
                {**T, **G},
                'v <= shift(p, along=t, offset=1, edge=0)',
                ['g', 't'],
                parameters={'p': {'dims': ['t', 'g'], 'missing': 'absent'}},
                variables={'v': _bounded('g', 't')},
            ),
            {'t': [1, 2], 'g': ['a', 'b'], 'p': pl.DataFrame({'t': [2], 'g': ['a'], 'value': [0.5]})},
            'shift(p, along=t, offset=1, edge=0)',
        ),
        id='a shift edge where the parameter has no row',
    ),
    pytest.param(
        _case(
            _model(
                {**T, **G},
                'shift(x + p, along=t, offset=1, edge=0) <= v',
                ['g', 't'],
                parameters={'p': {'dims': ['t', 'g'], 'missing': 'absent'}},
                variables={'v': _bounded('g', 't'), 'x': _bounded('g', 't')},
            ),
            {'t': [1, 2], 'g': ['a', 'b'], 'p': pl.DataFrame({'t': [2], 'g': ['a'], 'value': [0.5]})},
            'shift(x + p, along=t, offset=1, edge=0)',
        ),
        id='a shift edge beside a variable where the parameter has no row',
    ),
    pytest.param(
        _case(
            _model(
                G,
                'p1 == p0 * v',
                ['g'],
                parameters={'p0': {'dims': ['g']}, 'p1': {'dims': []}},
                variables={'v': {'dims': [], 'bounds': {'lower': -5, 'upper': 10}}, 'w': _bounded('g')},
            ),
            {'g': ['g1'], 'p0': pl.DataFrame({'g': ['g1'], 'value': [0.0]}), 'p1': 2.0},
        ),
        id='a row whose coefficient is a data zero',
    ),
    pytest.param(
        _case(
            _model(
                {**G, 'b': {'dtype': 'str'}, 'k': {'dtype': 'str'}},
                'sum(v, over=k, by=r[b]) == at(p, by=r[b])',
                ['b', 'g', 'k'],
                relations={'r': {'key': 'k', 'values': 'b'}},
                parameters={'p': {'dims': ['g', 'b']}},
                variables={'v': {'dims': ['k'], 'bounds': {'lower': 0, 'upper': 10}}},
            ),
            {
                'g': ['g1'],
                'b': ['b1'],
                'k': ['k1'],
                'p': pl.DataFrame({'g': ['g1'], 'b': ['b1'], 'value': [1.0]}),
                'r': pl.DataFrame({'k': ['k1'], 'b': ['b1']}),
            },
        ),
        id='oracle: a grouped sum beside a lookup of the same column',
    ),
    pytest.param(
        _case(
            _model(
                {**G, 'h': {'dtype': 'str'}},
                'v <= p',
                ['g', 'h'],
                parameters={'p': {'dims': ['g', 'h'], 'missing': 2.5}},
                variables={'v': {'dims': ['g', 'h'], 'bounds': {'lower': -5, 'upper': 10}}},
            ),
            {
                'g': ['g1', 'g2'],
                'h': ['h1', 'h2'],
                'p': pl.DataFrame({'g': ['g1', 'g2'], 'h': ['h1', 'h2'], 'value': [1.0, 2.0]}),
            },
        ),
        id='oracle: a missing value over two dimensions',
    ),
    pytest.param(
        _case(
            _model(
                {**G, 'h': {'dtype': 'str'}},
                'v <= 1',
                ['g', 'h'],
                parameters={'p': {'dims': ['g', 'h'], 'missing': 0.0}},
                variables={'v': _bounded('g', 'h') | {'where': 'p != 1'}},
            ),
            {
                'g': ['g1', 'g2'],
                'h': ['h1', 'h2'],
                'p': pl.DataFrame({'g': ['g1', 'g2'], 'h': ['h1', 'h2'], 'value': [2.0, 3.0]}),
            },
        ),
        id='oracle: a missing value over two dimensions in a where',
    ),
    pytest.param(
        _case(
            {
                'dimensions': T,
                'parameters': {'w': {'dims': [], 'dtype': 'int'}},
                'variables': {'v': _bounded('t')},
                'objective': {'sense': 'maximize', 'expression': 'sum(v)'},
            },
            {'t': [1, 2], 'w': 0},
            'sum_back(v, along=t, window=w)',
        ),
        id='a window parameter of zero',
    ),
    pytest.param(
        _case(
            _model(
                T,
                'sum_back(v, along=t, window=w) + x <= 0.5',
                ['t'],
                parameters={'w': {'dims': [], 'dtype': 'int'}},
                variables={'v': _bounded('t'), 'x': _bounded('t')},
            ),
            {'t': [1, 2], 'w': 0},
        ),
        id='a window of width zero beside another term',
    ),
    pytest.param(
        _case(
            _model(
                T,
                'sum_back(v, along=t, window=w) <= 0.5',
                ['t'],
                parameters={'w': {'dims': [], 'dtype': 'int'}},
                variables={'v': _bounded('t')},
            ),
            {'t': [1, 2], 'w': 0},
        ),
        id='a window of width zero alone in a row',
    ),
    pytest.param(
        _case(
            _model(
                T,
                'v <= sum_back(p, along=t, window=w)',
                ['t'],
                parameters={'w': {'dims': [], 'dtype': 'int'}, 'p': {'dims': ['t']}},
                variables={'v': _bounded('t')},
            ),
            {'t': [1, 2], 'w': 0, 'p': pl.DataFrame({'t': [1, 2], 'value': [1.0, 2.0]})},
            'sum_back(p, along=t, window=w)',
        ),
        id='a window of width zero over data',
    ),
    pytest.param(
        _case(
            _model(
                T,
                'sum_back(p, along=t, window=1) <= x',
                ['t'],
                parameters={'p': {'dims': ['t'], 'missing': 'neutral'}},
                variables={'x': _bounded('t')},
            ),
            {'t': [1, 2], 'p': pl.DataFrame({'t': [2], 'value': [0.25]})},
            'sum_back(p, along=t, window=1)',
        ),
        id='a window over a neutral parameter with no row in it',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'g': {'dtype': 'str'}, 'h': {'dtype': 'int'}, 'k': {'dtype': 'int'}},
                'parameters': {'p': {'dims': ['g', 'h'], 'missing': 'neutral'}},
                'variables': {
                    'x': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 5}, 'where': "g != 'g6'"},
                    'y': {'dims': ['k', 'h'], 'bounds': {'lower': -5, 'upper': 5}},
                },
                'objective': {'sense': 'maximize', 'expression': 'sum(y)'},
            },
            {
                'g': ['g6'],
                'h': [9],
                'k': [2],
                'p': pl.DataFrame(
                    {'g': [], 'h': [], 'value': []}, schema={'g': pl.String, 'h': pl.Int64, 'value': pl.Float64}
                ),
            },
            '(y + x) / p',
        ),
        id='a read divided by a neutral parameter with no row, where the quotient is absent',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'h': {'dtype': 'int'}, 'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
                'relations': {'r': {'key': ['g', 'h'], 'missing': 'absent'}},
                'parameters': {'p': {'dims': ['t', 'g']}},
                'variables': {'v': {'dims': ['h'], 'bounds': {'lower': 0, 'upper': 10}}},
                'constraints': {'c': {'dims': ['g', 't'], 'expression': 'p <= sum(v, over=h, by=r[g])'}},
                'objective': {'sense': 'maximize', 'expression': 'sum(v)'},
            },
            {
                'h': [9],
                'g': ['g7'],
                't': [9],
                'p': pl.DataFrame({'t': [9], 'g': ['g7'], 'value': [1.0]}),
                'r': pl.DataFrame({'g': [], 'h': []}, schema={'g': pl.String, 'h': pl.Int64}),
            },
        ),
        id='oracle: a sum through a bare relation that relates nothing',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'b': {'dtype': 'str'}},
                'parameters': {'p': {'dims': ['b']}},
                'variables': {
                    'v': {'dims': ['b'], 'bounds': {'lower': 0, 'upper': 10}, 'where': "b != 'b4'"},
                },
                'objective': {'sense': 'maximize', 'expression': 'sum(sum(v) / p)'},
            },
            {'b': ['b4', 'b7'], 'p': pl.DataFrame({'b': ['b4', 'b7'], 'value': [0.0, 2.0]})},
        ),
        id='oracle: a zero divisor under a sum in the objective',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'h': {'dtype': 'int'}, 'b': {'dtype': 'str'}, 't': {'dtype': 'int'}},
                'parameters': {'p': {'dims': ['t', 'h'], 'missing': 'neutral'}},
                'variables': {'v': {'dims': ['b'], 'bounds': {'lower': 0, 'upper': 5}}},
                'objective': {'sense': 'maximize', 'expression': 'sum(v)'},
            },
            {
                'h': [0],
                'b': ['b5'],
                't': [-1],
                'p': pl.DataFrame(
                    {'t': [], 'h': [], 'value': []}, schema={'t': pl.Int64, 'h': pl.Int64, 'value': pl.Float64}
                ),
            },
            'v / p - v / p',
        ),
        id='a read refused once per coordinate where two quotients by a missing row reach it',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'h': {'dtype': 'int', 'ordered': True}},
                'parameters': {'p': {'dims': ['h'], 'missing': 'neutral'}},
                'variables': {'v': {'dims': ['h'], 'bounds': {'lower': 0, 'upper': 1}}},
                'objective': {'sense': 'maximize', 'expression': 'sum(v)'},
            },
            {'h': [10, 4], 'p': pl.DataFrame({'h': [10], 'value': [-0.5]})},
            '-v / p + shift(shift(v, along=h, offset=1), along=h, offset=1, edge=0)',
        ),
        id='oracle: a read answered where a quotient by a missing row is absent beside a shift',
    ),
    pytest.param(
        _case(
            _model(
                {'b': {'dtype': 'str'}},
                'sum(v) + v * p <= 1',
                ['b'],
                parameters={'p': {'dims': ['b'], 'missing': 'neutral'}},
                variables={'v': {'dims': ['b'], 'bounds': {'lower': 0, 'upper': 10}, 'where': "b != 'b8'"}},
            ),
            {'b': ['b5', 'b8'], 'p': pl.DataFrame({'b': ['b5'], 'value': [2.0]})},
        ),
        id='oracle: a neutral factor with no row beside a variable absent there',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'b': {'dtype': 'str'}},
                'parameters': {'p': {'dims': ['b'], 'missing': 'neutral'}},
                'variables': {'v': {'dims': ['b'], 'bounds': {'lower': 0, 'upper': 1}}},
                'expressions': {'e': {'dims': ['b'], 'expression': 'v / p'}},
                'objective': {'sense': 'maximize', 'expression': 'sum(v)'},
            },
            {'b': ['b5'], 'p': pl.DataFrame({'b': [], 'value': []}, schema={'b': pl.String, 'value': pl.Float64})},
            'e',
        ),
        id='a declared expression read from a saved answer is refused by its own name',
    ),
    pytest.param(
        _case(
            {
                'dimensions': G,
                'parameters': {'p': {'dims': ['g'], 'dtype': 'int', 'missing': 'neutral'}},
                'variables': {'v': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 1}, 'where': "g != 'g8'"}},
                'expressions': {'e': {'dims': ['g'], 'expression': 'sum(v, over=g) / p'}},
                'constraints': {'c': {'dims': ['g'], 'expression': 'v == e'}},
                'objective': {'sense': 'maximize', 'expression': 'sum(v)'},
            },
            {'g': ['g7', 'g8'], 'p': pl.DataFrame({'g': ['g7'], 'value': [2]})},
        ),
        id='oracle: a divisor with no row where a variable beside the quotient takes the row',
    ),
    pytest.param(
        _case(
            _model(
                G,
                'v / sum(p, over=g) <= 1',
                ['g'],
                parameters={'p': {'dims': ['g'], 'missing': 'neutral'}},
                variables={'v': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 1}, 'where': "g != 'g8'"}},
            ),
            {'g': ['g0', 'g7', 'g8'], 'p': pl.DataFrame({'g': ['g7'], 'value': [2.0]})},
        ),
        id='oracle: a neutral parameter with no row summed inside a divisor',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'t': {'dtype': 'int'}, 'h': {'dtype': 'int', 'ordered': True}},
                'relations': {'r': {'key': 'h', 'values': 't'}},
                'parameters': {'p': {'dims': ['h', 't'], 'missing': 'absent'}},
                'variables': {'v': {'dims': [], 'bounds': {'lower': 0, 'upper': 1}}},
                'objective': {'sense': 'maximize', 'expression': 'v'},
            },
            {
                't': [-2],
                'h': [2],
                'p': pl.DataFrame(
                    {'h': [], 't': [], 'value': []}, schema={'h': pl.Int64, 't': pl.Int64, 'value': pl.Float64}
                ),
                'r': pl.DataFrame({'h': [2], 't': [-2]}),
            },
            'shift(p, along=h, offset=-1, edge=1.5, within=r[t])',
        ),
        id='oracle: a partitioned shift edge over data with no row',
    ),
    pytest.param(
        _case(
            _model(
                {'t': {'dtype': 'int'}},
                '3 / p <= v',
                ['t'],
                parameters={'p': {'dims': ['t'], 'missing': 'neutral'}},
                variables={
                    'v': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 10}, 'where': 't != 9', 'missing': 'neutral'}
                },
            ),
            {'t': [0, 9], 'p': pl.DataFrame({'t': [0], 'value': [2.0]})},
        ),
        id='oracle: a divisor with no row beside a neutral variable masked there',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {**T, **G},
                'parameters': {'w': {'dims': ['g'], 'dtype': 'int'}},
                'variables': {'v': _bounded('g', 't')},
                'objective': {'sense': 'maximize', 'expression': 'sum(v)'},
            },
            {'t': [1, 2], 'g': ['a', 'b'], 'w': pl.DataFrame({'g': ['a', 'b'], 'value': [2, -1]})},
            'sum_back(v, along=t, window=w)',
        ),
        id='a negative window parameter',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'k': {'dtype': 'int'}},
                'parameters': {'p0': {'dims': ['k']}},
                'variables': {
                    'v0': {'dims': ['k'], 'bounds': {'lower': -5, 'upper': 10}, 'where': 'p0 > 1'},
                    'v1': {'dims': ['k'], 'bounds': {'lower': -5, 'upper': 5}},
                },
                'constraints': {'c': {'dims': ['k'], 'expression': 'sum(v0) / p0 <= v1'}},
                'objective': {'sense': 'minimize', 'expression': 'sum(v1)'},
            },
            {'k': [11], 'p0': pl.DataFrame({'k': [11], 'value': [0.0]})},
        ),
        id='oracle: an empty sum over a zero divisor',
    ),
    pytest.param(
        _case(
            _model(
                {'h': {'dtype': 'int', 'ordered': True}, 't': {'dtype': 'int'}},
                'v <= 1',
                ['t'],
                parameters={'p': {'dims': ['h']}},
                variables={'v': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 5}}},
            ),
            {'h': [5, 6], 't': [1], 'p': pl.DataFrame({'h': [5, 6], 'value': [1.0, 2.0]})},
            'shift(v / p, along=h, offset=1, edge=0)',
        ),
        id='a quotient under a shift with an edge',
    ),
    pytest.param(
        _case(
            _model(
                {'t': {'dtype': 'int'}, 'b': {'dtype': 'str'}},
                'sum(v + p, over=t, by=r[b]) <= 1.5',
                ['b'],
                relations={'r': {'key': 't', 'values': 'b'}},
                parameters={'p': {'dims': ['t']}},
                variables={'v': {'dims': [], 'bounds': {'lower': 0, 'upper': 10}}},
            ),
            {
                't': [2],
                'b': ['b3'],
                'p': pl.DataFrame({'t': [2], 'value': [1.0]}),
                'r': pl.DataFrame({'t': [2], 'b': ['b3']}),
            },
        ),
        id='sum by over a term without the dimension',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'b': {'dtype': 'str'}, 'g': {'dtype': 'str'}},
                'variables': {
                    'x': {'dims': ['b', 'g'], 'bounds': {'lower': 0, 'upper': 5}, 'where': "g != 'g6'"},
                    'y': {'dims': ['g'], 'bounds': {'lower': -5, 'upper': 10}},
                },
                'objective': {'sense': 'minimize', 'expression': 'sum(x - y)'},
            },
            {'b': ['b8'], 'g': ['g6', 'g7']},
            'x - y',
        ),
        id='an absent summand beside a broadcast term',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'h': {'dtype': 'int'}},
                'parameters': {'p': {'dims': ['h'], 'missing': 'neutral'}},
                'variables': {'v': {'dims': [], 'bounds': {'lower': -5, 'upper': 10}}},
                'constraints': {'c': {'dims': ['h'], 'expression': 'v / p <= 1'}},
                'objective': {'sense': 'maximize', 'expression': 'v'},
            },
            {'h': [2], 'p': pl.DataFrame({'h': [], 'value': []}, schema={'h': pl.Int64, 'value': pl.Float64})},
        ),
        id='a neutral divisor beside a scalar dividend',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'k': {'dtype': 'int'}, 'g': {'dtype': 'str'}},
                'parameters': {'p': {'dims': ['g'], 'missing': 'neutral'}},
                'variables': {'v': {'dims': ['k'], 'bounds': {'lower': 0, 'upper': 5}}},
                'constraints': {'c': {'dims': ['g', 'k'], 'expression': '0.5 <= p * (3 + v)'}},
                'objective': {'sense': 'maximize', 'expression': 'sum(v)'},
            },
            {
                'k': [1],
                'g': ['a'],
                'p': pl.DataFrame({'g': [], 'value': []}, schema={'g': pl.String, 'value': pl.Float64}),
            },
        ),
        id='a neutral factor on a side with a variable',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'h': {'dtype': 'int'}, 't': {'dtype': 'int'}},
                'parameters': {'a': {'dims': ['t', 'h']}, 'b': {'dims': ['h', 't'], 'dtype': 'int'}},
                'variables': {'v': {'dims': [], 'bounds': {'lower': -5, 'upper': 10}}},
                'expressions': {
                    'e': {
                        'dims': ['h', 't'],
                        'cases': {'one': {'when': 'a != 1', 'expression': 'sum(a, over=[h, t])'}},
                        'otherwise': 'b',
                    }
                },
                'objective': {'sense': 'maximize', 'expression': 'sum(v / e)'},
            },
            {
                'h': [7, 5],
                't': [1, 2],
                'a': pl.DataFrame({'t': [1, 1, 2, 2], 'h': [7, 5, 7, 5], 'value': [2.0, 1.0, 3.0, -2.0]}),
                'b': pl.DataFrame({'h': [7, 7, 5, 5], 't': [1, 2, 1, 2], 'value': [1, 2, 3, 4]}),
            },
        ),
        id='a cased divisor',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'h': {'dtype': 'int'}},
                'parameters': {'p': {'dims': ['h']}},
                'variables': {'v': _bounded('h')},
                'objective': {'sense': 'maximize', 'expression': 'sum(v / p)'},
            },
            {'h': [1], 'p': pl.DataFrame({'h': [1], 'value': [0.0]})},
        ),
        id='the words for a zero divisor in the objective',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'b': {'dtype': 'str'}},
                'parameters': {'p': {'dims': ['b'], 'missing': 'absent'}},
                'variables': {'v': {'dims': [], 'bounds': {'lower': -5, 'upper': 5}}},
                'objective': {'sense': 'maximize', 'expression': 'v'},
            },
            {'b': ['b1'], 'p': pl.DataFrame({'b': [], 'value': []}, schema={'b': pl.String, 'value': pl.Float64})},
            'p ** 0',
        ),
        id='oracle: an absent base to the power 0',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'h': {'dtype': 'int'}},
                'parameters': {'p': {'dims': ['h'], 'missing': 'neutral'}},
                'variables': {'v': _bounded('h')},
                'objective': {'sense': 'maximize', 'expression': 'sum(v)'},
            },
            {'h': [1, 2], 'p': pl.DataFrame({'h': [1], 'value': [2.0]})},
            'v / p',
        ),
        id='oracle: a read that divides by a neutral parameter with no row',
    ),
    pytest.param(
        _case(
            _model(
                {'b': {'dtype': 'str'}, **G},
                'v <= 1',
                ['b', 'g'],
                relations={'r': {'key': ['g', 'b'], 'missing': 'absent'}},
                variables={'v': _bounded('b', 'g')},
            )
            | {'constraints': {'c': {'dims': ['b', 'g'], 'expression': 'v <= 1', 'where': 'r'}}},
            {'b': ['b0'], 'g': ['g0'], 'r': pl.DataFrame({'g': ['g0'], 'b': ['b0']})},
        ),
        id='oracle: a where that reads a bare relation',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'h': {'dtype': 'int'}},
                'parameters': {'a': {'dims': ['h']}, 'd': {'dims': ['h']}},
                'variables': {'v': _bounded('h')},
                'constraints': {'c': {'dims': ['h'], 'expression': 'a / d >= v', 'where': 'd > 0'}},
                'objective': {'sense': 'maximize', 'expression': 'sum(v)'},
            },
            {
                'h': [10, 6],
                'a': pl.DataFrame({'h': [10, 6], 'value': [-0.5, 1.0]}),
                'd': pl.DataFrame({'h': [10, 6], 'value': [0.0, 2.0]}),
            },
        ),
        id='oracle: a zero divisor the where excludes',
    ),
    pytest.param(
        _case(
            {
                'dimensions': T,
                'parameters': {'p': {'dims': ['t']}},
                'variables': {'v': _bounded('t')},
                'constraints': {'c': {'dims': ['t'], 'expression': 'shift(v / p, along=t, offset=1) >= v'}},
                'objective': {'sense': 'maximize', 'expression': 'sum(v)'},
            },
            {'t': [1, 2], 'p': pl.DataFrame({'t': [1, 2], 'value': [2.0, 0.0]})},
        ),
        id='oracle: a zero divisor a shift vacates',
    ),
    pytest.param(
        _case(
            _model(
                {'k': {'dtype': 'int'}, **G},
                'p >= v',
                ['g', 'k'],
                relations={'r': {'key': 'g', 'values': 'k'}},
                parameters={'p': {'dims': ['k', 'g']}, 'f': {'dims': ['k'], 'dtype': 'bool'}},
                variables={'v': {'dims': ['k'], 'bounds': {'lower': -5, 'upper': 10}}},
            )
            | {'constraints': {'c': {'dims': ['g', 'k'], 'expression': 'p >= v', 'where': 'at(f, by=r[k])'}}},
            {
                'k': [6],
                'g': ['g5'],
                'p': pl.DataFrame({'k': [6], 'g': ['g5'], 'value': [0.0]}),
                'f': pl.DataFrame({'k': [6], 'value': [True]}),
                'r': pl.DataFrame({'g': ['g5'], 'k': [6]}),
            },
        ),
        id='oracle: an at() in a where over a frame with its column',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'k': {'dtype': 'int'}},
                'parameters': {'f': {'dims': ['k'], 'dtype': 'bool', 'missing': 'neutral'}},
                'variables': {'v': {'dims': ['k'], 'bounds': {'lower': -5, 'upper': 10}}},
                'expressions': {
                    'e': {
                        'dims': ['k'],
                        'cases': {'one': {'when': 'count(f, over=k) == 1', 'expression': '8'}},
                        'otherwise': 'v / 2',
                    }
                },
                'constraints': {'c': {'dims': ['k'], 'expression': 'e <= 3'}},
                'objective': {'sense': 'maximize', 'expression': 'sum(v)'},
            },
            {'k': [7], 'f': pl.DataFrame({'k': [7], 'value': [False]})},
            'e',
        ),
        id='a counted case beside a quotient',
    ),
    pytest.param(
        _case(
            {
                'dimensions': G,
                'parameters': {'lo': {'dims': []}},
                'variables': {'v': {'dims': ['g'], 'bounds': {'lower': 'lo', 'upper': 10}, 'domain': 'integer'}},
                'constraints': {'c': {'dims': ['g'], 'expression': 'v <= 1.5'}},
                'objective': {'sense': 'minimize', 'expression': 'sum(v)'},
            },
            {'g': ['g3'], 'lo': -0.5},
        ),
        id='an integer column with a fractional lower bound',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'k': {'dtype': 'int'}},
                'variables': {
                    'v': {'dims': ['k'], 'bounds': {'lower': -5, 'upper': 5}},
                    'w': {'dims': [], 'bounds': {'lower': 1, 'upper': 1}},
                },
                'expressions': {
                    'e': {'dims': ['k'], 'cases': {'one': {'when': 'v', 'expression': 'w'}}, 'otherwise': 'v'}
                },
                'objective': {'sense': 'maximize', 'expression': 'sum(-v * e)'},
            },
            {'k': [-2]},
            'e',
            'sum(-v * e)',
            linear=False,
        ),
        id='oracle: a product with a case over a variable',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'h': {'dtype': 'int', 'ordered': True}, 't': {'dtype': 'int'}},
                'relations': {'r': {'key': 'h', 'values': 't'}},
                'variables': {'v': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 5}}},
                'objective': {'sense': 'maximize', 'expression': 'sum(sum_back(at(v, by=r[t]), along=h, window=2))'},
            },
            {'h': [-1, 1], 't': [5, 1], 'r': pl.DataFrame({'h': [-1, 1], 't': [1, 5]})},
        ),
        id='oracle: a sum_back over a lookup',
    ),
    pytest.param(
        _case(
            {
                'dimensions': {'k': {'dtype': 'int'}, 'b': {'dtype': 'str'}, 'h': {'dtype': 'int', 'ordered': True}},
                'parameters': {'o': {'dims': ['b'], 'dtype': 'int'}, 'q': {'dims': ['k', 'h']}},
                'variables': {
                    'v': {'dims': ['b', 'k'], 'bounds': {'lower': 0, 'upper': 10}},
                    'x': {'dims': ['b', 'h', 'k'], 'bounds': {'lower': 0, 'upper': 1}},
                },
                'constraints': {
                    'c': {'dims': ['b', 'h', 'k'], 'expression': 'shift(q + x, along=h, offset=o, edge=0) <= v'}
                },
                'objective': {'sense': 'maximize', 'expression': 'sum(v)'},
            },
            {
                'k': [11],
                'b': ['b0'],
                'h': [9, 10],
                'o': pl.DataFrame({'b': ['b0'], 'value': [1]}),
                'q': pl.DataFrame({'k': [11, 11], 'h': [9, 10], 'value': [1.0, 2.0]}),
            },
        ),
        id='a parameter offset over a term without its dimension',
    ),
]


@pytest.mark.parametrize('case', REDUCED)
def test_a_reduced_case_passes_every_oracle(case: Case) -> None:
    findings = check(case).findings
    assert not findings, '\n'.join(f'[{f.oracle}] {f.summary}\n{f.detail[-800:]}' for f in findings)
