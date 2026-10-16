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


def _case(spec: dict[str, Any], sources: dict[str, Any], *reads: str) -> Case:
    return Case(seed=0, spec=spec, sources=sources, reads=list(reads))


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
        marks=_found(
            '#2049 — a shift of a shift over a variable with a second dimension crashes polars: schema lengths differ'
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
        marks=_found(
            '#2049 — a shift over a sum with a term that does not carry the dimension raises an internal AssertionError'
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
        marks=_found(
            '#2049 — a sum_back over a sum with a term that does not carry the dimension raises an internal AssertionError'
        ),
        id='sum_back over a term without the dimension',
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
        marks=_found(
            '#2049 — an at() over a sum with a term that does not carry the looked-up column raises an internal AssertionError'
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
        marks=_found(
            '#2048 — a float parameter supplied as an integer column reads `missing: 2.5` as 2 on the relational lane'
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
        marks=_found(
            '#2048 — a numeric shift edge over a parameter stands only where the parameter has a row in the other dimensions'
        ),
        id='a shift edge where the parameter has no row',
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
        marks=_found(
            '#2048 — a row whose only coefficient is a data zero: the relational lane keeps `2 == 0`, the oracle drops it'
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
        marks=_found(
            '#2049 — the oracle crashes on a sum grouped by a column that an at() on the other side also adds'
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
        marks=_found(
            '#2049 — the oracle reads `missing: <value>` on a parameter over two dimensions as `neutral`, and refuses the data'
        ),
        id='oracle: a missing value over two dimensions',
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
        marks=_found(
            '#2048 — a window parameter of 0 is accepted, which a literal window is not, and the lanes read it apart'
        ),
        id='a window parameter of zero',
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
        marks=_found('#2048 — the oracle drops the whole row, v1 with it, when a sum with no terms is divided by zero'),
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
        marks=_found('#2049 — a quotient read under a shift with an edge crashes polars: schema lengths differ'),
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
        marks=_found(
            '#2049 — a sum through a relation over a sum with a term that does not carry the dimension raises an internal AssertionError'
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
        marks=_found(
            '#2048 — a summand is absent where its masked operand is, but the relational lane keeps the term it broadcasts beside it'
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
        marks=_found(
            '#2048 — a neutral divisor with no row is not refused on the relational lane when the dividend lacks its dimension'
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
        marks=_found(
            "#2049 — a neutral factor on a side with a variable is refused as a constant side, and the message names parameter ''"
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
        marks=_found(
            '#2049 — a cased divisor with a value at every coordinate is refused as short of rows on the relational lane'
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
        marks=_found(
            '#2049 — a zero divisor in the objective is refused as `objective:` on one lane and `the objective:` on the other'
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
        marks=_found('#2048 — the oracle reads an absent parameter to the power 0 as 1'),
        id='oracle: an absent base to the power 0',
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
        marks=_found('#1897 — the oracle crashes on a where that reads a bare relation: reduce() of empty iterable'),
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
        marks=_found(
            '#2049 — the oracle divides by a zero that the where already excludes, and crashes on the infinity'
        ),
        id='oracle: a zero divisor the where excludes',
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
        marks=_found('#2049 — the oracle crashes on an at() in a where whose frame also carries the column it reads'),
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
        marks=_found(
            '#2049 — a case whose when: counts, beside a quotient in otherwise:, crashes polars when it is read: no column __present__'
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
        marks=_found(
            '#2048 — an integer column with a lower bound of -0.5 solves optimal with no values here, and to -0.5 on the oracle'
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
        ),
        marks=_found(
            '#2049 — the oracle crashes on a product with a case whose when: reads a variable: unsupported operand for *'
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
        marks=_found(
            '#2049 — the oracle crashes on a sum_back over an at(): an auxiliary coordinate with conflicting values'
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
        marks=_found(
            '#2049 — a shift by a parameter offset over a term that lacks the offset dimension crashes polars: column not found'
        ),
        id='a parameter offset over a term without its dimension',
    ),
]


@pytest.mark.parametrize('case', REDUCED)
def test_a_reduced_case_passes_every_oracle(case: Case) -> None:
    findings = check(case).findings
    assert not findings, '\n'.join(f'[{f.oracle}] {f.summary}\n{f.detail[-800:]}' for f in findings)
