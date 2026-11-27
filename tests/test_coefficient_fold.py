"""A variable times a sum of parameters is the model the distributed product writes, built once.

`x * (a - b)` pairs `x` with each of `a` and `-b`, and the matrix summed the two
copies back. The query builder adds the parameters up first where that sum is the
same coefficient, so a piecewise chord row costs one term per variable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

import specsolve as sps
from specsolve._engine import assembly, queries
from specsolve.errors import DataError

if TYPE_CHECKING:
    from pathlib import Path

SPEC = {
    'dimensions': {'i': {'dtype': 'int', 'ordered': True}, 'j': {}},
    'parameters': {
        'a': {'dims': ['i']},
        'b': {'dims': ['i']},
        'c': {'dims': ['j']},
        'd': {'dims': ['i'], 'missing': 'absent'},
        'e': {'dims': ['i'], 'missing': 0},
        'flag': {'dims': ['i'], 'dtype': 'bool', 'missing': False},
        's': {'dims': []},
        'r': {'dims': []},
    },
    'variables': {
        'x': {'dims': ['i'], 'bounds': {'lower': 0, 'upper': 10}},
        'y': {'dims': ['i', 'j'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'expressions': {
        'capped': {'dims': ['i'], 'cases': {'flagged': {'when': 'flag', 'expression': 'a'}}, 'otherwise': 5}
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(x) + sum(y)'},
}

SOURCES = {
    'i': [0, 1, 2],
    'j': ['p', 'q'],
    'a': pl.DataFrame({'i': [0, 1, 2], 'value': [3.0, 4.0, 5.0]}),
    'b': pl.DataFrame({'i': [0, 1, 2], 'value': [1.0, 4.0, 2.0]}),
    'c': pl.DataFrame({'j': ['p', 'q'], 'value': [0.5, 1.5]}),
    'd': pl.DataFrame({'i': [0, 2], 'value': [7.0, 8.0]}),
    'e': pl.DataFrame({'i': [0, 2], 'value': [2.0, 4.0]}),
    'flag': pl.DataFrame({'i': [0, 1, 2], 'value': [True, False, True]}),
    's': 2.0,
    'r': 3.0,
}


def _written(constraint: dict, path: Path) -> str:
    sps.write({**SPEC, 'constraints': {'row': constraint}}, SOURCES, path)
    return path.read_text()


@pytest.mark.parametrize(
    'constraint',
    [
        pytest.param({'dims': ['i'], 'expression': 'x * (a - b) <= 10'}, id='a-difference-over-one-set-of-dimensions'),
        pytest.param({'dims': ['i', 'j'], 'expression': 'y * (a + c) <= 10'}, id='terms-over-different-dimensions'),
        pytest.param({'dims': ['i'], 'expression': 'x * (a + d) <= 10'}, id='a-term-absent-at-a-coordinate'),
        pytest.param({'dims': ['i'], 'expression': 'x * (capped + b) <= 10'}, id='terms-from-regions-of-cases'),
        pytest.param({'dims': ['i'], 'expression': 'x * (s + r) <= 10'}, id='scalars'),
        pytest.param({'dims': ['i'], 'expression': '(a - b) * x <= 10'}, id='the-sum-on-the-left'),
        pytest.param({'dims': ['i'], 'expression': 'x * (a + b / (1 + e)) <= 10'}, id='a-divisor-over-a-missing-row'),
    ],
)
def test_a_folded_coefficient_writes_the_model_the_distributed_product_writes(
    monkeypatch, tmp_path: Path, constraint: dict
) -> None:
    folded = _written(constraint, tmp_path / 'folded.lp')
    monkeypatch.setattr(queries, '_one_coefficient', lambda terms: terms)
    distributed = _written(constraint, tmp_path / 'distributed.lp')
    assert folded == distributed, 'adding the parameters up first changed the written model'


def test_a_variable_times_a_difference_stacks_one_term_per_cell(monkeypatch) -> None:
    """Distributed, the row stacked one table per parameter and summed the copies at assembly."""
    stacked = []
    share = assembly.Assembly._matrix_share

    def spied(self, tables, name, *expressions):
        if name == "constraint 'row'":
            stacked.append(len(tables))
        return share(self, tables, name, *expressions)

    monkeypatch.setattr(assembly.Assembly, '_matrix_share', spied)
    spec = {**SPEC, 'constraints': {'row': {'dims': ['i'], 'expression': 'x * (a - b) <= 10'}}}
    sps.build(spec, SOURCES).close()
    assert stacked == [1], f'x * (a - b) stacked {stacked} tables, not one'


def test_a_divisor_over_a_missing_row_under_a_folded_coefficient_reads_the_declared_zero() -> None:
    """`e` is `missing: 0` with no row at 1, so `b / (1 + e)` is 4 / 1 there and the row reads `x * (4 + 4) <= 10`."""
    spec = {**SPEC, 'constraints': {'row': {'dims': ['i'], 'expression': 'x * (a + b / (1 + e)) <= 10'}}}
    with sps.solve(spec, SOURCES) as solution:
        x = dict(solution.primal('x').select('i', 'value').iter_rows())
    assert x[1] == pytest.approx(10 / 8), 'the folded coefficient at 1 is a + b / 1, not a alone'


def _outcome(spec: dict, path: Path) -> str:
    """The written model, or the class and first line of the refusal."""
    try:
        sps.write(spec, SOURCES, path)
    except DataError as error:
        return f'{type(error).__name__}: {str(error).splitlines()[0]}'
    return path.read_text()


@pytest.mark.parametrize(
    ('expressions', 'constraint'),
    [
        pytest.param(
            {},
            {'dims': ['i'], 'expression': '(x + s) * (a + e) <= 10'},
            id='a-constant-beside-the-variable-reads-each-term',
        ),
        pytest.param(
            {'k': {'dims': ['i'], 'cases': {'flagged': {'when': 'flag', 'expression': 'a + b'}}, 'otherwise': 0}},
            {'dims': ['i'], 'expression': 'x + (x + s) * k <= 10'},
            id='a-constant-beside-the-variable-keeps-each-region',
        ),
    ],
)
def test_a_factor_with_a_constant_beside_its_variable_builds_as_distributed(
    monkeypatch, tmp_path: Path, expressions: dict, constraint: dict
) -> None:
    """Folding beside a constant keeps each constant product, and each region's claim, on its own."""
    spec = {**SPEC, 'expressions': {**SPEC['expressions'], **expressions}, 'constraints': {'row': constraint}}
    folded = _outcome(spec, tmp_path / 'folded.lp')
    monkeypatch.setattr(queries, '_one_coefficient', lambda terms: terms)
    distributed = _outcome(spec, tmp_path / 'distributed.lp')
    assert folded == distributed, 'folding beside a constant changed what the build does'


def test_a_quadratic_times_a_folded_sum_writes_the_model_the_distributed_product_writes(
    monkeypatch, tmp_path: Path
) -> None:
    """`x * x` meets `a - b` as the linear terms do, added up once."""
    spec = {
        **SPEC,
        'constraints': {'row': {'dims': ['i'], 'expression': 'x >= 1'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x * x * (a - b))'},
    }
    sps.write(spec, SOURCES, tmp_path / 'folded.lp')
    monkeypatch.setattr(queries, '_one_coefficient', lambda terms: terms)
    sps.write(spec, SOURCES, tmp_path / 'distributed.lp')
    folded, distributed = ((tmp_path / f'{n}.lp').read_text() for n in ('folded', 'distributed'))
    assert folded == distributed, 'adding the parameters up first changed the quadratic objective'
