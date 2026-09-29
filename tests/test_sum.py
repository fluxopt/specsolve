"""``sum()`` naming no dim: every dim the operand carries.

The spelling writes down a reduction a declaration would otherwise imply
(#1046). ``sum(x)`` and the nest that names each dim build one model, on both
lanes and through the LP file. The reduction sits in a **scalar constraint**,
since an objective sums what is left over anyway and hides a bare sum.
"""

from __future__ import annotations

import polars as pl
import pytest

import specsolve as sps
from tests.differential import differential

SNAPSHOTS = [0, 1]
GENERATORS = ['wind', 'gas']

#: Bare and nested say the same reduction: the nest names the two dims the
#: operand carries, in either order, and the bare form takes both unnamed.
SPELLINGS = {
    'bare': 'sum(p * cost)',
    'nested': 'sum(sum(p * cost, over=generator), over=snapshot)',
    'nested-the-other-way': 'sum(sum(p * cost, over=snapshot), over=generator)',
}


def _spec(budget: str, *, where: str | None = None) -> dict:
    variable: dict = {'dims': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}}
    if where is not None:
        variable['where'] = where
    return {
        'dimensions': {'snapshot': {'dtype': 'int'}, 'generator': {'dtype': 'str'}},
        'parameters': {
            'p_max': {'dims': ['generator']},
            'cost': {'dims': ['generator']},
            'cap': {'dims': []},
        },
        'variables': {'p': variable},
        'constraints': {'budget': {'dims': [], 'expression': f'{budget} <= cap'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(p)'},
    }


def _sources(p_max: list[float]) -> dict[str, pl.DataFrame]:
    return {
        'snapshot': pl.DataFrame({'snapshot': SNAPSHOTS}),
        'generator': GENERATORS,
        'p_max': pl.DataFrame({'generator': GENERATORS, 'value': p_max}),
        'cost': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 4.0]}),
        'cap': pl.DataFrame({'value': [60.0]}),
    }


@pytest.mark.parametrize('spelling', sorted(SPELLINGS), ids=sorted(SPELLINGS))
def test_a_bare_sum_builds_what_the_nest_builds(spelling, tmp_path):
    """The three spellings reach one optimum, and write one LP file.

    One budget of 60 buys wind at 1 up to its 20 per snapshot — 40 spent, 40
    generated — and what is left buys 5 of gas at 4. A budget row per
    coordinate instead of one for the model would let every generator spend the
    whole 60 and reach 70.
    """
    spec = _spec(SPELLINGS[spelling])
    sources = _sources([20.0, 20.0])

    with differential(spec, sources, lp=True) as run:
        assert run.result.objective == pytest.approx(45.0), 'the budget binds once for the model, not once per row'

    written = tmp_path / f'{spelling}.lp'
    sps.write(spec, sources, written)
    reference = tmp_path / 'reference.lp'
    sps.write(_spec(SPELLINGS['nested']), sources, reference)
    assert written.read_text() == reference.read_text(), 'a bare sum wrote a different model than the nest'


def test_a_bare_sum_skips_the_slots_its_operand_does_not_reach():
    """A reduction reads absent slots as skipped, and the bare form is a reduction.

    Gas is masked out of existence, so the budget row is wind's two summands
    and nothing standing in for a variable that is not there: 40 of wind at
    cost 1, which the budget of 60 does not bind.
    """
    spec = _spec(SPELLINGS['bare'], where='p_max > 0')
    with differential(spec, _sources([20.0, 0.0])) as run:
        assert run.result.objective == pytest.approx(40.0)
