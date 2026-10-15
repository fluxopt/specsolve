"""What a missing row means, as each parameter or relation declares it with ``missing:``.

One small model, two units `a` and `b`, each `x` in ``[0, 10]`` and the
objective their sum, so every reading moves the optimum by a number worked out
by hand. The data always gives `a` and leaves `b` out.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tests.conftest import override
from tests.differential import RTOL, both_lanes_refuse, differential

BASE: dict[str, Any] = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'cap': {'dims': ['f']}},
    'variables': {'x': {'dims': ['f'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'c': {'dims': ['f'], 'expression': 'x <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}


def _only_a(value: float | bool) -> pd.Series:
    """A table over `f` that has a row at `a` and none at `b`."""
    return pd.Series([value], index=pd.Index(['a'], name='f'))


def test_a_parameter_short_of_a_row_is_refused_by_default_and_the_coordinate_is_named():
    """With no `missing:`, every coordinate needs a row, so the gap is refused at attach on both lanes."""
    message = both_lanes_refuse(BASE, {'f': ['a', 'b'], 'cap': _only_a(4.0)}, match="parameter 'cap' has no row at f=b")
    assert 'missing: absent' in message, 'the refusal names the declarations that would accept the gap'


def test_a_relation_short_of_a_key_is_refused_by_default_and_the_key_is_named():
    """A map with no `missing:` has a row for every key, so a unit on no bus is refused on both lanes."""
    spec = override(
        BASE,
        **{
            'dimensions.bus': {'dtype': 'str'},
            'relations.on_bus': {'key': 'f', 'values': 'bus'},
            'parameters': {'load': {'dims': ['bus']}},
            'constraints.c': {'dims': ['bus'], 'expression': 'sum(x, over=f, by=on_bus[bus]) <= load'},
        },
    )
    sources = {
        'f': ['a', 'b'],
        'bus': ['n'],
        'on_bus': pd.DataFrame({'f': ['a'], 'bus': ['n']}),
        'load': pd.Series([4.0], index=pd.Index(['n'], name='bus')),
    }
    both_lanes_refuse(spec, sources, match="relation 'on_bus' has no row at f=b")


@pytest.mark.parametrize(
    ('patch', 'value', 'objective'),
    [
        pytest.param({}, 4.0, 14.0, id='absent-constant-side-drops-the-row'),
        pytest.param({'constraints.c.expression': 'x <= 12 / cap'}, 3.0, 14.0, id='absent-divisor-drops-the-row'),
        pytest.param(
            {
                'constraints.c': {'dims': [], 'expression': 'sum(cap * x, over=f) <= 6'},
            },
            2.0,
            13.0,
            id='absent-coefficient-is-one-summand-fewer',
        ),
        pytest.param(
            {
                'variables.x.bounds': {'lower': 0, 'upper': 'cap'},
                'constraints.c.expression': 'x <= 7',
            },
            4.0,
            11.0,
            id='absent-bound-leaves-that-side-open',
        ),
    ],
)
def test_an_absent_row_is_absence_in_every_position(patch, value, objective):
    """Under `missing: absent`, `b`'s missing row is absence, whatever position reads it.

    `a` meets its row in each case; `b` loses the row, the summand or the bound,
    and runs to whatever is left.
    """
    spec = override(BASE, **{'parameters.cap.missing': 'absent', **patch})
    with differential(spec, {'f': ['a', 'b'], 'cap': _only_a(value)}) as run:
        assert run.oracle == pytest.approx(objective, rel=RTOL), f'b gives up its row there, so {objective}'


@pytest.mark.parametrize(
    ('patch', 'objective'),
    [
        pytest.param({'parameters.cap.missing': 2.5}, 6.5, id='a-value-on-a-constant-side'),
        pytest.param(
            {
                'parameters.cap.missing': '.inf',
                'variables.x.bounds': {'lower': 0, 'upper': 'cap'},
                'constraints.c.expression': 'x <= 7',
            },
            11.0,
            id='an-infinite-bound-is-open',
        ),
        pytest.param({'parameters.cap.missing': 5.0, 'variables.x.where': 'cap > 3'}, 9.0, id='a-comparison-reads-it'),
        pytest.param({'parameters.cap.missing': 5.0, 'variables.x.where': 'cap'}, 4.0, id='a-bare-name-asks-for-a-row'),
    ],
)
def test_a_missing_row_reads_as_the_declared_value_wherever_a_value_is_read(patch, objective):
    """A declared value stands in for `b`'s row, but a bare name in a `where` still asks whether the data has one."""
    raw = override(BASE, **patch)
    if raw['parameters']['cap']['missing'] == '.inf':
        raw['parameters']['cap']['missing'] = float('inf')
    with differential(raw, {'f': ['a', 'b'], 'cap': _only_a(4.0)}) as run:
        assert run.oracle == pytest.approx(objective, rel=RTOL), f'b reads the value, so {objective}'


def test_a_bool_that_declares_true_masks_in_where_the_data_has_no_row():
    """A bare `bool` name reads its value, so `missing: true` keeps `b` and the data's `false` drops `a`."""
    spec = override(
        BASE,
        **{
            'parameters.cap': {'dims': ['f'], 'dtype': 'bool', 'missing': True},
            'variables.x.where': 'cap',
            'constraints.c.expression': 'x <= 7',
        },
    )
    with differential(spec, {'f': ['a', 'b'], 'cap': _only_a(False)}) as run:
        assert run.oracle == pytest.approx(7.0, rel=RTOL), 'only b is built, and it stops at 7'
