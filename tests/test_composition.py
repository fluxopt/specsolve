"""A spec composed from fragments builds once it is whole, and a fragment on its own is refused by name.

The fragments are the three files of mathspec's composition how-to: a network
that balances `Bus_injection` and minimises `total_cost`, and two components
that each add their term to those sums.
"""

from __future__ import annotations

import mathspec
import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import LanguageError

NETWORK = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'bus': {'dtype': 'str'}},
    'given': {'expressions': {'Bus_injection': {'dims': ['snapshot', 'bus']}, 'total_cost': {'dims': []}}},
    'constraints': {'Bus_balance': {'dims': ['snapshot', 'bus'], 'expression': 'Bus_injection == 0'}},
    'objective': {'sense': 'minimize', 'expression': 'total_cost'},
}

GENERATOR = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'bus': {'dtype': 'str'}, 'generator': {'dtype': 'str'}},
    'relations': {'Generator_bus': {'key': 'generator', 'values': 'bus'}},
    'parameters': {'Generator_p_nom': {'dims': ['generator']}, 'Generator_marginal_cost': {'dims': ['generator']}},
    'variables': {
        'Generator_p': {'dims': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'Generator_p_nom'}},
    },
    'given': {'expressions': {'Bus_injection': {'dims': ['snapshot', 'bus']}, 'total_cost': {'dims': []}}},
    'expressions': {
        'Generator_injection': {
            'expression': 'sum(Generator_p, by=Generator_bus, over=generator, into=bus)',
            'adds_to': 'Bus_injection',
        },
        'Generator_cost': {'expression': 'sum(Generator_p * Generator_marginal_cost)', 'adds_to': 'total_cost'},
    },
}

LOAD = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'bus': {'dtype': 'str'}, 'load': {'dtype': 'str'}},
    'relations': {'Load_bus': {'key': 'load', 'values': 'bus'}},
    'parameters': {'Load_p_set': {'dims': ['snapshot', 'load']}},
    'given': {'expressions': {'Bus_injection': {'dims': ['snapshot', 'bus']}}},
    'expressions': {
        'Load_injection': {
            'expression': '-sum(Load_p_set, by=Load_bus, over=load, into=bus)',
            'adds_to': 'Bus_injection',
        },
    },
}

SOURCES = {
    'snapshot': [0, 1],
    'bus': ['n'],
    'generator': ['cheap', 'dear'],
    'load': ['town'],
    'Generator_bus': pl.DataFrame({'generator': ['cheap', 'dear'], 'bus': ['n', 'n']}),
    'Load_bus': pl.DataFrame({'load': ['town'], 'bus': ['n']}),
    'Generator_p_nom': {'cheap': 10.0, 'dear': 10.0},
    'Generator_marginal_cost': {'cheap': 1.0, 'dear': 2.0},
    'Load_p_set': pl.DataFrame({'snapshot': [0, 1], 'load': ['town', 'town'], 'value': [15.0, 15.0]}),
}


def test_a_merged_spec_solves_to_the_cost_a_hand_computes():
    """Each snapshot takes all 10 of `cheap` at 1 and 5 of `dear` at 2: 20 a snapshot, over two."""
    result = sps.solve(mathspec.merge([NETWORK, GENERATOR, LOAD]), SOURCES)
    assert result.objective == pytest.approx(40.0), 'the terms each file adds reach the balance and the cost'


@pytest.mark.parametrize(
    ('fragment', 'reads'),
    [
        pytest.param(NETWORK, "'Bus_injection', 'total_cost'", id='a-network-that-reads-its-sums'),
        pytest.param(LOAD, "'Bus_injection'", id='a-component-that-adds-a-term'),
        pytest.param(
            {
                'dimensions': {'snapshot': {'dtype': 'int'}},
                'parameters': {'load': {'dims': ['snapshot']}},
                'given': {'variables': {'supply': {'dims': ['snapshot']}}},
                'constraints': {'meet': {'dims': ['snapshot'], 'expression': 'supply >= load'}},
            },
            "'supply'",
            id='a-variable-another-file-declares',
        ),
    ],
)
def test_a_fragment_on_its_own_is_refused_naming_what_it_reads(fragment, reads):
    """A name under `given:` is declared by a file this spec was not merged with, so there is nothing to build it from."""
    with pytest.raises(LanguageError, match=f'reads {reads} under given:'):
        sps.check(fragment)
