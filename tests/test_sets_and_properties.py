"""A set is a table of its rows, and a property a table of one value per row of its subject.

A property arrives under ``subject.property``, or as a column of the table under
the subject's own key. Both lanes read the one the door hands them, so each case
here is held on both.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from tests.differential import both_lanes_refuse, differential

GENERATORS = ['g1', 'g2', 'g3']

#: g2 has no fuel, so the fuel cap, which reads the bus and the fuel together, does not reach it.
NETWORK: dict[str, Any] = {
    'dimensions': {'bus': {'dtype': 'str'}, 'generator': {'dtype': 'str'}, 'carrier': {'dtype': 'str'}},
    'properties': {'generator': {'bus': 'bus', 'fuel': {'into': 'carrier', 'missing': 'absent'}}},
    'parameters': {'cost': {'dims': ['generator']}, 'load': {'dims': ['bus']}, 'cap': {'dims': ['carrier']}},
    'variables': {'p': {'dims': ['generator'], 'bounds': {'lower': 0}}},
    'constraints': {
        'balance': {'dims': ['bus'], 'expression': 'sum(p, over=generator, by=generator[bus]) >= load'},
        'fuel_cap': {
            'dims': ['bus', 'carrier'],
            'expression': 'sum(p, over=generator, by=generator[bus, fuel]) <= cap',
        },
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost, over=generator)'},
}

#: The cheapest dispatch: g1 at its fuel cap of 4, g2 the rest of bus a's 10, g3 bus b's 5.
OPTIMUM = 4 * 1.0 + 6 * 2.0 + 5 * 1.0


def _network(**tables: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        'bus': ['a', 'b'],
        'generator': GENERATORS,
        'carrier': ['gas', 'coal'],
        'cost': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 2.0, 1.0]}),
        'load': pl.DataFrame({'bus': ['a', 'b'], 'value': [10.0, 5.0]}),
        'cap': pl.DataFrame({'carrier': ['gas', 'coal'], 'value': [4.0, 100.0]}),
        'generator.bus': pl.DataFrame({'generator': GENERATORS, 'bus': ['a', 'a', 'b']}),
        'generator.fuel': pl.DataFrame({'generator': ['g1', 'g3'], 'fuel': ['gas', 'coal']}),
    }
    return {name: table for name, table in (base | tables).items() if table is not None}


def _wide() -> dict[str, Any]:
    """The network with both properties as columns of the table under ``generator``, its index."""
    wide = pl.DataFrame({'generator': GENERATORS, 'bus': ['a', 'a', 'b'], 'fuel': ['gas', None, 'coal']})
    return _network(**{'generator': wide, 'generator.bus': None, 'generator.fuel': None})


def _wide_from_pandas() -> dict[str, Any]:
    """The wide table as PyPSA holds a component: a frame indexed by name, a column per attribute."""
    import pandas as pd

    frame = pd.DataFrame({'bus': ['a', 'a', 'b'], 'fuel': ['gas', None, 'coal']}, index=pd.Index(GENERATORS))
    return _network(
        **{
            'generator': frame.rename_axis('generator').reset_index(),
            'generator.bus': None,
            'generator.fuel': None,
        }
    )


@pytest.mark.parametrize(
    'sources',
    [
        pytest.param(_network, id='a-table-per-property'),
        pytest.param(_wide, id='a-wide-table-under-the-subject'),
        pytest.param(_wide_from_pandas, id='a-wide-pandas-frame'),
        pytest.param(
            lambda: _network(**{'generator': _wide()['generator'].drop('fuel'), 'generator.bus': None}),
            id='one-property-wide-and-one-by-its-own-key',
        ),
    ],
)
def test_a_property_reads_the_same_by_its_own_key_or_as_a_column_of_its_subject(sources):
    with differential(NETWORK, sources()) as run:
        assert run.oracle == pytest.approx(OPTIMUM), 'g1 runs to its fuel cap, g2 covers the rest of bus a'


def test_a_call_reading_two_properties_reads_only_the_rows_that_have_both():
    """``generator[bus, fuel]`` is the inner join of the two tables: g2, with no fuel, is in no group."""
    with differential(NETWORK, _network()) as run:
        p = run.result.primal('p').sort('generator')
        assert p['value'].to_list() == pytest.approx([4.0, 6.0, 5.0]), 'only g1 and g3 are capped by fuel'


def test_a_property_given_by_its_own_key_and_as_a_column_is_refused():
    both_lanes_refuse(
        NETWORK,
        _wide() | {'generator.fuel': _network()['generator.fuel']},
        match=r"property 'fuel' of 'generator' is given twice: under key 'generator\.fuel', and as a column",
    )


def test_a_null_in_a_wide_column_is_a_row_without_the_property():
    """A wide table holds a row per label, so g2's null bus is read as g2 having no bus, which ``bus`` refuses."""
    wide = pl.DataFrame({'generator': GENERATORS, 'bus': ['a', None, 'b'], 'fuel': ['gas', None, 'coal']})
    both_lanes_refuse(
        NETWORK,
        _network(**{'generator': wide, 'generator.bus': None, 'generator.fuel': None}),
        match=r"property 'bus' of 'generator' has no row at generator=g2\. The spec does not declare `missing:`",
    )


def test_a_property_with_two_values_for_one_row_is_refused():
    twice = pl.DataFrame({'generator': [*GENERATORS, 'g1'], 'bus': ['a', 'a', 'b', 'b']})
    both_lanes_refuse(
        NETWORK,
        _network(**{'generator.bus': twice}),
        match=r"property 'bus' of 'generator' has 1 row\(s\) of 'generator' more than once: generator=g1",
    )


def test_a_property_nothing_supplies_names_both_ways_to_supply_it():
    message = both_lanes_refuse(
        NETWORK, _network(**{'generator.bus': None}), match=r"no data provided for property 'bus' of 'generator'"
    )
    assert "under key 'generator.bus'" in message, 'the table of its own is one way'
    assert "a 'bus' column of the table under 'generator'" in message, 'a column of the wide table is the other'


#: A set of the entity-port pairs that exist, and the fuel each one burns. e2 has no port p2.
CONNECTIONS: dict[str, Any] = {
    'dimensions': {'entity': {'dtype': 'str'}, 'port': {'dtype': 'str'}, 'carrier': {'dtype': 'str'}},
    'sets': {'connection': {'dims': ['entity', 'port'], 'missing': 'absent'}},
    'properties': {'connection': {'fuel': 'carrier'}},
    'parameters': {'demand': {'dims': ['carrier']}},
    'variables': {'q': {'dims': ['entity', 'port'], 'where': 'connection', 'bounds': {'lower': 0}}},
    'constraints': {
        'meet': {'dims': ['carrier'], 'expression': 'sum(q, over=[entity, port], by=connection[fuel]) >= demand'}
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(q, over=[entity, port])'},
}

PAIRS = {'entity': ['e1', 'e1', 'e2'], 'port': ['p1', 'p2', 'p1']}


def _connections(**tables: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        'entity': ['e1', 'e2'],
        'port': ['p1', 'p2'],
        'carrier': ['gas', 'coal'],
        'demand': pl.DataFrame({'carrier': ['gas', 'coal'], 'value': [3.0, 2.0]}),
        'connection': pl.DataFrame(PAIRS),
        'connection.fuel': pl.DataFrame(PAIRS | {'fuel': ['gas', 'coal', 'gas']}),
    }
    return {name: table for name, table in (base | tables).items() if table is not None}


@pytest.mark.parametrize(
    'sources',
    [
        pytest.param(_connections, id='a-table-per-property'),
        pytest.param(
            lambda: _connections(
                connection=pl.DataFrame(PAIRS | {'fuel': ['gas', 'coal', 'gas']}), **{'connection.fuel': None}
            ),
            id='a-wide-table-under-the-set',
        ),
    ],
)
def test_a_property_of_a_set_covers_the_rows_of_the_set_rather_than_every_combination(sources):
    """``fuel`` refuses a missing row, and (e2, p2) is no row of the set, so it needs none there."""
    with differential(CONNECTIONS, sources()) as run:
        assert run.oracle == pytest.approx(5.0), 'gas demand 3 and coal demand 2, one unit each'


def test_a_property_of_a_set_at_a_row_the_set_lacks_is_refused():
    stray = pl.DataFrame({'entity': [*PAIRS['entity'], 'e2'], 'port': [*PAIRS['port'], 'p2']})
    both_lanes_refuse(
        CONNECTIONS,
        _connections(**{'connection.fuel': stray.with_columns(fuel=pl.lit('gas'))}),
        match=r"property 'fuel' of 'connection' has 1 row\(s\) that are not rows of 'connection': entity=e2, port=p2",
    )


def test_a_row_of_a_set_without_a_property_that_refuses_a_missing_row_is_refused():
    short = pl.DataFrame({'entity': ['e1', 'e1'], 'port': ['p1', 'p2'], 'fuel': ['gas', 'coal']})
    both_lanes_refuse(
        CONNECTIONS,
        _connections(**{'connection.fuel': short}),
        match=(
            r"(?s)property 'fuel' of 'connection' has no row at entity=e2, port=p1\. .*"
            r"Declare `missing: absent` on the property, if a row of 'connection' without it is meant\."
        ),
    )
