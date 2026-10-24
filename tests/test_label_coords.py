"""Sets and properties: structure on a dimension, never an axis.

Everything under ``dimensions:`` is an axis. A label a dimension's members
carry is a property that maps into another dimension, and which combinations of
labels exist is a set.
"""

from __future__ import annotations

import re
import warnings

import polars as pl
import pytest
from mathspec import to_spec

import specsolve as sps
from specsolve.errors import DataError, SpecsolveError, SpecsolveWarning
from tests.conftest import by_coord


def _spec(objective: str = 'sum(x, over=snapshot)') -> dict:
    return {
        'dimensions': {
            'snapshot': {'dtype': 'int', 'ordered': True},
            'period': {'dtype': 'int', 'ordered': True},
        },
        'properties': {'snapshot': {'period': 'period'}},
        'parameters': {'load': {'dims': ['snapshot']}},
        'variables': {'x': {'dims': ['snapshot'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'c': {'dims': ['snapshot'], 'expression': 'x >= load'}},
        'objective': {'sense': 'minimize', 'expression': objective},
    }


def _index() -> pl.DataFrame:
    return pl.DataFrame({'snapshot': [0, 1, 2]})


def _periods() -> pl.DataFrame:
    return pl.DataFrame({'period': [1, 2]})


def _snapshot_period() -> pl.DataFrame:
    """`snapshot[period]` as its own table, under the key `snapshot.period`."""
    return pl.DataFrame({'snapshot': [0, 1, 2], 'period': [1, 1, 2]})


def _load() -> pl.DataFrame:
    return pl.DataFrame({'snapshot': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})


def test_a_property_is_keyed_by_its_subject_and_determines_one_column():
    schema = to_spec(
        {
            'dimensions': {'bus': {}, 'generator': {}},
            'properties': {'generator': {'bus': 'bus'}},
        }
    )
    (declared,) = schema.program.relations_of('generator').values()
    assert (declared.key, declared.values) == (('generator',), ('bus',)), (
        'the declaration fixes which columns identify a row and which they determine, and no direction'
    )


def test_a_set_joins_the_flat_namespace():
    spec = _spec()
    spec['sets'] = {'snapshot_period': {'dims': ['snapshot', 'period'], 'missing': 'absent'}}
    spec['parameters']['snapshot_period'] = {'dims': ['snapshot']}
    with pytest.raises(SpecsolveError, match="Parameter 'snapshot_period' collides with the set"):
        to_spec(spec)


def test_a_set_cannot_take_a_dimensions_name():
    spec = _spec()
    spec['sets'] = {'period': {'dims': ['snapshot', 'period'], 'missing': 'absent'}}
    with pytest.raises(SpecsolveError, match="Set 'period' collides with the dimension"):
        to_spec(spec)


@pytest.mark.parametrize(
    ('by', 'match', 'offered'),
    [
        pytest.param(
            'snapshat[bus]', r'by=snapshat\[...\]\) does not name a set or a dimension', "'snapshot'", id='subject'
        ),
        pytest.param('snapshot[bsu]', r"names \['bsu'\], which is no property of 'snapshot'", "'bus'", id='property'),
    ],
)
def test_a_by_typo_is_offered_the_names_it_could_have_meant(by, match, offered):
    """A misspelt ``by=`` names the subjects or properties on offer rather than only refusing."""
    spec = _spec()
    spec['dimensions']['bus'] = {'dtype': 'str'}
    spec['properties']['snapshot']['bus'] = 'bus'
    spec['constraints']['c'] = {'dims': ['bus'], 'expression': f'sum(x, over=snapshot, by={by}) >= load'}
    with pytest.raises(SpecsolveError, match=match) as caught:
        sps.check(spec)
    assert offered in str(caught.value), 'a typo is answered with the names that were declared'


def test_a_dimension_only_a_property_maps_into_draws_no_advice():
    """A property's dimension is reached: its members are the labels the property is checked against."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sps.check(_spec())


def test_check_advises_an_unused_dimension():
    spec = _spec()
    spec['dimensions']['scenario'] = {'dtype': 'str'}
    with pytest.warns(SpecsolveWarning, match="'scenario' is never used"):
        sps.check(spec)


def test_a_dimension_grouped_into_draws_no_advice():
    """`group_by=` lands terms on its target, so the target is an axis even
    when nothing is declared over it — an objective groups and implicitly
    sums, and no warning fires."""
    spec = {
        'dimensions': {'bus': {}, 'generator': {}},
        'properties': {'generator': {'bus': 'bus'}},
        'parameters': {'cost': {'dims': ['generator']}},
        'variables': {'p': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'c': {'dims': ['generator'], 'expression': 'p <= 1'}},
        'objective': {
            'sense': 'minimize',
            'expression': 'sum(sum(p * cost, over=generator, by=generator[bus]), over=bus)',
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sps.check(spec)


def test_a_property_arrives_under_its_own_key():
    sources = {'load': _load(), 'snapshot': _index(), 'period': _periods(), 'snapshot.period': _snapshot_period()}
    with sps.solve(_spec(), sources) as solution:
        assert solution.objective == pytest.approx(6.0)

    with pytest.raises(DataError, match="no data provided for property 'period' of 'snapshot'"):
        sps.build(_spec(), {'load': _load(), 'snapshot': _index(), 'period': _periods()})


def test_a_property_is_single_valued_per_row():
    doubled = pl.DataFrame({'snapshot': [0, 0, 1, 2], 'period': [1, 2, 1, 2]})
    sources = {'load': _load(), 'snapshot': _index(), 'period': _periods(), 'snapshot.period': doubled}
    with pytest.raises(DataError, match='more than once'):
        sps.build(_spec(), sources)


def _unused_target_spec(month: dict) -> dict:
    """#488's incremental multi-period shape.

    The flat ``snapshot`` index declares every property it will need, but no
    constraint groups into ``month`` yet — only ``period`` is used.
    """
    return {
        'dimensions': {
            'snapshot': {'dtype': 'int', 'ordered': True},
            'period': {'dtype': 'int', 'ordered': True},
            'month': month,
        },
        'properties': {'snapshot': {'period': 'period', 'month': 'month'}},
        'parameters': {'cap': {'dims': ['period']}},
        'variables': {'p': {'dims': ['snapshot'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {
            'budget': {'dims': ['period'], 'expression': 'sum(p, over=snapshot, by=snapshot[period]) <= cap'}
        },
        'objective': {'sense': 'maximize', 'expression': 'sum(p, over=snapshot)'},
    }


def _unused_target_sources() -> dict:
    return {
        'snapshot': pl.DataFrame({'snapshot': [0, 1, 2]}),
        'snapshot.period': pl.DataFrame({'snapshot': [0, 1, 2], 'period': [2030, 2030, 2050]}),
        'snapshot.month': pl.DataFrame({'snapshot': [0, 1, 2], 'month': ['jan', 'feb', 'jan']}),
        'period': pl.DataFrame({'period': [2030, 2050]}),
        'cap': pl.DataFrame({'period': [2030, 2050], 'value': [5.0, 5.0]}),
    }


@pytest.mark.parametrize(
    ('month', 'extra'),
    [
        pytest.param(
            {'dtype': 'str', 'ordered': True}, {'month': pl.DataFrame({'month': ['jan', 'feb']})}, id='a-table'
        ),
        pytest.param({'dtype': 'str', 'ordered': True}, {'month': ['jan', 'feb']}, id='a-bare-sequence'),
    ],
)
def test_a_property_may_map_into_a_dimension_nothing_spans_yet(month, extra):
    """#488: the first build after declaring a property, before its constraint exists."""
    with sps.solve(_unused_target_spec(month), _unused_target_sources() | extra) as solution:
        assert solution.objective == pytest.approx(10.0), 'each period caps its snapshots at 5, so the model builds'


@pytest.mark.parametrize('lane', ['relational', 'linopy'])
def test_an_unused_target_still_checks_containment(lane):
    """A property into a dimension no constraint groups by is checked all the same, on both lanes."""
    from tests.oracle import specsolve_linopy

    build = sps.build if lane == 'relational' else specsolve_linopy.build
    short = {'month': pl.DataFrame({'month': ['jan']})}
    with pytest.raises(DataError, match="not 'month' labels"):
        build(_unused_target_spec({'dtype': 'str', 'ordered': True}), _unused_target_sources() | short)


@pytest.mark.parametrize('lane', ['relational', 'linopy'])
def test_an_unused_target_without_an_index_is_refused_with_the_true_reason(lane):
    """The refusal names the missing index, not data the caller may well have supplied (#488)."""
    from tests.oracle import specsolve_linopy

    build = sps.build if lane == 'relational' else specsolve_linopy.build
    with pytest.raises(DataError, match='no index of its own') as caught:
        build(_unused_target_spec({'dtype': 'str', 'ordered': True}), _unused_target_sources())
    assert "Pass an index for 'month'" in str(caught.value), 'the refusal has to say what would satisfy it'


def test_both_lanes_read_the_same_index():
    """The `snapshot[period]` property is read the same way on the linopy lane too —
    both lanes reach the 6.0 the relational test above asserts.

    The oracle is imported in the body, so the bare install skips this one test.
    """
    from tests.differential import differential
    from tests.oracle import pd

    data = {'load': pd.Series({0: 1.0, 1: 2.0, 2: 3.0}).rename_axis('snapshot')}
    index = {
        'snapshot': pd.DataFrame({'snapshot': [0, 1, 2]}),
        'period': pd.DataFrame({'period': [1, 2]}),
        'snapshot.period': pd.DataFrame({'snapshot': [0, 1, 2], 'period': [1, 1, 2]}),
    }
    with differential(_spec(), data | index) as run:
        assert run.oracle == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# where on a property (#553)
# ---------------------------------------------------------------------------


#: A three-line network whose structure is entirely properties: each line has two
#: endpoints, and `spur` deliberately has an open end so the partial case is
#: reachable. `voltage` maps into an int dimension and `send`/`recv` into a
#: string one, so one model covers every property predicate.
NETWORK = {
    'dimensions': {'bus': {'dtype': 'str'}, 'line': {'dtype': 'str'}, 'kv': {'dtype': 'int'}},
    'properties': {'line': {'send': 'bus', 'recv': {'into': 'bus', 'missing': 'absent'}, 'voltage': 'kv'}},
    'parameters': {'cap': {'dims': ['line']}, 'price': {'dims': ['line']}},
    'variables': {'f': {'dims': ['line'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'ceiling': {'dims': ['line'], 'expression': 'f <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(f * price)'},
}

LINES = ['ring_a', 'ring_b', 'loop', 'spur']
SEND = ['north', 'south', 'north', 'north']
RECV = ['south', 'north', 'north']
VOLTAGE = [220, 380, 220, 380]
CAP = [10.0, 20.0, 30.0, 40.0]
PRICE = [1.0, 1.0, 1.0, 1.0]

#: `loop` starts and ends on the same bus; `spur` has no receiving end at all,
#: which `recv` says by having no row for it.
NETWORK_SOURCES = {
    'bus': pl.DataFrame({'bus': ['north', 'south']}),
    'line': pl.DataFrame({'line': LINES}),
    'kv': pl.DataFrame({'kv': [220, 380]}),
    'line.send': pl.DataFrame({'line': LINES, 'send': SEND}),
    'line.recv': pl.DataFrame({'line': LINES[:3], 'recv': RECV}),
    'line.voltage': pl.DataFrame({'line': LINES, 'voltage': VOLTAGE}),
    'cap': pl.DataFrame({'line': LINES, 'value': CAP}),
    'price': pl.DataFrame({'line': LINES, 'value': PRICE}),
}


@pytest.mark.parametrize(
    ('where', 'kept'),
    [
        pytest.param('line[voltage] == 220', ['loop', 'ring_a'], id='a-property-into-an-int-dimension'),
        pytest.param("line[send] == 'north'", ['loop', 'ring_a', 'spur'], id='a-property-against-a-label'),
        pytest.param('line[send] != line[recv]', ['ring_a', 'ring_b'], id='two-properties-into-one-dimension'),
        pytest.param('line[recv]', ['loop', 'ring_a', 'ring_b'], id='a-bare-property-is-the-partial-case'),
        pytest.param('NOT line[voltage] == 220', ['ring_b', 'spur'], id='negated'),
        pytest.param(
            'line[voltage] == 380 AND line[send] != line[recv]', ['ring_b'], id='conjoined-with-a-pair-comparison'
        ),
        pytest.param("at(bus == 'north', by=line[send])", ['loop', 'ring_a', 'spur'], id='a-predicate-read-at'),
        pytest.param("at(bus == 'north', by=line[recv])", ['loop', 'ring_b'], id='read-at-a-partial-property'),
    ],
)
def test_a_where_reads_a_property(where, kept):
    """The atom #553 asked for, in both shapes.

    `kept` is asserted rather than just a count: a predicate that inverted its
    sense would keep the complement, which is the same size on a symmetric
    case and a different model everywhere.

    `spur`'s missing `recv` is the reading law 8 fixes — a comparison over it is
    false, so the bare `line[recv]` keeps exactly the lines that have it.
    """
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': where}}}
    with sps.solve(spec, NETWORK_SOURCES) as result:
        built = sorted(row['line'] for row in result.primal('f').to_dicts())
    assert built == sorted(kept), f'where: {where!r} built the wrong set of variables'


@pytest.mark.parametrize(
    ('where', 'objective'),
    [
        pytest.param('line[send] != line[recv]', 30.0, id='the-two-ring-lines-survive'),
        pytest.param('NOT line[send] != line[recv]', 70.0, id='negated-over-a-partial-property'),
        # the linopy lane's null exclusion: numpy answers `None != 'north'` with True
        pytest.param("line[recv] != 'north'", 10.0, id='not-equal-over-a-null-value'),
        pytest.param('line[recv] != line[send]', 30.0, id='not-equal-between-two-properties'),
        pytest.param("at(bus == 'north', by=line[send])", 80.0, id='a-predicate-read-at'),
        pytest.param("NOT at(bus == 'south', by=line[recv])", 90.0, id='negated-read-at-a-partial-property'),
        pytest.param("at(bus == 'north', by=line[recv]) AND line[voltage] == 220", 30.0, id='read-at-conjoined'),
    ],
)
def test_a_property_where_agrees_with_the_oracle(where, objective):
    """Both lanes, one answer — the differential half of #553.

    A mask reading a property is a join on the dim table in the relational lane
    and an array read in the linopy one; nothing but this shows they agree on
    which rows survive, since a wrong mask still solves.
    """
    from tests.differential import differential
    from tests.oracle import pd

    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': where}}}
    data = {
        'cap': pd.Series(CAP, index=LINES),
        'price': pd.Series(PRICE, index=LINES),
    }
    index = {
        'bus': pd.Index(['north', 'south'], name='bus'),
        'line': pd.DataFrame({'line': LINES}),
        'kv': pd.DataFrame({'kv': [220, 380]}),
        'line.send': pd.DataFrame({'line': LINES, 'send': SEND}),
        'line.recv': pd.DataFrame({'line': LINES[:3], 'recv': RECV}),
        'line.voltage': pd.DataFrame({'line': LINES, 'voltage': VOLTAGE}),
    }
    with differential(spec, data | index) as run:
        assert run.result.objective == pytest.approx(objective), (
            f'where: {where!r} — the two lanes agree on the objective but not on this one'
        )


def test_a_where_on_a_property_outside_the_frame_is_refused():
    """A property is read on its subject's dims, so those dims have to be in the
    frame — otherwise the mask would silently reduce over an unlisted dim."""
    spec = {
        **NETWORK,
        'variables': {'f': {'dims': ['line'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {
            'ceiling': {
                'dims': ['bus'],
                'where': 'line[voltage] == 220',
                'expression': 'sum(f, over=line, by=line[send]) <= 100',
            }
        },
    }
    with pytest.raises(
        SpecsolveError, match=r"the read of 'line\[voltage\]' reads dims \['line'\] outside the declared dims"
    ):
        to_spec(spec)


def test_two_properties_of_subjects_over_different_dims_cannot_be_compared():
    """There is no row carrying both, so the comparison has nothing to test.

    The pair comparison is legal exactly because two properties of one subject
    are two columns of one index. Drop that condition and this reads as a join
    nobody wrote.
    """
    spec = {
        **NETWORK,
        'dimensions': {**NETWORK['dimensions'], 'zone': {'dtype': 'str'}},
        'properties': {**NETWORK['properties'], 'bus': {'zone': 'zone'}},
        'variables': {'f': {**NETWORK['variables']['f'], 'where': 'line[send] != bus[zone]'}},
    }
    with pytest.raises(SpecsolveError, match='over different dimensions'):
        to_spec(spec)


@pytest.mark.parametrize(
    ('extra', 'where'),
    [
        pytest.param({'area': 'zone'}, 'line[send] != line[area]', id='two-targets-of-the-same-dtype'),
        pytest.param({}, 'line[send] != line[voltage]', id='two-targets-of-different-dtypes'),
    ],
)
def test_two_properties_into_different_label_sets_cannot_be_compared(extra, where):
    """One dimension is necessary but not sufficient — the targets must match too.

    A bus label is never a zone label, so the predicate could only mask
    everything out.
    """
    spec = {
        **NETWORK,
        'dimensions': {**NETWORK['dimensions'], 'zone': {'dtype': 'str'}},
        'properties': {'line': {**NETWORK['properties']['line'], **extra}},
        'variables': {'f': {**NETWORK['variables']['f'], 'where': where}},
    }
    with pytest.raises(SpecsolveError, match='Compare two properties into the same dimension'):
        to_spec(spec)


def test_a_property_comparison_is_checked_against_its_dtype():
    """The same dtype check every other where-comparison gets (#460).

    A property's literal is as silent to get wrong as a dimension's: `voltage`
    is an int, so a quoted right-hand side matches nothing rather than
    erroring at run time.
    """
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': "line[voltage] == 'high'"}}}
    with pytest.raises(SpecsolveError, match=r"has dtype 'int'"):
        to_spec(spec)


def test_a_property_compares_against_a_label_the_target_lacks():
    """A stranger label masks everything out; it does not raise.

    The property column is compared as a string, because the target's `Enum`
    refuses a label outside it.
    """
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': "line[send] == 'atlantis'"}}}
    with sps.build(spec, NETWORK_SOURCES) as model:
        surviving = model._engine._model.variables['f'].frame.select(pl.len()).collect().item()
    assert surviving == 0, "a label no bus carries matches nothing, so no 'f' is built"


def test_a_relation_orders_bytewise_not_by_declaration():
    """Labels order bytewise, whatever order the dimension declared them.

    Attaching casts a property column to the target's `Enum`, which orders by
    *declaration*, so an ordering comparison read off it would answer a
    different question — and silently, since both readings return a mask.
    `south` is declared first here precisely so the two disagree.
    """
    sources = {**NETWORK_SOURCES, 'bus': pl.DataFrame({'bus': ['south', 'north']})}
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': "line[send] >= 'south'"}}}
    with sps.solve(spec, sources) as result:
        built = sorted(row['line'] for row in result.primal('f').to_dicts())
    assert built == ['ring_b'], "only ring_b sends from 'south'; declaration order would keep the 'north' lines too"


# ---------------------------------------------------------------------------
# a property, supplied under its own key
# ---------------------------------------------------------------------------


#: The model the section is written over: one property, and the caller holding
#: both its labels and its table. `g3` is on no bus — the partial case,
#: spelled by the row that is not there.
GENERATORS = ['g1', 'g2', 'g3']
COST = [1.0, 2.0, 0.5]
LOAD = [5.0, 4.0]

BASE = {
    'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}},
    'properties': {'generator': {'bus': {'into': 'bus', 'missing': 'absent'}}},
    'parameters': {'cost': {'dims': ['generator']}, 'load': {'dims': ['bus']}},
    'variables': {'p': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'balance': {'dims': ['bus'], 'expression': 'sum(p, over=generator, by=generator[bus]) >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}

BASE_SOURCES = {
    'bus': ['north', 'south'],
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': COST}),
    'load': pl.DataFrame({'bus': ['north', 'south'], 'value': LOAD}),
}


#: The same labels and property as one wide table under the subject's key, null where a label lacks it.
_LABELS_AND_PROPERTY = pl.DataFrame({'generator': GENERATORS, 'bus': ['north', 'south', None]})


def test_a_property_may_arrive_as_a_column_of_its_subjects_index():
    """The wide form: the index carries the property, and a null is a row that lacks it.

    It builds the model the narrow table under `generator.bus` builds below.
    """
    with sps.solve(BASE, {**BASE_SOURCES, 'generator': _LABELS_AND_PROPERTY}) as result:
        built = by_coord(result, 'p', 'generator')
        assert result.objective == pytest.approx(13.0), 'the wide table reads as the narrow one does'
    assert built['g3'] == pytest.approx(0.0), "a null in the index's 'bus' column is a generator on no bus"


def test_a_property_given_both_ways_is_refused():
    """Under its own key and as a column of the index at once is two claims, and neither is dropped."""
    with pytest.raises(DataError, match=re.escape("property 'bus' of 'generator' is given twice")):
        sps.solve(BASE, {**BASE_SOURCES, 'generator.bus': _PROPERTY, 'generator': _LABELS_AND_PROPERTY})


def test_a_property_alone_does_not_say_which_labels_exist():
    """A property is a table over a dimension, never the dimension.

    It may omit members and its key order is whatever someone typed, so reading
    the label set out of it would let an added entry create a member and a
    reordered table re-order the axis that ``shift`` reads positionally. With
    nothing supplying ``generator``'s labels, the property over it leaves the
    dimension without an index, and both lanes say so. Read off the property,
    `g3` — on no bus and still a generator — is the label that would vanish.
    """
    with pytest.raises(
        DataError, match=re.escape("has tables over it (sources['generator.bus']) but no index of its own")
    ):
        sps.solve(BASE, {**BASE_SOURCES, 'generator.bus': _PROPERTY})


# ---------------------------------------------------------------------------
# A property supplied under its own key
# ---------------------------------------------------------------------------


#: `generator.bus` takes its own source key, and the two columns say what it is
#: — the subject it is a fact about, and the property. `g3` is on no bus by
#: being in no row.
SUPPLIED = BASE

_PROPERTY = pl.DataFrame({'generator': ['g1', 'g2'], 'bus': ['north', 'south']})
_SUPPLIED_SOURCES = {**BASE_SOURCES, 'generator': GENERATORS, 'generator.bus': _PROPERTY}


def test_a_supplied_property_reaches_the_declared_map_without_touching_the_index():
    """The point: a caller adds a property to a dimension whose index is not theirs.

    `g3` is the cheapest generator and contributes nothing, which is what a
    label without the property means. The index goes in as a bare label list,
    so nothing here rewrites a table someone else generated.
    """
    with sps.solve(SUPPLIED, _SUPPLIED_SOURCES) as result:
        built = by_coord(result, 'p', 'generator')
        assert result.objective == pytest.approx(13.0)
    assert built['g3'] == pytest.approx(0.0), 'a generator in no row of the property is a generator on no bus'


def test_a_supplied_property_agrees_with_the_oracle():
    """Both lanes read the property through the one front door, so both see it."""
    from tests.differential import differential
    from tests.oracle import pd

    data = {
        'cost': pd.Series([1.0, 2.0, 0.5], index=['g1', 'g2', 'g3']),
        'load': pd.Series([5.0, 4.0], index=['north', 'south']),
        'bus': ['north', 'south'],
        'generator': ['g1', 'g2', 'g3'],
        'generator.bus': _PROPERTY,
    }
    with differential(SUPPLIED, data) as run:
        assert run.result.objective == pytest.approx(13.0)


def test_a_partial_property_is_supplied_as_the_rows_it_has():
    """Absence is the absent row here as everywhere else.

    The property says nothing about `g3`, and a null in its own table is
    refused the way a null in a parameter's values is.
    """
    holed = pl.DataFrame({'generator': ['g1', 'g2', 'g3'], 'bus': ['north', 'south', None]})
    with pytest.raises(DataError, match=r"property 'bus' of 'generator' carries 1 row\(s\) with a null in 'bus'"):
        sps.solve(SUPPLIED, {**_SUPPLIED_SOURCES, 'generator.bus': holed})


def test_a_where_reads_a_property_that_leaves_a_label_out():
    """A label the property has no row for satisfies no comparison over it.

    The reading law for a partial property, at the one place it changes the
    model: the constraint is placed only where the property says something.
    """
    spec = _spec()
    spec['properties']['snapshot']['period'] = {'into': 'period', 'missing': 'absent'}
    spec['constraints']['c']['where'] = 'snapshot[period] == 1'
    sources = {
        'snapshot': [0, 1, 2],
        'period': _periods(),
        'load': _load(),
        'snapshot.period': pl.DataFrame({'snapshot': [0, 1], 'period': [1, 1]}),
    }
    with sps.solve(spec, sources) as result:
        built = sorted(row['snapshot'] for row in result.primal('x').to_dicts())
    assert built == [0, 1, 2], 'the variable is unmasked; only the constraint reads the property'
    assert result.objective == pytest.approx(3.0), 'snapshot 2 is in no row of the property, so nothing constrains it'


@pytest.mark.parametrize(
    ('table', 'match'),
    [
        pytest.param(
            pl.DataFrame({'generator': ['g1'], 'value': ['north']}),
            r"must carry the columns \['generator', 'bus'\]",
            id='a-value-column-not-named-after-the-property',
        ),
        pytest.param(
            pl.DataFrame({'bus': ['north']}),
            r"must carry the columns \['generator', 'bus'\]",
            id='no-subject-column',
        ),
        pytest.param(
            pl.DataFrame({'generator': ['g1', 'g1'], 'bus': ['north', 'south']}),
            r"has 1 row\(s\) of 'generator' more than once: generator=g1",
            id='given-twice-for-one-row',
        ),
        pytest.param(
            pl.DataFrame({'generator': ['g9'], 'bus': ['north']}),
            r"property 'bus' of 'generator' has value\(s\) in 'generator' that are not 'generator' labels: 'g9'",
            id='subject-is-not-a-label',
        ),
    ],
)
def test_a_supplied_property_is_held_to_what_a_property_is(table, match):
    """Single-valued, about labels that exist, and spelled as the pair it is.

    Dropping a stray row instead of refusing it would place that generator's
    terms nowhere while the model built and solved.
    """
    with pytest.raises(DataError, match=match):
        sps.solve(SUPPLIED, {**_SUPPLIED_SOURCES, 'generator.bus': table})


def test_a_property_with_no_author_at_all_is_refused():
    """Neither is not a spelling of empty: a property nothing supplies is missing data."""
    with pytest.raises(DataError, match="no data provided for property 'bus' of 'generator'"):
        sps.solve(SUPPLIED, {**BASE_SOURCES, 'generator': GENERATORS})


@pytest.mark.parametrize('lane', ['relational', 'linopy'])
def test_a_supplied_property_is_refused_the_same_way_on_both_lanes(lane):
    """One defect, one sentence: the checks live in the door both lanes enter."""
    from tests.oracle import specsolve_linopy

    build = sps.solve if lane == 'relational' else specsolve_linopy.build
    with pytest.raises(DataError, match=r"has 1 row\(s\) of 'generator' more than once"):
        build(
            SUPPLIED,
            {
                **_SUPPLIED_SOURCES,
                'generator.bus': pl.DataFrame({'generator': ['g1', 'g1'], 'bus': ['north', 'south']}),
            },
        )


#: Two properties of one dimension into the same dimension — the PyPSA shape,
#: where a line has a sending and a receiving bus. Every check over a
#: dimension's properties has to run per property.
TWO_MAPS = {
    'dimensions': {'line': {}, 'bus': {'dtype': 'str'}},
    'properties': {'line': {'from_bus': 'bus', 'to_bus': 'bus'}},
    'parameters': {'flow_max': {'dims': ['line']}, 'load': {'dims': ['bus']}},
    'variables': {'f': {'dims': ['line'], 'bounds': {'lower': 0, 'upper': 'flow_max'}}},
    'constraints': {'served': {'dims': ['bus'], 'expression': 'sum(f, over=line, by=line[to_bus]) >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(f, over=line)'},
}

_TWO_MAP_SOURCES = {
    'line': ['l1', 'l2'],
    'bus': ['north', 'south'],
    'flow_max': pl.DataFrame({'line': ['l1', 'l2'], 'value': [10.0, 10.0]}),
    'load': pl.DataFrame({'bus': ['north', 'south'], 'value': [1.0, 2.0]}),
    'line.from_bus': pl.DataFrame({'line': ['l1', 'l2'], 'from_bus': ['south', 'north']}),
    'line.to_bus': pl.DataFrame({'line': ['l1', 'l2'], 'to_bus': ['north', 'south']}),
}


def test_two_properties_into_one_dimension_each_take_their_own_key():
    """Two properties into one dimension, told apart by the key and the column they arrive under."""
    with sps.solve(TWO_MAPS, _TWO_MAP_SOURCES) as result:
        assert result.objective == pytest.approx(3.0), 'each line serves the bus its to_bus names'


def test_a_property_column_is_named_after_the_property_not_its_dimension():
    """`from_bus` maps into `bus`, and its table carries a `from_bus` column, not a `bus` one."""
    table = pl.DataFrame({'line': ['l1', 'l2'], 'bus': ['south', 'north']})
    with pytest.raises(DataError, match=re.escape("must carry the columns ['line', 'from_bus']")):
        sps.solve(TWO_MAPS, {**_TWO_MAP_SOURCES, 'line.from_bus': table})


def test_the_second_property_is_checked_as_hard_as_the_first():
    """Per property, not per dimension: the check runs for `from_bus` as for `to_bus`."""
    index = pl.DataFrame({'line': ['l1', 'l2'], 'from_bus': ['south', 'north']})
    with pytest.raises(DataError, match=re.escape("property 'from_bus' of 'line' is given twice")):
        sps.solve(TWO_MAPS, {**_TWO_MAP_SOURCES, 'line': index})


#: A parameter written positionally over a dimension that carries a supplied
#: property. `cost` is a bare list, so which label each number belongs to is the
#: index's row order — the order a property joined onto it must not disturb.
#: `cap` pins the solution to `t = 0` alone, so the objective *is* that label's cost.
POSITIONAL = {
    'dimensions': {'t': {'dtype': 'int', 'ordered': True}, 'g': {'dtype': 'str'}},
    'properties': {'t': {'g': 'g'}},
    'parameters': {'cost': {'dims': ['t']}, 'cap': {'dims': ['t']}},
    'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'c': {'dims': ['t'], 'expression': 'x >= cap'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x * cost, over=t)'},
}


def test_a_supplied_property_does_not_reorder_the_index_it_joins_onto():
    """A label's position is its ordinal, and joining a property on may not move it.

    A positional shape is placed against the labels read back off the index
    *after* the property has been joined onto it, so a join free to reorder
    hands every one of these numbers to the wrong label. Both lanes would agree
    on that model, so the check is a number rather than a comparison between them.
    """
    sources = {
        't': pl.DataFrame({'t': [0, 1, 2]}),
        'g': ['n', 's'],
        't.g': pl.DataFrame({'t': [0, 1, 2], 'g': ['n', 'n', 's']}),
        'cost': [1.0, 10.0, 100.0],
        'cap': pl.DataFrame({'t': [0, 1, 2], 'value': [1.0, 0.0, 0.0]}),
    }
    with sps.solve(POSITIONAL, sources) as result:
        assert result.objective == pytest.approx(1.0), "the first number is the first label's, whatever the join did"


# ---------------------------------------------------------------------------
# the set and property shapes the relational lane builds
# ---------------------------------------------------------------------------


def _walked(declared: dict, expression: str) -> dict:
    """A model whose one constraint walks the *declared* sets and properties, with both ends named."""
    return {
        'dimensions': {
            'generator': {'dtype': 'str'},
            'bus': {'dtype': 'str'},
            'period': {'dtype': 'int', 'ordered': True},
        },
        **declared,
        'parameters': {'load': {'dims': ['bus']}},
        'variables': {'p': {'dims': ['generator', 'period'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'bal': {'dims': ['bus', 'period'], 'expression': expression}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }


@pytest.mark.parametrize(
    ('declared', 'expression', 'tables'),
    [
        pytest.param(
            {'properties': {'generator': {'bus': 'bus'}}},
            'sum(p, over=generator, by=generator[bus]) >= load',
            {'generator.bus'},
            id='a-property-of-a-dimension',
        ),
        pytest.param(
            {'sets': {'generator_bus': {'dims': ['generator', 'bus'], 'missing': 'absent'}}},
            'sum(p, over=generator, by=generator_bus[bus]) >= load',
            {'generator_bus'},
            id='a-set',
        ),
        pytest.param(
            {
                'sets': {'generator_period': {'dims': ['generator', 'period']}},
                'properties': {'generator_period': {'bus': 'bus'}},
            },
            'sum(p, over=generator, by=generator_period[bus]) >= load',
            {'generator_period', 'generator_period.bus'},
            id='a-property-of-a-set',
        ),
    ],
)
def test_every_set_and_property_shape_the_language_admits_passes_check(
    declared: dict, expression: str, tables: set[str]
) -> None:
    """A set or a property is any table walked any way, and `check` refuses none of them.

    What each shape builds to is `test_relation_shapes.py`'s subject.
    """
    program = sps.check(_walked(declared, expression))
    assert set(program.relations) == tables, 'every set and property the model declares, each its own table'
