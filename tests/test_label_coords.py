"""Relations: structure on a dimension, never an axis.

Everything under ``dimensions:`` is an axis, and a label a dimension's members
carry is a relation into another dimension.
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
            'snapshot': {'dtype': 'int'},
            'period': {'dtype': 'int'},
        },
        'relations': {'period_of': {'key': 'snapshot', 'values': 'period'}},
        'parameters': {'load': {'dims': ['snapshot']}},
        'variables': {'x': {'dims': ['snapshot'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'c': {'dims': ['snapshot'], 'expression': 'x >= load'}},
        'objective': {'sense': 'minimize', 'expression': objective},
    }


def _index() -> pl.DataFrame:
    return pl.DataFrame({'snapshot': [0, 1, 2]})


def _periods() -> pl.DataFrame:
    return pl.DataFrame({'period': [1, 2]})


def _period_of() -> pl.DataFrame:
    """`period_of` as a relation, which is the only way a map arrives as data."""
    return pl.DataFrame({'snapshot': [0, 1, 2], 'period': [1, 1, 2]})


def _load() -> pl.DataFrame:
    return pl.DataFrame({'snapshot': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})


def test_a_relation_names_the_columns_a_row_is_keyed_by_and_the_ones_they_determine():
    schema = to_spec(
        {
            'dimensions': {'bus': {}, 'generator': {}},
            'relations': {'gen_bus': {'key': 'generator', 'values': 'bus'}},
        }
    )
    (declared,) = schema.program.relations_of('generator').values()
    assert (declared.key, declared.values) == (('generator',), ('bus',)), (
        'the declaration fixes which columns identify a row and which they determine, and no direction'
    )


def test_a_relation_joins_the_flat_namespace():
    spec = _spec()
    spec['parameters']['period_of'] = {'dims': ['snapshot']}
    with pytest.raises(SpecsolveError, match="Parameter 'period_of' collides with the relation"):
        to_spec(spec)


def test_a_relation_cannot_take_a_dimensions_name():
    spec = _spec()
    spec['relations']['period'] = {'key': 'snapshot', 'values': 'period'}
    with pytest.raises(SpecsolveError, match="Relation 'period' collides with the dimension"):
        to_spec(spec)


def test_a_by_typo_is_offered_the_relations_it_could_have_meant():
    """A misspelt ``by=`` names the relations on offer rather than only refusing."""
    spec = _spec()
    spec['dimensions']['bus'] = {'dtype': 'str'}
    spec['relations']['bus_of'] = {'key': 'snapshot', 'values': 'bus'}
    spec['constraints']['c'] = {'dims': ['bus'], 'expression': 'sum(x, by=bus_ov, over=snapshot, into=bus) >= load'}
    with pytest.raises(SpecsolveError, match=r'by=bus_ov\) does not name a relation') as caught:
        sps.check(spec)
    assert "'bus_of'" in str(caught.value), 'a typo is answered with the relations that were declared'


def test_a_dimension_only_a_relation_targets_draws_no_advice():
    """A relation's target is reached: its members are the labels the map is checked against."""
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
        'relations': {'gen_bus': {'key': 'generator', 'values': 'bus'}},
        'parameters': {'cost': {'dims': ['generator']}},
        'variables': {'p': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'c': {'dims': ['generator'], 'expression': 'p <= 1'}},
        'objective': {
            'sense': 'minimize',
            'expression': 'sum(sum(p * cost, by=gen_bus, over=generator, into=bus), over=bus)',
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sps.check(spec)


def test_a_map_arrives_under_its_own_name():
    sources = {'load': _load(), 'snapshot': _index(), 'period': _periods(), 'period_of': _period_of()}
    with sps.solve(_spec(), sources) as solution:
        assert solution.objective == pytest.approx(6.0)

    with pytest.raises(DataError, match="no data provided for relation 'period_of'"):
        sps.build(_spec(), {'load': _load(), 'snapshot': _index(), 'period': _periods()})


def test_a_relation_is_single_valued_per_label():
    doubled = pl.DataFrame({'snapshot': [0, 0, 1, 2], 'period': [1, 2, 1, 2]})
    sources = {'load': _load(), 'snapshot': _index(), 'period': _periods(), 'period_of': doubled}
    with pytest.raises(DataError, match='more than once'):
        sps.build(_spec(), sources)


def _unused_target_spec(month: dict) -> dict:
    """#488's incremental multi-period shape.

    The flat ``snapshot`` index declares every relation it will need, but no
    constraint groups into ``month`` yet — only ``period`` is used.
    """
    return {
        'dimensions': {
            'snapshot': {'dtype': 'int'},
            'period': {'dtype': 'int'},
            'month': month,
        },
        'relations': {
            'period_of': {'key': 'snapshot', 'values': 'period'},
            'month_of': {'key': 'snapshot', 'values': 'month'},
        },
        'parameters': {'cap': {'dims': ['period']}},
        'variables': {'p': {'dims': ['snapshot'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {
            'budget': {'dims': ['period'], 'expression': 'sum(p, by=period_of, over=snapshot, into=period) <= cap'}
        },
        'objective': {'sense': 'maximize', 'expression': 'sum(p, over=snapshot)'},
    }


def _unused_target_sources() -> dict:
    return {
        'snapshot': pl.DataFrame({'snapshot': [0, 1, 2]}),
        'period_of': pl.DataFrame({'snapshot': [0, 1, 2], 'period': [2030, 2030, 2050]}),
        'month_of': pl.DataFrame({'snapshot': [0, 1, 2], 'month': ['jan', 'feb', 'jan']}),
        'period': pl.DataFrame({'period': [2030, 2050]}),
        'cap': pl.DataFrame({'period': [2030, 2050], 'value': [5.0, 5.0]}),
    }


@pytest.mark.parametrize(
    ('month', 'extra'),
    [
        pytest.param({'dtype': 'str'}, {'month': pl.DataFrame({'month': ['jan', 'feb']})}, id='a-table'),
        pytest.param({'dtype': 'str'}, {'month': ['jan', 'feb']}, id='a-bare-sequence'),
    ],
)
def test_a_relation_may_target_a_dimension_nothing_spans_yet(month, extra):
    """#488: the first build after declaring a relation, before its constraint exists."""
    with sps.solve(_unused_target_spec(month), _unused_target_sources() | extra) as solution:
        assert solution.objective == pytest.approx(10.0), 'each period caps its snapshots at 5, so the model builds'


@pytest.mark.parametrize('lane', ['relational', 'linopy'])
def test_an_unused_target_still_checks_containment(lane):
    """A map into a dimension no constraint groups by is checked all the same, on both lanes."""
    from tests.oracle import specsolve_linopy

    build = sps.build if lane == 'relational' else specsolve_linopy.build
    short = {'month': pl.DataFrame({'month': ['jan']})}
    with pytest.raises(DataError, match="not 'month' labels"):
        build(_unused_target_spec({'dtype': 'str'}), _unused_target_sources() | short)


@pytest.mark.parametrize('lane', ['relational', 'linopy'])
def test_an_unused_target_without_an_index_is_refused_with_the_true_reason(lane):
    """The refusal names the missing index, not data the caller may well have supplied (#488)."""
    from tests.oracle import specsolve_linopy

    build = sps.build if lane == 'relational' else specsolve_linopy.build
    with pytest.raises(DataError, match='no index of its own') as caught:
        build(_unused_target_spec({'dtype': 'str'}), _unused_target_sources())
    assert "Pass an index for 'month'" in str(caught.value), 'the refusal has to say what would satisfy it'


def test_both_lanes_read_the_same_index():
    """The `period_of` relation is read the same way on the linopy lane too —
    both lanes reach the 6.0 the relational test above asserts.

    The oracle is imported in the body, so the bare install skips this one test.
    """
    from tests.differential import differential
    from tests.oracle import pd

    data = {'load': pd.Series({0: 1.0, 1: 2.0, 2: 3.0}).rename_axis('snapshot')}
    index = {
        'snapshot': pd.DataFrame({'snapshot': [0, 1, 2]}),
        'period': pd.DataFrame({'period': [1, 2]}),
        'period_of': pd.DataFrame({'snapshot': [0, 1, 2], 'period': [1, 1, 2]}),
    }
    with differential(_spec(), data | index) as run:
        assert run.oracle == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# where on a relation (#553)
# ---------------------------------------------------------------------------


#: A three-line network whose structure is entirely relations: each line has two
#: endpoints, and `spur` deliberately has an open end so the partial case is
#: reachable. `voltage` targets an int dimension and `send`/`recv` a string one,
#: so one model covers every relation predicate.
NETWORK = {
    'dimensions': {'bus': {'dtype': 'str'}, 'line': {'dtype': 'str'}, 'kv': {'dtype': 'int'}},
    'relations': {
        'send': {'key': 'line', 'values': 'bus'},
        'recv': {'key': 'line', 'values': 'bus'},
        'voltage': {'key': 'line', 'values': 'kv'},
    },
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
    'send': pl.DataFrame({'line': LINES, 'bus': SEND}),
    'recv': pl.DataFrame({'line': LINES[:3], 'bus': RECV}),
    'voltage': pl.DataFrame({'line': LINES, 'kv': VOLTAGE}),
    'cap': pl.DataFrame({'line': LINES, 'value': CAP}),
    'price': pl.DataFrame({'line': LINES, 'value': PRICE}),
}


@pytest.mark.parametrize(
    ('where', 'kept'),
    [
        pytest.param('voltage == 220', ['loop', 'ring_a'], id='a-relation-into-an-int-dimension'),
        pytest.param("send == 'north'", ['loop', 'ring_a', 'spur'], id='a-relation-against-a-label'),
        pytest.param('send != recv', ['ring_a', 'ring_b'], id='two-relations-over-one-dimension'),
        pytest.param('recv', ['loop', 'ring_a', 'ring_b'], id='a-bare-relation-is-the-partial-case'),
        pytest.param('NOT voltage == 220', ['ring_b', 'spur'], id='negated'),
        pytest.param('voltage == 380 AND send != recv', ['ring_b'], id='conjoined-with-a-pair-comparison'),
        pytest.param(
            "at(bus == 'north', by=send, over=bus, into=line)", ['loop', 'ring_a', 'spur'], id='a-predicate-read-at'
        ),
        pytest.param(
            "at(bus == 'north', by=recv, over=bus, into=line)", ['loop', 'ring_b'], id='read-at-a-partial-relation'
        ),
    ],
)
def test_a_where_reads_a_relation(where, kept):
    """The atom #553 asked for, in both shapes.

    `kept` is asserted rather than just a count: a predicate that inverted its
    sense would keep the complement, which is the same size on a symmetric
    case and a different model everywhere.

    `spur`'s null `recv` is the reading law 8 fixes — a comparison over it is
    false, so the bare name keeps exactly the lines that map.
    """
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': where}}}
    with sps.solve(spec, NETWORK_SOURCES) as result:
        built = sorted(row['line'] for row in result.primal('f').to_dicts())
    assert built == sorted(kept), f'where: {where!r} built the wrong set of variables'


@pytest.mark.parametrize(
    ('where', 'objective'),
    [
        pytest.param('send != recv', 30.0, id='the-two-ring-lines-survive'),
        pytest.param('NOT send != recv', 70.0, id='negated-over-a-partial-relation'),
        # the linopy lane's null exclusion: numpy answers `None != 'north'` with True
        pytest.param("recv != 'north'", 10.0, id='not-equal-over-a-null-value'),
        pytest.param('recv != send', 30.0, id='not-equal-between-two-relations'),
        pytest.param("at(bus == 'north', by=send, over=bus, into=line)", 80.0, id='a-predicate-read-at'),
        pytest.param(
            "NOT at(bus == 'south', by=recv, over=bus, into=line)", 90.0, id='negated-read-at-a-partial-relation'
        ),
        pytest.param(
            "at(bus == 'north', by=recv, over=bus, into=line) AND voltage == 220", 30.0, id='read-at-conjoined'
        ),
    ],
)
def test_a_relation_where_agrees_with_the_oracle(where, objective):
    """Both lanes, one answer — the differential half of #553.

    A mask reading a relation is a join on the dim table in the relational lane
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
        'send': pd.DataFrame({'line': LINES, 'bus': SEND}),
        'recv': pd.DataFrame({'line': LINES[:3], 'bus': RECV}),
        'voltage': pd.DataFrame({'line': LINES, 'kv': VOLTAGE}),
    }
    with differential(spec, data | index) as run:
        assert run.result.objective == pytest.approx(objective), (
            f'where: {where!r} — the two lanes agree on the objective but not on this one'
        )


def test_a_where_on_a_relation_outside_the_frame_is_refused():
    """A relation is read on the dim it maps out of, so that dim has to be in the
    frame — otherwise the mask would silently reduce over an unlisted dim."""
    spec = {
        **NETWORK,
        'variables': {'f': {'dims': ['line'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {
            'ceiling': {
                'dims': ['bus'],
                'where': 'voltage == 220',
                'expression': 'sum(f, by=send, over=line, into=bus) <= 100',
            }
        },
    }
    with pytest.raises(SpecsolveError, match=r"where-relation 'voltage' reads dims \['line'\] outside the frame"):
        to_spec(spec)


def test_two_relations_over_different_dims_cannot_be_compared():
    """There is no row carrying both, so the comparison has nothing to test.

    The pair comparison is legal exactly because two relations over one dim are
    two columns of one index. Drop that condition and this reads as a join
    nobody wrote.
    """
    spec = {
        **NETWORK,
        'dimensions': {**NETWORK['dimensions'], 'zone': {'dtype': 'str'}},
        'relations': {**NETWORK['relations'], 'zone_of': {'key': 'bus', 'values': 'zone'}},
        'variables': {'f': {**NETWORK['variables']['f'], 'where': 'send != zone_of'}},
    }
    with pytest.raises(SpecsolveError, match='over different dimensions'):
        to_spec(spec)


@pytest.mark.parametrize(
    ('extra', 'where'),
    [
        pytest.param({'area': {'key': 'line', 'values': 'zone'}}, 'send != area', id='two-targets-of-the-same-dtype'),
        pytest.param({}, 'send != voltage', id='two-targets-of-different-dtypes'),
    ],
)
def test_two_relations_into_different_label_sets_cannot_be_compared(extra, where):
    """One dimension is necessary but not sufficient — the targets must match too.

    A bus label is never a zone label, so the predicate could only mask
    everything out.
    """
    spec = {
        **NETWORK,
        'dimensions': {**NETWORK['dimensions'], 'zone': {'dtype': 'str'}},
        'relations': {**NETWORK['relations'], **extra},
        'variables': {'f': {**NETWORK['variables']['f'], 'where': where}},
    }
    with pytest.raises(SpecsolveError, match='over the same dimension'):
        to_spec(spec)


def test_a_relation_comparison_is_checked_against_its_dtype():
    """The same dtype check every other where-comparison gets (#460).

    A relation's literal is as silent to get wrong as a dimension's: `voltage`
    is an int, so a quoted right-hand side matches nothing rather than
    erroring at run time.
    """
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': "voltage == 'high'"}}}
    with pytest.raises(SpecsolveError, match=r"has dtype 'int'"):
        to_spec(spec)


def test_a_relation_compares_against_a_label_the_target_lacks():
    """A stranger label masks everything out; it does not raise.

    The relation column is compared as a string, because the target's `Enum`
    refuses a label outside it.
    """
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': "send == 'atlantis'"}}}
    with sps.build(spec, NETWORK_SOURCES) as model:
        surviving = model._engine._model.variables['f'].frame.select(pl.len()).collect().item()
    assert surviving == 0, "a label no bus carries matches nothing, so no 'f' is built"


def test_a_relation_orders_bytewise_not_by_declaration():
    """Labels order bytewise, whatever order the dimension declared them.

    Attaching casts a relation column to the target's `Enum`, which orders by
    *declaration*, so an ordering comparison read off it would answer a
    different question — and silently, since both readings return a mask.
    `south` is declared first here precisely so the two disagree.
    """
    sources = {**NETWORK_SOURCES, 'bus': pl.DataFrame({'bus': ['south', 'north']})}
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': "send >= 'south'"}}}
    with sps.solve(spec, sources) as result:
        built = sorted(row['line'] for row in result.primal('f').to_dicts())
    assert built == ['ring_b'], "only ring_b sends from 'south'; declaration order would keep the 'north' lines too"


# ---------------------------------------------------------------------------
# a relation's map, supplied under its own key
# ---------------------------------------------------------------------------


#: The model the section is written over: one relation, and the caller holding
#: both its labels and its relation. `g3` maps to no bus — the partial case,
#: spelled by the row that is not there.
GENERATORS = ['g1', 'g2', 'g3']
COST = [1.0, 2.0, 0.5]
LOAD = [5.0, 4.0]

BASE = {
    'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}},
    'relations': {'gen_bus': {'key': 'generator', 'values': 'bus'}},
    'parameters': {'cost': {'dims': ['generator']}, 'load': {'dims': ['bus']}},
    'variables': {'p': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'balance': {'dims': ['bus'], 'expression': 'sum(p, by=gen_bus, over=generator, into=bus) >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}

BASE_SOURCES = {
    'bus': ['north', 'south'],
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': COST}),
    'load': pl.DataFrame({'bus': ['north', 'south'], 'value': LOAD}),
}


_LABELS_AND_MAP = pl.DataFrame({'generator': GENERATORS, 'gen_bus': ['south', 'north', None]})


def test_a_map_is_not_a_column_of_the_index_it_runs_over():
    """The one stray column that is refused rather than filtered away.

    An index may carry anything — attributes, other frameworks' fields — and
    the extras are dropped. A column named after a relation over that dimension
    is not an extra: it is a map somebody meant to supply, and dropping it
    would build the model they did not write.
    """
    with pytest.raises(DataError, match=re.escape("index for dimension 'generator' carries a 'gen_bus' column")):
        sps.solve(BASE, {**BASE_SOURCES, 'gen_bus': _RELATION, 'generator': _LABELS_AND_MAP})


def test_a_map_alone_does_not_say_which_labels_exist():
    """A map is a relation over a dimension, never the dimension.

    It may omit members and its key order is whatever someone typed, so reading
    the label set out of it would let an added entry create a member and a
    reordered map re-order the axis that ``shift`` reads positionally. With
    nothing supplying ``generator``'s labels, the map over it leaves the
    dimension without an index, and both lanes say so.
    """
    with pytest.raises(DataError, match=re.escape("has its maps (sources['gen_bus'])")):
        sps.solve(BASE, {**BASE_SOURCES, 'gen_bus': _RELATION})


# ---------------------------------------------------------------------------
# A map supplied under the relation's own key
# ---------------------------------------------------------------------------


#: `gen_bus` takes its own source key, and the two columns say what it is — the
#: dimension it runs over, and the space its values are labels of. `g3` is
#: mapped nowhere by being in no row.
SUPPLIED = BASE

_RELATION = pl.DataFrame({'generator': ['g1', 'g2'], 'bus': ['north', 'south']})
_SUPPLIED_SOURCES = {**BASE_SOURCES, 'generator': GENERATORS, 'gen_bus': _RELATION}


def test_a_supplied_relation_reaches_the_declared_map_without_touching_the_index():
    """The point: a caller adds a map to a dimension whose index is not theirs.

    `g3` is the cheapest generator and contributes nothing, which is what an
    unmapped label means. The index goes in as a bare label list, so nothing
    here rewrites a table someone else generated.
    """
    with sps.solve(SUPPLIED, _SUPPLIED_SOURCES) as result:
        built = by_coord(result, 'p', 'generator')
        assert result.objective == pytest.approx(13.0)
    assert built['g3'] == pytest.approx(0.0), 'a generator in no row of the map is a generator on no bus'


def test_a_supplied_relation_agrees_with_the_oracle():
    """Both lanes read the relation through the one front door, so both see it."""
    from tests.differential import differential
    from tests.oracle import pd

    data = {
        'cost': pd.Series([1.0, 2.0, 0.5], index=['g1', 'g2', 'g3']),
        'load': pd.Series([5.0, 4.0], index=['north', 'south']),
        'bus': ['north', 'south'],
        'generator': ['g1', 'g2', 'g3'],
        'gen_bus': _RELATION,
    }
    with differential(SUPPLIED, data) as run:
        assert run.result.objective == pytest.approx(13.0)


def test_a_partial_map_is_supplied_as_the_rows_it_has():
    """Absence is the absent row here as everywhere else.

    The relation says nothing about `g3`, and a null in it is refused the way a
    null in a parameter's values is.
    """
    holed = pl.DataFrame({'generator': ['g1', 'g2', 'g3'], 'bus': ['north', 'south', None]})
    with pytest.raises(DataError, match=r"relation 'gen_bus' carries 1 row\(s\) with a null in 'bus'"):
        sps.solve(SUPPLIED, {**_SUPPLIED_SOURCES, 'gen_bus': holed})


def test_a_where_reads_a_map_that_leaves_a_label_out():
    """A label the map has no row for satisfies no comparison over it.

    The reading law for a partial map, at the one place it changes the model:
    the constraint is placed only where the map says something.
    """
    spec = _spec()
    spec['constraints']['c']['where'] = 'period_of == 1'
    sources = {
        'snapshot': [0, 1, 2],
        'period': _periods(),
        'load': _load(),
        'period_of': pl.DataFrame({'snapshot': [0, 1], 'period': [1, 1]}),
    }
    with sps.solve(spec, sources) as result:
        built = sorted(row['snapshot'] for row in result.primal('x').to_dicts())
    assert built == [0, 1, 2], 'the variable is unmasked; only the constraint reads the map'
    assert result.objective == pytest.approx(3.0), 'snapshot 2 is in no row of the map, so nothing constrains it'


@pytest.mark.parametrize(
    ('relation', 'match'),
    [
        pytest.param(
            pl.DataFrame({'generator': ['g1'], 'gen_bus': ['north']}),
            r"must carry a column per column it declares, \['generator', 'bus'\]",
            id='named-after-itself-not-its-target',
        ),
        pytest.param(
            pl.DataFrame({'bus': ['north']}),
            r"must carry a column per column it declares, \['generator', 'bus'\]",
            id='no-key-column',
        ),
        pytest.param(
            pl.DataFrame({'generator': ['g1', 'g1'], 'bus': ['north', 'south']}),
            r"maps 1 key\(s\) more than once: generator='g1'",
            id='mapped-twice',
        ),
        pytest.param(
            pl.DataFrame({'generator': ['g9'], 'bus': ['north']}),
            r"relation 'gen_bus' has value\(s\) in 'generator' that are not 'generator' labels: 'g9'",
            id='key-is-not-a-label',
        ),
    ],
)
def test_a_supplied_relation_is_held_to_what_a_map_is(relation, match):
    """Single-valued, keyed by labels that exist, and spelled as the pair it is.

    Dropping a stray key instead of refusing it would place that generator's
    terms nowhere while the model built and solved.
    """
    with pytest.raises(DataError, match=match):
        sps.solve(SUPPLIED, {**_SUPPLIED_SOURCES, 'gen_bus': relation})


def test_a_map_with_no_author_at_all_is_refused():
    """Neither is not a spelling of empty: a relation nothing supplies is missing data."""
    with pytest.raises(DataError, match="no data provided for relation 'gen_bus'"):
        sps.solve(SUPPLIED, {**BASE_SOURCES, 'generator': GENERATORS})


def test_a_supplied_map_does_not_say_which_labels_exist():
    """A relation over a dimension is not the dimension, whoever holds it.

    Reading the label set out of the map would let an omitted row delete a
    member, and `g3` — mapped nowhere and still a generator — is exactly the
    row that would vanish.
    """
    with pytest.raises(DataError, match=re.escape("has its maps (sources['gen_bus'])")):
        sps.solve(SUPPLIED, {**BASE_SOURCES, 'gen_bus': _RELATION})


@pytest.mark.parametrize('lane', ['relational', 'linopy'])
def test_a_supplied_relation_is_refused_the_same_way_on_both_lanes(lane):
    """One defect, one sentence: the checks live in the door both lanes enter."""
    from tests.oracle import specsolve_linopy

    build = sps.solve if lane == 'relational' else specsolve_linopy.build
    with pytest.raises(DataError, match=r'maps 1 key\(s\) more than once'):
        build(
            SUPPLIED,
            {**_SUPPLIED_SOURCES, 'gen_bus': pl.DataFrame({'generator': ['g1', 'g1'], 'bus': ['north', 'south']})},
        )


#: Two maps out of one dimension into the same target — the PyPSA shape, where
#: a line has a sending and a receiving bus. Every check over a dimension's maps
#: has to run per map.
TWO_MAPS = {
    'dimensions': {'line': {}, 'bus': {'dtype': 'str'}},
    'relations': {
        'line_from': {'key': 'line', 'values': 'bus'},
        'line_to': {'key': 'line', 'values': 'bus'},
    },
    'parameters': {'flow_max': {'dims': ['line']}, 'load': {'dims': ['bus']}},
    'variables': {'f': {'dims': ['line'], 'bounds': {'lower': 0, 'upper': 'flow_max'}}},
    'constraints': {'served': {'dims': ['bus'], 'expression': 'sum(f, by=line_to, over=line, into=bus) >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(f, over=line)'},
}

_TWO_MAP_SOURCES = {
    'line': ['l1', 'l2'],
    'bus': ['north', 'south'],
    'flow_max': pl.DataFrame({'line': ['l1', 'l2'], 'value': [10.0, 10.0]}),
    'load': pl.DataFrame({'bus': ['north', 'south'], 'value': [1.0, 2.0]}),
    'line_from': pl.DataFrame({'line': ['l1', 'l2'], 'bus': ['south', 'north']}),
    'line_to': pl.DataFrame({'line': ['l1', 'l2'], 'bus': ['north', 'south']}),
}


def test_two_maps_into_one_target_each_take_their_own_key():
    """Two relations of identical schema, told apart by the key they arrive under."""
    with sps.solve(TWO_MAPS, _TWO_MAP_SOURCES) as result:
        assert result.objective == pytest.approx(3.0), 'each line serves the bus line_to sends it to'


def test_the_second_map_is_checked_as_hard_as_the_first():
    """Per-map, not per-dimension: the check runs for `line_from` as for `line_to`."""
    index = pl.DataFrame({'line': ['l1', 'l2'], 'line_from': ['south', 'north']})
    with pytest.raises(DataError, match=re.escape("carries a 'line_from' column")):
        sps.solve(TWO_MAPS, {**_TWO_MAP_SOURCES, 'line': index})


#: A parameter written positionally over a dimension that carries a supplied
#: map. `cost` is a bare list, so which label each number belongs to is the
#: index's row order — the order a map joined onto it must not disturb. `cap`
#: pins the solution to `t = 0` alone, so the objective *is* that label's cost.
POSITIONAL = {
    'dimensions': {'t': {'dtype': 'int'}, 'g': {'dtype': 'str'}},
    'relations': {'g_of': {'key': 't', 'values': 'g'}},
    'parameters': {'cost': {'dims': ['t']}, 'cap': {'dims': ['t']}},
    'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'c': {'dims': ['t'], 'expression': 'x >= cap'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x * cost, over=t)'},
}


def test_a_supplied_map_does_not_reorder_the_index_it_joins_onto():
    """A label's position is its ordinal, and joining a map on may not move it.

    A positional shape is placed against the labels read back off the index
    *after* the map has been joined onto it, so a join free to reorder hands
    every one of these numbers to the wrong label. Both lanes would agree on
    that model, so the check is a number rather than a comparison between them.
    """
    sources = {
        't': pl.DataFrame({'t': [0, 1, 2]}),
        'g': ['n', 's'],
        'g_of': pl.DataFrame({'t': [0, 1, 2], 'g': ['n', 'n', 's']}),
        'cost': [1.0, 10.0, 100.0],
        'cap': pl.DataFrame({'t': [0, 1, 2], 'value': [1.0, 0.0, 0.0]}),
    }
    with sps.solve(POSITIONAL, sources) as result:
        assert result.objective == pytest.approx(1.0), "the first number is the first label's, whatever the join did"


# ---------------------------------------------------------------------------
# the relation shapes the relational lane builds
# ---------------------------------------------------------------------------


def _walked(relations: dict, expression: str) -> dict:
    """A model whose one constraint walks *relations*, with both ends named."""
    return {
        'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}, 'period': {'dtype': 'int'}},
        'relations': relations,
        'parameters': {'load': {'dims': ['bus']}},
        'variables': {'p': {'dims': ['generator', 'period'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'bal': {'dims': ['bus', 'period'], 'expression': expression}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }


@pytest.mark.parametrize(
    ('relations', 'expression'),
    [
        pytest.param(
            {'gen_bus': {'key': 'generator', 'values': 'bus'}},
            'sum(p, by=gen_bus, over=generator, into=bus) >= load',
            id='a-map-keyed-by-one-column',
        ),
        pytest.param(
            {'gen_bus': {'key': ['generator', 'bus']}},
            'sum(p, by=gen_bus, over=generator, into=bus) >= load',
            id='a-bare-relation',
        ),
        pytest.param(
            {'zone_of': {'key': ['generator', 'period'], 'values': 'bus'}},
            'sum(p, by=zone_of, over=generator, into=bus) >= load',
            id='a-map-keyed-by-two-columns',
        ),
    ],
)
def test_every_relation_shape_the_language_admits_passes_check(relations: dict, expression: str) -> None:
    """The language's relation is any table walked any way, and `check` refuses none of them.

    What each shape builds to is `test_relation_shapes.py`'s subject.
    """
    program = sps.check(_walked(relations, expression))
    assert set(program.relations) == set(relations), 'every relation the model declares, under its own name'
