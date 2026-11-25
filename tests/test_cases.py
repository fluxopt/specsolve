"""`cases:` — one quantity, a value per region, on both lanes.

Each lane builds a region against its own condition and adds the results. The tests
hold three things:

* a region's value reaches the coordinates it claims, and only those;
* a region's **absence** stays inside it — an ``otherwise`` that shifts with no
  ``edge=`` has nothing at the first position, and must not unmake a row the
  other regions do cover;
* a region reads its data **where that region applies**, so rows outside it
  change nothing, and a missing row inside it takes the parameter's declared
  ``missing:`` value there.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml
from mathspec.errors import SchemaError

import specsolve as sps
from tests.differential import RTOL, differential
from tests.oracle import specsolve_linopy

CAPPED_BY_REGION = {
    'dimensions': {'t': {'dtype': 'int', 'ordered': True}},
    'parameters': {
        'flag': {'dims': ['t'], 'dtype': 'bool', 'missing': False},
        'hi': {'dims': ['t'], 'missing': 0},
        'cost': {'dims': ['t']},
    },
    'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 1000}}},
    'expressions': {
        'cap': {
            'dims': ['t'],
            'cases': {'flagged': {'when': 'flag', 'expression': 'hi'}},
            'otherwise': 5,
        }
    },
    'constraints': {'under_cap': {'dims': ['t'], 'expression': 'x <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost)'},
}

#: `flag` holds at 0 and 2, so `hi` carries those and `otherwise`'s 5 the rest.
CAPPED_SOURCES = {
    't': [0, 1, 2, 3],
    'flag': {'t': [0, 1, 2, 3], 'value': [True, False, True, False]},
    'hi': {'t': [0, 2], 'value': [40.0, 60.0]},
    'cost': {'t': [0, 1, 2, 3], 'value': [1.0, 1.0, 1.0, 1.0]},
}


def _tables(sources):
    """A dimension's labels as a Series and every parameter as a table, the shape both lanes take."""
    import polars as pl

    built = {}
    for name, value in sources.items():
        if not isinstance(value, list):
            built[name] = pl.DataFrame(value)
            continue
        dtype = pl.Int64 if isinstance(value[0], int) else pl.String
        built[name] = pl.Series(name, value, dtype=dtype)
    return built


def test_each_region_carries_the_coordinates_it_claims():
    with differential(CAPPED_BY_REGION, _tables(CAPPED_SOURCES), lp=True, outputs=frozenset({'activity'})) as run:
        assert run.oracle == pytest.approx(110.0, rel=RTOL), 'the flagged steps cap at 40 and 60, the rest at 5'
        caps = run.result.activity('under_cap').sort('t')
        assert list(caps.get_column('value')) == [40.0, 5.0, 60.0, 5.0], (
            'a region reaches its own coordinates and the otherwise carries the rest, in t order'
        )


@pytest.mark.parametrize(
    ('hi', 'objective'),
    [
        pytest.param({'t': [0, 2], 'value': [40.0, 60.0]}, 110.0, id='every flagged step has a cap'),
        pytest.param(
            {'t': [0, 1, 2, 3], 'value': [40.0, 9.0, 60.0, 9.0]}, 110.0, id='a cap outside the region is spare'
        ),
    ],
)
def test_a_constant_side_is_asked_for_data_only_where_its_region_applies(hi, objective):
    """`hi` is the flagged region's cap, so it answers for that region and is never asked about the rest."""
    with differential(CAPPED_BY_REGION, _tables(CAPPED_SOURCES | {'hi': hi})) as run:
        assert run.oracle == pytest.approx(objective, rel=RTOL), 'rows outside the region change nothing'


def test_a_missing_row_inside_the_region_reads_as_zero():
    """`hi` is `missing: 0`, so the flagged step it has no row at caps at 0, on both lanes."""
    with differential(CAPPED_BY_REGION, _tables(CAPPED_SOURCES | {'hi': {'t': [0], 'value': [40.0]}})) as run:
        assert run.oracle == pytest.approx(40.0 + 5.0 + 0.0 + 5.0, rel=RTOL), (
            't=2 is flagged and hi has no row there, so it caps at 0'
        )


@pytest.mark.parametrize(
    ('flagged', 'objective', 'reads'),
    [
        pytest.param(
            True, 130.0, 'the region takes every coordinate, so hi caps every step', id='a region that is everywhere'
        ),
        pytest.param(
            False, 20.0, 'the region takes nothing, so the otherwise 5 caps every step', id='a region that is nowhere'
        ),
    ],
)
def test_a_region_whose_condition_reads_no_dimension(flagged, objective, reads):
    """A scalar `when` names no coordinate set to cut the region down by, so it cuts by its own constant."""
    spec = CAPPED_BY_REGION | {
        'parameters': CAPPED_BY_REGION['parameters'] | {'flag_all': {'dims': [], 'dtype': 'bool'}},
        'expressions': {
            'cap': {'dims': ['t'], 'cases': {'flagged': {'when': 'flag_all', 'expression': 'hi'}}, 'otherwise': 5}
        },
    }
    sources = _tables(
        CAPPED_SOURCES
        | {'hi': {'t': [0, 1, 2, 3], 'value': [40.0, 10.0, 60.0, 20.0]}, 'flag_all': {'value': [flagged]}}
    )
    with differential(spec, sources) as run:
        assert run.oracle == pytest.approx(objective, rel=RTOL), reads


CARRIED_IN = {
    'dimensions': {'t': {'dtype': 'int', 'ordered': True}, 'g': {}},
    'parameters': {
        'switchable': {'dims': ['g'], 'dtype': 'bool', 'missing': False},
        'before': {'dims': ['g']},
        'cap': {'dims': ['g']},
        'step': {'dims': ['g']},
        'first_step': {'dims': ['g']},
        'load': {'dims': ['t']},
        'cost': {'dims': ['g']},
    },
    'variables': {
        'p': {'dims': ['t', 'g'], 'bounds': {'lower': 0, 'upper': 'cap'}},
        'on': {'dims': ['t', 'g'], 'domain': 'binary'},
    },
    'expressions': {
        'carried': {
            'dims': ['t', 'g'],
            'cases': {
                'never_off': {'when': 'not switchable', 'expression': 1},
                'boundary': {'when': 'switchable and position(t) == 0', 'expression': 'before'},
            },
            # no `edge=`, so this region has nothing at t == 0 - which no region claims it at
            'otherwise': 'shift(on, along=t, offset=1)',
        }
    },
    'constraints': {
        'meet_load': {'dims': ['t'], 'expression': 'sum(p, over=g) == load'},
        'runs_only_when_on': {'dims': ['t', 'g'], 'expression': 'p <= on * cap'},
        'ramp': {
            'dims': ['t', 'g'],
            'expression': 'p - shift(p, along=t, offset=1, edge=0) <= step * carried + first_step * (1 - carried)',
        },
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}


def _carried_sources(switchable, before):
    return _tables(
        {
            't': [0, 1, 2, 3],
            'g': ['base', 'peak'],
            'switchable': {'g': ['base', 'peak'], 'value': switchable},
            'before': {'g': ['base', 'peak'], 'value': before},
            'cap': {'g': ['base', 'peak'], 'value': [80.0, 60.0]},
            'step': {'g': ['base', 'peak'], 'value': [40.0, 35.0]},
            'first_step': {'g': ['base', 'peak'], 'value': [70.0, 55.0]},
            'cost': {'g': ['base', 'peak'], 'value': [12.0, 45.0]},
            'load': {'t': [0, 1, 2, 3], 'value': [90.0, 60.0, 60.0, 60.0]},
        }
    )


def test_a_region_that_claims_no_coordinate_does_not_unmake_the_row():
    """The `otherwise` shifts with no `edge=`, so it is empty at t == 0 — where no region claims it.

    ``never_off`` and ``boundary`` between them carry every unit at the first
    position, so the row stays.
    """
    with differential(CARRIED_IN, _carried_sources([False, True], [1.0, 0.0]), outputs=frozenset({'activity'})) as run:
        rows = run.result.activity('ramp')
        assert rows.height == 8, 'every (t, g) coordinate has a ramp row, the first position included'
        assert sorted(set(rows.get_column('t'))) == [0, 1, 2, 3], 't == 0 is built like any other position'
        assert int((run.model.constraints['ramp'].labels == -1).sum()) == 0, (
            'the linopy lane leaves no ramp row absent either'
        )


@pytest.mark.parametrize(
    ('switchable', 'before', 'objective', 'reads'),
    [
        pytest.param(
            [False, True],
            [1.0, 0.0],
            4890.0,
            'boundary: peak was off before the first snapshot, so it may start at first_step 55',
            id='peak starts cold',
        ),
        pytest.param(
            [True, True],
            [0.0, 0.0],
            3900.0,
            'never_off no longer claims base, so boundary gives it first_step 70 instead of step 40',
            id='base becomes switchable',
        ),
    ],
)
def test_a_region_is_read_and_its_own_data_decides_the_answer(switchable, before, objective, reads):
    """Vary the data each region reads and the answer moves — which is the only proof the region is built."""
    with differential(CARRIED_IN, _carried_sources(switchable, before)) as run:
        assert run.oracle == pytest.approx(objective, rel=RTOL), reads


def test_a_region_binding_tighter_makes_the_model_infeasible_on_both_lanes():
    """`boundary` reading a unit that was already on holds it to `step`, and the load can no longer be met.

    Asserted lane by lane: ``differential`` needs a finite objective.
    """
    sources = _carried_sources([False, True], [1.0, 1.0])
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / 'carried.yaml'
        path.write_text(yaml.safe_dump(CARRIED_IN))

        relational = sps.solve(path, sources, solver_name='highs').objective
        linopy_lane = specsolve_linopy.build(path, dict(sources))
        linopy_lane.solve(solver_name='highs', output_flag=False)

    assert relational != relational, 'the relational lane reports no objective — peak is held to step 35'
    assert linopy_lane.objective.value != linopy_lane.objective.value, (
        'and the linopy lane reaches the same infeasibility'
    )


def test_a_region_that_claims_nothing_does_not_unmake_the_row():
    """The complement of a condition reading no dimension claims nothing, and so may restrict nothing.

    With a true scalar condition the ``otherwise`` claims nothing, while its
    ``shift`` with no ``edge=`` is still absent at the first position.
    """
    spec = CARRIED_IN | {
        'parameters': CARRIED_IN['parameters'] | {'everywhere': {'dims': [], 'dtype': 'bool'}},
        'expressions': {
            'carried': {
                'dims': ['t', 'g'],
                'cases': {'always': {'when': 'everywhere', 'expression': 1}},
                'otherwise': 'shift(on, along=t, offset=1)',
            }
        },
    }
    sources = _carried_sources([False, True], [1.0, 0.0]) | _tables({'everywhere': {'value': [True]}})
    sources['load'] = _tables({'load': {'t': [0, 1, 2, 3], 'value': [70.0, 60.0, 60.0, 60.0]}})['load']
    with differential(spec, sources, outputs=frozenset({'activity'})) as run:
        rows = run.result.activity('ramp')
        assert rows.height == 8, 'every (t, g) coordinate has a ramp row, the first position included'
        assert int((run.model.constraints['ramp'].labels == -1).sum()) == 0, (
            'the linopy lane leaves no ramp row absent either'
        )
        assert run.oracle == pytest.approx(3990.0, rel=RTOL), (
            'carried is 1 everywhere, so the first position is held to step rather than first_step'
        )


def test_one_parameter_answering_for_two_regions():
    """`hi` caps the flagged steps and, doubled, the rest — which is one name owed two answers."""
    spec = CAPPED_BY_REGION | {
        'expressions': {
            'cap': {
                'dims': ['t'],
                'cases': {
                    'flagged': {'when': 'flag', 'expression': 'hi'},
                    'unflagged': {'when': 'not flag', 'expression': 'hi * 2'},
                },
                'otherwise': 0,
            }
        },
    }
    sources = _tables(CAPPED_SOURCES | {'hi': {'t': [0, 1, 2, 3], 'value': [40.0, 10.0, 60.0, 20.0]}})
    with differential(spec, sources) as run:
        assert run.oracle == pytest.approx(160.0, rel=RTOL), 'the flagged steps read hi and the rest read twice it'


def _divided_by_region(rate):
    """`CAPPED_BY_REGION` with the flagged cap divided by a `rate` declared *rate*."""
    return CAPPED_BY_REGION | {
        'expressions': {
            'cap': {
                'dims': ['t'],
                'cases': {'flagged': {'when': 'flag', 'expression': 'hi / rate'}},
                'otherwise': 5,
            },
        },
        'parameters': CAPPED_BY_REGION['parameters'] | {'rate': rate},
    }


@pytest.mark.parametrize(
    ('rate', 'objective', 'reads'),
    [
        pytest.param(
            {'t': [0, 2], 'value': [2.0, 2.0]},
            20.0 + 5.0 + 30.0 + 5.0,
            'the flagged steps cap at 40/2 and 60/2',
            id='every flagged step has a rate',
        ),
        pytest.param(
            {'t': [0], 'value': [2.0]},
            20.0 + 5.0 + 60.0 + 5.0,
            't=2 has no rate, so it divides by the declared 1',
            id='a flagged step without a rate',
        ),
    ],
)
def test_a_divisor_inside_a_region_reads_its_declared_value_where_it_has_no_row(rate, objective, reads):
    """`rate` is `missing: 1`, so the steps outside the region owe it nothing and a flagged step without a row divides by 1."""
    spec = _divided_by_region({'dims': ['t'], 'missing': 1})
    with differential(spec, _tables(CAPPED_SOURCES | {'rate': rate})) as run:
        assert run.oracle == pytest.approx(objective, rel=RTOL), reads


def test_a_divisor_inside_a_region_that_a_missing_row_makes_zero_is_refused_at_load():
    """`missing: 0` under a divisor is refused by mathspec before any data arrives, region or not."""
    with pytest.raises(SchemaError, match="the divisor contains 'rate', which is declared missing: 0"):
        sps.build(_divided_by_region({'dims': ['t'], 'missing': 0}), _tables(CAPPED_SOURCES))


#: `cap` is a sum over `g` inside the flagged region: `hi` owes two coordinates
#: per flagged step, and none for the steps the `otherwise` carries.
SUMMED_BY_REGION = CAPPED_BY_REGION | {
    'dimensions': CAPPED_BY_REGION['dimensions'] | {'g': {'dtype': 'str'}},
    'parameters': CAPPED_BY_REGION['parameters'] | {'hi': {'dims': ['t', 'g'], 'missing': 0}},
    'expressions': {
        'cap': {
            'dims': ['t'],
            'cases': {'flagged': {'when': 'flag', 'expression': 'sum(hi, over=g)'}},
            'otherwise': 5,
        }
    },
}

#: The flagged steps are 0 and 2, and both are covered at both `g`.
SUMMED_SOURCES = CAPPED_SOURCES | {
    'g': ['u', 'v'],
    'hi': {'t': [0, 0, 2, 2], 'g': ['u', 'v', 'u', 'v'], 'value': [30.0, 10.0, 30.0, 30.0]},
}


def test_a_region_narrows_what_a_summed_constant_side_owes():
    """Data for the steps a region claims, at every coordinate the sum reads — and no more."""
    with differential(SUMMED_BY_REGION, _tables(SUMMED_SOURCES)) as run:
        assert run.oracle == pytest.approx(110.0, rel=RTOL), 'the flagged steps cap at 30+10 and 30+30, the rest at 5'


#: The whole cased quantity under the sum, rather than a sum inside a region:
#: `bound` is cased over `(k, t)` and the row reads its total over `t`.
SUMMED_CASES = {
    'dimensions': {'k': {'dtype': 'str'}, 't': {'dtype': 'int', 'ordered': True}},
    'parameters': {'hi': {'dims': ['k', 't']}, 'lo': {'dims': ['k', 't']}},
    'variables': {'x': {'dims': ['k'], 'bounds': {'lower': 0, 'upper': 100}}},
    'expressions': {
        'bound': {'dims': ['k', 't'], 'cases': {'a': {'when': "k == 'a'", 'expression': 'hi'}}, 'otherwise': 'lo'},
    },
    'constraints': {'cap': {'dims': ['k'], 'expression': 'x <= sum(bound, over=t)'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x, over=k)'},
}

#: Both parameters dense over the whole product, so nothing is owed anywhere.
SUMMED_CASES_SOURCES = {
    'k': ['a', 'b'],
    't': [0, 1],
    'hi': {'k': ['a', 'a', 'b', 'b'], 't': [0, 1, 0, 1], 'value': [5.0, 6.0, 7.0, 8.0]},
    'lo': {'k': ['a', 'a', 'b', 'b'], 't': [0, 1, 0, 1], 'value': [1.0, 2.0, 3.0, 4.0]},
}


def test_a_cased_quantity_summed_onto_a_constant_side_builds():
    """A region's claim survives the reduction that keeps the dimensions it reads.

    After the sum over `t` the `a` term has no row at `k = b` and the
    `otherwise` term none at `k = a`; neither is a hole.
    """
    with differential(SUMMED_CASES, _tables(SUMMED_CASES_SOURCES)) as run:
        assert run.oracle == pytest.approx(18.0, rel=RTOL), 'k=a caps at 5+6 and k=b at 3+4'


def test_a_hole_the_summed_region_reads_is_one_term_fewer():
    """A missing row under the sum inside the region adds 0 (#1465).

    Step 2 loses `v`'s 30, so the flagged steps cap at 30+10 and 30, the rest at 5.
    """
    holed = {'t': [0, 0, 2], 'g': ['u', 'v', 'u'], 'value': [30.0, 10.0, 30.0]}
    with differential(SUMMED_BY_REGION, _tables(SUMMED_SOURCES | {'hi': holed})) as run:
        assert run.oracle == pytest.approx(80.0, rel=RTOL), 'the flagged steps cap at 40 and 30, the rest at 5'


#: The flow an outage takes off its line: the line's flow where it stands,
#: nothing where it does not. Every line stands, so the `otherwise` region is
#: empty, and the pullback through `outage_line` cannot carry its claim.
OUTAGE_FLOW = {
    'dimensions': {'line': {'dtype': 'str'}, 'outage': {'dtype': 'str'}},
    'relations': {'outage_line': {'key': 'outage', 'values': 'line'}},
    'parameters': {
        'active': {'dims': ['line'], 'dtype': 'bool', 'missing': False},
        'share': {'dims': ['line', 'outage'], 'missing': 0},
        'cap': {'dims': ['line']},
    },
    'variables': {'s': {'dims': ['line'], 'bounds': {'lower': 0, 'upper': 100}}},
    'expressions': {
        'monitored': {'dims': ['line'], 'cases': {'standing': {'when': 'active', 'expression': 's'}}, 'otherwise': 0},
        'outage_s': {'dims': ['outage'], 'expression': 'at(monitored, by=outage_line[line])'},
    },
    'constraints': {
        'after': {'dims': ['line', 'outage'], 'expression': 'monitored + share * outage_s <= cap'},
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(s, over=line)'},
}


def test_a_region_worth_zero_read_through_a_pullback_owes_nothing():
    """`otherwise: 0` adds nothing anywhere, so no row is short of it.

    The zero was a constant term claimed by the region where a line does not
    stand. The pullback dropped that claim, so the term was owed at every row
    and, with every line standing, had none: each row was refused as short of
    `share` and `cap`, which are short of nothing. `a` going out puts half its flow on `b`, so `s_b + s_a / 2 <= 10` and
    `s_a <= 10` with `s_b <= 10` alone; the most is `s_a = 10, s_b = 5`.
    """
    sources = {
        'line': ['a', 'b'],
        'outage': ['out_a'],
        'outage_line': {'outage': ['out_a'], 'line': ['a']},
        'active': {'line': ['a', 'b'], 'value': [True, True]},
        'share': {'line': ['a', 'b'], 'outage': ['out_a', 'out_a'], 'value': [0.0, 0.5]},
        'cap': {'line': ['a', 'b'], 'value': [10.0, 10.0]},
    }
    with differential(OUTAGE_FLOW, _tables(sources)) as run:
        assert float(run.result.objective) == pytest.approx(15.0, rel=RTOL), 's_a at 10 leaves s_b 5'


def test_a_read_through_a_cased_divisor_whose_otherwise_is_missing_zero_is_refused_at_load():
    """`b` answers where `a` is not positive and is `missing: 0`, so its missing row at 3 would make `v / e` divide by 0.

    mathspec refuses the read at load, before it looks at which rows `b` has.
    """
    spec = {
        'dimensions': {'h': {'dtype': 'int'}},
        'parameters': {'a': {'dims': ['h']}, 'b': {'dims': ['h'], 'missing': 0}},
        'variables': {'v': {'dims': ['h'], 'bounds': {'lower': 0, 'upper': 1}}},
        'expressions': {
            'e': {'dims': ['h'], 'cases': {'flagged': {'when': 'a > 0', 'expression': 'a'}}, 'otherwise': 'b'}
        },
        'objective': {'sense': 'maximize', 'expression': 'sum(v)'},
    }
    sources = _tables(
        {'h': [1, 2, 3], 'a': {'h': [1, 2, 3], 'value': [2.0, -1.0, -1.0]}, 'b': {'h': [2], 'value': [4.0]}}
    )
    with (
        sps.solve(spec, sources) as result,
        pytest.raises(SchemaError, match="the divisor contains 'b', which is declared missing: 0"),
    ):
        result.evaluate('v / e')


#: `delayed` claims a unit only where its `delay` is positive, as PyPSA's links
#: read a port's flow an earlier snapshot only where the port has a delay.
DELAYED = {
    'dimensions': {'t': {'dtype': 'int', 'ordered': True}, 'g': {}},
    'parameters': {'delay': {'dims': ['g'], 'dtype': 'int'}},
    'variables': {'p': {'dims': ['t', 'g'], 'bounds': {'lower': 0, 'upper': 1}}},
    'expressions': {
        'arrival': {
            'dims': ['t', 'g'],
            'cases': {'delayed': {'when': 'delay > 0', 'expression': 'shift(p, along=t, offset=1, edge=0)'}},
            'otherwise': 'p',
        }
    },
    'constraints': {'held': {'dims': ['t', 'g'], 'expression': 'arrival <= 0.5'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(p)'},
}


@pytest.mark.parametrize(
    ('delay', 'terms', 'objective', 'reads'),
    [
        pytest.param([0, 0], 1, 4.0, 'every p is held to 0.5', id='a-region-that-claims-no-unit-builds-no-term'),
        pytest.param(
            [1, 0],
            2,
            4.5,
            'a reads p one step late, so its last p is free up to 1',
            id='a-region-that-claims-a-unit-builds-its-term',
        ),
    ],
)
def test_a_region_builds_a_term_only_where_it_claims_a_coordinate(monkeypatch, delay, terms, objective, reads):
    """PyPSA's spec writes a delayed and a curved arrival for every link port, and most networks claim neither.

    Each term of a region carries presences over the full product of its dimensions,
    so a region claiming nothing still read that product to cut it away.
    """
    from specsolve._engine import queries as queries_module

    built = []
    expression = queries_module.QueryBuilder.expression

    def spied(self, e, what, **kwargs):
        terms = expression(self, e, what, **kwargs)
        if what == "constraint 'held' lhs":
            built.append(len(terms.linear))
        return terms

    monkeypatch.setattr(queries_module.QueryBuilder, 'expression', spied)
    sources = _tables({'t': [0, 1, 2, 3], 'g': ['a', 'b'], 'delay': {'g': ['a', 'b'], 'value': delay}})
    with differential(DELAYED, sources) as run:
        assert run.oracle == pytest.approx(objective, rel=RTOL), reads
    assert built == [terms], f'`held` builds {terms} term(s): one per region that claims a unit'


@pytest.mark.parametrize(
    'solved',
    [
        pytest.param(True, id='an-expression-read-from-a-solve'),
        pytest.param(False, id='a-calculation-evaluated-without-a-solve'),
    ],
)
def test_a_read_keeps_the_dimension_only_a_region_that_claims_nothing_spans(solved):
    """`flat` claims every step, so `otherwise` claims none, and `otherwise` is the one region over `g`.

    A build places each term on a row by its own dimensions, so a region that claims
    nothing can be left out of it. A read takes its columns from the terms it
    is given, and with `otherwise` left out it came back over `h` alone, one
    value where the expression declares two.
    """
    spec = {
        'dimensions': {'g': {}, 'h': {'dtype': 'int', 'ordered': True}},
        'parameters': {'on': {'dims': ['h'], 'dtype': 'bool'}, 'w': {'dims': ['h', 'g']}},
        'expressions': {
            'e': {
                'dims': ['g', 'h'],
                'cases': {'flat': {'when': 'on', 'expression': '0.5'}},
                'otherwise': 'v' if solved else 'w',
            }
        },
    }
    if solved:
        spec['variables'] = {'v': {'dims': ['h', 'g'], 'bounds': {'lower': 0, 'upper': 5}}}
        spec['objective'] = {'sense': 'maximize', 'expression': 'sum(v)'}
    sources = _tables(
        {
            'g': ['a', 'b'],
            'h': [7],
            'on': {'h': [7], 'value': [True]},
            'w': {'h': [7, 7], 'g': ['a', 'b'], 'value': [1.0, 2.0]},
        }
    )
    read = sps.solve(spec, sources).evaluate('e') if solved else sps.evaluate(spec, sources, 'e')
    assert read.columns == ['g', 'h', 'value'], '`e` is declared over `g` and `h`, whichever region claims a coordinate'
    assert read.sort('g').get_column('value').to_list() == [0.5, 0.5], '`flat` is worth 0.5 at each of the two units'


@pytest.mark.parametrize(
    ('delay', 'objective', 'reads'),
    [
        pytest.param([0, 0], 4.0, 'no unit is delayed, so every p is held to 0.5', id='settled-false'),
        pytest.param(
            [1, 1], 5.0, 'every unit reads p one step late, so each last p is free up to 1', id='settled-true'
        ),
    ],
)
def test_a_region_the_data_settles_is_built_without_reading_its_condition(monkeypatch, delay, objective, reads):
    """`delay > 0` holds at no unit, or at every unit, and either is read once for the build rather than per region."""
    from specsolve._engine import queries as queries_module

    read, real = [], queries_module.narrowed

    def spied(scope, dims, where):
        read.append(tuple(dims))
        return real(scope, dims, where)

    monkeypatch.setattr(queries_module, 'narrowed', spied)
    sources = _tables({'t': [0, 1, 2, 3], 'g': ['a', 'b'], 'delay': {'g': ['a', 'b'], 'value': delay}})
    with differential(DELAYED, sources) as run:
        assert run.oracle == pytest.approx(objective, rel=RTOL), reads
    assert not read, f'a region whose condition the data settles was read over {read}'


@pytest.mark.parametrize('delay', [pytest.param([0, 0], id='settled-false'), pytest.param([1, 1], id='settled-true')])
def test_a_region_the_data_settles_is_written_out_of_the_program_a_build_reads(monkeypatch, delay):
    """One region of `arrival` settles false and the other true, so `specialized` leaves no `cases:` to build.

    The false one leaves, and the true one, left alone, stands for the `cases:`. Either kept builds the same model
    over a filter no query needs, which only this sees.
    """
    from mathspec import program

    from specsolve._engine import assembly as assembly_module

    programs, real = [], assembly_module.Assembly.__init__

    def spied(self, compiled, *args):
        programs.append(compiled)
        real(self, compiled, *args)

    monkeypatch.setattr(assembly_module.Assembly, '__init__', spied)
    sources = _tables({'t': [0, 1, 2, 3], 'g': ['a', 'b'], 'delay': {'g': ['a', 'b'], 'value': delay}})
    sps.build(DELAYED, sources).close()
    held = programs[0].constraints['held']
    left = [e for e in program.walk(held.lhs, held.rhs) if isinstance(e, program.Cases)]
    assert not left, f'the build reads {left}, where the data settled every region of it'
