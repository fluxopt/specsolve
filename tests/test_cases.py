"""`cases:` — one quantity, a value per region, on both lanes.

Each lane builds a region against its own mask and adds the results. The tests
hold three things:

* a region's value reaches the coordinates it claims, and only those;
* a region's **absence** stays inside it — an ``otherwise`` that shifts with no
  ``edge=`` has nothing at the first position, and must not unmake a row the
  other regions do cover;
* a region's data is required **where that region applies**, so a parameter
  standing for one region is not asked to cover the whole frame, and a hole
  inside the region it does stand for is still refused.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

import specsolve as sps
from specsolve.errors import DataError
from tests.differential import RTOL, both_lanes_refuse, differential
from tests.oracle import specsolve_linopy

CAPPED_BY_REGION = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {
        'flag': {'dims': ['t'], 'dtype': 'bool'},
        'hi': {'dims': ['t']},
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


def _frames(sources):
    """A dimension's labels as a Series and every parameter as a frame, the shape both lanes take."""
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
    with differential(CAPPED_BY_REGION, _frames(CAPPED_SOURCES), lp=True) as run:
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
    with differential(CAPPED_BY_REGION, _frames(CAPPED_SOURCES | {'hi': hi})) as run:
        assert run.oracle == pytest.approx(objective, rel=RTOL), 'rows outside the region change nothing'


@pytest.mark.parametrize('lane', ['relational', 'linopy'])
def test_a_hole_inside_the_region_is_still_refused_on_each_lane(lane):
    """Narrowing the question to the region must not stop it being asked there.

    Asserted lane by lane: one lane refusing satisfies a ``pytest.raises``
    around both.
    """
    sources = _frames(CAPPED_SOURCES | {'hi': {'t': [0], 'value': [40.0]}})
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / 'capped.yaml'
        path.write_text(yaml.safe_dump(CAPPED_BY_REGION))
        build = (
            (lambda: sps.build(path, sources))
            if lane == 'relational'
            else (lambda: specsolve_linopy.build(path, dict(sources)))
        )
        with pytest.raises(DataError, match=r"parameter 'hi' covers 1 fewer coordinate"):
            build()


@pytest.mark.parametrize(
    ('flagged', 'objective', 'reads'),
    [
        pytest.param(
            True, 130.0, 'the region takes the whole frame, so hi caps every step', id='a region that is everywhere'
        ),
        pytest.param(
            False, 20.0, 'the region takes nothing, so the otherwise 5 caps every step', id='a region that is nowhere'
        ),
    ],
)
def test_a_region_whose_mask_reads_no_dimension(flagged, objective, reads):
    """A scalar `when` names no coordinate set to cut the region down by, so it cuts by its own constant."""
    spec = CAPPED_BY_REGION | {
        'parameters': CAPPED_BY_REGION['parameters'] | {'flag_all': {'dims': [], 'dtype': 'bool'}},
        'expressions': {
            'cap': {'dims': ['t'], 'cases': {'flagged': {'when': 'flag_all', 'expression': 'hi'}}, 'otherwise': 5}
        },
    }
    sources = _frames(
        CAPPED_SOURCES
        | {'hi': {'t': [0, 1, 2, 3], 'value': [40.0, 10.0, 60.0, 20.0]}, 'flag_all': {'value': [flagged]}}
    )
    with differential(spec, sources) as run:
        assert run.oracle == pytest.approx(objective, rel=RTOL), reads


CARRIED_IN = {
    'dimensions': {'t': {'dtype': 'int'}, 'g': {}},
    'parameters': {
        'switchable': {'dims': ['g'], 'dtype': 'bool'},
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
    return _frames(
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
    with differential(CARRIED_IN, _carried_sources([False, True], [1.0, 0.0])) as run:
        rows = run.result.activity('ramp')
        assert rows.height == 8, 'every (t, g) coordinate has a ramp row, the first position included'
        assert sorted(set(rows.get_column('t'))) == [0, 1, 2, 3], 't == 0 is built like any other position'
        assert int((run.model.constraints['ramp'].labels == -1).sum()) == 0, (
            'the linopy lane masks out no ramp row either'
        )


@pytest.mark.parametrize(
    ('switchable', 'before', 'objective', 'reads'),
    [
        pytest.param(
            [False, True],
            [1.0, 0.0],
            4890.0,
            'boundary: peak was off before the horizon, so it may start at first_step 55',
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
    """The complement of a mask reading no dimension claims nothing, and so may restrict nothing.

    With a true scalar mask the ``otherwise`` claims nothing, while its
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
    sources = _carried_sources([False, True], [1.0, 0.0]) | _frames({'everywhere': {'value': [True]}})
    sources['load'] = _frames({'load': {'t': [0, 1, 2, 3], 'value': [70.0, 60.0, 60.0, 60.0]}})['load']
    with differential(spec, sources) as run:
        rows = run.result.activity('ramp')
        assert rows.height == 8, 'every (t, g) coordinate has a ramp row, the first position included'
        assert int((run.model.constraints['ramp'].labels == -1).sum()) == 0, (
            'the linopy lane masks out no ramp row either'
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
    sources = _frames(CAPPED_SOURCES | {'hi': {'t': [0, 1, 2, 3], 'value': [40.0, 10.0, 60.0, 20.0]}})
    with differential(spec, sources) as run:
        assert run.oracle == pytest.approx(160.0, rel=RTOL), 'the flagged steps read hi and the rest read twice it'


def test_a_divisor_is_asked_for_data_only_where_its_region_applies():
    """A rate stated for the steps its region claims is not asked about the rest."""
    spec = CAPPED_BY_REGION | {
        'expressions': {
            'cap': {
                'dims': ['t'],
                'cases': {'flagged': {'when': 'flag', 'expression': 'hi / rate'}},
                'otherwise': 5,
            },
        },
        'parameters': CAPPED_BY_REGION['parameters'] | {'rate': {'dims': ['t']}},
    }
    sources = _frames(CAPPED_SOURCES | {'rate': {'t': [0, 2], 'value': [2.0, 2.0]}})
    with differential(spec, sources) as run:
        assert run.oracle == pytest.approx(60.0, rel=RTOL), 'the flagged steps cap at 40/2 and 60/2, the rest at 5'


def test_a_hole_in_a_divisor_inside_its_region_is_still_refused():
    """Narrowing the divisor's question to the region must not stop it being asked there (#1465)."""
    spec = CAPPED_BY_REGION | {
        'expressions': {
            'cap': {
                'dims': ['t'],
                'cases': {'flagged': {'when': 'flag', 'expression': 'hi / rate'}},
                'otherwise': 5,
            },
        },
        'parameters': CAPPED_BY_REGION['parameters'] | {'rate': {'dims': ['t']}},
    }
    sources = _frames(CAPPED_SOURCES | {'rate': {'t': [0], 'value': [2.0]}})
    both_lanes_refuse(spec, sources, match=r"parameter 'rate' is used as a divisor but covers 1 fewer coordinate")


#: `cap` is a sum over `g` inside the flagged region: `hi` owes two coordinates
#: per flagged step, and none for the steps the `otherwise` carries.
SUMMED_BY_REGION = CAPPED_BY_REGION | {
    'dimensions': CAPPED_BY_REGION['dimensions'] | {'g': {'dtype': 'str'}},
    'parameters': CAPPED_BY_REGION['parameters'] | {'hi': {'dims': ['t', 'g']}},
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
    with differential(SUMMED_BY_REGION, _frames(SUMMED_SOURCES)) as run:
        assert run.oracle == pytest.approx(110.0, rel=RTOL), 'the flagged steps cap at 30+10 and 30+30, the rest at 5'


#: The whole cased quantity under the sum, rather than a sum inside a region:
#: `bound` is cased over `(k, t)` and the row reads its total over `t`.
SUMMED_CASES = {
    'dimensions': {'k': {'dtype': 'str'}, 't': {}},
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
    """A region's claim survives the reduction that keeps the dims it reads.

    After the sum over `t` the `a` piece has no row at `k = b` and the
    `otherwise` piece none at `k = a`; neither is a hole.
    """
    with differential(SUMMED_CASES, _frames(SUMMED_CASES_SOURCES)) as run:
        assert run.oracle == pytest.approx(18.0, rel=RTOL), 'k=a caps at 5+6 and k=b at 3+4'


def test_a_hole_the_summed_region_reads_is_still_refused():
    """A hole under the sum inside the region is refused (#1465)."""
    holed = {'t': [0, 0, 2], 'g': ['u', 'v', 'u'], 'value': [30.0, 10.0, 30.0]}
    both_lanes_refuse(
        SUMMED_BY_REGION, _frames(SUMMED_SOURCES | {'hi': holed}), match=r"parameter 'hi' covers 1 fewer coordinate"
    )


#: The flow an outage takes off its line: the line's flow where it stands,
#: nothing where it does not. Every line stands, so the `otherwise` region is
#: empty, and the pullback through `outage_line` cannot carry its claim.
OUTAGE_FLOW = {
    'dimensions': {'line': {'dtype': 'str'}, 'outage': {'dtype': 'str'}},
    'relations': {'outage_line': {'key': 'outage', 'values': 'line'}},
    'parameters': {
        'active': {'dims': ['line'], 'dtype': 'bool'},
        'share': {'dims': ['line', 'outage']},
        'cap': {'dims': ['line']},
    },
    'variables': {'s': {'dims': ['line'], 'bounds': {'lower': 0, 'upper': 100}}},
    'expressions': {
        'monitored': {'dims': ['line'], 'cases': {'standing': {'when': 'active', 'expression': 's'}}, 'otherwise': 0},
        'outage_s': {'dims': ['outage'], 'expression': 'at(monitored, by=outage_line, over=line, into=outage)'},
    },
    'constraints': {
        'after': {'dims': ['line', 'outage'], 'expression': 'monitored + share * outage_s <= cap'},
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(s, over=line)'},
}


def test_a_region_worth_zero_read_through_a_pullback_owes_nothing():
    """`otherwise: 0` adds nothing anywhere, so no row is short of it.

    The zero was a constant piece claimed by the region where a line does not
    stand. The pullback dropped that claim, so the piece was owed at every row
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
    with differential(OUTAGE_FLOW, _frames(sources)) as run:
        assert float(run.result.objective) == pytest.approx(15.0, rel=RTOL), 's_a at 10 leaves s_b 5'
