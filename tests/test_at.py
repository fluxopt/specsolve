"""``at()`` — the pullback, and the models that had no formulation without it.

`sum` walks a relation from the fine dim into the coarse one; `at` walks it
back out. `examples/multi_period.yaml` is ragged — four snapshots in 2030
against two in 2050 — and its capacity bound reads a per-period variable at
every snapshot.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import specsolve as sps
from tests.conftest import EXAMPLES_DIR, by_coord, relation

MULTI_PERIOD = EXAMPLES_DIR / 'multi_period.yaml'
PAGE = Path('docs/examples/multi_period.md')

#: 2030 at four snapshots and 2050 at two; a 2050 snapshot weighs four hours.
SNAPSHOTS = [0, 1, 2, 3, 4, 5]
PERIOD_OF = [2030, 2030, 2030, 2030, 2050, 2050]
GENERATORS = ['wind', 'gas']


def _sources(capex_2050_wind: float = 8.0):
    return {
        'snapshot': pl.DataFrame({'snapshot': SNAPSHOTS}),
        'period_of': pl.DataFrame({'snapshot': SNAPSHOTS, 'period': PERIOD_OF}),
        'period': pl.DataFrame({'period': [2030, 2050]}),
        'generator': pl.DataFrame({'generator': GENERATORS}),
        'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [10.0, 20.0, 30.0, 20.0, 40.0, 60.0]}),
        'weight': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [1.0, 1.0, 1.0, 1.0, 4.0, 4.0]}),
        'opex': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 5.0]}),
        'capex': pl.DataFrame(
            {
                'generator': GENERATORS * 2,
                'period': [2030, 2030, 2050, 2050],
                'value': [10.0, 2.0, capex_2050_wind, 2.0],
            }
        ),
    }


def test_the_multi_period_page_number():
    """The optimum `docs/examples/multi_period.md` quotes, and the per-period build behind it."""
    with sps.solve(MULTI_PERIOD, _sources()) as result:
        nominal = result.primal('p_nom').sort('period', 'generator')
        assert result.objective == pytest.approx(750.0)

    built = {(row['period'], row['generator']): row['value'] for row in nominal.to_dicts()}
    assert built[(2030, 'wind')] == pytest.approx(20.0), '2030 peaks at 30 and splits the build'
    assert built[(2030, 'gas')] == pytest.approx(10.0), '2030 peaks at 30 and splits the build'
    assert built[(2050, 'wind')] == pytest.approx(60.0), (
        '2050 peaks at 60 with every snapshot weighted four times, so the operating term takes it all to wind'
    )
    assert built[(2050, 'gas')] == pytest.approx(0.0), 'gas is priced out of 2050 entirely'

    assert '750.0' in PAGE.read_text(), 'the page quotes an optimum this test does not hold'


def test_a_period_bound_actually_binds():
    """Not vacuous: making 2050 capacity dearer moves the answer."""
    with sps.solve(MULTI_PERIOD, _sources()) as unbounded:
        base = unbounded.objective

    with sps.solve(MULTI_PERIOD, _sources(capex_2050_wind=80.0)) as dearer:
        assert dearer.objective > base, 'the per-period capacity bound is not reaching the snapshots'


COMPONENT_GATE = {
    'dimensions': {
        'flow': {'dtype': 'str'},
        'component': {'dtype': 'str'},
        't': {'dtype': 'int'},
    },
    'relations': {'component_of': {'key': 'flow', 'values': 'component'}},
    'parameters': {'cost': {'dims': ['flow']}, 'oncost': {'dims': ['component']}},
    'variables': {
        'rate': {'dims': ['flow', 't'], 'bounds': {'lower': 0, 'upper': 10}},
        'on': {'dims': ['component', 't'], 'domain': 'binary'},
    },
    'constraints': {
        'gate': {
            'dims': ['flow', 't'],
            'expression': 'rate <= at(on, by=component_of, over=component, into=flow) * 10',
        },
        'need': {'dims': ['t'], 'expression': 'sum(rate, over=flow) >= 12'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(rate * cost) + sum(on * oncost)'},
}


def test_one_binary_gates_every_flow_of_its_component():
    """A per-component binary variable read on each of that component's flows (#185)."""
    flows, components = ['f1', 'f2', 'f3'], ['c1', 'c2']
    sources = {
        't': [0, 1],
        'flow': pl.DataFrame({'flow': flows}),
        'component_of': relation('flow', 'component', flows, ['c1', 'c1', 'c2']),
        'component': pl.DataFrame({'component': components}),
        'cost': pl.DataFrame({'flow': flows, 'value': [1.0, 2.0, 1.5]}),
        'oncost': pl.DataFrame({'component': components, 'value': [5.0, 7.0]}),
    }
    with sps.solve(COMPONENT_GATE, sources) as result:
        assert result.objective == pytest.approx(38.0)
        running = by_coord(result, 'on', 'component', 't')
        rates = by_coord(result, 'rate', 'flow', 't')

    for t in (0, 1):
        for flow, component in zip(flows, ['c1', 'c1', 'c2'], strict=True):
            if rates[(flow, t)] > 1e-9:
                assert running[(component, t)] == pytest.approx(1.0), (
                    f'{flow} runs at t={t} while its component {component} is off — the gate did not reach this flow'
                )


def test_at_agrees_with_the_oracle_through_a_reduction():
    """A pullback summed back over `flow` brings two copies of one label into a row, and they add.

    The oracle is imported in the body so the rest of the module runs on the
    bare install.
    """
    from tests.differential import differential
    from tests.oracle import pd

    spec = {
        'dimensions': {
            'flow': {'dtype': 'str'},
            'component': {'dtype': 'str'},
        },
        'relations': {'component_of': {'key': 'flow', 'values': 'component'}},
        'parameters': {'cost': {'dims': ['flow']}, 'share': {'dims': ['flow']}},
        'variables': {
            'level': {'dims': ['component'], 'bounds': {'lower': 0, 'upper': 10}},
            'take': {'dims': ['flow'], 'bounds': {'lower': 0, 'upper': 10}},
        },
        'constraints': {
            'draw': {
                'dims': [],
                'expression': 'sum(at(level, by=component_of, over=component, into=flow) * share, over=flow) >= 9',
            },
            'link': {'dims': ['flow'], 'expression': 'take <= at(level, by=component_of, over=component, into=flow)'},
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(level * 1.0) + sum(take * cost)'},
    }
    flows, components = ['f1', 'f2', 'f3'], ['c1', 'c2']
    data = {
        'cost': pd.Series([1.0, 2.0, 1.5], index=flows),
        'share': pd.Series([1.0, 2.0, 3.0], index=flows),
    }
    index = {
        'flow': pd.DataFrame({'flow': flows}),
        'component_of': relation('flow', 'component', flows, ['c1', 'c1', 'c2']),
        'component': pd.Index(components, name='component'),
    }
    with differential(spec, data | index) as run:
        assert run.result.objective > 0


def test_a_window_whose_length_is_read_from_data_is_an_incidence_table():
    """A minimum up time read per unit from data is an incidence table, not a shift chain.

    The window is one row per pair of snapshots inside it, contracted along a
    mirror axis `tf`; `at()` reads the commitment onto that axis. The slow unit
    is cheaper, so it runs and stays up its three hours: 13.0, where relaxing
    the window gives 11.0. Relational lane only: the harness cannot carry a
    3-D parameter (#60).
    """
    up_time = {'slow': 3, 'fast': 1}
    hours = list(range(6))
    spec = {
        'dimensions': {'unit': {'dtype': 'str'}, 't': {'dtype': 'int'}, 'tf': {'dtype': 'int'}},
        'relations': {'same_moment': {'key': 'tf', 'values': 't'}},
        'parameters': {
            'window': {'dims': ['unit', 't', 'tf']},
            'load': {'dims': ['t']},
            'cap': {'dims': ['unit']},
            'run_cost': {'dims': ['unit']},
            'idle_cost': {'dims': ['unit']},
        },
        'variables': {
            'p': {'dims': ['unit', 't'], 'bounds': {'lower': 0}},
            'on': {'dims': ['unit', 't'], 'domain': 'binary'},
            'started': {'dims': ['unit', 'tf'], 'domain': 'binary'},
        },
        'constraints': {
            'a_start_turns_it_on': {
                'dims': ['unit', 'tf'],
                'expression': (
                    'started >= at(on, by=same_moment, over=t, into=tf) - shift(at(on, by=same_moment, over=t, into=tf), along=tf, offset=1, edge=0)'
                ),
            },
            'stays_up_its_own_time': {
                'dims': ['unit', 't'],
                'expression': 'sum(started * window, over=tf) <= on',
            },
            'within_capacity': {'dims': ['unit', 't'], 'expression': 'p <= on * cap'},
            'meet_load': {'dims': ['t'], 'expression': 'sum(p, over=unit) >= load'},
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(p * run_cost) + sum(on * idle_cost)'},
    }
    rows = [(u, t, tf) for u, k in up_time.items() for t in hours for tf in hours if 0 <= t - tf < k]
    sources = {
        'window': pl.DataFrame(
            {
                'unit': [r[0] for r in rows],
                't': [r[1] for r in rows],
                'tf': [r[2] for r in rows],
                'value': [1.0] * len(rows),
            }
        ),
        'load': pl.DataFrame({'t': hours, 'value': [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]}),
        'cap': pl.DataFrame({'unit': list(up_time), 'value': [10.0, 10.0]}),
        'run_cost': pl.DataFrame({'unit': list(up_time), 'value': [1.0, 5.0]}),
        'idle_cost': pl.DataFrame({'unit': list(up_time), 'value': [1.0, 1.0]}),
    }
    sources |= {
        'unit': pl.DataFrame({'unit': list(up_time)}),
        't': pl.DataFrame({'t': hours}),
        'tf': pl.DataFrame({'tf': hours}),
        'same_moment': relation('tf', 't', hours, hours),
    }
    with sps.solve(spec, sources) as solution:
        assert solution.objective == pytest.approx(13.0), (
            'the slow unit runs and is held up its own three hours; 11.0 would mean the window read nothing'
        )
        on = solution.primal('on').filter(pl.col('value') > 0.5)
        assert on.height == 3, 'exactly the three snapshots its own minimum up time forces'
        assert set(on['unit']) == {'slow'}, 'and it is the slow unit that is held, not the fast one'


#: `f3` maps nowhere. With its row gone `take[f3]` goes to its bound of 10; with
#: the row built, its right-hand side is zero.
DANGLING = {
    'dimensions': {'flow': {'dtype': 'str'}, 'component': {'dtype': 'str'}},
    'relations': {'component_of': {'key': 'flow', 'values': 'component'}},
    'variables': {
        'level': {'dims': ['component'], 'bounds': {'lower': 0, 'upper': 10}},
        'take': {'dims': ['flow'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {
        'link': {'dims': ['flow'], 'expression': 'take <= at(level, by=component_of, over=component, into=flow)'}
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(take, over=flow) - 1000 * sum(level, over=component)'},
}
DANGLING_MAP = ['c1', 'c1', None]
FLOWS, COMPONENTS = ['f1', 'f2', 'f3'], ['c1', 'c2']


def _dangling_sources(map_: list | None = None, **extra):
    """The three flows and two components every case here attaches, the map under its own key."""
    from tests.oracle import pd

    return {
        'flow': pd.DataFrame({'flow': FLOWS}),
        'component_of': relation('flow', 'component', FLOWS, DANGLING_MAP if map_ is None else map_),
        'component': pd.Index(COMPONENTS, name='component'),
        **extra,
    }


def test_at_through_a_null_relation_takes_the_row_with_it():
    """A label mapping nowhere has no value to read, so the row is not asserted (#897).

    Linopy lane only; the differential case below carries the relational lane.
    """
    from tests.oracle import specsolve_linopy

    built = specsolve_linopy.build(DANGLING, _dangling_sources())
    labels = built.constraints['link'].labels.to_series().to_dict()
    assert labels['f3'] == -1, 'a flow mapping nowhere has no row, and -1 is how linopy spells one that was not built'
    assert labels['f1'] != -1 and labels['f2'] != -1, 'the flows that do map keep theirs'

    built.solve(solver_name='highs', output_flag=False)
    assert built.objective.value == pytest.approx(10.0), (
        'take[f3] is held by its own bound alone — 0.0 would mean the row was built and bound it at zero'
    )


def test_at_through_a_null_relation_agrees_between_lanes():
    """The same model on both lanes (#897, #968)."""
    from tests.differential import differential

    with differential(DANGLING, _dangling_sources(), lp=True) as run:
        assert run.oracle == pytest.approx(10.0)
        assert run.engine.diagnostics().rows == 2, 'the two flows that map somewhere have a row, and f3 has none'


#: Two columns read at once, through a relation that leaves `f3` out.
DANGLING_PAIR = {
    'dimensions': {'flow': {'dtype': 'str'}, 'component': {'dtype': 'str'}, 'kind': {'dtype': 'str'}},
    'relations': {'placed': {'key': 'flow', 'values': ['component', 'kind']}},
    'variables': {
        'level': {'dims': ['component', 'kind'], 'bounds': {'lower': 0, 'upper': 10}},
        'take': {'dims': ['flow'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {
        'link': {'dims': ['flow'], 'expression': 'take <= at(level, by=placed, over=[component, kind], into=flow)'}
    },
    'objective': {
        'sense': 'maximize',
        'expression': 'sum(take, over=flow) - 1000 * sum(sum(level, over=component), over=kind)',
    },
}


def test_at_through_a_pair_the_relation_leaves_out_takes_the_row_with_it():
    """A pullback reads a tuple of labels, so one null anywhere leaves nothing."""
    from tests.differential import differential
    from tests.oracle import pd

    flows = ['f1', 'f2', 'f3']
    with differential(
        DANGLING_PAIR,
        {
            'flow': pd.DataFrame({'flow': flows}),
            'placed': pd.DataFrame({'flow': flows[:2], 'component': ['c1', 'c1'], 'kind': ['k1', 'k1']}),
            'component': pd.Index(['c1', 'c2'], name='component'),
            'kind': pd.Index(['k1'], name='kind'),
        },
        lp=True,
    ) as run:
        assert run.oracle == pytest.approx(10.0), 'take[f3] is held by its own bound, not by a row reading nothing'
        assert run.engine.diagnostics().rows == 2, 'the two flows whose whole tuple maps have a row, and f3 has none'


#: The same shape with a total relation; `level` is masked away at `c2`.
MASKED = DANGLING | {
    'parameters': {'usable': {'dims': ['component']}},
    'variables': DANGLING['variables'] | {'level': DANGLING['variables']['level'] | {'where': 'usable > 0'}},
}


def test_at_over_a_masked_variable_takes_the_row_with_it():
    """A pullback carries the mask under it, not only the relation's own gaps (#968)."""
    from tests.differential import differential
    from tests.oracle import pd

    usable = pd.Series([1.0, 0.0], index=pd.Index(COMPONENTS, name='component'))
    with differential(MASKED, _dangling_sources(['c1', 'c1', 'c2'], usable=usable), lp=True) as run:
        assert run.oracle == pytest.approx(10.0), 'take[f3] is held by its own bound — 0.0 would mean a row bound it'
        assert run.engine.diagnostics().rows == 2, 'only the flows whose component has a level are asserted'


#: A pullback's absence is keyed by the fine dim alone; `u` is a second dim the shift edge spans.
DANGLING_SHIFTED = {
    'dimensions': {
        'flow': {'dtype': 'str'},
        'component': {'dtype': 'str'},
        't': {'dtype': 'int'},
        'u': {'dtype': 'str'},
    },
    'relations': {'component_of': {'key': 'flow', 'values': 'component'}},
    'variables': {
        'level': {'dims': ['component', 't', 'u'], 'bounds': {'lower': 0, 'upper': 10}},
        'take': {'dims': ['flow', 't', 'u'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {
        'link': {
            'dims': ['flow', 't', 'u'],
            'expression': 'take <= shift(at(level, by=component_of, over=component, into=flow), along=t, offset=1, edge=0)',
        }
    },
    'objective': {
        'sense': 'maximize',
        'expression': (
            'sum(sum(sum(take, over=flow), over=t), over=u) '
            '- 1000 * sum(sum(sum(level, over=component), over=t), over=u)'
        ),
    },
}


def test_a_pullbacks_absence_reaches_a_shift_that_spans_more_dims():
    """The shift edge places itself over dims the pullback's presence omits.

    An absence that arrived before the shift is not the edge, so it is not
    filled, and f3 keeps no row at either t (#987).
    """
    from tests.differential import differential
    from tests.oracle import pd

    extra = {'t': pd.Index([0, 1], name='t'), 'u': pd.Index(['a', 'b'], name='u')}
    with differential(DANGLING_SHIFTED, _dangling_sources(**extra), lp=True) as run:
        assert run.engine.diagnostics().rows == 8, (
            'the flows that map somewhere are asserted at both t, and f3 at neither'
        )
        assert run.oracle == pytest.approx(40.0), (
            "f3's four coordinates are held by their own bound alone, every other row by a level worth 1000"
        )
