"""sum: the transport YAML through both backends, and what coordinates buy.

Three-way differential on examples/transport.yaml:
  1. specsolve_linopy.build + solve (sum via linopy groupby)
  2. lowered Program -> Engine -> the `highs` solver, plus the LP file
  3. hand-built indicator-matrix linopy model (an independent oracle that
     involves no sum at all)

Plus ``examples/monthly_budget.yaml``, which is the same primitive over *time*:
a relation over ``snapshot`` groups it into months exactly as a relation over
``generator`` groups onto buses. The gallery page quotes its dual and prints
its snapshot index, so a test has to hold both.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from mathspec.program import (
    Add,
    GroupSum,
    Named,
    Negate,
    Variable,
)

import specsolve as sps
from specsolve.errors import DataError
from specsolve.relational.engine.engine import Engine
from specsolve.sources import tidy_sources
from tests.conftest import EXAMPLES_DIR, override, schema_of
from tests.differential import RTOL, differential
from tests.oracle import pd, specsolve_linopy, transport_linopy_objective

TRANSPORT_YAML = EXAMPLES_DIR / 'transport.yaml'


def _inputs(gens, lines, load):
    data = {
        'p_max': gens.set_index('generator')['p_max'],
        'cost': gens.set_index('generator')['cost'],
        'cap': lines.set_index('line')['cap'],
        'neg_cap': -lines.set_index('line')['cap'],
        'load': load,
    }
    return data | {
        'snapshot': pd.Index(sorted(load['snapshot'].unique()), name='snapshot'),
        'generator': gens[['generator']],
        'bus': pd.Index(sorted(load['bus'].unique()), name='bus'),
        'line': lines[['line']],
        'gen_bus': gens[['generator', 'bus']],
        'line_from': lines[['line', 'from_bus']].rename(columns={'from_bus': 'bus'}),
        'line_to': lines[['line', 'to_bus']].rename(columns={'to_bus': 'bus'}),
    }


def test_transport_yaml_agrees_with_an_independent_oracle(transport_data):
    gens, lines, load = transport_data
    data = _inputs(gens, lines, load)

    independent = transport_linopy_objective(gens, lines, load)
    assert np.isfinite(independent), 'indicator matrices, no sum involved — an oracle for the oracle'

    with differential(TRANSPORT_YAML, data, lp=True) as run:
        assert run.oracle == pytest.approx(independent, rel=RTOL)


# ---------------------------------------------------------------------------
# lowering
# ---------------------------------------------------------------------------


def _flatten(expr):
    if isinstance(expr, Add):
        return _flatten(expr.left) + _flatten(expr.right)
    if isinstance(expr, Negate):
        return _flatten(expr.operand)
    if isinstance(expr, Named):
        return _flatten(expr.body)
    return [expr]


def test_sum_lowers_to_one_node_per_injection_term():
    program = schema_of(TRANSPORT_YAML).program

    (c,) = program.constraints.values()
    assert c.dims == ('snapshot', 'bus')
    terms = _flatten(c.lhs)
    grouped = {
        (t.operand, t.direction.consumed_dims, t.direction.name, t.direction.produced_dims)
        for t in terms
        if isinstance(t, GroupSum)
    }
    assert (Variable('p'), ('generator',), 'gen_bus', ('bus',)) in grouped, (
        'generation is grouped out of generator onto bus, through gen_bus and nothing else'
    )
    assert (Variable('f'), ('line',), 'line_to', ('bus',)) in grouped, (
        'inflow is grouped out of line onto bus, through line_to and nothing else'
    )


# ---------------------------------------------------------------------------
# the hole a coordinate closes: a label that is not a coordinate of its target
# ---------------------------------------------------------------------------


def _relationally(data):
    schema = schema_of(TRANSPORT_YAML)
    program = schema.program
    with Engine() as engine:
        engine.build(program, tidy_sources(program, data))


def test_a_mistyped_coordinate_is_refused_on_both_lanes(transport_data):
    """A mapping value that is not a declared label would drop its generator out of the balance silently."""
    gens, lines, load = transport_data
    bad = gens.copy()
    bad.loc[bad.index[0], 'bus'] = 'nowhere'  # a bus that does not exist
    data = _inputs(bad, lines, load)

    with pytest.raises(DataError, match="not 'bus' labels"):
        _relationally(data)
    with pytest.raises(DataError, match="not 'bus' labels"):
        specsolve_linopy.build(TRANSPORT_YAML, data)


def test_a_coordinate_must_be_single_valued(transport_data):
    """Two rows disagreeing about a generator's bus is a data bug, not a
    silently-picked winner.

    Only the *index* is doubled here. Doubling the generator frame outright
    duplicates the parameters it also feeds, and the keyed-parameter check
    below catches that first — which is correct, and would leave this test
    asserting the wrong message.
    """
    gens, lines, load = transport_data
    other = 's' if gens['bus'].iloc[0] != 's' else 'n'
    data = _inputs(gens, lines, load)
    data['gen_bus'] = pd.concat([data['gen_bus'], data['gen_bus'].head(1).assign(bus=other)])

    with pytest.raises(DataError, match='more than once'):
        _relationally(data)


def test_a_parameter_carrying_a_coordinate_twice_is_refused(transport_data):
    """A parameter is a function of its dims, so two rows for one coordinate
    has no answer — and the linopy lane will not lay such a source out either.
    """
    gens, lines, load = transport_data
    data = _inputs(gens, lines, load)
    data['p_max'] = pd.concat([data['p_max'], data['p_max'].head(1)])

    with pytest.raises(DataError, match="parameter 'p_max' has more than one row"):
        _relationally(data)


def test_a_coordinate_bearing_dim_needs_an_index_source(transport_data):
    """A coordinate cannot be inferred from the parameters that use the dim —
    inferring it is what would let a typo extend the label space.

    Nor from the map over it: `gen_bus` still says which generators sit where,
    and a relation over a dimension never says which of its members exist.
    """
    gens, lines, load = transport_data
    data = _inputs(gens, lines, load)
    del data['generator']

    with pytest.raises(DataError, match=re.escape("has its maps (sources['gen_bus'])")):
        _relationally(data)


PARTIAL_YAML = """
dimensions:
  g: {dtype: str}
  item: {dtype: str}
relations:
  grp: {key: item, values: g}
parameters:
  cap: {dims: [item]}
  target: {dims: [g]}
variables:
  x:
    dims: [item]
    bounds: {lower: 0, upper: cap}
constraints:
  meet:
    dims: [g]
    expression: sum(x, by=grp, over=item, into=g) >= target
objective:
  sense: minimize
  expression: sum(x, over=item)
"""


def _partial_inputs():
    """`item` carries relation `grp`: i0 and i1 in group g0, i2 in none."""
    items = ['i0', 'i1', 'i2']
    index = pd.DataFrame({'item': items})
    grp = pd.DataFrame({'item': ['i0', 'i1'], 'g': ['g0', 'g0']})
    return (
        {  # relational sources
            'item': index,
            'grp': grp,
            'g': pd.DataFrame({'g': ['g0']}),
            'cap': pd.DataFrame({'item': items, 'value': [5.0, 5.0, 5.0]}),
            'target': pd.DataFrame({'g': ['g0'], 'value': [3.0]}),
        },
        {  # the same, in the shapes the linopy lane is usually fed
            'cap': pd.Series([5.0, 5.0, 5.0], index=pd.Index(items, name='item')),
            'target': pd.Series([3.0], index=pd.Index(['g0'], name='g')),
            'item': index,
            'grp': grp,
            'g': pd.Index(['g0'], name='g'),
        },
    )


def test_a_partial_coordinate_places_its_orphans_nowhere(tmp_path):
    """A null coordinate means "this label is in no group", not "typo".

    `i2` belongs to no group, so `sum` places its terms nowhere and only
    `i0`/`i1` can meet the target of 3.
    """
    path = tmp_path / 'partial.yaml'
    path.write_text(PARTIAL_YAML)
    sources, data = _partial_inputs()

    with sps.solve(path, sources) as result:
        assert result.is_ok
        assert result.objective == pytest.approx(3.0)
        assert result.to_pandas('x').set_index('item')['value']['i2'] == pytest.approx(0.0), (
            'the orphan is still a variable; it just carries no group obligation'
        )

    model = specsolve_linopy.build(path, data)
    model.solve(solver_name='highs', output_flag=False)
    assert float(model.objective.value) == pytest.approx(3.0)


GROUPED_ONTO_BUS = {
    'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}},
    'relations': {'gen_bus': {'key': 'generator', 'values': 'bus'}},
    'parameters': {'p_max': {'dims': ['generator']}, 'load': {'dims': ['bus']}},
    'variables': {'p': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'balance': {'dims': ['bus'], 'expression': 'sum(p, by=gen_bus, over=generator, into=bus) >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p, over=generator)'},
}

#: The two ways a grouped sum's own index departs from the dimension it
#: produces. Both instances carry 10 of load in total, so landing the group on
#: the declared index is the only thing between them and the same optimum.
GROUPED_ONTO_BUS_SOURCES = {
    'a label no member reaches': {
        'bus': pl.DataFrame({'bus': ['north', 'south', 'east']}),
        'generator': ['g1', 'g2'],
        'gen_bus': pl.DataFrame({'generator': ['g1', 'g2'], 'bus': ['north', 'south']}),
        'p_max': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [10.0, 10.0]}),
        'load': pl.DataFrame({'bus': ['north', 'south', 'east'], 'value': [4.0, 6.0, 0.0]}),
    },
    'a declared order that is not sorted': {
        'bus': pl.DataFrame({'bus': ['south', 'north']}),
        'generator': ['g1', 'g2'],
        'gen_bus': pl.DataFrame({'generator': ['g1', 'g2'], 'bus': ['north', 'south']}),
        'p_max': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [10.0, 10.0]}),
        'load': pl.DataFrame({'bus': ['north', 'south'], 'value': [4.0, 6.0]}),
    },
}


@pytest.mark.parametrize('sources', GROUPED_ONTO_BUS_SOURCES.values(), ids=GROUPED_ONTO_BUS_SOURCES.keys())
def test_a_grouped_sum_lands_on_the_dimension_it_declares(sources):
    """The result spans ``bus``'s declared index, not the labels the relation reaches.

    A groupby yields only the labels some member points at, and in sorted
    order. Either departure — a bus no generator sits on, or a declared order
    that is not alphabetical — leaves the linopy lane holding a ``bus`` that is
    not the model's ``bus``, and linopy v1 refuses the next combination, since
    it aligns on membership and order alike. The relational lane never faces
    the question: it joins on the label and takes its rows from the dims
    product, not from whatever the group produced.
    """
    with differential(GROUPED_ONTO_BUS, sources) as run:
        assert run.oracle == pytest.approx(10.0, rel=RTOL), 'every bus must be served, including one with no generator'


BROADCAST_GROUP_SUM = {
    'dimensions': {
        'snapshot': {'dtype': 'int'},
        'generator': {'dtype': 'str'},
        'bus': {'dtype': 'str'},
    },
    'relations': {'gen_bus': {'key': 'generator', 'values': 'bus'}},
    'parameters': {'w': {'dims': ['generator']}, 'limit': {'dims': ['snapshot', 'bus']}},
    'variables': {'x': {'dims': ['snapshot'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {
        'cap': {
            'dims': ['snapshot', 'bus'],
            'expression': 'sum(x * w, by=gen_bus, over=generator, into=bus) <= limit',
        }
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}

#: g1 and g2 share a bus, so grouping merges two rows carrying the *same*
#: variable — which is the case a broadcast `over` creates and a `dims:` one
#: cannot.
BROADCAST_SOURCES = {
    'snapshot': [0, 1],
    'w': pl.DataFrame({'generator': ['g1', 'g2', 'g3'], 'value': [1.0, 2.0, 5.0]}),
    'limit': pl.DataFrame({'snapshot': [0, 0, 1, 1], 'bus': ['b1', 'b2'] * 2, 'value': [9.0, 100.0, 9.0, 100.0]}),
    'generator': ['g1', 'g2', 'g3'],
    'gen_bus': pl.DataFrame({'generator': ['g1', 'g2', 'g3'], 'bus': ['b1', 'b1', 'b2']}),
    'bus': pl.DataFrame({'bus': ['b1', 'b2']}),
}


def test_sum_over_a_broadcast_dim_still_collapses_its_terms():
    """The variable does not carry the grouped dim, so a group holds it twice.

    `sum(x * w, by=gen_bus, over=generator, into=bus)` with `x` indexed by snapshot
    alone: `generator` reaches the fragment by broadcast from `w`, so two
    generators on one bus put the *same* `var_label` on one row. Nothing after
    this point can tell them apart — a solver handed a row with a column twice
    is entitled to reject the whole model, and HiGHS does.
    """
    with sps.build(BROADCAST_GROUP_SUM, BROADCAST_SOURCES) as model:
        tables = model._engine._model.handoff
        matrix = tables.matrix_block(0, tables.row_count).sort('row', 'col')
        assert matrix.height == 4, 'a column appears twice on a row'
        assert matrix['coeff'].to_list() == [3.0, 5.0, 3.0, 5.0], 'the 1.0 and the 2.0 merged'

        result = model.solve()
    assert result.termination_condition == 'optimal'
    assert result.objective == pytest.approx(6.0), '3x <= 9 at b1, over two snapshots'


def test_sum_over_a_declared_dim_needs_no_such_collapse():
    """The counterpart: when the variable carries the grouped dim, each merged
    row has its own label and there is nothing to add."""
    spec = override(
        BROADCAST_GROUP_SUM,
        **{
            'variables.x.dims': ['snapshot', 'generator'],
            'constraints.cap.expression': 'sum(x * w, by=gen_bus, over=generator, into=bus) <= limit',
        },
    )
    with sps.build(spec, BROADCAST_SOURCES) as model:
        tables = model._engine._model.handoff
        matrix = tables.matrix_block(0, tables.row_count).sort('row', 'col')
        assert matrix.height == 6, 'one entry per (row, generator-on-that-bus), not one per bus'
        assert model.solve().termination_condition == 'optimal'


# ---------------------------------------------------------------------------
# the objective's own key — the projection the matrix does not do
# ---------------------------------------------------------------------------

#: `y` is indexed by bus and `w` by snapshot, so `y * w` holds one row per
#: (bus, snapshot) and one *column* per bus. It is the objective's projection
#: down to `(col, coeff)` that drops the dims and merges those rows.
BROADCAST_OBJECTIVE = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'bus': {'dtype': 'str'}},
    'parameters': {'w': {'dims': ['snapshot']}, 'floor': {'dims': ['bus']}},
    'variables': {'y': {'dims': ['bus'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {'atleast': {'dims': ['bus'], 'expression': 'y >= floor'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(y * w)'},
}

#: ``w`` is deliberately unequal across snapshots, so last-write-wins is not
#: the same number as the sum.
BROADCAST_OBJECTIVE_SOURCES = {
    'snapshot': [0, 1, 2, 3],
    'w': pl.DataFrame({'snapshot': [0, 1, 2, 3], 'value': [1.0, 10.0, 100.0, 1000.0]}),
    'floor': pl.DataFrame({'bus': ['b0', 'b1', 'b2'], 'value': [1.0, 2.0, 3.0]}),
    'bus': pl.DataFrame({'bus': ['b0', 'b1', 'b2']}),
}


def test_an_objective_term_carrying_dims_is_still_summed_per_column():
    """A coefficient is the *sum* over the dims the objective projects away.

    The matrix keeps a fragment's dims — a constraint row is a function of dims
    that include them — so one row there is one `(row, col)` cell. The
    objective drops them, and a fragment that still carries one then holds
    several rows per column.
    """
    with sps.build(BROADCAST_OBJECTIVE, BROADCAST_OBJECTIVE_SOURCES) as model:
        obj = model._engine._model.handoff.obj.sort('col')
        assert obj.height == 3, 'one row per column, not one per (bus, snapshot)'
        assert obj['coeff'].to_list() == [1111.0] * 3, 'sum(w), not w[-1]'


def test_the_broadcast_objective_agrees_with_the_linopy_lane():
    """The same model end to end, against linopy — 6666.0, not 6000.0."""
    data = {
        'w': pd.Series([1.0, 10.0, 100.0, 1000.0], index=pd.Index([0, 1, 2, 3], name='snapshot')),
        'floor': pd.Series([1.0, 2.0, 3.0], index=pd.Index(['b0', 'b1', 'b2'], name='bus')),
    }
    index = {'snapshot': [0, 1, 2, 3], 'bus': pd.Index(['b0', 'b1', 'b2'], name='bus')}
    with differential(BROADCAST_OBJECTIVE, data | index, lp=True) as run:
        assert run.oracle == pytest.approx(6666.0)


def test_an_objective_over_the_variables_own_dims_keeps_its_coefficients():
    """The counterpart: a projection that merges nothing must change nothing.

    `y * floor` reaches the objective carrying `bus` alone — `y`'s own dim — so
    each column holds exactly one row and the sum over it is that row. The
    aggregate must not turn a coefficient into anything but itself.
    """
    spec = override(BROADCAST_OBJECTIVE, **{'objective.expression': 'sum(y * floor)'})
    with sps.build(spec, BROADCAST_OBJECTIVE_SOURCES) as model:
        obj = model._engine._model.handoff.obj.sort('col')
        assert obj.height == 3
        assert obj['coeff'].to_list() == [1.0, 2.0, 3.0], 'floor itself, un-summed'


# ---------------------------------------------------------------------------
# the same construct, grouping time
# ---------------------------------------------------------------------------

MONTHLY_YAML = EXAMPLES_DIR / 'monthly_budget.yaml'
MONTHLY_PAGE = Path('docs/examples/monthly_budget.md')


@pytest.fixture
def monthly():
    """The snapshot index and sources: six snapshots over three calendar months,
    wind capped in the first.

    The `month_of` relation is data prep: one polars expression.
    """
    import datetime as dt

    hours = [dt.datetime(2030, 1, 1) + dt.timedelta(days=15 * i) for i in range(6)]
    snapshots = pl.DataFrame({'snapshot': hours})
    month_of = snapshots.with_columns(pl.col('snapshot').dt.strftime('%Y-%m').alias('month'))
    months = sorted(set(month_of['month']))
    gens = ['wind', 'gas']
    return (
        month_of,
        {
            'snapshot': snapshots,
            'month_of': month_of,
            'month': pl.DataFrame({'month': months}),
            'generator': pl.DataFrame({'generator': gens}),
            'p_max': pl.DataFrame({'generator': gens, 'value': [10.0, 100.0]}),
            'cost': pl.DataFrame({'generator': gens, 'value': [1.0, 50.0]}),
            'load': pl.DataFrame({'snapshot': hours, 'value': [20.0] * 6}),
            'monthly_cap': pl.DataFrame(
                {
                    'month': [m for m in months for _ in gens],
                    'generator': gens * len(months),
                    'value': [5.0 if (m == months[0] and g == 'wind') else 1e4 for m in months for g in gens],
                }
            ),
        },
    )


def test_a_monthly_budget_binds_and_prices_itself(monthly):
    """The number the gallery page quotes, held by a test.

    January caps wind at 5 where three snapshots could carry 30, so the cap
    binds and its shadow price is the cost of covering that energy with gas
    instead — 50 against 1. February and March are slack and price at zero,
    which is what distinguishes a binding budget from a decorative one.
    """
    month_of, sources = monthly
    with sps.solve(MONTHLY_YAML, sources) as result:
        assert result.is_ok
        wind = (
            result.primal('p')
            .filter(pl.col('generator') == 'wind')
            .join(month_of, on='snapshot')
            .group_by('month')
            .agg(pl.col('value').sum())
            .sort('month')
        )
        assert wind['value'].to_list() == pytest.approx([5.0, 10.0, 20.0]), (
            '3 snapshots in Jan (capped at 5), 1 in Feb, 2 in Mar — unequal groups'
        )

        duals = result.dual('monthly_budget').filter(pl.col('generator') == 'wind').sort('month')
        assert duals['value'].to_list() == pytest.approx([-49.0, 0.0, 0.0])


def test_the_monthly_grouping_is_a_column_and_nothing_else(monthly):
    """Re-grouping the same snapshots re-states the budget, model untouched.

    Quarters instead of months: one different column in the snapshot index,
    and the constraint now spans three-month blocks. That is the claim the
    page makes about weeks, seasons and representative periods, checked once.
    """
    month_of, sources = monthly
    quarters = month_of.with_columns(pl.lit('2030-Q1').alias('month'))
    regrouped = {
        **sources,
        'month_of': quarters,
        'month': pl.DataFrame({'month': ['2030-Q1']}),
        'monthly_cap': pl.DataFrame({'month': ['2030-Q1'] * 2, 'generator': ['wind', 'gas'], 'value': [5.0, 1e4]}),
    }
    with sps.solve(MONTHLY_YAML, regrouped) as result:
        assert result.is_ok
        assert result.dual('monthly_budget').height == 2, 'one row per group, and there is now one group'
        wind = result.primal('p').filter(pl.col('generator') == 'wind')['value'].sum()
        assert wind == pytest.approx(5.0), 'the cap now binds across the whole quarter'


def test_a_mistyped_month_is_a_typo_and_not_a_new_group(monthly):
    """Why the target of a coordinate has to be a declared dimension.

    Without one there is nothing to check the snapshot index against, and
    `2030-3` beside `2030-03` would quietly become a fourth group with a budget
    of its own — the model then solves a smaller problem and says nothing. The
    same check catches a generator assigned to a bus that does not exist.
    """
    month_of, sources = monthly
    typo = month_of.with_columns(
        pl.when(pl.col('month') == '2030-03').then(pl.lit('2030-3')).otherwise(pl.col('month')).alias('month')
    )
    with pytest.raises(DataError, match=r"relation 'month_of' has value\(s\) in 'month' that are not 'month' labels"):
        sps.solve(MONTHLY_YAML, {**sources, 'month_of': typo})


def test_the_index_the_page_prints_is_the_index_it_solves(monthly):
    """The frame printed on the page is the frame these tests build.

    `test_doc_examples.py` sweeps `python` and `yaml` fences and runs neither,
    so a pasted *output* block is the one kind of doc claim nothing checks —
    change the timestamps here and the page would keep printing the old ones.
    Defaults are restored while formatting, so a contributor's `POLARS_FMT_*`
    environment cannot fail this.
    """
    month_of, _sources = monthly
    fences = re.findall(r'^```text\n(.*?)^```', MONTHLY_PAGE.read_text(), re.MULTILINE | re.DOTALL)
    printed = [block for block in fences if block.startswith('shape: (')]
    assert len(printed) == 1, 'the page prints exactly one frame'
    with pl.Config(restore_defaults=True):
        assert printed[0].rstrip('\n') == str(month_of)


#: Every load stands, so the `otherwise` region of `demand` holds no member
#: at all and its grouped sum lands on no bus.
DEMAND_BY_REGION = {
    'dimensions': {'bus': {'dtype': 'str'}, 'load': {'dtype': 'str'}},
    'relations': {'load_bus': {'key': 'load', 'values': 'bus'}},
    'parameters': {'p_set': {'dims': ['load']}, 'active': {'dims': ['load'], 'dtype': 'bool'}},
    'variables': {'g': {'dims': ['bus'], 'bounds': {'lower': 0}}},
    'expressions': {
        'demand': {'dims': ['load'], 'cases': {'on': {'when': 'active', 'expression': 'p_set'}}, 'otherwise': 0},
    },
    'constraints': {'balance': {'dims': ['bus'], 'expression': 'g == sum(demand, by=load_bus, over=load, into=bus)'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(g)'},
}


def test_a_region_that_holds_no_member_of_a_group_adds_the_empty_sum():
    """The `otherwise` region of `demand` is empty at both buses, which is a sum of nothing, not a missing value.

    The zero an empty group adds was written only for a bus the relation maps
    no load to. Here the relation maps a load to every bus and the region
    leaves each out, so the group was empty with no zero written, and the
    balance was refused as short of `p_set`.
    """
    sources = {
        'bus': ['north', 'south'],
        'load': ['a', 'b'],
        'load_bus': pd.DataFrame({'load': ['a', 'b'], 'bus': ['north', 'south']}),
        'p_set': pd.Series({'a': 3.0, 'b': 4.0}).rename_axis('load'),
        'active': pd.Series({'a': True, 'b': True}).rename_axis('load'),
    }
    with differential(DEMAND_BY_REGION, sources) as run:
        assert float(run.result.objective) == pytest.approx(7.0, rel=RTOL), 'each bus generates its own load'


#: `tap` has no labels, so the angle each bus reads is a sum over nothing that
#: still keeps the bus it is read at.
SUM_OVER_NOTHING = {
    'dimensions': {'bus': {'dtype': 'str'}, 'tap': {'dtype': 'str'}},
    'parameters': {'angle': {'dims': ['bus', 'tap']}},
    'variables': {'g': {'dims': ['bus'], 'bounds': {'lower': 0}}},
    'constraints': {'floor': {'dims': ['bus'], 'expression': 'g >= sum(angle, over=tap) + 1'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(g)'},
}


def test_a_sum_over_a_dimension_with_no_labels_is_zero():
    """`sum(angle, over=tap)` with no tap is the empty sum, so each bus floors at one.

    A constant summed over a dimension kept no row at a coordinate no member
    landed in, so the row read a gap there and was refused as short of
    `angle`, which has no row to be short of.
    """
    sources = {
        'bus': ['north', 'south'],
        'tap': pd.Index([], name='tap', dtype=str),
        'angle': pd.DataFrame({'bus': pd.Series([], dtype=str), 'tap': pd.Series([], dtype=str), 'value': []}),
    }
    with differential(SUM_OVER_NOTHING, sources) as run:
        assert float(run.result.objective) == pytest.approx(2.0, rel=RTOL), 'two buses at one each'
