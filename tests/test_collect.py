"""How a table is materialised: joins in the order specsolve wrote them, a coordinate product in index order."""

from __future__ import annotations

import polars as pl
import pytest

import specsolve as sps
from specsolve import _collect as collect
from specsolve._collect import collected
from specsolve._engine import assembly
from specsolve._engine.scope import Scope, position_column
from specsolve._names import COL


def _shift_shaped(snapshots: int, stores: int) -> pl.LazyFrame:
    """A constraint's rows joined to a variable moved one snapshot along, as a cyclic ``shift`` builds it."""
    keys = pl.DataFrame(
        {
            'snapshot': [s for s in range(snapshots) for _ in range(stores)],
            'store': [f's{k}' for _ in range(snapshots) for k in range(stores)],
        }
    )
    positions = pl.DataFrame({'snapshot': range(snapshots), 'position': range(snapshots)}).lazy()
    moved = (
        keys.with_row_index('col_index')
        .lazy()
        .join(positions.rename({'position': 'position_in'}), on='snapshot')
        .with_columns(((pl.col('position_in') + 1) % snapshots).alias('position_out'))
        .drop('snapshot', 'position_in')
        .join(positions.rename({'position': 'position_out'}), on='position_out')
        .drop('position_out')
    )
    return keys.with_row_index('row').lazy().join(moved, on=['snapshot', 'store']).select('row', 'col_index')


def test_a_join_runs_in_the_order_it_was_written(monkeypatch):
    """polars reorders joins by cost unless told not to.

    On this plan it runs the rows' join first, keyed on ``store`` alone, which
    holds every pair of snapshots per store before the second key cuts it down.
    """
    plans = []
    original = pl.LazyFrame.collect

    def recording(self, *args, **kwargs):
        plans.append(self.explain(optimizations=kwargs.get('optimizations', pl.QueryOptFlags())))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, 'collect', recording)
    assert collected(_shift_shaped(50, 4)).height == 200, 'one row per (snapshot, store)'

    keyed = [line.strip() for line in plans[-1].splitlines() if 'LEFT PLAN ON' in line]
    assert keyed[0] == 'LEFT PLAN ON: [col("snapshot"), col("store")]', (
        f'the outermost join is the one written last, on both keys; polars ran {keyed}'
    )


def test_a_coordinate_product_arrives_in_index_order():
    """The streaming engine picks which side of a cross join it buffers, and the product's order with it.

    A solver index is row-major over the positions, so a product out of that
    order has to be sorted before it is numbered.
    """
    spec = {
        'dimensions': {'a': {'dtype': 'int'}, 'b': {'dtype': 'int'}},
        'variables': {'x': {'dims': ['a', 'b']}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x)'},
    }
    with sps.build(spec, {'a': list(range(2)), 'b': list(range(50))}) as model:
        built = model._built
        product = Scope(built.program, built.attached, built.variables).product(('a', 'b')).pipe(collected)
    positions = product.select(position_column('a'), position_column('b')).rows()
    assert positions == sorted(positions), 'row-major: `a` varies slowest'


def test_the_bounds_are_collected_in_memory(monkeypatch, dispatch_yaml, dispatch_table_inputs):
    """polars 2.0's streaming engine runs the bounds' ordered join 4.6 to 5.5 times slower than in memory (#1857).

    The limit is 0, so the rest of the build is left to polars: a small one is in memory throughout.
    """
    monkeypatch.setattr(assembly, 'IN_MEMORY_REACH', 0)
    engines = {}
    original = pl.LazyFrame.collect

    def recording(self, *args, **kwargs):
        engines.setdefault(tuple(self.collect_schema().names()), set()).add(kwargs.get('engine'))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, 'collect', recording)
    sps.solve(dispatch_yaml, dispatch_table_inputs)
    assert engines[(COL, 'lb', 'ub')] == {'in-memory'}, f'every bounds collect names the in-memory engine: {engines}'


@pytest.fixture
def engines(monkeypatch) -> list[str]:
    """Every engine a collect of specsolve's names, in the order it named them.

    The engine is read where specsolve picks it rather than off polars' own
    ``collect``: polars 1.x runs an eager ``DataFrame.select`` as a lazy collect
    that names no engine, and in memory whatever it names.
    """
    named: list[str] = []
    original = collect._engine

    def recording() -> str:
        named.append(original())
        return named[-1]

    monkeypatch.setattr(collect, '_engine', recording)
    return named


def test_a_small_model_is_read_and_built_on_the_in_memory_engine(engines, dispatch_yaml, dispatch_table_inputs):
    """polars 2.0's streaming engine costs a fixed time per query, which is most of a small model's build."""
    sps.build(dispatch_yaml, dispatch_table_inputs)
    assert engines, 'the build collected something'
    assert set(engines) == {'in-memory'}, (
        f'{engines.count("auto")} of {len(engines)} collects left the engine to polars'
    )


def test_a_collect_after_a_build_is_left_to_polars_again(engines, dispatch_yaml, dispatch_table_inputs):
    """A block hands the engine back on exit, so the caller's own query after a small build is polars' choice."""
    sps.build(dispatch_yaml, dispatch_table_inputs)
    collected(pl.LazyFrame({'probe': [0]}))
    assert engines[-1] == 'auto', 'the query after the build still named the in-memory engine'


def _build_wide() -> None:
    """A constraint over dimensions no variable spans: four labels on each of two dimensions reach 8 columns and 16 rows."""
    labels = ['p', 'q', 'r', 's']
    sps.build(
        {
            'dimensions': {'a': {'dtype': 'str'}, 'b': {'dtype': 'str'}},
            'parameters': {'c': {'dims': ['a', 'b']}},
            'variables': {'x': {'dims': ['a']}, 'y': {'dims': ['b']}},
            'constraints': {'cap': {'dims': ['a', 'b'], 'expression': 'x + y <= c'}},
            'objective': {'sense': 'maximize', 'expression': 'sum(x) + sum(y)'},
        },
        {
            'a': labels,
            'b': labels,
            'c': pl.DataFrame({'a': [a for a in labels for _ in labels], 'b': labels * 4, 'value': 1.0}),
        },
    )


@pytest.mark.parametrize(
    ('limit', 'spec', 'left_to_polars'),
    [
        pytest.param(0, 'dispatch', True, id='columns-over-the-limit'),
        pytest.param(15, 'wide', True, id='rows-over-the-limit'),
        pytest.param(16, 'wide', False, id='rows-at-the-limit'),
    ],
)
def test_the_reach_against_the_limit_decides_the_build_engine(
    engines, monkeypatch, limit, spec, left_to_polars, dispatch_yaml, dispatch_table_inputs
):
    """Above `IN_MEMORY_REACH` the streaming engine keeps the build's peak down, so polars chooses; at it, memory.

    A constraint over dimensions no variable spans counts too: its 16 rows reach past
    a limit of 15 where its 8 columns do not.
    """
    monkeypatch.setattr(assembly, 'IN_MEMORY_REACH', limit)
    if spec == 'dispatch':
        sps.build(dispatch_yaml, dispatch_table_inputs)
    else:
        _build_wide()
    if left_to_polars:
        assert 'auto' in engines, 'the build left its engine to polars'
    else:
        assert set(engines) == {'in-memory'}, (
            f'{engines.count("auto")} of {len(engines)} collects left the engine to polars at the limit'
        )


TWO_ROWS = {
    'dimensions': {'t': {'dtype': 'int', 'ordered': True}},
    'parameters': {'load': {'dims': ['t']}},
    'variables': {'p': {'dims': ['t'], 'bounds': {'lower': 0}}, 'q': {'dims': ['t'], 'bounds': {'lower': 0}}},
    'constraints': {
        'balance': {'dims': ['t'], 'expression': 'p + q == load'},
        'cap': {'dims': ['t'], 'expression': 'p <= load'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p + 2 * q, over=t)'},
}
TWO_ROWS_DATA = {'t': [0, 1, 2], 'load': pl.DataFrame({'t': [0, 1, 2], 'value': [5.0, 4.0, 6.0]})}


@pytest.mark.parametrize(
    ('limit', 'order', 'shared'),
    [
        pytest.param(0, ['plan balance', 'read balance', 'plan cap', 'read cap'], False, id='above-the-reach'),
        pytest.param(10**9, ['plan balance', 'plan cap', 'read balance', 'read cap'], True, id='within-the-reach'),
    ],
)
def test_above_the_reach_a_pass_reads_one_entry_and_lets_it_go_before_the_next(monkeypatch, limit, order, shared):
    """A shared pass holds every entry's tables until it has read them all, which a large build cannot afford.

    Above `IN_MEMORY_REACH` each constraint's terms are read and made into its
    shares before the next constraint's are planned, and no two queries share a
    collect; within it every constraint's are planned, then read in one.
    """
    monkeypatch.setattr(assembly, 'IN_MEMORY_REACH', limit)
    events, batched = [], []
    planned, shares, together = (
        assembly.coverage.constant_side_planned,
        assembly.Assembly._shares,
        assembly.collected_all,
    )

    def spied_planned(scope, rows, terms, c):
        events.append(f'plan {"balance" if c.sense == "==" else "cap"}')
        return planned(scope, rows, terms, c)

    def spied_shares(self, p, *args):
        events.append(f'read {p.name}')
        return shares(self, p, *args)

    def spied_together(queries):
        batched.append(len(queries))
        return together(queries)

    monkeypatch.setattr(assembly.coverage, 'constant_side_planned', spied_planned)
    monkeypatch.setattr(assembly.Assembly, '_shares', spied_shares)
    monkeypatch.setattr(assembly, 'collected_all', spied_together)
    sps.build(TWO_ROWS, TWO_ROWS_DATA).close()
    assert events == order, f'the terms pass ran in the order {events}'
    assert any(n > 1 for n in batched) == shared, f'queries per collect_all: {batched}'
