"""How a frame is materialised: joins in the order specsolve wrote them, a coordinate product in label order."""

from __future__ import annotations

import polars as pl
import pytest

import specsolve as sps
from specsolve.relational import collect
from specsolve.relational.collect import collected
from specsolve.relational.engine import assembly
from specsolve.relational.engine.scope import Scope, ordinal


def _shift_shaped(snapshots: int, stores: int) -> pl.LazyFrame:
    """A constraint's rows joined to a variable moved one snapshot along, as a cyclic ``shift`` builds it."""
    keys = pl.DataFrame(
        {
            'snapshot': [s for s in range(snapshots) for _ in range(stores)],
            'store': [f's{k}' for _ in range(snapshots) for k in range(stores)],
        }
    )
    ordinals = pl.DataFrame({'snapshot': range(snapshots), 'ord': range(snapshots)}).lazy()
    moved = (
        keys.with_row_index('var_label')
        .lazy()
        .join(ordinals.rename({'ord': 'ord_in'}), on='snapshot')
        .with_columns(((pl.col('ord_in') + 1) % snapshots).alias('ord_out'))
        .drop('snapshot', 'ord_in')
        .join(ordinals.rename({'ord': 'ord_out'}), on='ord_out')
        .drop('ord_out')
    )
    return keys.with_row_index('row').lazy().join(moved, on=['snapshot', 'store']).select('row', 'var_label')


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


def test_a_coordinate_product_arrives_in_label_order():
    """The streaming engine picks which side of a cross join it buffers, and the product's order with it.

    A label is row-major over the declared ordinals, so a product out of that
    order has to be sorted before it is numbered.
    """
    spec = {
        'dimensions': {'a': {'dtype': 'int'}, 'b': {'dtype': 'int'}},
        'variables': {'x': {'dims': ['a', 'b']}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x)'},
    }
    with sps.build(spec, {'a': list(range(2)), 'b': list(range(50))}) as model:
        built = model._engine._model
        product = Scope(built.program, built.attached, built.variables).product(('a', 'b')).pipe(collected)
    ordinals = product.select(ordinal('a'), ordinal('b')).rows()
    assert ordinals == sorted(ordinals), 'row-major: `a` varies slowest'


def test_the_bounds_are_collected_in_memory(monkeypatch, dispatch_yaml, dispatch_frame_inputs):
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
    sps.solve(dispatch_yaml, dispatch_frame_inputs)
    assert engines[('var_label', 'lb', 'ub')] == {'in-memory'}, (
        f'every bounds collect names the in-memory engine: {engines}'
    )


@pytest.fixture
def engines(monkeypatch) -> list[str]:
    """Every engine a collect names, in the order it named them, from after polars is probed for streaming."""
    collect.collect_engine()
    named: list[str] = []
    one, many = pl.LazyFrame.collect, pl.collect_all

    def recording(self, *args, engine='auto', **kwargs):
        named.append(engine)
        return one(self, *args, engine=engine, **kwargs)

    def recording_all(frames, *args, engine='auto', **kwargs):
        named.append(engine)
        return many(frames, *args, engine=engine, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, 'collect', recording)
    monkeypatch.setattr(pl, 'collect_all', recording_all)
    return named


def test_a_small_model_is_read_and_built_on_the_in_memory_engine(engines, dispatch_yaml, dispatch_frame_inputs):
    """polars 2.0's streaming engine costs a fixed time per query, which is most of a small model's build."""
    sps.build(dispatch_yaml, dispatch_frame_inputs)
    assert engines, 'the build collected something'
    assert set(engines) == {'in-memory'}, (
        f'{engines.count("auto")} of {len(engines)} collects left the engine to polars'
    )


def test_a_collect_after_a_build_is_left_to_polars_again(engines, dispatch_yaml, dispatch_frame_inputs):
    """A block hands the engine back on exit, so the caller's own query after a small build is polars' choice."""
    sps.build(dispatch_yaml, dispatch_frame_inputs)
    collected(pl.LazyFrame({'probe': [0]}))
    assert engines[-1] == 'auto', 'the query after the build still named the in-memory engine'


@pytest.mark.parametrize(
    ('limit', 'spec'),
    [
        pytest.param(0, 'dispatch', id='columns-over-the-limit'),
        pytest.param(10, 'wide', id='rows-over-the-limit'),
    ],
)
def test_a_model_over_the_limit_is_built_on_polars_choice(
    engines, monkeypatch, limit, spec, dispatch_yaml, dispatch_frame_inputs
):
    """Above `IN_MEMORY_REACH` the streaming engine keeps the build's peak down, so polars chooses.

    A constraint over dims no variable spans counts too: four labels on each of
    two dims reach 8 columns and 16 rows.
    """
    monkeypatch.setattr(assembly, 'IN_MEMORY_REACH', limit)
    if spec == 'dispatch':
        sps.build(dispatch_yaml, dispatch_frame_inputs)
    else:
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
    assert 'auto' in engines, 'the build left its engine to polars'
