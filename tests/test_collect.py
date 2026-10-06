"""How a frame is materialised: on an engine sized to the model, joins in the order written, a product in label order."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import specsolve as sps
from specsolve.relational import collect
from specsolve.relational.collect import collected
from specsolve.relational.engine.assembly import Assembly
from specsolve.relational.engine.scope import Scope, ordinal
from tests.conftest import expanded, port_sources

if TYPE_CHECKING:
    from specsolve.relational.sinks.handoff import Handoff

#: The two engines round ``a * b / c`` differently, by a unit in the last place.
ROUNDING = 1e-12


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


def test_every_port_builds_the_same_model_on_either_engine(port: dict[str, Any], monkeypatch):
    """A model below [`IN_MEMORY_BELOW`][] builds in memory, every larger one streams; both hand over one model.

    Every port is small, so the suite's own builds take the in-memory engine;
    this is where the streaming one is held to the same answer. Equal to
    rounding, not to the bit: the engines associate ``a * b / c`` differently.
    """

    def handoff(threshold: int) -> Handoff:
        monkeypatch.setattr(collect, 'IN_MEMORY_BELOW', threshold)
        with sps.build(expanded(port['spec']), port_sources(port['name'])) as model:
            return model._engine._model.handoff

    small, streamed = handoff(collect.IN_MEMORY_BELOW), handoff(0)
    for field in ('cols', 'rows', 'matrix', 'quad', 'qmatrix', 'sos'):
        assert_frame_equal(getattr(small, field), getattr(streamed, field), rel_tol=ROUNDING, abs_tol=ROUNDING)
    assert_frame_equal(small.obj.sort('col'), streamed.obj.sort('col'), rel_tol=ROUNDING, abs_tol=ROUNDING)
    assert np.array_equal(small.row_starts, streamed.row_starts), 'one CSR row span on both engines'


@pytest.mark.parametrize(
    ('coordinates', 'engine'),
    [
        pytest.param(collect.IN_MEMORY_BELOW - 1, 'in-memory', id='below'),
        pytest.param(collect.IN_MEMORY_BELOW, 'streaming', id='at'),
    ],
)
def test_the_engine_is_sized_to_the_model(coordinates, engine, monkeypatch):
    ran = []
    original = pl.LazyFrame.collect

    def recording(self, *args, **kwargs):
        ran.append(kwargs['engine'])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, 'collect', recording)
    with collect.sized(coordinates):
        collected(pl.LazyFrame({'a': [1]}))
    collected(pl.LazyFrame({'a': [1]}))
    assert ran == [engine, 'streaming'], 'sized chooses inside its block, and the default returns after it'


@pytest.mark.parametrize(
    ('threshold', 'engine'),
    [
        pytest.param(collect.IN_MEMORY_BELOW, 'in-memory', id='small'),
        pytest.param(100, 'streaming', id='at-the-threshold'),
    ],
)
def test_a_build_collects_on_the_engine_its_largest_declaration_sizes(threshold, engine, monkeypatch):
    """The largest declaration here is ``x`` over ``a`` and ``b``: one hundred coordinates."""
    seen = []
    original = Assembly.run

    def run(self):
        seen.append(collect._engine())
        return original(self)

    monkeypatch.setattr(Assembly, 'run', run)
    monkeypatch.setattr(collect, 'IN_MEMORY_BELOW', threshold)
    spec = {
        'dimensions': {'a': {'dtype': 'int'}, 'b': {'dtype': 'int'}},
        'variables': {'x': {'dims': ['a', 'b']}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x)'},
    }
    with sps.build(spec, {'a': list(range(2)), 'b': list(range(50))}):
        pass
    assert seen == [engine], 'one build, on the engine its size chose'
