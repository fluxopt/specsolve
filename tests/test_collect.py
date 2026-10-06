"""How a frame is materialised: every plan runs its joins in the order specsolve wrote them."""

from __future__ import annotations

import polars as pl

from specsolve.relational.collect import collected


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
