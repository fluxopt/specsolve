"""`transport` as a matrix: one incidence block, tiled per snapshot.

Columns are one snapshot's generators followed by its lines, repeated per
snapshot, so the balance is ``kron(I(n_snapshot), [A_generator | A_line])``.

The load vector is read in file order, which `_transport_data` writes
snapshot-major with buses within.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bench.models import Lp

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


def _incidence(rows: Any, columns: Any, at: Any, sign: float, shape: tuple[int, int]) -> Any:
    """One ``bus x entity`` block: ``sign`` where the entity touches the bus."""
    from scipy import sparse

    position = {b: i for i, b in enumerate(rows)}
    row = np.fromiter((position[b] for b in at), dtype=np.int64, count=len(columns))
    return sparse.csr_matrix((np.full(len(columns), sign), (row, np.arange(len(columns)))), shape=shape)


def build(tables: Mapping[str, Any]) -> Lp:
    from scipy import sparse

    p_max = tables['p_max']['value'].to_numpy()
    cost = tables['cost']['value'].to_numpy()
    cap = tables['cap']['value'].to_numpy()
    neg_cap = tables['neg_cap']['value'].to_numpy()
    load = tables['load']['value'].to_numpy()

    buses = tables['bus']['bus'].to_list()
    generators = tables['generator']['generator'].to_list()
    lines = tables['line']['line'].to_list()
    n_bus, n_generator, n_line = len(buses), len(generators), len(lines)
    n_snapshot = len(load) // n_bus

    block = sparse.hstack(
        [
            _incidence(buses, generators, tables['gen_bus']['bus'], 1.0, (n_bus, n_generator)),
            _incidence(buses, lines, tables['line_to']['bus'], 1.0, (n_bus, n_line))
            - _incidence(buses, lines, tables['line_from']['bus'], 1.0, (n_bus, n_line)),
        ],
        format='csr',
    )

    return Lp(
        lower=np.tile(np.concatenate([np.zeros(n_generator), neg_cap]), n_snapshot),
        upper=np.tile(np.concatenate([p_max, cap]), n_snapshot),
        obj=np.tile(np.concatenate([cost, np.zeros(n_line)]), n_snapshot),
        matrix=sparse.kron(sparse.eye(n_snapshot), block, format='csr'),
        senses=np.full(len(load), '='),
        rhs=load,
    )
