"""`dispatch` as a matrix: one column block, one balance row per snapshot.

What a performance-minded user of either solver's bulk API writes. The balance
matrix is ``kron(I(n_snapshot), ones(1, n_generator))``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bench.models import Lp

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


def build(tables: Mapping[str, Any]) -> Lp:
    from scipy import sparse

    p_max = tables['p_max']['value'].to_numpy()
    cost = tables['cost']['value'].to_numpy()
    load = tables['load']['value'].to_numpy()

    live = p_max > 0
    p_max, cost = p_max[live], cost[live]
    n_snapshot, n_generator = len(load), len(p_max)

    return Lp(
        lower=np.zeros(n_snapshot * n_generator),
        upper=np.tile(p_max, n_snapshot),
        obj=np.tile(cost, n_snapshot),
        matrix=sparse.kron(sparse.eye(n_snapshot), np.ones((1, n_generator)), format='csr'),
        senses=np.full(n_snapshot, '='),
        rhs=load,
    )
