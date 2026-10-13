"""`sector` as a matrix: a column per installed pair, tiled per snapshot.

Columns are one snapshot's installed (node, tech) pairs, in the order the
`installed` table lists them, repeated per snapshot. Rows are one snapshot's
served (node, carrier) pairs, repeated per snapshot. The balance is
``kron(I(n_snapshot), B)``, where ``B`` is the ``served x pair`` block holding
the `produces` factor where the pair's technology delivers the row's carrier.

The served pairs and the demand vector are read off the demand table in file
order, which `_sector_data` writes snapshot-major with nodes, then carriers,
within, and the same served pairs in every snapshot.
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

    installed = tables['installed']
    produces = tables['produces']
    demand = tables['demand']
    cost = dict(zip(tables['cost']['tech'], tables['cost']['value'], strict=True))

    n_pair = len(installed)
    n_snapshot = len(tables['snapshot'])
    n_served = len(demand) // n_snapshot
    served = {
        key: i
        for i, key in enumerate(
            zip(demand['node'].to_list()[:n_served], demand['carrier'].to_list()[:n_served], strict=True)
        )
    }

    rows, columns, factors = [], [], []
    delivers: dict[Any, list[tuple[Any, float]]] = {}
    for tech, carrier, factor in zip(
        produces['tech'].to_list(), produces['carrier'].to_list(), produces['value'].to_list(), strict=True
    ):
        delivers.setdefault(tech, []).append((carrier, factor))
    for j, (node, tech) in enumerate(zip(installed['node'].to_list(), installed['tech'].to_list(), strict=True)):
        for carrier, factor in delivers.get(tech, ()):
            rows.append(served[node, carrier])
            columns.append(j)
            factors.append(factor)
    block = sparse.csr_matrix((factors, (rows, columns)), shape=(n_served, n_pair))

    return Lp(
        lower=np.zeros(n_snapshot * n_pair),
        upper=np.tile(installed['value'].to_numpy(), n_snapshot),
        obj=np.tile(np.fromiter((cost[t] for t in installed['tech']), dtype=np.float64, count=n_pair), n_snapshot),
        matrix=sparse.kron(sparse.eye(n_snapshot), block, format='csr'),
        senses=np.full(n_snapshot * n_served, '='),
        rhs=demand['value'].to_numpy(),
    )
