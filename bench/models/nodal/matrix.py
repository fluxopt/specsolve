"""`nodal` as a matrix: a column per installed pair, tiled per snapshot.

Columns are one snapshot's installed (node, tech) pairs, in the order the
`installed` table lists them, repeated per snapshot. An absent pair has no
column, so the matrix is as sparse as the portfolio. The balance is
``kron(I(n_snapshot), B)``, where ``B`` is the ``node x pair`` block with a 1
where the pair sits at the node.

The demand vector is read in file order, which `_nodal_data` writes
snapshot-major with nodes within, in the order of the `node` table.
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
    cost = dict(zip(tables['cost']['tech'], tables['cost']['value'], strict=True))
    demand = tables['demand']['value'].to_numpy()

    nodes = tables['node']['node'].to_list()
    position = {n: i for i, n in enumerate(nodes)}
    n_pair, n_node = len(installed), len(nodes)
    n_snapshot = len(demand) // n_node

    row = np.fromiter((position[n] for n in installed['node']), dtype=np.int64, count=n_pair)
    block = sparse.csr_matrix((np.ones(n_pair), (row, np.arange(n_pair))), shape=(n_node, n_pair))

    return Lp(
        lower=np.zeros(n_snapshot * n_pair),
        upper=np.tile(installed['value'].to_numpy(), n_snapshot),
        obj=np.tile(np.fromiter((cost[t] for t in installed['tech']), dtype=np.float64, count=n_pair), n_snapshot),
        matrix=sparse.kron(sparse.eye(n_snapshot), block, format='csr'),
        senses=np.full(n_snapshot * n_node, '='),
        rhs=demand,
    )
