"""`network` as a matrix.

Columns are one snapshot's ``[p | f | charge | discharge | soc]`` repeated per
snapshot, where ``p`` is the installed (bus, tech) pairs in the order the
`installed` table lists them; an absent pair has no column. The balance is
``kron(I, B)`` over a bus x column block ``B``, and a state of charge reaches
the previous snapshot, and the first reaches none, which is the YAML's
`edge=0`:

    kron(I, own) + kron(P, previous)     where P[t, t-1] = 1 for t > 0

`p_max`, `load` are read in file order, which `_network_data` writes
snapshot-major, with the installed pairs and the buses in their tables' order.
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
    cap = tables['cap']['value'].to_numpy()
    neg_cap = tables['neg_cap']['value'].to_numpy()
    e_max = tables['e_max']['value'].to_numpy()
    p_store = tables['p_store']['value'].to_numpy()
    eta = tables['eta']['value'].to_numpy()
    load = tables['load']['value'].to_numpy()

    bus_at = {b: i for i, b in enumerate(tables['bus']['bus'])}
    n_bus, n_pair, n_line, n_store = len(bus_at), len(installed), len(cap), len(p_store)
    n_snapshot = len(load) // n_bus
    width = n_pair + n_line + 3 * n_store
    lines = n_pair + np.arange(n_line)
    charge = n_pair + n_line + np.arange(n_store)
    discharge = charge + n_store
    state = discharge + n_store

    balance = np.zeros((n_bus, width))
    balance[[bus_at[b] for b in installed['bus']], np.arange(n_pair)] = 1.0
    balance[[bus_at[b] for b in tables['line_to']['bus']], lines] += 1.0
    balance[[bus_at[b] for b in tables['line_from']['bus']], lines] -= 1.0
    store_bus = [bus_at[b] for b in tables['store_bus']['bus']]
    balance[store_bus, discharge] = 1.0
    balance[store_bus, charge] = -1.0

    own = np.zeros((n_store, width))
    own[np.arange(n_store), state] = 1.0
    own[np.arange(n_store), charge] = -eta
    own[np.arange(n_store), discharge] = 1.0
    earlier = np.zeros((n_store, width))
    earlier[np.arange(n_store), state] = -1.0

    before = sparse.eye(n_snapshot, k=-1, format='csr')
    matrix = sparse.vstack(
        [
            sparse.kron(sparse.eye(n_snapshot), sparse.csr_matrix(balance), format='csr'),
            sparse.kron(sparse.eye(n_snapshot), own, format='csr') + sparse.kron(before, earlier, format='csr'),
        ],
        format='csr',
    )

    pair_cost = np.fromiter((cost[t] for t in installed['tech']), dtype=np.float64, count=n_pair)
    zeros = np.zeros(n_store)
    upper = np.hstack(
        [
            tables['p_max']['value'].to_numpy().reshape(n_snapshot, n_pair),
            np.tile(np.concatenate([cap, p_store, p_store, e_max]), (n_snapshot, 1)),
        ]
    )
    return Lp(
        lower=np.tile(np.concatenate([np.zeros(n_pair), neg_cap, zeros, zeros, zeros]), n_snapshot),
        upper=upper.reshape(-1),
        obj=np.tile(np.concatenate([pair_cost, np.zeros(n_line + 3 * n_store)]), n_snapshot),
        matrix=matrix,
        senses=np.full(matrix.shape[0], '='),
        rhs=np.concatenate([load, np.zeros(n_snapshot * n_store)]),
    )
