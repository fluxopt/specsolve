"""`storage` as a matrix.

Columns are one snapshot's ``[p | charge | discharge | soc]`` repeated per
snapshot. A state of charge reaches the previous snapshot, so its rows add a
`kron` against the cyclic shift matrix:

    kron(I, own) + kron(P, previous)     where P[t, t-1 mod n] = 1

The wrap row of `P` is ``edge='wrap'`` in the YAML.
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
    e_max = tables['e_max']['value'].to_numpy()
    p_store = tables['p_store']['value'].to_numpy()
    eta = tables['eta']['value'].to_numpy()

    n_snapshot, n_generator, n_store = len(load), len(p_max), len(p_store)
    width = n_generator + 3 * n_store
    generators = slice(0, n_generator)
    charge = slice(n_generator, n_generator + n_store)
    discharge = slice(n_generator + n_store, n_generator + 2 * n_store)
    state = slice(n_generator + 2 * n_store, width)

    balance = np.zeros((1, width))
    balance[0, generators] = 1.0
    balance[0, discharge] = 1.0
    balance[0, charge] = -1.0

    own = np.zeros((n_store, width))
    own[np.arange(n_store), np.arange(state.start, state.stop)] = 1.0
    own[np.arange(n_store), np.arange(charge.start, charge.stop)] = -eta
    own[np.arange(n_store), np.arange(discharge.start, discharge.stop)] = 1.0

    earlier = np.zeros((n_store, width))
    earlier[np.arange(n_store), np.arange(state.start, state.stop)] = -1.0

    ring = sparse.eye(n_snapshot, format='csr')[np.arange(-1, n_snapshot - 1)]
    matrix = sparse.vstack(
        [
            sparse.kron(sparse.eye(n_snapshot), balance, format='csr'),
            sparse.kron(sparse.eye(n_snapshot), own, format='csr') + sparse.kron(ring, earlier, format='csr'),
        ],
        format='csr',
    )

    return Lp(
        lower=np.zeros(n_snapshot * width),
        upper=np.tile(np.concatenate([p_max, p_store, p_store, e_max]), n_snapshot),
        obj=np.tile(np.concatenate([cost, np.zeros(3 * n_store)]), n_snapshot),
        matrix=matrix,
        senses=np.full(matrix.shape[0], '='),
        rhs=np.concatenate([load, np.zeros(n_snapshot * n_store)]),
    )
