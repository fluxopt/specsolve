"""What the ``linopy`` export can ingest, asked without importing linopy.

The export itself is ``specsolve.linopy``, outside the engine, because it
builds linopy and xarray objects. What it refuses is read off the hand-off
alone, so it lives here, where [`check`][specsolve.Model.check] reaches it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from specsolve.relational.sinks.capabilities import Capabilities

if TYPE_CHECKING:
    from specsolve.relational.sinks.handoff import Declared, Handoff

__all__ = ['LINOPY_CAPABILITIES', 'refusal']

#: linopy has no quadratic constraint; it holds every other construct.
LINOPY_CAPABILITIES = Capabilities(
    supports=frozenset({'integrality', 'sos', 'quadratic_objective', 'nonconvex_quadratic_objective'})
)


def refusal(handoff: Handoff, declared: Declared) -> str | None:
    """What linopy would lose from the built model in *handoff*, or ``None`` where it loses nothing.

    linopy's objective holds no constant, and linopy drops a row without
    terms, so ``0 >= 10`` would vanish and an infeasible model would solve. A
    termless row every point meets drops without changing the answer.
    """
    if handoff.objective_sense is not None and handoff.objective_constant:
        return (
            f"the objective has the constant {handoff.objective_constant!r}, and linopy's objective holds none. "
            'Move the constant out of the objective, or solve the model directly.'
        )
    empty = np.flatnonzero(np.diff(handoff.row_starts) == 0)
    if not empty.size:
        return None
    rows = handoff.rows[empty]
    sense, rhs = rows.get_column('sense').cast(pl.String).to_numpy(), rows.get_column('rhs').to_numpy()
    unmet = ((sense == '<=') & (rhs < 0)) | ((sense == '>=') & (rhs > 0)) | ((sense == '==') & (rhs != 0))
    if not unmet.any():
        return None
    name, coordinate = _owner(declared, int(empty[np.flatnonzero(unmet)[0]]))
    return (
        f"constraint '{name}' at {coordinate} has no term left and reads 0 {sense[unmet][0]} {rhs[unmet][0]}, "
        'which no point meets. linopy drops a row without terms, so the exported model would solve. '
        'Solve the model directly to read the infeasibility.'
    )


def _owner(declared: Declared, row: int) -> tuple[str, dict[str, object]]:
    """The constraint that owns *row*, and the row's coordinate; every row has one."""
    run = next(r for r in declared.constraints if r.start <= row < r.start + r.height)
    return run.name, run.coordinates.row(row - run.start, named=True) if run.dims else {}
