"""What every solver member promises about release, asked of each.

The contract is :meth:`~specsolve._relational.sinks.solvers.base.Solver.close`'s
docstring; this is where it is held to, one row per member of ``SOLVERS``.
The ``solver_name`` fixture skips a member this environment cannot run.
"""

from __future__ import annotations

import gc
import weakref

import specsolve as sps
from specsolve._relational.sinks import SOLVERS
from tests.conftest import CASES


def _tables(solver_name: str):
    del solver_name
    with sps.build(*CASES['LP']) as model:
        return model._engine._model.handoff


def test_close_leaves_no_model(solver_name: str) -> None:
    """``close()`` is the release, leaving a ``with``, and it is idempotent."""
    with SOLVERS[solver_name](_tables(solver_name)) as solver:
        assert solver.handle is not None, 'a loaded solver exposes the native object it holds'
    assert solver.handle is None, 'after close the handle is gone from the holder'
    solver.close()


def test_a_dropped_holder_releases_its_model(solver_name: str) -> None:
    """A holder dropped without ``close()`` still releases what it holds.

    Asked through a weak reference to the native object: the member whose
    library releases on collection passes by refcount, the one with a
    finalizer passes because the finalizer holds the objects and not the
    holder. Either way nothing in this package keeps the model alive.
    """
    solver = SOLVERS[solver_name](_tables(solver_name))
    reference = weakref.ref(solver.handle)
    del solver
    gc.collect()
    assert reference() is None, f'{solver_name}: the native model outlived the holder that was dropped'
