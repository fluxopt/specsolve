"""Every arm, and the four verbs the harness asks of one.

An arm is one library's answer to the same question: same parquet, same model,
same seam. Adding one is a module here and an entry in `ARMS`.

Each arm module defines:

    prepare(case_name, size, paths, options) -> Prepared
    build_and_emit(sink, prepared) -> Counts
    build_only(prepared) -> Counts
    objective(prepared) -> float

and, where the library has a rolling-horizon answer, both or neither of:

    window_setup(sink, prepared, following) -> (args, kwargs)
    window(*args, **kwargs) -> Counts

``window_setup`` is pytest-benchmark's pedantic ``setup``: it runs untracked in
the spawned child before each sample, and what it returns feeds ``window``, the
one window that gets timed — the one *following* prepares. An arm whose
``Counts`` carry ``reloaded`` says whether that window loaded its solver from
scratch, and the harness holds it to the change the window made.

``Prepared`` is opaque to the harness. ``prepare`` runs before the clock, so
work the harness rather than the library imposes is charged to nobody.

Every verb is top-level and picklable, because ``benchmem(isolate=True)`` sends
it to a fresh process; pre-clock state is built by ``window_setup`` in the
child, never returned as a closure (#1617).

The library is imported inside the verb, never at module scope, because the
import is part of what an arm costs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bench.arms import gurobipy_loop, gurobipy_matrix, highspy_matrix, linopy, pyomo, specsolve

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

#: What every verb returns: the counts the published tables carry, read after the action.
Counts = dict[str, Any]

#: Name to the module that speaks for it.
ARMS: dict[str, ModuleType] = {
    'specsolve': specsolve,
    'linopy': linopy,
    'pyomo': pyomo,
    'gurobipy-loop': gurobipy_loop,
    'gurobipy-matrix': gurobipy_matrix,
    'highspy-matrix': highspy_matrix,
}


def unmeasurable(arm: str, case_name: str, sink: str) -> str | None:
    """Why this cell is not measured, or None when it is.

    The reason is a missing library, an unreachable sink, or a case with no
    formulation in the arm's dialect.
    """
    import importlib.util

    module = ARMS[arm]
    absent = [r for r in getattr(module, 'REQUIRES', ()) if importlib.util.find_spec(r) is None]
    if absent:
        return f'{arm} needs {", ".join(absent)}, which this environment does not have'
    if sink not in module.SINKS:
        return f'{arm} does not reach the {sink} sink — it has {", ".join(module.SINKS)}'
    dialect = getattr(module, 'DIALECT', None)
    if dialect is not None:
        from bench.models import formulation

        if formulation(case_name, dialect) is None:
            return f'{case_name} has no {dialect} formulation'
    return None


def solved(arm: str, case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]) -> float:
    """Prepare and solve on *arm*, unmeasured — what `bench.floor` checks its own answer against."""
    module = ARMS[arm]
    return float(module.objective(module.prepare(case_name, size, paths, options)))
