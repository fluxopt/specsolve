"""The `gurobipy-matrix` arm, the floor under the `gurobi` sink: `bench.arms.gurobipy`'s verbs over the `gurobipy-matrix` formulations.

The published number is `load`: one `loadModel` over arrays `load_setup` built
outside the clock, so it is what handing a model to Gurobi costs and nothing
before that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bench.arms import gurobipy as runtime

if TYPE_CHECKING:
    from collections.abc import Mapping

DIALECT = 'gurobipy-matrix'
SINKS = runtime.SINKS

#: The runtime's, plus `scipy` for the CSR the shared formulation builds.
REQUIRES = (*runtime.REQUIRES, 'scipy')

build_and_emit = runtime.build_and_emit
build_only = runtime.build_only
objective = runtime.objective
written = runtime.written
load = runtime.matrix_load


def load_setup(sink: str, prepared: runtime.Prepared) -> tuple[tuple[runtime.Arrays], dict[str, Any]]:
    """The finished arrays `load` hands over, built before the clock starts."""
    del sink
    return (runtime.matrix_arrays(prepared.case_name, runtime.tables(prepared)),), {}


def prepare(case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]) -> Any:
    return runtime.prepare(DIALECT, case_name, size, paths, options)
