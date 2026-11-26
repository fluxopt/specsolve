"""The `gurobipy-matrix` arm, the floor under the `gurobi` sink: `bench.arms.gurobipy`'s verbs over the `gurobipy-matrix` formulations.

The published number is `load`: one `loadModel` over the arrays `prepare`
built and `load_setup` read back, so it is what handing a model to Gurobi costs
and nothing before that.
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
    """The finished arrays `load` hands over, read back from where `prepare` wrote them."""
    from bench.arms import read_finished

    del sink
    return (read_finished(prepared.paths, DIALECT, runtime.Arrays),), {}


def prepare(case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]) -> runtime.Prepared:
    """Where the parquet is, with the finished arrays built from it and written beside it."""
    from bench.arms import write_finished

    prepared = runtime.prepare(DIALECT, case_name, size, paths, options)
    write_finished(prepared.paths, DIALECT, runtime.matrix_arrays(case_name, runtime.tables(prepared)))
    return prepared
