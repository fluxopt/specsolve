"""The `gurobipy-loop` arm: `bench.arms.gurobipy`'s verbs over the `gurobipy-loop` formulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bench.arms import gurobipy as runtime

if TYPE_CHECKING:
    from collections.abc import Mapping

DIALECT = 'gurobipy-loop'
SINKS = runtime.SINKS
REQUIRES = runtime.REQUIRES

build_and_emit = runtime.build_and_emit
build_only = runtime.build_only
objective = runtime.objective


def prepare(case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]) -> Any:
    return runtime.prepare(DIALECT, case_name, size, paths, options)
