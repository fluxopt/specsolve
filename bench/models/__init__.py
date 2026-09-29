"""One directory per case: the model in every dialect that can express it.

`bench/cases.py` holds what a case is: its ladder, its cardinalities, the
parquet its generator writes. Here is the model: `spec.yaml` for specsolve, and
one module per hand-written dialect, named in the case's `FORMULATIONS` map.
The two matrix arms share `matrix.py` and differ only in the solver they push
the `Lp` into.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import numpy as np


class Lp(NamedTuple):
    """A case as a matrix, in the vocabulary every solver's bulk API shares.

    Columns carry `lower`, `upper` and `obj`; rows carry `matrix`, one `senses`
    character each and `rhs`. An arm meeting a sense other than `'='` or `'<'`
    raises.

    Attributes:
        lower: Column lower bounds, one per column.
        upper: Column upper bounds, one per column.
        obj: Objective coefficient per column; the sense is always minimise.
        matrix: The constraint matrix, scipy CSR, rows x columns.
        senses: One of `'='` or `'<'` per row.
        rhs: The right-hand side, one per row.
    """

    lower: np.ndarray
    upper: np.ndarray
    obj: np.ndarray
    matrix: Any
    senses: np.ndarray
    rhs: np.ndarray


def formulation(case_name: str, dialect: str):
    """The case's model in *dialect*, or None where nobody has written one.

    What `build` takes is the arm's contract: `gurobipy-loop` passes an `Env`
    and the tables, the matrix dialects take the tables alone and return an
    `Lp`.
    """
    import importlib

    try:
        case = importlib.import_module(f'bench.models.{case_name}')
    except ModuleNotFoundError:
        return None
    return getattr(case, 'FORMULATIONS', {}).get(dialect)
