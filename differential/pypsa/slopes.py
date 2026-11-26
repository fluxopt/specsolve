"""The range of duals an LP admits on a row, measured as the objective's slope on each side of its right-hand side."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

#: How far a row's right-hand side moves to measure the objective's slope on
#: each side of it.
SLOPE_STEP = 1e-3


def slope_ranges(model, rows: set[int]) -> dict[int, tuple[float, float]]:
    """The objective's slope below and above each row's right-hand side, by row label.

    An LP whose optimum is degenerate has more than one optimal dual on a row:
    every value between these two one-sided slopes is one. The slopes are
    measured on HiGHS over *model*'s LP file, which leaves *model* as it is.
    *model* minimizes, as PyPSA's does: a side the row cannot move to without
    making the model infeasible has an infinite slope.
    """
    import highspy

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'model.lp'
        model.to_file(path)
        h = highspy.Highs()
        h.setOptionValue('output_flag', False)
        h.readModel(str(path))
        h.run()
        base = h.getInfo().objective_function_value
        ranges = {}
        for label in rows:
            _, index = h.getRowByName(f'c{label}')
            lp = h.getLp()
            lower, upper = lp.row_lower_[index], lp.row_upper_[index]
            slopes = []
            for step in (-SLOPE_STEP, SLOPE_STEP):
                h.changeRowBounds(
                    index, lower + step if lower > -math.inf else lower, upper + step if upper < math.inf else upper
                )
                h.run()
                optimal = h.getModelStatus() == highspy.HighsModelStatus.kOptimal
                slopes.append(
                    (h.getInfo().objective_function_value - base) / step if optimal else math.copysign(math.inf, step)
                )
            h.changeRowBounds(index, lower, upper)
            ranges[label] = (min(slopes), max(slopes))
    return ranges


def inside(value: float, bounds: tuple[float, float]) -> bool:
    """Whether *value* lies in *bounds*, to the 1e-6 the gate compares duals at, relative above 1."""
    low, high = bounds
    slack = 1e-6 * max(1.0, abs(value))
    return low - slack <= value <= high + slack
