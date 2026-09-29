"""Expansion means the same on both lanes.

The rules for `macros:` and `expressions:` live with the language's own suite
(#1150). Both constructs expand to core AST before dispatch, so one end-to-end
case on which the lanes agree covers them.
"""

from __future__ import annotations

import numpy as np

from tests.conftest import DISPATCH_COST, DISPATCH_GENERATORS, DISPATCH_P_MAX
from tests.differential import differential
from tests.oracle import pd

#: The same model the language's own expansion suite expands, copied rather
#: than imported: that suite travels with the language and this one does not.
EXPANSION_YAML = """
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}
parameters:
  p_max: {dims: [generator]}
  cost: {dims: [generator]}
  load: {dims: [snapshot]}
expressions:
  total_generation: sum(p, over=generator)
macros:
  weighted_sum:
    args: [array, weights]
    kwargs: [over]
    template: sum(array * weights, over=over)
variables:
  p:
    dims: [snapshot, generator]
    where: "p_max > 0"
    bounds: {lower: 0, upper: p_max}
constraints:
  balance:
    dims: [snapshot]
    expression: total_generation == load
objective:
  sense: minimize
  expression: sum(weighted_sum(p, cost, over=generator))
"""


def test_a_macro_and_a_named_expression_mean_the_same_on_both_lanes():
    rng = np.random.default_rng(5)
    n_s = 24
    data = {
        'p_max': pd.Series(dict(zip(DISPATCH_GENERATORS, DISPATCH_P_MAX, strict=True))),
        'cost': pd.Series(dict(zip(DISPATCH_GENERATORS, DISPATCH_COST, strict=True))),
        'load': pd.Series(
            (rng.uniform(0.2, 0.8, n_s) * sum(DISPATCH_P_MAX)).round(3),
            index=pd.RangeIndex(n_s, name='snapshot'),
        ),
    }
    index = {'snapshot': pd.RangeIndex(n_s, name='snapshot'), 'generator': list(DISPATCH_GENERATORS)}

    with differential(EXPANSION_YAML, data | index):
        pass  # agreement on the objective is the whole assertion
