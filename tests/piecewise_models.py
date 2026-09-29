"""The curves the piecewise tests solve.

A copy of `mathspec`'s test models, since a test package is not shipped.
Watch it for drift.
"""

from __future__ import annotations

import polars as pl

from tests.fixtures import override, raw_of

NONCONVEX_YAML = """
dimensions:
  snapshot: {dtype: int}
  bp: {dtype: int}

parameters:
  load: {dims: [snapshot]}
  bp_x: {dims: [bp]}
  bp_y: {dims: [bp]}

variables:
  p:
    dims: [snapshot]
    bounds: {lower: 0, upper: 100}
  op_cost:
    dims: [snapshot]
    bounds: {lower: 0}

piecewise:
  cost_curve:
    over: bp
    links:
      - [p, bp_x]
      - [op_cost, bp_y]

constraints:
  balance:
    dims: [snapshot]
    expression: p == load

objective:
  sense: minimize
  expression: sum(op_cost, over=snapshot)
"""
#: The same restriction said as a set; it reaches the default's optimum on every sink.
SOS2_SPEC = override(raw_of(NONCONVEX_YAML), **{'piecewise.cost_curve.method': 'sos2'})
#: two dims in the frame, so the emitted ``dims`` has an order to get wrong.
TWO_DIM_YAML = """
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}
  bp: {dtype: int}

parameters:
  load: {dims: [snapshot]}
  bp_x: {dims: [generator, bp]}
  bp_y: {dims: [generator, bp]}

variables:
  p:
    dims: [snapshot, generator]
    bounds: {lower: 0, upper: 100}
  op_cost:
    dims: [snapshot, generator]
    bounds: {lower: 0}

piecewise:
  cost_curve:
    over: bp
    links:
      - [p, bp_x]
      - [op_cost, bp_y]

constraints:
  balance:
    dims: [snapshot]
    expression: sum(p, over=generator) == load

objective:
  sense: minimize
  expression: sum(sum(op_cost, over=generator), over=snapshot)
"""
CHP_YAML = """
dimensions:
  snapshot: {dtype: int}
  bp: {dtype: int}

parameters:
  load: {dims: [snapshot]}
  power_bp: {dims: [bp]}
  fuel_bp: {dims: [bp]}
  heat_bp: {dims: [bp]}

variables:
  power:
    dims: [snapshot]
    bounds: {lower: 0, upper: 100}
  fuel:
    dims: [snapshot]
    bounds: {lower: 0}
  heat:
    dims: [snapshot]
    bounds: {lower: 0}

piecewise:
  chp:
    over: bp
    links:
      - [power, power_bp]
      - [fuel, fuel_bp]
      - [heat, heat_bp]

constraints:
  balance:
    dims: [snapshot]
    expression: power == load

objective:
  sense: minimize
  expression: sum(fuel, over=snapshot)
"""
GATED_YAML = """
dimensions:
  snapshot: {dtype: int}
  bp: {dtype: int}

parameters:
  load: {dims: [snapshot]}
  on_flag: {dims: [snapshot]}
  bp_x: {dims: [bp]}
  bp_y: {dims: [bp]}

variables:
  u:
    dims: [snapshot]
    domain: binary
  p:
    dims: [snapshot]
    bounds: {lower: 0, upper: 100}
  op_cost:
    dims: [snapshot]
    bounds: {lower: 0}

piecewise:
  cost_curve:
    over: bp
    links:
      - [p, bp_x]
      - [op_cost, bp_y]
    activity: u

constraints:
  commit:
    dims: [snapshot]
    expression: u == on_flag
  balance:
    dims: [snapshot]
    expression: p == load * on_flag

objective:
  sense: minimize
  expression: sum(op_cost, over=snapshot)
"""


LP_SPEC = """
description: dispatch whose cost is read off a convex curve, stated as its segment lines

dimensions:
  snapshot: {dtype: int, description: dispatch periods}
  bp: {dtype: int, description: breakpoints of the cost curve}

parameters:
  load: {dims: [snapshot], description: demand to be met}
  bp_x: {dims: [bp], description: breakpoint output levels}
  bp_y: {dims: [bp], description: cost at each breakpoint}

variables:
  p:
    dims: [snapshot]
    bounds: {lower: 0, upper: 100}
    description: dispatched power
  op_cost:
    dims: [snapshot]
    bounds: {lower: 0}
    description: operating cost, read off the curve
  running:
    dims: [snapshot]
    domain: binary
    description: unused here; a gate for the case lp cannot take one

piecewise:
  cost_curve:
    description: cost bounded below by the curve, which is exact where the curve is convex
    over: bp
    links:
      - [p, bp_x]
      - [op_cost, bp_y, '>=']
    method: lp

constraints:
  balance:
    dims: [snapshot]
    expression: p == load
    description: output meets demand

objective:
  sense: minimize
  expression: sum(op_cost, over=snapshot)
  description: total operating cost
"""


def curve_frame(points: dict) -> pl.DataFrame:
    """A per-generator curve ``{(generator, bp): value}`` as the tidy frame it is supplied as."""
    return pl.DataFrame(
        {
            'generator': [g for g, _ in points],
            'bp': [k for _, k in points],
            'value': list(points.values()),
        }
    )
