"""vtype: a unit-commitment MILP through both backends.

Binary commitment variables u with p <= p_max * u and a fixed commitment
cost. Verifies the relational backend's vtype path end to end: cols vtype
column, HiGHS changeColsIntegrality in the `highs` solver, and the LP binary
section.
"""

from __future__ import annotations

from typing import get_args

import numpy as np
import polars as pl
import pytest
from mathspec import program

from specsolve.relational.sinks.solvers.highs import Highs
from tests.differential import differential

COMMITMENT_YAML = """
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}

parameters:
  p_max: {dims: [generator]}
  cost: {dims: [generator]}
  fix_cost: {dims: [generator]}
  load: {dims: [snapshot]}

variables:
  u:
    dims: [snapshot, generator]
    domain: binary
  p:
    dims: [snapshot, generator]
    bounds: {lower: 0, upper: p_max}

constraints:
  commitment:
    dims: [snapshot, generator]
    expression: p <= p_max * u
  balance:
    dims: [snapshot]
    expression: sum(p, over=generator) == load

objective:
  sense: minimize
  expression: sum(p * cost) + sum(u * fix_cost)
"""


@pytest.fixture
def commitment_run(commitment_inputs):
    """The commitment model solved through both lanes, engine still open."""
    data = commitment_inputs
    with differential(COMMITMENT_YAML, data) as run:
        yield run


def test_commitment_milp_agrees_and_stays_integral(commitment_inputs):
    """Both lanes agree, the binaries are integral, and the LP file says so."""
    data = commitment_inputs

    with differential(COMMITMENT_YAML, data, lp=True) as run:
        assert float(run.model.solution['u'].sum()) < run.model.solution['u'].size, (
            'u is not all-1 at the optimum, so commitment actually binds'
        )

        u = run.result.to_pandas('u')['value'].to_numpy()
        assert np.allclose(u, np.round(u), atol=1e-6), 'a binary variable takes an integral value'
        assert set(np.round(u)) <= {0.0, 1.0}, 'and that value is 0 or 1'

        assert 'binary' in run.lp.read_text(), 'the LP file carries integrality, not just bounds'


def test_the_highs_solver_takes_integrality_over_the_whole_column_index(commitment_run):
    """Every column's variable type crosses, in label order, in the one load.

    Integrality is applied by column index, so a vector that is short, shifted
    or in another order relabels which variables are integral and the
    objective moves. A binary model is the sharp case.
    """
    tables = commitment_run.engine._model.handoff
    with Highs(tables, None, None) as sink:
        loaded = sink.run(tables)
    assert loaded.status.is_ok
    assert loaded.objective == pytest.approx(commitment_run.oracle, rel=1e-9)

    held = commitment_run.engine._model.variables['u']
    u = held.share(loaded.primal).to_numpy()
    assert set(np.round(u)) <= {0.0, 1.0}, 'integrality landed on the wrong columns'


def test_highs_numbers_a_continuous_column_zero_and_an_integer_one():
    """The two numbers the hand-off's boolean integrality vector *is*.

    ``_built`` widens ``cols.integral`` — one boolean per column — straight to
    the ``int32`` vector HiGHS reads as variable types, which is only the same
    vector while these two enum members keep these two values. A HiGHS release
    that renumbers them would silently declare every continuous column
    something else, so it fails here instead.
    """
    highspy = pytest.importorskip('highspy')
    assert int(highspy.HighsVarType.kContinuous) == 0, 'a continuous column is what a False widens to'
    assert int(highspy.HighsVarType.kInteger) == 1, 'an integer column is what a True widens to'


def test_cols_vtype_is_an_enum_over_every_declared_domain(commitment_run):
    """``cols.vtype`` is an Enum, and its members are ``program.VariableDomain``.

    An Enum rejects a value outside it, so a fourth domain added to the plan
    and not reaching the column fails where the column is built.
    """
    vtype = commitment_run.engine._model.handoff.cols.schema['vtype']
    held = set(commitment_run.engine._model.handoff.cols['vtype'].unique().to_list())

    assert isinstance(vtype, pl.Enum), f'vtype is {vtype}, so it stores a word per column'
    assert set(vtype.categories.to_list()) == set(get_args(program.VariableDomain))
    assert held == {'continuous', 'binary'}, 'this model declares both, so both must survive the stack'
