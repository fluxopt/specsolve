"""Every referenced model, built on both lanes.

``test_ports.py`` asks whether the relational lane reaches a published
optimum; this module asks whether the linopy lane builds the same model, and
whether the written LP file re-solves to it. It needs the oracle, where
``test_ports.py`` runs on the bare install.
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl
import pytest

from tests.conftest import PORT_REFERENCES, PORTS_DIR, port_sources, port_spec
from tests.differential import differential
from tests.linopy_lane.loader import OracleCannotBuildError

#: The instance files the ports attach, unfiltered.
PORTS_DATA = PORTS_DIR / 'data'

#: What the linopy lane accepts and cannot build, keyed by model; a strict xfail on `OracleCannotBuildError`.
LANE_GAPS: dict[str, str] = {
    'osemosys_utopia': '#894 — linopy has no objective-constant slot',
}


def _case(name: str) -> Any:
    reason = LANE_GAPS.get(name)
    marks = [pytest.mark.xfail(reason=reason, raises=OracleCannotBuildError, strict=True)] if reason else []
    return pytest.param(name, marks=marks, id=name)


@pytest.mark.parametrize('name', [_case(n) for n in sorted(PORT_REFERENCES)])
def test_both_lanes_and_the_lp_file_reach_one_objective(name: str) -> None:
    """The harness is the whole assertion: it builds both lanes and re-solves the LP."""
    with differential(port_spec(name), port_sources(name), lp=True) as run:
        _same_matrix(name, run)
        _linopy_matches_the_recorded_duals(name, run)


def _same_matrix(name: str, run: Any) -> None:
    """The two lanes wrote the same coefficients, not merely the same shape.

    Each constraint becomes the sorted multiset of its rows, each row the
    sorted multiset of its coefficients, since the lanes number rows and
    columns differently. Structure is compared exactly and values
    approximately, since the lanes reach a coefficient by a different order of
    operations.
    """
    tables = run.engine._model.handoff
    for constraint, block in run.engine._model.constraints.items():
        if not block.height:
            continue
        got = _canonical(tables.matrix_block(block.start, block.start + block.height))
        want = _linopy_matrix(run.model, constraint)
        assert [len(r) for r in got] == [len(r) for r in want], (
            f'{name}.{constraint}: the lanes wrote a different number of terms per row'
        )
        flat, expected = [c for r in got for c in r], [c for r in want for c in r]
        assert flat == pytest.approx(expected, rel=1e-9, abs=1e-12), (
            f'{name}.{constraint}: the lanes wrote different coefficients'
        )


def _canonical(matrix: pl.DataFrame) -> list[tuple[float, ...]]:
    """``(row, col, coeff)`` as a sorted multiset of sorted coefficient rows."""
    rows = matrix.group_by('row').agg(pl.col('coeff').sort()).get_column('coeff').to_list()
    return sorted(tuple(r) for r in rows)


def _linopy_matrix(linopy_lane: Any, constraint: str) -> list[tuple[float, ...]]:
    """The same, off linopy's dense arrays — duplicate terms collapsed first, as the relational lane does."""
    import numpy as np

    c = linopy_lane.constraints[constraint]
    labels = np.asarray(c.labels).reshape(-1)
    variables = np.asarray(c.vars).reshape(len(labels), -1)
    coefficients = np.asarray(c.coeffs).reshape(len(labels), -1)

    rows = []
    for i, label in enumerate(labels):
        if label < 0:
            continue
        collapsed: dict[int, float] = {}
        for column, coefficient in zip(variables[i], coefficients[i], strict=True):
            if column >= 0:
                collapsed[int(column)] = collapsed.get(int(column), 0.0) + float(coefficient)
        rows.append(tuple(sorted(v for v in collapsed.values() if round(v, 12) != 0)))
    return sorted(rows)


def _linopy_matches_the_recorded_duals(name: str, run: Any) -> None:
    """The linopy lane against the published price, where there is one.

    Against the recording rather than the other lane: a recorded dual claims
    this instance has a unique one, where two lanes need not agree on a dual.
    """
    _check_recorded_duals(name, PORT_REFERENCES[name], run)


def _check_recorded_duals(name: str, entry: dict[str, Any], run: Any) -> None:
    """*entry*'s recorded duals against the linopy lane, split out so a probe can pass a wrong one."""
    recorded = entry.get('duals')
    if not recorded:
        return
    for constraint, table in recorded.items():
        want = pl.DataFrame(table)
        dims = [c for c in want.columns if c != 'value']
        got = _tidy(run.model.constraints[constraint].dual, dims, want)
        want = want.with_columns(pl.col(d).cast(got.schema[d]) for d in dims).sort(dims)
        assert got[dims].equals(want[dims]), f'{name}.{constraint}: the linopy dual is keyed differently'
        assert got['value'].to_list() == pytest.approx(want['value'].to_list(), rel=entry['rtol'], abs=1e-9), (
            f'{name}.{constraint}: the linopy lane disagrees with {entry["provenance"]}'
        )


def _tidy(dual: Any, dims: list[str], like: pl.DataFrame) -> pl.DataFrame:
    """A linopy dual as ``(dims…, value)``, keyed and sorted like *like*."""
    if not dims:
        return pl.DataFrame({'value': [float(dual.values.reshape(-1)[0])]})
    series = dual.to_series().dropna()
    keys = list(series.index)
    columns = {d: [k[i] if isinstance(k, tuple) else k for k in keys] for i, d in enumerate(dims)}
    frame = pl.DataFrame({**columns, 'value': [float(v) for v in series.to_numpy()]})
    return frame.with_columns(pl.col(d).cast(like.schema[d]) for d in dims).sort(dims)


def test_the_linopy_dual_check_would_notice_a_wrong_price() -> None:
    """A recording one step from the truth is refused.

    ``monthly_budget``'s dual is a short vector with distinct values, so a
    perturbed entry cannot coincide with another.
    """
    name = 'monthly_budget'
    recorded = PORT_REFERENCES[name].get('duals')
    assert recorded, f'{name} is the probe because it records duals — give the probe another model'

    constraint, table = next(iter(recorded.items()))
    wrong = {**table, 'value': [v + 1.0 for v in table['value']]}
    entry = {**PORT_REFERENCES[name], 'duals': {constraint: wrong}}

    with differential(port_spec(name), port_sources(name)) as run, pytest.raises(AssertionError, match=constraint):
        _check_recorded_duals(name, entry, run)


#: PyPSA's secant loss mode on ``pypsa_losses``' own network — its default, where
#: the ported tangent mode is deprecated. From
#: ``examples/ports/references/pypsa/pypsa_losses.py``'s ``secant_objective``,
#: pypsa 1.2.4, under the tolerances that script records.
SECANT_OBJECTIVE = 24432.20488089685

#: The loss each lossy line settles on, ``(snapshot, line)`` in sorted order.
#: Three low-demand snapshots make the flows reach the early segments too.
SECANT_LOSSES = [
    4.319999999999999,
    1.6236570501210394,
    4.319999999999999,
    1.2607240286010466,
    4.319999999999999,
    2.034868195461712,
    0.05571402370778852,
    0.031231008229264477,
    0.3029679080268904,
    0.14940704839558197,
    1.4593292560812914,
    0.753085939715055,
]


def test_the_two_loss_approximations_are_one_model() -> None:
    """PyPSA's other loss mode is the same rows with different numbers in them.

    ``pypsa_losses`` ports the tangent approximation; the secant one emits the
    same half-planes, ``loss ± slope * f >= offset``, with other coefficients.
    The coefficients are dumped from PyPSA
    (``pypsa_losses.py::secant_coefficients``), not re-derived here.
    """
    sources = dict(port_sources('pypsa_losses'))
    sources |= {
        name: pl.DataFrame(table)
        for name, table in json.loads((PORTS_DATA / 'pypsa_losses_secants.json').read_text()).items()
    }

    with differential(port_spec('pypsa_losses'), sources, lp=True) as run:
        assert run.result.objective == pytest.approx(SECANT_OBJECTIVE, rel=1e-9), (
            'the secant instance reaches the number PyPSA reaches under its own default mode'
        )
        loss = run.result.primal('loss').sort(['snapshot', 'line'])['value'].to_list()
        assert loss == pytest.approx(SECANT_LOSSES, rel=1e-9), (
            'and sits where PyPSA sits on the approximated curve, line by line'
        )
