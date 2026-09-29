"""What a ``where:`` may say about data beyond one comparison per parameter.

A comparison of two expressions reads arithmetic, a count reduces a dimension
away, and a shift reads a predicate at the neighbouring coordinate. Each is
asserted through the differential harness. The masks sit on *variables*, and an
admitted member is worth one in the objective, so the optimum counts exactly
the coordinates the mask keeps.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from tests.differential import RTOL, differential

#: ``a`` is cheap and runs at three consecutive steps, ``b`` is dear and runs
#: at two that are not consecutive, ``c`` is cheap and runs at one. Chosen so
#: the three masks keep three different sets.
UNITS = ['a', 'b', 'c']
STEPS = [0, 1, 2, 3]

P_MIN = {'a': 1.0, 'b': 4.0, 'c': 2.0}
P_MAX = {'a': 10.0, 'b': 6.0, 'c': 8.0}
LIVE = {
    ('a', 0): True, ('a', 1): True, ('a', 2): True, ('a', 3): False,
    ('b', 0): True, ('b', 1): False, ('b', 2): True, ('b', 3): False,
    ('c', 0): True, ('c', 1): False, ('c', 2): False, ('c', 3): False,
}  # fmt: skip

BASE: dict[str, Any] = {
    'dimensions': {'unit': {'dtype': 'str'}, 'step': {'dtype': 'int'}},
    'parameters': {
        'p_min': {'dims': ['unit']},
        'p_max': {'dims': ['unit']},
        'live': {'dims': ['unit', 'step'], 'dtype': 'bool'},
    },
}


def spec(dims: list[str], where: str) -> dict[str, Any]:
    """One variable bounded in ``[0, 1]`` under *where*, maximised."""
    summed = 'sum(pick, over=unit)' if dims == ['unit'] else 'sum(sum(pick, over=unit), over=step)'
    return BASE | {
        'variables': {'pick': {'dims': dims, 'where': where, 'bounds': {'lower': 0, 'upper': 1}}},
        'objective': {'sense': 'maximize', 'expression': summed},
    }


def sources(p_min: dict[str, float] | None = None, live: dict[tuple[str, int], bool] | None = None) -> dict[str, Any]:
    """The fixture, either table replaceable so a missing row can be what a case is about."""
    lower, runs = p_min if p_min is not None else P_MIN, live if live is not None else LIVE
    return {
        'unit': UNITS,
        'step': STEPS,
        'p_min': pl.DataFrame({'unit': list(lower), 'value': list(lower.values())}),
        'p_max': pl.DataFrame({'unit': UNITS, 'value': [P_MAX[u] for u in UNITS]}),
        'live': pl.DataFrame(
            {'unit': [u for u, _ in runs], 'step': [s for _, s in runs], 'value': list(runs.values())}
        ),
    }


def both(written: dict[str, Any], data: dict[str, Any]) -> float:
    """The objective both lanes reach, the harness having asserted they agree."""
    with differential(written, data) as run:
        return float(run.result.objective)


# ---------------------------------------------------------------------------
# a comparison of two expressions
# ---------------------------------------------------------------------------


def test_a_where_comparing_arithmetic_admits_the_coordinates_it_holds_at():
    """``p_min <= 0.5 * p_max`` is read per unit, and the arithmetic is the language's own."""
    admitted = both(spec(['unit'], 'p_min <= 0.5 * p_max'), sources())

    assert admitted == pytest.approx(2.0, rel=RTOL), "'a' and 'c' clear half their cap; 'b' does not"


def test_a_side_of_numbers_alone_is_one_of_the_two_sides():
    """``0.5 <= p_max`` compares expressions too, and the literal side carries no coordinate."""
    admitted = both(spec(['unit'], '4.0 <= p_max'), sources())

    assert admitted == pytest.approx(3.0, rel=RTOL), 'every cap clears four'


def test_a_side_with_no_value_makes_the_comparison_false():
    """A missing row is absence, and a comparison over absence admits nothing —
    the zero a coefficient reads it as would admit ``0 <= 5`` instead."""
    thin = {unit: value for unit, value in P_MIN.items() if unit != 'c'}

    admitted = both(spec(['unit'], 'p_min <= 0.5 * p_max'), sources(thin))

    assert admitted == pytest.approx(1.0, rel=RTOL), "'c' has no p_min row, so nothing is compared there"


def test_a_null_spreads_through_the_arithmetic_of_a_side():
    """`p_min + p_max` has no value wherever either operand has none.

    The absence rules: through arithmetic a null spreads, and only out of a
    summing operator does it not. A row's *constant side* reads a parameter
    with no row as a zero instead, so the two readings part here — and taken
    as a zero this admits every unit rather than the two with both rows.
    """
    thin = {unit: value for unit, value in P_MIN.items() if unit != 'c'}

    admitted = both(spec(['unit'], 'p_min + p_max >= 1'), sources(thin))

    assert admitted == pytest.approx(2.0, rel=RTOL), "'c' has no p_min, so the sum has no value there"


# ---------------------------------------------------------------------------
# a count along a dimension
# ---------------------------------------------------------------------------


def test_a_where_counting_a_predicate_reduces_the_dimension_it_counts_along():
    """``count(live, over=step)`` is one number per unit, so the mask is over ``unit`` alone."""
    admitted = both(spec(['unit'], 'count(live, over=step) >= 2'), sources())

    assert admitted == pytest.approx(2.0, rel=RTOL), "'a' runs three steps and 'b' two; 'c' runs one"


def test_a_count_of_a_predicate_nothing_admits_is_zero_rather_than_absent():
    """A unit the table has no row for is counted, not dropped.

    Counted as absent it would be missing from the answer, and the mask would
    admit nobody rather than the one unit that runs nowhere.
    """
    grounded = {coordinate: runs for coordinate, runs in LIVE.items() if coordinate[0] != 'c'}

    admitted = both(spec(['unit'], 'count(live, over=step) == 0'), sources(live=grounded))

    assert admitted == pytest.approx(1.0, rel=RTOL), "'c' has no row in the table at all, and no step it runs at"


# ---------------------------------------------------------------------------
# a predicate read at the neighbouring coordinate
# ---------------------------------------------------------------------------


def test_a_where_shifting_a_predicate_reads_it_one_coordinate_back():
    """``shift(live, along=step, offset=1)`` is ``live`` at the step before.

    ``a`` runs 0, 1, 2, so steps 1 and 2 have a live predecessor; ``b`` runs 0
    and 2 with nothing between them, and ``c`` runs only 0.
    """
    admitted = both(spec(['unit', 'step'], 'live AND shift(live, along=step, offset=1)'), sources())

    assert admitted == pytest.approx(2.0, rel=RTOL), "only 'a' has a step whose predecessor also runs"


def test_the_end_a_shift_vacates_is_false_rather_than_wrapped():
    """Step 0 has no predecessor, so nothing is carried round from the last step."""
    admitted = both(spec(['unit', 'step'], 'shift(live, along=step, offset=-1)'), sources())

    assert admitted == pytest.approx(3.0, rel=RTOL), "the successors that run are a's 0 and 1, and b's 1"
