"""Row order carries no meaning: no answer depends on it.

Polars promises no row order from a join, a group-by or a ``unique`` that was
not asked for one, and a source table's rows arrive in whatever order its
writer chose. Two checks hold every referenced model to that:

**The engine's own order.** Every join, group-by and ``unique`` that promises
no order is shuffled, and the model built must digest exactly as a plain
build does: matrix, bounds, costs and right-hand sides, bit for bit. A saved
answer is checked against that digest, so a digest that moves refuses an
archive whose data nothing changed. The same model then solves to the same
objective, and each variable reads back in the same order at the same values.

**The sources' order.** Every parameter and relation table is shuffled, and the
solve must reach the same optimum, with each variable's coordinates in the same
order. Dimension tables keep their order: a dimension's row order is its
coordinate order, which ``shift`` reads.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

import specsolve as sps
from tests.conftest import expanded
from tests.conftest import port_sources as sources

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from polars.dataframe.group_by import GroupBy
    from polars.lazyframe.group_by import LazyGroupBy

SEEDS = [pytest.param(0, id='seed-0'), pytest.param(1, id='seed-1')]

#: Ports whose built model moves with the engine's row order, and why.
MOVES_WITH_ROW_ORDER = {
    'osemosys_utopia': (
        'a cost or a right-hand side summed over rows in no fixed order differs in its last bit, '
        'so the digest moves, and an archive refuses an undeclared read as built from other data'
    ),
}

_SHUFFLE_KEY = '__shuffled__'
_LAZY_JOIN, _EAGER_JOIN = pl.LazyFrame.join, pl.DataFrame.join
_LAZY_UNIQUE, _EAGER_UNIQUE, _SERIES_UNIQUE = pl.LazyFrame.unique, pl.DataFrame.unique, pl.Series.unique
_LAZY_GROUP_BY, _EAGER_GROUP_BY = pl.LazyFrame.group_by, pl.DataFrame.group_by


class _Shuffler:
    """Each call permutes what it is given, by a seed of its own drawn from one sequence, and is counted."""

    def __init__(self, seed: int) -> None:
        self._seeds: Iterator[int] = itertools.count(seed * 1_000_003)
        self.calls = 0

    def _seed(self) -> int:
        self.calls += 1
        return next(self._seeds)

    def lazy(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        order = pl.int_range(pl.len(), dtype=pl.UInt32).shuffle(self._seed())
        return frame.with_columns(order.alias(_SHUFFLE_KEY)).sort(_SHUFFLE_KEY).drop(_SHUFFLE_KEY)

    def eager(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.sample(fraction=1.0, shuffle=True, seed=self._seed())

    def series(self, values: pl.Series) -> pl.Series:
        return values.sample(fraction=1.0, shuffle=True, seed=self._seed())


class _ShuffledGroups:
    """A group-by asked for no order, whose every aggregate comes back shuffled."""

    def __init__(self, groups: LazyGroupBy | GroupBy, shuffle: Callable[[Any], Any]) -> None:
        self._groups = groups
        self._shuffle = shuffle

    def __iter__(self) -> Iterator[Any]:
        return iter(self._groups)

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._groups, name)
        if not callable(attribute):
            return attribute
        return lambda *args, **kwargs: self._shuffle(attribute(*args, **kwargs))


def _shuffle_what_promises_no_order(monkeypatch: pytest.MonkeyPatch, seed: int) -> _Shuffler:
    """Patch polars so that every join, group-by and ``unique`` asked for no order returns its rows shuffled."""
    shuffle = _Shuffler(seed)

    def join(original: Callable[..., Any], shuffled: Callable[[Any], Any]) -> Callable[..., Any]:
        def patched(self: Any, other: Any, *args: Any, **kwargs: Any) -> Any:
            out = original(self, other, *args, **kwargs)
            return shuffled(out) if kwargs.get('maintain_order') in (None, 'none') else out

        return patched

    def unique(original: Callable[..., Any], shuffled: Callable[[Any], Any]) -> Callable[..., Any]:
        def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
            out = original(self, *args, **kwargs)
            return out if kwargs.get('maintain_order', False) else shuffled(out)

        return patched

    def group_by(original: Callable[..., Any], shuffled: Callable[[Any], Any]) -> Callable[..., Any]:
        def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
            groups = original(self, *args, **kwargs)
            return groups if kwargs.get('maintain_order', False) else _ShuffledGroups(groups, shuffled)

        return patched

    def eager_groups(out: Any) -> Any:
        return shuffle.eager(out) if isinstance(out, pl.DataFrame) else out

    monkeypatch.setattr(pl.LazyFrame, 'join', join(_LAZY_JOIN, shuffle.lazy))
    monkeypatch.setattr(pl.DataFrame, 'join', join(_EAGER_JOIN, shuffle.eager))
    monkeypatch.setattr(pl.LazyFrame, 'unique', unique(_LAZY_UNIQUE, shuffle.lazy))
    monkeypatch.setattr(pl.DataFrame, 'unique', unique(_EAGER_UNIQUE, shuffle.eager))
    monkeypatch.setattr(pl.Series, 'unique', unique(_SERIES_UNIQUE, shuffle.series))
    monkeypatch.setattr(pl.LazyFrame, 'group_by', group_by(_LAZY_GROUP_BY, shuffle.lazy))
    monkeypatch.setattr(pl.DataFrame, 'group_by', group_by(_EAGER_GROUP_BY, eager_groups))
    return shuffle


def _solved(port: dict[str, Any]) -> tuple[str, float, dict[str, pl.DataFrame]]:
    """The port built and solved: its model's digest, its objective, and each variable's values as read back."""
    spec = expanded(port['spec'])
    with sps.build(spec, sources(port['name'])) as model, model.solve() as result:
        primals = {name: result.primal(name) for name in sps.check(spec).variables} if result.has_primal else {}
        return model._model_digest(), result.objective, primals


@pytest.mark.parametrize('seed', SEEDS)
def test_a_model_builds_and_answers_the_same_whatever_order_the_engine_meets_its_rows_in(
    port: dict[str, Any], seed: int, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    if reason := MOVES_WITH_ROW_ORDER.get(port['name']):
        request.applymarker(pytest.mark.xfail(reason=reason))
    digest, objective, primals = _solved(port)
    shuffle = _shuffle_what_promises_no_order(monkeypatch, seed)
    shuffled_digest, shuffled_objective, shuffled_primals = _solved(port)
    assert shuffle.calls > 0, 'nothing was shuffled, so this compared a build with itself'
    assert shuffled_digest == digest, (
        'the same sources built a different model once the engine met its rows in another order'
    )
    assert shuffled_objective == objective or math.isnan(objective), 'the same model solved to another objective'
    for name, values in primals.items():
        assert shuffled_primals[name].equals(values), f'{name} was read back in another order, or at other values'


def _shuffled_tables(port: dict[str, Any], seed: int) -> dict[str, Any]:
    """The port's sources, every table but a dimension's with its rows shuffled."""
    dimensions = set(sps.check(expanded(port['spec'])).dimensions)
    return {
        name: source.sample(fraction=1.0, shuffle=True, seed=seed)
        if name not in dimensions and isinstance(source, pl.DataFrame)
        else source
        for name, source in sources(port['name']).items()
    }


@pytest.mark.parametrize('seed', SEEDS)
def test_a_model_answers_the_same_whatever_order_its_tables_arrive_in(port: dict[str, Any], seed: int) -> None:
    variables = sps.check(expanded(port['spec'])).variables
    with (
        sps.solve(expanded(port['spec']), sources(port['name'])) as plain,
        sps.solve(expanded(port['spec']), _shuffled_tables(port, seed)) as shuffled,
    ):
        assert shuffled.status == plain.status
        if not plain.has_primal:
            return
        assert shuffled.objective == pytest.approx(plain.objective, rel=1e-9), (
            'shuffling the rows of a table moved the optimum, to within 1e-9 relative'
        )
        for name in variables:
            assert shuffled.primal(name).drop('value').equals(plain.primal(name).drop('value')), (
                f"{name}'s coordinates came back in another order, though only the tables' rows moved"
            )
