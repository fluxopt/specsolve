"""A `where` or a bound over a parameter with a row at every coordinate reads it by position.

The condition's own walk is unchanged: only how a parameter's column reaches
the table differs, a read at each row's position where it was a join on every
dimension (`Scope.parameter_join`). Each case is built both ways and the written models
compared, and each says which parameters were read by position.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

import specsolve as sps
from specsolve._engine import dense
from tests.conftest import nothing_by_position

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures('read_by_position', 'read_from_the_first_row')


@pytest.fixture
def read_from_the_first_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tables here hold a few dozen rows, far under the size from which a read by position pays."""
    monkeypatch.setattr(dense, 'READ_BY_POSITION_FROM', 0)


LABELS = list(range(12))
PAIR = ['p', 'q']

SPEC = {
    'dimensions': {'i': {'dtype': 'int', 'ordered': True}, 'j': {}, 'gapped': {'dtype': 'int'}},
    'parameters': {
        'on': {'dims': ['i'], 'dtype': 'bool'},
        'size': {'dims': ['i']},
        'kind': {'dims': ['j'], 'dtype': 'str'},
        'both': {'dims': ['j', 'i'], 'dtype': 'bool'},
        'short': {'dims': ['i'], 'missing': 0},
        'far': {'dims': ['gapped'], 'dtype': 'bool'},
    },
    'variables': {
        'x': {'dims': ['i', 'j'], 'bounds': {'lower': 0, 'upper': 10}},
        'z': {'dims': ['gapped'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(x) + sum(z)'},
}

SOURCES = {
    'i': LABELS,
    'j': PAIR,
    'gapped': [10, 20, 40],
    'on': pl.DataFrame({'i': LABELS, 'value': [n % 3 != 0 for n in LABELS]}),
    'size': pl.DataFrame({'i': LABELS, 'value': [float(n) for n in LABELS]}),
    'kind': pl.DataFrame({'j': PAIR, 'value': ['wind', 'gas']}),
    'both': pl.DataFrame(
        {'j': [j for j in PAIR for _ in LABELS], 'i': LABELS * 2, 'value': [n % 5 != 0 for n in range(24)]}
    ),
    'short': pl.DataFrame({'i': LABELS[::2], 'value': [1.0] * 6}),
    'far': pl.DataFrame({'gapped': [10, 20, 40], 'value': [True, False, True]}),
}


def _written(spec: dict, sources: dict, path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, list[str]]:
    """The model as written, and the parameters read by position on the way, each once, sorted."""
    read: set[str] = set()
    column = dense.column

    def spied(scope, name: str) -> pl.Series:
        read.add(name)
        return column(scope, name)

    with monkeypatch.context() as patched:
        patched.setattr(dense, 'column', spied)
        sps.write(spec, sources, path)
    return path.read_text(), sorted(read)


def _joined(spec: dict, sources: dict, path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    with monkeypatch.context() as patched:
        nothing_by_position(patched.setattr)
        text, read = _written(spec, sources, path, monkeypatch)
    assert not read, 'with every read by position switched off, no parameter is read by it'
    return text


@pytest.mark.parametrize(
    ('entries', 'by_position'),
    [
        pytest.param(
            {'constraints': {'row': {'dims': ['i', 'j'], 'where': 'on', 'expression': 'x <= 5'}}},
            ['on'],
            id='a-flag',
        ),
        pytest.param(
            {'constraints': {'row': {'dims': ['i', 'j'], 'where': 'size > 4', 'expression': 'x <= 5'}}},
            ['size'],
            id='a-comparison-with-a-number',
        ),
        pytest.param(
            {'constraints': {'row': {'dims': ['i', 'j'], 'where': 'kind == "wind"', 'expression': 'x <= 5'}}},
            ['kind'],
            id='a-comparison-with-a-string',
        ),
        pytest.param(
            {'constraints': {'row': {'dims': ['i', 'j'], 'where': 'both and not on', 'expression': 'x <= 5'}}},
            ['both', 'on'],
            id='a-parameter-declared-in-another-order-than-the-row',
        ),
        pytest.param(
            {'constraints': {'row': {'dims': ['i', 'j'], 'where': 'on or short', 'expression': 'x <= 5'}}},
            ['on'],
            id='one-parameter-of-two-is-short-of-a-row',
        ),
        pytest.param(
            {'constraints': {'row': {'dims': ['gapped'], 'where': 'far', 'expression': 'z <= 5'}}},
            [],
            id='labels-that-are-not-their-own-positions-are-joined',
        ),
        pytest.param(
            {
                'variables': {
                    **SPEC['variables'],
                    'y': {'dims': ['i'], 'where': 'on', 'bounds': {'lower': 0, 'upper': 'size'}},
                },
                'objective': {'sense': 'maximize', 'expression': 'sum(x) + sum(z) + sum(y)'},
            },
            ['on', 'size'],
            id='a-bound-on-a-variable-narrowed-by-its-where',
        ),
    ],
)
def test_a_parameter_read_by_position_gives_the_model_the_join_gives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entries: dict, by_position: list[str]
) -> None:
    spec = {**SPEC, **entries}
    text, read = _written(spec, SOURCES, tmp_path / 'built.lp', monkeypatch)
    assert read == by_position, f'{read} were read by position, and this shape reads {by_position} so'
    assert text == _joined(spec, SOURCES, tmp_path / 'joined.lp', monkeypatch), (
        'reading a parameter by position changed the written model'
    )


def test_a_table_out_of_order_is_read_by_position_at_the_right_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = {**SPEC, 'constraints': {'row': {'dims': ['i', 'j'], 'where': 'both and size > 4', 'expression': 'x <= 5'}}}
    reversed_rows = {**SOURCES, 'both': SOURCES['both'].reverse(), 'size': SOURCES['size'].reverse()}
    in_order, read = _written(spec, SOURCES, tmp_path / 'in_order.lp', monkeypatch)
    out_of_order, _ = _written(spec, reversed_rows, tmp_path / 'reversed.lp', monkeypatch)
    assert read == ['both', 'size'], f'{read} were read by position, not both parameters of the where'
    assert out_of_order == in_order, "the order of a table's rows changed the rows its where keeps"


def test_a_table_too_small_to_be_worth_reading_is_joined(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`on` spans 12 coordinates: reading its column costs a query of its own, which the join it saves does not repay."""
    monkeypatch.setattr(dense, 'READ_BY_POSITION_FROM', 13)
    spec = {**SPEC, 'constraints': {'row': {'dims': ['i', 'j'], 'where': 'on and both', 'expression': 'x <= 5'}}}
    _, read = _written(spec, SOURCES, tmp_path / 'built.lp', monkeypatch)
    assert read == ['both'], f'{read} were read by position, and only `both` spans 13 coordinates or more'
