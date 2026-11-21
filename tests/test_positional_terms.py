"""A constraint whose terms are read by position is the model the joins build.

Where a parameter has a row at every coordinate and a variable a column at
most of them, a term is held by position (`_engine/dense.py`) and a constraint
places it on its rows by index. Each case here is built both ways and the
written models compared, and each says which way the constraint was placed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import polars as pl
import pytest

import specsolve as sps
from specsolve._engine import assembly, coverage, dense
from specsolve._engine.attaching import AttachedSources
from specsolve._engine.compiler import Compiler
from specsolve._engine.scope import Scope
from specsolve._engine.terms import Presence, Term
from specsolve.errors import DataError, SpecsolveError

if TYPE_CHECKING:
    from pathlib import Path

SPEC = {
    'dimensions': {'i': {'dtype': 'int', 'ordered': True}, 'j': {}},
    'parameters': {
        'a': {'dims': ['i']},
        'b': {'dims': ['j', 'i']},
        'zero_at_one': {'dims': ['i']},
        'short': {'dims': ['i'], 'missing': 'neutral'},
        'gone': {'dims': ['i'], 'missing': 'absent'},
        'on': {'dims': ['i'], 'dtype': 'bool'},
        'rare': {'dims': ['i', 'j'], 'dtype': 'bool'},
        's': {'dims': []},
    },
    'variables': {
        'x': {'dims': ['i'], 'bounds': {'lower': 0, 'upper': 10}},
        'y': {'dims': ['i', 'j'], 'bounds': {'lower': 0, 'upper': 10}},
        'masked': {'dims': ['i'], 'where': 'on', 'bounds': {'lower': 0, 'upper': 10}},
        'silent': {'dims': ['i'], 'where': 'on', 'missing': 'neutral', 'bounds': {'lower': 0, 'upper': 10}},
        'total': {'dims': [], 'bounds': {'lower': 0, 'upper': 10}},
        'late': {'dims': ['i'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(x) + sum(y) + sum(masked) + sum(silent) + total + sum(late)'},
}

#: `i` has 40 labels so that `rare`, true at one coordinate of 80, leaves a
#: variable masked by it past `dense.CELLS_PER_COLUMN` coordinates per column.
LABELS = list(range(40))
PAIR = ['p', 'q']
SOURCES = {
    'i': LABELS,
    'j': PAIR,
    'a': pl.DataFrame({'i': LABELS, 'value': [float(n + 2) for n in LABELS]}),
    'b': pl.DataFrame({'j': [j for j in PAIR for _ in LABELS], 'i': LABELS * 2, 'value': [0.5 + n for n in range(80)]}),
    'zero_at_one': pl.DataFrame({'i': LABELS, 'value': [0.0 if n == 1 else 3.0 for n in LABELS]}),
    'short': pl.DataFrame({'i': LABELS[::2], 'value': [4.0] * 20}),
    'gone': pl.DataFrame({'i': LABELS[::2], 'value': [4.0] * 20}),
    'on': pl.DataFrame({'i': LABELS, 'value': [n % 3 != 0 for n in LABELS]}),
    'rare': pl.DataFrame(
        {'i': [n for n in LABELS for _ in PAIR], 'j': PAIR * 40, 'value': [n == 7 for n in range(80)]}
    ),
    's': 2.0,
}

BY_POSITION, BY_JOIN = 'position', 'join'

pytestmark = pytest.mark.usefixtures('terms_by_position')


def _built(spec: dict, sources: dict, path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, list[str]]:
    """The model as written, and how each constraint with a linear term placed its terms, in file order."""
    placed: list[str] = []
    by_position, by_join = assembly.Assembly._placed_by_position, assembly.Assembly._placed_by_join

    def spied_position(self, *args, **kwargs):
        placed.append(BY_POSITION)
        return by_position(self, *args, **kwargs)

    def spied_join(self, linear, *args, **kwargs):
        if linear:
            placed.append(BY_JOIN)
        return by_join(self, linear, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(assembly.Assembly, '_placed_by_position', spied_position)
        patched.setattr(assembly.Assembly, '_placed_by_join', spied_join)
        sps.write(spec, sources, path)
    return path.read_text(), placed


def nothing_by_position(patched: pytest.MonkeyPatch) -> None:
    """Every read by position switched off, as ``--terms-by-join`` switches it off for the suite."""
    patched.setattr(dense, 'constant', lambda *args: None)
    patched.setattr(dense, 'variable', lambda *args: None)
    patched.setattr(dense, 'covers', lambda *args: False)


def _joined(spec: dict, sources: dict, path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """The model as the join path writes it, no leaf being held by position."""
    with monkeypatch.context() as patched:
        nothing_by_position(patched)
        text, placed = _built(spec, sources, path, monkeypatch)
    assert BY_POSITION not in placed, 'with no leaf held by position, no constraint is placed by it'
    return text


@pytest.mark.parametrize(
    ('constraint', 'how'),
    [
        pytest.param({'dims': ['i'], 'expression': 'a * x <= 10'}, BY_POSITION, id='a-parameter-times-a-variable'),
        pytest.param({'dims': ['i'], 'expression': 'x <= 10 - 2 * late'}, BY_POSITION, id='a-number-times-a-variable'),
        pytest.param(
            {'dims': ['i', 'j'], 'expression': 'y <= a * x'}, BY_POSITION, id='a-term-over-fewer-dims-than-the-row'
        ),
        pytest.param(
            {'dims': ['i', 'j'], 'expression': 'b * y - s * total <= 10'},
            BY_POSITION,
            id='a-parameter-declared-in-another-order-and-a-scalar',
        ),
        pytest.param(
            {'dims': ['i'], 'expression': 'late - x <= 10'}, BY_POSITION, id='terms-that-arrive-out-of-column-order'
        ),
        pytest.param({'dims': ['i'], 'expression': 'x + a * x <= 10'}, BY_POSITION, id='one-variable-twice-in-a-row'),
        pytest.param({'dims': ['i'], 'expression': 'x - x <= 10'}, BY_POSITION, id='a-row-whose-terms-cancel'),
        pytest.param(
            {'dims': ['i'], 'expression': 'zero_at_one * x <= 10'}, BY_POSITION, id='a-coefficient-that-is-zero'
        ),
        pytest.param(
            {'dims': ['i'], 'expression': 'masked + x <= 10'},
            BY_POSITION,
            id='a-masked-variable-removes-the-rows-it-is-absent-from',
        ),
        pytest.param(
            {'dims': ['i'], 'expression': 'a * masked + x <= 10'},
            BY_POSITION,
            id='a-masked-variable-under-a-product-still-removes-its-rows',
        ),
        pytest.param(
            {'dims': ['i'], 'expression': 'silent + x <= 10'},
            BY_POSITION,
            id='a-masked-variable-that-reads-as-zero-keeps-its-rows',
        ),
        pytest.param(
            {'dims': ['i'], 'expression': 'silent <= 10'},
            BY_POSITION,
            id='a-row-left-with-no-term-is-not-built',
        ),
        pytest.param(
            {'dims': ['i'], 'where': 'a > 5', 'expression': 'masked - a * x <= 10'},
            BY_POSITION,
            id='a-where-on-the-constraint',
        ),
        pytest.param({'dims': [], 'expression': 's * total <= 10'}, BY_POSITION, id='a-constraint-over-no-dims'),
        pytest.param(
            {'dims': ['i'], 'expression': 'x <= gone'}, BY_POSITION, id='a-right-hand-side-absent-at-a-coordinate'
        ),
        pytest.param({'dims': ['i'], 'expression': 'short * x <= 10'}, BY_JOIN, id='a-parameter-short-of-a-row'),
        pytest.param({'dims': ['i'], 'expression': 'gone * x <= 10'}, BY_JOIN, id='a-parameter-absent-at-a-coordinate'),
        pytest.param({'dims': ['i'], 'expression': 'x / a <= 10'}, BY_JOIN, id='a-quotient'),
        pytest.param({'dims': ['i'], 'expression': 'sum(y, over=j) <= 10'}, BY_JOIN, id='a-sum'),
        pytest.param(
            {'dims': ['i'], 'expression': 'x + shift(x, along=i, offset=1) <= 10'},
            BY_JOIN,
            id='one-term-of-two-is-a-frame',
        ),
    ],
)
def test_a_constraint_placed_by_position_is_the_one_the_joins_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, constraint: dict, how: str
) -> None:
    spec = {**SPEC, 'constraints': {'row': constraint}}
    text, placed = _built(spec, SOURCES, tmp_path / 'built.lp', monkeypatch)
    assert placed == [how], f'the constraint was placed by {placed}, and this shape is read by {how}'
    assert text == _joined(spec, SOURCES, tmp_path / 'joined.lp', monkeypatch), (
        'reading the terms by position changed the written model'
    )


def _sided(spec: dict, sources: dict, path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, list[str]]:
    """The model as written, and how each constraint's right-hand side was read, in file order."""
    read: list[str] = []
    by_position, by_join = assembly.Assembly._sides_by_position, coverage.constant_side

    def spied_position(self, *args, **kwargs):
        read.append(BY_POSITION)
        return by_position(self, *args, **kwargs)

    def spied_join(*args, **kwargs):
        read.append(BY_JOIN)
        return by_join(*args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(assembly.Assembly, '_sides_by_position', spied_position)
        patched.setattr(coverage, 'constant_side', spied_join)
        sps.write(spec, sources, path)
    return path.read_text(), read


@pytest.mark.parametrize(
    ('constraint', 'how'),
    [
        pytest.param({'dims': ['i'], 'expression': 'x <= a'}, BY_POSITION, id='a-parameter'),
        pytest.param({'dims': ['i'], 'expression': 'x + a >= -3'}, BY_POSITION, id='a-constant-on-each-side'),
        pytest.param({'dims': ['i'], 'expression': 'x <= a * s + 3'}, BY_POSITION, id='a-product-and-a-number'),
        pytest.param(
            {'dims': ['i', 'j'], 'expression': 'y <= b - a'},
            BY_POSITION,
            id='a-parameter-declared-in-another-order-and-one-over-fewer-dims',
        ),
        pytest.param(
            {'dims': ['i'], 'expression': 'sum(y, over=j) <= a'}, BY_POSITION, id='under-terms-that-are-joined'
        ),
        pytest.param(
            {'dims': ['i'], 'where': 'a > 5', 'expression': 'sum(y, over=j) <= a'}, BY_POSITION, id='under-a-where'
        ),
        pytest.param(
            {'dims': ['i'], 'expression': 'masked + x <= a'}, BY_POSITION, id='at-the-rows-a-masked-variable-keeps'
        ),
        pytest.param({'dims': ['i'], 'expression': 'x - late <= 0 * a'}, BY_POSITION, id='a-side-that-is-zero'),
        pytest.param({'dims': ['i'], 'expression': 'x <= late'}, BY_POSITION, id='no-constant-term-at-all'),
        pytest.param({'dims': [], 'expression': 'total <= s'}, BY_POSITION, id='a-constraint-over-no-dims'),
        pytest.param({'dims': ['i'], 'expression': 'x <= gone'}, BY_JOIN, id='a-parameter-absent-at-a-coordinate'),
        pytest.param({'dims': ['i'], 'expression': 'x <= 10 / a'}, BY_JOIN, id='a-quotient'),
        pytest.param({'dims': ['i'], 'expression': 'x <= sum(b, over=j)'}, BY_JOIN, id='a-sum'),
        pytest.param(
            {'dims': ['i'], 'expression': 'x <= a + sum(b, over=j)'}, BY_JOIN, id='one-constant-of-two-is-a-frame'
        ),
    ],
)
def test_a_right_hand_side_read_by_position_is_the_one_the_joins_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, constraint: dict, how: str
) -> None:
    spec = {**SPEC, 'constraints': {'row': constraint}}
    text, read = _sided(spec, SOURCES, tmp_path / 'built.lp', monkeypatch)
    assert read == [how], f'the right-hand side was read by {read}, and this shape is read by {how}'
    assert text == _joined(spec, SOURCES, tmp_path / 'joined.lp', monkeypatch), (
        'reading the right-hand side by position changed the written model'
    )


def test_a_right_hand_side_short_of_a_row_is_refused_as_before(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`short` has a row at every second label, so it is not held by position and the rows pass still asks for the rest."""
    spec = {**SPEC, 'constraints': {'row': {'dims': ['i'], 'expression': 'x <= short'}}}
    with pytest.raises(DataError, match="constraint 'row'"):
        _sided(spec, SOURCES, tmp_path / 'built.lp', monkeypatch)


def test_a_table_whose_rows_are_out_of_order_is_read_at_the_right_coordinates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A table in the order of its dims is read as it stands; any other order is moved into place first."""
    spec = {**SPEC, 'constraints': {'row': {'dims': ['i', 'j'], 'expression': 'b * y + a * x <= 10'}}}
    reversed_rows = {**SOURCES, 'a': SOURCES['a'].reverse(), 'b': SOURCES['b'].reverse()}
    in_order, placed = _built(spec, SOURCES, tmp_path / 'in_order.lp', monkeypatch)
    out_of_order, _ = _built(spec, reversed_rows, tmp_path / 'reversed.lp', monkeypatch)
    assert placed == [BY_POSITION], f'two terms over whole tables were placed by {placed}'
    assert out_of_order == in_order, "the order of a table's rows changed the model read from it by position"


def test_terms_that_name_a_later_variable_first_are_placed_without_a_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    """`late` is declared after `x`, so in file order its entry would lead each row and the share would be sorted."""
    sorts = []
    monkeypatch.setattr(assembly, '_sorted', lambda stacked, keys: sorts.append(keys) or stacked.sort(*keys))
    spec = {**SPEC, 'constraints': {'row': {'dims': ['i'], 'expression': 'late - a * x <= 10'}}}
    sps.build(spec, SOURCES).close()
    assert not sorts, f'the share of two terms over two variables was sorted by {sorts}'


def test_a_variable_masked_to_few_coordinates_is_not_held_by_position(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Held by position it would cost an integer per coordinate of its dims, 80 here for the one column it has."""
    spec = {
        **SPEC,
        'variables': {**SPEC['variables'], 'few': {'dims': ['i', 'j'], 'where': 'rare', 'bounds': {'lower': 0}}},
        'constraints': {'row': {'dims': ['i', 'j'], 'expression': 'few <= 10'}},
    }
    text, placed = _built(spec, SOURCES, tmp_path / 'built.lp', monkeypatch)
    assert placed == [BY_JOIN], f'a variable with one column over 80 coordinates was placed by {placed}'
    assert text == _joined(spec, SOURCES, tmp_path / 'joined.lp', monkeypatch)


def test_rows_a_masked_variable_removes_are_counted_as_not_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """`on` is false at 14 of the 40 labels, and `masked` is absent there."""
    spec = {**SPEC, 'constraints': {'row': {'dims': ['i'], 'expression': 'masked + x <= 10'}}}
    counts = {}
    for label, by_join in (('by position', False), ('by join', True)):
        with monkeypatch.context() as patched:
            if by_join:
                nothing_by_position(patched)
            with sps.build(spec, SOURCES) as model:
                omissions = model.diagnostics().omissions.to_dicts()
        counts[label] = {r['constraint']: r['rows_not_built'] for r in omissions}
    assert counts == {'by position': {'row': 14}, 'by join': {'row': 14}}, (
        'each way counts the 14 rows where the masked variable is absent'
    )


@pytest.mark.parametrize(
    'constraint',
    [
        pytest.param({'dims': ['i', 'j'], 'expression': 'y <= a * x'}, id='every-coordinate-holds-a-row'),
        pytest.param({'dims': ['i', 'j'], 'expression': 'y + masked <= 10'}, id='a-masked-variable-removes-rows'),
        pytest.param({'dims': ['i'], 'where': 'a > 5', 'expression': 'masked + x <= 10'}, id='under-a-where'),
    ],
)
def test_rows_labelled_from_their_positions_read_back_at_the_coordinates_the_joins_give(
    monkeypatch: pytest.MonkeyPatch, constraint: dict
) -> None:
    """A constraint without a `where` is labelled from the positions its terms keep, not from the product of its dims."""
    spec = {**SPEC, 'constraints': {'row': constraint}}
    read = {}
    for label, by_join in (('by position', False), ('by join', True)):
        with monkeypatch.context() as patched:
            if by_join:
                nothing_by_position(patched)
            result = sps.solve(spec, SOURCES, outputs={'activity'})
        read[label] = (result.dual('row'), result.activity('row'))
    for by_position, by_join in zip(read['by position'], read['by join'], strict=True):
        assert by_position.equals(by_join), 'a row read back by coordinate sits where the joins put it'


def test_a_coefficient_that_is_not_finite_is_refused_by_position_as_by_join(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = {**SOURCES, 'a': pl.DataFrame({'i': LABELS, 'value': [float('inf') if n == 3 else 1.0 for n in LABELS]})}
    spec = {**SPEC, 'constraints': {'row': {'dims': ['i'], 'expression': 'a * x <= 10'}}}
    with pytest.raises(DataError, match="constraint 'row'") as by_position:
        sps.build(spec, sources)
    with monkeypatch.context() as patched:
        nothing_by_position(patched)
        with pytest.raises(DataError, match="constraint 'row'") as by_join:
            sps.build(spec, sources)
    assert str(by_position.value) == str(by_join.value)


def test_a_table_as_tall_as_its_product_that_misses_a_coordinate_is_not_read_as_whole() -> None:
    """The door refuses a repeated row, so only a caller of the engine can hold one; unchecked, the gap would read as whatever the array held."""
    program = sps.check(
        {
            'dimensions': {'i': {'dtype': 'int'}},
            'parameters': {'a': {'dims': ['i']}},
            'variables': {'x': {'dims': ['i']}},
            'objective': {'sense': 'minimize', 'expression': 'sum(a * x)'},
        }
    )
    data = AttachedSources(
        parameters={'a': pl.LazyFrame({'i': [0, 0, 2], 'value': [1.0, 2.0, 3.0]})},
        dimensions={'i': pl.LazyFrame({'val': [0, 1, 2], 'ord': [0, 1, 2]})},
        relations={},
        cardinality={'i': 3},
        parameter_rows={'a': 3},
        consecutive={'i': 0},
        given={},
    )
    values = dense.parameter(Scope(program, data, {}), 'a')
    assert values is not None, 'a table of three rows over three labels is taken for whole'
    with pytest.raises(SpecsolveError, match='does not cover'):
        values(dense.At(('i',), {'i': 3}))


def test_a_term_given_a_presence_its_body_does_not_hold_is_placed_by_join(monkeypatch: pytest.MonkeyPatch) -> None:
    """No operator does this today: each that adds a presence makes the term a frame.

    One that kept the body would leave it saying less than the term, and read
    by position the constraint would keep the rows the presence removes. Here
    `x` is given a presence at the first two labels only.
    """
    variable_term = Compiler._variable_term

    def narrowed(self, name: str) -> Term:
        term = variable_term(self, name)
        if name != 'x':
            return term
        first_two = Presence(term.frame.filter(pl.col('i') < 2).select('i'), ('i',))
        return replace(term, presences=(*term.presences, first_two))

    spec = {**SPEC, 'constraints': {'row': {'dims': ['i'], 'expression': 'a * x <= 10'}}}
    placed: list[str] = []
    by_position = assembly.Assembly._placed_by_position

    def spied(self, *args, **kwargs):
        placed.append(BY_POSITION)
        return by_position(self, *args, **kwargs)

    monkeypatch.setattr(Compiler, '_variable_term', narrowed)
    monkeypatch.setattr(assembly.Assembly, '_placed_by_position', spied)
    with sps.build(spec, SOURCES) as model:
        rows = model.diagnostics().rows
    assert not placed, 'a term whose presence its body does not hold is not read by position'
    assert rows == 2, f'the presence admits two rows, and {rows} were built'
