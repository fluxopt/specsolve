"""The LP sink renders doubles exactly, and writes the same bytes twice.

The renderer is ``writers.base``'s, shared with the MPS sink. ``lp_file`` writes
numbers by casting them to string, which is only correct if the cast is
shortest-*round-trip*; a golden file proves the bytes did not move, not that
they are correct. Reproducibility (#109) is pinned here too: a golden file
proves one write, and the failure mode is two writes of one model differing.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import TYPE_CHECKING

import polars as pl
import pytest

import specsolve as sps
from specsolve.relational.sinks.writers import lp_file
from specsolve.relational.sinks.writers.base import number
from specsolve.relational.sinks.writers.lp_file import _signed
from tests.conftest import DISPATCH_SPEC, override

if TYPE_CHECKING:
    from pathlib import Path

#: Doubles that break naive formatters: repeating binary fractions, the
#: extremes of the exponent range, a denormal, and the signed zeros.
AWKWARD = [
    0.1,
    1 / 3,
    2 / 3,
    0.3,
    1e-17,
    1.7976931348623157e308,
    2.2250738585072014e-308,
    5e-324,
    123456789.12345679,
    1e20,
    -0.0,
    0.0,
]


def _rendered(render, values: list[float]) -> list[str]:
    """*render* applied to ``values`` as the sink applies it, in order."""
    frame = pl.DataFrame({'v': values}, schema={'v': pl.Float64})
    return frame.select(render(pl.col('v'))).to_series().to_list()


def _signed_text(value: pl.Expr) -> pl.Expr:
    """The sign and magnitude ``_signed`` hands to ``concat_str``, as one string."""
    return pl.concat_str(*_signed(value))


def _bits(x: float) -> bytes:
    """The bit pattern, so ``-0.0`` and ``0.0`` compare unequal."""
    return struct.pack('<d', x)


@pytest.mark.parametrize('value', AWKWARD, ids=repr)
def test_plain_cast_round_trips(value: float) -> None:
    """``number`` — how bounds and right-hand sides are written."""
    (text,) = _rendered(number, [value])
    assert _bits(float(text)) == _bits(value)


@pytest.mark.parametrize('value', AWKWARD, ids=repr)
def test_signed_coefficient_round_trips(value: float) -> None:
    """``_signed`` — how objective and matrix coefficients are written.

    The sign is normalised away for ``-0.0`` (see :func:`_signed`), so the
    round-trip is checked on magnitude there: ``+0.0`` and ``-0.0`` are the
    same coefficient, and only one of them is expressible after a ``+``.
    """
    (text,) = _rendered(_signed_text, [value])
    assert text[0] in '+-', f'coefficient {text!r} carries no explicit sign'
    assert text[:2] != '+-', f'coefficient {text!r} carries two signs'
    assert float(text) == value, '-0.0 == 0.0, which is the whole point'


def test_negative_zero_coefficient_is_written_once() -> None:
    """The trap the spelled-out zero in :func:`_signed` exists to close.

    ``-0.0`` is reachable — any negative coefficient times a zero parameter —
    and it satisfies ``>= 0``, so a naive sign arm emits ``+`` in front of a
    cast that still reads ``-0.0``.
    """
    (text,) = _rendered(_signed_text, [-0.0])
    assert text == '+0.0'


def test_extremes_do_not_become_infinite() -> None:
    """A formatter that drops exponent digits turns ``1e308`` into ``inf``."""
    for text in _rendered(number, [1.7976931348623157e308, 5e-324]):
        assert math.isfinite(float(text))
        assert float(text) != 0.0


def test_written_bounds_are_bit_exact(tmp_path: Path) -> None:
    """End to end: awkward data in, the same doubles back out of the file."""
    upper = [1 / 3, 1e-17]
    cost = [2 / 3, 1.7976931348623157e308]
    data = {
        'generator': ['wind', 'gas'],
        'p_max': pl.DataFrame({'generator': ['wind', 'gas'], 'value': upper}),
        'cost': pl.DataFrame({'generator': ['wind', 'gas'], 'value': cost}),
        'snapshot': pl.DataFrame({'snapshot': [0]}),
        'load': pl.DataFrame({'snapshot': [0], 'value': [0.0]}),
    }
    lp = tmp_path / 'model.lp'
    with sps.build(DISPATCH_SPEC, data) as model:
        model.write(lp)
    text = lp.read_text()

    section = text.split('bounds\n')[1].split('\nend')[0]
    written = sorted(float(line.rsplit('<=', 1)[1]) for line in section.strip().splitlines())
    assert written == sorted(upper)

    objective = text.split('obj:\n')[1].split('\ns.t.')[0]
    coefficients = sorted(float(line.split(' x')[0]) for line in objective.strip().splitlines())
    assert coefficients == sorted(cost)


def _scaled_dispatch(n_generators: int, n_snapshots: int) -> tuple[dict, dict]:
    """``DISPATCH_SPEC`` widened to the given size, with data to match."""
    generators = [f'g{i}' for i in range(n_generators)]
    data = {
        'generator': generators,
        'p_max': pl.DataFrame({'generator': generators, 'value': [100.0 + i for i in range(n_generators)]}),
        'cost': pl.DataFrame({'generator': generators, 'value': [1.0 + i / 8 for i in range(n_generators)]}),
        'snapshot': pl.DataFrame({'snapshot': list(range(n_snapshots))}),
        'load': pl.DataFrame(
            {'snapshot': list(range(n_snapshots)), 'value': [50.0 + t % 7 for t in range(n_snapshots)]}
        ),
    }
    return DISPATCH_SPEC, data


@pytest.mark.parametrize(
    ('sense', 'keyword'),
    [
        pytest.param('minimize', 'min', id='minimize'),
        pytest.param('maximize', 'max', id='maximize'),
        pytest.param(None, 'min', id='no-objective-at-all'),
    ],
)
def test_the_direction_keyword_is_the_files_own(sense: str | None, keyword: str, tmp_path: Path) -> None:
    """The LP format's word for the direction, including where the file names none.

    A feasibility model reaches the writer with ``objective_sense=None``, and
    the format still opens with a keyword. ``min`` over an empty objective is
    the one every direction agrees on.
    """
    schema, data = _scaled_dispatch(n_generators=2, n_snapshots=3)
    if sense is None:
        schema = {k: v for k, v in schema.items() if k != 'objective'}
    else:
        schema = override(schema, **{'objective.sense': sense})

    lp = tmp_path / 'model.lp'
    with sps.build(schema, data) as model:
        model.write(lp)
    assert lp.read_text().splitlines()[0] == keyword, 'the opening keyword is the direction the file asked for'


def test_one_model_writes_the_same_bytes_every_time(tmp_path: Path) -> None:
    """#109 — reproducible output, which is a property of the whole file.

    Sized well past one morsel: a parallel engine hands back one row's joined
    terms in whatever order it finished them, so the model needs many terms per
    row and enough rows for the engine to split the work.
    """
    schema, data = _scaled_dispatch(n_generators=20, n_snapshots=200)

    written = []
    with sps.build(schema, data) as model:
        for attempt in range(3):
            lp = tmp_path / f'{attempt}.lp'
            model.write(lp)
            written.append(hashlib.sha256(lp.read_bytes()).hexdigest())

    assert len(set(written)) == 1, 'the same model wrote different bytes'


def test_chunking_the_constraint_section_leaves_the_bytes_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seams are invisible — chunking bounds the writer's peak, not its output.

    The suite's models fit inside one `EMIT_BUDGET`, so a budget of a few
    nonzeros forces a seam every handful of rows.
    """
    schema, data = _scaled_dispatch(n_generators=5, n_snapshots=40)

    with sps.build(schema, data) as model:
        model.write(tmp_path / 'one.lp')
        monkeypatch.setattr(lp_file, 'EMIT_BUDGET', 3)
        model.write(tmp_path / 'many.lp')

    assert (tmp_path / 'one.lp').read_bytes() == (tmp_path / 'many.lp').read_bytes()


def test_section_keywords_survive_sections_far_larger_than_a_buffer(tmp_path: Path) -> None:
    """The sink writes the keywords itself and polars writes the sections.

    Two writers on one handle agree only while polars goes through the handle's
    buffer. Each section here is megabytes, so a keyword out of place lands in
    the middle of one.
    """
    n_generators, n_snapshots = 50, 2000
    schema, data = _scaled_dispatch(n_generators, n_snapshots)

    lp = tmp_path / 'model.lp'
    with sps.build(schema, data) as model:
        model.write(lp)
    lines = lp.read_text().splitlines()

    keywords = ['min', 'obj:', 's.t.', 'bounds', 'end']
    at = [i for i, line in enumerate(lines) if line in keywords]
    assert [lines[i] for i in at] == keywords, 'a section keyword is missing, doubled or out of order'

    variables = n_generators * n_snapshots
    objective, bounds = at[1], at[3]
    assert lp.stat().st_size > 4_000_000, 'sections too small for the buffer boundary to be crossed'
    assert sum(1 for line in lines[objective + 1 : at[2]] if line.startswith(('+', '-'))) == variables
    assert sum(1 for line in lines[bounds + 1 : at[4]] if ' <= x' in line) == variables
