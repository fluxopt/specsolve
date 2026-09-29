"""The MPS sink writes the model the LP sink writes, in the format's own order.

MPS is column-major and LP is row-major, so the claim under test is "this is
the same model", not "these bytes": every check reaches an optimum through the
written file and compares it against the same model reached another way.
Reproducibility (#109) and chunk-invariance are pinned as for the LP sink.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError
from specsolve.relational.sinks.writers import mps_file
from tests.conftest import (
    DISPATCH_SPEC,
    PORT_REFERENCES,
    expanded,
    port_sources,
    port_spec,
    schema_of,
    solve_written_file,
)
from tests.test_milp import COMMITMENT_YAML
from tests.test_quadratic_objective import SOURCES as QUADRATIC_DATA
from tests.test_quadratic_objective import SPEC as QUADRATIC_OBJECTIVE_SPEC
from tests.test_sos import DATA as SOS_DATA
from tests.test_sos import best
from tests.test_sos import spec as sos_spec

if TYPE_CHECKING:
    from pathlib import Path

#: The quadratic-objective fixture with its objective moved into a row — the
#: other position degree 2 reaches, and the one whose rows would arrive empty.
QUADRATIC_ROW_SPEC = {
    **QUADRATIC_OBJECTIVE_SPEC,
    'constraints': {
        **QUADRATIC_OBJECTIVE_SPEC['constraints'],
        'coupled': {'dims': ['g'], 'expression': 'p * p <= 9'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p, over=g)'},
}

DISPATCH_DATA = {
    'generator': pl.DataFrame({'generator': ['wind', 'gas']}),
    'p_max': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [40.0, 200.0]}),
    'cost': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [1.0, 50.0]}),
    'snapshot': pl.DataFrame({'snapshot': [0, 1, 2, 3]}),
    'load': pl.DataFrame({'snapshot': [0, 1, 2, 3], 'value': [80.0, 60.0, 100.0, 45.0]}),
}

#: The MILP fixture as a loaded model — a ``str`` reaching the API is a path.
COMMITMENT = schema_of(COMMITMENT_YAML)

COMMITMENT_DATA = {
    'p_max': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [40.0, 200.0]}),
    'cost': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [1.0, 50.0]}),
    'fix_cost': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [10.0, 300.0]}),
    'load': pl.DataFrame({'snapshot': [0, 1, 2, 3], 'value': [80.0, 60.0, 100.0, 45.0]}),
    'generator': pl.DataFrame({'generator': ['wind', 'gas']}),
    'snapshot': pl.DataFrame({'snapshot': [0, 1, 2, 3]}),
}

#: A free column beside bounded ones, so the bounds section is written through
#: all four of its spellings in one model; ``spill``, which the objective never
#: names; and ``idle``, which nothing names at all — the column MPS could drop
#: without the file looking wrong.
FREE_SPEC: dict[str, Any] = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'load': {'dims': ['t']}},
    'variables': {
        'p': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 100}},
        'slack': {'dims': ['t']},
        'spill': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 5}},
        'idle': {'dims': ['t'], 'bounds': {'lower': 1, 'upper': 7}},
    },
    'constraints': {'meet': {'dims': ['t'], 'expression': 'p + slack + spill == load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p, over=t) + sum(slack * 2, over=t)'},
}

FREE_DATA = {'t': [0, 1], 'load': pl.DataFrame({'t': [0, 1], 'value': [30.0, 70.0]})}


CASES = [
    pytest.param(DISPATCH_SPEC, DISPATCH_DATA, id='lp'),
    pytest.param(COMMITMENT, COMMITMENT_DATA, id='milp'),
    pytest.param(FREE_SPEC, FREE_DATA, id='free-and-bounded-columns'),
]


def _written(spec: Any, data: Any, directory: Path) -> Path:
    """*model* built once and written as MPS text."""
    return sps.write(spec, data, directory / 'model.mps')


@pytest.mark.parametrize(('spec', 'data'), CASES)
def test_a_written_model_reaches_the_optimum_the_engine_reaches(spec: Any, data: Any, tmp_path: Path) -> None:
    """The whole claim: MPS in, the same answer out.

    Against the direct sink rather than against the LP file, so a fault shared
    by both writers cannot hide — the LP file is checked here too, but as a
    third opinion.
    """
    with sps.build(spec, data) as model:
        direct = model.solve().objective
        model.write(tmp_path / 'model.mps')
        model.write(tmp_path / 'model.lp')

    assert solve_written_file(tmp_path / 'model.mps') == pytest.approx(direct), (
        'the MPS file describes a different model from the one the engine solved'
    )
    assert solve_written_file(tmp_path / 'model.mps') == pytest.approx(solve_written_file(tmp_path / 'model.lp'))


@pytest.mark.parametrize('name', sorted(PORT_REFERENCES), ids=str)
def test_every_referenced_model_reaches_its_optimum_through_the_file(name: str, tmp_path: Path) -> None:
    """The corpus, against somebody else's published number rather than a sink.

    The three fixtures above are chosen for the sections they exercise; this
    covers every construct the ports use, at their own sizes.

    Every formulation is written out first, so a port's set reaches the file
    as the binaries HiGHS reads; the set as a section is checked above.
    """
    path = tmp_path / f'{name}.mps'
    sps.write(expanded(port_spec(name)), port_sources(name), path)
    assert solve_written_file(path) == pytest.approx(PORT_REFERENCES[name]['objective'], rel=1e-6)


def test_a_maximised_model_says_so(tmp_path: Path) -> None:
    """``OBJSENSE`` — MPS minimises unless told otherwise, and LP carries the word."""
    text = _written(sos_spec(1) | {'sos': {}}, SOS_DATA, tmp_path).read_text()
    assert text.startswith('NAME\nOBJSENSE\n    MAX\n'), 'a maximised model must declare its sense before ROWS'


@pytest.mark.parametrize('sos_type', [1, 2])
def test_a_set_survives_the_file(sos_type: int, tmp_path: Path) -> None:
    """SOS through MPS, against the enumerated optimum rather than a sink.

    HiGHS has no SOS concept and refuses the section, so the reader here is
    Gurobi — the same split ``test_sos`` already makes for the LP file.
    """
    pytest.importorskip('gurobipy', reason='no reader here takes an MPS SOS section without it')
    import gurobipy as gp

    path = _written(sos_spec(sos_type), SOS_DATA, tmp_path)
    model = gp.read(str(path))
    model.setParam('OutputFlag', 0)
    model.optimize()
    assert model.ObjVal == pytest.approx(best(sos_type)), 'the written sets do not restrict what they should'


def test_an_objective_constant_is_written_negated(tmp_path: Path) -> None:
    """MPS carries the constant as the objective row's right-hand side, sign flipped."""
    spec = DISPATCH_SPEC | {'objective': DISPATCH_SPEC['objective'] | {'expression': 'sum(p * cost) + 12.5'}}
    with sps.build(spec, DISPATCH_DATA) as model:
        direct = model.solve().objective
        model.write(tmp_path / 'model.mps')

    text = (tmp_path / 'model.mps').read_text()
    assert '    rhs obj -12.5\n' in text, 'the constant is the objective row RHS, negated'
    assert solve_written_file(tmp_path / 'model.mps') == pytest.approx(direct)


def test_only_the_integer_columns_are_wrapped_in_markers(tmp_path: Path) -> None:
    """The markers are positional, so what they enclose is the assertion.

    Checked against the LP file's own ``binary`` section rather than against
    the built tables: the two writers name their columns the same way, so
    either one disagreeing with the other is the failure this is looking for.
    """
    with sps.build(COMMITMENT, COMMITMENT_DATA) as model:
        model.write(tmp_path / 'model.mps')
        model.write(tmp_path / 'model.lp')

    section = (tmp_path / 'model.mps').read_text().split('COLUMNS\n')[1].split('RHS\n')[0]
    wrapped, depth = set(), 0
    for line in section.splitlines():
        if 'INTORG' in line:
            depth += 1
        elif 'INTEND' in line:
            depth -= 1
        elif depth:
            wrapped.add(line.split()[0])
    assert depth == 0, 'every INTORG needs its INTEND'

    binaries = (tmp_path / 'model.lp').read_text().split('\nbinary\n')[1].split('\n\n')[0]
    assert wrapped == set(binaries.split()), 'the two writers disagree about which columns are integral'


def test_an_unbounded_column_takes_the_format_s_own_spelling(tmp_path: Path) -> None:
    """``MI``/``PL`` rather than a number MPS has no way to write."""
    section = _written(FREE_SPEC, FREE_DATA, tmp_path).read_text().split('BOUNDS\n')[1].split('ENDATA')[0]
    spellings = {line.split()[0] for line in section.strip().splitlines()}
    assert spellings == {'LO', 'UP', 'MI', 'PL'}, 'every column is written with both its bounds, infinite or not'
    assert section.index(' UP ') > section.rindex(' MI '), (
        'every lower bound precedes every upper, so no reader has to guess one from a negative UP'
    )


def test_a_column_in_no_row_and_no_objective_term_still_reaches_the_file(tmp_path: Path) -> None:
    """MPS defines a column by *naming* one, and such a column has nothing to be
    named in.

    LP declares every column in its bounds section whatever it appears in, so
    the same model loses two columns in one format and not the other — which a
    reader notices only as a different answer.
    """
    text = _written(FREE_SPEC, FREE_DATA, tmp_path).read_text()
    named = {line.split()[0] for line in text.split('COLUMNS\n')[1].split('RHS\n')[0].splitlines()}
    bounded = {line.split()[2] for line in text.split('BOUNDS\n')[1].split('ENDATA')[0].strip().splitlines()}
    assert named == bounded, 'a column the file bounds but never names is a column the reader does not have'


def test_one_model_writes_the_same_bytes_every_time(tmp_path: Path) -> None:
    """#109, for the writer whose section order comes out of a sort."""
    digests = []
    for attempt in range(3):
        path = tmp_path / f'model{attempt}.mps'
        sps.write(COMMITMENT, COMMITMENT_DATA, path)
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    assert len(set(digests)) == 1, 'three writes of one model produced different bytes'


@pytest.mark.parametrize('budget', [1, 3, 10, 2_000_000], ids=lambda n: f'budget-{n}')
def test_chunking_the_columns_section_leaves_the_bytes_alone(
    budget: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chunk width is a memory knob and nothing else."""
    reference = tmp_path / 'reference.mps'
    sps.write(COMMITMENT, COMMITMENT_DATA, reference)

    monkeypatch.setattr(mps_file, 'EMIT_BUDGET', budget)
    chunked = tmp_path / 'chunked.mps'
    sps.write(COMMITMENT, COMMITMENT_DATA, chunked)

    assert chunked.read_bytes() == reference.read_bytes(), f'a budget of {budget} moved the bytes'


def test_a_format_nothing_writes_names_both_of_the_ones_that_do(tmp_path: Path) -> None:
    """The registry's error is where a caller learns MPS shipped."""
    with pytest.raises(ValueError, match=r'\.lp, \.mps'):
        sps.write(DISPATCH_SPEC, DISPATCH_DATA, tmp_path / 'model.nl')


#: The two constructs this writer has no section for, each in the position it
#: is declared over. ``.lp`` writes both, which is what makes the refusal the
#: *format's* rather than a ban on degree 2.
QUADRATIC = {
    'a-quadratic-objective': QUADRATIC_OBJECTIVE_SPEC,
    'a-quadratic-constraint': QUADRATIC_ROW_SPEC,
}


@pytest.mark.parametrize('spec', list(QUADRATIC.values()), ids=list(QUADRATIC))
def test_a_construct_this_format_cannot_spell_is_refused_rather_than_written(spec: Any, tmp_path: Path) -> None:
    """MPS spells a quadratic term in an extension section this writer does not
    write, so writing the model without it would hand back a file that parses,
    solves, and is a different model.

    Without the refusal the quadratic rows arrive empty, and Gurobi reads the
    file back at 30.0 against the 9.0 the model itself reaches.
    """
    with pytest.raises(SpecsolveError, match=r"the '\.mps' sink cannot take a quadratic"):
        sps.write(spec, QUADRATIC_DATA, tmp_path / 'model.mps')

    sps.write(spec, QUADRATIC_DATA, tmp_path / 'model.lp')
    assert '[' in (tmp_path / 'model.lp').read_text(), 'the same model is a section the LP writer does emit'
