"""Shared fixtures and schema helpers for specsolve tests.

Everything here is linopy-free and pandas-free at import, so it loads on a bare
install. A module that needs the oracle imports ``tests.oracle`` or
``tests.differential``, whose ``importorskip`` skips it. A fixture that hands
out pandas objects imports pandas in its own body.
"""

from __future__ import annotations

import contextlib
import difflib
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest
import yaml as pyyaml

from specsolve.relational.sinks import SOLVERS
from specsolve.sources import attachable
from tests.fixtures import (  # noqa: F401
    DISPATCH_SPEC,
    expanded,
    override,
    raw_of,
    schema_of,
)
from tools import constructs

if TYPE_CHECKING:
    from collections.abc import Sequence

EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'

#: The referenced models, with their data and the optimum each should reach.
PORTS_DIR = EXAMPLES_DIR / 'ports'
PORT_REFERENCES: dict[str, dict[str, Any]] = constructs.REFERENCES


def relation(over: str, into: str, labels: Sequence[Any], values: Sequence[Any]) -> pl.DataFrame:
    """A relation's map from one value per label, `None` where the label maps nowhere."""
    rows = [(a, b) for a, b in zip(labels, values, strict=True) if b is not None]
    return pl.DataFrame({over: [a for a, _ in rows], into: [b for _, b in rows]})


def port_sources(name: str) -> dict[str, Any]:
    """One JSON per port, filtered to what its model declares.

    The file keeps the upstream dump whole as provenance, and attaching refuses a
    name the model does not declare.
    """
    data = json.loads((PORTS_DIR / 'data' / f'{name}.json').read_text())
    tables = {k: pl.DataFrame(v) if isinstance(v, dict) else v for k, v in data.items()}
    spec = PORTS_DIR / f'{name}.yaml'
    program = expanded(spec if spec.exists() else EXAMPLES_DIR / f'{name}.yaml', 'piecewise').program
    return {k: v for k, v in tables.items() if k in attachable(program)}


def port_spec(name: str) -> Path:
    """The file behind a referenced model's name, in ``examples/ports/`` or ``examples/``."""
    spec = PORTS_DIR / f'{name}.yaml'
    return spec if spec.exists() else EXAMPLES_DIR / f'{name}.yaml'


@pytest.fixture(params=sorted(PORT_REFERENCES), ids=str)
def port(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Each referenced model in turn: its name, its file, and what it should reach."""
    return {'name': request.param, 'spec': port_spec(request.param)} | PORT_REFERENCES[request.param]


#: Every model in the repo, ports included, from the list the gallery and docs build from.
SPEC_PATHS = [p for _, p in constructs.models()]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        '--update-golden',
        action='store_true',
        default=False,
        help='rewrite committed golden output (examples/*.out) from this run instead of asserting on it',
    )
    parser.addoption(
        '--sweep-depth',
        type=int,
        default=2,
        help='how deep tests/test_expression_sweep.py enumerates; 3 is minutes and has its own CI job',
    )
    parser.addoption(
        '--sweep-shard',
        default='0/1',
        help='run the i-th of n equal strides of the sweep, as `i/n`; the CI job takes one per matrix leg',
    )


# ---------------------------------------------------------------------------
# building schemas to test against
# ---------------------------------------------------------------------------


def solve_written_file(path: Path | str) -> float:
    """Objective HiGHS reaches reading a written model back from disk.

    The third opinion in a differential: without it a sink that writes a wrong
    file is invisible. The format is the path's.
    """
    import highspy

    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.readModel(str(path))
    h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    return h.getInfo().objective_function_value


def by_coord(result: Any, name: str, *dims: str) -> dict[Any, float]:
    """A variable's primal as ``{coordinate: value}`` — tuple keys past one dim.

    One ``primal`` call: two separate calls are not promised to line up.
    """
    frame = result.primal(name)
    columns = [frame[dim] for dim in dims]
    keys = columns[0] if len(dims) == 1 else zip(*columns, strict=True)
    return dict(zip(keys, frame['value'], strict=True))


# ---------------------------------------------------------------------------
# examples as evidence
# ---------------------------------------------------------------------------


def run_example(path: Path, name: str) -> str:
    """Import a script as module ``name`` and capture what its ``main()`` prints, unstyled."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            module.main()
    finally:
        del sys.modules[name]
    return buffer.getvalue()


def assert_golden(output: str, golden: Path, pytestconfig: pytest.Config, drifted: str) -> None:
    """``output`` matches the committed golden, or ``--update-golden`` rewrites it.

    Args:
        output: What this run printed.
        golden: The committed file to compare against or rewrite.
        pytestconfig: The fixture, for the ``--update-golden`` flag.
        drifted: What a mismatch means for this example — opens the failure
            message, above the diff, and should say how to regenerate.
    """
    if pytestconfig.getoption('--update-golden'):
        golden.write_text(output)
        pytest.skip(f'rewrote {golden.name} from this run')
    expected = golden.read_text()
    if output != expected:
        diff = '\n'.join(difflib.unified_diff(expected.splitlines(), output.splitlines(), 'committed', 'this run'))
        pytest.fail(f'{drifted}\n{diff}', pytrace=False)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@pytest.fixture
def dispatch_yaml() -> Path:
    return EXAMPLES_DIR / 'dispatch.yaml'


@pytest.fixture
def dispatch_spec_inputs():
    """``DISPATCH_SPEC``'s data as pandas, index included — one mapping, both lanes."""
    import pandas as pd

    return {
        'p_max': pd.Series({'wind': 100.0, 'gas': 200.0}),
        'cost': pd.Series({'wind': 0.0, 'gas': 50.0}),
        'load': pd.Series([80.0] * 4, index=pd.RangeIndex(4, name='snapshot')),
        'snapshot': pd.RangeIndex(4, name='snapshot'),
        'generator': ['wind', 'gas'],
    }


def dispatch_spec_path(directory: Path, **patch: Any) -> Path:
    """``DISPATCH_SPEC``, varied and written to disk — the linopy lane only takes a path."""
    path = directory / 'spec.yaml'
    path.write_text(pyyaml.safe_dump(override(DISPATCH_SPEC, **patch)))
    return path


#: Generators of ``examples/dispatch.yaml``, and the snapshot count. Distinct
#: costs, so the optimal vertex is unique and primals are comparable across
#: lanes.
DISPATCH_GENERATORS = ('wind', 'solar', 'gas')
DISPATCH_P_MAX = (100.0, 60.0, 200.0)
DISPATCH_COST = (1.0, 2.0, 50.0)
DISPATCH_SNAPSHOTS = 48


def _dispatch_load() -> np.ndarray:
    """The load series both shapes below carry — one draw, one seed."""
    rng = np.random.default_rng(3)
    return (rng.uniform(0.2, 0.8, DISPATCH_SNAPSHOTS) * sum(DISPATCH_P_MAX)).round(3)


@pytest.fixture
def dispatch_inputs():
    """Dispatch data as pandas — the shape the linopy oracle takes."""
    import pandas as pd

    return {
        'p_max': pd.Series(dict(zip(DISPATCH_GENERATORS, DISPATCH_P_MAX, strict=True))),
        'cost': pd.Series(dict(zip(DISPATCH_GENERATORS, DISPATCH_COST, strict=True))),
        'load': pd.Series(_dispatch_load(), index=pd.RangeIndex(DISPATCH_SNAPSHOTS, name='snapshot')),
        'snapshot': pd.RangeIndex(DISPATCH_SNAPSHOTS, name='snapshot'),
        'generator': list(DISPATCH_GENERATORS),
    }


@pytest.fixture
def dispatch_frame_inputs():
    """The same data as tidy frames — the shape the engine documents."""
    generators = list(DISPATCH_GENERATORS)
    return {
        'p_max': pl.DataFrame({'generator': generators, 'value': list(DISPATCH_P_MAX)}),
        'cost': pl.DataFrame({'generator': generators, 'value': list(DISPATCH_COST)}),
        'load': pl.DataFrame({'snapshot': list(range(DISPATCH_SNAPSHOTS)), 'value': _dispatch_load()}),
        'snapshot': pl.DataFrame({'snapshot': range(DISPATCH_SNAPSHOTS)}),
        'generator': pl.DataFrame({'generator': generators}),
    }


@pytest.fixture
def commitment_inputs():
    """Data for the unit-commitment MILP in ``tests.test_milp``."""
    import pandas as pd

    rng = np.random.default_rng(5)
    n_s = 24
    p_max = pd.Series({'coal': 120.0, 'gas': 80.0, 'peaker': 60.0})
    data = {
        'p_max': p_max,
        'cost': pd.Series({'coal': 10.0, 'gas': 30.0, 'peaker': 90.0}),
        'fix_cost': pd.Series({'coal': 400.0, 'gas': 150.0, 'peaker': 20.0}),
        'load': pd.Series(
            (rng.uniform(0.3, 0.9, n_s) * p_max.sum()).round(1),
            index=pd.RangeIndex(n_s, name='snapshot'),
        ),
    }
    return data | {
        'snapshot': pd.RangeIndex(n_s, name='snapshot'),
        'generator': pd.Index(p_max.index, name='generator'),
    }


# ---------------------------------------------------------------------------
# the law fixture: one model, one masked dimension
# ---------------------------------------------------------------------------

#: The two dimensions every expression in the law and sweep tests is written over.
LAW_DIMS = {'f': {'dtype': 'str'}, 't': {'dtype': 'int'}}


def law_data() -> dict[str, Any]:
    """``gate`` masks ``y`` at ``f=b``; ``w`` is a dense coefficient."""
    import pandas as pd

    return {
        'f': ['a', 'b'],
        't': [0, 1],
        'gate': pd.Series({'a': True}),
        'w': pd.Series({'a': 2.0, 'b': 3.0}),
    }


def law_spec(
    expression: str,
    *,
    dims: list[str],
    objective: str = 'sum(x)',
    also: dict | None = None,
) -> dict:
    """A model whose only variable content is *expression*, in a binding row.

    Args:
        expression: The constraint the model exists to state.
        dims: The dimensions the row is repeated over.
        objective: What is maximised.
        also: A second named constraint, for the cases that need two rules.
    """
    return {
        'dimensions': dict(LAW_DIMS),
        'parameters': {'gate': {'dims': ['f'], 'dtype': 'bool'}, 'w': {'dims': ['f']}},
        'variables': {
            'x': {'dims': ['f', 't'], 'bounds': {'lower': 0, 'upper': 100}},
            'y': {'dims': ['f', 't'], 'where': 'gate', 'bounds': {'lower': 0, 'upper': 50}},
        },
        'constraints': {'c': {'dims': dims, 'expression': expression}, **(also or {})},
        'objective': {'sense': 'maximize', 'expression': objective},
    }


def masked_operand_spec(constraint: str, expression: str, *, grouped: bool = False, masked: bool = True) -> dict:
    """The probe behind the shift and window edge cases, over one masked operand.

    ``level`` is masked where ``usable`` says so (unmasked under
    ``masked=False``), and ``take`` is capped only by *expression*'s row. The
    1000x penalty on ``level`` makes "row dropped" and "row built and binding"
    separable from the objective alone. ``grouped`` adds ``season_of``.
    """
    spec: dict[str, Any] = {
        'dimensions': {'t': {'dtype': 'int'}},
        'parameters': {'usable': {'dims': ['t']}},
        'variables': {
            'level': {'dims': ['t'], 'where': 'usable > 0', 'bounds': {'lower': 0, 'upper': 10}},
            'take': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 10}},
        },
        'constraints': {constraint: {'dims': ['t'], 'expression': expression}},
        'objective': {'sense': 'maximize', 'expression': 'sum(take, over=t) - 1000 * sum(level, over=t)'},
    }
    if grouped:
        spec['dimensions']['season'] = {'dtype': 'str'}
        spec['relations'] = {'season_of': {'key': 't', 'values': 'season'}}
    if not masked:
        del spec['parameters']
        del spec['variables']['level']['where']
    return spec


@pytest.fixture
def transport_data():
    """A four-bus ring plus one chord, feasible with zero flow and cheaper with some."""
    import pandas as pd

    rng = np.random.default_rng(11)
    n_s, n_b, n_g, n_l = 24, 4, 9, 5
    buses = [f'b{i}' for i in range(n_b)]
    gens = pd.DataFrame(
        {
            'generator': [f'g{i}' for i in range(n_g)],
            'bus': [buses[i % n_b] for i in range(n_g)],
            'p_max': rng.uniform(80, 150, n_g).round(3),
            'cost': rng.uniform(5, 100, n_g).round(3),
        }
    )
    pairs = [(buses[i], buses[(i + 1) % n_b]) for i in range(n_b)] + [(buses[0], buses[2])]
    lines = pd.DataFrame(
        {
            'line': [f'l{i}' for i in range(n_l)],
            'from_bus': [a for a, _ in pairs],
            'to_bus': [b for _, b in pairs],
            'cap': rng.uniform(60, 120, n_l).round(3),
        }
    )
    local_cap = gens.groupby('bus')['p_max'].sum().reindex(buses).to_numpy()
    factors = rng.uniform(0.3, 0.8, (n_s, n_b))
    load = pd.DataFrame(
        {
            'snapshot': np.repeat(np.arange(n_s), n_b),
            'bus': buses * n_s,
            'value': (factors * local_cap).round(3).ravel(),
        }
    )
    return gens, lines, load


def recomputed_row_values(engine, result) -> Any:
    """Every row's left-hand side at the solution, recomputed from the model.

    ``Ax`` for a linear row and ``xᵀQx + Ax`` for a quadratic one, from the
    built frames and the primal alone. Scattered rather than ``reduceat``-ed: a
    purely quadratic row owns no linear entries, and ``reduceat`` repeats the
    previous row on an empty span.
    """
    tables = engine._model.handoff
    x = np.zeros(tables.column_count)
    for name, block in engine._model.variables.items():
        x[block.start : block.start + block.height] = result.primal(name)['value'].to_numpy()

    values = np.zeros(tables.row_count)
    spans = np.diff(tables.row_starts)
    rows = np.repeat(np.arange(tables.row_count), spans)
    np.add.at(values, rows, tables.matrix['coeff'].to_numpy() * x[tables.matrix['col'].to_numpy()])

    quadratic = tables.qmatrix
    if quadratic.height:
        pairs = quadratic['coeff'].to_numpy() * x[quadratic['col_l'].to_numpy()] * x[quadratic['col_r'].to_numpy()]
        np.add.at(values, quadratic['row'].to_numpy(), pairs)
    return values


#: Three columns and three rows, every declared row built.
SOLVER_VECTOR_SPEC = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'load': {'dims': ['t']}},
    'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'meet': {'dims': ['t'], 'expression': 'x >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x, over=t)'},
}


SOLVER_VECTOR_LOAD = {'t': [0, 1, 2], 'load': pl.DataFrame({'t': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})}


# ---------------------------------------------------------------------------
# the sink cases: what a second solver has to earn
# ---------------------------------------------------------------------------

LP = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'load': {'dims': ['t']}, 'price': {'dims': ['t']}},
    'variables': {'p': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {'meet': {'dims': ['t'], 'expression': 'p >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * price, over=t)'},
}

#: A convex quadratic objective.
QP = {
    'dimensions': {'g': {'dtype': 'str'}},
    'parameters': {'need': {'dims': []}, 'toll': {'dims': ['g']}},
    'variables': {
        'p': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 10}},
        'q': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'meet': {'dims': [], 'expression': 'sum(p, over=g) + sum(q, over=g) >= need'}},
    #: A linear term beside the quadratic one: a hand-off that passed only ``Q`` drops it.
    'objective': {'sense': 'minimize', 'expression': 'sum(p * p + p * q + q * q + q * toll, over=g)'},
}

QP_SOURCES = {
    'g': ['a', 'b'],
    'need': pl.DataFrame({'value': [24.0]}),
    'toll': pl.DataFrame({'g': ['a', 'b'], 'value': [1.0, 7.0]}),
}

#: Maximisation and an objective constant, the two things a sink states outside the frames.
MAX = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'cap': {'dims': ['t']}},
    'variables': {'p': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'lim': {'dims': ['t'], 'expression': 'p <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(p, over=t) + 5'},
}

MIP = {
    'dimensions': {'i': {'dtype': 'int'}, 'one': {'dtype': 'int'}},
    'parameters': {'w': {'dims': ['i']}, 'cap': {'dims': ['one']}},
    'variables': {'x': {'dims': ['i'], 'domain': 'binary'}},
    'constraints': {'budget': {'dims': ['one'], 'expression': 'sum(x * w, over=i) <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * w, over=i)'},
}

INFEASIBLE = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'load': {'dims': ['t']}},
    'variables': {'p': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 1}}},
    'constraints': {'meet': {'dims': ['t'], 'expression': 'p == load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}

#: Each case is the ``(model, data)`` pair a call site unpacks:
#: ``sps.solve(*CASES['MIP'])``.
CASES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    'LP': (
        LP,
        {
            't': [0, 1, 2],
            'load': pl.DataFrame({'t': [0, 1, 2], 'value': [1.0, 2.0, 3.0]}),
            'price': pl.DataFrame({'t': [0, 1, 2], 'value': [10.0, 20.0, 30.0]}),
        },
    ),
    'MAX': (MAX, {'t': [0, 1], 'cap': pl.DataFrame({'t': [0, 1], 'value': [3.0, 4.0]})}),
    'MIP': (
        MIP,
        {
            'i': [0, 1, 2],
            'one': [0],
            'w': pl.DataFrame({'i': [0, 1, 2], 'value': [2.0, 3.0, 4.0]}),
            'cap': pl.DataFrame({'one': [0], 'value': [5.0]}),
        },
    ),
    'INFEASIBLE': (INFEASIBLE, {'t': [0], 'load': pl.DataFrame({'t': [0], 'value': [99.0]})}),
    'QP': (QP, QP_SOURCES),
}


def assert_agrees_with_highs(solver_name: str, case: str, variable: str, constraint: str, *, has_duals: bool) -> None:
    """A second solver agrees with HiGHS on objective, primal, activity and duals.

    Coordinates as well as values, since columns loaded in a different order
    still reach the same objective; and duals under ``maximize``, where a sign
    convention could differ.
    """
    import specsolve as sps

    with sps.solve(*CASES[case]) as highs, sps.solve(*CASES[case], solver_name=solver_name) as other:
        assert other.termination_condition == highs.termination_condition
        assert other.objective == pytest.approx(highs.objective)

        expected, got = highs.primal(variable), other.primal(variable)
        assert got.columns == expected.columns
        assert got.drop('value').equals(expected.drop('value'))
        assert got['value'].to_list() == pytest.approx(expected['value'].to_list())

        assert other.activity(constraint)['value'].to_list() == pytest.approx(
            highs.activity(constraint)['value'].to_list()
        )
        if has_duals:
            assert other.dual(constraint)['value'].to_list() == pytest.approx(highs.dual(constraint)['value'].to_list())


def assert_infeasible_reports_both_axes(solver_name: str) -> None:
    """The status pair, and the solver's own word for it where a user reads it."""
    import specsolve as sps
    from specsolve.errors import NoSolutionError

    with sps.solve(*CASES['INFEASIBLE'], solver_name=solver_name) as solution:
        assert solution.status == 'warning'
        assert solution.termination_condition == 'infeasible'
        assert not solution.has_primal
        assert solution.objective != solution.objective, 'nan, not 0.0'
        with pytest.raises(NoSolutionError, match='INFEASIBLE'):
            solution.primal('p')


#: A knapsack: the discrete model the update and warm-start tests share.
ITEMS = [f'item{i}' for i in range(12)]
KNAPSACK = {
    'dimensions': {'item': {'dtype': 'str'}},
    'parameters': {'worth': {'dims': ['item']}, 'weight': {'dims': ['item']}, 'capacity': {'dims': []}},
    'variables': {'take': {'dims': ['item'], 'domain': 'binary'}},
    'constraints': {'fits': {'dims': [], 'expression': 'sum(weight * take, over=item) <= capacity'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(take * worth)'},
}


def knapsack_sources(items: list[str] = ITEMS) -> dict[str, pl.DataFrame]:
    return {
        'item': pl.DataFrame({'item': items}),
        'worth': pl.DataFrame({'item': items, 'value': [float(7 * i % 13 + 1) for i in range(len(items))]}),
        'weight': pl.DataFrame({'item': items, 'value': [float(5 * i % 11 + 1) for i in range(len(items))]}),
        'capacity': pl.DataFrame({'value': [20.0]}),
    }


@pytest.fixture(params=sorted(SOLVERS))
def solver_name(request: pytest.FixtureRequest) -> str:
    """Every sink that can stay loaded, skipping one this build cannot run."""
    if not SOLVERS[request.param].is_available():
        pytest.skip(f'{request.param} is not installed here')
    return str(request.param)
