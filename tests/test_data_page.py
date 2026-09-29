"""The data-preparation page shows files that exist and code that runs.

The CSV shown matches the committed file, the committed file matches the JSON
instance, and the preparation code produces tables that solve.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl
import pytest

import specsolve as sps

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / 'docs' / 'howto' / 'data.md'
FOLDER = ROOT / 'examples' / 'ports' / 'data' / 'dispatch'


def _fences(lang: str) -> list[str]:
    return re.findall(rf'^```{lang}\n(.*?)^```', PAGE.read_text(), re.MULTILINE | re.DOTALL)


def _block(marker: str) -> str:
    """The one python fence containing *marker* — each block has its test."""
    (block,) = [b for b in _fences('python') if marker in b]
    return block


@pytest.mark.parametrize('name', ['generators', 'load'])
def test_the_page_shows_the_file_that_is_committed(name: str) -> None:
    text = (FOLDER / f'{name}.csv').read_text()
    assert text in _fences('csv'), f'docs/howto/data.md has drifted from {name}.csv'


def test_the_preparation_code_runs_and_solves(monkeypatch: pytest.MonkeyPatch) -> None:
    """The page's preparation code, executed verbatim, reaches the recorded optimum."""
    monkeypatch.chdir(ROOT)
    scope: dict = {}
    exec(_block('read_csv'), scope)  # the page's code, run as the reader would
    with sps.solve(str(ROOT / 'examples' / 'dispatch.yaml'), scope['sources']) as solution:
        assert solution.objective == pytest.approx(10500.0, rel=1e-9), (
            'the prepared sources do not reach the optimum references.json records for dispatch'
        )


def test_the_linopy_shapes_block_runs_and_solves() -> None:
    """Indexed Series pass as sources unconverted — proven by solving with them."""
    pytest.importorskip('pandas')
    scope: dict = {}
    exec(_block('pd.Series'), scope)
    with sps.solve(str(ROOT / 'examples' / 'dispatch.yaml'), scope['sources']) as solution:
        assert solution.objective == pytest.approx(10500.0, rel=1e-9), (
            'linopy-shaped Series sources do not reach the recorded dispatch optimum'
        )


def test_the_pypsa_shapes_block_produces_tidy_sources() -> None:
    """The PyPSA conversion, run against a stub carrying PyPSA's real shapes.

    PyPSA is not a test dependency; the stub's index and column names
    (``name``, ``snapshot``) match what pypsa 1.2.4 produces.
    """
    from types import SimpleNamespace

    pd = pytest.importorskip('pandas')

    generators = pd.DataFrame(
        {'bus': ['north', 'south'], 'p_nom': [60.0, 150.0], 'marginal_cost': [10.0, 40.0]},
        index=pd.Index(['wind_n', 'gas_s'], name='name'),
    )
    loads = pd.DataFrame({'bus': ['north', 'south']}, index=pd.Index(['ln', 'ls'], name='name'))
    p_set = pd.DataFrame({'ln': [20.0, 30.0], 'ls': [70.0, 80.0]}, index=pd.Index([0, 1], name='snapshot')).rename_axis(
        columns='name'
    )
    n = SimpleNamespace(generators=generators, loads=loads, loads_t=SimpleNamespace(p_set=p_set))

    scope: dict = {'n': n}
    exec(_block('n.generators'), scope)
    instance = json.loads((FOLDER.parent / 'transport.json').read_text())
    sources = scope['sources'] | {
        key: pl.DataFrame(instance[key]) for key in ('line', 'cap', 'neg_cap', 'generator', 'line_from', 'line_to')
    }
    with sps.solve(str(ROOT / 'examples' / 'transport.yaml'), sources) as solution:
        assert solution.objective == pytest.approx(4400.0, rel=1e-9), (
            'pypsa-shaped sources do not reach the recorded transport optimum'
        )


def test_the_folder_and_the_instance_agree() -> None:
    """The CSV folder and dispatch.json carry the same instance."""
    instance = json.loads((FOLDER.parent / 'dispatch.json').read_text())
    generators = pl.read_csv(FOLDER / 'generators.csv')
    for parameter in ('p_max', 'cost'):
        from_csv = generators.select('generator', pl.col(parameter).alias('value'))
        assert from_csv.equals(pl.DataFrame(instance[parameter])), f'{parameter} differs between CSV and JSON'
    assert pl.read_csv(FOLDER / 'load.csv').equals(pl.DataFrame(instance['load'])), 'load differs between CSV and JSON'
