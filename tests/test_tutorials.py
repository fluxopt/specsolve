"""The tutorial and how-to pages whose blocks the site runs keep running, and keep claiming true things.

The five tutorials (``docs/run.md``, ``docs/tables.md``, ``docs/change.md``,
``docs/commitment.md`` and ``docs/sweep.md``) and
``docs/howto/fix-relax-remove.md`` run every executed block, and each test pins
one claim the prose makes. Blocks are exec'd in order in one namespace, as
markdown-exec's ``session=`` runs them on the site, and in the directory a
fence's ``workdir=`` names.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import re
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

import specsolve as sps
from tests.conftest import EXAMPLES_DIR

if TYPE_CHECKING:
    from pathlib import Path

REPO = EXAMPLES_DIR.parent
DOCS_DIR = REPO / 'docs'
FIRST = DOCS_DIR / 'run.md'
DATA = DOCS_DIR / 'tables.md'
ON_OFF = DOCS_DIR / 'commitment.md'
TIME = DOCS_DIR / 'sweep.md'
LOOPS = DOCS_DIR / 'change.md'
LIFECYCLE = DOCS_DIR / 'howto' / 'fix-relax-remove.md'

#: A block markdown-exec runs, and only those: the fence carries `exec="true"`.
EXECUTED = re.compile(r'^```python(?P<info>[^\n]*\bexec="true"[^\n]*)$\n(?P<code>.*?)^```$', re.DOTALL | re.MULTILINE)
WORKDIR = re.compile(r'\bworkdir="(?P<dir>[^"]*)"')


def run(page: Path) -> tuple[dict[str, Any], str]:
    """One top-to-bottom run: the namespace it ends with, and what it printed.

    Runs from the repository root, where zensical runs the build, so
    ``examples/dispatch.yaml`` resolves, or from the ``workdir`` a fence names.
    """
    blocks = list(EXECUTED.finditer(page.read_text()))
    assert blocks, f'{page.name} has no executed block, so this test would assert nothing'
    namespace: dict[str, Any] = {'__name__': '__tutorial__'}
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        for block in blocks:
            workdir = WORKDIR.search(block['info'])
            with contextlib.chdir(REPO / workdir['dir'] if workdir else REPO):
                exec(compile(block['code'], str(page), 'exec'), namespace)
    return namespace, printed.getvalue()


@pytest.fixture(scope='module')
def first() -> tuple[dict[str, Any], str]:
    return run(FIRST)


@pytest.fixture(scope='module')
def session() -> tuple[dict[str, Any], str]:
    return run(LOOPS)


@pytest.fixture(scope='module')
def lifecycle() -> tuple[dict[str, Any], str]:
    return run(LIFECYCLE)


def test_the_first_model_solves_to_the_cost_the_page_quotes(first: tuple[dict[str, Any], str]) -> None:
    namespace, _ = first
    assert namespace['result'].objective == pytest.approx(10500.0), 'the page says the load is met for 10500'


def test_the_first_model_quotes_the_refusal_it_asks_the_reader_to_cause() -> None:
    """The page has the reader drop the sum from ``power_balance``, and quotes what ``check`` says."""
    import yaml

    spec = yaml.safe_load((EXAMPLES_DIR / 'dispatch.yaml').read_text())
    spec['constraints']['power_balance']['expression'] = 'p == load'
    with pytest.raises(sps.errors.SpecsolveError) as refused:
        sps.check(spec)
    quoted = f'{type(refused.value).__name__}: {refused.value}'
    assert f'```text\n{quoted}\n```' in FIRST.read_text(), f'docs/run.md no longer quotes the refusal whole: {quoted}'


@pytest.fixture(scope='module')
def data() -> tuple[dict[str, Any], str]:
    return run(DATA)


def test_the_data_from_files_solves_as_the_first_model_did(data: tuple[dict[str, Any], str]) -> None:
    namespace, _ = data
    assert namespace['result'].objective == pytest.approx(10500.0), 'the page says the files hold the same data'
    assert namespace['spend']['spend'].to_list() == pytest.approx([7500.0, 3000.0]), (
        'gas then wind, and the page says the two make the objective'
    )


def test_an_archive_reads_back_the_answer_it_kept(data: tuple[dict[str, Any], str]) -> None:
    namespace, _ = data
    assert namespace['archived'].result.objective == pytest.approx(namespace['result'].objective), (
        'the page says load_archive gives the answer without solving again'
    )


@pytest.fixture(scope='module')
def on_off() -> tuple[dict[str, Any], str]:
    return run(ON_OFF)


def test_on_off_generators_cost_what_the_page_says(on_off: tuple[dict[str, Any], str]) -> None:
    namespace, _ = on_off
    assert namespace['result'].objective == pytest.approx(14000.0), 'the page says the cost goes up to 14000'


def test_gas_runs_at_its_minimum_and_wind_gives_way(on_off: tuple[dict[str, Any], str]) -> None:
    namespace, _ = on_off
    wide = namespace['result'].primal('p').pivot(on='generator', index='snapshot', values='value').sort('snapshot')
    assert wide['gas'].to_list() == pytest.approx([0.0, 50.0, 100.0, 50.0]), (
        'gas is off in snapshot 0, at its minimum in 1 and 3, and above it in 2'
    )
    assert wide['wind'].to_list() == pytest.approx([60.0, 70.0, 80.0, 40.0]), 'wind gives way in snapshots 1 and 3'


def test_a_mip_has_no_prices(on_off: tuple[dict[str, Any], str]) -> None:
    _, printed = on_off
    assert 'duals are undefined for a mixed-integer model' in printed, 'the page shows the refusal of dual'


@pytest.fixture(scope='module')
def over_time() -> tuple[dict[str, Any], str]:
    return run(TIME)


def test_the_battery_lets_gas_switch_off(over_time: tuple[dict[str, Any], str]) -> None:
    namespace, _ = over_time
    assert namespace['result'].objective == pytest.approx(10850.0), 'the page says the battery brings 14000 to 10850'


def test_a_cold_scenario_keeps_gas_on(over_time: tuple[dict[str, Any], str]) -> None:
    namespace, _ = over_time
    wide = namespace['gas_on'].pivot(on='scenario', index='snapshot', values='value').sort('snapshot')
    assert wide['mild'].to_list() == pytest.approx([0.0, 1.0, 1.0, 0.0]), 'mild: gas is off in snapshot 3'
    assert wide['cold'].to_list() == pytest.approx([0.0, 1.0, 1.0, 1.0]), 'cold: gas stays on in snapshot 3'


def test_a_window_starts_from_what_the_window_before_left(over_time: tuple[dict[str, Any], str]) -> None:
    namespace, _ = over_time
    soc = namespace['rolling'].primal('soc').sort('hour')['value'].to_list()
    gas = namespace['rolling'].primal('on').filter(pl.col('generator') == 'gas').sort('hour')['value'].to_list()
    assert soc[2] == pytest.approx(10.0), 'the page says the first window ends with 10 in the battery'
    assert gas[3] == pytest.approx(0.0), 'and that the next window spends it in hour 3, with gas off'


def test_lookahead_closes_the_gap_to_one_solve(over_time: tuple[dict[str, Any], str]) -> None:
    namespace, _ = over_time
    costs = [row['cost'] for row in namespace['rows']]
    whole = namespace['whole'].objective
    assert costs[0] / whole - 1 == pytest.approx(0.12, abs=0.005), 'the page says no lookahead costs 12% more'
    assert costs[0] > costs[1] > costs[2], f'each hour of lookahead buys something: {costs}'
    assert costs[2] == pytest.approx(whole, abs=0.1), 'the page says three hours close all of the gap'


def test_the_update_loop_stays_on_the_fast_path(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    numbers = namespace['after_numbers']
    assert (numbers.loads, numbers.solves) == (1, 5), 'the page says five answers came off one loaded model'


def test_more_wind_saves_less_each_time(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    objectives = [row['objective'] for row in namespace['rows']]
    savings = [a - b for a, b in itertools.pairwise(objectives)]
    assert savings == sorted(savings, reverse=True) and savings[-1] > 0, (
        f'the page says each step of wind saves less than the one before: {savings}'
    )


def test_a_update_answers_what_a_fresh_build_answers(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    assert namespace['fresh'].objective == pytest.approx(namespace['rows'][-1]['objective']), (
        'the equality the page offers as the oracle for a loop that looks wrong'
    )


def test_growing_a_coordinate_set_loads_again(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    assert namespace['after_rows'].loads == 2, 'the page says new snapshots cost a reload'
    assert namespace['longer'].objective == pytest.approx(2 * 10500.0), 'the page says two days cost twice one day'


def test_pinning_a_variable_stays_on_the_fast_path(lifecycle: tuple[dict[str, Any], str]) -> None:
    """A pin writes bounds, not a row, so the loaded model is kept."""
    namespace, _ = lifecycle
    assert namespace['pinning'].loads == 1, 'a pin writes bounds, so the solver keeps the model it has loaded'
    assert namespace['held'] > namespace['unpinned'], 'holding gas at 60 has to cost something, or it pins nothing'


def test_integrality_is_a_declaration_and_costs_the_duals(lifecycle: tuple[dict[str, Any], str]) -> None:
    """The relax page's claim: a domain edit is a rebuild, and a MILP has no prices."""
    namespace, printed = lifecycle
    assert namespace['milp'].has_primal, 'the integer solve still answers'
    assert 'duals are undefined for a mixed-integer model' in printed, 'and says why it cannot be priced'
    assert namespace['relaxed'].dual('power_balance').height == 6, 'the continuous declaration prices every row'


def test_removing_a_constraint_moves_the_answer(lifecycle: tuple[dict[str, Any], str]) -> None:
    namespace, _ = lifecycle
    assert namespace['with_ramp'] > namespace['without_ramp'], 'popping the key has to give the objective back'
