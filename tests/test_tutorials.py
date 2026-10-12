"""The tutorial and how-to pages whose blocks the site runs keep running, and keep claiming true things.

The four tutorials (``docs/run.md``, ``docs/tables.md``, ``docs/change.md``
and ``docs/sweep.md``) and
``docs/howto/fix-relax-remove.md`` and ``docs/howto/scenarios.md`` run every executed block, and each test pins
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
TIME = DOCS_DIR / 'sweep.md'
LOOPS = DOCS_DIR / 'change.md'
LIFECYCLE = DOCS_DIR / 'howto' / 'fix-relax-remove.md'
SCENARIOS = DOCS_DIR / 'howto' / 'scenarios.md'

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
def over_time() -> tuple[dict[str, Any], str]:
    return run(TIME)


def test_one_day_with_the_battery_costs_what_it_did(over_time: tuple[dict[str, Any], str]) -> None:
    namespace, _ = over_time
    assert namespace['result'].objective == pytest.approx(10850.0), 'the day the windows repeat solves to 10850'


def test_each_day_adds_the_same_size(over_time: tuple[dict[str, Any], str]) -> None:
    namespace, _ = over_time
    sizes = {row['days']: (row['columns'], row['rows']) for row in namespace['sizes']}
    assert sizes == {days: (28 * days, 24 * days) for days in (1, 3, 30, 365)}, 'the page says each day adds 28 and 24'


def test_a_window_keeps_its_size(over_time: tuple[dict[str, Any], str]) -> None:
    namespace, _ = over_time
    assert namespace['windows'].metrics['columns'].to_list() == [21] * 4, 'the page says each window has 21 columns'


def test_a_window_without_lookahead_empties_the_battery(over_time: tuple[dict[str, Any], str]) -> None:
    namespace, _ = over_time
    windows = namespace['windows']
    ends = windows.primal('soc').filter(pl.col('hour') % 3 == 2)['value'].to_list()
    assert ends == pytest.approx([0.0] * 4), 'the page says every window ends with the battery empty'
    gap = namespace['schedule_cost'](windows) / namespace['whole'].objective - 1
    assert gap == pytest.approx(0.12, abs=0.005), 'the page says the windows cost 12% more'


def test_lookahead_closes_the_gap_and_grows_the_window(over_time: tuple[dict[str, Any], str]) -> None:
    namespace, _ = over_time
    rows = namespace['rows']
    whole = namespace['whole'].objective
    costs = [row['cost'] for row in rows]
    assert costs[0] > costs[1] > costs[2], f'each hour of lookahead buys something: {costs}'
    assert costs[1] - whole < (costs[0] - whole) / 2, 'the page says one hour closes most of the gap'
    assert costs[2] == pytest.approx(whole, abs=0.1), 'the page says three hours close all of the gap'
    assert [row['columns'] for row in rows] == [21, 28, 42], 'each hour of lookahead makes every window larger'


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


def test_a_mip_is_priced_once_its_decisions_are_fixed(lifecycle: tuple[dict[str, Any], str]) -> None:
    namespace, _ = lifecycle
    assert namespace['priced'].objective == pytest.approx(namespace['mip'].objective), (
        'the page says the fixed LP has the same optimum as the MIP'
    )
    prices = namespace['priced'].dual('power_balance').sort('snapshot')['value'].to_list()
    assert prices == pytest.approx([10.0, 10.0, 50.0, 10.0]), 'gas sets the price in snapshot 2, and wind elsewhere'


def test_removing_a_constraint_moves_the_answer(lifecycle: tuple[dict[str, Any], str]) -> None:
    namespace, _ = lifecycle
    assert namespace['with_ramp'] > namespace['without_ramp'], 'popping the key has to give the objective back'


@pytest.fixture(scope='module')
def scenarios() -> tuple[dict[str, Any], str]:
    return run(SCENARIOS)


def test_a_cold_scenario_costs_more_and_burns_more_gas(scenarios: tuple[dict[str, Any], str]) -> None:
    namespace, _ = scenarios
    objectives = dict(namespace['scenarios'].record.select('scenario', 'objective').iter_rows())
    assert objectives == pytest.approx({'cold': 16530.0, 'mild': 10500.0}), 'one solve for each label of scenario'
    wide = namespace['gas'].pivot(on='scenario', index='snapshot', values='value').sort('snapshot')
    assert (wide['cold'] >= wide['mild']).all(), 'gas covers the extra load of the cold scenario'


def test_a_second_table_varies_with_the_same_column(scenarios: tuple[dict[str, Any], str]) -> None:
    namespace, _ = scenarios
    objectives = dict(namespace['both'].record.select('scenario', 'objective').iter_rows())
    assert objectives['mild'] == pytest.approx(10500.0), 'the mild scenario keeps its cost table'
    assert objectives['cold'] > 16530.0, 'dearer gas in the cold scenario has to cost more'
