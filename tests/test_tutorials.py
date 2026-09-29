"""The pages whose blocks the site runs must keep running, and keep claiming true things.

``docs/change.md`` teaches the three loops a session actually has — update,
grow a coordinate set, patch the spec — and ``docs/howto/fix-relax-remove.md``
aims them at linopy's `fix` / `relax` / `remove`. Every executed block is a real call, so a
signature change breaks this test rather than leaving a page that reads fine and
errors in a reader's session.

Running is the weaker half, as with ``test_walkthrough.py``. The prose also
*claims* things: that the update loop loaded one model for three solves, that
growing an axis loads a second, and that a pin moves bounds rather than labels. A page that
executed but had stopped doing any of that would still be green here without
these assertions, and would teach the wrong loop.

Blocks are exec'd in order in one namespace, which is what markdown-exec's
``session=`` gives them on the site. The site does run them — a build is the
second place this would fail, several minutes later and only on a push.
"""

from __future__ import annotations

import contextlib
import io
import re
from typing import TYPE_CHECKING, Any

import pytest
from mathspec import to_spec

from tests.conftest import EXAMPLES_DIR

if TYPE_CHECKING:
    from pathlib import Path

REPO = EXAMPLES_DIR.parent
DOCS_DIR = REPO / 'docs'
LOOPS = DOCS_DIR / 'change.md'
LIFECYCLE = DOCS_DIR / 'howto' / 'fix-relax-remove.md'

#: A block markdown-exec runs, and only those: the fence carries `exec="true"`.
#: A plain ```python fence on the same page is a listing, and running it would
#: make this test disagree with the site about what the page does.
EXECUTED = re.compile(r'^```python[^\n]*\bexec="true"[^\n]*$\n(?P<code>.*?)^```$', re.DOTALL | re.MULTILINE)


def run(page: Path) -> tuple[dict[str, Any], str]:
    """One top-to-bottom run: the namespace it ends with, and what it printed.

    Runs from the repository root, which is where zensical runs the build and
    so where markdown-exec starts a block — which is what makes
    ``examples/dispatch.yaml`` resolve.
    """
    blocks = [match['code'] for match in EXECUTED.finditer(page.read_text())]
    assert blocks, f'{page.name} has no executed block, so this test would assert nothing'
    namespace: dict[str, Any] = {'__name__': '__tutorial__'}
    printed = io.StringIO()
    with contextlib.chdir(REPO), contextlib.redirect_stdout(printed):
        for block in blocks:
            exec(compile(block, str(page), 'exec'), namespace)
    return namespace, printed.getvalue()


@pytest.fixture(scope='module')
def session() -> tuple[dict[str, Any], str]:
    return run(LOOPS)


@pytest.fixture(scope='module')
def lifecycle() -> tuple[dict[str, Any], str]:
    return run(LIFECYCLE)


def test_the_update_loop_stays_on_the_fast_path(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    reused = namespace['reused']
    assert (reused.loads, reused.solves) == (1, 3), 'the page says three answers came off one loaded model'


def test_growing_a_coordinate_set_loads_again(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    assert namespace['grown'].loads == 2, 'the page says new coordinates cost a reload, and why that is fine'
    assert namespace['schedule'].height == 36, 'twelve snapshots against three generators'


def test_a_update_answers_what_a_fresh_build_answers(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    assert namespace['updated'] == pytest.approx(namespace['fresh']), (
        'the equality the page offers as the oracle for a loop that looks wrong'
    )


def test_the_added_constraint_changes_the_answer(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    assert namespace['ramped'] > namespace['base'], (
        'the ramp limit has to bind — a structural edit with no effect teaches nothing'
    )


def test_pinning_a_variable_stays_on_the_fast_path(lifecycle: tuple[dict[str, Any], str]) -> None:
    """The claim that makes a fix worth spelling as bounds rather than as a row."""
    namespace, _ = lifecycle
    assert namespace['pinning'].loads == 1, 'a pin writes bounds, so the solver keeps the model it has loaded'
    assert namespace['held'] > namespace['unpinned'], 'holding gas at 60 has to cost something, or it pins nothing'


def test_a_refused_edit_says_what_is_wrong(session: tuple[dict[str, Any], str]) -> None:
    _, printed = session
    assert 'does not name a declared dimension' in printed, "the load-time error is the page's error message"


def test_the_session_leaves_a_file(session: tuple[dict[str, Any], str]) -> None:
    """``to_yaml`` on the patched spec is what the reader diffs against the model."""
    import yaml

    namespace, _ = session
    patched = to_spec(namespace['spec'])
    written = patched.to_yaml()
    assert to_spec(yaml.safe_load(written)).to_dict() == patched.to_dict(), (
        'the review copy has to reload as the model it was written from'
    )
    assert 'ramp_up' in written and 'ramp_up' not in (EXAMPLES_DIR / 'dispatch.yaml').read_text(), (
        'and to differ from the file on disk — that difference is what the reader commits'
    )


def test_integrality_is_a_declaration_and_costs_the_duals(lifecycle: tuple[dict[str, Any], str]) -> None:
    """The relax page's claim: a domain edit is a rebuild, and a MILP has no prices."""
    namespace, printed = lifecycle
    assert namespace['milp'].has_primal, 'the integer solve still answers'
    assert 'duals are undefined for a mixed-integer model' in printed, 'and says why it cannot be priced'
    assert namespace['relaxed'].dual('power_balance').height == 6, 'the continuous declaration prices every row'


def test_removing_a_constraint_moves_the_answer(lifecycle: tuple[dict[str, Any], str]) -> None:
    namespace, _ = lifecycle
    assert namespace['with_ramp'] > namespace['without_ramp'], 'popping the key has to give the objective back'
