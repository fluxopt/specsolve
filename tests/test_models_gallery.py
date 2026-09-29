"""The gallery says what the repo actually contains.

A docs page that shows a model is a **copy** of that model, and a copy rots
unless something asserts it. Three things are checked, each a different way for
the page to become a lie:

- a model exists with no page, so the gallery quietly under-sells the language;
- a model has a page the catalogue does not list, which under-sells it just as
  quietly — the reader who never scrolls to the construct matrix never sees it;
- a page shows YAML that no longer matches the file CI runs;
- a page shows a reference implementation that no longer matches the script;
- the construct matrix says a model exercises something it does not;
- a model, or a declaration in one, says nothing about what it is, so the
  generated legend has an empty `Meaning` column where the reader needs one.

The same trade the linopy lane's v1-absence helpers already make: copying is fine when a
test asserts it, and rots when nothing does.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest
from mathspec import to_spec

from tools import constructs, gallery_math

GALLERY = Path(__file__).resolve().parent.parent / 'docs' / 'examples'


@pytest.fixture(params=constructs.models(), ids=lambda m: m[0])
def example(request: pytest.FixtureRequest) -> tuple[str, Path]:
    return request.param


def test_every_model_has_a_page(example: tuple[str, Path]) -> None:
    name, _ = example
    assert (GALLERY / f'{name}.md').exists(), (
        f'{name} has no gallery page. A model with no page is invisible to a reader '
        f'deciding whether the language can say theirs.'
    )


def _fences(markdown: str, lang: str) -> list[str]:
    """Every ``lang`` fence, dedented.

    A fence inside a content tab is indented by the tab's four spaces, and the
    byte-for-byte checks below compare against file text — so the tab indent
    is stripped from every line that carries it, and a blank line (which
    markdown allows to stay empty inside an indented block) passes through.
    """
    bodies = []
    for match in re.finditer(rf'^([ \t]*)```{lang}\n(.*?)^\1```', markdown, re.MULTILINE | re.DOTALL):
        prefix, body = match.group(1), match.group(2)
        bodies.append(''.join(line.removeprefix(prefix) for line in body.splitlines(keepends=True)))
    return bodies


def test_the_page_shows_the_model_that_runs(example: tuple[str, Path]) -> None:
    """A YAML fence on the page equals the model file, byte for byte."""
    name, path = example
    fences = _fences((GALLERY / f'{name}.md').read_text(), 'yaml')
    assert path.read_text().rstrip() + '\n' in fences, f'docs/examples/{name}.md has drifted from {path}'


def test_the_model_says_what_it_is(example: tuple[str, Path]) -> None:
    """Every gallery model carries a file `description:`.

    The legend a page generates opens with it, so a model without one opens
    with nothing — and the sentence is the only part of the page a reader who
    never opens the YAML is certain to read.
    """
    _, path = example
    assert to_spec(path).description, (
        f'{path} has no top-level `description:` — the generated legend on its page opens with it'
    )


def test_no_page_without_a_model() -> None:
    """The reverse: a page for a model that was deleted or renamed."""
    named = {name for name, _ in constructs.models()} | {'index', 'data', 'pypsa_ladder'}
    orphans = sorted(p.stem for p in GALLERY.glob('*.md') if p.stem not in named)
    assert not orphans, f'gallery pages with no model behind them: {orphans}'


def test_the_gallery_math_is_current() -> None:
    """Each page's math block equals what the model produces.

    The fourth way a page becomes a lie, and the one that had already
    happened: three pages stated math their model does not build. A summary
    can be loose — it is prose, and it is meant to be read at a glance — but
    the exact statement beside it has to be exact, and only a generator keeps
    it that way.
    """
    assert gallery_math.main(['--check']) == 0, 'stale gallery math'


def test_every_page_with_a_model_carries_a_math_block() -> None:
    """A page added without the markers would silently opt out of the check
    above, which is the failure mode the check exists to prevent."""
    missing = [
        name for name, _ in constructs.models() if gallery_math.BEGIN not in (GALLERY / f'{name}.md').read_text()
    ]
    assert not missing, (
        f'gallery pages with no math block: {missing} — add the '
        f'{gallery_math.BEGIN}/{gallery_math.END} markers under "## The model"'
    )


def test_every_math_block_opts_into_markdown_inside_html(example: tuple[str, Path]) -> None:
    """`<details>` without `markdown="1"` renders its contents as literal text
    on the site, and the strict build does not notice — literal text is valid.

    The two renderers disagree here and only one of them complains. GitHub
    processes Markdown inside `<details>` regardless and drops the unknown
    attribute; mkdocs has `md_in_html`, which needs it. So the attribute is
    free on one side and load-bearing on the other, which is exactly the kind
    of thing that ships broken.
    """
    name, _ = example
    page = (GALLERY / f'{name}.md').read_text()
    assert '<details markdown="1">' in page, (
        f'docs/examples/{name}.md has a math block whose <details> does not carry '
        f'markdown="1" — its tables and $$ blocks will be literal text on the site'
    )


def test_the_catalogue_lists_every_model(example: tuple[str, Path]) -> None:
    """The section head says *every* model, and the nav is what makes that true.

    ``strict: true`` fails the build on a page missing from the nav, and the
    catalogue is generated from the nav — but only from the *groups* under
    Models. A page filed anywhere else, or as a loose entry beside the data
    page, still builds and still disappears from the list.
    """
    name, _ = example
    listed = {name for _, pages in constructs.nav_groups() for _, name in pages}
    assert name in listed, (
        f'{name} is in no group under `Models:` in mkdocs.yml, so the gallery catalogue omits it — '
        f'a reader deciding whether the language can say their model never sees it'
    )


def test_the_generated_blocks_are_current() -> None:
    """All three of the gallery's blocks, which is the whole point of generating them.

    The catalogue comes from the nav and from each page's opening line, so a
    model cannot be renamed, regrouped or re-described into a list that still
    claims to be every model. The construct matrix comes from the resolved
    plan, so a model that gains a construct and a table that does not mention
    it cannot both be committed. The reference table comes from
    ``references.json``, the same file ``test_ports.py`` asserts against — so
    the *published* optimum and the *asserted* one cannot disagree.
    """
    page = constructs.PAGE.read_text()
    assert constructs.rendered(page) == page, 'the gallery page is stale — run `pixi run python -m tools.constructs`'


@pytest.fixture(scope='module')
def exercised() -> set[str]:
    """The union of every construct some model in the corpus exercises."""
    return set().union(*(constructs.constructs(path) for _, path in constructs.models()))


@pytest.mark.parametrize('column', constructs.COLUMNS)
def test_every_construct_is_exercised_by_some_model(column: str, exercised: set[str]) -> None:
    """No column of the construct matrix is all dots.

    The checks above keep the matrix *true*; this one keeps it *full*. A
    construct the language ships that no model exercises renders as a column
    of `·`, visible only to a reader scanning for the hole — the same claim
    `test_resolution_parity.test_every_resolved_predicate_is_parity_tested`
    makes one level down.
    """
    assert column in exercised, (
        f'`{column}` ships, but no model in examples/ or examples/ports/ exercises it — '
        f'the gallery matrix renders it as an empty column'
    )


PORTS = Path(__file__).resolve().parent.parent / 'examples' / 'ports' / 'references'

#: Directory name under ``references/`` -> the tab title the gallery shows.
#: Adding an arm (``pyomo/``) means adding its display name here — the tests
#: below then demand a tab for every script it holds, and refuse a tab with no
#: script, so the docs and the corpus cannot say different things about which
#: libraries a model is shown in.
ARMS = {'linopy': 'linopy', 'pypsa': 'PyPSA'}


@pytest.fixture(params=sorted(PORTS.glob('*/*.py')), ids=lambda p: f'{p.parent.name}-{p.stem}')
def reference(request: pytest.FixtureRequest) -> Path:
    return request.param


def test_every_arm_directory_is_named(reference: Path) -> None:
    arm = reference.parent.name
    assert arm in ARMS, (
        f'{reference} sits in an arm directory `{arm}` that ARMS does not name — '
        f'without a display name the tab checks below cannot see it'
    )


def test_the_page_shows_the_reference_that_runs(reference: Path) -> None:
    """The reference tab embeds the script's modelling half, and a comparison
    about readability has to show code that still exists in that form.

    Caught its own first regression: `ruff format` reflowed a `pivot` chain in
    `transport_dantzig.py` after the page had copied it, and nothing else would
    have noticed.
    """
    page = GALLERY / f'{reference.stem}.md'
    text = page.read_text()
    title = ARMS[reference.parent.name]
    assert f'=== "{title}"' in text, (
        f'{page} shows no `=== "{title}"` tab — a reference with no tab is invisible to the reader it was written for'
    )
    assert '=== "specsolve"' in text, f'{page} has an arm tab but no `=== "specsolve"` tab beside it'
    assert _build_slice(reference) in _fences(text, 'python'), (
        f'{page} has drifted from the build function of {reference}'
    )


def _build_slice(reference: Path) -> str:
    """The reference's ``build`` function, exactly as the file holds it.

    The slice ends at the next top-level statement, so the run harness — the
    solve, the printing protocol, the dual extraction — never reaches the
    page. A reader comparing formulations wants the modelling against the
    YAML, like against like; the harness is a click away at the file itself.
    """
    body = reference.read_text()
    body = body[body.index('def build') :]
    end = min(i for i in (body.find('\n\n\ndef '), body.find('\n\n\nif ')) if i != -1)
    return body[:end].rstrip() + '\n'


def _call_snippet(name: str) -> str:
    """The specsolve tab's call block, derived rather than copied.

    Three lines: the solve, the objective it reaches, the dual the corpus
    checks — a projection of `references.json`, and this is its one home. A
    model with a formulation is handed over written out, which is the door's
    rule.
    `test_ports.py` executes the same call against the committed instance;
    the page only has to match it.
    """
    entry = constructs.REFERENCES[name]
    ports_yaml = constructs.ROOT / 'examples' / 'ports' / f'{name}.yaml'
    path = f'examples/ports/{name}.yaml' if ports_yaml.exists() else f'examples/{name}.yaml'
    written = to_spec(constructs.ROOT / path)
    handed = f"to_spec('{path}').expand()" if written.piecewise or written.sos else f"'{path}'"
    lines = [
        '# sources: parameter name -> frame or parquet path',
        f'with sps.solve({handed}, sources) as solution:',
        f'    solution.objective  # {entry["objective"]!r}',
    ]
    if entry.get('duals'):
        lines.append(f"    solution.dual('{next(iter(entry['duals']))}')")
    return '\n'.join(lines) + '\n'


def test_the_specsolve_tab_shows_the_call(reference: Path) -> None:
    """Beside a runnable script, a bare YAML file is half an answer.

    The arm tab is a complete program — build, solve, read the duals — so the
    specsolve tab carries the same journey: the model, then the call that takes
    the committed instance to the verified optimum.
    """
    page = GALLERY / f'{reference.stem}.md'
    assert _call_snippet(reference.stem) in _fences(page.read_text(), 'python'), (
        f'{page} does not show the call for {reference.stem} — regenerate it from _call_snippet, '
        f'which derives it from references.json'
    )


def test_no_tab_without_a_reference() -> None:
    """The reverse: a tab claiming an arm must have a script behind it.

    The arm tabs exist to show code that ran and matched the recorded optimum;
    a tab whose script was deleted or renamed would keep showing code nobody
    can run, which is the drift the whole side-by-side machinery exists to
    prevent.
    """
    arm_of = {display: arm for arm, display in ARMS.items()}
    for page in sorted(GALLERY.glob('*.md')):
        for title in re.findall(r'^=== "(.+)"$', page.read_text(), re.MULTILINE):
            if title == 'specsolve':
                continue
            arm = arm_of.get(title)
            assert arm is not None, f'{page} has a tab `{title}` that is neither specsolve nor a named arm'
            assert (PORTS / arm / f'{page.stem}.py').exists(), (
                f'{page} shows a `{title}` tab but examples/ports/references/{arm}/{page.stem}.py does not exist'
            )


#: Hand-written prose that shows model YAML, checked against the models that
#: run. `docs/run.md` is not here: the language is mathspec's and the tutorial
#: links it rather than teaching it, so it shows no expressions to check.
#: `README.md`'s block is what `docs/index.md` includes as the whole thing in
#: one model.
TEACHING_PAGES = (Path(__file__).resolve().parent.parent / 'README.md',)
#: Both spellings of a declaration: a list item under `constraints:` and a
#: mapping key under a named one.
_TAUGHT_START = re.compile(r'^([ \t]*)(?:-\s*)?(?:expression|where):\s*\S')
_QUOTED = ('expression', 'where')


def _normalise(text: str) -> str:
    """One expression, whitespace-flattened, so folding is not a difference.

    A YAML folded scalar joins its lines with single spaces, so the same math
    written on one line and over five parses to the same string only after the
    runs of whitespace collapse. Comparing *parsed* values rather than source
    lines is what lets the guide and the models each wrap where they read best.
    """
    return ' '.join(text.split())


def _taught(markdown: str) -> list[str]:
    """Every expression and ``where`` a page shows, as parsed strings.

    A snippet is its opening line plus the lines indented under it, so a folded
    scalar is collected whole rather than truncated to its ``>-`` header.
    """
    lines = markdown.split('\n')
    snippets: list[str] = []
    i = 0
    while i < len(lines):
        start = _TAUGHT_START.match(lines[i])
        if start is None:
            i += 1
            continue
        indent, block = len(start.group(1)), [lines[i]]
        i += 1
        while i < len(lines) and lines[i].strip() and _indent_of(lines[i]) > indent:
            block.append(lines[i])
            i += 1
        parsed = yaml.safe_load(textwrap.dedent('\n'.join(block)))
        entry = parsed[0] if isinstance(parsed, list) else parsed
        snippets.extend(_normalise(v) for k, v in entry.items() if k in _QUOTED)
    return snippets


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _declared(node: object) -> Iterator[str]:
    """Every ``expression`` and ``where`` string anywhere in a model file."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _QUOTED and isinstance(value, str):
                yield _normalise(value)
            yield from _declared(value)
    elif isinstance(node, list):
        for item in node:
            yield from _declared(item)


@pytest.mark.parametrize('page', TEACHING_PAGES, ids=lambda p: p.name)
def test_the_prose_teaches_lines_that_exist(page: Path) -> None:
    """Every expression a prose page shows is copied from a model that runs.

    Prose is prose, so nothing else would notice it drifting — and a page
    demonstrating syntax the compiler no longer accepts is worse than no page.
    Only expressions and `where` clauses are checked: the dimension blocks are
    deliberately written in the compact form to be read, not to be pasted.

    Compared as **parsed values**, not as source lines. The line form was
    equivalent only while every expression fitted on one line; once a model
    folds one across several, a line-wise check silently degrades to comparing
    ``- expression: >-`` against itself and asserts nothing about the math.

    The corpus is ``constructs.models()`` — the same list the gallery and the
    matrix are built from — rather than a glob of ``examples/*.yaml``, which
    silently excluded the two ports one directory down. A line taken from a
    port would have failed here for not existing.
    """
    declared = {text for _, path in constructs.models() for text in _declared(yaml.safe_load(path.read_text()))}
    taught = _taught(page.read_text())
    assert taught, f'no expressions found in {page.name} — the extractor is broken'
    for expression in taught:
        assert expression in declared, f'{page.name} teaches an expression no example model contains:\n  {expression}'


#: Every sentence that states how many ports there are, as ``(page, template)``.
#: A third one is added here, not asserted separately.
COUNTED_IN_PROSE = (
    ('examples/README.md', '| `ports/` | {} models somebody else already solved'),
    ('docs/examples/index.md', 'Two rows from {} ports'),
)


@pytest.mark.parametrize(('page', 'sentence'), COUNTED_IN_PROSE, ids=[page for page, _ in COUNTED_IN_PROSE])
def test_the_prose_counts_the_ports_there_are(page: str, sentence: str) -> None:
    """The stated size of the port corpus is the size of the port corpus.

    Written by hand in two files, and the failure it guards is not a stale
    number but a *silent* one: two port PRs each bump the count from the same
    starting number, git merges the identical edit with nothing to conflict on,
    and the total lands one short. Nothing else in the suite reads these
    sentences, so without this they are wrong until a person notices.
    """
    expected = sentence.format(len(constructs.ports()))
    assert expected in (constructs.ROOT / page).read_text(), (
        f'{page} does not say "{expected}", and examples/ports/ holds '
        f'{len(constructs.ports())} models — the count in the prose has drifted from the corpus'
    )
