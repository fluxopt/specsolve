"""The docs are read on GitHub and on the site; these checks keep them right in both.

A link inside ``docs/`` is relative and the strict build validates it. A link
outside ``docs/`` is a full GitHub URL, because the site has nothing above
``docs/`` to resolve to. The builder checks neither a relative link that
escapes ``docs/`` nor a blob URL.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any, get_args

import yaml

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / 'docs'
REPO_URL = 'https://github.com/fluxopt/specsolve'
BLOB = f'{REPO_URL}/blob/main'

#: `](target)` and `[label]: target`, the two ways markdown names a destination.
_TARGETS = re.compile(r'\]\(\s*([^)\s]+)|^\[[^\]]+\]:\s+(\S+)', re.MULTILINE)

#: Already absolute, a bare fragment, or a protocol that names no path.
_ABSOLUTE = re.compile(r'^([a-z][a-z0-9+.-]*:|//|#|/)', re.IGNORECASE)


@functools.cache
def _pages() -> tuple[Path, ...]:
    """Every Markdown file under `docs/`, `README.md` included."""
    return tuple(sorted(DOCS.rglob('*.md')))


def _targets(page: Path) -> list[str]:
    return [inline or reference for inline, reference in _TARGETS.findall(page.read_text())]


def test_no_relative_link_escapes_the_docs_tree():
    """`[x](../CONTRIBUTING.md)` works in the repo, and the build ships it as a 404 on the site."""
    escaping = []
    for page in _pages():
        for target in _targets(page):
            if _ABSOLUTE.match(target):
                continue
            path = target.partition('#')[0]
            if not path:
                continue
            resolved = (page.parent / path).resolve()
            if resolved != DOCS and DOCS not in resolved.parents:
                escaping.append(f'{page.relative_to(REPO)} -> {target}')
    assert not escaping, (
        f'relative links pointing outside docs/, which 404 on the site: {escaping}\nwrite them as {BLOB}/<path> instead'
    )


def test_every_blob_url_names_a_file_that_exists():
    """A blob URL is checked by nothing else: the build treats it as external and never follows it."""
    broken = []
    for page in _pages():
        for target in _targets(page):
            if not target.startswith(BLOB):
                continue
            relative = target.removeprefix(f'{BLOB}/').partition('#')[0]
            if not (REPO / relative).exists():
                broken.append(f'{page.relative_to(REPO)} -> {relative}')
    assert not broken, f'links to repo files that no longer exist: {broken}'


def test_links_to_our_own_files_are_all_spelled_the_same_way():
    """One spelling, so the check above cannot be dodged.

    A file link written as `tree/`, `raw/`, a permalinked sha or another
    branch skips the existence check, which only recognises `blob/main`. Issue
    and PR links are not file links.
    """
    file_shaped = re.compile(rf'^{re.escape(REPO_URL)}/(blob|tree|raw|blame)/')
    stray = [
        f'{page.relative_to(REPO)} -> {target}'
        for page in _pages()
        for target in _targets(page)
        if file_shaped.match(target) and not target.startswith(f'{BLOB}/')
    ]
    assert not stray, f'links at repo files not written as {BLOB}/<path>: {stray}'


def test_the_convention_is_actually_in_use():
    """The assertions above pass vacuously on a docs tree with no outbound links, so pin that the links exist."""
    urls = [t for page in _pages() for t in _targets(page) if t.startswith(BLOB)]
    assert len(urls) >= 15, f'expected the docs to link out to the repo; found {len(urls)}'


def _nav_pages(entries: list[Any]) -> list[str]:
    """Every page the nav points at, depth first, as `mkdocs.yml` spells it.

    An entry is a bare path or a one-key mapping whose value is a path or a
    deeper list. A value that does not end in `.md` is not a page of this tree.
    """
    found: list[str] = []
    for entry in entries:
        target = entry if isinstance(entry, str) else next(iter(entry.values()))
        if isinstance(target, list):
            found.extend(_nav_pages(target))
        elif isinstance(target, str) and target.endswith('.md'):
            found.append(target)
    return found


def test_every_page_under_docs_has_a_nav_entry():
    """Every page has a nav entry and every nav entry has a page; the build checks neither.

    `README.md` is the folder view GitHub renders and belongs in no nav.
    """
    config = yaml.safe_load((REPO / 'mkdocs.yml').read_text())
    nav = set(_nav_pages(config['nav']))
    pages = {page.relative_to(DOCS).as_posix() for page in _pages()} - {'README.md'}
    assert pages == nav, (
        f'pages with no nav entry in mkdocs.yml: {sorted(pages - nav)}; nav entries with no page: {sorted(nav - pages)}'
    )


def test_the_home_page_still_carries_its_math_block():
    """`docs/index.md` keeps the markers `tools.gallery_math --check` fills.

    The generator fills only the markers it finds, so a dropped marker would
    stop the check without a failure.
    """
    from tools import gallery_math

    page = (DOCS / 'index.md').read_text()
    assert gallery_math.HOME_BEGIN in page and gallery_math.HOME_END in page, (
        f'docs/index.md lost its {gallery_math.HOME_BEGIN}/{gallery_math.HOME_END} markers — '
        f'the LaTeX tabs are generated, and an unmarked page silently opts out'
    )


# --------------------------------------------------------------------------
# the ten rules, and the pages that elaborate them
# --------------------------------------------------------------------------

LANGUAGE = DOCS / 'reference' / 'language'
RULES = LANGUAGE / 'index.md'

#: A rule row: `| 7 | text | [Absence](absence.md#how-absence-travels) |`
_RULE_ROW = re.compile(r'^\|\s*(\d+)\s*\|(.+?)\|([^|]*)\|\s*$', re.MULTILINE)


def _rules() -> list[tuple[str, str, str]]:
    text = RULES.read_text()
    start = text.index('## Ten rules the language reduces to')
    return _RULE_ROW.findall(text[start : text.index('\n## The pages', start)])


def _headings(page: Path) -> set[str]:
    """Every heading in *page* as GitHub would slug it."""
    slugs = set()
    for line in page.read_text().splitlines():
        if line.startswith('#'):
            title = line.lstrip('#').strip()
            slugs.add(re.sub(r'[^a-z0-9 -]', '', title.lower()).replace(' ', '-'))
    return slugs


# --------------------------------------------------------------------------
# the operators as math


# --------------------------------------------------------------------------
# every construct as math


# --------------------------------------------------------------------------
# the error tree


# the lane, as a translation


def test_the_translation_table_names_every_built_in_operator():
    """`What a construct becomes` is a copy of the builder, so something checks it."""
    from mathspec import BUILTIN_NAMES

    page = (DOCS / 'about' / 'linopy.md').read_text()
    section = page.split('### What a construct becomes')[1].split('### The same language')[0]
    expressions = section.split('| In an expression |')[1].split('| A `where:` |')[0]
    shown = set(re.findall(r'^\| `(\w+)\(', expressions, re.MULTILINE))

    assert shown == set(BUILTIN_NAMES), (
        f"the translation table shows {sorted(shown)} against the language's "
        f'{sorted(BUILTIN_NAMES)} — every built-in needs the linopy call it becomes'
    )


def _gen_bus_direction(program: Any) -> Any:
    """One map read one way — what a `by=` node stands on, whichever operator takes it."""
    gen_bus = program.RelationDeclaration((('g', 'g'), ('bus', 'bus')), ('g',))
    return program.Direction('gen_bus', gen_bus, ('g',), ('bus',), ())


def test_the_plan_table_names_every_expression_node():
    """`The plan, node for node` is a copy of two dispatches, so something checks it.

    The fan-in cell is read back off :func:`fan_in`, because the compiler acts
    on that column.
    """
    from mathspec import program

    from specsolve.relational.engines.polars.fragments import fan_in

    page = (DOCS / 'about' / 'architecture.md').read_text()
    section = page.split('## The plan, node for node')[1].split('## The relational lane')[0]
    rows = dict(re.findall(r'^\| `(\w+)` \|[^|]*\| ([^|]*) \|', section, re.MULTILINE))

    x = program.Variable('x')
    nodes = {
        type(node).__name__: node
        for node in (
            program.Constant(1.0),
            program.Parameter('p'),
            x,
            program.Dual('c'),
            program.Negate(x),
            program.Add(x, x),
            program.Multiply(x, x),
            program.Divide(x, program.Parameter('p')),
            program.Power(program.Parameter('p'), program.Constant(2.0)),
            program.Sum(x, ('t',)),
            program.GroupSum(x, _gen_bus_direction(program)),
            program.Pullback(x, _gen_bus_direction(program)),
            program.Translate(x, 't', 1, wrap=False),
            program.WindowSum(x, 't', 3, wrap=False),
            program.Cases((program.Region(program.Mask(program.BooleanLiteral(True)), x),)),
            program.Named('e', x),
        )
    }
    assert set(nodes) == {c.__name__ for c in get_args(program.Expression)}, (
        'the instances below stand for every expression node, so a node added to the language is one here too'
    )
    assert set(rows) == set(nodes), (
        f"the table shows {sorted(rows)} against the plan's {sorted(nodes)} — "
        f'every expression node needs the two readings it becomes'
    )
    shown = {name: cell.strip() for name, cell in rows.items()}
    declared = {name: fan_in(node) for name, node in nodes.items()}
    assert shown == declared, f'the table calls these {shown}, the compiler answers {declared}'


#: A ``:::`` entry, which mkdocstrings renders from the named object's docstring.
API_ENTRY = re.compile(r'^::: (\S+)$', re.MULTILINE)


def test_every_name_the_package_exports_has_an_entry_on_the_api_page():
    """The reference is the docstrings, so a name without an entry has no reference at all."""
    import specsolve

    rendered = set(API_ENTRY.findall((DOCS / 'reference' / 'api.md').read_text()))
    missing = sorted(name for name in specsolve.__all__ if f'specsolve.{name}' not in rendered)
    assert not missing, f'names in specsolve.__all__ with no ::: entry on reference/api.md: {missing}'


#: A Sphinx role, which mkdocstrings prints as it stands.
SPHINX_ROLE = re.compile(r':(?:func|class|meth|attr|mod|data|exc|obj):`')


def test_no_docstring_links_with_a_sphinx_role():
    """A ``:func:`build``` prints on the site as the literal text, so a docstring links as ``[`build`][]``.

    The strict build fails on a link of that form that resolves to nothing, but
    it cannot tell a Sphinx role from prose.
    """
    found = [
        f'{path.relative_to(REPO)}:{number}'
        for path in sorted((REPO / 'src').rglob('*.py'))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if SPHINX_ROLE.search(line)
    ]
    assert not found, f'Sphinx roles, which the site prints literally: {found}'
