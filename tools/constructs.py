"""The gallery's catalogue and the two evidence tables beside the verification:
which models there are, what each exercises, and what somebody else says its
answer is.

    pixi run python -m tools.constructs           # rewrite every block on PAGES
    pixi run python -m tools.constructs --check   # fail if any has drifted

``tests/test_models_gallery.py`` asserts the committed blocks equal what this
produces.

The catalogue is read off ``mkdocs.yml``'s nav and each page's opening line.
Constructs are read off the lowered plan, not the YAML text. References are read
off ``examples/ports/references.json``, the file ``tests/test_ports.py`` asserts
against; adding a port is a JSON entry and a regenerate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterator

from mathspec import program, to_spec

ROOT = Path(__file__).resolve().parent.parent
GALLERY = ROOT / 'docs' / 'examples'
PAGE = GALLERY / 'index.md'
#: Every page that carries a generated block: the catalogue on the gallery, the
#: construct matrix and the reference table on the verification page.
PAGES = (PAGE, ROOT / 'docs' / 'development' / 'verification.md')
MKDOCS = ROOT / 'mkdocs.yml'
REFERENCES = json.loads((ROOT / 'examples' / 'ports' / 'references.json').read_text())
CAT_BEGIN, CAT_END = '<!-- catalogue:begin -->', '<!-- catalogue:end -->'
BEGIN, END = '<!-- constructs:begin -->', '<!-- constructs:end -->'
REF_BEGIN, REF_END = '<!-- references:begin -->', '<!-- references:end -->'

#: Column order is the order a reader meets these in docs/reference/language/.
COLUMNS = ('sum', 'sum(by=)', 'at()', 'shift', "shift(edge='wrap')", 'where', 'bounds', 'piecewise', 'sos', 'MILP')


def walk(node: Any) -> Iterator[Any]:
    """Every dataclass node reachable from *node*, itself included."""
    if is_dataclass(node) and not isinstance(node, type):
        yield node
        for f in fields(node):
            yield from walk(getattr(node, f.name))
    elif isinstance(node, Mapping):
        for item in node.values():
            yield from walk(item)
    elif isinstance(node, tuple | list):
        for item in node:
            yield from walk(item)


def constructs(spec: Path) -> set[str]:
    """The set of columns *spec* exercises.

    A bound counts only where it is not the open default. ``piecewise:`` is read
    off the surface schema, because it lowers away into a lambda formulation.
    """
    schema = to_spec(spec)
    lowered = schema.expand('piecewise').program
    nodes = list(walk(lowered))
    used: set[str] = set()

    for node in nodes:
        if isinstance(node, program.Sum):
            grouped = isinstance(node.operand, program.Join) and node.operand.columns.axes
            used.add('sum(by=)' if grouped else 'sum')
        elif isinstance(node, program.Join) and not node.columns.axes:
            used.add('at()')
        elif isinstance(node, program.Translate):
            used.add("shift(edge='wrap')" if node.wrap else 'shift')

    if any(isinstance(n, program.Condition) for n in nodes):
        used.add('where')
    if lowered.footprint.domains - {'continuous'}:
        used.add('MILP')
    if any(_bounded(v) for v in lowered.variables.values()):
        used.add('bounds')
    if schema.piecewise:
        used.add('piecewise')
    if lowered.sos:
        used.add('sos')
    return used


def _bounded(v: program.Variable) -> bool:
    open_at = {float('-inf'): 'lower', float('inf'): 'upper'}
    for side in ('lower', 'upper'):
        bound = getattr(v, side)
        if not (isinstance(bound, program.Constant) and open_at.get(bound.value) == side):
            return True
    return False


class _NavLoader(yaml.SafeLoader):
    """A loader that reads ``mkdocs.yml``'s ``!!python/`` tags as ``None``, since the nav needs none of them."""


_NavLoader.add_multi_constructor('tag:yaml.org,2002:python/', lambda *_: None)


def nav_groups() -> list[tuple[str, list[tuple[str, str]]]]:
    """The model groups of the site nav: group title, then its pages.

    The groups under Examples come first, then PyPSA examples, which the nav
    files under Development. Each page is ``(label, name)``: the sidebar label
    and the model's name, which is also its page and its YAML file. An entry
    that is not a group is skipped.
    """
    nav = yaml.load(MKDOCS.read_text(), Loader=_NavLoader)['nav']
    section = [*_section(nav, 'Examples'), {'PyPSA examples': _section(nav, 'PyPSA examples')}]
    groups = []
    for entry in section:
        if not isinstance(entry, dict):
            continue
        ((title, target),) = entry.items()
        if isinstance(target, list):
            groups.append((title, [(label, Path(path).stem) for page in target for label, path in page.items()]))
    return groups


def _section(entries: list, title: str) -> list:
    """The entries under the nav section called ``title``, at any depth."""
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ((label, target),) = entry.items()
        if label == title:
            return target
        if isinstance(target, list) and (found := _section(target, title)):
            return found
    return []


def _summary(name: str) -> str:
    """The opening paragraph of a model's page, as one line."""
    _, _, body = (GALLERY / f'{name}.md').read_text().partition('\n')
    summary = next(block for block in body.split('\n\n') if block.strip())
    return ' '.join(summary.split())


def catalogue() -> str:
    """Every model, grouped and described exactly as the sidebar groups it."""
    blocks = []
    for title, pages in nav_groups():
        rows = '\n'.join(f'| [{label}]({name}.md) | {_summary(name)} |' for label, name in pages)
        blocks.append(f'### {title}\n\n| | |\n|---|---|\n{rows}')
    return '\n\n'.join(blocks)


def _named(name: str, gallery: str) -> str:
    """*name* linked to its gallery page, or plain where the gallery shows no page for it."""
    return f'[{name}]({gallery}{name}.md)' if (GALLERY / f'{name}.md').exists() else f'`{name}`'


def table(models: list[tuple[str, Path]], gallery: str = '') -> str:
    """Markdown, one row per model, `·` where a construct is absent.

    The ``verified`` badge marks a port checked against an external optimum.
    """
    lines = [
        '| model | verified | ' + ' | '.join(f'`{c}`' if c != 'MILP' else c for c in COLUMNS) + ' |',
        '|---' * (len(COLUMNS) + 2) + '|',
    ]
    for name, path in models:
        used = constructs(path)
        cells = ['**✓**' if c in used else '·' for c in COLUMNS]
        badge = f'**✔** {REFERENCES[name]["objective"]:g}' if name in REFERENCES else '·'
        lines.append(f'| {_named(name, gallery)} | {badge} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def references_table(gallery: str = '') -> str:
    """One row per verified port, straight from ``references.json``.

    The optimum is the ``repr`` the assertion uses, not a rounding.
    Corroboration lands under the table as a footnote.
    """
    lines = [
        '| port | optimum | `rtol` | duals | reference |',
        '|---|---|---|---|---|',
    ]
    notes = []
    for name, entry in sorted(REFERENCES.items()):
        duals = '**✔**' if entry.get('duals') else '·'
        mark = f'[^{name}]' if entry.get('corroborated_by') else ''
        lines.append(
            f'| {_named(name, gallery)} | {entry["objective"]!r} | {entry["rtol"]:g} | '
            f'{duals} | {entry["provenance"]}{mark} |'
        )
        if corroborated := entry.get('corroborated_by'):
            notes.append(f'[^{name}]: {corroborated}')
    return '\n'.join(lines) + ('\n\n' + '\n\n'.join(notes) if notes else '')


def ports() -> list[Path]:
    """The ported models — somebody else's model, against somebody else's optimum."""
    return sorted((ROOT / 'examples' / 'ports').glob('*.yaml'))


def models() -> list[tuple[str, Path]]:
    """Every model the gallery shows, examples before ports."""
    examples = sorted((ROOT / 'examples').glob('*.yaml'))
    return [(p.stem, p) for p in examples] + [(p.stem, p) for p in ports()]


def replace_between(page: str, begin: str, end: str, body: str) -> str:
    """*page* with the generated block between two markers replaced by *body*."""
    i, j = page.index(begin) + len(begin), page.index(end)
    return page[:i] + '\n' + body + '\n' + page[j:]


def rendered(page: str, path: Path = PAGE) -> str:
    """*page*, which lives at *path*, with each generated block it carries replaced.

    A link in a block is relative to *path*, so a block reaches the model pages
    from whichever page carries it.
    """
    gallery = os.path.relpath(GALLERY, path.parent).replace(os.sep, '/') + '/'
    gallery = '' if gallery == './' else gallery
    blocks = (
        (CAT_BEGIN, CAT_END, catalogue),
        (BEGIN, END, lambda: table(models(), gallery)),
        (REF_BEGIN, REF_END, lambda: references_table(gallery)),
    )
    for begin, end, body in blocks:
        if begin in page:
            page = replace_between(page, begin, end, body())
    return page


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='fail if any committed block has drifted')
    opts = ap.parse_args(argv)

    stale = []
    for path in PAGES:
        page = path.read_text()
        updated = rendered(page, path)
        if updated == page:
            continue
        if opts.check:
            stale.append(str(path.relative_to(ROOT)))
        else:
            path.write_text(updated)
            print(f'{path.relative_to(ROOT)} refreshed')
    if stale:
        print(f'stale: {", ".join(stale)} — run `pixi run python -m tools.constructs`', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
