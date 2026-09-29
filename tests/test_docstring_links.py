"""Every link a docstring in `src/` writes lands on a specsolve object.

The strict site build checks the links of the docstrings it renders, which is
the public surface. Most links sit in modules the site never renders, where a
target that was renamed or removed leaves a link to nothing and no build
notices. This walks every docstring the package holds, resolves each link the
way the site does — in the docstring's own scope, walking outward — and
refuses one that lands nowhere, or on another package, whose names are plain
code rather than links.

griffe is the site's own reader, loaded with the extension the site uses to read
`#:` attribute comments, and ships with the docs toolchain: this runs under
`pixi run docs-test` and skips where the toolchain is absent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

griffe = pytest.importorskip('griffe', reason='the docs toolchain reads the docstrings; `pixi run docs-test`')

SRC = Path(__file__).resolve().parent.parent / 'src'

#: A link as mkdocstrings writes it: ``[`name`][]`` or ``[`name`][dotted.path]``.
LINK = re.compile(r'\[`([^`]+)`\]\[([^\]]*)\]')


def _target(scope: griffe.Object, name: str) -> str | None:
    """The full path *name* reaches from *scope*, walking outward the way a scoped link does."""
    if name.startswith('specsolve.'):
        return name
    first, _, rest = name.partition('.')
    while scope is not None:
        try:
            found = scope.resolve(first)
        except griffe.NameResolutionError:
            scope = scope.parent
            continue
        return f'{found}.{rest}' if rest else found
    return None


def _broken(package: griffe.Module) -> tuple[list[str], int]:
    """The links that land nowhere or outside specsolve, and how many links were read."""
    broken, read = [], 0
    stack = [package]
    while stack:
        obj = stack.pop()
        if obj.is_alias:
            continue
        stack.extend(obj.members.values())
        if obj.docstring is None:
            continue
        for display, written in LINK.findall(obj.docstring.value):
            read += 1
            path = _target(obj, written or display)
            try:
                if path is None or not path.startswith('specsolve.'):
                    raise KeyError(path)
                package[path.removeprefix('specsolve.')]
            except (KeyError, griffe.AliasResolutionError):
                broken.append(f'{obj.path}: [`{display}`][{written}] -> {path}')
    return sorted(broken), read


def test_every_docstring_link_lands_on_a_specsolve_object():
    package = griffe.load(
        'specsolve', search_paths=[SRC], extensions=griffe.load_extensions('griffe_sphinx'), resolve_aliases=False
    )
    broken, read = _broken(package)
    assert read > 260, f'the walk read {read} links, so it no longer reaches the docstrings it is for'
    assert not broken, (
        f'docstring links that land nowhere, or outside specsolve: {broken} — link a name as [`name`][] '
        f'where the module imports it, [`name`][dotted.path] where it does not, and write a name from '
        f'another package as plain code'
    )
