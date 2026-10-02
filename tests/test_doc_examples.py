"""The examples in the docs, checked against the code.

:func:`test_every_block_is_covered` polices blocks that were matched, and
:func:`test_every_fence_is_seen` asserts every ``python``/``yaml`` fence was
matched. A block may be indented inside a list item, carry an info string after
the language (``python title="a.py"``), or use tilde fences; the code is
dedented before parsing.

Annotations go in an HTML comment on the line before the fence, so they are
invisible in rendered markdown:

    <!-- doctest: wrap=constraints -->   nest the block under that schema key
    <!-- doctest: skip -->               excluded, and the reason belongs in a comment

A YAML block with no annotation is validated whole, so it must resolve its
own cross-references.

In module docstrings, an example is an indented run introduced by ``::`` —
the reST literal-block marker.
"""

from __future__ import annotations

import ast
import functools
import re
import textwrap
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, NamedTuple, get_args

import pytest
import yaml
from mathspec import Spec, to_spec

import specsolve as sps
from specsolve.api import Model
from specsolve.relational.result import Result

try:
    from tests import linopy_lane
except ModuleNotFoundError:
    linopy_lane = None  # bare install, no oracle

REPO = Path(__file__).resolve().parent.parent
TRACKED = [
    'README.md',
    'docs/run.md',
    'docs/howto/warm-start.md',
    'docs/howto/fix-relax-remove.md',
    'docs/howto/solvers.md',
    'docs/howto/parallel.md',
    'docs/howto/debug.md',
    'docs/howto/archiving.md',
    'docs/howto/pandas-xarray.md',
    'docs/reference/api.md',
    'docs/reference/sweeps.md',
    'docs/about/linopy.md',
    'docs/about/architecture.md',
    'docs/about/roadmap.md',
    'docs/about/decomposition.md',
]

#: Names an example may dot into, and the object that decides what is valid.
#: Anything else (pd, np, network, ...) is external and not our contract.
ROOTS: dict[str, Any] = {
    'sps': sps,
    'specsolve_linopy': linopy_lane,
    'result': Result,
    'model': Model,
}

#: Every root an example may name, whether or not this install can resolve it.
ROOT_NAMES = frozenset(ROOTS)
ROOTS = {name: obj for name, obj in ROOTS.items() if obj is not None}


def _unresolvable(code: str) -> set[str]:
    """Roots this example names that the install cannot supply."""
    return {root for root in ROOT_NAMES - set(ROOTS) if f'{root}.' in code}


_EXTRA = 'needs the linopy oracle to check {}'

#: A fence may be ``` or ~~~, three or more, indented (inside a list item), and
#: may carry an info string after the language (```python title="a.py").
_FENCE = re.compile(
    r'(?:^[ \t]*<!--\s*doctest:\s*(?P<note>[^>]*?)\s*-->[ \t]*\n)?'
    r'^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?P<lang>python|yaml)\b[^\n]*\n'
    r'(?P<code>.*?)'
    r'^[ \t]*(?P=fence)[ \t]*$',
    re.DOTALL | re.MULTILINE,
)

#: Any fenced block, whatever its language.
_ANY_FENCE = re.compile(r'^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^\n]*)$', re.MULTILINE)


def _fence_openings(text: str) -> list[tuple[int, str]]:
    """(line, language) for every opening fence, by walking open/close pairs."""
    out: list[tuple[int, str]] = []
    open_delim: str | None = None
    for m in _ANY_FENCE.finditer(text):
        delim, info = m.group('fence'), m.group('info').strip()
        line = text.count('\n', 0, m.start()) + 1
        if open_delim is None:
            open_delim = delim
            out.append((line, info.split()[0] if info else ''))
        elif delim == open_delim:
            open_delim = None
    return out


class Block(NamedTuple):
    doc: str
    lang: str
    index: int
    code: str
    note: str
    line: int

    @property
    def where(self) -> str:
        return f'{self.doc}:{self.line} ({self.lang} block #{self.index})'


@functools.cache
def _blocks(lang: str | None = None) -> list[Block]:
    """Every tracked fenced block, optionally narrowed to one language, dedented.

    The recorded line is the fence itself, not the doctest comment above it.
    """
    out: list[Block] = []
    for doc in TRACKED:
        text = (REPO / doc).read_text()
        counters: dict[str, int] = {}
        for m in _FENCE.finditer(text):
            got = m.group('lang')
            i = counters.get(got, 0)
            counters[got] = i + 1
            out.append(
                Block(
                    doc=doc,
                    lang=got,
                    index=i,
                    code=textwrap.dedent(m.group('code')),
                    note=(m.group('note') or '').strip(),
                    line=text.count('\n', 0, m.start('fence')) + 1,
                )
            )
    return [b for b in out if lang is None or b.lang == lang]


def _public(obj: Any) -> set[str]:
    """Attribute names an example may use — dataclass fields included, since a
    field with no default is not a class attribute and ``dir`` misses it."""
    names = {n for n in dir(obj) if not n.startswith('_')}
    if is_dataclass(obj):
        names |= {f.name for f in fields(obj) if not f.name.startswith('_')}
    return names


def _undefined_attributes(tree: ast.AST) -> list[str]:
    """``root.attr`` uses whose root is ours and whose attr does not exist."""
    return sorted(
        {
            f'{n.value.id}.{n.attr}'
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id in ROOTS
            and n.attr not in _public(ROOTS[n.value.id])
        }
    )


# --------------------------------------------------------------------------
# python blocks
# --------------------------------------------------------------------------


@pytest.mark.parametrize('block', _blocks('python'), ids=lambda b: b.where)
def test_python_block_parses(block: Block) -> None:
    if block.note == 'skip':
        pytest.skip('explicitly skipped')
    try:
        ast.parse(block.code)
    except SyntaxError as exc:  # pragma: no cover - only on a broken doc
        pytest.fail(f'{block.where} is not valid Python: {exc}')


@pytest.mark.parametrize('block', _blocks('python'), ids=lambda b: b.where)
def test_python_block_uses_real_api(block: Block) -> None:
    """Every ``sps.x`` / ``result.x`` an example shows must exist."""
    if block.note == 'skip':
        pytest.skip('explicitly skipped')
    if missing := _unresolvable(block.code):
        pytest.skip(_EXTRA.format(sorted(missing)))
    bad = _undefined_attributes(ast.parse(block.code))
    assert not bad, f'{block.where} uses names that do not exist: {bad}. Fix the example, or the API it documents.'


def test_readme_example_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The README example solves to the objective its trailing comment states."""
    yaml_blocks = [b for b in _blocks('yaml') if b.doc == 'README.md']
    py_blocks = [b for b in _blocks('python') if b.doc == 'README.md']
    model = next(b for b in yaml_blocks if '# dispatch.yaml' in b.code)
    script = next(b for b in py_blocks if 'sps.solve' in b.code)

    (tmp_path / 'dispatch.yaml').write_text(model.code)
    monkeypatch.chdir(tmp_path)

    ns: dict[str, Any] = {}
    exec(compile(script.code, 'README.md', 'exec'), ns)

    result = ns['result']
    assert result.is_ok

    claimed = re.search(r'#\s*([0-9]+\.?[0-9]*)\s*$', script.code, re.MULTILINE)
    assert claimed, 'README example no longer states its objective in a comment'
    assert result.objective == pytest.approx(float(claimed.group(1))), (
        f'README claims objective {claimed.group(1)}, run produced {result.objective}'
    )


# --------------------------------------------------------------------------
# yaml blocks
# --------------------------------------------------------------------------


def _entry_model(section: str) -> Any:
    """The per-entry model behind a schema section, e.g. constraints -> ConstraintDef."""
    args = get_args(Spec.model_fields[section].annotation)
    return args[1] if len(args) == 2 else None


@pytest.mark.parametrize('block', _blocks('yaml'), ids=lambda b: b.where)
def test_yaml_block_validates(block: Block) -> None:
    """A YAML example must be a thing the schema accepts.

    A ``wrap=`` block is checked against its section's entry model; everything
    else is validated whole.
    """
    if block.note == 'skip':
        pytest.skip('explicitly skipped')

    doc = yaml.safe_load(block.code)
    assert isinstance(doc, dict), f'{block.where} is not a YAML mapping'

    if block.note.startswith('wrap='):
        section = block.note.removeprefix('wrap=')
        assert section in Spec.model_fields, f'{block.where}: wrap={section!r} is not a schema section'
        model = _entry_model(section)
        for name, entry in doc.items():
            try:
                model.model_validate(entry)
            except Exception as exc:
                pytest.fail(f'{block.where}: entry {name!r} does not validate:\n{exc}')
        return

    try:
        to_spec(doc)
    except Exception as exc:
        pytest.fail(
            f'{block.where} does not validate:\n{exc}\n\n'
            'If the block is not meant to be a complete model — a section shown '
            'on its own still has to resolve its cross-references, so a lone '
            '`parameters:` must declare the dims it names — annotate the fence '
            'instead:\n'
            '  <!-- doctest: wrap=<section> -->  a single entry of that section\n'
            '  <!-- doctest: skip -->            not a model, or wrong on purpose'
        )


# --------------------------------------------------------------------------
# the anti-rot guard
# --------------------------------------------------------------------------


def test_every_fence_is_seen() -> None:
    """`_FENCE` must match every python/yaml block a language-agnostic scan finds."""
    missed = []
    for doc in TRACKED:
        text = (REPO / doc).read_text()
        seen = {b.line for b in _blocks() if b.doc == doc}
        for line, lang in _fence_openings(text):
            if lang in ('python', 'yaml') and line not in seen:
                missed.append(f'{doc}:{line} (```{lang})')
    assert not missed, (
        'these blocks exist but _FENCE did not match them, so nothing checks '
        'them and no other test would notice:\n  ' + '\n  '.join(missed)
    )


def test_every_block_is_covered() -> None:
    """A new example must be checkable or explicitly skipped — never ignored."""
    unhandled = []
    for block in _blocks():
        if block.note == 'skip' or block.note.startswith('wrap='):
            continue
        if block.lang == 'python':
            continue
        keys = yaml.safe_load(block.code)
        if not isinstance(keys, dict) or not set(keys) <= set(Spec.model_fields):
            unhandled.append(block.where)
    assert not unhandled, (
        'these YAML blocks are neither whole schema sections nor annotated, so '
        'nothing checks them:\n  ' + '\n  '.join(unhandled) + '\n'
        'Add <!-- doctest: wrap=<section> --> or <!-- doctest: skip --> above the fence.'
    )


# --------------------------------------------------------------------------
# module docstrings
# --------------------------------------------------------------------------

DOCSTRING_MODULES = ['src/specsolve/__init__.py', 'src/specsolve/api.py', 'tests/linopy_lane/__init__.py']


def _docstring_examples(path: Path) -> list[str]:
    """Indented runs introduced by ``::`` — reST literal blocks."""
    tree = ast.parse(path.read_text())
    doc = ast.get_docstring(tree) or ''
    runs: list[tuple[str, list[str]]] = []
    current: list[str] = []
    lead = ''
    prev = ''
    for line in doc.splitlines():
        if not line.strip() or line.startswith('    '):
            if not current:
                lead = prev
            current.append(line)
            continue
        if current:
            runs.append((lead, current))
            current = []
        prev = line
    if current:
        runs.append((lead, current))

    out = []
    for lead, run in runs:
        text = '\n'.join(run).strip('\n')
        if not text.strip() or not lead.rstrip().endswith('::'):
            continue
        out.append(textwrap.dedent(text))
    return out


class Example(NamedTuple):
    module: str
    index: int
    code: str

    @property
    def where(self) -> str:
        return f'{self.module} (docstring example #{self.index})'


def _docstring_cases() -> list[Example]:
    """One case per example, so a skip names the example rather than the module."""
    return [
        Example(module, i, code)
        for module in DOCSTRING_MODULES
        for i, code in enumerate(_docstring_examples(REPO / module))
    ]


@pytest.mark.parametrize('module', DOCSTRING_MODULES)
def test_module_documents_its_api(module: str) -> None:
    """A module docstring with no example would leave the per-example tests collecting nothing."""
    assert _docstring_examples(REPO / module), f'{module}: no API example found in the module docstring'


@pytest.mark.parametrize('example', _docstring_cases(), ids=lambda e: e.where)
def test_docstring_example_uses_real_api(example: Example) -> None:
    """Every name a module docstring's example dots into has to exist; the name check needs the extra."""
    try:
        tree = ast.parse(example.code)
    except SyntaxError as exc:
        pytest.fail(f'{example.where} is not valid Python: {exc}\n{example.code}')
    if missing := _unresolvable(example.code):
        pytest.skip(_EXTRA.format(sorted(missing)))
    bad = _undefined_attributes(tree)
    assert not bad, f'{example.where} uses names that do not exist: {bad}'


def test_tracked_docs_exist() -> None:
    """Renaming a doc must not silently drop its examples from the sweep."""
    missing = [d for d in TRACKED if not (REPO / d).is_file()]
    assert not missing, f'tracked docs missing (update TRACKED): {missing}'
    assert _blocks('python'), 'no python examples found — the regex has drifted'
    assert _blocks('yaml'), 'no yaml examples found — the regex has drifted'
