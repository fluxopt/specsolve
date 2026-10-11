"""The gallery's math, and every model in it rendered.

The renderer's own tests are mathspec's. This module asserts that every model
in **this repository's corpus** renders in every format, and that each gallery
page's generated math block is current.
"""

from __future__ import annotations

import re
import subprocess  # noqa: F401  — used by the typst compile check
from pathlib import Path

import pytest
from mathspec import FORMATS, SymbolTable, to_latex, to_markdown, to_spec, to_typst, typeset

from tests.conftest import SPEC_PATHS
from tools import gallery_math

#: Every format, as a parametrize mark.
EVERY_FORMAT = pytest.mark.parametrize('fmt', list(FORMATS))

_OPERATORS = frozenset({r'\sum', r'\min', r'\max', r'\prod', r'\int'})
_SUBSCRIPTED = re.compile(r'(\\[a-zA-Z]+|[A-Za-z])\s*_\s*(?:\{([^{}]*)\}|(\S))')


@pytest.fixture
def typst():
    """The `typst` binary, or a skip — the only check the Typst output is real."""
    from shutil import which

    if which('typst') is None:
        pytest.skip('needs the typst binary')
    return which('typst')


#: Every model a format has to handle: the gallery corpus, plus the operator
#: probes the operators page renders as math.
TYPESET_PATHS = SPEC_PATHS


@EVERY_FORMAT
def test_every_example_renders(fmt):
    """The walk consumes the same AST as lowering, so anything ``check``
    accepts it must print — a node it forgot is an exception, not a blank."""
    for path in TYPESET_PATHS:
        assert typeset(path, fmt).strip()


#: Syntax that could only have come from one family of formats. Markdown is
#: absent: it *is* LaTeX math in a Markdown wrapper.
_FINGERPRINTS = {
    'latex': (r'\mathcal{', r'\mathit{', r'\sum_{', r'\begin{align'),
    'typst': ('cal(', 'italic("', 'sum_(', '#set '),
}


@pytest.mark.parametrize(
    ('name', 'foreign'),
    [('latex', 'typst'), ('typst', 'latex'), ('markdown', 'typst')],
)
def test_no_format_leaks_another_formats_syntax(name: str, foreign: str):
    """The seam's whole job. Typst syntax in the LaTeX output means the walk is
    spelling something itself instead of asking the format to.
    """
    text = '\n'.join(typeset(p, name, standalone=True) for p in TYPESET_PATHS)
    assert any(mark in text for mark in _FINGERPRINTS[name if name != 'markdown' else 'latex']), (
        f'{name} output contains none of its own syntax — is this test still reading anything?'
    )
    for mark in _FINGERPRINTS[foreign]:
        assert mark not in text, f'{foreign} syntax {mark!r} leaked into the {name} output'


def _generated(stem: str, legend: bool = False) -> str:
    """A shipped example rendered to Markdown with its committed symbol table."""
    return to_markdown(f'examples/{stem}.yaml', symbols=f'examples/symbols/{stem}.yaml', legend=legend)


def test_markdown_avoids_escapes_github_eats_inside_math():
    r"""GitHub runs Markdown's backslash-escape processing *inside* `$$`.

    `\,` arrives as a literal comma and `\;` as a semicolon, so `\forall\, s`
    renders as "\u2200, s" and `\,:\,` as ",:,". Letter-named macros are
    untouched and MathJax treats them identically, so the Markdown format uses
    those. LaTeX and Typst are unaffected — no Markdown processor sees them.
    """
    md = _generated('dispatch', legend=True)
    for block in md.split('$$')[1::2]:
        for eaten in (r'\,', r'\;', r'\!', r'\:'):
            assert eaten not in block, f'{eaten!r} does not survive GitHub inside math: {block!r}'


GALLERY = Path(__file__).resolve().parent.parent / 'docs' / 'examples'
#: Pages whose summary deliberately says something *else*, each with its reason.
#: `test_every_summary_declares_itself` fails on a page in neither list.
DIVERGENT = {
    'transport_dantzig': (
        'is the textbook statement of the transportation problem, with an abstract '
        'c_{ij}. The model is the GAMS instance, whose cost is distance times freight over 1000.'
    ),
    'tsp_mtz': (
        'is DFJ subtour elimination — the formulation the language refuses, which is '
        'the point of the section it sits in. The model is MTZ.'
    ),
}


def _summary(stem: str) -> str:
    """The ``$$`` blocks of a gallery page, *minus* the generated block.

    Keyed on the generated block's markers, since neither position nor heading
    holds for every page. A missing closing marker and one above its partner
    are the same failure.
    """
    path = GALLERY / f'{stem}.md'
    page = path.read_text()
    if gallery_math.BEGIN in page:
        begin = page.index(gallery_math.BEGIN)
        end = page.find(gallery_math.END, begin)
        assert end != -1, (
            f'{path}: has {gallery_math.BEGIN} with no {gallery_math.END} after it, '
            f'so the generated block cannot be separated from the hand-written math'
        )
        page = page[:begin] + page[end:]
    return '\n'.join(page.split('$$')[1::2])


def _symbols(latex: str) -> set[str]:
    """Every subscripted quantity, as `head_subscript` with braces dropped.

    Brace-insensitive because the two sides spell single-character subscripts
    differently by convention — a summary writes `c_g`, the generator `c_{g}` —
    and that is a spelling difference, not a disagreement about the math.
    """
    found = set()
    for head, braced, bare in _SUBSCRIPTED.findall(latex):
        if head in _OPERATORS:
            continue
        found.add(f'{head}_{f"{braced}{bare}".strip()}')
    return found


def test_every_summary_declares_itself():
    """A page with hand-written math says why it differs from the generated math."""
    with_math = {p.stem for p in GALLERY.glob('*.md') if p.stem != 'index' and _summary(p.stem).strip()}
    undeclared = with_math - set(DIVERGENT)
    assert not undeclared, (
        f'gallery summaries that do not explain how they differ from the generated math: '
        f'{sorted(undeclared)} — add each to DIVERGENT with its reason'
    )
    stale = set(DIVERGENT) - with_math
    assert not stale, f'declared pages that no longer carry hand-written math: {sorted(stale)}'


def test_typst_output_compiles(typst, tmp_path: Path):
    """The only check that the Typst output is real."""
    for path in TYPESET_PATHS:
        source = tmp_path / f'{path.stem}.typ'
        source.write_text(to_typst(path, standalone=True))
        typst.compile(str(source), output=str(tmp_path / f'{path.stem}.pdf'))


def _structural_errors(tex: str) -> list[str]:
    """The three ways generated LaTeX usually fails to compile.

    Not a substitute for running TeX — it cannot know whether ``\\mathcal``
    takes an argument — but brace balance, environment nesting and
    ``\\left``/``\\right`` pairing are exactly what a *generator* gets wrong,
    and they are checkable without a toolchain.
    """
    errors = []
    depth = 0
    for i, c in enumerate(tex):
        escaped = i > 0 and tex[i - 1] == '\\'
        if c == '{' and not escaped:
            depth += 1
        elif c == '}' and not escaped:
            depth -= 1
            if depth < 0:
                errors.append(f'unbalanced closing brace at offset {i}')
                break
    if depth > 0:
        errors.append(f'{depth} unclosed brace(s)')

    stack: list[str] = []
    for verb, environment in re.findall(r'\\(begin|end)\{(\w+\*?)\}', tex):
        if verb == 'begin':
            stack.append(environment)
        elif not stack:
            errors.append(rf'\end{{{environment}}} with nothing open')
        elif stack.pop() != environment:
            errors.append(rf'\end{{{environment}}} does not close the open environment')
    if stack:
        errors.append(f'environments left open: {stack}')

    left, right = tex.count(r'\left'), tex.count(r'\right')
    if left != right:
        errors.append(rf'\left/\right mismatch: {left} vs {right}')
    return errors


@pytest.mark.parametrize('path', TYPESET_PATHS, ids=lambda p: p.stem)
def test_the_latex_is_structurally_well_formed(path: Path):
    assert _structural_errors(to_latex(path, standalone=True)) == []


def test_a_generated_variable_carries_the_description_its_expander_gave_it():
    """`piecewise:` invents the λ weights, so nothing the author wrote can
    describe them — the expander is the only thing that knows what they are.

    Asked of the expanded model, because a block left as written typesets as
    the curve it states rather than as the rows that state it."""
    expanded = to_spec('examples/piecewise.yaml').expand('piecewise')

    assert 'convex-combination weight on a breakpoint' in to_latex(expanded)


# ---------------------------------------------------------------------------
# the committed symbol tables
# ---------------------------------------------------------------------------

#: Committed tables written in typst, by what they declare rather than by what
#: they are called — the notation is the file's own word (#740).
TYPST_TABLES = sorted(p for p in Path('examples/symbols').glob('*.yaml') if SymbolTable.load(p).notation == 'typst')


def test_the_typst_path_has_a_committed_artifact():
    assert TYPST_TABLES, (
        'no committed typst symbol table — the file-load, checked-against and compile paths '
        'for a typst table would again be reached only by an inline dict in this file'
    )


@pytest.mark.parametrize('table', TYPST_TABLES, ids=lambda p: p.name)
def test_a_committed_typst_table_compiles_beside_its_model(typst, tmp_path: Path, table: Path):
    """The table on disk, against the model on disk, through the compiler."""
    source = tmp_path / f'{table.name}.typ'
    source.write_text(to_typst(_spec_of(table), symbols=table, standalone=True))
    typst.compile(str(source), output=str(tmp_path / f'{table.name}.pdf'))


def test_the_table_loads_from_a_file_and_the_committed_one_applies():
    expanded = to_spec('examples/piecewise.yaml').expand('piecewise')
    tex = to_latex(expanded, symbols='examples/symbols/piecewise.yaml')
    assert r'\lambda_{' in tex
    assert r'k \in \mathcal{K}' in tex
    assert 'breakpoints of the cost curve' in tex


def _spec_of(table: Path) -> Path:
    """The model a committed symbol table belongs to.

    The name up to the first dot, so one model may carry a table per notation:
    `transport_dantzig.yaml` and `transport_dantzig.typst.yaml` both name
    `transport_dantzig`. The unsuffixed file is the one `tools/gallery_math.py`
    renders the page with.
    """
    stem = table.name.split('.')[0]
    candidates = [Path('examples') / f'{stem}.yaml', Path('examples/ports') / f'{stem}.yaml']
    spec = next((c for c in candidates if c.exists()), None)
    assert spec is not None, f'{table} names no model: looked in {[str(c) for c in candidates]}'
    return spec


@pytest.mark.parametrize('table', sorted(Path('examples/symbols').glob('*.yaml')), ids=lambda p: p.name)
def test_every_committed_symbol_table_still_fits_its_model(table: Path):
    """A sidecar is matched to its model by filename alone, so renaming a
    parameter leaves the table naming nothing; `checked_against` makes that an
    error, run here for every committed pair in its declared notation."""
    assert typeset(_spec_of(table), SymbolTable.load(table).notation, symbols=table).strip()
