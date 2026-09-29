"""The math a page prints for GitHub has to reach MathJax on the site as well.

``mathspec.to_markdown`` writes inline math as the verbatim pair ``$`…`$``,
and Arithmatex reads neither that pair nor a ```math fence. The fence is a
``superfences`` entry in ``mkdocs.yml`` and the pair is
``tools.mdx_github_math``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / 'docs'

#: A math span as mathspec's typesetter prints it for GitHub — a fence, which
#: may be indented inside a tab or a list, and the verbatim inline pair.
_FENCED_MATH = re.compile(r'^[ \t]*```math$', re.MULTILINE)
_INLINE_MATH = re.compile(r'\$`[^`\n]+`\$')


def _site_markdown():
    """A renderer configured exactly as the site's, from `mkdocs.yml` itself."""
    markdown = pytest.importorskip('markdown', reason='the docs group; the default environment skips it')
    config = pytest.importorskip('zensical.config', reason='the docs group; the default environment skips it')
    site = config.parse_config(str(REPO / 'mkdocs.yml'))
    return markdown.Markdown(extensions=site['markdown_extensions'], extension_configs=site['mdx_configs'])


@pytest.mark.parametrize(
    'page',
    sorted(p for p in DOCS.rglob('*.md') if _FENCED_MATH.search(p.read_text())),
    ids=lambda p: p.stem,
)
def test_the_site_renders_the_math_the_page_prints_for_github(page: Path) -> None:
    source = page.read_text()
    printed = len(_FENCED_MATH.findall(source)) + len(_INLINE_MATH.findall(source))
    html = _site_markdown().convert(source)
    rendered = html.count('class="arithmatex"')
    assert rendered >= printed, (
        f'{page.relative_to(REPO)} prints {printed} math spans and the site renders '
        f'{rendered} — the rest reach the reader as literal text'
    )
    assert '$`' not in html and 'language-math' not in html, 'no delimiter is left for the reader to see'


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        pytest.param('| $`\\mathcal{T}`$ | index $`t`$ |', '| $\\mathcal{T}$ | index $t$ |', id='inline-math'),
        pytest.param('the pair ``$`x`$`` inline', 'the pair ``$`x`$`` inline', id='the-syntax-quoted-in-prose'),
        pytest.param('```math\n\\mathrm{a\\_b}\n```', '```math\n\\mathrm{a\\_b}\n```', id='a-math-fence'),
        pytest.param(
            '    ```math\n    \\mathrm{a\\_b}\n    ```',
            '    ```math\n    \\mathrm{a\\_b}\n    ```',
            id='a-math-fence-indented-in-a-tab',
        ),
        pytest.param('```yaml\nname: $`x`$\n```', '```yaml\nname: $`x`$\n```', id='a-model-that-shows-the-syntax'),
        pytest.param(
            '````text\n$`x`$\n````\n\nand $`y`$ after',
            '````text\n$`x`$\n````\n\nand $y$ after',
            id='a-fence-opened-with-more-than-three-backticks',
        ),
    ],
)
def test_the_extension_rewrites_math_and_nothing_that_only_quotes_it(source: str, expected: str) -> None:
    """Inline math becomes `$…$`; a fence and a code span are left exactly as they are.

    A backtick on the outer edge is a code span quoting the delimiter. The
    last case holds the closing run to the opening one's length, because
    markdown-exec fences its source block with eight backticks.
    """
    pytest.importorskip('markdown', reason='the docs group; the default environment skips it')
    from tools.mdx_github_math import rewrite

    assert rewrite(source) == expected
