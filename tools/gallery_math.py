"""Each gallery page's math, generated from the model the page shows.

    pixi run python -m tools.gallery_math           # rewrite every page's math block
    pixi run python -m tools.gallery_math --check   # fail if any has drifted

Notation comes from ``examples/symbols/<model>.yaml`` where one exists, and is
derived otherwise.

The block is a ``<details markdown="1">``, because the gallery pages render on
GitHub as well as on the site: GitHub keeps ``<details>`` and drops the unknown
attribute, and the site's ``md_in_html`` needs ``markdown="1"`` to render the
tables and math inside it.

``docs/index.md`` renders only on the site, so its block (marker ``home-math:``)
uses tabs and shows the LaTeX source beside the math.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml as pyyaml
from mathspec import to_latex, to_markdown

from tools.constructs import GALLERY, ROOT, models, replace_between

SYMBOLS = ROOT / 'examples' / 'symbols'
BEGIN, END = '<!-- math:begin -->', '<!-- math:end -->'

#: The home page shows one model end to end, and it is the quickstart's.
HOME = ROOT / 'docs' / 'index.md'
HOME_MODEL = ROOT / 'examples' / 'dispatch.yaml'
HOME_BEGIN, HOME_END = '<!-- home-math:begin -->', '<!-- home-math:end -->'


def _block(name: str, path: Path) -> str:
    """The generated section for one model: a disclosure holding its math."""
    table = SYMBOLS / f'{name}.yaml'
    math = to_markdown(path, symbols=table if table.exists() else None, legend=True)
    return f'<details markdown="1">\n<summary>The same model, as math</summary>\n\n{math}\n</details>'


def _indent(text: str) -> str:
    """Tab content — four spaces, and blank lines stay blank rather than ragged."""
    return '\n'.join(f'    {line}' if line else '' for line in text.splitlines())


def _literal(table: Path) -> str:
    """*table* as the Python dict literal that `symbols=` accepts.

    The output must be what `ruff format` would write, because ruff reaches into
    ```python fences in Markdown.
    """
    raw = pyyaml.safe_load(Path(table).read_text())
    lines = ['symbols = {']
    for section, entries in raw.items():
        if not isinstance(entries, dict):
            lines.append(f'    {section!r}: {entries!r},')
            continue
        lines.append(f'    {section!r}: {{')
        lines += [f'        {key!r}: {value!r},' for key, value in entries.items()]
        lines.append('    },')
    lines.append('}')
    return '\n'.join(lines)


def _home_block() -> str:
    """The home page's tabs: the math, its LaTeX source, and the call that printed it."""
    table = SYMBOLS / 'dispatch.yaml'
    options = {'symbols': table, 'legend': True}
    return '\n'.join(
        f'=== "{title}"\n\n{_indent(body)}\n'
        for title, body in {
            'The math': to_markdown(HOME_MODEL, **options),
            'LaTeX': f'```latex\n{to_latex(HOME_MODEL, **options).rstrip()}\n```',
            'How': _HOW.format(symbols=_literal(table)),
        }.items()
    )


_HOW = """```python
import mathspec as ms

{symbols}

ms.to_latex('dispatch.yaml', symbols=symbols)  # amsmath align
ms.to_typst('dispatch.yaml')  # compiles without a TeX toolchain
ms.to_markdown('dispatch.yaml')  # renders as-is on GitHub
```

`symbols` is optional — drop it and the same model prints as
$\\mathit{{load}}_t$, $p^{{\\mathrm{{max}}}}_g$. A dict, a YAML path or a
`SymbolTable`; a key naming nothing in the model is an error, not a symbol that
silently never applies. Every spelling is printed verbatim — `notation` says
which language they are, and a render in the other one refuses.

Or from a shell, where the table is that same YAML on disk and `--standalone`
emits a document that compiles rather than a fragment to `\\input`:

```bash
python -m mathspec latex dispatch.yaml --symbols dispatch.symbols.yaml
python -m mathspec typst dispatch.yaml --standalone -o dispatch.typ
```

The renderer is [mathspec](https://mathspec.readthedocs.io/en/latest/reference/typeset/)'s,
and reads the same file this page solves."""


def rendered(page: str, name: str, path: Path) -> str:
    """*page* with the block between the markers replaced."""
    return replace_between(page, BEGIN, END, _block(name, path))


def rendered_home(page: str) -> str:
    """``docs/index.md`` with its tabbed block replaced."""
    return replace_between(page, HOME_BEGIN, HOME_END, _home_block().rstrip('\n'))


def pages() -> list[tuple[str, Path, Path]]:
    """Every (name, model, page) the gallery covers whose page has the markers."""
    found = []
    for name, path in models():
        page = GALLERY / f'{name}.md'
        if page.exists() and BEGIN in page.read_text():
            found.append((name, path, page))
    return found


def _home_has_block(home: str, ap: argparse.ArgumentParser) -> bool:
    """Whether ``docs/index.md`` carries the tabbed block; error on anything but one pair or none."""
    found = (home.count(HOME_BEGIN), home.count(HOME_END))
    if found == (1, 1):
        return True
    if found != (0, 0):
        ap.error(
            f'{HOME.relative_to(ROOT)}: found {found[0]}x {HOME_BEGIN} and {found[1]}x {HOME_END}; '
            f'expected exactly one of each, or neither. Restore both markers around the tabs.'
        )
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='fail if any committed block has drifted')
    opts = ap.parse_args(argv)

    work = [(name, page_path, rendered(page_path.read_text(), name, path)) for name, path, page_path in pages()]

    home = HOME.read_text()
    if _home_has_block(home, ap):
        work.append(('index', HOME, rendered_home(home)))

    stale = []
    for name, page_path, updated in work:
        if updated == page_path.read_text():
            continue
        if opts.check:
            stale.append(name)
        else:
            page_path.write_text(updated)

    if opts.check:
        if stale:
            print(
                f'stale math on {len(stale)} page(s): {", ".join(stale)}\nrun `pixi run python -m tools.gallery_math`',
                file=sys.stderr,
            )
            return 1
        print(f'{len(work)} pages match their models')
        return 0
    print(f'{len(work)} pages refreshed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
