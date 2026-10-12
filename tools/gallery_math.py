"""Each gallery page's math, generated from the model the page shows.

    pixi run python -m tools.gallery_math           # rewrite every page's math block
    pixi run python -m tools.gallery_math --check   # fail if any has drifted

Notation comes from ``examples/symbols/<model>.yaml`` where one exists, and is
derived otherwise.

The block is a ``<details markdown="1">``, because the gallery pages render on
GitHub as well as on the site: GitHub keeps ``<details>`` and drops the unknown
attribute, and the site's ``md_in_html`` needs ``markdown="1"`` to render the
tables and math inside it.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from mathspec import to_markdown

from tools.constructs import GALLERY, ROOT, models, replace_between

if TYPE_CHECKING:
    from pathlib import Path

SYMBOLS = ROOT / 'examples' / 'symbols'
BEGIN, END = '<!-- math:begin -->', '<!-- math:end -->'


def _block(name: str, path: Path) -> str:
    """The generated section for one model: a disclosure holding its math."""
    table = SYMBOLS / f'{name}.yaml'
    math = to_markdown(path, symbols=table if table.exists() else None, legend=True)
    return f'<details markdown="1">\n<summary>The same model, as math</summary>\n\n{math}\n</details>'


def rendered(page: str, name: str, path: Path) -> str:
    """*page* with the block between the markers replaced."""
    return replace_between(page, BEGIN, END, _block(name, path))


def pages() -> list[tuple[str, Path, Path]]:
    """Every (name, model, page) the gallery covers whose page has the markers."""
    found = []
    for name, path in models():
        page = GALLERY / f'{name}.md'
        if page.exists() and BEGIN in page.read_text():
            found.append((name, path, page))
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='fail if any committed block has drifted')
    opts = ap.parse_args(argv)

    work = [(name, page_path, rendered(page_path.read_text(), name, path)) for name, path, page_path in pages()]

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
