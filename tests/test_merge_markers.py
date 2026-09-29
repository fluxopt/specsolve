"""No file in the tree carries a conflict marker.

A file nothing reads can carry markers through a merge unnoticed, and a test
that asks whether a sentence is *in* a file stays true with markers around it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent

#: A marker git writes, anchored at the start of a line. The trailing space
#: keeps this file's own prose, and a markdown `=======` rule, from matching.
MARKER = re.compile(r'^(<<<<<<< |>>>>>>> |=======$)', re.MULTILINE)


def _tracked() -> list[Path]:
    files = subprocess.run(
        ['git', 'ls-files', '-z'], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split('\0')
    return [REPO / f for f in files if f]


@pytest.mark.parametrize('path', _tracked(), ids=lambda p: str(p.relative_to(REPO)))
def test_no_file_carries_a_conflict_marker(path: Path) -> None:
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        pytest.skip('binary, or a path git tracks that the checkout does not hold')

    if path == Path(__file__):
        pytest.skip('this module names the markers it looks for')

    found = MARKER.search(text)
    assert not found, (
        f'{path.relative_to(REPO)} line {text[: found.start()].count(chr(10)) + 1} is a conflict '
        f'marker: {found.group(0)!r} — a merge was committed unresolved'
    )
