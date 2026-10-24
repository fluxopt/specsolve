"""Build a wheel of each dependency ``pyproject.toml`` references directly, by git commit, into the directory given.

    python tools/pyodide/direct.py dist/pyodide/direct

micropip installs only wheels, and a requirement such as
``mathspec @ git+https://…@<commit>`` is a git URL it cannot read. ``run.mjs``
installs these wheels first, so specsolve's own wheel finds that requirement
met. With no direct reference this builds nothing.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

DIRECT = re.compile(r'^[A-Za-z0-9._-]+ @ git\+\S+$')


def main() -> None:
    (out,) = sys.argv[1:]
    declared = tomllib.loads(Path('pyproject.toml').read_text())['project']['dependencies']
    if direct := [spec for spec in declared if DIRECT.match(spec)]:
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--wheel-dir', out, *direct], check=True)


if __name__ == '__main__':
    main()
