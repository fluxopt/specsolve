# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "specsolve[gurobi] @ git+https://github.com/fluxopt/specsolve@7bf011473",
#   "linopy @ git+https://github.com/PyPSA/linopy@master",
#   "pyomo>=6.7",
#   "pytest==9.1.1",
#   "pytest-benchmem>=0.5",
#   "pyarrow>=16",
# ]
# ///
"""Re-run the published comparison on the versions that produced it.

    uv run --locked bench/reproduce.py            # the published selection
    uv run --locked bench/reproduce.py --sizes xs # a smaller look

`bench/reproduce.py.lock` freezes every dependency, git commits included, and
`--locked` refuses to run if the resolution has drifted. The selection is read
from the `pixi run ladder` task in `pyproject.toml`, and the harness in `bench/`
runs it, so this needs the repository checked out.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def published() -> list[str]:
    """The selection `pixi run ladder` takes, with its argument defaults, read out of `pyproject.toml`."""
    import tomllib

    manifest = Path(__file__).resolve().parent.parent / 'pyproject.toml'
    task = tomllib.loads(manifest.read_text())['tool']['pixi']['feature']['bench']['tasks']['ladder']
    command = ' '.join(task['cmd'].split())
    for argument in task['args']:
        command = command.replace('{{ ' + argument['arg'] + ' }}', argument['default'])
    return command.split()[1:]


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent.parent
    command = [sys.executable, '-m', 'pytest', '-q', *published(), *argv]
    print(f'$ {" ".join(command)}\n', flush=True)
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
