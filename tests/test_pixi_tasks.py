"""A pixi task's ``cmd`` is a shell script, so its newlines separate commands.

A task written across lines for readability therefore runs as several commands,
and the shape that hides it is a long invocation whose first line is already
valid on its own::

    cmd = \"\"\"
    pytest bench --benchmark-memory
      --cases dispatch transport storage fleet
      --budget 30 --benchmark-json=bench/results/latest.json
    \"\"\"

There the selection, the budget and the results file are three separate
commands that never reach pytest. A line ending in an operator (``&&``, ``|``)
is continued and is fine; what this refuses is a line that ends in the middle
of an argument list.
"""

import re
import tomllib
from pathlib import Path

import pytest

CONTINUATIONS = ('&&', '||', '|', '\\', '(', '{', ';')

REPO = Path(__file__).resolve().parents[1]

#: The shards `sweep.yml` names, e.g. `shard: ['0/2', '1/2']`.
_MATRIX = re.compile(r'shard:\s*\[([^\]]*)\]')


def tasks() -> dict[str, str]:
    """Every pixi task in the file, ``name`` to ``cmd``, features included."""
    config = tomllib.loads((REPO / 'pyproject.toml').read_text())
    pixi = config['tool']['pixi']
    tables = [pixi.get('tasks', {})] + [f.get('tasks', {}) for f in pixi.get('feature', {}).values()]
    return {
        name: body['cmd'] if isinstance(body, dict) else body
        for table in tables
        for name, body in table.items()
        if isinstance(body, str) or 'cmd' in body
    }


@pytest.mark.parametrize('name', sorted(tasks()), ids=str)
def test_a_task_wrapped_across_lines_is_still_one_command(name: str) -> None:
    lines = [line.strip() for line in tasks()[name].strip().splitlines() if line.strip()]
    dangling = [line for line in lines[:-1] if not line.endswith(CONTINUATIONS)]
    assert not dangling, (
        f'`{name}` wraps onto the next line after {dangling} — pixi runs each line as its own '
        f'command, so everything below it is dropped from the invocation and then executed as a '
        f'program. End the line with `\\` to wrap it, or with `&&` to mean a second command.'
    )


def test_the_sweep_matrix_covers_every_shard_it_divides_the_space_into() -> None:
    """A leg missing from the matrix is a slice of the sweep nobody runs.

    The task's own `i/n` is the divisor, so the matrix has to name every offset
    it implies.
    """
    matrix = _MATRIX.search((REPO / '.github' / 'workflows' / 'sweep.yml').read_text())
    assert matrix, 'sweep.yml names no `shard:` matrix — the job takes one leg per shard'
    legs = [leg.strip().strip('\'"') for leg in matrix[1].split(',') if leg.strip()]
    divisors = {leg.partition('/')[2] for leg in legs}
    assert len(divisors) == 1, f'the legs {legs} divide the sweep into different numbers of shards'
    n = int(divisors.pop())
    assert sorted(legs) == sorted(f'{i}/{n}' for i in range(n)), (
        f'sweep.yml runs {legs}, which is not every shard of {n} — the missing offsets are cases '
        f'no job runs, and every job still passes'
    )
