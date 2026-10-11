"""Does an interior-point solve without crossover pay on the ladder? HiGHS, three ways.

    pixi run -e bench python -m bench.crossover nodal transport --sizes s m l

Each case and rung is solved three ways, each on a model built for it, so no
solver state carries from one way to the next: HiGHS's own choice
(``solver='choose'``, simplex on these LPs), interior point followed by
crossover, and interior point with crossover off. It is not an arm and writes
no results file.

What each prints beside the solve's wall seconds: whether the answer carries
duals, and how far its objective is from the simplex one, relative to it. An
answer without crossover carries no basis, which ``tests/test_basis.py`` holds
on every sink, so it is not asked for here, and its read-back stays out of the
clock. Wall time is read off ``Model.diagnostics().seconds['solve']``, so it
counts the solve and not the build, and it prints the load averages beside
itself.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import specsolve as sps
from bench.cases import CASES

#: The ways each rung is solved, as HiGHS options.
METHODS: dict[str, dict[str, Any]] = {
    'simplex': {},
    'ipm': {'solver': 'ipm', 'run_crossover': 'on'},
    'ipm, no crossover': {'solver': 'ipm', 'run_crossover': 'off'},
}


def measured(case: str, size: str) -> list[tuple[str, float, float, bool]]:
    """``(method, solve seconds, objective, has duals)`` for each of [`METHODS`][] on one rung."""
    shape = CASES[case].shape(size)
    spec, data = CASES[case].spec_path(shape), CASES[case].data(shape)
    rows = []
    for method, options in METHODS.items():
        with sps.build(spec, data) as model:
            answer = model.solve(solver_options=options)
            rows.append((method, model.diagnostics().seconds['solve'], answer.objective, _has_duals(answer)))
    return rows


def _has_duals(answer: sps.types.Result) -> bool:
    """Whether *answer* carries duals, rather than a refusal saying why not."""
    try:
        answer._names('dual')
    except sps.errors.SpecsolveError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m bench.crossover', description=__doc__)
    parser.add_argument('cases', nargs='+', choices=sorted(CASES), help='the cases to solve')
    parser.add_argument('--sizes', nargs='+', default=['m'], help='the rungs of each case')
    args = parser.parse_args(argv)

    for case in args.cases:
        for size in args.sizes:
            rows = measured(case, size)
            simplex = rows[0][2]
            print(f'\n{case} {size}, simplex objective {simplex:.6g}')
            print('  method               solve s   relative gap   duals')
            for method, seconds, objective, duals in rows:
                gap = (objective - simplex) / abs(simplex)
                print(f'  {method:18}  {seconds:8.3f}   {gap:+.1e}        {duals!s}')
    one, five, fifteen = os.getloadavg()
    print(f'\n(load averages {one:.2f} {five:.2f} {fifteen:.2f} — only meaningful near zero)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
