"""A rolling horizon on specsolve, checked against the horizon it approximates.

    pixi run python examples/rolling/run.py

A storage schedule solved a window at a time with `solve_over`, against the
answer full foresight gives.

One file, `horizon.yaml`, written over a *local* index `t`. The same YAML is
solved as one window over the whole horizon — the reference — and then as
rolling windows of increasing lookahead, each handing its state of charge to
the next through `carry`.

**The store cycles in every schedule**: wind blows nightly and load triples by
day, so charging and discharging is worth doing inside any window. What
lookahead buys is the *end* of one. Charge a window cannot spend before its
horizon runs out is worth nothing to it, so it arrives empty and hands zero to
the next; a window that can see far enough has somewhere to spend it.

Windows advance eight hours against a twelve-hour cycle, so a boundary lands at
a different point of the day each time.

Three properties are asserted rather than printed:

- the schedule the sweep answers covers every snapshot exactly once — no overlap
  double-counted, no tail dropped
- rolling never beats full foresight, which is what myopia means
- the store is used in every schedule, so the gap is a quality difference and
  not storage quietly disappearing
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import specsolve as sps

HERE = Path(__file__).parent
MODEL = HERE / 'horizon.yaml'

DAY = 12  #: hours in a day — six of night, then six of daylight
DAYS = 4
PERIODS = DAY * DAYS
GENERATORS = ['wind', 'gas']

#: Wind blows at night when load is low; by day there is none and load triples.
NIGHT = [t % DAY < DAY // 2 for t in range(PERIODS)]
LOAD = [25.0 if night else 85.0 for night in NIGHT]
WIND = [70.0 if night else 0.0 for night in NIGHT]

#: Windows advance eight hours against a twelve-hour cycle.
STEP = 8

SOURCES = {
    'generator': pl.DataFrame({'generator': GENERATORS}),
    'p_max': pl.DataFrame(
        {
            'snapshot': [t for t in range(PERIODS) for _ in GENERATORS],
            'generator': GENERATORS * PERIODS,
            'value': [v for t in range(PERIODS) for v in (WIND[t], 200.0)],
        }
    ),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 40.0]}),
    'load': pl.DataFrame({'snapshot': range(PERIODS), 'value': LOAD}),
    'soc_initial': pl.DataFrame({'value': [0.0]}),
}


def full_foresight() -> sps.Sweep:
    """One window over the whole horizon — the answer rolling is measured against."""
    return sps.solve_over(
        MODEL,
        SOURCES,
        sps.EachWindow('snapshot', steps=PERIODS, lookahead=0, into='t'),
    )


def rolling(steps: int, lookahead: int) -> sps.Sweep:
    """Windows keeping *steps* coordinates and seeing *lookahead* beyond them.

    The carry names no coordinate: `soc` is over `(t)` and `soc_initial` over
    `()`, so `t` is what it collapses, and the row handed on is the last one the
    window *keeps* rather than the last it solved. With lookahead those differ,
    and the last solved row is a level the next window is about to recompute.
    """
    return sps.solve_over(
        MODEL,
        SOURCES,
        sps.EachWindow('snapshot', steps=steps, lookahead=lookahead, into='t'),
        carry={'soc_initial': 'soc'},
    )


def cost_of(sweep: sps.Sweep) -> float:
    """What the schedule cost, summed over the snapshots each window owns.

    A window objective covers its lookahead too, so summing them double-counts.
    `spend` is the model's own per-snapshot definition, and the sweep's answer
    keeps only the rows a window owns.
    """
    return float(sweep.evaluate('spend')['value'].sum())


def main() -> None:
    reference = full_foresight()
    best = cost_of(reference)
    peak = reference.primal('soc')['value'].max()
    print(f'{DAYS} days of {DAY} hours: six of cheap wind, then six of none.')
    print(f'full foresight   one window                      cost {best:>9.2f}   peak soc {peak:>6.1f}')
    print()

    for lookahead in (0, 4, 8):
        sweep = rolling(STEP, lookahead)
        schedule = sweep.primal('soc')
        assert schedule['snapshot'].to_list() == list(range(PERIODS)), 'the answer must cover the horizon'

        cost = cost_of(sweep)
        assert cost >= best - 1e-6, 'rolling cannot beat full foresight'
        assert schedule['value'].max() > 0, 'the store must be used in every schedule'
        print(
            f'rolling  steps={STEP:<3} lookahead={lookahead:<3} '
            f'windows {len(sweep):>2}   cost {cost:>9.2f}   peak soc {schedule["value"].max():>6.1f}'
            f'   +{100 * (cost - best) / best:>5.1f}%'
        )

    print()
    print('The store cycles in every schedule — this is not a model where storage')
    print('stops being worth having. What myopia costs is the end of each window:')
    print('charge a window cannot spend before its horizon runs out is worth')
    print('nothing to it, so it arrives empty and the next window starts from zero.')
    print('Lookahead gives it somewhere to spend that charge, and the gap closes.')
    print()
    print(f'every schedule above covers all {PERIODS} snapshots exactly once')


if __name__ == '__main__':
    main()
