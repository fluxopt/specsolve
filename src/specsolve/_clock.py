"""The clock a model's life is timed on: every lap, the run it belongs to, and the solves each run took.

One mark moves through a call. Each public call starts it, and each step ends
a lap at it, so the laps tile the call and add up to its wall time
(docs/development/architecture.md, "The phases").
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Callable
from time import perf_counter
from typing import Literal

import numpy as np
import polars as pl

from specsolve._answer.records import PHASES, Metrics

#: A name in [`PHASES`][specsolve._answer.records.PHASES], so a lap under a phase no
#: [`Metrics`][specsolve._answer.records.Metrics] column carries is refused by the
#: type checker.
Phase = Literal['load', 'tidy', 'attach', 'build', 'handoff', 'solve', 'readback', 'write', 'archive']

#: What a lap did inside its phase: a step of ``tidy``, ``attach``, ``build``
#: or ``handoff``, as [`Model.timings`][specsolve._api.Model.timings] names them.
#: A step is not checked against its phase.
Step = Literal[
    'read',
    'shape',
    'encode',
    'content',
    'fill',
    'settle',
    'assumptions',
    'specialize',
    'splice',
    'variable',
    'sos',
    'constraint',
    'objective',
    'matrix',
    'sink_tables',
    'digest',
    'match',
    'capability',
    'push',
    'solver_instance',
    'start',
]

#: How a step ends a lap: ``(phase, step, entry)``, as [`Clock.lap`][] takes them.
Lap = Callable[[Phase, Step | None, str | None], None]

#: The columns of [`Model.timings`][specsolve._api.Model.timings]: one row per lap.
TIMINGS_SCHEMA = {
    'run': pl.UInt32,
    'phase': pl.String,
    'step': pl.String,
    'entry': pl.String,
    'seconds': pl.Float64,
}


def untimed(phase: Phase, step: Step | None, entry: str | None) -> None:
    """A lap no clock takes: a step run outside a model."""


class Clock:
    """Every lap of one model's life, held as columns so a long-lived model's log stays small.

    Each step, ``(phase, step, entry)``, is stored once and a lap holds its
    number: 21.5 bytes a lap held, where a tuple of the five took 145 (#2207).
    Each run's seconds per phase are added up as its laps arrive, so a run's
    clocks are read without the log, and the whole life's are the runs' summed
    exactly and rounded once, so no run passes the whole, not even in the last
    place.
    """

    def __init__(self) -> None:
        self._steps: dict[tuple[str, str | None, str | None], int] = {}
        self._runs = array('I')
        self._codes = array('I')
        self._seconds = array('d')
        self._totals: list[dict[str, float]] = []
        #: Per run, how many solves it took and how many solver instances they took.
        self._counts: list[list[int]] = [[0, 0]]
        #: When the last lap ended, or the call began: where the next lap measures from.
        self._mark = perf_counter()

    def start(self) -> None:
        """Begin a call: the next [`lap`][] measures from now."""
        self._mark = perf_counter()

    def lap(self, phase: Phase, step: Step | None = None, entry: str | None = None) -> None:
        """Log the wall time since the last lap, or since [`start`][], under *phase*.

        Each lap begins where the one before it ended, so the laps of a call
        add up to the whole call, and a step between two named steps is in the
        lap after it. *step* names what the lap did inside *phase*, as the
        build names each entry. A call that raises keeps the laps it finished.
        """
        now = perf_counter()
        run = self.run
        self._runs.append(run)
        self._codes.append(self._steps.setdefault((phase, step, entry), len(self._steps)))
        self._seconds.append(now - self._mark)
        while len(self._totals) <= run:
            self._totals.append(dict.fromkeys(PHASES, 0.0))
        self._totals[run][phase] += now - self._mark
        self._mark = now

    def count_a_solve(self, *, reloaded: bool) -> None:
        """Count one solve toward this run, and one solver instance where it was *reloaded*."""
        counts = self._counts[-1]
        counts[0] += 1
        counts[1] += int(reloaded)

    def next_run(self) -> None:
        """End the run: a result has been handed back, so the laps from here on are the next one's."""
        self._counts.append([0, 0])

    @property
    def run(self) -> int:
        """Which run the laps belong to now: how many results the model has handed back."""
        return len(self._counts) - 1

    def metrics(self, columns: int, rows: int, nonzeros: int, run: int | None = None) -> Metrics:
        """The sizes given, with the counters and clocks over the whole life, or over *run* alone.

        The one place a [`Metrics`][specsolve._answer.records.Metrics] is built from
        a model: a result's, an archive's and a sweep slice's row is its
        run's. The clocks are the laps summed by phase.
        """
        clocks = self._clocks(run)
        solves, instances = (
            (sum(c[0] for c in self._counts), sum(c[1] for c in self._counts)) if run is None else self._counts[run]
        )
        return Metrics(
            columns=columns,
            rows=rows,
            nonzeros=nonzeros,
            solves=solves,
            solver_instances=instances,
            load_seconds=clocks['load'],
            tidy_seconds=clocks['tidy'],
            attach_seconds=clocks['attach'],
            build_seconds=clocks['build'],
            handoff_seconds=clocks['handoff'],
            solve_seconds=clocks['solve'],
            readback_seconds=clocks['readback'],
            write_seconds=clocks['write'],
            archive_seconds=clocks['archive'],
        )

    def _clocks(self, run: int | None) -> dict[str, float]:
        """Seconds per phase over every lap, or over *run*'s, every phase present."""
        if run is not None:
            return dict(self._totals[run])
        return {phase: math.fsum(totals[phase] for totals in self._totals) for phase in PHASES}

    def timings(self) -> pl.DataFrame:
        """Every lap as a row of [`TIMINGS_SCHEMA`][], in the order they ran."""
        steps = list(self._steps)
        codes = pl.Series(np.frombuffer(self._codes, dtype=np.uint32).copy())
        named = {
            column: pl.Series([step[i] for step in steps], dtype=pl.String).gather(codes)
            for i, column in enumerate(('phase', 'step', 'entry'))
        }
        return pl.DataFrame(
            {
                'run': np.frombuffer(self._runs, dtype=np.uint32).copy(),
                **named,
                'seconds': np.frombuffer(self._seconds, dtype=np.float64).copy(),
            },
            schema=TIMINGS_SCHEMA,
        )
