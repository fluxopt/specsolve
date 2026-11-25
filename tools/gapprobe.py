"""Where the unclocked time of a `tests/test_phases.py` call sits. A diagnostic plugin, not for merging.

    GAPLOG=gaps.jsonl pytest -p tools.gapprobe tests/test_phases.py

Every `Engine.start`, `Engine.lap`, engine construction, and the start and end
of each timed call is stamped with the real clock, the thread's CPU time and
its context switches. Each garbage collection is stamped too. For a call with
more than ``GAPMS`` (default 1) milliseconds unclocked, one JSON line names each
span between two marks that no lap covers: its wall and CPU milliseconds, the
voluntary and involuntary context switches inside it, and the collections that
started in it. A span whose wall time far exceeds its CPU time had the thread
off the CPU; one whose CPU time matches and holds a collection was the collector.
"""

from __future__ import annotations

import gc
import itertools
import json
import os
import resource
import time

_EVENTS: list[tuple[float, float, int, int, str, object]] = []
_THRESHOLD = float(os.environ.get('GAPMS', '1')) / 1e3


def _stamp(event: str, detail: object = None) -> None:
    usage = resource.getrusage(resource.RUSAGE_THREAD)
    _EVENTS.append((time.perf_counter(), time.thread_time(), usage.ru_nvcsw, usage.ru_nivcsw, event, detail))


def _collected(phase: str, info: dict) -> None:
    _stamp(f'gc-{phase}', info.get('generation'))


def pytest_configure(config) -> None:
    gc.callbacks.append(_collected)
    from specsolve._engine import engine

    start, lap, init = engine.Engine.start, engine.Engine.lap, engine.Engine.__init__

    def stamped_start(self):
        start(self)
        _stamp('start')

    def stamped_lap(self, phase, step=None, entry=None):
        lap(self, phase, step, entry)
        _stamp(f'lap {phase}/{step}', entry)

    def stamped_init(self):
        init(self)
        _stamp('engine-init')

    engine.Engine.start, engine.Engine.lap, engine.Engine.__init__ = stamped_start, stamped_lap, stamped_init

    import tests.test_phases as phases

    timed = phases._timed

    def probed(call, out, clock=time.perf_counter):
        _EVENTS.clear()

        def stamped_call():
            _stamp('call-begin')
            model = call()
            _stamp('call-end')
            return model

        result = timed(stamped_call, out, clock)
        _report(os.environ.get('PYTEST_CURRENT_TEST', '?'), result.wall - sum(result.kept.values()), list(_EVENTS))
        return result

    phases._timed = probed


def _report(test: str, gap: float, events: list) -> None:
    log = os.environ.get('GAPLOG', 'gaps.jsonl')
    with open(f'{log}.all', 'a') as every:
        every.write(f'{gap * 1e3:.3f} {test}\n')
    if gap <= _THRESHOLD:
        return
    marks = [e for e in events if not e[4].startswith('gc')]
    starts = [e for e in events if e[4] == 'gc-start']
    stops = [e for e in events if e[4] == 'gc-stop']
    spans = []
    for a, b in itertools.pairwise(marks):
        if b[4].startswith('lap'):
            continue
        spans.append(
            {
                'wall_ms': round((b[0] - a[0]) * 1e3, 3),
                'cpu_ms': round((b[1] - a[1]) * 1e3, 3),
                'voluntary': b[2] - a[2],
                'involuntary': b[3] - a[3],
                'from': a[4],
                'to': b[4],
                'gc_ms': [
                    (round((e[0] - s[0]) * 1e3, 3), s[5])
                    for s, e in zip(starts, stops, strict=False)
                    if a[0] <= s[0] < b[0]
                ],
            }
        )
    spans.sort(key=lambda s: -s['wall_ms'])
    record = {'test': test, 'pid': os.getpid(), 'gap_ms': round(gap * 1e3, 3), 'unclocked': spans[:4]}
    with open(log, 'a') as out:
        out.write(json.dumps(record) + '\n')
