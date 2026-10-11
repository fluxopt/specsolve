"""One first window in a fresh interpreter: what a process started for each solve pays.

    python -m bench.fresh <payload.pickle> <answer.json>

`test_fresh` pickles ``(arm, sink, prepared)`` before the clock and times this
process from launch to exit. Here the arm's own ``build_and_emit`` is timed
once, so the gap between the two is the interpreter, the harness and the
library's import. The answer is one JSON object, written to a file because a
library may print to stdout: the arm's counts, with ``first_window`` among its
phases, and ``peak_rss_bytes``, this process's high-water mark.

Nothing here imports a modelling library: the arm imports its own inside the
verb, so the import lands in ``first_window``.
"""

from __future__ import annotations

import importlib
import json
import pickle
import resource
import sys
import time
from pathlib import Path
from typing import Any


def run(payload: Path) -> dict[str, Any]:
    """Unpickle *payload*, time one ``build_and_emit``, and return its counts with the clock and the peak."""
    arm, sink, prepared = pickle.loads(payload.read_bytes())
    module = importlib.import_module(f'bench.arms.{arm.replace("-", "_")}')
    started = time.perf_counter()
    counts = module.build_and_emit(sink, prepared)
    seconds = time.perf_counter() - started
    return {
        **counts,
        'phases': {**(counts.get('phases') or {}), 'first_window': seconds},
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


if __name__ == '__main__':
    Path(sys.argv[2]).write_text(json.dumps(run(Path(sys.argv[1]))))
