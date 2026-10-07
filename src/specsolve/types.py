"""What specsolve hands back: the objects a call returns, the rows they hold, and the outputs and starts one can pass.

A caller never constructs one of these. A call returns it, and a caller names it
to annotate a function, to check one with ``isinstance``, or to read its
reference. [`Output`][] and [`Start`][] are the exceptions: they name what
``outputs=`` and ``start=`` take, so a caller that builds one writes its type.

Example::

    from specsolve.types import Result


    def total_cost(result: Result) -> float:
        return result.objective
"""

from specsolve.api import Model
from specsolve.archive import ResultArchive, SweepArchive
from specsolve.relational.answer_layout import Metrics, Output, Provenance, Record
from specsolve.relational.result import ConstraintRow, Diagnostics, Result, Start
from specsolve.sweep import Sweep

__all__ = [
    'ConstraintRow',
    'Diagnostics',
    'Metrics',
    'Model',
    'Output',
    'Provenance',
    'Record',
    'Result',
    'ResultArchive',
    'Start',
    'Sweep',
    'SweepArchive',
]
