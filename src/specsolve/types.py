"""What specsolve hands back: the objects a call returns, and the rows they hold.

A caller never constructs one of these. A call returns it, and a caller names it
to annotate a function, to check one with ``isinstance``, or to read its
reference.

Example::

    from specsolve.types import Result


    def total_cost(result: Result) -> float:
        return result.objective
"""

from specsolve.api import Model
from specsolve.archive import ResultArchive, SweepArchive
from specsolve.relational.answer_layout import Metrics, Provenance, Record
from specsolve.relational.result import ConstraintRow, Diagnostics, Result
from specsolve.sweep import Sweep

__all__ = [
    'ConstraintRow',
    'Diagnostics',
    'Metrics',
    'Model',
    'Provenance',
    'Record',
    'Result',
    'ResultArchive',
    'Sweep',
    'SweepArchive',
]
