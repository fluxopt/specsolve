"""Declarative optimisation: YAML math on a relational engine.

Specs build relationally on polars and stream to the solver — see
docs/development/architecture.md.

Example::

    import specsolve as sps

    result = sps.solve('spec.yaml', {'p_max': 'p_max.parquet', 'load': 'load.parquet'})
    result.objective
    result.primal('p')  # tidy polars.DataFrame
    result.to_dataarray('p')  # labelled, for array post-processing

What a call hands back is in ``specsolve.types``, and what it raises in
``specsolve.errors``. Nothing else under ``specsolve.`` is public.

``__version__`` reads the installed metadata; a source tree with nothing
installed reads ``0.0.0``.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _installed_version

from specsolve import errors as errors
from specsolve import types as types
from specsolve._api import build, check, evaluate, solve, tidy, write
from specsolve._archive import read_archive, read_inputs, scan_archive
from specsolve._sweep.axes import EachLabel, EachWindow
from specsolve._sweep.solve_over import build_slice, solve_over

__all__ = [
    'EachLabel',
    'EachWindow',
    'build',
    'build_slice',
    'check',
    'evaluate',
    'read_archive',
    'read_inputs',
    'scan_archive',
    'solve',
    'solve_over',
    'tidy',
    'write',
]

try:
    __version__ = _installed_version('specsolve')
except _PackageNotFoundError:
    __version__ = '0.0.0'
