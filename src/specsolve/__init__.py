"""Declarative optimisation: YAML math on a streaming engine.

Specs build relationally on polars and stream to the solver — see
docs/about/architecture.md.

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
from specsolve.api import build, check, evaluate, load_result, scan_result, solve, tidy, write
from specsolve.archive import load_archive, scan_archive
from specsolve.axes import EachCoordinate, EachWindow
from specsolve.manifest import load_manifest
from specsolve.strategy import solve_over
from specsolve.sweep import load_sweep, scan_sweep

__all__ = [
    'EachCoordinate',
    'EachWindow',
    'build',
    'check',
    'evaluate',
    'load_archive',
    'load_manifest',
    'load_result',
    'load_sweep',
    'scan_archive',
    'scan_result',
    'scan_sweep',
    'solve',
    'solve_over',
    'tidy',
    'write',
]

try:
    __version__ = _installed_version('specsolve')
except _PackageNotFoundError:
    __version__ = '0.0.0'
