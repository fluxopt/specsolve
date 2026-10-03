"""Declarative optimisation: YAML math on a streaming engine.

Specs build relationally on polars and stream to the solver — see
docs/about/architecture.md.

Example::

    import specsolve as sps

    result = sps.solve('spec.yaml', {'p_max': 'p_max.parquet', 'load': 'load.parquet'})
    result.objective
    result.primal('p')  # tidy polars.DataFrame
    result.to_dataarray('p')  # labelled, for array post-processing

``__version__`` reads the installed metadata; a source tree with nothing
installed reads ``0.0.0``.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _installed_version

from specsolve.api import Model, build, check, evaluate, load_result, scan_result, solve, tidy, write
from specsolve.archive import ResultArchive, SweepArchive, load_archive, scan_archive
from specsolve.axes import EachCoordinate, EachWindow
from specsolve.errors import (
    DataError,
    DimensionError,
    LanguageError,
    LayoutError,
    NoSolutionError,
    SchemaError,
    SpecsolveError,
    SpecsolveWarning,
)
from specsolve.relational.result import Result
from specsolve.strategy import solve_over
from specsolve.sweep import Sweep, load_sweep, scan_sweep

__all__ = [
    'DataError',
    'DimensionError',
    'EachCoordinate',
    'EachWindow',
    'LanguageError',
    'LayoutError',
    'Model',
    'NoSolutionError',
    'Result',
    'ResultArchive',
    'SchemaError',
    'SpecsolveError',
    'SpecsolveWarning',
    'Sweep',
    'SweepArchive',
    'build',
    'check',
    'evaluate',
    'load_archive',
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
