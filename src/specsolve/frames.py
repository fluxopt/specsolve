"""The one place that knows what a caller's table library is.

What a caller hands over is read through the Arrow PyCapsule protocol, so
neither pyarrow nor pandas is a dependency. A ``pandas.Series`` is unwrapped
first, and only when pandas is already imported. An ``xarray.DataArray`` is not
a table and is not read.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from specsolve.inputs import ArrowTable

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from specsolve.inputs import Source


__all__ = ['as_frame', 'is_dense_array', 'is_multi_indexed', 'to_pandas']


def to_pandas(table: pl.DataFrame) -> pd.DataFrame:
    """A polars frame as pandas, column by column, without pyarrow."""
    import pandas as pd

    encoded = [name for name, kind in table.schema.items() if kind in (pl.Categorical, pl.Enum)]
    if encoded:
        table = table.with_columns(pl.col(name).cast(pl.String) for name in encoded)
    return pd.DataFrame({name: table[name].to_numpy() for name in table.columns})


def as_frame(obj: Source, dims: Sequence[str] = ()) -> pl.LazyFrame | None:
    """One source as a lazy frame: a parquet path scanned, or an in-memory table normalised.

    The one place a string is read as a parquet path. *dims* names the columns
    a pandas index becomes.

    Returns:
        The frame, or ``None`` where *obj* is not table-shaped.
    """
    import sys

    if isinstance(obj, (str, Path)):
        return pl.scan_parquet(obj)
    if isinstance(obj, pl.LazyFrame):
        return obj
    if isinstance(obj, pl.DataFrame):
        return obj.lazy()

    if 'pandas' in sys.modules:
        import pandas as pd

        if isinstance(obj, pd.Series):
            frame = _series_to_frame(obj, dims)
            return _from_pandas(frame) if frame is not None else None
        if isinstance(obj, pd.DataFrame):
            return _from_pandas(obj)

    if isinstance(obj, ArrowTable):
        try:
            return pl.DataFrame(obj).lazy()
        except (TypeError, ValueError, pl.exceptions.PolarsError):
            return None
    return None


def is_dense_array(obj: object) -> bool:
    """Whether *obj* is an ``xarray.DataArray``, the one shape recognised and not read."""
    import sys

    xr = sys.modules.get('xarray')
    return xr is not None and isinstance(obj, xr.DataArray)


def is_multi_indexed(obj: Source) -> bool:
    """Whether *obj* is a pandas Series carrying more than one index level."""
    import sys

    if 'pandas' not in sys.modules:
        return False
    import pandas as pd

    return isinstance(obj, pd.Series) and obj.index.nlevels > 1


def _series_to_frame(series: pd.Series, dims: Sequence[str]) -> pd.DataFrame | None:
    """A pandas Series with its one index level promoted to a column.

    A named level keeps its name: renaming it to *dims* would transpose the
    data when two dims share a label space.

    Returns:
        The tidy frame, or ``None`` where the declaration is not one dimension.
    """
    if len(dims) != 1:
        return None
    if series.index.name is None:
        series = series.rename_axis(dims[0])
    return series.rename('value').reset_index()


#: The numpy datetime units polars reads; a coarser one is cast to microseconds without loss.
_POLARS_DATETIMES = frozenset({'datetime64[ms]', 'datetime64[us]', 'datetime64[ns]'})


def _from_pandas(frame: pd.DataFrame) -> pl.LazyFrame:
    """A pandas frame, column by column, without pyarrow.

    Object arrays go through a list so numpy's ``nan`` becomes a null rather
    than a string. A datetime in a unit polars does not read, such as the
    seconds ``pd.to_datetime`` gives a date, is cast to microseconds first.
    """
    columns: dict[str, object] = {}
    for name in frame.columns:
        values = frame[name].to_numpy()
        if values.dtype == object:
            columns[name] = pl.Series(name, [None if _is_missing(v) else v for v in values], strict=False)
        elif values.dtype.kind == 'M' and values.dtype.name not in _POLARS_DATETIMES:
            columns[name] = values.astype('datetime64[us]')
        else:
            columns[name] = values
    return pl.DataFrame(columns).lazy()


def _is_missing(value: object) -> bool:
    """Whether an object-array entry is pandas' rendering of "no value"."""
    return value is None or (isinstance(value, float) and value != value)
