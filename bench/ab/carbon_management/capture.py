"""Write PyPSA's `carbon_management` example as specsolve reads it: `spec.yaml`, `sources/`, `scalars.json`.

    uv run --python 3.12 --with "pypsa @ git+https://github.com/PyPSA/PyPSA@7f36f24f27ae84e67387dc770620ec6fc989baaf" \\
        python bench/ab/carbon_management/capture.py

The spec and sources are what PyPSA's `feat/mathspec-formulation` branch hands
`linopy.Model.from_spec`, intercepted there. `README.md` beside this file names
the source, its license and what is changed from it.
"""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path

import linopy
import numpy as np
import pandas as pd
import polars as pl
import pypsa
import yaml

HERE = Path(__file__).resolve().parent
UNBOUNDED = ('Generator_p_nom_max', 'Link_p_nom_max', 'StorageUnit_p_nom_max', 'Store_e_nom_max')


def captured() -> tuple[object, dict[str, object]]:
    """The spec and sources PyPSA builds `carbon_management` from."""
    got = {}

    def grab(spec, sources, **_):
        got['spec'], got['sources'] = spec, sources
        raise _CapturedError

    linopy.Model.from_spec = staticmethod(grab)
    pypsa.options.general.allow_network_requests = True
    with contextlib.suppress(_CapturedError):
        pypsa.examples.carbon_management().optimize.create_model()
    return got['spec'], got['sources']


class _CapturedError(Exception):
    """Raised from the intercepted `from_spec`, so PyPSA builds no linopy model."""


def with_unbounded_absent(text: str) -> str:
    """*text* with `missing: absent` under each parameter in `UNBOUNDED`, the rest of the file as PyPSA wrote it."""
    for name in UNBOUNDED:
        entry = re.search(rf'^(\s+){name}:\n', text, re.MULTILINE)
        text = f'{text[: entry.end()]}{entry.group(1)}  missing: absent\n{text[entry.end() :]}'
    return text


def table(name: str, value: object) -> pl.DataFrame:
    """The source *name* as a table, a DataArray's NaN cells dropped."""
    if isinstance(value, pd.Index):
        return pl.DataFrame({name: _labels(value)})
    if isinstance(value, pd.DataFrame):
        frame = value.reset_index(drop=True)
        out = pl.from_pandas(frame)
        return out.with_columns(pl.Series(c, _labels(frame[c])) for c in out.columns if out.schema[c] == pl.Object)
    grids = np.meshgrid(*(np.arange(len(value.coords[d])) for d in value.dims), indexing='ij')
    columns = [
        pl.Series(d, _labels(value.coords[d].values)).gather(g.ravel()) for d, g in zip(value.dims, grids, strict=True)
    ]
    out = pl.DataFrame([*columns, pl.Series('value', value.values.ravel())])
    return out.filter(pl.col('value').is_not_nan()) if out.schema['value'].is_float() else out


def _labels(values: object) -> np.ndarray | list:
    index = pd.Index(values)
    if index.dtype == object and len(index) and isinstance(index[0], pd.Timestamp):
        return pd.to_datetime(index).to_numpy()
    return index.tolist() if index.dtype == object else index.to_numpy()


def as_declared(frame: pl.DataFrame, dtypes: dict[str, str]) -> pl.DataFrame:
    """*frame* with every column over a dimension cast to that dimension's `dtype`."""
    casts = []
    for column, given in frame.schema.items():
        declared = dtypes.get(column)
        if declared is None:
            continue
        if declared == 'int' and given != pl.Int64:
            casts.append(pl.col(column).cast(pl.Int64))
        elif declared == 'str' and given.is_temporal():
            casts.append(pl.col(column).dt.strftime('%Y-%m-%d %H:%M:%S'))
        elif declared == 'str' and given != pl.String:
            casts.append(pl.col(column).cast(pl.String))
    return frame.with_columns(casts)


def main() -> None:
    spec, sources = captured()
    text = with_unbounded_absent(spec.to_yaml())
    (HERE / 'spec.yaml').write_text(text)
    declared = yaml.safe_load(text)['dimensions']
    dtypes = {name: (entry or {}).get('dtype', 'str') for name, entry in declared.items()}
    folder = HERE / 'sources'
    folder.mkdir(exist_ok=True)
    for stale in folder.glob('*.parquet'):
        stale.unlink()
    scalars = {}
    for name, value in sources.items():
        if isinstance(value, (bool, int, float, np.generic)):
            scalars[name] = value.item() if isinstance(value, np.generic) else value
            continue
        as_declared(table(name, value), dtypes).write_parquet(
            folder / f'{name}.parquet', compression='zstd', compression_level=22, statistics=False
        )
    (HERE / 'scalars.json').write_text(json.dumps(scalars) + '\n')
    print(f'{len(sources)} sources: {len(scalars)} scalars, {len(sources) - len(scalars)} tables')


if __name__ == '__main__':
    main()
