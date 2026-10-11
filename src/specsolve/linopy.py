"""The ``linopy`` export: the built model as a ``linopy.Model``.

One variable and one constraint per declaration, each over every label of its
dims and masked to the coordinates the build produced, so a coordinate a
``where`` removed is masked out. A row holds the build's numbers: it is a flat
sum of terms, not the formula the file wrote.

linopy is imported when the export is called and never by the engine, which
is why this module sits outside ``relational/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Mapping

    import linopy
    import pandas as pd

    from specsolve.relational.sinks.handoff import Declared, Handoff, Run

__all__ = ['UNAVAILABLE', 'to_linopy']

#: What calling the export without the extra says.
UNAVAILABLE = 'to_linopy requires the [linopy] extra: pip install "specsolve[linopy]"'

#: linopy's spelling of each comparison.
_SIGNS = {'<=': '<=', '>=': '>=', '==': '='}


def to_linopy(handoff: Handoff, declared: Declared, dimensions: Mapping[str, pl.DataFrame]) -> linopy.Model:
    """The built model as a ``linopy.Model``.

    *dimensions* holds each dim's ``(val, ord)`` frame, its labels in ordinal
    order, which is the order the model's coordinates take.

    The caller has asked [`check`][specsolve.Model.check] for ``linopy``
    already, so nothing here is one linopy would drop.

    Raises:
        SpecsolveError: linopy is not installed.
    """
    try:
        import linopy
        import pandas as pd
    except ModuleNotFoundError as missing:
        raise SpecsolveError(UNAVAILABLE) from missing

    coords = {d: pd.Index(frame.get_column('val').to_list(), name=d) for d, frame in dimensions.items()}
    m = linopy.Model()
    columns = np.full(handoff.column_count, -1, dtype=np.int64)
    for run in declared.variables:
        _variable(m, handoff, run, coords, dimensions, columns)
    for run in declared.constraints:
        _constraint(m, handoff, run, coords, dimensions, columns)
    dims = {run.name: run.dims for run in declared.variables}
    for sets in declared.sets:
        along = dims[sets.variable][sets.along]
        m.add_sos_constraints(m.variables[sets.variable], sos_type=sets.sos_type, sos_dim=along)
    if handoff.objective_sense is not None:
        m.add_objective(
            _objective(m, handoff, columns), sense='max' if handoff.objective_sense == 'maximize' else 'min'
        )
    return m


def _placed(run: Run, dimensions: Mapping[str, pl.DataFrame]) -> np.ndarray:
    """Each entry's flat position in the dense array over the run's dims, row-major by ordinal."""
    if not run.dims:
        return np.zeros(run.height, dtype=np.int64)
    frame = run.coordinates
    for d in run.dims:
        frame = frame.join(dimensions[d].rename({'val': d, 'ord': f'#{d}'}), on=d, how='left', maintain_order='left')
    flat = pl.lit(0, dtype=pl.Int64)
    for d in run.dims:
        flat = flat * dimensions[d].height + pl.col(f'#{d}')
    return frame.select(flat.alias('#flat')).get_column('#flat').to_numpy()


def _shape(run: Run, dimensions: Mapping[str, pl.DataFrame]) -> tuple[int, ...]:
    return tuple(dimensions[d].height for d in run.dims)


def _variable(
    m: linopy.Model,
    handoff: Handoff,
    run: Run,
    coords: Mapping[str, pd.Index],
    dimensions: Mapping[str, pl.DataFrame],
    columns: np.ndarray,
) -> None:
    """Add one variable over its dims' labels, masked to its columns, and record each column's linopy label.

    linopy refuses a ``NaN`` constant and a binary bound other than 0 or 1
    even where the mask hides it, so a masked entry carries the bounds 0 and 1.
    """
    import xarray as xr

    shape, flat = _shape(run, dimensions), _placed(run, dimensions)
    owned = handoff.cols.slice(run.start, run.height)
    lower, upper, mask = np.zeros(shape), np.ones(shape), np.zeros(shape, dtype=bool)
    lower.flat[flat], upper.flat[flat], mask.flat[flat] = owned['lb'].to_numpy(), owned['ub'].to_numpy(), True
    on = {d: coords[d] for d in run.dims}
    vtype = owned.item(0, 'vtype') if run.height else 'continuous'
    variable = m.add_variables(
        lower=xr.DataArray(lower, coords=on),
        upper=xr.DataArray(upper, coords=on),
        coords=list(on.values()) or None,
        mask=xr.DataArray(mask, coords=on),
        name=run.name,
        binary=vtype == 'binary',
        integer=vtype == 'integer',
    )
    columns[run.start : run.start + run.height] = variable.labels.values.flat[flat]


def _constraint(
    m: linopy.Model,
    handoff: Handoff,
    run: Run,
    coords: Mapping[str, pd.Index],
    dimensions: Mapping[str, pl.DataFrame],
    columns: np.ndarray,
) -> None:
    """Add one constraint, each row's terms along ``_term``, padded with linopy's absent variable.

    A family with no term at all is left out, since linopy refuses a
    constraint with no variable; the check has already refused any of its rows
    no point meets.
    """
    import xarray as xr
    from linopy.expressions import LinearExpression

    starts = handoff.row_starts[run.start : run.start + run.height + 1]
    counts = np.diff(starts)
    width = int(counts.max()) if counts.size else 0
    if not width:
        return
    shape, flat = _shape(run, dimensions), _placed(run, dimensions)
    size = int(np.prod(shape, dtype=np.int64))
    span = handoff.matrix.slice(int(starts[0]), int(starts[-1] - starts[0]))
    owner = np.repeat(flat, counts)
    within = np.arange(span.height) - np.repeat(starts[:-1] - starts[0], counts)
    variables, coefficients = np.full((size, width), -1, dtype=np.int64), np.zeros((size, width))
    variables[owner, within] = columns[span['col'].to_numpy()]
    coefficients[owner, within] = span['coeff'].to_numpy()
    rows = handoff.rows.slice(run.start, run.height)
    rhs, mask = np.zeros(size), np.zeros(size, dtype=bool)
    rhs[flat], mask[flat] = rows['rhs'].to_numpy(), True

    on = {d: coords[d] for d in run.dims}
    dims = [*run.dims, '_term']
    expression = LinearExpression(
        xr.Dataset(
            {
                'coeffs': xr.DataArray(coefficients.reshape((*shape, width)), coords=on, dims=dims),
                'vars': xr.DataArray(variables.reshape((*shape, width)), coords=on, dims=dims),
            }
        ),
        m,
    )
    m.add_constraints(
        expression,
        _SIGNS[str(rows.item(0, 'sense'))],
        xr.DataArray(rhs.reshape(shape), coords=on, dims=list(run.dims)),
        name=run.name,
        mask=xr.DataArray(mask.reshape(shape), coords=on, dims=list(run.dims)),
    )


def _objective(
    m: linopy.Model, handoff: Handoff, columns: np.ndarray
) -> linopy.LinearExpression | linopy.QuadraticExpression:
    """The objective's terms and its quadratic pairs, each pair whole."""
    import xarray as xr
    from linopy.expressions import LinearExpression, QuadraticExpression

    linear = LinearExpression(
        xr.Dataset(
            {
                'coeffs': xr.DataArray(handoff.obj['coeff'].to_numpy(), dims=['_term']),
                'vars': xr.DataArray(columns[handoff.obj['col'].to_numpy()], dims=['_term']),
            }
        ),
        m,
    )
    if not handoff.quad.height:
        return linear
    pairs = np.stack([columns[handoff.quad['col_l'].to_numpy()], columns[handoff.quad['col_r'].to_numpy()]], axis=1)
    quadratic = QuadraticExpression(
        xr.Dataset(
            {
                'coeffs': xr.DataArray(handoff.quad['coeff'].to_numpy(), dims=['_term']),
                'vars': xr.DataArray(pairs, dims=['_term', '_factor']),
                'const': 0.0,
            }
        ),
        m,
    )
    return linear + quadratic
