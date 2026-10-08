"""The messages two modules raise, written once.

A message raised from one place stays beside its ``raise``. One raised from two
lives here, so the two cannot drift. The engine may import this module: it is a
leaf that knows no YAML, schema or AST.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from mathspec import did_you_mean

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

#: How a datetime label prints: without the ``.000000`` that polars casts a
#: whole second to.
_DATETIME = '%Y-%m-%d %H:%M:%S%.f'


def coordinate_expr(schema: Mapping[str, pl.DataType]) -> pl.Expr:
    """``t=1, g=gas`` per row of a frame holding the columns *schema* names, in its order.

    The one way specsolve writes a coordinate, in a printed row and in a
    refusal alike. A missing label writes ``null``, and an empty *schema*
    writes ``''``.
    """
    parts: list[pl.Expr] = []
    for at, (dim, dtype) in enumerate(schema.items()):
        label = pl.col(dim)
        if isinstance(dtype, pl.Datetime):
            label = label.dt.to_string(_DATETIME + ('%:z' if dtype.time_zone else ''))
        parts += [pl.lit(f'{", " if at else ""}{dim}='), label.cast(pl.String).fill_null('null')]
    return pl.concat_str(parts) if parts else pl.lit('')


def coordinate_text(coordinate: Mapping[str, object]) -> str:
    """One coordinate, ``dim: label`` in dim order, as [`coordinate_expr`][] writes it."""
    if not coordinate:
        return ''
    frame = pl.DataFrame([dict(coordinate)])
    return frame.select(coordinate_expr(frame.schema)).item()


def coordinates_text(dims: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Coordinates as a refusal lists them: ``f=b; f=c``."""
    return '; '.join(coordinate_text(dict(zip(dims, row, strict=True))) for row in rows)


def uncovered_constant_message(names: str, missing: int, subject: str) -> str:
    """The message for a constant side covering fewer coordinates than its rows."""
    return (
        f"{subject}: parameter '{names}' covers {missing} fewer coordinates than the rows "
        f'built here. A missing row is read as 0, and on the constant side that zero is a '
        f'bound rather than an absence — the row still exists, and it binds.\n'
        f'  Supply the missing rows, if the value is what was meant.\n'
        f'  Mask them out with a where, if the row should not exist there.\n'
        f'  Drop the declaration, if the spec has no such quantity at all.'
    )


def sparse_divisor_message(name: str, missing: int) -> str:
    """The message for a divisor parameter that is sparse over its index."""
    return (
        f"parameter '{name}' is used as a divisor but covers {missing} fewer "
        f'coordinates than it is indexed over. A missing row means a zero '
        f'coefficient everywhere else, and zero is not a divisor: the term '
        f'would drop and the constraint would silently stop constraining.\n'
        f'  Supply the missing rows, or mask the coordinates out with a where.'
    )


def zero_divisor_dims(dims_of: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """The dims a zero divisor's coordinate is named over: each divisor's own in *dims_of*, in name order, once each.

    Both lanes name the same coordinate by this order.
    """
    return tuple(dict.fromkeys(d for name in sorted(dims_of) for d in dims_of[name]))


def zero_divisor_message(name: str, zeros: int, at: str) -> str:
    """The message for a divisor parameter that is zero where the model divides by it, *at* one such coordinate."""
    such_as = f', such as {at}' if at else ''
    return (
        f"parameter '{name}' is used as a divisor and is zero at {zeros} of the "
        f'coordinates the model divides at{such_as}. A quotient by zero has no value, so '
        f'there is no coefficient to build there.\n'
        f'  Supply a non-zero value, or mask the coordinates out with a where '
        f'that admits only a non-zero divisor.'
    )


def null_bounds_message(name: str, rows: int) -> str:
    """The message for a bound parameter missing values at some coordinates."""
    return (
        f"variable '{name}': {rows} rows have NULL bounds — a bound parameter is missing "
        f'values for some coordinates. The two ways out build different models, so the '
        f'language will not pick one:\n'
        f'  supply the value           the variable exists there, bounded (`inf` is a value)\n'
        f'  where: "<the parameter>"   the variable does not exist there at all'
    )


def no_model_behind_this_answer_message() -> str:
    """An expression to read in an answer that has no model behind it."""
    return (
        'this answer has no model behind it, so a quantity the file never named cannot be read from '
        'it: an answer read back off disk carries the values without the model to splice the '
        'expression into. Re-ask with sps.solve(archive.spec, archive.sources), which reads any '
        'expression; a name the spec declares is readable either way.'
    )


def position_out_of_range_message(name: str, op: str, position: int, at: int, cardinality: int) -> str:
    """A ``position(dim)`` boundary naming no coordinate of the dimension."""
    return (
        f'where: position({name}) {op} {position} names position {at} of '
        f"'{name}', which has {cardinality} coordinate(s). A boundary that "
        f'names no coordinate leaves the rows it was to seed unseeded.'
    )


def short_groups_message(name: str, by: str, op: str, position: int, short: Sequence[str]) -> str:
    """A grouped ``position(dim, by=)`` boundary that some group is too short to reach."""
    return (
        f'where: position({name}, by={by}) {op} {position} names position '
        f'{position} within each group, and {len(short)} of them are shorter than '
        f'that: {list(short[:5])}. A boundary that names no coordinate leaves the rows it '
        f'was to seed unseeded.'
    )


def unknown_name_message(kind: str, name: str, known: Iterable[str]) -> str:
    """``unknown <kind> '<name>'``, plus the near miss, or every declaration the name prefixes.

    Single-line: it is raised as ``KeyError``, whose ``str`` reprs a newline.
    """
    candidates = sorted(known)

    family = [c for c in candidates if c.startswith(f'{name}_')]
    if family:
        return (
            f"unknown {kind} '{name}': no declaration has that name, but "
            f'{len(family)} begin with it — {", ".join(family)}.'
        )

    return f"unknown {kind} '{name}'. {did_you_mean(name, candidates)}"
