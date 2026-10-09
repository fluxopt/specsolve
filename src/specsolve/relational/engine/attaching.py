"""What a caller's ``sources`` become: the frames the engine reads by name.

The door ([`tidy_sources`][specsolve.sources.tidy_sources]) has already read and checked
every source; this shapes each and encodes the string dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from specsolve.relational.collect import collected, in_memory

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mathspec import program


@dataclass(frozen=True)
class AttachedSources:
    """The data a program is built against, after attaching.

    ``parameters`` are tidy ``(dims…, value)``; ``dimensions`` are
    ``(val, ord)``; ``relations`` hold no row for a key that maps nowhere.
    ``cardinality`` and ``parameter_rows`` are cached frame heights.
    """

    parameters: Mapping[str, pl.LazyFrame]
    dimensions: Mapping[str, pl.LazyFrame]
    relations: Mapping[str, pl.LazyFrame]
    cardinality: Mapping[str, int]
    parameter_rows: Mapping[str, int]

    def is_enum_encoded(self, dim: str) -> bool:
        """Whether *dim* was given an ``Enum``."""
        return self.dimensions[dim].collect_schema()['val'] == pl.Enum


def attach(program: program.Program, sources: Mapping[str, pl.LazyFrame]) -> AttachedSources:
    """Shape the door's frames into what *program* is written against.

    A dimension's ``Enum`` is built from its labels, then every frame that
    carries the dimension is re-encoded against it. Each frame is read
    [`in_memory`][], since every read holds one table whole.
    """
    with in_memory():
        dimensions = {d: _ordinal_frame(d, sources[d]).pipe(collected) for d in program.dimensions}
        relations = {name: sources[name].pipe(collected) for name in program.relations}
        parameters = {name: sources[name].pipe(collected) for name in program.parameters}

    enums = {d: pl.Enum(f['val']) for d, f in dimensions.items() if f.schema['val'] == pl.String}
    for d, enum in enums.items():
        dimensions[d] = dimensions[d].with_columns(pl.col('val').cast(enum))
    for name, relation in program.relations.items():
        casts = [pl.col(role).cast(enums[dim]) for role, dim in relation.columns if dim in enums]
        if casts:
            relations[name] = relations[name].with_columns(casts)
    for name, p in program.parameters.items():
        frame = _plain_strings(parameters[name], p.dims)
        casts = [pl.col(d).cast(enums[d]) for d in p.dims if d in enums]
        parameters[name] = frame.with_columns(casts) if casts else frame

    return AttachedSources(
        parameters={name: f.lazy() for name, f in parameters.items()},
        dimensions={d: f.lazy() for d, f in dimensions.items()},
        relations={name: f.lazy() for name, f in relations.items()},
        cardinality={d: f.height for d, f in dimensions.items()},
        parameter_rows={name: f.height for name, f in parameters.items()},
    )


def _ordinal_frame(d: str, index: pl.LazyFrame) -> pl.LazyFrame:
    """A dimension's ``(val, ord)`` from its index, a label's ordinal being its row."""
    return index.with_row_index('ord').select(pl.col(d).alias('val'), pl.col('ord').cast(pl.Int64))


def _plain_strings(frame: pl.DataFrame, dims: tuple[str, ...]) -> pl.DataFrame:
    """Dim columns as plain strings, so a writer's own dictionary cannot block the cast into an ``Enum``."""
    categorical = [d for d, dtype in frame.schema.items() if d in dims and dtype in (pl.Categorical, pl.Enum)]
    if not categorical:
        return frame
    return frame.with_columns(pl.col(d).cast(pl.String) for d in categorical)
