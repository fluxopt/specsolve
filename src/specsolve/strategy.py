"""Solving strategies: one plan per slice, folded.

A plan cannot contain a loop; a process may loop over plans. A strategy is a
driver above [`specsolve.api`][], built from the public verbs.

    scenario / sweep    ``EachCoordinate('scenario')``            independent
    myopic pathway      ``EachCoordinate('period')``              + ``carry``
    rolling horizon     ``EachWindow('snapshot', steps=24, lookahead=24, into='t')``  + ``carry``

A partition filters the sources, rows and index together: the containment
check refuses parameter rows outside a narrowed index.

The caller-facing rules are [sweeps](https://specsolve.readthedocs.io/en/latest/reference/sweeps/).
"""

from __future__ import annotations

import io
import json
import shutil
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import polars as pl
from mathspec import did_you_mean
from mathspec.program import parameters_of

from specsolve.api import build, check
from specsolve.archive_layout import ANSWER_DIR, beside, check_the_target, write_archive
from specsolve.errors import (
    DataError,
    LayoutError,
    SpecsolveError,
    SpecsolveWarning,
    no_model_behind_this_answer_message,
)
from specsolve.frames import as_frame
from specsolve.lanes import declared, lower
from specsolve.relational.answer_layout import (
    KINDS,
    METRICS_FILE,
    METRICS_SCHEMA,
    RECORD_FILE,
    RECORD_SCHEMA,
    RUN,
    Metrics,
    Record,
    check_format,
    checked_kind,
    consolidated,
    read_reasons,
    refuse_reserved,
    row_of,
    saved_frames,
    write_format,
    write_reasons,
    write_whole,
)
from specsolve.relational.result import tidy_to_dataarray, tidy_to_dataset, tidy_to_pandas
from specsolve.sources import least_value, numbered, tidy_sources

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence

    import pandas as pd
    import xarray as xr
    from mathspec import Spec
    from mathspec.program import Expression, Program

    from specsolve.api import Model
    from specsolve.lanes import Buildable, Label, Source
    from specsolve.relational.result import Diagnostics, Keep, Result

#: A frame lazy or not, going in and coming back out the same way.
_Frame = TypeVar('_Frame', pl.DataFrame, pl.LazyFrame)

#: What each of the [`KINDS`][specsolve.relational.answer_layout.KINDS] is a frame of, as a message names it.
_LABELS = {'primal': 'variable', 'dual': 'constraint', 'expression': 'named expression'}

#: The codec a frame is written with to cross a process.
_COMPRESSION = 'zstd'


class _Slice(NamedTuple):
    """One slice of a sweep: the key, the sources that build it, and what it owns.

    ``owns`` counts the coordinates of the re-indexed dimension this slice
    keeps, the rest being lookahead; a ``carry`` reads the seam off it. ``None``
    where the axis re-indexed nothing.
    """

    key: Label
    sources: Mapping[str, Source]
    owns: int | None = None


#: A spilled sweep's manifest, its keys as their own type, and the coordinates
#: each window owns.
_MANIFEST_FILE = 'sweep.json'
_KEYS_FILE = 'keys.parquet'
_OWNED_FILE = 'owned.parquet'


def _slice_metrics(after: Diagnostics, before: Diagnostics | None) -> Metrics:
    """One slice's row of [`Sweep.metrics`][], off the model's cumulative counters.

    A serial fold's model keeps summing across slices, so this slice's share is
    *after* minus *before*; a model built for one slice has no *before*.
    """
    now = after.metrics()
    return now if before is None else now.since(before.metrics())


@dataclass(frozen=True)
class _CarryRule:
    """One resolved carry: which variable moves into a parameter, and how.

    ``dropped`` is the one dimension the carry collapses, and ``None`` where the
    whole frame moves forward.
    """

    variable: str
    dropped: str | None

    @classmethod
    def resolved(cls, program: Program, parameter: str, variable: str) -> _CarryRule:
        """One carry checked against the plan, without reading data."""
        if parameter not in program.parameters:
            raise SpecsolveError(f'carry writes parameter {parameter!r}, which the spec does not declare')
        if variable not in program.variables:
            raise SpecsolveError(f'carry reads variable {variable!r}, which the spec does not declare')
        over = list(program.parameters[parameter].dims)
        source = list(program.variables[variable].dims)
        if missing := [d for d in over if d not in source]:
            raise SpecsolveError(
                f'carry {parameter!r} <- {variable!r} cannot line up: {parameter!r} is over {over}, and '
                f'{variable!r} is over {source}, which has no {missing}. A carry copies a variable into a '
                f'parameter, so the parameter cannot be over more than the variable is.'
            )
        dropped = [d for d in source if d not in over]
        if len(dropped) > 1:
            raise SpecsolveError(
                f'carry {parameter!r} <- {variable!r} would collapse {dropped} at once: {variable!r} is over '
                f'{source} and {parameter!r} over {over}. A carry collapses the one dimension the sweep '
                f'advances along, so reduce the others in the YAML — a derived variable is where the oracle '
                f'can see the math.'
            )
        return cls(variable, dropped[0] if dropped else None)

    def value_from(
        self, frames: Mapping[str, pl.DataFrame], parameter: str, key: Label, owns: int | None
    ) -> pl.DataFrame:
        """What this rule hands the next slice, read out of one slice's primals.

        The coordinate handed on is ``owns - 1``, the last one the slice keeps,
        never a lookahead row.
        """
        frame = frames[self.variable]
        if self.dropped is None:
            return frame
        assert owns is not None, 'a carry that drops a dimension is refused unless the axis owns it'
        seam = owns - 1
        picked = frame.filter(pl.col(self.dropped) == seam).drop(self.dropped)
        if picked.is_empty():
            raise SpecsolveError(
                f'carry {parameter!r} <- {self.variable!r} has nothing to copy: slice {key!r} built no '
                f'{self.variable!r} at {self.dropped} == {seam}, the last coordinate it owns, so there is no '
                f'value there to hand forward.'
            )
        return picked


@dataclass(frozen=True)
class _Answer:
    """One slice, solved and read out; plain data only, so it can cross a process."""

    meta: Record
    #: This slice's row of [`Sweep.metrics`][].
    metrics: Metrics
    #: ``{kind: {name: frame}}`` over [`KINDS`][specsolve.relational.answer_layout.KINDS]:
    #: every variable, every constraint's dual and every declared named
    #: expression, evaluated at this slice's solution. A kind the slice
    #: produced nothing of is absent.
    frames: dict[str, dict[str, pl.DataFrame]] = field(default_factory=dict)
    #: Why this slice has no duals, when it has none.
    no_duals: str | None = None
    #: Per expression, why this slice could not evaluate it.
    no_expressions: dict[str, str] = field(default_factory=dict)

    def sliced(self, key_name: str, key: Label) -> _Answer:
        """This answer with its record and metrics rows naming the slice they are."""
        text = str(key)
        return replace(
            self,
            meta=self.meta._replace(slice_axis=key_name, slice=text),
            metrics=self.metrics._replace(slice_axis=key_name, slice=text),
        )


@dataclass(frozen=True)
class _Stitch:
    """The way from a windowed sweep's frames to its answer over the dimension it sliced.

    ``owned`` is ``(key, local, dim)`` for the coordinates each window owns;
    the lookahead rows are not in it.
    """

    local: str
    dim: str
    owned: pl.DataFrame

    def unstitchable(self, frame: pl.DataFrame | pl.LazyFrame) -> str | None:
        """Why *frame* has no answer over [`dim`][], or ``None`` where it has one.

        A frame with no [`local`][] column is over no coordinate a window
        owns, so no row of it belongs to one coordinate of the sliced
        dimension.
        """
        if self.local in frame.collect_schema().names():
            return None
        return (
            f'this has no answer over {self.dim!r}: the frame has no {self.local!r} column, because the '
            f'quantity is not over the windowed dimension — each row covers a whole window, lookahead '
            f'included under an overlapping window. Read it with per_window=True for the value of each '
            f'window, or read a quantity that keeps {self.local!r} and aggregate its answer.'
        )

    def restore(self, frame: _Frame, key_name: str) -> _Frame:
        """*frame* over the dimension the axis sliced, sorted on it; lazy in, lazy out.

        The inner join on ``owned`` drops the lookahead rows.

        Raises:
            SpecsolveError: *frame* is [`unstitchable`][].
        """
        if why := self.unstitchable(frame):
            raise SpecsolveError(why)
        columns = frame.collect_schema().names()
        keys = [key_name, self.local]
        rest = [column for column in columns if column not in (*keys, 'value')]
        restored = frame.lazy().join(self.owned.lazy(), on=keys, how='inner').drop(keys)
        stitched = restored.select(self.dim, *rest, 'value').sort(self.dim, *rest)
        return stitched if isinstance(frame, pl.LazyFrame) else stitched.collect()  # pyrefly: ignore[bad-return]  — the branch matches the frame's own kind


def _one_key_type(keys: Sequence[Label], key_name: str) -> pl.DataType:
    """The type every file writes *key_name* as; keys of mixed types are refused, never coerced."""
    try:
        typed = pl.Series(keys)
    except TypeError as mixed:
        kinds = sorted({type(key).__name__ for key in keys})
        raise SpecsolveError(
            f'the keys of this sweep are of more than one type ({", ".join(kinds)}), so its files could not '
            f'all write {key_name!r} as one. Every file carries the key, and a column that changes type '
            f'between them cannot be concatenated or loaded into one table. Key the slices consistently.'
        ) from mixed
    _one_slice_per_text(keys, typed.to_list())
    return typed.dtype


def _one_slice_per_text(keys: Sequence[Label], typed: Sequence[Label]) -> None:
    """Refuse keys whose text, which the record and metrics name a slice by, does not find one slice.

    ``_rekeyed`` matches each row's ``slice`` text against the text of the
    keys as the sweep's one type holds them, when the fold ends and when a
    spill is scanned. A key that type rewrites, or two keys of one text, would
    fail there, after every slice has solved.
    """
    for given, held in zip(keys, typed, strict=True):
        if str(given) != str(held):
            raise SpecsolveError(
                f'the key {str(given)!r} is written as {str(held)!r} once every key of this sweep shares one '
                f'type, and the record names a slice by its key as text. Key the slices consistently.'
            )
    texts = Counter(str(key) for key in keys)
    repeated = [text for text, count in texts.items() if count > 1]
    if repeated:
        raise SpecsolveError(
            f'the key {repeated[0]!r} names more than one slice of this sweep, and the record names a slice '
            f'by its key as text, so those slices could not be told apart. Give each slice its own key.'
        )


def _keyed(frame: pl.DataFrame, key_name: str, key: Label, dtype: pl.DataType) -> pl.DataFrame:
    """*frame* with the slice key prepended as *dtype*, the whole sweep's type rather than ``pl.lit``'s."""
    return frame.select(pl.lit(key, dtype=dtype).alias(key_name), pl.all())


@dataclass(frozen=True)
class _Spill:
    """A sweep's answers on disk instead of in memory, one file per slice and name.

    ``<kind>/<name>/<position>.parquet`` holds the frames, keyed, and
    ``record/`` and ``metrics/`` the rows, which name their slice in
    ``slice_axis`` and ``slice`` rather than in a column of the key's own
    name and type. Every file lands whole, and the record file is written
    last: it marks a slice done. ``sweep.json`` names the key and the keys, so
    a directory answers for one sweep, and ``keys.parquet`` holds the keys as
    their own type, which the rows do not. ``sweep.json`` lands after the
    files a scan reads beside it: it marks the directory stamped.
    """

    directory: Path
    key_name: str
    #: The key column's type, settled over the sweep's keys, never per file.
    key_dtype: pl.DataType

    @classmethod
    def opened(
        cls,
        directory: str | Path,
        key_name: str,
        keys: Sequence[Label],
        key_dtype: pl.DataType,
        stitch: _Stitch | None,
    ) -> _Spill:
        """The directory ready to take this sweep: stamped if it holds none, checked and never re-stamped if it does."""
        directory = Path(directory)
        manifest: dict[str, object] = {
            'key_name': key_name,
            'keys': [str(key) for key in keys],
            'stitch': None if stitch is None else {'local': stitch.local, 'dim': stitch.dim},
        }
        record = directory / _MANIFEST_FILE
        if record.exists():
            check_format(directory)
            found = json.loads(record.read_text())
            if found != manifest:
                raise SpecsolveError(
                    f'{str(directory)!r} holds a sweep keyed by {found["key_name"]!r} over {found["keys"]}, and '
                    f'this one is keyed by {key_name!r} over {manifest["keys"]}. A directory holds one sweep: '
                    f'point spill_to= at an empty one, or delete this one to solve it again.'
                )
        else:
            write_format(directory)
            write_whole(pl.DataFrame([pl.Series(key_name, keys, dtype=key_dtype)]), directory / _KEYS_FILE)
            if stitch is not None:
                write_whole(stitch.owned, directory / _OWNED_FILE)
            record.write_text(json.dumps(manifest))
        return cls(directory, key_name, key_dtype)

    def _file(self, kind: str, position: int, name: str | None = None) -> Path:
        under = self.directory / kind if name is None else self.directory / kind / name
        return under / f'{position:06d}.parquet'

    def done(self, position: int) -> bool:
        return self._file('record', position).exists()

    def write(self, position: int, key: Label, answer: _Answer) -> _Answer:
        """*answer*'s frames and record on disk, and the answer with the frames released."""
        answer = answer.sliced(self.key_name, key)
        for kind, produced in answer.frames.items():
            for name, frame in produced.items():
                write_whole(_keyed(frame, self.key_name, key, self.key_dtype), self._file(kind, position, name))
        write_whole(pl.DataFrame([answer.metrics._asdict()], schema=METRICS_SCHEMA), self._file('metrics', position))
        write_whole(pl.DataFrame([answer.meta._asdict()], schema=RECORD_SCHEMA), self._file('record', position))
        return replace(answer, frames={})

    def read_back(self, position: int) -> _Answer:
        """A done slice's record, with no frames."""
        row = pl.read_parquet(self._file('record', position)).row(0, named=True)
        held = pl.read_parquet(self._file('metrics', position)).row(0, named=True)
        return _Answer(row_of(Record, row, self.directory), row_of(Metrics, held, self.directory))

    def primals(self, position: int, names: Iterable[str]) -> dict[str, pl.DataFrame]:
        """The named primals a done slice wrote; a name it did not write is absent."""
        found = {name: self._file('primal', position, name) for name in names}
        return {name: pl.read_parquet(path).drop(self.key_name) for name, path in found.items() if path.exists()}

    def held(self, kind: str) -> list[str]:
        under = self.directory / kind
        return sorted(path.name for path in under.iterdir()) if under.is_dir() else []

    def scan(self, kind: str, name: str) -> pl.LazyFrame | None:
        """Every slice's frame of *name*, lazily and in slice order, or ``None`` where no slice wrote one."""
        under = self.directory / kind / name
        return pl.scan_parquet(sorted(under.glob('*.parquet'))).drop(RUN, strict=False) if under.is_dir() else None

    def whole(self, kind: str, name: str) -> list[pl.DataFrame]:
        """The same frames read into memory, one per slice that wrote one, each keeping its key column."""
        under = self.directory / kind / name
        return [pl.read_parquet(file).drop(RUN, strict=False) for file in sorted(under.glob('*.parquet'))]


def _listed(entries: Mapping[str, str]) -> str:
    return '\n'.join(f'  {label}: {reason}' for label, reason in entries.items())


# ---------------------------------------------------------------------------
# axes — how the sources are sliced
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EachCoordinate:
    """One slice per coordinate of *dim* — a column the sources carry.

    Scenarios, draws, investment periods. Parameters and relations carrying
    *dim* are filtered to one coordinate and the column dropped, so the model
    never mentions it — a *dim* the spec declares is refused, and so is an
    index that carries it; every other source passes through untouched. The
    slices run in the coordinates' sorted order, which is the order a
    ``carry`` chains them in.
    """

    dim: str

    def slices(self, sources: Mapping[str, Source]) -> list[tuple[Label, Mapping[str, Source]]]:
        """The ``(key, sources)`` list this axis would run — what ``axis=`` takes hand-built.

        For building one slice alone: ``sps.build(spec, axis.slices(sources)[3][1])``.
        """
        return [(current.key, current.sources) for current in self._slice(sources, self._key_name())[0]]

    def _key_name(self) -> str:
        return self.dim

    def _check_the_program(self, program: Program, sources: Mapping[str, Source]) -> None:
        """Refuse a *dim* the spec declares: every source drops it, so nothing would supply it."""
        del sources
        if self.dim in program.dimensions:
            raise SpecsolveError(
                f'EachCoordinate({self.dim!r}) drops {self.dim!r} from every source, and the spec declares '
                f'it, so each slice would build a dimension nothing supplies. A coordinate sweep slices an '
                f'axis the model does not have; to slice one it does, window it.'
            )

    def _slice(self, sources: Mapping[str, Source], key_name: str) -> tuple[list[_Slice], _Stitch | None]:
        """One slice per coordinate, keyed by it, and no [`_Stitch`][]: the slices are the answer."""
        del key_name
        carrying, coordinates = _coordinates(sources, self.dim, 'slice')
        out: list[_Slice] = []
        for key in coordinates:
            filtered = {name: table.filter(pl.col(self.dim) == key).drop(self.dim) for name, table in carrying.items()}
            out.append(_Slice(key, {**sources, **filtered}))
        return out, None


@dataclass(frozen=True)
class EachWindow:
    """One slice per window of consecutive coordinates of *dim*.

    ``steps`` is what each window keeps and ``lookahead`` is what it sees beyond
    that, so a window is ``steps + lookahead`` coordinates long and a
    ``lookahead`` above zero is overlap. An ``int`` keeps the same number every
    window; a sequence keeps those numbers in order, which is a telescoping
    horizon or a month at a time. Both count coordinates rather than coordinate
    values, so *dim* need only be orderable — datetimes, strings and gapped
    integers all work. The dimension is re-indexed rather than dropped, into a
    dense ``0..n-1`` column the model addresses by the name ``into`` gives it,
    which the spec has to declare.

    Whether the model *can* be cut this way is asked before a slice is taken
    ([separability](https://mathspec.readthedocs.io/en/latest/reference/reading/#asking-whether-an-axis-can-be-cut)):
    a coupling along ``into`` is refused, naming the declaration and the
    change that would lift it; ``lookahead`` has to cover what the rows read
    ahead; and a ``position()`` the model counts warns, since every window
    restarts it. What the rows read *behind* is the rolling-horizon seed, met
    by the edge policy, and is not refused.
    """

    dim: str
    steps: int | Sequence[int] = field(kw_only=True)
    lookahead: int = field(kw_only=True)
    into: str = field(kw_only=True)

    def slices(self, sources: Mapping[str, Source]) -> list[tuple[Label, Mapping[str, Source]]]:
        """The ``(key, sources)`` list this axis would run — what ``axis=`` takes hand-built.

        For building one window alone: ``sps.build(spec, axis.slices(sources)[37][1])``.
        The pairs carry no ownership: solved as a list, the slices need
        ``key_name=``, the answer is keyed by slice rather than stitched, and a
        ``carry`` cannot collapse a dimension.
        """
        return [(current.key, current.sources) for current in self._slice(sources, self._key_name())[0]]

    def __post_init__(self) -> None:
        blocks = [self.steps] if isinstance(self.steps, int) else list(self.steps)
        if not blocks:
            raise ValueError('steps is empty, so no window would keep anything — pass an int, or one size per window')
        if short := [block for block in blocks if block < 1]:
            raise ValueError(f'every window must keep at least one coordinate (got steps with {short})')
        if self.lookahead < 0:
            raise ValueError(
                f'lookahead={self.lookahead} is negative; zero is contiguous windows, and above it overlaps'
            )
        if not isinstance(self.steps, int):
            object.__setattr__(self, 'steps', tuple(blocks))
        if not self.into:
            raise ValueError('into must name the local index the spec declares — it has no default')
        if self.into == self.dim:
            raise ValueError(f'into={self.into!r} must differ from dim — the local index replaces the global one')

    def _key_name(self) -> str:
        """Where the window started, never ``dim``, which would join silently against data over *dim*."""
        return f'{self.dim}_start'

    def _check_the_program(self, program: Program, sources: Mapping[str, Source]) -> None:
        """Refuse a window the program's rows cannot be whole inside, before one is taken.

        The program's `separability` answers; a reach the data decides is
        resolved from the parameter's least value. What the rows read behind
        is not refused: a window's first rows meet the edge policy there.
        """
        if self.into not in program.dimensions:
            raise SpecsolveError(
                f'EachWindow(into={self.into!r}) names the local index the model addresses a window by, and '
                f'the spec declares no such dimension. ' + did_you_mean(self.into, program.dimensions)
            )
        verdict = program.separability[self.into]
        named = {reach.name for reach in verdict.undecided if reach.kind == 'offset'}
        verdict = verdict.resolved({name: least_value(program, sources, name) for name in named})
        if verdict.coupled:
            raise SpecsolveError(
                f"EachWindow('{self.dim}', …, into='{self.into}') slices '{self.into}', which the model ties "
                f'together, so no window holds every row whole:\n{_listed(verdict.coupled)}\n'
                f'Each names the change that would lift it.'
            )
        if verdict.undecided:
            raise SpecsolveError(
                f"EachWindow('{self.dim}', …, into='{self.into}') slices '{self.into}', which the model reaches "
                f'along through a relation whose groups a window may split, and this driver does not resolve a '
                f'reach the relation decides:\n'
                f'{_listed({r.label: f"through the relation {r.name!r}" for r in verdict.undecided})}\n'
                f'Cut a dimension the relation does not group.'
            )
        if self.lookahead < verdict.ahead:
            raise SpecsolveError(
                f'EachWindow(lookahead={self.lookahead}) looks ahead by {self.lookahead} coordinate(s), and '
                f"the model reads {verdict.ahead} ahead along '{self.into}' — a row near a window's end would "
                f'read past it. Raise lookahead to at least {verdict.ahead}.'
            )
        if verdict.restarts:
            warnings.warn(
                f"the model counts a position along '{self.into}':\n{_listed(verdict.restarts)}\n"
                f'Every window restarts the count at its first row — what a rolling horizon seeding its '
                f'opening state means, and once per horizon otherwise.',
                SpecsolveWarning,
                stacklevel=3,
            )

    def _slice(self, sources: Mapping[str, Source], key_name: str) -> tuple[list[_Slice], _Stitch]:
        """One slice per window, keyed by its first coordinate.

        A window owns the coordinates its block names, and the
        [`_Stitch`][] records which.
        """
        carrying, coordinates = _coordinates(sources, self.dim, 'window')
        out: list[_Slice] = []
        owned: list[dict[str, object]] = []
        start = 0
        for owns in self._blocks(len(coordinates)):
            window = coordinates[start : start + owns + self.lookahead]
            local = {coordinate: position for position, coordinate in enumerate(window)}
            filtered = {
                name: (
                    table.filter(pl.col(self.dim).is_in(window))
                    .with_columns(pl.col(self.dim).replace_strict(local, return_dtype=pl.Int64).alias(self.into))
                    .drop(self.dim)
                )
                for name, table in carrying.items()
            }
            out.append(_Slice(window[0], {**sources, **filtered, self.into: range(len(window))}, owns))
            owned.extend(
                {key_name: window[0], self.into: position, self.dim: coordinate}
                for position, coordinate in enumerate(window[:owns])
            )
            start += owns
        return out, _Stitch(self.into, self.dim, pl.DataFrame(owned))

    def _blocks(self, total: int) -> list[int]:
        """How many coordinates each window owns, in order, summing to exactly *total*.

        An ``int`` repeats, the last window owning what is left. A sequence that
        stops short of the axis is refused.
        """
        if isinstance(self.steps, int):
            blocks = [self.steps] * -(-total // self.steps)
        else:
            blocks = list(self.steps)
            if sum(blocks) < total:
                raise DataError(
                    f'steps keeps {sum(blocks)} coordinate(s) across {len(blocks)} window(s), and '
                    f"'{self.dim}' has {total} — the last {total - sum(blocks)} would be solved by no window. "
                    f'List a block for them, or pass an int to repeat one size to the end.'
                )
        out: list[int] = []
        left = total
        for block in blocks:
            if left <= 0:
                break
            out.append(min(block, left))
            left -= block
        return out


#: An axis that cuts one set of sources into slices.
Axis = EachCoordinate | EachWindow

#: Slices written by hand, one ``(key, sources)`` pair each; ``axis=`` takes
#: this beside an [`Axis`][].
type HandBuilt = Sequence[tuple[Label, Mapping[str, Source]]]


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sweep:
    """What a fold returned: the answer over the model's own coordinates, and a record per slice.

    [`Result`][specsolve.relational.result.Result]'s readers — same names,
    same shapes — and every one returns **the answer** by default. An
    [`EachWindow`][] sweep is read over the dimension it sliced: each
    coordinate comes from the window that owns it, and the lookahead rows
    every overlapping window recomputed are dropped. An [`EachCoordinate`][]
    sweep, or a hand-built one, is keyed by slice, the key prepended, since
    each slice is a whole answer of its own. Nothing is combined across
    slices.

    ``per_window=True`` reads an EachWindow sweep one window at a time
    instead: keyed by where each window started, over the index inside it,
    lookahead rows included.
    """

    key_name: str
    #: One [`Record`][specsolve.relational.answer_layout.Record] per slice, in slice
    #: order — how every slice terminated, whether or not it produced an
    #: answer. The key column comes first, as its own type, so the table joins
    #: to the frames; ``slice_axis`` and ``slice`` name the slice again as
    #: text, and are what a saved sweep or an archive writes in its place. A
    #: slice that reached no objective holds null there rather than ``nan``,
    #: so the column aggregates over the slices that solved.
    record: pl.DataFrame
    #: One [`Metrics`][specsolve.relational.answer_layout.Metrics] per slice, keyed as
    #: [`record`][] is and in slice order — [`diagnostics`][specsolve.api.Model.diagnostics]
    #: one dimension wider, its counts and clocks only. Each row is the slice's
    #: own share: ``solves`` is ``1``, and ``loads`` is ``1`` where the solver
    #: took the model from scratch. Under a serial fold the first slice does
    #: and the rest are pushed values, so a later ``1`` is a slice whose data
    #: moved a mask; under an executor every slice builds alone and every one
    #: loads. So a slow sweep says which slice, and which phase of it.
    metrics: pl.DataFrame
    #: ``{kind: {name: frames}}``, one frame per slice, not concatenated; a
    #: kind no slice produced is absent.
    _frames: dict[str, dict[str, list[pl.DataFrame]]] = field(repr=False, default_factory=dict)
    _no_duals: str | None = field(repr=False, default=None)
    #: ``{kind: {name: reason}}`` for a name some slice could not produce, or
    #: one an archive holds no answer for.
    _absent: dict[str, dict[str, str]] = field(repr=False, default_factory=dict)
    _stitch: _Stitch | None = field(repr=False, default=None)
    #: Where the per-slice frames are instead, for a sweep solved with
    #: ``spill_to=`` or one read off disk.
    _spill: _Spill | None = field(repr=False, default=None)
    #: Where the frames lie when they are on disk rather than in memory, which
    #: the frame readers refuse and [`scan`][] reads.
    _disk: Path | None = field(repr=False, default=None)
    #: ``{kind: {name: frame}}``, the answer as a sweep archive holds it, read
    #: instead of folding it from the slices; ``None`` everywhere else.
    _answer: dict[str, dict[str, pl.LazyFrame]] | None = field(repr=False, default=None)
    #: Whether the per-window frames are there to read: not on an archive
    #: written without ``keep_windows=True``.
    _windows: bool = field(repr=False, default=True)
    #: What [`evaluate`][] lowers an undeclared expression through; ``None``
    #: on a live solve's Sweep, which retains no model.
    _evaluate: Callable[[str | Mapping[str, object]], pl.DataFrame] | None = field(repr=False, default=None)

    @classmethod
    def _folded(
        cls,
        key_name: str,
        stitch: _Stitch | None,
        answered: Generator[tuple[Label, _Answer], None, None],
        spill: _Spill | None,
        key_dtype: pl.DataType,
    ) -> Sweep:
        """Every slice's answer absorbed, in the order they arrive.

        Closing the stream releases the serial fold's model when a fold is
        abandoned. A reason a slice lacks something is kept from the first
        slice that gave one.
        """
        keys: list[Label] = []
        rows: list[Record] = []
        taken: list[Metrics] = []
        frames: defaultdict[str, defaultdict[str, list[pl.DataFrame]]] = defaultdict(lambda: defaultdict(list))
        no_duals: str | None = None
        no_expressions: dict[str, str] = {}
        with closing(answered) as stream:
            for key, answer in stream:
                no_duals = no_duals or answer.no_duals
                for name, reason in answer.no_expressions.items():
                    no_expressions.setdefault(name, reason)
                named = answer.sliced(key_name, key)
                keys.append(key)
                rows.append(named.meta)
                taken.append(named.metrics)
                for kind, produced in answer.frames.items():
                    for name, frame in produced.items():
                        frames[kind][name].append(_keyed(frame, key_name, key, key_dtype))
        keyed = pl.Series(key_name, keys, dtype=key_dtype)
        return cls(
            key_name=key_name,
            record=_rekeyed(pl.DataFrame([row._asdict() for row in rows], schema=RECORD_SCHEMA), keyed),
            metrics=_rekeyed(pl.DataFrame([row._asdict() for row in taken], schema=METRICS_SCHEMA), keyed),
            _frames={kind: dict(named) for kind, named in frames.items()},
            _no_duals=no_duals,
            _absent={'expression': no_expressions} if no_expressions else {},
            _stitch=stitch,
            _spill=spill,
            _disk=None if spill is None else spill.directory,
        )

    @property
    def keys(self) -> list[Label]:
        return self.record[self.key_name].to_list()

    def _held_here(self) -> None:
        if self._disk is not None:
            raise SpecsolveError(
                f"this sweep's frames are on disk under {str(self._disk)!r} rather than in memory: "
                f'sweep.scan(name) reads them back as a LazyFrame, and collecting it is the choice this '
                f'reader would otherwise make for you.'
            )

    def _check_per_window(self) -> None:
        """Refuse ``per_window=True`` where there are no windows to read."""
        if self._stitch is None:
            raise SpecsolveError(
                f'per_window=True reads an EachWindow sweep one window at a time, and this sweep was not cut '
                f'into windows: its answer already is one frame per slice, keyed by {self.key_name!r}. Read '
                f'it without per_window.'
            )
        if not self._windows:
            raise SpecsolveError(_NO_WINDOWS)

    def _named(self, kind: str, name: str, *, per_window: bool) -> pl.LazyFrame:
        """*name*'s frame of *kind*, lazily: the answer, or the frames per window.

        An archive's answer is read off its own files; any other answer is
        stitched from the slices.
        """
        if per_window:
            self._check_per_window()
        if self._answer is not None and not per_window:
            held: Mapping[str, object] = self._answer[kind]
            frame = self._answer[kind].get(name)
        elif self._spill is not None:
            held = dict.fromkeys(self._spill.held(kind))
            frame = self._spill.scan(kind, name)
        else:
            slices = self._frames.get(kind, {})
            held, frame = slices, pl.concat(slices[name]).lazy() if name in slices else None
        if frame is None:
            absent = self._absent.get(kind, {}).get(name) or (self._no_duals if kind == 'dual' else None)
            raise SpecsolveError(absent or _nothing_to_read(_LABELS[kind], name, held, self.record))
        if self._answer is not None and not per_window:
            return frame
        return self._answered(frame, per_window=per_window)

    def _answered(self, frame: _Frame, *, per_window: bool) -> _Frame:
        """*frame*, as the slices produced it, read the way the caller asked.

        [`EachWindow`][] stitches through its [`_Stitch`][]. Any other axis's
        slices are the answer, so the frame comes back unchanged, as it does
        per window, which the caller has already checked there are.
        """
        if per_window or self._stitch is None:
            return frame
        return self._stitch.restore(frame, self.key_name)

    def _unstitchable(self, frame: pl.DataFrame | pl.LazyFrame) -> str | None:
        """Why *frame*, as the slices produced it, has no answer; ``None`` where it has one.

        Only an EachWindow sweep stitches, so only its frames can lack an
        answer. The archive leaves out the file of a name this refuses, and
        [`to_dataset`][] leaves the name out, so the two agree.
        """
        return None if self._stitch is None else self._stitch.unstitchable(frame)

    def scan(self, name: str, kind: str = 'primal', *, per_window: bool = False) -> pl.LazyFrame:
        """One name's answer as a `polars.LazyFrame`.

        The reader for a sweep whose frames are on disk — solved with
        ``spill_to=``, or read by [`scan_sweep`][] or
        [`scan_archive`][specsolve.archive.scan_archive]. On one held in
        memory it is [`primal`][], [`dual`][] or [`evaluate`][] made lazy, so
        the same line reads either.

        Args:
            name: A variable, a constraint or a named expression the spec
                declares, as *kind* says.
            kind: ``primal``, ``dual`` or ``expression`` — the reader this
                stands in for.
            per_window: Read an EachWindow sweep one window at a time instead
                of its answer.

        Raises:
            SpecsolveError: No slice produced *name*, a *kind* that names no
                reader, or ``per_window`` where there are no windows to read.
        """
        if self._disk is None:
            return self._frame(name, kind, per_window=per_window).lazy()
        return self._named(checked_kind(kind), name, per_window=per_window)

    def primal(self, name: str, *, per_window: bool = False) -> pl.DataFrame:
        """One variable's answer.

        A slice that reached no solution contributes no rows, so this can be
        shorter than the sweep; [`record`][] is one row per slice always.

        Args:
            name: A variable the sweep's spec declares.
            per_window: Read an EachWindow sweep one window at a time instead
                of its answer: keyed by where each window started, lookahead
                rows included.

        Raises:
            SpecsolveError: No slice of the sweep produced *name*; a variable
                that is not over an EachWindow sweep's windowed dimension,
                which has an answer only per window; ``per_window`` on a sweep
                that was not cut into windows, or on an archive written
                without them.
        """
        self._held_here()
        return self._named('primal', name, per_window=per_window).collect()

    def dual(self, name: str, *, per_window: bool = False) -> pl.DataFrame:
        """One constraint's shadow prices.

        [`primal`][]'s shape and arguments. A slice whose model had an
        integer variable contributes no duals. In the answer of an EachWindow
        sweep each coordinate carries the price of the window that owns it,
        never a blend of several.

        Raises:
            SpecsolveError: No slice produced duals for *name* — the message says
                which of the two it was — or as [`primal`][] raises.
        """
        self._held_here()
        return self._named('dual', name, per_window=per_window).collect()

    def evaluate(self, expression: str | Mapping[str, object], *, per_window: bool = False) -> pl.DataFrame:
        """The value of *expression* at every slice's solution, as an answer.

        [`evaluate`][specsolve.relational.result.Result.evaluate] over the
        sweep, and [`primal`][]'s shape and arguments. *expression* is what
        one ``expressions:`` entry takes: a name the file declares, an
        expression string, or the mapping carrying ``cases:`` with ``dims:``
        and ``otherwise:``.

        A declared name is read from what the sweep holds, live or off
        disk. Anything else is valued at each slice's own solution with no
        re-solve, so it is available on the sweep
        [`load_archive`][specsolve.archive.load_archive] hands back, which
        carries the spec, sources and axis; a Sweep a live solve returned says
        it retains no model. An expression over a
        parameter the sweep **carried** is refused, that value being a
        previous slice's answer rather than stored data.

        In the answer of an EachWindow sweep each coordinate carries the
        value of the window that owns it, so summing it does not double-count
        the lookahead.

        Args:
            expression: A declared name, an expression string, or the ``cases:``
                mapping, as one ``expressions:`` entry takes.
            per_window: Read an EachWindow sweep one window at a time instead
                of its answer.

        Raises:
            SpecsolveError: No slice produced a declared *expression* — an
                evaluation that failed on every slice carries its own reason —
                a sweep on disk, which [`scan`][] reads instead; an undeclared
                expression on a Sweep with no model behind it, or one that reads
                a parameter the sweep carried; a quantity reduced over an
                EachWindow sweep's windowed dimension, which has an answer only
                per window; or ``per_window`` as [`primal`][] raises it.
            LanguageError: A construct outside the language, or a name the spec
                does not declare.
        """
        if isinstance(expression, str) and (
            expression in self._expression_names() or expression in self._absent.get('expression', {})
        ):
            self._held_here()
            return self._named('expression', expression, per_window=per_window).collect()
        if self._evaluate is None:
            raise SpecsolveError(self._nothing_to_evaluate(expression))
        if per_window:
            self._check_per_window()
        return self._answered(self._evaluate(expression), per_window=per_window)

    def _expression_names(self) -> Mapping[str, object]:
        """The declared expressions the sweep holds, in memory or on disk."""
        if self._answer is not None:
            return self._answer['expression']
        return (
            dict.fromkeys(self._spill.held('expression'))
            if self._spill is not None
            else self._frames.get('expression', {})
        )

    def _nothing_to_evaluate(self, expression: str | Mapping[str, object]) -> str:
        """Why a sweep with no model behind it cannot value *expression*; a string is also answered as a name."""
        no_model = no_model_behind_this_answer_message()
        if not isinstance(expression, str):
            return no_model
        return (
            f'{_nothing_to_read(_LABELS["expression"], expression, self._expression_names(), self.record)} {no_model}'
        )

    def _frame(self, name: str, kind: str, *, per_window: bool) -> pl.DataFrame:
        """*name* through the reader *kind* names."""
        reader = {'primal': self.primal, 'dual': self.dual, 'expression': self.evaluate}[checked_kind(kind)]
        return reader(name, per_window=per_window)

    def to_pandas(self, name: str, kind: str = 'primal', *, per_window: bool = False) -> pd.DataFrame:
        """One name's answer as a tidy `pandas.DataFrame`.

        The name is resolved before pandas is imported, so a sweep that never
        held *name* says so on any install.

        Args:
            name: A variable, a constraint or a named expression, as *kind*
                says.
            kind: ``primal``, ``dual`` or ``expression`` — the reader this
                stands in for.
            per_window: Read an EachWindow sweep one window at a time instead
                of its answer.
        """
        return tidy_to_pandas(self._frame(name, kind, per_window=per_window))

    def to_dataarray(self, name: str, kind: str = 'primal', *, per_window: bool = False) -> xr.DataArray:
        """One name's answer as a `xarray.DataArray`; [`to_pandas`][]'s arguments.

        An EachWindow sweep's answer is indexed by the dimension it sliced, so
        a rolling horizon's dispatch comes back indexed by time. Any other
        sweep adds the slice key as a dimension, named by the axis:
        ``(scenario, …)``. There, and per window, where the extra dimension is
        ``<dim>_start``, a slice that reached no solution has no rows and
        comes back NaN, the same answer a masked coordinate gets from
        ``Result``.
        """
        return tidy_to_dataarray(self.to_pandas(name, kind, per_window=per_window), name)

    def to_dataset(self, *names: str, kind: str = 'primal', per_window: bool = False) -> xr.Dataset:
        """The named answers of one *kind* as one `xarray.Dataset`; all of that kind by default.

        One kind per call, since a dual and a variable may share a name;
        [`save`][] writes every kind.

        Args:
            names: What to include; none means every name of *kind* the
                sweep has an answer for. A name an EachWindow sweep cannot
                stitch is left out, as an archive leaves out its file; named,
                or read ``per_window``, it is read as [`primal`][] reads it.
            kind: ``primal``, ``dual`` or ``expression``.
            per_window: Read an EachWindow sweep one window at a time instead
                of its answer.

        Raises:
            SpecsolveError: The sweep has no answer of *kind* at all — the
                message names each name it left out, and why — or its frames
                are on disk; or as [`primal`][] raises.
        """
        held = names or self._names_held(kind, per_window=per_window)
        return tidy_to_dataset(held, lambda name: self.to_dataarray(name, kind, per_window=per_window))

    def save(self, directory: str | Path) -> Path:
        """Everything the sweep holds, per slice, written as ``spill_to=`` would have written it.

        The same layout: ``<kind>/<name>/<position>.parquet`` for every
        primal, dual and expression, the slice key a column of each, with
        ``record/``, ``metrics/`` and the manifest beside them. So the
        directory is a spilled sweep: [`scan`][] reads it, and the call
        that made this sweep, pointed at it with ``spill_to=``, reads it back
        without solving a slice.

        A sweep whose every slice terminated without values writes each
        slice's record and no frames, as one such solve does, rather than
        refusing.

        Returns:
            The directory.

        Raises:
            SpecsolveError: The sweep's frames are on disk already, or it was
                read off an archive written without its windows.
        """
        self._held_here()
        by_key = {kind: _slice_index(self, kind) for kind in KINDS}
        spill = _Spill.opened(directory, self.key_name, self.keys, self.record[self.key_name].dtype, self._stitch)
        write_reasons(spill.directory, self._no_duals, self._absent)
        for position, key in enumerate(self.keys):
            meta = Record(**self.record.drop(self.key_name).row(position, named=True))
            taken = Metrics(**self.metrics.select(Metrics._fields).row(position, named=True))
            frames = {
                kind: {name: keyed[key] for name, keyed in names.items() if key in keyed}
                for kind, names in by_key.items()
            }
            answer = _Answer(meta, taken, frames)
            spill.write(position, key, answer)
        return spill.directory

    def _names_held(self, kind: str, *, per_window: bool) -> tuple[str, ...]:
        """Every name of *kind* there is an answer for, sorted; none at all is refused.

        Per window, every name the windows hold. A live sweep leaves out what
        ``_unstitchable`` refuses, which an archive's answer already lacks.
        """
        self._held_here()
        kind = checked_kind(kind)
        if per_window:
            self._check_per_window()
        left_out = dict(self._absent.get(kind, {}))
        if self._answer is not None and not per_window:
            held: Mapping[str, object] = self._answer[kind]
        else:
            held = {}
            for name, frames in self._frames.get(kind, {}).items():
                if not per_window and (why := self._unstitchable(frames[0])):
                    left_out[name] = why
                else:
                    held[name] = frames
        if not held:
            absent = (self._no_duals if kind == 'dual' else None) or _none_answered(_LABELS[kind], left_out)
            raise SpecsolveError(absent or _nothing_to_read(_LABELS[kind], 'anything', held, self.record))
        return tuple(sorted(held))

    def __len__(self) -> int:
        return self.record.height


#: Why an archive's sweep has nothing to read per window.
_NO_WINDOWS = (
    'this archive holds the answer only, because it was written without keep_windows=True, so it has no '
    'per-window frames to read. Solving again from the archived spec and sources restores them: '
    'load_archive gives both, with the axis and the carry, so '
    'sps.solve_over(archive.spec, archive.sources, archive.axis, carry=archive.carry) runs the sweep again.'
)

#: Where an archive keeps a windowed sweep's per-window frames, under its ``answer/``.
_WINDOWS_DIR = 'windows'


def _by_key(frames: Sequence[pl.DataFrame], key_name: str) -> dict[Label, pl.DataFrame]:
    """One name's held frames by the slice key each carries, the key column dropped; an empty frame is left out."""
    return {frame[key_name][0]: frame.drop(key_name) for frame in frames if frame.height}


def _none_answered(kind: str, left_out: Mapping[str, str]) -> str | None:
    """The message for a sweep whose every *kind* was left out of its answer, or ``None`` where none was."""
    if not left_out:
        return None
    return f'no {kind} of this sweep has an answer, and each one says why:\n{_listed(dict(sorted(left_out.items())))}'


def _nothing_to_read(kind: str, name: str, held: Mapping[str, object], record: pl.DataFrame) -> str:
    """The message for *name* having no frame, whether undeclared or produced by no slice."""
    conditions = ', '.join(sorted(set(record['termination_condition'].to_list())))
    if held:
        listed = ', '.join(repr(k) for k in sorted(held))
        return (
            f'no {kind} {name!r} in this sweep — it holds {listed}. '
            f'If the spec declares it, no slice produced one: all {record.height} terminated {conditions}.'
        )
    return (
        f'this sweep holds no {kind} frames at all — every one of its {record.height} slices '
        f'terminated {conditions}. The fold ran; the models did not solve. '
        f'sweep.record carries the status of each slice.'
    )


def load_sweep(directory: str | Path) -> Sweep:
    """Read back a sweep [`Sweep.save`][] wrote, or one ``solve_over(spill_to=)`` spilled.

    The sweep comes back **held**: every slice's frames are in memory when this
    returns, so it is the value a sweep solved without ``spill_to=`` is —
    [`Sweep.primal`][], [`Sweep.to_dataset`][] and [`Sweep.save`][] all
    answer, and it owes *directory* nothing afterwards. A sweep larger than
    memory is [`scan_sweep`][] instead.

    [`Sweep.record`][] and [`Sweep.metrics`][] are one row per slice
    either way, and the readers return the answer on both, the manifest
    carrying what a window owns.

    Args:
        directory: Where the sweep was written.

    Returns:
        The sweep, keyed as it was solved.

    Raises:
        LayoutError: *directory* holds no ``sweep.json``, misses a record every
            fold writes, or is in a layout that has moved since it was written.
    """
    scanned = scan_sweep(directory)
    spill = scanned._spill
    assert spill is not None, 'scan_sweep returns a spilled sweep, which is what there is to hold here'
    return replace(_holding(scanned, spill), _spill=None, _disk=None)


def _holding(sweep: Sweep, spill: _Spill) -> Sweep:
    """*sweep* with *spill*'s per-slice frames read into memory."""
    return replace(
        sweep, _frames={kind: {name: spill.whole(kind, name) for name in spill.held(kind)} for kind in KINDS}
    )


def scan_sweep(directory: str | Path) -> Sweep:
    """The sweep under *directory*, its frames left where they lie.

    [`load_sweep`][]'s other half, and the value a sweep solved with
    ``spill_to=`` already is: nothing but the record is read, and
    [`Sweep.scan`][] reads a name back as a `polars.LazyFrame` when one
    is asked for. That is the reader for a sweep too large to hold, and it
    costs the frame readers: [`Sweep.primal`][] and its siblings refuse,
    naming [`Sweep.scan`][].

    *directory* has to outlive the sweep.

    Args:
        directory: As [`load_sweep`][] takes it.

    Raises:
        LayoutError: As [`load_sweep`][] raises it.
    """
    under = Path(directory)
    opened = _opened_sweep(under)
    return replace(opened, _spill=_Spill(under, opened.key_name, opened.record[opened.key_name].dtype), _disk=under)


def _opened_sweep(under: Path) -> Sweep:
    """The sweep whose manifest is under *under*: its record and reasons, and none of its frames."""
    manifest = under / _MANIFEST_FILE
    if not manifest.is_file():
        raise LayoutError(
            f'{str(under)!r} holds no {_MANIFEST_FILE!r}, so it is not a sweep save() or spill_to= wrote. A '
            f'single solve writes no manifest and is read by load_result.'
        )
    check_format(under)
    found = json.loads(manifest.read_text())
    stitch = found['stitch']
    no_duals, absent = read_reasons(under)
    key_name = found['key_name']
    keys = pl.read_parquet(under / _KEYS_FILE, columns=[key_name]).to_series()
    return Sweep(
        key_name=key_name,
        record=_rekeyed(consolidated(under, RECORD_FILE), keys),
        metrics=_rekeyed(consolidated(under, METRICS_FILE), keys),
        _no_duals=no_duals,
        _absent=absent,
        _stitch=None
        if stitch is None
        else _Stitch(stitch['local'], stitch['dim'], pl.read_parquet(under / _OWNED_FILE).drop(RUN, strict=False)),
    )


def read_archived_sweep(under: Path, *, whole: bool) -> Sweep:
    """The sweep an archive's ``answer/`` holds: one file per name holding its answer, and the windows where kept.

    The readers read the answer files; ``per_window=True`` reads the windows,
    and is refused naming ``keep_windows=True`` where the archive has none.
    The ``specsolve_run`` column every archived frame carries is left on disk,
    so a frame read out of one equals the frame the live sweep returns.

    Args:
        under: The archive's ``answer/``.
        whole: Read every frame into memory, as [`load_sweep`][] does, rather
            than at the call that asks, as [`scan_sweep`][] does.
    """
    opened = _opened_sweep(under)
    answer = {kind: saved_frames(under / kind, whole=whole) for kind in KINDS}
    kept = json.loads((under / _MANIFEST_FILE).read_text())['windows']
    opened = replace(opened, _answer=answer, _windows=kept)
    spill = _Spill(under / _WINDOWS_DIR, opened.key_name, opened.record[opened.key_name].dtype) if kept else None
    if not whole:
        return replace(opened, _spill=spill, _disk=under)
    return opened if spill is None else _holding(opened, spill)


def _rekeyed(table: pl.DataFrame, keys: pl.Series) -> pl.DataFrame:
    """*table* with the typed key prepended, matched on the text each row's ``slice`` holds.

    Matched rather than placed by position, so a spill an interrupted fold
    left reads back the slices it finished.
    """
    texts = [str(key) for key in keys.to_list()]
    return table.select(pl.col('slice').replace_strict(texts, keys, return_dtype=keys.dtype).alias(keys.name), pl.all())


def axis_manifest(axis: Axis) -> dict[str, Any]:  # pyrefly: ignore[explicit-any] — the archive's own JSON
    """*axis* as the JSON an archive carries, read back by [`axis_from`][]."""
    if isinstance(axis, EachCoordinate):
        return {'each': 'coordinate', 'dim': axis.dim}
    steps = axis.steps if isinstance(axis.steps, int) else list(axis.steps)
    return {'each': 'window', 'dim': axis.dim, 'steps': steps, 'lookahead': axis.lookahead, 'into': axis.into}


def axis_from(manifest: Mapping[str, Any]) -> Axis:  # pyrefly: ignore[explicit-any] — the archive's own JSON
    """The axis [`axis_manifest`][] wrote."""
    if manifest['each'] == 'coordinate':
        return EachCoordinate(manifest['dim'])
    return EachWindow(manifest['dim'], steps=manifest['steps'], lookahead=manifest['lookahead'], into=manifest['into'])


def _archiving(archive: str | Path | None, axis: Axis | HandBuilt, *, keep_windows: bool) -> tuple[Path, Axis] | None:
    """Where the archive goes and the axis that re-runs it, or ``None`` for no archive."""
    if keep_windows and archive is None:
        raise SpecsolveError(
            'keep_windows=True says what an archive keeps beside the answer, and there is no archive=. The '
            'sweep in memory reads its windows through per_window=True already. Pass archive=, or drop '
            'keep_windows.'
        )
    if keep_windows and not isinstance(axis, EachWindow):
        raise SpecsolveError(
            "keep_windows=True keeps an EachWindow sweep's frames per window beside its answer, and this axis "
            'does not cut windows: its answer already is one frame per slice, which the archive holds. Drop '
            'keep_windows.'
        )
    if archive is None:
        return None
    if not isinstance(axis, Axis):
        raise SpecsolveError(
            'archive= takes a sweep cut by EachCoordinate or EachWindow, which say how one set of sources '
            'was cut and so how the archive can be re-run. A hand-built list is a set of sources per '
            'slice, which are unrelated questions — archive one solve each.'
        )
    out = Path(archive)
    check_the_target(out)
    return out, axis


def solve_over(
    spec: Buildable,
    sources: Mapping[str, Source],
    axis: Axis | HandBuilt,
    *,
    carry: Mapping[str, str] | None = None,
    key_name: str | None = None,
    executor: Executor | None = None,
    workers_share_fs: bool | None = None,
    solver_options: Mapping[str, object] | None = None,
    record_options: Sequence[str] | None = None,
    solver_name: str = 'highs',
    keep: Keep = 'solver',
    spill_to: str | Path | None = None,
    archive: str | Path | None = None,
    keep_windows: bool = False,
) -> Sweep:
    """Solve *spec* once per slice of *axis* and fold the answers together.

    The rules — what a carry copies, how the key column is named, which
    executor to choose — are [sweeps](https://specsolve.readthedocs.io/en/latest/reference/sweeps/).

    Args:
        spec: As [`check`][specsolve.api.check] takes it. Parsed once, whichever
            executor runs the slices.
        sources: As [`build`][specsolve.api.build] takes them, every shape
            included; the axis filters the parameters and relations that carry
            it and passes the rest through.
        axis: [`EachCoordinate`][], [`EachWindow`][], or a list of
            ``(key, sources)`` written by hand.
        carry: ``{parameter: variable}``: one slice's answer copied into the
            next slice's data. Where the two are over different dimensions, the
            last coordinate the slice owns is handed on. The first slice takes
            the parameter from *sources* as its seed.
        key_name: What to call the slice column; a class axis names its own,
            a hand-built list has to be told.
        executor: Any `concurrent.futures.Executor`; ``None`` runs the
            slices in order on one model. A process pool must be ``spawn``
            or ``forkserver`` — a forked worker hangs.
        workers_share_fs: Whether the executor's workers can read this
            process's paths. Decided for the stdlib pools; anything else is
            assumed not to, and paths travel as bytes.
        solver_options: As [`solve`][specsolve.api.Model.solve] takes them.
        record_options: As [`solve`][specsolve.api.Model.solve] takes them,
            reaching every slice.
        solver_name: As [`solve`][specsolve.api.Model.solve] takes it.
        keep: As [`solve`][specsolve.api.Model.solve] takes it, reaching every
            slice. Under an executor every slice is a first solve and keeps
            nothing, whatever was asked.
        spill_to: A directory to write each slice's frames to as the fold goes,
            so the sweep's memory stays at one slice however many there
            are. Read back through [`Sweep.scan`][]. A directory holds
            one sweep: run the same sweep at it again and the slices already
            there are not solved again, which is how an interrupted sweep
            resumes.
        archive: Where to write the whole thing — the model, the sources the
            sweep was cut from, the axis that cut them, and every slice's
            answer — so that ``sps.load_archive`` gives all four back and the
            sweep runs again from the file alone. A ``.zip`` suffix packs it
            into one file and anything else is a directory. The archive holds
            the answer, one file per name at ``answer/<kind>/<name>.parquet``
            as a single solve's archive does; a name with no answer, such as
            a quantity reduced over an EachWindow sweep's windowed dimension,
            is left out and ``answer/reasons.parquet`` says why. Given beside
            *spill_to*, the spill is what the archive packs, so a sweep too
            large to hold is archived without ever being held. The archive is
            a second copy of the answers on disk; the memory is what
            *spill_to* bounds. A sliced source is archived whole, the column
            the axis cuts on included, and one number over a window's local
            index as a table over the axis. A hand-built axis is refused,
            since a list of ``(key, sources)`` is a set of sources per slice:
            archive one solve each.
        keep_windows: Also archive an [`EachWindow`][] sweep's frames per
            window, lookahead rows included, under ``answer/windows/``, so
            that ``per_window=True`` reads off the archive. Refused for any
            other axis, and without *archive*.

    Returns:
        The sweep, which reads its answer.

    Raises:
        SpecsolveError: A carry that cannot line up, has no seed, collapses a
            dimension the axis does not advance along, or is asked together with
            an executor; a key that collides with a column the frames carry;
            an axis the program does not allow; a *spill_to* directory holding
            another sweep; *keep_windows* without *archive* or on an axis that
            does not cut windows. All refused before a slice is taken, and every
            one answerable from the declarations before a source is read.
            Keys of more than one type, or two keys of one text, are refused
            before a slice is taken too.
        DataError: No source carries the axis, an index of another
            dimension carries it, or the axis produced no slices.

    Warns:
        SpecsolveWarning: A source carrying the axis that is short of a
            coordinate another has — that slice builds it empty — or a
            position the model counts, which every window restarts.
    """
    if carry and executor is not None:
        raise SpecsolveError(
            'carry and executor are mutually exclusive: a carried value makes slice i+1 depend on '
            "slice i's answer, so the slices cannot run concurrently. Drop the executor, or drop the carry."
        )
    document = declared(spec)
    sources = _held(sources)
    archiving = _archiving(archive, axis, keep_windows=keep_windows)
    program = check(document)
    plan = {p: _CarryRule.resolved(program, p, v) for p, v in (carry or {}).items()}
    key_name = _key_column(axis, key_name, program)

    if isinstance(axis, Axis):
        _check_the_carry(plan, axis, sources)
        _check_no_index_is_cut(program, sources, axis)
        axis._check_the_program(program, sources)
        slices, stitch = axis._slice(sources, key_name)
    else:
        slices = [_Slice(*entry) for entry in axis]
        stitch = None
        _check_the_carry(plan, axis, slices[0].sources if slices else {})
    if not slices:
        raise DataError('the axis produced no slices')
    solving = {
        'solver_name': solver_name,
        'solver_options': dict(solver_options or {}) or None,
        'record_options': record_options,
    }
    keys = [current.key for current in slices]
    key_dtype = _one_key_type(keys, key_name)
    spill = None if spill_to is None else _Spill.opened(spill_to, key_name, keys, key_dtype, stitch)
    answered = (
        _serially(program, document, slices, solving, plan, keep, spill)
        if executor is None
        else _pooled(executor, workers_share_fs, program, document, slices, solving, spill)
    )
    folded = Sweep._folded(key_name, stitch, answered, spill, key_dtype)
    if spill is not None:
        write_reasons(spill.directory, folded._no_duals, folded._absent)
    if archiving is not None:
        out, cut = archiving
        _archive_the_sweep(
            out, document, program, cut, dict(carry or {}), sources, folded, slices[0].sources, keep_windows
        )
    return folded


def _held(sources: Mapping[str, Source]) -> dict[str, Source]:
    """*sources* with each one-shot iterator read into a list.

    Every slice reads the sources it does not cut, and the archive reads them
    again, so an iterator would be spent after the first read.
    """
    return {name: list(obj) if isinstance(obj, Iterator) else obj for name, obj in sources.items()}


def _archive_the_sweep(
    out: Path,
    spec: Spec,
    program: Program,
    axis: Axis,
    carry: Mapping[str, str],
    sources: Mapping[str, Source],
    folded: Sweep,
    one_slice: Mapping[str, Source],
    keep_windows: bool,
) -> None:
    """Write the sweep's question and its answer to *out*.

    Each source's tidy table comes from *one_slice*; the ones the axis cuts
    are written uncut, as [`_uncut`][] gives them, and a number over a
    window's local index over the axis, as [`_spread_over_the_axis`][] gives
    it. A spilled sweep's answer is read off its spill, so it is never held;
    a held one is spilled to scratch first.
    """
    manifest = axis_manifest(axis)
    if carry:
        manifest['carry'] = dict(carry)
    tidied = numbered(program, tidy_sources(program, one_slice))
    carried = carries(sources, axis.dim)
    cut = {name: _uncut(program, axis, name, table) for name, table in carried.items()}
    held = {**tidied, **_spread_over_the_axis(program, axis, sources, tidied, carried), **cut}
    tables = {name: held[name] for name in sources}
    with beside(out) as scratch:
        spilled = folded if folded._spill is not None else scan_sweep(folded.save(scratch / 'slices'))
        answer = _the_answer(spilled, scratch / ANSWER_DIR, keep_windows=keep_windows)
        write_archive(out, spec, tables, axis=manifest, answer=answer)


def _the_answer(sweep: Sweep, under: Path, *, keep_windows: bool) -> Path:
    """The ``answer/`` an archive holds for a spilled *sweep*, laid out under *under*.

    One file per kind and name, holding the answer as the readers return it
    and streamed from the spill. A name an EachWindow sweep cannot stitch
    has no file, and ``reasons.parquet`` says why. The record and metrics are
    written in a single solve's columns, so ``keys.parquet`` is copied
    beside them to give the keys their type back. With *keep_windows*, the
    spill's per-window files are copied under ``windows/``, and ``sweep.json``
    records that they were kept: a zip holds files only, so a ``windows/``
    no window wrote a file to is not there to say so.
    """
    spill = sweep._spill
    assert spill is not None, 'the answer is read off a spill, so a sweep too large to hold is never held'
    write_format(under)
    manifest = json.loads((spill.directory / _MANIFEST_FILE).read_text())
    (under / _MANIFEST_FILE).write_text(json.dumps({**manifest, 'windows': keep_windows}))
    shutil.copyfile(spill.directory / _KEYS_FILE, under / _KEYS_FILE)
    if sweep._stitch is not None:
        shutil.copyfile(spill.directory / _OWNED_FILE, under / _OWNED_FILE)
    write_whole(sweep.record.drop(sweep.key_name), under / RECORD_FILE)
    write_whole(sweep.metrics.drop(sweep.key_name), under / METRICS_FILE)
    absent = {kind: dict(names) for kind, names in sweep._absent.items()}
    for kind in KINDS:
        for name in spill.held(kind):
            frame = spill.scan(kind, name)
            assert frame is not None, 'a name the spill lists has files'
            if why := sweep._unstitchable(frame):
                absent.setdefault(kind, {})[name] = why
                continue
            write_whole(sweep._answered(frame, per_window=False), under / kind / f'{name}.parquet')
        if keep_windows and (spill.directory / kind).is_dir():
            shutil.copytree(spill.directory / kind, under / _WINDOWS_DIR / kind)
    write_reasons(under, sweep._no_duals, absent)
    return under


def _uncut(program: Program, axis: Axis, name: str, table: pl.LazyFrame) -> pl.LazyFrame:
    """A source the axis cuts, as its tidy columns with the axis column first.

    A window's local index is not a column of the uncut table: the axis
    column stands where it would be.
    """
    declared = (
        [*program.parameters[name].dims, 'value'] if name in program.parameters else program.relations[name].roles
    )
    local = axis.into if isinstance(axis, EachWindow) else None
    return table.select(list(dict.fromkeys([axis.dim, *(column for column in declared if column != local)])))


def _spread_over_the_axis(
    program: Program,
    axis: Axis,
    sources: Mapping[str, Source],
    tidied: Mapping[str, pl.LazyFrame],
    carried: Mapping[str, pl.LazyFrame],
) -> dict[str, pl.LazyFrame]:
    """Each parameter given as one number over a window's local index, as a table over the axis.

    The number spreads over the labels of the window it attaches to, and the
    local index has one label per coordinate the window holds, so no one
    window's table is what a window of another length read. Over the axis,
    each window cuts what its own solve read. A parameter given in any other
    shape, a relation, and an index other than the local one attach as one
    table in every slice that builds, so the archive holds the first slice's.
    """
    if not isinstance(axis, EachWindow):
        return {}
    numbers = [
        name
        for name, declared in program.parameters.items()
        if axis.into in declared.dims and isinstance(sources[name], (bool, int, float))
    ]
    coordinates = pl.concat([table.select(axis.dim) for table in carried.values()], how='vertical_relaxed')
    coordinates = coordinates.unique().sort(axis.dim)
    return {
        name: coordinates.join(tidied[name].drop(axis.into).unique(maintain_order=True), how='cross')
        for name in numbers
    }


def _check_no_index_is_cut(program: Program, sources: Mapping[str, Source], axis: Axis) -> None:
    """Refuse an index of a dimension other than the axis's own that carries the axis column.

    An index says which labels the model has, and a sweep cuts the tables that
    carry the axis, so a carried index would make each slice a model over
    other labels. The axis's own index carries the axis as its labels, and
    the axis refuses that dimension in its own words.
    """
    for dim in program.dimensions:
        table = as_frame(sources[dim]) if dim in sources and dim != axis.dim else None
        if table is None or axis.dim not in table.collect_schema().names():
            continue
        raise DataError(
            f"index for dimension '{dim}' carries a '{axis.dim}' column, and {type(axis).__name__}"
            f"('{axis.dim}') cuts every table that carries '{axis.dim}'. An index is not cut: it lists "
            f"the labels every slice has. Pass the '{dim}' labels alone, and say which of them each "
            f"slice has in a parameter or a relation over ('{dim}', '{axis.dim}'), where a missing row "
            f'already reads as absent.'
        )


def attach_sweep_readers(
    sweep: Sweep,
    spec: Spec,
    sources: Mapping[str, Source],
    axis: Axis,
    carry: Mapping[str, str],
) -> Sweep:
    """*sweep* with an undeclared expression readable through [`Sweep.evaluate`][], over a sweep archive's own inputs.

    Each slice's saved primal is put back against its rebuilt model, so nothing
    is re-solved.
    """
    return replace(sweep, _evaluate=_sweep_evaluator(sweep, spec, sources, axis, carry))


def _per_slice(
    sweep: Sweep, spec: Spec, sources: Mapping[str, Source], axis: Axis
) -> Iterator[tuple[Label, Callable[[str | Mapping[str, object]], pl.DataFrame]]]:
    """``(key, evaluate)`` for each slice that produced a solution, its model rebuilt from its cut of the sources."""
    primal, dual = _slice_index(sweep, 'primal'), _slice_index(sweep, 'dual')
    for key, slice_sources in axis.slices(sources):
        slice_primals = {name: by_key[key] for name, by_key in primal.items() if key in by_key}
        if not slice_primals:
            continue
        slice_duals = {name: by_key[key] for name, by_key in dual.items() if key in by_key} or None
        yield key, build(spec, slice_sources).evaluator(slice_primals, slice_duals, sweep._no_duals)


def _refuse_carried(carried: set[str], nodes: Iterable[Expression]) -> None:
    """Refuse a block that reads a parameter the sweep carried — its value is not stored per slice."""
    if touched := sorted({name for node in nodes for name in parameters_of(node)} & carried):
        raise SpecsolveError(
            f'this expression reads {touched}, which the sweep carried from one slice into the next, and a '
            f"carried value is a previous slice's answer rather than stored data — so it cannot be put back "
            f'per slice from the archive. Re-run the sweep with sps.solve_over(spec, sources, axis, carry=...) '
            f'and evaluate on what comes back, or read a quantity over the sweep that reads no carried parameter.'
        )


def _sweep_evaluator(
    sweep: Sweep,
    spec: Spec,
    sources: Mapping[str, Source],
    axis: Axis,
    carry: Mapping[str, str],
) -> Callable[[str | Mapping[str, object]], pl.DataFrame]:
    """One expression at every slice's solution, keyed by slice."""
    carried = set(carry)
    key_dtype = sweep.record.schema[sweep.key_name]

    def evaluate(expression: str | Mapping[str, object]) -> pl.DataFrame:
        _refuse_carried(carried, [lower(spec, expression)])
        pieces = [
            _keyed(evaluate_one(expression), sweep.key_name, key, key_dtype)
            for key, evaluate_one in _per_slice(sweep, spec, sources, axis)
        ]
        if not pieces:
            raise SpecsolveError('no slice of this sweep produced a solution, so an expression has nothing to read at.')
        return pl.concat(pieces)

    return evaluate


def _slice_index(sweep: Sweep, kind: str) -> dict[str, dict[Label, pl.DataFrame]]:
    """``{name: {slice key: frame}}`` for *kind*, whether the sweep is held, spilled or archived.

    An archived sweep that was not cut into windows holds its slices as its
    answer, keyed already. One that was holds them only where the windows were
    kept.
    """
    if sweep._answer is not None and sweep._stitch is None:
        return {
            name: _by_key(frame.collect().partition_by(sweep.key_name, maintain_order=True), sweep.key_name)
            for name, frame in sweep._answer[kind].items()
        }
    if not sweep._windows:
        raise SpecsolveError(_NO_WINDOWS)
    if sweep._spill is None:
        return {name: _by_key(frames, sweep.key_name) for name, frames in sweep._frames.get(kind, {}).items()}
    index: dict[str, dict[Label, pl.DataFrame]] = {}
    for name in sweep._spill.held(kind):
        frame = sweep._spill.scan(kind, name)
        if frame is not None:
            index[name] = _by_key(frame.collect().partition_by(sweep.key_name), sweep.key_name)
    return index


def _check_the_carry(
    plan: Mapping[str, _CarryRule],
    axis: Axis | HandBuilt,
    first: Mapping[str, Source],
) -> None:
    """Refuse a carry with no seed, or one that collapses a dimension other than [`EachWindow.into`][].

    Reads no source: the seed is a key of *first*, and the owned dimension is
    the axis's own.
    """
    for parameter, rule in plan.items():
        if parameter not in first:
            raise SpecsolveError(
                f'carry writes {parameter!r} from the second slice on, and the first slice has nothing to '
                f'start from: supply {parameter!r} in sources as the seed.'
            )
        if rule.dropped is None:
            continue
        if isinstance(axis, EachWindow) and rule.dropped == axis.into:
            continue
        owned = f'this axis advances along {axis.into!r}' if isinstance(axis, EachWindow) else 'this axis owns none'
        raise SpecsolveError(
            f'carry {parameter!r} <- {rule.variable!r} collapses {rule.dropped!r}, and {owned}: a carry hands '
            f'on the last coordinate a slice owns, so it can only collapse the dimension the sweep advances '
            f'along. Reduce {rule.dropped!r} in the YAML — a derived variable is where the oracle can see the '
            f'math — so that {parameter!r} and {rule.variable!r} are over the same dimensions.'
        )


def _serially(
    program: Program,
    document: Spec,
    slices: Sequence[_Slice],
    solving: Mapping[str, Any],  # pyrefly: ignore[explicit-any] — the verb's own keywords, forwarded
    plan: Mapping[str, _CarryRule],
    keep: Keep,
    spill: _Spill | None,
) -> Generator[tuple[Label, _Answer], None, None]:
    """Each slice's answer, off one model updated in place.

    A slice naming other sources than the last is rebuilt, since ``update`` is
    partial; a rebuild closes the previous model first. A generator because
    slice ``i+1``'s carry is read from slice ``i``'s frames after the yield;
    the caller closes it to release the model. A slice the spill holds is read
    back, and one solved here is written before it is yielded.
    """
    model: Model | None = None
    named: frozenset[str] | None = None
    state: dict[str, pl.DataFrame] = {}
    try:
        for position, current in enumerate(slices):
            if spill is not None and spill.done(position):
                answer = spill.read_back(position)
                primals = spill.primals(position, {rule.variable for rule in plan.values()})
                yield current.key, answer
                state = _carried(plan, primals, current, position, slices, answer)
                continue
            sources = {**current.sources, **state}
            names = frozenset(sources)
            with _named_slice(current.key, position, len(slices)):
                if model is not None and names == named:
                    before = model.diagnostics()
                    model.update(sources)
                else:
                    if model is not None:
                        model.close()
                    model, named, before = build(document, sources), names, None
                result = model.solve(**solving, keep=keep)
                answer = _answers(result, program, _slice_metrics(model.diagnostics(), before))
            primals = answer.frames.get('primal', {})
            if spill is not None:
                answer = spill.write(position, current.key, answer)
            yield current.key, answer
            state = _carried(plan, primals, current, position, slices, answer)
    finally:
        if model is not None:
            model.close()


def _carried(
    plan: Mapping[str, _CarryRule],
    primals: Mapping[str, pl.DataFrame],
    current: _Slice,
    position: int,
    slices: Sequence[_Slice],
    answer: _Answer,
) -> dict[str, pl.DataFrame]:
    """What the next slice starts from, read out of *primals*; nothing for the last slice, or with no plan."""
    if not plan or position == len(slices) - 1:
        return {}
    if not primals:
        raise SpecsolveError(
            f'slice {current.key!r} ({position + 1} of {len(slices)}) terminated '
            f'{answer.meta.termination_condition}, so slice {slices[position + 1].key!r} has no '
            f'{sorted({rule.variable for rule in plan.values()})} to start from. A carried sweep '
            f'stops at the first slice that leaves nothing to carry; the {position} before it solved.'
        )
    return {p: rule.value_from(primals, p, current.key, current.owns) for p, rule in plan.items()}


@contextmanager
def _named_slice(key: Label, position: int, count: int) -> Generator[None, None, None]:
    """Whatever a slice raises leaves naming the slice, as a note, so the error stays the engine's own."""
    try:
        yield
    except Exception as exc:
        exc.add_note(f'in slice {key!r} ({position + 1} of {count})')
        raise


def _pooled(
    executor: Executor,
    workers_share_fs: bool | None,
    program: Program,
    document: Spec,
    slices: Sequence[_Slice],
    solving: Mapping[str, Any],  # pyrefly: ignore[explicit-any] — the verb's own keywords, forwarded
    spill: _Spill | None,
) -> Generator[tuple[Label, _Answer], None, None]:
    """The same, from slices built independently and possibly elsewhere.

    Yielded in slice order, never completion order. A built model cannot cross
    a process, so each slice builds its own. A slice the spill holds is never
    submitted; one that comes back is written here, by the process that owns
    the directory.
    """
    crosses = _crosses_a_process(executor)
    shared = _shares_filesystem(executor, workers_share_fs)
    memo: dict[str, tuple[Any, Any]] = {}  # pyrefly: ignore[explicit-any] — a source beside its encoding
    futures = [
        None
        if spill is not None and spill.done(position)
        else executor.submit(
            _run_slice,
            program,
            document,
            _encode(current.sources, memo, workers_share_fs=shared) if crosses else dict(current.sources),
            crosses,
            solving,
        )
        for position, current in enumerate(slices)
    ]
    for position, (current, future) in enumerate(zip(slices, futures, strict=True)):
        if future is None:
            assert spill is not None, 'a slice is skipped only where a spill holds it'
            yield current.key, spill.read_back(position)
            continue
        with _named_slice(current.key, position, len(slices)):
            answer = future.result()
        answer = replace(answer, frames={kind: _decode(named) for kind, named in answer.frames.items()})
        yield current.key, spill.write(position, current.key, answer) if spill is not None else answer


def _answers(result: Result, program: Program, metrics: Metrics) -> _Answer:
    """One slice's answer, read out of *result*, every declared expression evaluated now.

    A slice with no primal, or with undefined duals, is not a failure: the
    reason ``Result.dual`` gives is carried.
    """
    meta = Record.of(
        result.termination_condition,
        result.objective,
        has_primal=result.has_primal,
        spec_digest=result.spec_digest,
        solved_at=result.solved_at,
        provenance=result.provenance,
    )
    if not result.has_primal:
        return _Answer(meta, metrics)
    frames = {'primal': {name: result.primal(name) for name in program.variables}, 'expression': {}}
    no_expressions: dict[str, str] = {}
    for name in program.expressions:
        try:
            frames['expression'][name] = result.evaluate(name)
        except SpecsolveError as exc:
            no_expressions[name] = str(exc)
    try:
        frames['dual'] = {name: result.dual(name) for name in program.constraints}
    except SpecsolveError as exc:
        return _Answer(meta, metrics, frames, str(exc), no_expressions)
    return _Answer(meta, metrics, frames, None, no_expressions)


def _run_slice(
    program: Program,
    document: Spec,
    encoded: dict[str, Any],  # pyrefly: ignore[explicit-any] — what crossed to the worker
    encode_out: bool,
    call: Mapping[str, Any],  # pyrefly: ignore[explicit-any] — the verb's own keywords, forwarded
) -> _Answer:
    """One slice, start to finish, over plain data; module-level so a remote executor can pickle it."""
    with build(document, _decode(encoded)) as model, model.solve(**call) as result:
        answer = _answers(result, program, _slice_metrics(model.diagnostics(), None))
        if not encode_out:
            return answer
        return replace(answer, frames={kind: _encode(named, {}) for kind, named in answer.frames.items()})


def _key_column(
    axis: Axis | HandBuilt,
    key_name: str | None,
    program: Program,
) -> str:
    """What to call the column holding the slice key; never a column the frames already carry."""
    if key_name is None:
        if not isinstance(axis, Axis):
            raise SpecsolveError(
                'a hand-built axis needs key_name=: a list of slices does not say what its keys are '
                "coordinates of, and 'slice' would be this library naming your axis for you. Pass "
                "key_name='draw', key_name='period', or whatever the keys actually are."
            )
        key_name = axis._key_name()
    refuse_reserved(key_name, f'key_name={key_name!r}')
    if key_name in program.dimensions:
        raise SpecsolveError(
            f'key_name={key_name!r} is a dimension the spec declares, so the slice key would collide '
            f'with a column the frames already carry. Name it something the spec does not use.'
        )
    fixed = tuple(dict.fromkeys(('value', *Record._fields, *Metrics._fields)))
    if key_name in fixed:
        raise SpecsolveError(
            f'key_name={key_name!r} is a column a sweep frame carries ({", ".join(fixed)}), so the slice '
            f'key would replace it rather than sit beside it. Name it something else.'
        )
    return key_name


# ---------------------------------------------------------------------------
# the wire
# ---------------------------------------------------------------------------


def _shares_filesystem(executor: Executor, declared: bool | None) -> bool:
    """Whether *executor*'s workers can read this process's paths; an unknown executor is assumed remote."""
    if declared is not None:
        return declared
    return isinstance(executor, ProcessPoolExecutor)


def _crosses_a_process(executor: Executor) -> bool:
    """Whether a slice's sources have to be encoded to reach *executor*: all but a thread pool's."""
    return not isinstance(executor, ThreadPoolExecutor)


def _encode(
    sources: Mapping[str, Source],
    memo: dict[str, tuple[Any, Any]],  # pyrefly: ignore[explicit-any] — a source beside its encoding
    *,
    workers_share_fs: bool = False,
) -> dict[str, Any]:  # pyrefly: ignore[explicit-any] — a frame crosses as parquet bytes
    """Sources in the shape a worker can be handed.

    A path the workers reach stays a path; one they cannot travels as its own
    bytes. A table becomes parquet ``bytes``; anything else crosses as itself.
    *memo* encodes a source no slice rewrote once.
    """
    out: dict[str, Any] = {}  # pyrefly: ignore[explicit-any] — a frame crosses as parquet bytes
    for name, obj in sources.items():
        if isinstance(obj, (str, Path)) and workers_share_fs:
            out[name] = obj
            continue
        cached = memo.get(name)
        if cached is not None and cached[0] is obj:
            out[name] = cached[1]
            continue
        if isinstance(obj, (str, Path)):
            out[name] = Path(obj).read_bytes()
        elif (table := as_frame(obj)) is None:
            out[name] = obj
        else:
            buffer = io.BytesIO()
            table.collect().write_parquet(buffer, compression=_COMPRESSION)
            out[name] = buffer.getvalue()
        memo[name] = (obj, out[name])
    return out


def _decode(encoded: Mapping[str, Any]) -> dict[str, Any]:  # pyrefly: ignore[explicit-any] — a frame crosses as parquet bytes
    """The inverse of [`_encode`][]; anything not ``bytes`` passes through."""
    return {name: pl.read_parquet(io.BytesIO(v)) if isinstance(v, bytes) else v for name, v in encoded.items()}


# ---------------------------------------------------------------------------
# reading a source without attaching it
# ---------------------------------------------------------------------------


def carries(sources: Mapping[str, Source], dim: str) -> dict[str, pl.LazyFrame]:
    """The sources that carry a column called *dim*, by name; a sweep and its archive both cut these."""
    tables = {name: table for name, obj in sources.items() if (table := as_frame(obj)) is not None}
    return {name: table for name, table in tables.items() if dim in table.collect_schema().names()}


def _coordinates(sources: Mapping[str, Source], dim: str, verb: str) -> tuple[dict[str, pl.LazyFrame], list[Label]]:
    """The sources a slice has to filter, by name, and the coordinates to slice, sorted by value."""
    carrying = carries(sources, dim)
    if not carrying:
        raise DataError(
            f"no source carries a '{dim}' column, so there is nothing to {verb} over. "
            f'EachCoordinate names a column the data has; a span of consecutive coordinates is EachWindow.'
        )
    held = {name: set(table.select(pl.col(dim).unique()).collect()[dim]) for name, table in carrying.items()}
    coordinates = sorted(set().union(*held.values()))
    for name, mine in held.items():
        if missing := sorted(set(coordinates) - mine):
            other = next(o for o, theirs in held.items() if missing[0] in theirs)
            warnings.warn(
                f"'{name}' has no rows for {dim} {missing[0]!r}, which '{other}' has, so the slice at "
                f"{missing[0]!r} builds '{name}' empty and every row of it reads as absent. Supply the rows "
                f'if a value was meant; where the absence is, this is the record of it.',
                SpecsolveWarning,
                stacklevel=4,
            )
    return carrying, coordinates
