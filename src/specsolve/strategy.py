"""Solving strategies: one plan per slice, folded.

A plan cannot contain a loop; a process may loop over plans. A strategy is a
driver above [`specsolve.api`][], built from the public verbs.

    scenario / sweep    ``EachCoordinate('scenario')``                          independent
    myopic pathway      ``EachCoordinate('period', carry={...})``               chained
    rolling horizon     ``EachWindow('snapshot', steps=24, lookahead=24, into='t', carry={...})``
    horizon per case    ``(EachCoordinate('scenario'), EachWindow(..., carry={...}))``  one chain per scenario

The axes are [`specsolve.axes`][] and what a fold returns is
[`specsolve.sweep`][]. The caller-facing rules are [sweeps](https://specsolve.readthedocs.io/en/latest/reference/sweeps/).
"""

from __future__ import annotations

import io
import json
import shutil
import warnings
from collections.abc import Iterator
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import polars as pl

from specsolve.api import build, check
from specsolve.archive_layout import check_the_target, write_archive
from specsolve.axes import (
    Axes,
    Axis,
    EachWindow,
    HandBuilt,
    Slice,
    axis_manifest,
    check_no_index_is_cut,
    checked_axes,
    cut_by,
    sources_with_column,
)
from specsolve.errors import DataError, SpecsolveError, SpecsolveWarning
from specsolve.frames import as_frame
from specsolve.inputs import declared
from specsolve.relational.answer_layout import (
    KINDS,
    METRICS_FILE,
    RECORD_FILE,
    Metrics,
    Record,
    checked_outputs,
    kinds_of,
    write_format,
    write_reasons,
    write_whole,
)
from specsolve.relational.collect import collected
from specsolve.relational.names import VALUE, refuse_reserved
from specsolve.relational.result import Result
from specsolve.sources import numbered, refuse_unknown_start, refuse_unknown_start_word, tidy_sources
from specsolve.sweep import (
    KEYS_FILE,
    MANIFEST_FILE,
    OWNED_FILE,
    WINDOWS_DIR,
    KeyColumns,
    SliceAnswer,
    Spill,
    Sweep,
    key_label,
    sweep_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence

    from mathspec import Spec
    from mathspec.program import Program

    from specsolve.api import Model
    from specsolve.inputs import Buildable, Label, Source
    from specsolve.relational.answer_layout import FrameWriter, Output
    from specsolve.relational.result import Diagnostics, Start


#: The codec a frame is written with to cross a process.
_COMPRESSION = 'zstd'


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
class _Carry:
    """One axis's carry: where its label sits in a slice's key, and its rules by parameter.

    The value is handed on from the last slice that holds a label of this
    axis to every slice of the next label under the same outer keys, and
    reset to the seed where an outer key changes.
    """

    depth: int
    rules: Mapping[str, _CarryRule]

    def hands_on(self, current: Slice, following: Slice | None) -> bool:
        """Whether *current* is the last slice of its label of this axis, with a next label after it."""
        if following is None:
            return False
        parent = slice(0, self.depth)
        return following.key[parent] == current.key[parent] and following.key[self.depth] != current.key[self.depth]


def _carries(program: Program, axes: Axes | None, first: Mapping[str, Source]) -> tuple[_Carry, ...]:
    """Each axis's carry, outer first, checked against the plan and the first slice's sources before any is read.

    Raises:
        SpecsolveError: A carry that cannot line up, has no seed, collapses a
            dimension its axis does not advance along, or a parameter two
            axes carry.
    """
    out: list[_Carry] = []
    seen: dict[str, str] = {}
    for depth, axis in enumerate(axes or ()):
        rules = {
            parameter: _CarryRule.resolved(program, parameter, variable) for parameter, variable in axis.carry.items()
        }
        for parameter in rules:
            if parameter in seen:
                raise SpecsolveError(
                    f'carry writes {parameter!r} from both {seen[parameter]} and {axis!r}, so a slice would take '
                    f'two values for it. Carry each parameter on one axis.'
                )
            seen[parameter] = repr(axis)
        _check_the_carry(rules, axis, first)
        if rules:
            out.append(_Carry(depth, rules))
    return tuple(out)


def _archiving(archive: str | Path | None, axes: Axes | None, *, keep_windows: bool) -> tuple[Path, Axes] | None:
    """Where the archive goes and the axes that re-run it, or ``None`` for no archive; *axes* is ``None`` for slices written by hand."""
    if keep_windows and archive is None:
        raise SpecsolveError(
            'keep_windows=True says what an archive keeps beside the answer, and there is no archive=. The '
            'sweep in memory reads its windows through per_window=True already. Pass archive=, or drop '
            'keep_windows.'
        )
    if keep_windows and not (axes is not None and any(isinstance(each, EachWindow) for each in axes)):
        raise SpecsolveError(
            "keep_windows=True keeps an EachWindow sweep's frames per window beside its answer, and this axis "
            'does not cut windows: its answer already is one frame per slice, which the archive holds. Drop '
            'keep_windows.'
        )
    if archive is None:
        return None
    if axes is None:
        raise SpecsolveError(
            'archive= takes a sweep cut by EachCoordinate or EachWindow, which say how one set of sources '
            'was cut and so how the archive can be re-run. A hand-built list is a set of sources per '
            'slice, which are unrelated questions — archive one solve each.'
        )
    out = Path(archive)
    check_the_target(out)
    return out, axes


def solve_over(
    spec: Buildable,
    sources: Mapping[str, Source],
    axis: Axis | Axes | HandBuilt,
    *,
    key_name: str | None = None,
    executor: Executor | None = None,
    workers_share_fs: bool | None = None,
    solver_options: Mapping[str, object] | None = None,
    record_options: Sequence[str] | None = None,
    solver_name: str = 'highs',
    spill_to: str | Path | None = None,
    archive: str | Path | None = None,
    keep_windows: bool = False,
    outputs: Iterable[Output] = (),
    start: Sweep | Result | Start | Literal['previous'] | None = 'previous',
) -> Sweep:
    """Solve *spec* once per slice of *axis* and fold the answers together.

    The rules are [sweeps](https://specsolve.readthedocs.io/en/latest/reference/sweeps/).

    Args:
        spec: As [`check`][specsolve.api.check] takes it. Parsed once, whichever
            executor runs the slices.
        sources: As [`build`][specsolve.api.build] takes them, every shape
            included; the axis filters the parameters and relations that carry
            it and passes the rest through.
        axis: [`EachCoordinate`][], [`EachWindow`][], a tuple of them, outer
            first, or a list of ``(key, sources)`` written by hand. A tuple
            cuts with each axis in turn, and any of them may be windows. An
            axis's ``carry=``
            chains its slices; the slices that share the keys of every axis
            outside the outermost one that carries are a chain, and the answer
            carries one key column per axis.
        key_name: What to call the slice column; a class axis names its own,
            a hand-built list has to be told.
        executor: Any `concurrent.futures.Executor`; ``None`` runs the
            slices in order on one model. Where an axis carries, it runs the
            chains concurrently, each one in order, and otherwise every
            slice. A process pool must be ``spawn``
            or ``forkserver`` — a forked worker hangs.
        workers_share_fs: Whether the executor's workers can read this
            process's paths. Decided for the stdlib pools; anything else is
            assumed not to, and paths travel as bytes.
        solver_options: As [`solve`][specsolve.api.Model.solve] takes them.
        record_options: As [`solve`][specsolve.api.Model.solve] takes them,
            reaching every slice.
        solver_name: As [`solve`][specsolve.api.Model.solve] takes it.
        spill_to: A directory each slice's frames are written to as the fold
            goes, so the sweep holds one slice in memory however many there
            are; [`Sweep.scan`][] reads a name off it lazily. A directory
            holds one sweep: the same sweep run at it again does not solve the
            slices already there, which is how an interrupted sweep resumes.
        archive: Where to write the model, the sources the sweep was cut from,
            the axis that cut them and every slice's answer, so that
            ``sps.load_archive`` gives all four back and the sweep runs again
            from the file alone. A ``.zip`` suffix packs it into one file;
            anything else is a directory. The answer is one file per name at
            ``answer/<kind>/<name>.parquet``, as for a single solve; a name
            with no answer, such as a quantity reduced over an EachWindow
            sweep's windowed dimension, is left out and
            ``answer/reasons.parquet`` says why. Beside *spill_to* the archive
            packs the spill, so a sweep too large to hold is archived without
            being held; the archive is a second copy of the answers on disk.
            A sliced source is archived whole, and one number over
            a window's local index as a table over the axis. Refused for a
            hand-built axis, which is a set of sources per slice: archive one
            solve each.
        keep_windows: Also archive an [`EachWindow`][] sweep's frames per
            window, lookahead rows included, under ``answer/windows/``, so
            that ``per_window=True`` reads off the archive. Refused for any
            other axis, and without *archive*.
        outputs: As [`solve`][specsolve.api.Model.solve] takes them, reaching
            every slice. The sweep, its spill and its archive carry these and
            nothing else; a *spill_to* directory solved with others is
            refused rather than resumed.
        start: What each slice starts from, as
            [`solve`][specsolve.api.Model.solve] takes it: an earlier answer,
            an earlier sweep, or a [`Start`][specsolve.types.Start] of tables.
            Each table is cut by the axis as a source is: one that carries the
            sliced dimension, or a hand-built axis's key column, gives each
            slice its own rows, and one without reaches every slice whole. An
            earlier sweep is its answer, so each slice starts from the earlier
            slice of its key, and each window from the hours it covers.
            ``'previous'``, the default, carries each slice on from the one
            before it on the model updated in place, as
            [`solve`][specsolve.api.Model.solve] does. Under an executor the
            first slice of each task starts from nothing: every slice, or
            every chain where an axis carries. ``None`` starts every slice
            from nothing.

    Returns:
        The sweep, which reads its answer.

    Raises:
        SpecsolveError: Before a slice is taken: a tuple of axes with two
            windows over one local index, or two axes over one dimension; a carry that
            cannot line up, has no seed, collapses a dimension its axis does
            not advance along, writes a parameter another axis carries too, or
            is asked with an executor on a sweep of one chain; a key that collides with a column the frames carry; an
            axis the program does not allow; a *spill_to* directory holding
            another sweep; *keep_windows* without *archive* or on an axis
            that does not cut windows; *outputs* that
            [`solve`][specsolve.api.Model.solve] refuses; a *start* word other
            than ``'previous'`` —
            each answerable from the declarations alone; keys of more than one
            type, or two keys of one text; a *start* table over an EachWindow
            sweep's local index alone, or one
            [`solve`][specsolve.api.Model.solve] refuses.
        DataError: No source carries the axis, an index of another
            dimension carries it, or the axis produced no slices.

    Warns:
        SpecsolveWarning: A source carrying the axis that is short of a
            coordinate another has — that slice builds it empty — a position
            the model counts, which every window restarts, or a *start* that
            leaves a slice no row, which starts that slice from nothing.
    """
    document = declared(spec)
    sources = _materialised(sources)
    axes = checked_axes(cast('Axis | Axes', axis)) if _is_axes(axis) else None
    archiving = _archiving(archive, axes, keep_windows=keep_windows)
    asked = checked_outputs(outputs)
    program = check(document)
    carries = _carries(program, axes, sources)
    key_names = _key_columns(axes, key_name, program)

    if axes is not None:
        for each in axes:
            check_no_index_is_cut(program, sources, each)
            each._check_the_program(program, sources)
        slices, stitch = cut_by(axes, sources, key_names)
    else:
        slices = [Slice((key,), named) for key, named in cast('HandBuilt', axis)]
        stitch = None
    if not slices:
        raise DataError('the axis produced no slices')
    depth = carries[0].depth if carries else len(key_names) - 1
    slices = [current._replace(chain=current.key[:depth]) for current in slices]
    one_chain = len({current.chain for current in slices}) == 1
    if carries and executor is not None and one_chain:
        raise SpecsolveError(
            'carry and executor are mutually exclusive on a sweep of one chain: a carried value makes slice '
            "i+1 depend on slice i's answer, so the slices cannot run concurrently, and an executor runs "
            'chains concurrently, one per combination of the outer axes. Drop the executor, or drop the carry.'
        )
    solving = {
        'solver_name': solver_name,
        'solver_options': dict(solver_options or {}) or None,
        'record_options': record_options,
        'outputs': asked,
    }
    keys = [current.key for current in slices]
    columns = KeyColumns.of(keys, key_names)
    starts = _slice_starts(start, axes, slices, key_names, program)
    spill = None if spill_to is None else Spill.opened(spill_to, columns, keys, stitch, asked, document)
    if executor is None:
        answered = _serially(program, document, slices, solving, carries, spill, starts)
    else:
        answered = _pooled(executor, workers_share_fs, program, document, slices, solving, carries, spill, starts)
    folded = Sweep._folded(columns, stitch, answered, spill, asked, document)
    if spill is not None:
        write_reasons(spill.directory, folded._no_duals, folded._absent, write_whole)
    if archiving is not None:
        out, cut = archiving
        _archive_the_sweep(out, document, program, cut, sources, folded, slices[0].sources, keep_windows)
    return folded


def _is_axes(axis: Axis | Axes | HandBuilt) -> bool:
    """Whether *axis* is one axis or a tuple of axes, rather than slices written by hand.

    A tuple holding an axis is a tuple of axes, so one that mixes in anything
    else is refused by [`checked_axes`][] rather than read as slices.
    """
    return isinstance(axis, Axis) or (isinstance(axis, tuple) and any(isinstance(each, Axis) for each in axis))


def _slice_starts(
    start: Sweep | Result | Start | Literal['previous'] | None,
    axes: Axes | None,
    slices: Sequence[Slice],
    key_names: Sequence[str],
    program: Program,
) -> Sequence[Start | Literal['previous'] | None]:
    """What each slice starts from: its cut of *start*, as it cut its sources, or the slice before it.

    Every cut is taken and checked here, before a slice is built, so a start
    that cannot start a slice stops the sweep before anything is solved; it
    also lets a slice solved in another process take its start as data. Each
    axis cuts a table that carries its column, and a table without any
    reaches every slice whole.

    Raises:
        SpecsolveError: A word other than ``'previous'``; a name
            [`refuse_unknown_start`][specsolve.sources.refuse_unknown_start]
            refuses; or a table over an EachWindow sweep's local index alone.

    Warns:
        SpecsolveWarning: A slice the cut leaves nothing, which starts from
            nothing.
    """
    refuse_unknown_start_word(start)
    if start is None or isinstance(start, str):
        return [start] * len(slices)
    columns = [each.dim for each in axes] if axes is not None else [key_names[-1]]
    tables = _start_tables(start, columns[-1])
    refuse_unknown_start(cast('Start', tables), program)
    for axis in axes or ():
        if isinstance(axis, EachWindow) and (local := _over_the_local_index(tables, axis)):
            raise SpecsolveError(
                f"start= gives {local} over the windows' local index {axis.into!r} and not over {axis.dim!r}, so "
                f'one table would lay the same {axis.dim!r} onto every window. Write it over {axis.dim!r}, as a '
                f'source is, and each window takes the rows it covers.'
            )
    starts: list[Start | None] = []
    empty: list[Label] = []
    for current in slices:
        cut = current.cut or partial(_one_key, key_names[-1], current.key[-1])
        given: dict[str, dict[str, Source]] = {}
        for reader, named in tables.items():
            pieces = {name: _cut_one(obj, cut) for name, obj in named.items()}
            if held := {name: piece for name, piece in pieces.items() if piece is not None}:
                given[reader] = held
        if not given:
            empty.append(key_label(current.key))
        starts.append(cast('Start', given) if given else None)
    if empty:
        warnings.warn(
            f'start= gives the slices {empty} no row, so they start from nothing: no table carries their '
            f'{columns}, and none reaches every slice. The answer is the same; only the time it takes changes.',
            SpecsolveWarning,
            stacklevel=3,
        )
    return starts


def _cut_one(obj: Source, cut: Callable[[pl.LazyFrame], pl.LazyFrame]) -> Source | None:
    """*obj* as one slice takes it: a table cut, read into memory to cross a process, or ``None`` where the cut leaves no row.

    A shape that is not a table, such as one number, carries no column to cut
    on and reaches the slice whole, as a table that carries no axis's column does.
    """
    table = as_frame(obj)
    if table is None:
        return obj
    piece = cut(table).pipe(collected)
    return piece if piece.height else None


def _start_tables(start: Sweep | Result | Start, column: str) -> dict[str, dict[str, Source]]:
    """*start* as tables keyed as [`Start`][specsolve.types.Start] is, an earlier sweep's keyed by *column*.

    A sweep's slices are keyed by its own key column, which for one cut by
    coordinate holds the coordinates a sweep cut by *column* cuts.
    """
    if isinstance(start, Result):
        return {
            reader: {name: frame.pipe(collected) for name, frame in frames.items()}
            for reader, frames in start._start().items()
        }
    if isinstance(start, Sweep):
        keyed = start._stitch is None and start.key_name != column
        return {
            reader: {name: table.rename({start.key_name: column}) if keyed else table for name, table in named.items()}
            for reader, named in cast('Mapping[str, Mapping[str, pl.DataFrame]]', start._start()).items()
        }
    return {reader: dict(named) for reader, named in cast('Mapping[str, Mapping[str, Source]]', start).items()}


def _over_the_local_index(tables: Mapping[str, Mapping[str, Source]], axis: EachWindow) -> list[str]:
    """The names of *tables* over *axis*'s local index and not its sliced dimension, which no window can place."""
    over = {
        name: frame.collect_schema().names()
        for named in tables.values()
        for name, obj in named.items()
        if (frame := as_frame(obj)) is not None
    }
    return sorted(name for name, columns in over.items() if axis.into in columns and axis.dim not in columns)


def _one_key(key_name: str, key: Label, table: pl.LazyFrame) -> pl.LazyFrame:
    """*table*'s rows of the hand-built slice *key*, without the key column; a table without it, whole."""
    if key_name not in table.collect_schema().names():
        return table
    return table.filter(pl.col(key_name) == key).drop(key_name)


def _materialised(sources: Mapping[str, Source]) -> dict[str, Source]:
    """*sources* with each one-shot iterator read into a list.

    Every slice and the archive read a source the axis does not cut, so an
    iterator would be spent after the first read.
    """
    return {name: list(obj) if isinstance(obj, Iterator) else obj for name, obj in sources.items()}


def _archive_the_sweep(
    out: Path,
    spec: Spec,
    program: Program,
    axes: Axes,
    sources: Mapping[str, Source],
    folded: Sweep,
    one_slice: Mapping[str, Source],
    keep_windows: bool,
) -> None:
    """Write the sweep's question and its answer to *out*.

    Each source's tidy table comes from *one_slice*, except the ones
    [`_uncut`][] and [`_spread_over_the_axis`][] give.
    """
    manifest = axis_manifest(axes if len(axes) > 1 else axes[0])
    tidied = numbered(program, tidy_sources(program, one_slice))
    sliced = {name: table for each in axes for name, table in sources_with_column(sources, each.dim).items()}
    cut = {name: _uncut(program, axes, name, table) for name, table in sliced.items()}
    spread = {
        name: table
        for each in axes
        if isinstance(each, EachWindow)
        for name, table in _spread_over_the_axis(program, each, sources, tidied).items()
    }
    held = {**tidied, **spread, **cut}
    tables = {name: held[name] for name in sources}
    write_archive(out, spec, tables, axis=manifest, answer=partial(_the_answer, folded, keep_windows=keep_windows))


def _the_answer(sweep: Sweep, under: Path, write: FrameWriter, *, keep_windows: bool) -> None:
    """The ``answer/`` an archive holds for *sweep*, laid out under *under*, each table written with *write*.

    One file per kind and name, as the readers return it; a name an
    EachWindow sweep cannot stitch has none, and ``reasons.parquet`` says
    why. The record and metrics take a single solve's columns, so
    ``keys.parquet`` beside them gives the keys their type back.
    ``sweep.json`` records *keep_windows*: a zip holds files only, so an empty
    ``windows/`` is not there to say so. What each window owns is kept in
    ``windows/`` beside them, the answer being stitched already: copied off
    the spill where the sweep has one, so a sweep too large to hold streams,
    and written from memory where it is held.
    """
    write_format(under, sweep._outputs)
    manifest = sweep_manifest(sweep._key_columns(), sweep._key_rows(), sweep._stitch, sweep._outputs)
    (under / MANIFEST_FILE).write_text(json.dumps({**manifest, 'windows': keep_windows}))
    write(sweep.record.select(sweep.key_names), under / KEYS_FILE)
    write(sweep.record.drop(sweep.key_names), under / RECORD_FILE)
    write(sweep.metrics.drop(sweep.key_names), under / METRICS_FILE)
    absent = {kind: dict(names) for kind, names in sweep._absent.items()}
    for kind in KINDS:
        for name, frame in sweep._slices.get(kind, {}).items():
            if why := sweep._unstitchable(frame):
                absent.setdefault(kind, {})[name] = why
                continue
            write(sweep._answered(frame, per_window=False), under / kind / f'{name}.parquet')
    write_reasons(under, sweep._no_duals, absent, write)
    if keep_windows:
        _the_windows(sweep, under / WINDOWS_DIR, write)


def _the_windows(sweep: Sweep, under: Path, write: FrameWriter) -> None:
    """What each window owns, and its frames as the spill lays them out, under *under*, each table written with *write*."""
    assert sweep._stitch is not None, 'only an EachWindow sweep keeps its windows'
    under.mkdir()
    write(sweep._stitch.owned, under / OWNED_FILE)
    if sweep._spill is None:
        windows = Spill(under, sweep._key_columns())
        for position, key, frames in sweep._frames_by_slice():
            windows.write_frames(position, key, frames, write)
        return
    for kind in KINDS:
        if (sweep._spill.directory / kind).is_dir():
            shutil.copytree(
                sweep._spill.directory / kind,
                under / kind,
                copy_function=lambda source, target: write(pl.scan_parquet(source), Path(target)),
            )


def _uncut(program: Program, axes: Axes, name: str, table: pl.LazyFrame) -> pl.LazyFrame:
    """A source the axes cut, as its tidy columns with each axis column it carries first, outer first.

    A window's local index is not a column of the uncut table: the axis
    column stands where it would be.
    """
    declared = [*program.parameters[name].dims, VALUE] if name in program.parameters else program.relations[name].roles
    locals_ = {each.into for each in axes if isinstance(each, EachWindow)}
    held = table.collect_schema().names()
    carried = [each.dim for each in axes if each.dim in held]
    return table.select(list(dict.fromkeys([*carried, *(column for column in declared if column not in locals_)])))


def _spread_over_the_axis(
    program: Program,
    axis: Axis,
    sources: Mapping[str, Source],
    tidied: Mapping[str, pl.LazyFrame],
) -> dict[str, pl.LazyFrame]:
    """Each parameter given as one number over a window's local index, as a table over the axis.

    The local index has one label per coordinate a window holds, so no one
    window's table is what a window of another length read; over the axis,
    each window cuts what its own solve read. Every other source attaches as
    one table in every slice, so the archive holds the first slice's.
    """
    if not isinstance(axis, EachWindow):
        return {}
    numbers = [
        name
        for name, declared in program.parameters.items()
        if axis.into in declared.dims and isinstance(sources[name], (bool, int, float))
    ]
    windowed = sources_with_column(sources, axis.dim)
    coordinates = pl.concat([table.select(axis.dim) for table in windowed.values()], how='vertical_relaxed')
    coordinates = coordinates.unique().sort(axis.dim)
    return {
        name: coordinates.join(tidied[name].drop(axis.into).unique(maintain_order=True), how='cross')
        for name in numbers
    }


def _check_the_carry(
    plan: Mapping[str, _CarryRule],
    axis: Axis,
    first: Mapping[str, Source],
) -> None:
    """Refuse a carry with no seed, or one that collapses a dimension other than [`EachWindow.into`][].

    Reads no source: the seed is a key of *first*, and the owned dimension is
    that of *axis*, the axis that carries.
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
    slices: Sequence[Slice],
    solving: Mapping[str, Any],  # pyrefly: ignore[explicit-any] — the verb's own keywords, forwarded
    carries: Sequence[_Carry],
    spill: Spill | None,
    starts: Sequence[Start | Literal['previous'] | None],
) -> Generator[tuple[tuple[Label, ...], SliceAnswer], None, None]:
    """Each slice's answer, off one model updated in place.

    Each axis's carry reaches the slices of its next label, and resets to the
    seed where an outer key changes. A slice naming other sources than the
    last is rebuilt, since ``update`` is partial. A generator because slice ``i+1``'s carry is read from slice
    ``i``'s frames after the yield; the caller closes it to release the model.
    A slice the spill holds is read back, and one solved here is written
    before it is yielded.
    """
    model: Model | None = None
    named: frozenset[str] | None = None
    state: dict[int, dict[str, pl.DataFrame]] = {}
    variables = {rule.variable for carry in carries for rule in carry.rules.values()}
    try:
        for position, current in enumerate(slices):
            if position:
                before_this = slices[position - 1]
                state = {depth: held for depth, held in state.items() if current.key[:depth] == before_this.key[:depth]}
            if spill is not None and spill.done(position):
                answer = spill.read_back(position)
                primals = spill.written('primal', position, variables)
                yield current.key, answer
                state.update(_carried(carries, primals, current, position, slices, answer))
                continue
            sources = {
                **current.sources,
                **{name: value for depth in sorted(state) for name, value in state[depth].items()},
            }
            names = frozenset(sources)
            with _named_slice(current.key, position, len(slices)):
                if model is not None and names == named:
                    before = model.diagnostics()
                    model.update(sources)
                else:
                    if model is not None:
                        model.close()
                    model, named, before = build(document, sources), names, None
                result = model.solve(**solving, start=starts[position])
                answer = _answers(result, program, _slice_metrics(model.diagnostics(), before), solving['outputs'])
            primals = answer.frames.get('primal', {})
            if spill is not None:
                answer = spill.write(position, current.key, answer)
            yield current.key, answer
            state.update(_carried(carries, primals, current, position, slices, answer))
    finally:
        if model is not None:
            model.close()


def _carried(
    carries: Sequence[_Carry],
    primals: Mapping[str, pl.DataFrame],
    current: Slice,
    position: int,
    slices: Sequence[Slice],
    answer: SliceAnswer,
) -> dict[int, dict[str, pl.DataFrame]]:
    """What each axis hands the slices of its next label, read out of *primals*, by the axis's depth.

    Only an axis whose label ends with *current* hands anything on, so a slice
    in the middle of an outer label hands on its own axis's carry alone.

    Raises:
        SpecsolveError: *current* left nothing to hand on.
    """
    following = slices[position + 1] if position + 1 < len(slices) else None
    handing = [carry for carry in carries if carry.hands_on(current, following)]
    if not handing:
        return {}
    if not primals:
        assert following is not None, 'only a slice with one after it hands anything on'
        variables = sorted({rule.variable for carry in handing for rule in carry.rules.values()})
        raise SpecsolveError(
            f'slice {key_label(current.key)!r} ({position + 1} of {len(slices)}) terminated '
            f'{answer.meta.termination_condition}, so slice {key_label(following.key)!r} has no '
            f'{variables} to start from. A carried sweep '
            f'stops at the first slice that leaves nothing to carry; the {position} before it solved.'
        )
    return {
        carry.depth: {
            p: rule.value_from(primals, p, key_label(current.key), current.owns[carry.depth])
            for p, rule in carry.rules.items()
        }
        for carry in handing
    }


@contextmanager
def _named_slice(key: tuple[Label, ...], position: int, count: int) -> Generator[None, None, None]:
    """Whatever a slice raises leaves naming the slice, as a note, so the error stays the engine's own."""
    try:
        yield
    except Exception as exc:
        exc.add_note(f'in slice {key_label(key)!r} ({position + 1} of {count})')
        raise


def _pooled(
    executor: Executor,
    workers_share_fs: bool | None,
    program: Program,
    document: Spec,
    slices: Sequence[Slice],
    solving: Mapping[str, Any],  # pyrefly: ignore[explicit-any] — the verb's own keywords, forwarded
    carries: Sequence[_Carry],
    spill: Spill | None,
    starts: Sequence[Start | Literal['previous'] | None],
) -> Generator[tuple[tuple[Label, ...], SliceAnswer], None, None]:
    """The same, from slices or chains run independently and possibly elsewhere.

    Where an axis carries, each chain is one task, its slices run in order
    on one model; otherwise each slice is. Yielded in slice
    order, never completion order. A built model cannot cross a process, so
    each task builds its own. A task the spill holds every slice of is never
    submitted, and one that comes back is written here, by the process that
    owns the directory: a chain the spill holds only part of runs whole again.
    """
    crosses = _crosses_a_process(executor)
    shared = _shares_filesystem(executor, workers_share_fs)
    memo: dict[str, tuple[Any, Any]] = {}  # pyrefly: ignore[explicit-any] — a source beside its encoding
    tasks = _chains(slices) if carries else [[position] for position in range(len(slices))]
    futures = []
    for positions in tasks:
        if spill is not None and all(spill.done(position) for position in positions):
            futures.append(None)
            continue
        entries = [
            (
                slices[position].key,
                _encode(slices[position].sources, memo, workers_share_fs=shared)
                if crosses
                else dict(slices[position].sources),
                slices[position].owns,
                slices[position].chain,
            )
            for position in positions
        ]
        given = [starts[position] for position in positions]
        futures.append(executor.submit(_run_chain, program, document, entries, crosses, solving, carries, given))
    for positions, future in zip(tasks, futures, strict=True):
        if future is None:
            assert spill is not None, 'a task is skipped only where a spill holds every slice of it'
            for position in positions:
                yield slices[position].key, spill.read_back(position)
            continue
        with _named_slice(slices[positions[0]].key, positions[0], len(slices)):
            answers = future.result()
        for position, answer in zip(positions, answers, strict=True):
            key = slices[position].key
            answer = replace(answer, frames={kind: _decode(named) for kind, named in answer.frames.items()})
            yield key, spill.write(position, key, answer) if spill is not None else answer


def _chains(slices: Sequence[Slice]) -> list[list[int]]:
    """The positions of each chain's slices, in slice order; a chain's slices are consecutive, as the axes cut them."""
    out: list[list[int]] = []
    for position, current in enumerate(slices):
        if out and slices[out[-1][-1]].chain == current.chain:
            out[-1].append(position)
        else:
            out.append([position])
    return out


def _answers(result: Result, program: Program, metrics: Metrics, outputs: frozenset[Output]) -> SliceAnswer:
    """One slice's answer, read out of *result*, every declared expression evaluated now.

    It carries the *outputs* the sweep asked for and no others, so a basis
    read only to start the next slice from is not kept. A slice with no
    primal, or with undefined duals, is not a failure: the reason
    ``Result.dual`` gives is carried.
    """
    meta = Record.of(
        result.termination_condition,
        result.objective,
        has_primal=result.has_primal,
        solved_at=result.solved_at,
        provenance=result.provenance,
    )
    if not result.has_primal:
        return SliceAnswer(meta, metrics)
    frames: dict[str, dict[str, pl.DataFrame]] = {
        'primal': {name: result.primal(name) for name in program.variables},
        'expression': {},
    }
    for kind in kinds_of(outputs):
        if laid := (result._outputs or {}).get(kind):
            frames[kind] = {name: frame.pipe(collected) for name, frame in laid.items()}
    no_expressions: dict[str, str] = {}
    for name in program.expressions:
        try:
            frames['expression'][name] = result.evaluate(name)
        except SpecsolveError as exc:
            no_expressions[name] = str(exc)
    try:
        frames['dual'] = {name: result.dual(name) for name in program.constraints}
    except SpecsolveError as exc:
        return SliceAnswer(meta, metrics, frames, str(exc), no_expressions)
    return SliceAnswer(meta, metrics, frames, None, no_expressions)


def _run_chain(
    program: Program,
    document: Spec,
    entries: Sequence[tuple[tuple[Label, ...], dict[str, Any], tuple[int | None, ...], tuple[Label, ...]]],  # pyrefly: ignore[explicit-any] — what crossed to the worker
    encode_out: bool,
    call: Mapping[str, Any],  # pyrefly: ignore[explicit-any] — the verb's own keywords, forwarded
    carries: Sequence[_Carry],
    starts: Sequence[Start | Literal['previous'] | None],
) -> list[SliceAnswer]:
    """One task's slices, start to finish, over plain data, as [`_serially`][] runs them; module-level so a remote executor can pickle it.

    A task of one slice is a chain of one, so a slice of an unchained sweep
    runs here too.
    """
    slices = [Slice(key, _decode(encoded), owns, chain=chain) for key, encoded, owns, chain in entries]
    answers = [answer for _, answer in _serially(program, document, slices, call, carries, None, starts)]
    if not encode_out:
        return answers
    return [
        replace(answer, frames={kind: _encode(named, {}) for kind, named in answer.frames.items()})
        for answer in answers
    ]


def _key_columns(axes: Axes | None, key_name: str | None, program: Program) -> tuple[str, ...]:
    """What to call the columns holding the slice key, one per axis, outer first; never a column the frames already carry.

    *axes* is ``None`` for slices written by hand.
    """
    if axes is None:
        if key_name is None:
            raise SpecsolveError(
                'a hand-built axis needs key_name=: a list of slices does not say what its keys are '
                "coordinates of, and 'slice' would be this library naming your axis for you. Pass "
                "key_name='draw', key_name='period', or whatever the keys actually are."
            )
        return (_key_column(key_name, program),)
    if key_name is not None and len(axes) > 1:
        raise SpecsolveError(
            f'key_name={key_name!r} names the key of one axis, and this sweep has {len(axes)}, each of which names '
            f'its own key column: {[each._key_name() for each in axes]}. Drop key_name.'
        )
    names = (key_name,) if key_name is not None else tuple(each._key_name() for each in axes)
    return tuple(_key_column(name, program) for name in names)


def _key_column(key_name: str, program: Program) -> str:
    """*key_name*, refused where it is a column the frames already carry."""
    refuse_reserved(key_name, f'key_name={key_name!r}')
    if key_name in program.dimensions:
        raise SpecsolveError(
            f'key_name={key_name!r} is a dimension the spec declares, so the slice key would collide '
            f'with a column the frames already carry. Name it something the spec does not use.'
        )
    fixed = tuple(dict.fromkeys((VALUE, *Record._fields, *Metrics._fields)))
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
            table.pipe(collected).write_parquet(buffer, compression=_COMPRESSION)
            out[name] = buffer.getvalue()
        memo[name] = (obj, out[name])
    return out


def _decode(encoded: Mapping[str, Any]) -> dict[str, Any]:  # pyrefly: ignore[explicit-any] — a frame crosses as parquet bytes
    """The inverse of [`_encode`][]; anything not ``bytes`` passes through."""
    return {name: pl.read_parquet(io.BytesIO(v)) if isinstance(v, bytes) else v for name, v in encoded.items()}
