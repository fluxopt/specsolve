"""Solving strategies: one plan per slice, folded.

A plan cannot contain a loop; a process may loop over plans. A strategy is a
driver above [`specsolve.api`][], built from the public verbs.

    scenario / sweep    ``EachCoordinate('scenario')``            independent
    myopic pathway      ``EachCoordinate('period')``              + ``carry``
    rolling horizon     ``EachWindow('snapshot', steps=24, lookahead=24, into='t')``  + ``carry``

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
from specsolve.archive_layout import ANSWER_DIR, beside, check_the_target, write_archive
from specsolve.axes import Axis, EachWindow, HandBuilt, Slice, axis_manifest, check_no_index_is_cut, sources_with_column
from specsolve.errors import DataError, SpecsolveError, SpecsolveWarning
from specsolve.frames import as_frame
from specsolve.inputs import declared
from specsolve.relational.answer_layout import (
    KINDS,
    METRICS_FILE,
    RECORD_FILE,
    VALUE,
    Metrics,
    Record,
    checked_outputs,
    kinds_of,
    refuse_reserved,
    write_format,
    write_reasons,
    write_whole,
)
from specsolve.relational.collect import collected
from specsolve.relational.result import Result
from specsolve.sources import numbered, refuse_unknown_start, refuse_unknown_start_word, tidy_sources
from specsolve.sweep import (
    KEYS_FILE,
    MANIFEST_FILE,
    OWNED_FILE,
    WINDOWS_DIR,
    SliceAnswer,
    Spill,
    Sweep,
    one_key_type,
    scan_sweep,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence

    from mathspec import Spec
    from mathspec.program import Program

    from specsolve.api import Model
    from specsolve.inputs import Buildable, Label, Source
    from specsolve.relational.answer_layout import Output
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
            [`solve`][specsolve.api.Model.solve] does, and under an executor
            starts every slice from nothing. ``None`` starts every slice from
            nothing.

    Returns:
        The sweep, which reads its answer.

    Raises:
        SpecsolveError: Before a slice is taken: a carry that cannot line up,
            has no seed, collapses a dimension the axis does not advance
            along, or is asked together with an executor; a key that collides
            with a column the frames carry; an axis the program does not
            allow; a *spill_to* directory holding another sweep;
            *keep_windows* without *archive* or on an axis that does not cut
            windows; *outputs* that [`solve`][specsolve.api.Model.solve]
            refuses; a *start* word other than ``'previous'`` — each answerable from the declarations alone; keys of more than one
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
    if carry and executor is not None:
        raise SpecsolveError(
            'carry and executor are mutually exclusive: a carried value makes slice i+1 depend on '
            "slice i's answer, so the slices cannot run concurrently. Drop the executor, or drop the carry."
        )
    document = declared(spec)
    sources = _materialised(sources)
    archiving = _archiving(archive, axis, keep_windows=keep_windows)
    asked = checked_outputs(outputs)
    program = check(document)
    plan = {p: _CarryRule.resolved(program, p, v) for p, v in (carry or {}).items()}
    key_name = _key_column(axis, key_name, program)

    if isinstance(axis, Axis):
        _check_the_carry(plan, axis, sources)
        check_no_index_is_cut(program, sources, axis)
        axis._check_the_program(program, sources)
        slices, stitch = axis._slice(sources, key_name)
    else:
        slices = [Slice(*entry) for entry in axis]
        stitch = None
        _check_the_carry(plan, axis, slices[0].sources if slices else {})
    if not slices:
        raise DataError('the axis produced no slices')
    solving = {
        'solver_name': solver_name,
        'solver_options': dict(solver_options or {}) or None,
        'record_options': record_options,
        'outputs': asked,
    }
    keys = [current.key for current in slices]
    key_dtype = one_key_type(keys, key_name)
    starts = _slice_starts(start, axis, slices, key_name, program)
    spill = None if spill_to is None else Spill.opened(spill_to, key_name, keys, key_dtype, stitch, asked)
    if executor is None:
        answered = _serially(program, document, slices, solving, plan, spill, starts)
    else:
        answered = _pooled(executor, workers_share_fs, program, document, slices, solving, spill, starts)
    folded = Sweep._folded(key_name, stitch, answered, spill, key_dtype, asked)
    if spill is not None:
        write_reasons(spill.directory, folded._no_duals, folded._absent)
    if archiving is not None:
        out, cut = archiving
        _archive_the_sweep(
            out, document, program, cut, dict(carry or {}), sources, folded, slices[0].sources, keep_windows
        )
    return folded


def _slice_starts(
    start: Sweep | Result | Start | Literal['previous'] | None,
    axis: Axis | HandBuilt,
    slices: Sequence[Slice],
    key_name: str,
    program: Program,
) -> Sequence[Start | Literal['previous'] | None]:
    """What each slice starts from: its cut of *start*, as it cut its sources, or the slice before it.

    Every cut is taken and checked here, before a slice is built, so a start
    that cannot start a slice stops the sweep before anything is solved; it
    also lets a slice solved in another process take its start as data. A
    table without the sliced column reaches every slice whole.

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
    column = axis.dim if isinstance(axis, Axis) else key_name
    tables = _start_tables(start, column)
    refuse_unknown_start(cast('Start', tables), program)
    if isinstance(axis, EachWindow) and (local := _over_the_local_index(tables, axis)):
        raise SpecsolveError(
            f"start= gives {local} over the windows' local index {axis.into!r} and not over {axis.dim!r}, so one "
            f'table would lay the same {axis.dim!r} onto every window. Write it over {axis.dim!r}, as a source is, '
            f'and each window takes the rows it covers.'
        )
    starts: list[Start | None] = []
    empty: list[Label] = []
    for current in slices:
        cut = current.cut or partial(_one_key, key_name, current.key)
        given: dict[str, dict[str, Source]] = {}
        for reader, named in tables.items():
            pieces = {name: _cut_one(obj, column, cut) for name, obj in named.items()}
            if held := {name: piece for name, piece in pieces.items() if piece is not None}:
                given[reader] = held
        if not given:
            empty.append(current.key)
        starts.append(cast('Start', given) if given else None)
    if empty:
        warnings.warn(
            f'start= gives the slices {empty} no row, so they start from nothing: no table carries their '
            f'{column!r}, and none reaches every slice. The answer is the same; only the time it takes changes.',
            SpecsolveWarning,
            stacklevel=3,
        )
    return starts


def _cut_one(obj: Source, column: str, cut: Callable[[pl.LazyFrame], pl.LazyFrame]) -> Source | None:
    """*obj* as one slice takes it: a table cut, read into memory to cross a process, or ``None`` where the cut leaves no row.

    A shape that is not a table, such as one number, carries no column to cut
    on and reaches the slice whole.
    """
    table = as_frame(obj)
    if table is None:
        return obj
    piece = (cut(table) if column in table.collect_schema().names() else table).pipe(collected)
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
    """*table*'s rows of the hand-built slice *key*, without the key column."""
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
    axis: Axis,
    carry: Mapping[str, str],
    sources: Mapping[str, Source],
    folded: Sweep,
    one_slice: Mapping[str, Source],
    keep_windows: bool,
) -> None:
    """Write the sweep's question and its answer to *out*.

    Each source's tidy table comes from *one_slice*, except the ones
    [`_uncut`][] and [`_spread_over_the_axis`][] give. A held sweep is spilled
    to scratch first, so the answer is always read off a spill.
    """
    manifest = axis_manifest(axis)
    if carry:
        manifest['carry'] = dict(carry)
    tidied = numbered(program, tidy_sources(program, one_slice))
    sliced = sources_with_column(sources, axis.dim)
    cut = {name: _uncut(program, axis, name, table) for name, table in sliced.items()}
    held = {**tidied, **_spread_over_the_axis(program, axis, sources, tidied, sliced), **cut}
    tables = {name: held[name] for name in sources}
    with beside(out) as scratch:
        spilled = folded if folded._spill is not None else scan_sweep(folded.save(scratch / 'slices'))
        answer = _the_answer(spilled, scratch / ANSWER_DIR, keep_windows=keep_windows)
        write_archive(out, spec, tables, axis=manifest, answer=answer)


def _the_answer(sweep: Sweep, under: Path, *, keep_windows: bool) -> Path:
    """The ``answer/`` an archive holds for a spilled *sweep*, laid out under *under*.

    One file per kind and name, as the readers return it, streamed from the
    spill; a name an EachWindow sweep cannot stitch has none, and
    ``reasons.parquet`` says why. The record and metrics take a single solve's
    columns, so ``keys.parquet`` beside them gives the keys their type back.
    ``sweep.json`` records *keep_windows*: a zip holds files only, so an empty
    ``windows/`` is not there to say so. What each window owns is kept in
    ``windows/`` beside them, the answer being stitched already.
    """
    spill = sweep._spill
    assert spill is not None, 'the answer is read off a spill, so a sweep too large to hold is never held'
    write_format(under, sweep._outputs)
    manifest = json.loads((spill.directory / MANIFEST_FILE).read_text())
    (under / MANIFEST_FILE).write_text(json.dumps({**manifest, 'windows': keep_windows}))
    shutil.copyfile(spill.directory / KEYS_FILE, under / KEYS_FILE)
    if keep_windows:
        (under / WINDOWS_DIR).mkdir()
        shutil.copyfile(spill.directory / OWNED_FILE, under / WINDOWS_DIR / OWNED_FILE)
    write_whole(sweep.record.drop(sweep.key_name), under / RECORD_FILE)
    write_whole(sweep.metrics.drop(sweep.key_name), under / METRICS_FILE)
    absent = {kind: dict(names) for kind, names in sweep._absent.items()}
    for kind in KINDS:
        for name, frame in sweep._slices.get(kind, {}).items():
            if why := sweep._unstitchable(frame):
                absent.setdefault(kind, {})[name] = why
                continue
            write_whole(sweep._answered(frame, per_window=False), under / kind / f'{name}.parquet')
        if keep_windows and (spill.directory / kind).is_dir():
            shutil.copytree(spill.directory / kind, under / WINDOWS_DIR / kind)
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
    sliced: Mapping[str, pl.LazyFrame],
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
    coordinates = pl.concat([table.select(axis.dim) for table in sliced.values()], how='vertical_relaxed')
    coordinates = coordinates.unique().sort(axis.dim)
    return {
        name: coordinates.join(tidied[name].drop(axis.into).unique(maintain_order=True), how='cross')
        for name in numbers
    }


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
    slices: Sequence[Slice],
    solving: Mapping[str, Any],  # pyrefly: ignore[explicit-any] — the verb's own keywords, forwarded
    plan: Mapping[str, _CarryRule],
    spill: Spill | None,
    starts: Sequence[Start | Literal['previous'] | None],
) -> Generator[tuple[Label, SliceAnswer], None, None]:
    """Each slice's answer, off one model updated in place.

    A slice naming other sources than the last is rebuilt, since ``update`` is
    partial. A generator because slice ``i+1``'s carry is read from slice
    ``i``'s frames after the yield; the caller closes it to release the model.
    A slice the spill holds is read back, and one solved here is written
    before it is yielded.
    """
    model: Model | None = None
    named: frozenset[str] | None = None
    state: dict[str, pl.DataFrame] = {}
    try:
        for position, current in enumerate(slices):
            if spill is not None and spill.done(position):
                answer = spill.read_back(position)
                primals = spill.written('primal', position, {rule.variable for rule in plan.values()})
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
                result = model.solve(**solving, start=starts[position])
                answer = _answers(result, program, _slice_metrics(model.diagnostics(), before), solving['outputs'])
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
    current: Slice,
    position: int,
    slices: Sequence[Slice],
    answer: SliceAnswer,
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
    slices: Sequence[Slice],
    solving: Mapping[str, Any],  # pyrefly: ignore[explicit-any] — the verb's own keywords, forwarded
    spill: Spill | None,
    starts: Sequence[Start | Literal['previous'] | None],
) -> Generator[tuple[Label, SliceAnswer], None, None]:
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
            starts[position],
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
        spec_digest=result.spec_digest,
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


def _run_slice(
    program: Program,
    document: Spec,
    encoded: dict[str, Any],  # pyrefly: ignore[explicit-any] — what crossed to the worker
    encode_out: bool,
    call: Mapping[str, Any],  # pyrefly: ignore[explicit-any] — the verb's own keywords, forwarded
    start: Start | Literal['previous'] | None,
) -> SliceAnswer:
    """One slice, start to finish, over plain data; module-level so a remote executor can pickle it."""
    with build(document, _decode(encoded)) as model, model.solve(**call, start=start) as result:
        answer = _answers(result, program, _slice_metrics(model.diagnostics(), None), call['outputs'])
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
