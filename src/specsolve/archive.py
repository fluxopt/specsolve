"""Reading an archive back: the spec, the data it was solved with, and what came back.

[`load_archive`][] reads it whole; [`scan_archive`][] leaves the frames on
disk; [`load_inputs`][] reads the spec and the data without the answer.
``archive=`` on the verbs that solve writes one, through
[`specsolve.archive_layout`][].
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from mathspec import to_spec
from mathspec.program import parameters_of

from specsolve.api import build, load_result, scan_result
from specsolve.archive_layout import (
    ANSWER_DIR,
    AXIS_MEMBER,
    INPUTS_LAYOUT,
    SOURCES_DIR,
    SPEC_MEMBER,
    opened,
)
from specsolve.axes import axis_from
from specsolve.errors import LayoutError, SpecsolveError
from specsolve.inputs import lower
from specsolve.relational.answer_layout import (
    ANSWER_LAYOUT,
    KINDS,
    METRICS_FILE,
    RUN,
    Metrics,
    other_layout,
    row_of,
    saved_frames,
)
from specsolve.relational.collect import collected
from specsolve.sweep import (
    MANIFEST_FILE,
    OWNED_FILE,
    WINDOWS_DIR,
    Spill,
    Sweep,
    opened_sweep,
    slice_index,
    with_key,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from mathspec import Spec
    from mathspec.program import Expression

    from specsolve.axes import Axis
    from specsolve.inputs import Buildable, Label, Source
    from specsolve.relational.result import Result

__all__ = ['ArchivedInputs', 'ResultArchive', 'SweepArchive', 'load_archive', 'load_inputs', 'scan_archive']


@dataclass(frozen=True)
class ResultArchive:
    """A spec, the data it was solved with, and what one solve of it returned.

    ``sps.solve(archive.spec, archive.sources)`` asks the question again.

    Attributes:
        spec: The spec as written, read back as one ``Spec`` whatever went in.
        sources: What was attached, keyed as the file declares it, as
            [`tidy`][specsolve.api.tidy] returns it: a table from
            [`load_archive`][], the path to one from [`scan_archive`][],
            which also holds the ``specsolve_run`` column.
        result: What the solve returned.
        metrics: What reaching the answer took, as one
            [`Metrics`][specsolve.relational.answer_layout.Metrics].
    """

    spec: Spec
    sources: Mapping[str, Source]
    result: Result
    metrics: Metrics


@dataclass(frozen=True)
class SweepArchive:
    """A spec, the data a sweep was solved over, the axis that cut it, and what came back.

    ``sps.solve_over(archive.spec, archive.sources, archive.axis, carry=archive.carry)``
    runs it again.

    Attributes:
        spec: The spec as written.
        sources: What the sweep was given, uncut, as [`ResultArchive`][]
            holds them. A source the axis cuts holds the axis column first; a
            parameter given as one number over a window's local index is held
            over the axis, so each slice cuts from it what it attached.
        axis: What cut them.
        carry: ``{parameter: variable}`` the slices were chained with, empty
            where they were not.
        sweep: The archived answer, in memory from [`load_archive`][], on
            disk from [`scan_archive`][]. ``per_window=True`` reads an
            EachWindow sweep's windows where ``keep_windows=True`` kept them,
            and is refused otherwise.
    """

    spec: Spec
    sources: Mapping[str, Source]
    axis: Axis
    carry: Mapping[str, str]
    sweep: Sweep


@dataclass(frozen=True)
class ArchivedInputs:
    """A spec and the data it was solved with, read off an archive without its answer.

    ``sps.solve(inputs.spec, inputs.sources)`` asks the question again, and
    ``sps.solve_over(inputs.spec, inputs.sources, inputs.axis, carry=inputs.carry)``
    runs a sweep again.

    Attributes:
        spec: The spec as written.
        sources: What was attached, keyed as the file declares it, as the
            tables [`tidy`][specsolve.api.tidy] returns. A sweep's are uncut,
            as [`SweepArchive`][] holds them.
        axis: What cut the sources, or ``None`` for an archive of one solve.
        carry: ``{parameter: variable}`` the slices were chained with, empty
            where they were not and for an archive of one solve.
    """

    spec: Spec
    sources: Mapping[str, Source]
    axis: Axis | None
    carry: Mapping[str, str]


def load_inputs(path: str | Path, into: str | Path | None = None) -> ArchivedInputs:
    """Read the spec and the data an archive holds, without its answer.

    The spec and the sources have a layout of their own, so this reads an
    archive whose answer [`load_archive`][] refuses because its layout has
    moved.

    Args:
        path: The archive, a ``.zip`` or the directory one was written to.
        into: Where to unpack a zip, kept afterwards. Without it a zip
            unpacks to a scratch directory removed before this returns.
            Refused for a directory archive, which is read where it lies.

    Raises:
        LanguageError: A ``spec.yaml`` the language does not accept.
        LayoutError: A member outside the layout, an *into* given for a
            directory, or a spec and sources whose layout has moved since
            they were written.
        zipfile.BadZipFile: A file that is not a zip archive.
    """
    return _whole(path, into, partial(_inputs, whole=True))


def load_archive(path: str | Path, into: str | Path | None = None) -> ResultArchive | SweepArchive:
    """Read an archive back whole: the sources as tables, the answer's frames in memory.

    Args:
        path: The archive, a ``.zip`` or the directory one was written to.
        into: Where to unpack a zip, kept afterwards. Without it a zip
            unpacks to a scratch directory removed before this returns.
            Refused for a directory archive, which is read where it lies.

    Returns:
        A [`SweepArchive`][] where the archive carries an axis, a
        [`ResultArchive`][] where it does not.

    Raises:
        LanguageError: A ``spec.yaml`` the language does not accept.
        LayoutError: A member outside the layout, an *into* given for a
            directory, or a spec, sources or answer whose layout has moved
            since it was written. Where only the answer's has,
            [`load_inputs`][] still reads the spec and the sources.
        SpecsolveError: An answer that names a different spec than the one
            beside it.
        zipfile.BadZipFile: A file that is not a zip archive.
    """
    return _whole(path, into, partial(_read, whole=True))


def _whole[T](path: str | Path, into: str | Path | None, read: Callable[[Path, Path], T]) -> T:
    """What *read* makes of the archive at *path*, given where its members are and the path to name.

    A zip with no *into* is unpacked to a scratch directory that is gone when
    this returns, so *read* must not hand back a path into it.
    """
    held = Path(path)
    if into is not None or held.is_dir():
        return read(opened(held, into), held)
    with tempfile.TemporaryDirectory() as scratch:
        return read(opened(held, scratch), held)


def scan_archive(path: str | Path, into: str | Path | None = None) -> ResultArchive | SweepArchive:
    """Read an archive back off disk: the sources as paths, each frame read at the call that asks for it.

    As [`load_archive`][], except that *into* is required for a zip, and kept,
    since the members have to outlive the value; a zip with no *into* raises
    ``LayoutError``.
    """
    return _read(opened(path, into), Path(path), whole=False)


def _inputs(under: Path, archive: Path, *, whole: bool) -> ArchivedInputs:
    """The spec, the sources and the axis the archive *under* holds, its sources as tables where *whole*, else as paths."""
    _refuse_other_inputs(under, archive)
    spec = to_spec(under / SPEC_MEMBER)
    sources: dict[str, Source] = {
        member.stem: pl.read_parquet(member).drop(RUN, strict=False) if whole else member
        for member in sorted((under / SOURCES_DIR).glob('*.parquet'))
    }
    axis_member = under / AXIS_MEMBER
    if not axis_member.is_file():
        return ArchivedInputs(spec, sources, None, {})
    manifest = json.loads(axis_member.read_text())
    return ArchivedInputs(spec, sources, axis_from(manifest), manifest.get('carry', {}))


def _read(under: Path, archive: Path, *, whole: bool) -> ResultArchive | SweepArchive:
    inputs = _inputs(under, archive, whole=whole)
    _refuse_another_layout(under / ANSWER_DIR, archive)
    spec, sources = inputs.spec, inputs.sources
    saved = under / ANSWER_DIR
    if inputs.axis is None:
        answer = _attach_readers((load_result if whole else scan_result)(saved), spec, sources)
        metrics = row_of(Metrics, pl.read_parquet(saved / METRICS_FILE).row(0, named=True), saved / METRICS_FILE)
        return ResultArchive(spec, sources, answer, metrics)
    answer = _attach_sweep_readers(_read_archived_sweep(saved, whole=whole), spec, sources, inputs.axis, inputs.carry)
    return SweepArchive(spec, sources, inputs.axis, inputs.carry, answer)


def _refuse_other_inputs(under: Path, archive: Path) -> None:
    """Refuse a spec and sources in another layout, naming the files that solve the model again by hand."""
    if (other := other_layout(under, INPUTS_LAYOUT)) is not None:
        raise LayoutError(
            f'{str(archive)!r} holds a spec and sources {other} and this package reads layout '
            f'{INPUTS_LAYOUT}. The layout moves before 1.0 and nothing reads another one back. The files are '
            f"plain: 'spec.yaml' is the spec, and each 'sources/<key>.parquet' is one source as the solve read "
            f'it. Read the spec with mathspec.to_spec, pass each source as its path, and solve again. A sweep '
            f"archive also holds 'axis.json', which names the axis and the carry the sweep ran with. A .zip "
            f'archive holds the same files as members.'
        )


def _refuse_another_layout(saved: Path, archive: Path) -> None:
    """Refuse an archived answer in another layout, and name the reader that still takes the spec and the sources.

    Checked before the answer is read, so this message wins over the generic
    one [`load_result`][] gives, which names a scratch directory for a zip.
    """
    if (other := other_layout(saved)) is not None:
        raise LayoutError(
            f'{str(archive)!r} holds an answer {other} and this package reads layout {ANSWER_LAYOUT}, so it '
            f'does not read the answer back. The spec and the data are in a layout this package reads: '
            f'sps.load_inputs({str(archive)!r}) returns them, to solve again.'
        )


def _attach_readers(answer: Result, spec: Buildable, sources: Mapping[str, Source]) -> Result:
    """*answer* with an undeclared expression readable through [`evaluate`][specsolve.relational.result.Result.evaluate].

    *spec* is rebuilt over *sources* at the first undeclared read, never
    solved, and cached. *answer* comes back unchanged where the solve left no
    values.
    """
    if not answer._primals:
        return answer
    frames = answer._primals
    dual_frames = answer._duals
    no_duals = answer._no_duals
    built: list[Callable[[str | Mapping[str, object]], pl.DataFrame]] = []

    def evaluate(written: str | Mapping[str, object]) -> pl.DataFrame:
        if not built:
            primals = {name: frame.pipe(collected) for name, frame in frames.items()}
            duals = (
                {name: frame.pipe(collected) for name, frame in dual_frames.items()}
                if no_duals is None and dual_frames
                else None
            )
            built.append(build(spec, sources)._evaluator(primals, duals, no_duals))
        return built[0](written)

    return replace(answer, _evaluate=evaluate)


def _read_archived_sweep(under: Path, *, whole: bool) -> Sweep:
    """The sweep an archive's ``answer/`` holds: one file per name, and the windows where kept.

    ``per_window=True`` is refused naming ``keep_windows=True`` where the
    archive has no windows. The ``specsolve_run`` column every archived frame
    carries is left on disk, so a frame read out of one equals the frame the
    live sweep returns.

    Args:
        under: The archive's ``answer/``.
        whole: Read every frame into memory, as [`load_sweep`][] does, rather
            than at the call that asks, as [`scan_sweep`][] does.
    """
    kept = json.loads((under / MANIFEST_FILE).read_text())['windows']
    opened = opened_sweep(under, under / WINDOWS_DIR / OWNED_FILE if kept else None)
    answer = {kind: saved_frames(under / kind, whole=whole) for kind in KINDS}
    windows = Spill(under / WINDOWS_DIR, opened.key_name, opened.record[opened.key_name].dtype)
    return replace(opened, _answer=answer, _windows=kept, _slices=windows.frames(whole=whole) if kept else {})


def _attach_sweep_readers(
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
    primal, dual = slice_index(sweep, 'primal'), slice_index(sweep, 'dual')
    for key, slice_sources in axis.slices(sources):
        slice_primals = {name: by_key[key] for name, by_key in primal.items() if key in by_key}
        if not slice_primals:
            continue
        slice_duals = {name: by_key[key] for name, by_key in dual.items() if key in by_key} or None
        yield key, build(spec, slice_sources)._evaluator(slice_primals, slice_duals, sweep._no_duals)


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
            with_key(evaluate_one(expression), sweep.key_name, key, key_dtype)
            for key, evaluate_one in _per_slice(sweep, spec, sources, axis)
        ]
        if not pieces:
            raise SpecsolveError('no slice of this sweep produced a solution, so an expression has nothing to read at.')
        return pl.concat(pieces)

    return evaluate
