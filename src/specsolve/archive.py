"""Reading an archive back: the spec, the data it was solved with, and what came back.

[`load_archive`][] reads it whole; [`scan_archive`][] leaves the frames on
disk. ``archive=`` on the verbs that solve writes one, through
[`specsolve.archive_layout`][].
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from mathspec import to_spec
from mathspec.program import parameters_of

from specsolve.api import build, load_result, scan_result
from specsolve.archive_layout import ANSWER_DIR, AXIS_MEMBER, DIGESTS_MEMBER, SOURCES_DIR, SPEC_MEMBER, opened
from specsolve.axes import axis_from
from specsolve.errors import SpecsolveError
from specsolve.lanes import lower
from specsolve.relational.answer_layout import KINDS, METRICS_FILE, RUN, Metrics, digest_of, row_of, saved_frames
from specsolve.sweep import (
    MANIFEST_FILE,
    WINDOWS_DIR,
    Spill,
    Sweep,
    held_in_memory,
    opened_sweep,
    slice_index,
    with_key,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from mathspec import Spec
    from mathspec.program import Expression

    from specsolve.api import Model
    from specsolve.axes import Axis
    from specsolve.lanes import Buildable, Label, Source
    from specsolve.relational.result import Result

__all__ = ['SolveArchive', 'SweepArchive', 'load_archive', 'scan_archive']


@dataclass(frozen=True)
class SolveArchive:
    """A spec, the data it was solved with, and what one solve of it returned.

    ``sps.solve(archive.spec, archive.sources)`` asks the question again.

    Attributes:
        spec: The spec as written, read back as one ``Spec`` whatever went in.
        sources: What was attached, keyed as the file declares it, as
            [`tidy`][specsolve.api.tidy] returns it: a table from
            [`load_archive`][], the path to one from [`scan_archive`][],
            which also holds the ``specsolve_run`` column.
        answer: What came back.
        source_digests: ``(specsolve_run, source, digest)``, one row per
            source, so two archives of one spec over different numbers name
            the input that moved. A digest is of the tidy table's parquet
            bytes, written before the run was stamped on, so two archives of
            one table under different names digest it alike; two polars
            versions can write one table to different digests, and reading
            an archive does not verify them.
        metrics: What reaching the answer took, as one
            [`Metrics`][specsolve.relational.answer_layout.Metrics].
    """

    spec: Spec
    sources: Mapping[str, Source]
    answer: Result
    source_digests: pl.DataFrame
    metrics: Metrics


@dataclass(frozen=True)
class SweepArchive:
    """A spec, the data a sweep was solved over, the axis that cut it, and what came back.

    ``sps.solve_over(sweep.spec, sweep.sources, sweep.axis, carry=sweep.carry)``
    runs it again.

    Attributes:
        spec: The spec as written.
        sources: What the sweep was given, uncut. A table or a path, as
            [`SolveArchive`][] holds them; a source the axis cuts holds the
            axis column first, and a parameter given as one number over a
            window's local index is held over the axis instead, so each
            slice cuts from it what that slice attached.
        axis: What cut them.
        carry: ``{parameter: variable}`` the slices were chained with, empty
            where they were not.
        answer: The sweep, whose readers return the answer the archive
            holds. Held from [`load_archive`][], on disk from
            [`scan_archive`][]. ``per_window=True`` reads an EachWindow
            sweep's windows where ``keep_windows=True`` kept them, and is
            refused otherwise.
        source_digests: As [`SolveArchive`][] holds it, of the uncut
            sources.
    """

    spec: Spec
    sources: Mapping[str, Source]
    axis: Axis
    carry: Mapping[str, str]
    answer: Sweep
    source_digests: pl.DataFrame


def load_archive(path: str | Path, into: str | Path | None = None) -> SolveArchive | SweepArchive:
    """Read an archive back whole: the sources as tables, the answer's frames in memory.

    Args:
        path: The archive, a ``.zip`` or the directory one was written to.
        into: Where to unpack a zip, kept afterwards. Without it a zip
            unpacks to a scratch directory removed before this returns.
            Refused for a directory archive, which is read where it lies.

    Returns:
        A [`SweepArchive`][] where the archive carries an axis, a
        [`SolveArchive`][] where it does not.

    Raises:
        LanguageError: A ``spec.yaml`` the language does not accept.
        LayoutError: A member outside the layout, an *into* given for a
            directory, or an answer whose layout has moved since it was
            written.
        SpecsolveError: An answer that names a different spec than the one
            beside it.
        zipfile.BadZipFile: A file that is not a zip archive.
    """
    held = Path(path)
    if into is not None or held.is_dir():
        return _read(opened(held, into), whole=True)
    with tempfile.TemporaryDirectory() as scratch:
        return _read(opened(held, scratch), whole=True)


def scan_archive(path: str | Path, into: str | Path | None = None) -> SolveArchive | SweepArchive:
    """Read an archive back off disk: the sources as paths, each frame read at the call that asks for it.

    As [`load_archive`][], except that *into* is required for a zip, and kept:
    the members have to outlive the value. ``LayoutError`` for a zip with no
    *into*.
    """
    return _read(opened(path, into), whole=False)


def _read(under: Path, *, whole: bool) -> SolveArchive | SweepArchive:
    spec = to_spec(under / SPEC_MEMBER)
    sources: dict[str, Source] = {
        member.stem: pl.read_parquet(member).drop(RUN, strict=False) if whole else member
        for member in sorted((under / SOURCES_DIR).glob('*.parquet'))
    }
    digests = pl.read_parquet(under / DIGESTS_MEMBER)
    saved = under / ANSWER_DIR
    axis_member = under / AXIS_MEMBER
    if not axis_member.is_file():
        answer = _attach_readers((load_result if whole else scan_result)(saved), spec, sources)
        _check_the_pairing(spec, [answer.spec_digest])
        metrics = row_of(Metrics, pl.read_parquet(saved / METRICS_FILE).row(0, named=True), saved / METRICS_FILE)
        return SolveArchive(spec, sources, answer, digests, metrics)
    manifest = json.loads(axis_member.read_text())
    axis, carry = axis_from(manifest), manifest.get('carry', {})
    answer = _attach_sweep_readers(_read_archived_sweep(saved, whole=whole), spec, sources, axis, carry)
    _check_the_pairing(spec, answer.record['spec_digest'].to_list())
    return SweepArchive(spec, sources, axis, carry, answer, digests)


def _check_the_pairing(spec: Spec, answered: Sequence[str | None]) -> None:
    """Refuse an archive whose answer came back from a different spec than the one beside it.

    A ``None`` digest is not compared.
    """
    mine = digest_of(spec)
    if others := sorted({other for other in answered if other is not None and other != mine}):
        raise SpecsolveError(
            f'this archive holds an answer that came back from a different spec: the answer carries '
            f'{others} and the spec.yaml beside it digests to {mine}, so re-solving it would give another answer.'
        )


def _refuse_another_model(answer: Result, model: Model) -> None:
    """Refuse a saved answer against a model built from other data than the one it answered.

    The spec is compared where the pair is read; the data needs a build, so it
    is compared here. An answer carrying no digest is taken as given.

    Raises:
        SpecsolveError: Sources that build a model other than the answered one.
    """
    answered = answer.model_digest()
    if answered is not None and answered != (rebuilt := model._model_digest()):
        raise SpecsolveError(
            f'this answer came back from another model: it answered the model digesting to {answered} '
            f'and the spec and sources beside it build {rebuilt}. The document matched, so what differs '
            f'is the data — and reading a quantity the file never named against other numbers would '
            f'value it at an answer nobody solved for.\n'
            f'  Read the answer against the data the solve ran on. An archive holds that pair, so one '
            f'refused here has had a source replaced since it was written.'
        )


def _attach_readers(answer: Result, spec: Buildable, sources: Mapping[str, Source]) -> Result:
    """*answer* with an undeclared expression readable through [`evaluate`][specsolve.relational.result.Result.evaluate].

    *spec* is rebuilt over *sources* at the first undeclared read, never
    solved, and cached. A rebuild from other data than the solve ran on is
    refused. *answer* comes back unchanged where the solve left no values.

    Args:
        answer: A saved solve, as [`load_result`][specsolve.api.load_result] or [`scan_result`][specsolve.api.scan_result]
            read it back.
        spec: The model the answer solved, as [`build`][specsolve.api.build] takes it.
        sources: What it was solved with, as [`build`][specsolve.api.build] takes them.
    """
    if not answer._primals:
        return answer
    frames = answer._primals
    dual_frames = answer._duals
    no_duals = answer._no_duals
    built: list[Callable[[str | Mapping[str, object]], pl.DataFrame]] = []

    def evaluate(written: str | Mapping[str, object]) -> pl.DataFrame:
        if not built:
            primals = {name: frame.collect() for name, frame in frames.items()}
            duals = (
                {name: frame.collect() for name, frame in dual_frames.items()}
                if no_duals is None and dual_frames
                else None
            )
            model = build(spec, sources)
            _refuse_another_model(answer, model)
            built.append(model.evaluator(primals, duals, no_duals))
        return built[0](written)

    return replace(answer, _evaluate=evaluate)


def _read_archived_sweep(under: Path, *, whole: bool) -> Sweep:
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
    opened = opened_sweep(under)
    answer = {kind: saved_frames(under / kind, whole=whole) for kind in KINDS}
    kept = json.loads((under / MANIFEST_FILE).read_text())['windows']
    opened = replace(opened, _answer=answer, _windows=kept)
    spill = Spill(under / WINDOWS_DIR, opened.key_name, opened.record[opened.key_name].dtype) if kept else None
    if not whole:
        return replace(opened, _spill=spill, _disk=under)
    return opened if spill is None else held_in_memory(opened, spill)


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
            with_key(evaluate_one(expression), sweep.key_name, key, key_dtype)
            for key, evaluate_one in _per_slice(sweep, spec, sources, axis)
        ]
        if not pieces:
            raise SpecsolveError('no slice of this sweep produced a solution, so an expression has nothing to read at.')
        return pl.concat(pieces)

    return evaluate
