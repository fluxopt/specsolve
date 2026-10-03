"""Reading an archive back: the spec, the data it was solved with, and what came back.

[`load_archive`][] reads it whole; [`scan_archive`][] leaves the frames on
disk. ``archive=`` on the verbs that solve writes one, through
[`specsolve.layout`][].
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from mathspec import to_spec

from specsolve.api import build, load_result, scan_result
from specsolve.errors import SpecsolveError
from specsolve.layout import ANSWER_DIR, AXIS_MEMBER, DIGESTS_MEMBER, SOURCES_DIR, SPEC_MEMBER, opened
from specsolve.relational.parquet import METRICS_FILE, RUN, Metrics, digest_of, row_of
from specsolve.strategy import (
    EachCoordinate,
    EachWindow,
    Sweep,
    attach_sweep_readers,
    axis_from,
    read_archived_sweep,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mathspec import Spec

    from specsolve.api import Model
    from specsolve.lanes import Buildable, Source
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
            [`Metrics`][specsolve.relational.parquet.Metrics].
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
    axis: EachCoordinate | EachWindow
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
    answer = attach_sweep_readers(read_archived_sweep(saved, whole=whole), spec, sources, axis, carry)
    _check_the_pairing(spec, answer.record['spec_digest'].to_list())
    return SweepArchive(spec, sources, axis, carry, answer, digests)


def _check_the_pairing(spec: Spec, answered: Sequence[str | None]) -> None:
    """Refuse an archive whose answer came back from a different spec than the one beside it.

    A ``None`` digest is not compared.
    """
    mine = digest_of(spec.to_yaml())
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
