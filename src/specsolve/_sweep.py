"""What a sweep returns: the answer over the model's own coordinates, a record per slice, and its spill on disk.

[`load_sweep`][] and [`scan_sweep`][] read one back. The caller-facing rules
are [sweeps](https://specsolve.readthedocs.io/en/latest/reference/sweeps/).
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import polars as pl

from specsolve import _parquet as parquet
from specsolve._answer_layout import (
    KEYS_FILE,
    MANIFEST_FILE,
    METRICS_FILE,
    OWNED_FILE,
    RECORD_FILE,
    check_format,
    clear_the_answer,
    consolidated,
    read_outputs,
    read_reasons,
    read_spec,
    row_of,
    write_format,
    write_reasons,
    write_spec,
    write_whole,
)
from specsolve._axes import Stitch, bulleted
from specsolve._collect import collected
from specsolve._names import RUN
from specsolve._outputs import BASES, KINDS, NO_BASIS, OUTPUT_KINDS, PRICED, checked_kind, not_requested_message
from specsolve._records import METRICS_SCHEMA, RECORD_SCHEMA, Metrics, Record
from specsolve._result import tidy_to_dataarray, tidy_to_dataset, tidy_to_pandas
from specsolve.errors import LayoutError, SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence

    import pandas as pd
    import xarray as xr
    from mathspec.spec import Spec

    from specsolve._answer_layout import FrameWriter
    from specsolve._inputs import Label
    from specsolve._outputs import Output


#: What each of the [`KINDS`][specsolve._outputs.KINDS] is a frame of, as a message names it.
_LABELS: Mapping[str, str] = MappingProxyType(
    {
        'primal': 'variable',
        'dual': 'constraint',
        'expression': 'named expression',
        **{kind: carried.per for kind, carried in OUTPUT_KINDS.items()},
    }
)


@dataclass(frozen=True)
class SliceAnswer:
    """One slice, solved and read out; plain data only, so it can cross a process."""

    record: Record
    #: This slice's row of [`Sweep.metrics`][].
    metrics: Metrics
    #: ``{kind: {name: frame}}`` over [`KINDS`][specsolve._outputs.KINDS]:
    #: every variable, every constraint's dual, every declared named
    #: expression evaluated at this slice's solution, and each output the
    #: sweep was asked for. A kind the slice produced nothing of is absent.
    frames: dict[str, dict[str, pl.DataFrame]] = field(default_factory=dict)
    #: Why this slice has no duals, when it has none.
    no_duals: str | None = None
    #: Per expression, why this slice could not evaluate it.
    no_expressions: dict[str, str] = field(default_factory=dict)

    def sliced(self, key_name: str, key: Label) -> SliceAnswer:
        """This answer with its record and metrics rows naming the slice they are."""
        text = str(key)
        return replace(
            self,
            record=self.record._replace(slice_axis=key_name, slice=text),
            metrics=self.metrics._replace(slice_axis=key_name, slice=text),
        )


def one_key_type(keys: Sequence[Label], key_name: str) -> pl.DataType:
    """The type every file writes *key_name* as; keys that fit no one type are refused, never coerced."""
    try:
        typed = pl.Series(keys)
    except TypeError as apart:
        kinds = sorted({type(key).__name__ for key in keys})
        raise SpecsolveError(
            f'the keys of this sweep fit no one column type ({", ".join(kinds)}), so its files could not all '
            f'write {key_name!r} as one. Every file carries the key, and a column that changes type between '
            f'them cannot be concatenated or loaded into one table. Give the labels one type in every source.'
        ) from apart
    _one_slice_per_text(keys)
    return typed.dtype


def _one_slice_per_text(keys: Sequence[Label]) -> None:
    """Refuse two keys of one text, which the record and metrics name a slice by.

    ``_rekeyed`` matches each row's ``slice`` text against the keys' text,
    when the fold ends and when a spill is scanned, so two keys of one text
    would fail there, after every slice has solved.
    """
    texts = Counter(str(key) for key in keys)
    repeated = [text for text, count in texts.items() if count > 1]
    if repeated:
        raise SpecsolveError(
            f'the key {repeated[0]!r} names more than one slice of this sweep, and the record names a slice '
            f'by its key as text, so those slices could not be told apart. Give each slice its own key.'
        )


def with_key(frame: pl.DataFrame, key_name: str, key: Label, dtype: pl.DataType) -> pl.DataFrame:
    """*frame* with the slice key prepended as *dtype*, the whole sweep's type rather than ``pl.lit``'s."""
    return frame.select(pl.lit(key, dtype=dtype).alias(key_name), pl.all())


def sweep_manifest(
    key_name: str, keys: Sequence[Label], stitch: Stitch | None, outputs: frozenset[Output]
) -> dict[str, object]:
    """What ``sweep.json`` names: the key, the keys as text, the stitch and the outputs."""
    return {
        'key_name': key_name,
        'keys': [str(key) for key in keys],
        'stitch': None if stitch is None else {'dim': stitch.dim},
        'outputs': sorted(outputs),
    }


@dataclass(frozen=True)
class Spill:
    """A sweep's answers on disk instead of in memory, one file per slice and name.

    ``<kind>/<name>/<position>.parquet`` holds the keyed frames, and
    ``record/`` and ``metrics/`` the rows, which name their slice as text in
    ``slice_axis`` and ``slice``; ``keys.parquet`` holds the keys as their own
    type. ``sweep.json`` names the key, the keys and the outputs, so a
    directory answers for one sweep, and ``spec.yaml`` is the spec it answered. Every file lands whole. A slice's record file is written
    last and marks it done; ``sweep.json`` lands after the files a scan reads
    beside it and marks the directory stamped.
    """

    directory: Path
    key_name: str
    #: The key column's type, settled over the sweep's keys, never per file.
    key_dtype: pl.DataType
    #: How each table is written, where an archive adds its run; ``None`` is
    #: [`write_whole`][specsolve._answer_layout.write_whole].
    writer: FrameWriter | None = None

    def _write(self, frame: pl.DataFrame | pl.LazyFrame, path: Path) -> None:
        """*frame* at *path*, through [`writer`][]."""
        (self.writer or write_whole)(frame, path)

    @classmethod
    def opened(
        cls,
        directory: str | Path,
        key_name: str,
        keys: Sequence[Label],
        key_dtype: pl.DataType,
        stitch: Stitch | None,
        outputs: frozenset[Output],
        spec: Spec | None,
        writer: FrameWriter | None = None,
    ) -> Spill:
        """The directory ready to take this sweep: stamped if it holds none, checked and never re-stamped if it does."""
        directory = Path(directory)
        spill = cls(directory, key_name, key_dtype, writer)
        manifest = sweep_manifest(key_name, keys, stitch, outputs)
        record = directory / MANIFEST_FILE
        if record.exists():
            check_format(directory)
            found = json.loads(record.read_text())
            if {**found, 'outputs': None} != {**manifest, 'outputs': None}:
                raise SpecsolveError(
                    f'{str(directory)!r} holds a sweep keyed by {found["key_name"]!r} over {found["keys"]}, and '
                    f'this one is keyed by {key_name!r} over {manifest["keys"]}. A directory holds one sweep: '
                    f'point spill_to= at an empty one, or delete this one to solve it again.'
                )
            if found['outputs'] != manifest['outputs']:
                raise SpecsolveError(
                    f'{str(directory)!r} holds this sweep solved with outputs={found["outputs"]}, and this '
                    f'run asks for outputs={manifest["outputs"]}. A slice on disk carries only what it was '
                    f'solved with, so the two could not be read as one sweep: ask for the same outputs to '
                    f'resume it, or point spill_to= at an empty directory.'
                )
        else:
            write_format(directory, outputs)
            write_spec(directory, spec)
            spill._write(pl.DataFrame([pl.Series(key_name, keys, dtype=key_dtype)]), directory / KEYS_FILE)
            if stitch is not None:
                spill._write(stitch.owned, directory / OWNED_FILE)
            record.write_text(json.dumps(manifest))
        return spill

    def _file(self, kind: str, position: int, name: str | None = None) -> Path:
        under = self.directory / kind if name is None else self.directory / kind / name
        return under / f'{position:06d}.parquet'

    def done(self, position: int) -> bool:
        return self._file('record', position).exists()

    def write(self, position: int, key: Label, answer: SliceAnswer) -> SliceAnswer:
        """*answer*'s frames and record on disk, and the answer with the frames released."""
        answer = answer.sliced(self.key_name, key)
        self._keep_reasons(answer)
        self.write_tables(
            position,
            key,
            answer.frames,
            pl.DataFrame([answer.record._asdict()], schema=RECORD_SCHEMA),
            pl.DataFrame([answer.metrics._asdict()], schema=METRICS_SCHEMA),
        )
        return replace(answer, frames={})

    def _keep_reasons(self, answer: SliceAnswer) -> None:
        """*answer*'s reasons for what it lacks, added to the sweep's on disk before its record lands.

        A slice read back on resume gives no reason again, so one written
        only when the fold ends is lost to a sweep that stopped before then.
        The first slice to give a reason keeps it.
        """
        if answer.no_duals is None and not answer.no_expressions:
            return
        no_duals, absent = read_reasons(self.directory)
        known = absent.get('expression', {})
        if (answer.no_duals is None or no_duals is not None) and set(answer.no_expressions) <= set(known):
            return
        gathered = {**absent, 'expression': {**answer.no_expressions, **known}}
        write_reasons(self.directory, no_duals or answer.no_duals, gathered, self._write)

    def write_tables(
        self,
        position: int,
        key: Label,
        frames: Mapping[str, Mapping[str, pl.DataFrame]],
        record: pl.DataFrame,
        metrics: pl.DataFrame,
    ) -> None:
        """One slice on disk: its *frames*, and its one row of *record* and of *metrics*.

        The two rows are taken as tables, so a sweep saves what it holds
        without reading a value out. A UTC timestamp read out as a Python
        value needs a time zone database, and the browser has none.
        """
        for kind, produced in frames.items():
            for name, frame in produced.items():
                self._write(with_key(frame, self.key_name, key, self.key_dtype), self._file(kind, position, name))
        self._write(metrics, self._file('metrics', position))
        self._write(record, self._file('record', position))

    def take(self, other: Spill) -> None:
        """Every slice *other* holds, written here file for file, so a sweep too large to hold is copied without being held.

        The records land last, as [`write`][] lands one, since a record is
        what says its slice is done.
        """
        for held in (*KINDS, 'metrics', 'record'):
            if (other.directory / held).is_dir():
                shutil.copytree(
                    other.directory / held,
                    self.directory / held,
                    copy_function=lambda source, target: self._write(parquet.scan(Path(source)), Path(target)),
                    dirs_exist_ok=True,
                )

    def read_back(self, position: int) -> SliceAnswer:
        """A done slice's record, with no frames."""
        row = parquet.read(self._file('record', position)).row(0, named=True)
        held = parquet.read(self._file('metrics', position)).row(0, named=True)
        return SliceAnswer(row_of(Record, row, self.directory), row_of(Metrics, held, self.directory))

    def written(self, kind: str, position: int, names: Iterable[str]) -> dict[str, pl.DataFrame]:
        """The named frames of *kind* a done slice wrote; a name it did not write is absent."""
        found = {name: self._file(kind, position, name) for name in names}
        return {name: parquet.read(path).drop(self.key_name) for name, path in found.items() if path.exists()}

    def frames(self, *, eager: bool) -> dict[str, dict[str, pl.LazyFrame]]:
        """``{kind: {name: frame}}``, every slice's frame of a name as one, keyed and in slice order.

        A kind or a name no slice wrote is absent. *eager* reads the files into
        memory now, rather than as a `polars.scan_parquet` collected at the
        first read.
        """
        out: dict[str, dict[str, pl.LazyFrame]] = {}
        for kind in KINDS:
            under = self.directory / kind
            for named in sorted(under.iterdir()) if under.is_dir() else []:
                files = sorted(named.glob('*.parquet'))
                frame = parquet.read(files).lazy() if eager else parquet.scan(files)
                out.setdefault(kind, {})[named.name] = frame.drop(RUN, strict=False)
        return out


@dataclass(frozen=True)
class Sweep:
    """What a fold returned: the answer over the model's own coordinates, and a record per slice.

    [`Result`][specsolve._result.Result]'s readers, same names and
    shapes, and each returns **the answer** by default. An [`EachWindow`][]
    sweep is read over the dimension it sliced: each coordinate comes from
    the window that owns it, and the lookahead rows are dropped. An
    [`EachLabel`][] sweep is keyed by slice, the key
    prepended, each slice being a whole answer. Nothing is combined across
    slices.

    ``per_window=True`` reads an EachWindow sweep one window at a time:
    keyed by where each window started, over the window's own labels,
    lookahead rows included.
    """

    key_name: str
    #: One [`Record`][specsolve._records.Record] per slice, in slice
    #: order — how every slice terminated, whether or not it produced an
    #: answer. The key column comes first, as its own type, so the table joins
    #: to the frames; ``slice_axis`` and ``slice`` name the slice again as
    #: text, and are what a saved sweep or an archive writes in its place. A
    #: slice that reached no objective holds null there rather than ``nan``,
    #: so the column aggregates over the slices that solved.
    records: pl.DataFrame
    #: One [`Metrics`][specsolve._records.Metrics] per slice, keyed as
    #: [`records`][] is and in slice order — [`diagnostics`][specsolve._api.Model.diagnostics]
    #: one dimension wider, its counts and clocks only. Each row is the slice's
    #: own share: ``solves`` is ``1``, and ``loads`` is ``1`` where the solver
    #: took the model from scratch. Under a serial fold the first slice does
    #: and the rest are pushed values, so a later ``1`` is a slice whose data
    #: moved a mask; under an executor every slice builds alone and every one
    #: loads. So a slow sweep says which slice, and which phase of it.
    metrics: pl.DataFrame
    #: The spec every slice answered, as written; ``None`` only for a directory
    #: written without one.
    spec: Spec | None
    #: ``{kind: {name: frame}}``, every slice's frame of a name as one, keyed,
    #: in memory or scanned off disk; a kind no slice produced is absent.
    _slices: dict[str, dict[str, pl.LazyFrame]] = field(repr=False, default_factory=dict)
    _no_duals: str | None = field(repr=False, default=None)
    #: ``{kind: {name: reason}}`` for a name some slice could not produce.
    _absent: dict[str, dict[str, str]] = field(repr=False, default_factory=dict)
    _stitch: Stitch | None = field(repr=False, default=None)
    #: The directory the slices lie in, for a sweep solved with ``spill_to=``
    #: or scanned off one; what an archive of it is packed from.
    _spill: Spill | None = field(repr=False, default=None)
    #: Each [`Output`][specsolve._outputs.Output] the sweep was
    #: asked for, and so carries.
    _outputs: frozenset[Output] = field(repr=False, default=frozenset())
    #: What [`evaluate`][] lowers an undeclared expression through; ``None``
    #: on a live solve's Sweep, which retains no model.
    _evaluate: Callable[[str | Mapping[str, object]], pl.DataFrame] | None = field(repr=False, default=None)

    @classmethod
    def _folded(
        cls,
        key_name: str,
        stitch: Stitch | None,
        answered: Generator[tuple[Label, SliceAnswer], None, None],
        spill: Spill | None,
        key_dtype: pl.DataType,
        outputs: frozenset[Output],
        spec: Spec,
    ) -> Sweep:
        """Every slice's answer absorbed, in the order they arrive.

        Closing the stream releases the serial fold's model when a fold is
        abandoned. A reason a slice lacks something is kept from the first
        slice that gave one. A spilled sweep reads its reasons off the
        directory, where each slice left its own before its record, so the
        slices a resume reads back are counted too.
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
                rows.append(named.record)
                taken.append(named.metrics)
                for kind, produced in answer.frames.items():
                    for name, frame in produced.items():
                        frames[kind][name].append(with_key(frame, key_name, key, key_dtype))
        keyed = pl.Series(key_name, keys, dtype=key_dtype)
        texts = [str(key) for key in keys]
        absent: dict[str, dict[str, str]] = {'expression': no_expressions} if no_expressions else {}
        if spill is not None:
            slices = spill.frames(eager=False)
            no_duals, absent = read_reasons(spill.directory)
        else:
            slices = {
                kind: {name: pl.concat(held).lazy() for name, held in named.items()} for kind, named in frames.items()
            }
        return cls(
            key_name=key_name,
            records=_rekeyed(pl.DataFrame([row._asdict() for row in rows], schema=RECORD_SCHEMA), keyed, texts),
            metrics=_rekeyed(pl.DataFrame([row._asdict() for row in taken], schema=METRICS_SCHEMA), keyed, texts),
            spec=spec,
            _slices=slices,
            _no_duals=no_duals,
            _absent=absent,
            _stitch=stitch,
            _spill=spill,
            _outputs=outputs,
        )

    @property
    def keys(self) -> list[Label]:
        return self.records[self.key_name].to_list()

    def _start(self, column: str) -> dict[str, dict[str, pl.DataFrame]]:
        """This sweep's answer as tables to start a sweep cut by *column* from, keyed as [`Start`][specsolve.types.Start] is.

        The primal, and the basis where the sweep carries one, each read as
        [`scan`][] reads it: over the dimension an EachWindow sweep cut, or
        keyed by slice, the key column renamed to *column*, whose coordinates
        it holds.
        """
        rename = {} if self._stitch is not None else {self.key_name: column}
        given: dict[str, dict[str, pl.DataFrame]] = {}
        for kind in ('primal', *(sorted(BASES) if 'basis' in self._outputs else ())):
            held, _ = self._answerable(kind, per_window=False)
            if held:
                given[kind] = {name: self.scan(name, kind).rename(rename).pipe(collected) for name in held}
        return given

    def _check_per_window(self) -> None:
        """Refuse ``per_window=True`` where there are no windows to read."""
        if self._stitch is None:
            raise SpecsolveError(
                f'per_window=True reads an EachWindow sweep one window at a time, and this sweep was not cut '
                f'into windows: its answer already is one frame per slice, keyed by {self.key_name!r}. Read '
                f'it without per_window.'
            )

    def _check_requested(self, kind: str, name: str) -> None:
        """Refuse a kind of the [`OUTPUT_KINDS`][specsolve._outputs.OUTPUT_KINDS] the sweep was not asked for."""
        if kind in OUTPUT_KINDS and OUTPUT_KINDS[kind].output not in self._outputs:
            raise SpecsolveError(not_requested_message(kind, name))

    def _named(self, kind: str, name: str, *, per_window: bool) -> pl.LazyFrame:
        """*name*'s frame of *kind*, lazily: the answer, or the frames per window."""
        self._check_requested(kind, name)
        if per_window:
            self._check_per_window()
        held = self._slices.get(kind, {})
        frame = held.get(name)
        if frame is None:
            absent = self._absent.get(kind, {}).get(name) or _whole_kind_absent(
                kind, held, self._no_duals, self.records
            )
            raise SpecsolveError(absent or _nothing_to_read(_LABELS[kind], name, held, self.records))
        return self._answered(frame, per_window=per_window)

    def _answered[F: (pl.DataFrame, pl.LazyFrame)](self, frame: F, *, per_window: bool) -> F:
        """*frame*, as the slices produced it, read the way the caller asked.

        [`EachWindow`][] stitches through its [`Stitch`][]. Any other axis, and
        a read per window the caller has already checked, gets *frame* unchanged.
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
        """One name's answer as a `polars.LazyFrame`: [`primal`][], [`dual`][], [`evaluate`][] or an output, not collected.

        On a sweep whose frames lie on disk — solved with ``spill_to=``, or
        read by [`scan_sweep`][] or
        [`scan_archive`][specsolve._archive.scan_archive] — nothing is read
        until the frame is collected, so a filter or a select runs before the
        bytes move.

        Args:
            name: A variable, a constraint or a named expression the spec
                declares, as *kind* says.
            kind: ``primal``, ``dual``, ``expression``, or an
                [`Output`][specsolve.types.Output] the sweep was asked for —
                the reader this stands in for.
            per_window: Read an EachWindow sweep one window at a time instead
                of its answer.

        Raises:
            SpecsolveError: No slice produced *name*, a *kind* that names no
                reader, an output the sweep was not asked for, or
                ``per_window`` where there are no windows to read.
        """
        kind = checked_kind(kind)
        if kind == 'expression' and not self._holds_expression(name):
            return self.evaluate(name, per_window=per_window).lazy()
        return self._named(kind, name, per_window=per_window)

    def primal(self, name: str, *, per_window: bool = False) -> pl.DataFrame:
        """One variable's answer.

        A slice that reached no solution contributes no rows, so this can be
        shorter than the sweep; [`records`][] is one row per slice always.

        Args:
            name: A variable the sweep's spec declares.
            per_window: Read an EachWindow sweep one window at a time instead
                of its answer.

        Raises:
            SpecsolveError: No slice of the sweep produced *name*; a variable
                that is not over an EachWindow sweep's windowed dimension,
                which has an answer only per window; or ``per_window`` on a
                sweep that was not cut into windows.
        """
        return self._named('primal', name, per_window=per_window).pipe(collected)

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
        return self._named('dual', name, per_window=per_window).pipe(collected)

    def activity(self, name: str, *, per_window: bool = False) -> pl.DataFrame:
        """One constraint's left-hand side at every slice's solution.

        [`primal`][]'s shape and arguments, and carried only where the sweep
        was asked for it with ``outputs={'activity'}``. In the answer of an
        EachWindow sweep each coordinate carries the window that owns it.

        Raises:
            SpecsolveError: The sweep was not asked for its activity, or as
                [`primal`][] raises.
        """
        return self._named('activity', name, per_window=per_window).pipe(collected)

    def slack(self, name: str, *, per_window: bool = False) -> pl.DataFrame:
        """One constraint's distance to binding at every slice's solution.

        [`primal`][]'s shape and arguments, and
        [`Result.slack`][specsolve._result.Result.slack]'s sign.
        Carried only where the sweep was asked for it with
        ``outputs={'slack'}``.

        Raises:
            SpecsolveError: The sweep was not asked for its slack, or as
                [`primal`][] raises.
        """
        return self._named('slack', name, per_window=per_window).pipe(collected)

    def reduced_cost(self, name: str, *, per_window: bool = False) -> pl.DataFrame:
        """One variable's reduced costs at every slice's solution.

        [`primal`][]'s shape and arguments, and
        [`Result.reduced_cost`][specsolve._result.Result.reduced_cost]'s
        sign. Carried only where the sweep was asked for it with
        ``outputs={'reduced_cost'}``. A slice whose model had an integer
        variable contributes none, as it contributes no duals.

        Raises:
            SpecsolveError: The sweep was not asked for reduced costs, no
                slice produced them — the message says why — or as
                [`primal`][] raises.
        """
        return self._named('reduced_cost', name, per_window=per_window).pipe(collected)

    def variable_basis(self, name: str, *, per_window: bool = False) -> pl.DataFrame:
        """One variable's basis status where every slice's solve ended.

        [`primal`][]'s shape and arguments, and
        [`Result.variable_basis`][specsolve._result.Result.variable_basis]'s
        words. Carried only where the sweep was asked for it with
        ``outputs={'basis'}``. A slice that ended on no basis contributes
        none.

        Raises:
            SpecsolveError: The sweep was not asked for its basis, no slice
                ended on one, or as [`primal`][] raises.
        """
        return self._named('variable_basis', name, per_window=per_window).pipe(collected)

    def constraint_basis(self, name: str, *, per_window: bool = False) -> pl.DataFrame:
        """One constraint's basis status where every slice's solve ended.

        [`primal`][]'s shape and arguments, and
        [`Result.constraint_basis`][specsolve._result.Result.constraint_basis]'s
        words. Carried only where the sweep was asked for it with
        ``outputs={'basis'}``. A slice that ended on no basis contributes
        none.

        Raises:
            SpecsolveError: The sweep was not asked for its basis, no slice
                ended on one, or as [`primal`][] raises.
        """
        return self._named('constraint_basis', name, per_window=per_window).pipe(collected)

    def evaluate(self, expression: str | Mapping[str, object], *, per_window: bool = False) -> pl.DataFrame:
        """The value of *expression* at every slice's solution, as an answer.

        [`evaluate`][specsolve._result.Result.evaluate] over the
        sweep, with [`primal`][]'s shape. A declared name is read from what
        the sweep holds, live or off disk. Anything else is valued at each
        slice's own solution with no re-solve, which needs the spec, sources
        and axis the sweep [`load_archive`][specsolve._archive.load_archive]
        hands back carries; a Sweep a live solve returned retains no model. A
        parameter the sweep **carried** reads as each slice was solved with
        it, the answer of the slice before.

        In an EachWindow sweep's answer each coordinate carries the value of
        the window that owns it, so a sum does not double-count the lookahead.

        Args:
            expression: A declared name, an expression string, or the ``cases:``
                mapping, as one ``expressions:`` entry takes.
            per_window: Read an EachWindow sweep one window at a time instead
                of its answer.

        Raises:
            SpecsolveError: No slice produced a declared *expression* (an
                evaluation that failed on every slice carries its own reason);
                an undeclared expression on a Sweep with no model behind it; a
                quantity reduced over an EachWindow sweep's windowed dimension, which
                has an answer only per window; or ``per_window`` as
                [`primal`][] raises it.
            LanguageError: A construct outside the language, or a name the spec
                does not declare.
        """
        if isinstance(expression, str) and self._holds_expression(expression):
            return self._named('expression', expression, per_window=per_window).pipe(collected)
        if self._evaluate is None:
            raise SpecsolveError(self._nothing_to_evaluate(expression))
        if per_window:
            self._check_per_window()
        return self._answered(self._evaluate(expression), per_window=per_window)

    def _expression_names(self) -> Mapping[str, object]:
        """The declared expressions the sweep holds."""
        return self._slices.get('expression', {})

    def _holds_expression(self, name: str) -> bool:
        """Whether *name* is a declared expression the sweep holds, or one it says why it lacks."""
        return name in self._expression_names() or name in self._absent.get('expression', {})

    def _nothing_to_evaluate(self, expression: str | Mapping[str, object]) -> str:
        """Why a sweep with no model behind it cannot value *expression*; a string is also answered as a name.

        The way back to its models is an archive, which holds what rebuilds
        each slice. One solve of the sweep's sources is another question, and
        for a label sweep one the door refuses.
        """
        no_model = (
            'this sweep has no model behind it, so a quantity the file never named cannot be read from it: a '
            'sweep holds the values of each slice without the models that solved them. Solve it with '
            "archive=, and sps.load_archive(path).sweep reads any expression at each slice's solution; a name "
            'the spec declares is readable either way.'
        )
        if not isinstance(expression, str):
            return no_model
        return (
            f'{_nothing_to_read(_LABELS["expression"], expression, self._expression_names(), self.records)} {no_model}'
        )

    def _frame(self, name: str, kind: str, *, per_window: bool) -> pl.DataFrame:
        """*name* through the reader *kind* names."""
        kind = checked_kind(kind)
        if kind == 'expression':
            return self.evaluate(name, per_window=per_window)
        return self._named(kind, name, per_window=per_window).pipe(collected)

    def to_pandas(self, name: str, kind: str = 'primal', *, per_window: bool = False) -> pd.DataFrame:
        """One name's answer as a tidy `pandas.DataFrame`; [`scan`][]'s arguments.

        The name is resolved before pandas is imported, so a sweep that never
        held *name* says so on any install.
        """
        return tidy_to_pandas(self._frame(name, kind, per_window=per_window))

    def to_dataarray(self, name: str, kind: str = 'primal', *, per_window: bool = False) -> xr.DataArray:
        """One name's answer as a `xarray.DataArray`; [`to_pandas`][]'s arguments.

        An EachWindow sweep's answer is indexed by the dimension it sliced.
        Any other sweep adds the slice key as a dimension named by the axis,
        ``(scenario, …)``, and a read per window adds ``<dim>_start``; there a
        slice that reached no solution comes back NaN, as a masked coordinate
        does from ``Result``.
        """
        return tidy_to_dataarray(self.to_pandas(name, kind, per_window=per_window), name)

    def to_dataset(self, *names: str, kind: str = 'primal', per_window: bool = False) -> xr.Dataset:
        """The named answers of one *kind* as one `xarray.Dataset`; all of that kind by default.

        Args:
            names: What to include; none means every name of *kind* the
                sweep has an answer for. A name an EachWindow sweep cannot
                stitch is left out, as an archive leaves out its file; named,
                or read ``per_window``, it is read as [`primal`][] reads it.
            kind: As [`scan`][] takes it.
            per_window: Read an EachWindow sweep one window at a time instead
                of its answer.

        Raises:
            SpecsolveError: The sweep has no answer of *kind* at all — the
                message names each name it left out, and why — or as
                [`primal`][] raises.
        """
        held = names or self._names_held(kind, per_window=per_window)
        return tidy_to_dataset(held, lambda name: self.to_dataarray(name, kind, per_window=per_window))

    def save(self, directory: str | Path) -> Path:
        """Everything the sweep holds, per slice, in the layout ``spill_to=`` writes.

        ``<kind>/<name>/<position>.parquet`` for every primal, dual,
        expression and output it carries, the slice key a column of each, with ``record/``,
        ``metrics/`` and the manifest beside them. [`scan`][] reads it, and
        the call that made this sweep, pointed at it with ``spill_to=``, reads
        it back without solving a slice. A sweep whose every slice terminated
        without values writes each slice's record and no frames.

        Whatever a previous save left in *directory* is removed first; files
        outside the layout are left alone. A sweep whose slices already lie
        in *directory* is left as it is.

        Returns:
            The directory.

        Raises:
            SpecsolveError: This polars has no parquet.
        """
        parquet.require('Sweep.save')
        return self._save(Path(directory))

    def _save(self, out: Path, write: FrameWriter | None = None) -> Path:
        """[`save`][], each table written with *write*: an archive passes the writer that adds its run.

        A sweep whose slices lie on disk is copied file for file, so one too
        large to hold is saved without being held.
        """
        on_disk = self._spill
        if on_disk is not None and on_disk.directory.resolve() == out.resolve():
            return out
        by_slice = self._frames_by_slice() if on_disk is None else []
        clear_the_answer(out)
        spill = Spill.opened(
            out,
            self.key_name,
            self.keys,
            self.records[self.key_name].dtype,
            self._stitch,
            self._outputs,
            self.spec,
            write,
        )
        write_reasons(spill.directory, self._no_duals, self._absent, spill._write)
        if on_disk is not None:
            spill.take(on_disk)
        records = self.records.select(Record._fields)
        metrics = self.metrics.select(Metrics._fields)
        for position, key, frames in by_slice:
            spill.write_tables(position, key, frames, records.slice(position, 1), metrics.slice(position, 1))
        return spill.directory

    def _frames_by_slice(
        self, kinds: Sequence[str] = KINDS
    ) -> list[tuple[int, Label, dict[str, dict[str, pl.DataFrame]]]]:
        """``(position, key, {kind: {name: frame}})`` per slice over *kinds*, the key column dropped; a slice with no rows of a name lacks it."""
        by_key = {kind: self._by_slice(kind) for kind in kinds}
        return [
            (
                position,
                key,
                {
                    kind: {name: keyed[key] for name, keyed in names.items() if key in keyed}
                    for kind, names in by_key.items()
                },
            )
            for position, key in enumerate(self.keys)
        ]

    def _by_slice(self, kind: str) -> dict[str, dict[Label, pl.DataFrame]]:
        """``{name: {slice key: frame}}`` for *kind*, the key column dropped; a slice with no rows is absent."""
        key = self.key_name
        return {
            name: {
                part[key][0]: part.drop(key) for part in frame.pipe(collected).partition_by(key, maintain_order=True)
            }
            for name, frame in self._slices.get(kind, {}).items()
        }

    def _names_held(self, kind: str, *, per_window: bool) -> tuple[str, ...]:
        """Every name of *kind* there is an answer for, sorted; none at all is refused.

        Per window, every name the windows hold. A live sweep leaves out what
        ``_unstitchable`` refuses, which an archive's answer already lacks.
        """
        kind = checked_kind(kind)
        self._check_requested(kind, 'anything')
        if per_window:
            self._check_per_window()
        held, left_out = self._answerable(kind, per_window=per_window)
        if not held:
            absent = _whole_kind_absent(kind, held, self._no_duals, self.records) or _none_answered(
                _LABELS[kind], left_out
            )
            raise SpecsolveError(absent or _nothing_to_read(_LABELS[kind], 'anything', held, self.records))
        return tuple(sorted(held))

    def _answerable(self, kind: str, *, per_window: bool) -> tuple[dict[str, object], dict[str, str]]:
        """Every name of *kind* there is an answer for, and why each name left out of the answer is."""
        left_out = dict(self._absent.get(kind, {}))
        held: dict[str, object] = {}
        for name, frame in self._slices.get(kind, {}).items():
            if not per_window and (why := self._unstitchable(frame)):
                left_out[name] = why
            else:
                held[name] = frame
        return held, left_out

    def __len__(self) -> int:
        return self.records.height


def _whole_kind_absent(
    kind: str, held: Mapping[str, object], no_duals: str | None, records: pl.DataFrame
) -> str | None:
    """Why no slice holds *kind*: the duals' reason for a priced kind, or [`NO_BASIS`][] where slices solved and none ended on a basis."""
    if kind in PRICED:
        return no_duals
    if kind in BASES and not held and records['has_primal'].any():
        return NO_BASIS
    return None


def _none_answered(kind: str, left_out: Mapping[str, str]) -> str | None:
    """The message for a sweep whose every *kind* was left out of its answer, or ``None`` where none was."""
    if not left_out:
        return None
    return f'no {kind} of this sweep has an answer, and each one says why:\n{bulleted(dict(sorted(left_out.items())))}'


def _nothing_to_read(kind: str, name: str, held: Mapping[str, object], records: pl.DataFrame) -> str:
    """The message for *name* having no frame, whether undeclared or produced by no slice.

    A sweep with no frame of a kind solved nothing only where no slice left
    values: a spec that declares no expression leaves none however it solved.
    """
    conditions = ', '.join(sorted(set(records['termination_condition'].to_list())))
    if held:
        listed = ', '.join(repr(k) for k in sorted(held))
        return (
            f'no {kind} {name!r} in this sweep — it holds {listed}. '
            f'If the spec declares it, no slice produced one: all {records.height} terminated {conditions}.'
        )
    if solved := records['has_primal'].sum():
        return (
            f'this sweep holds no {kind} frames, though {solved} of its {records.height} slices left values '
            f'(they terminated {conditions}). If the spec declares one, no slice produced it.'
        )
    return (
        f'this sweep holds no {kind} frames at all — every one of its {records.height} slices '
        f'terminated {conditions}. The fold ran; the models did not solve. '
        f'sweep.records carries the status of each slice.'
    )


def load_sweep(directory: str | Path) -> Sweep:
    """Read back a sweep [`Sweep.save`][] wrote, or one ``solve_over(spill_to=)`` spilled.

    Every slice's frames are in memory when this returns, so the sweep owes
    *directory* nothing afterwards; a sweep larger than memory is
    [`scan_sweep`][] instead. The readers return the answer, the manifest
    carrying what each window owns.

    Returns:
        The sweep, keyed as it was solved.

    Raises:
        LayoutError: *directory* holds no ``sweep.json``, misses a record every
            fold writes, or is in a layout that has moved since it was written.
    """
    under = Path(directory)
    opened = _opened_sweep(under)
    spill = Spill(under, opened.key_name, opened.records[opened.key_name].dtype)
    return replace(opened, _slices=spill.frames(eager=True))


def scan_sweep(directory: str | Path) -> Sweep:
    """The sweep under *directory*, its frames left where they lie; *directory* has to outlive it.

    [`load_sweep`][]'s other half, and what a sweep solved with ``spill_to=``
    already is: only the record is read until a reader asks for a name.
    [`Sweep.primal`][] and its siblings read that name into memory;
    [`Sweep.scan`][] hands it back as a `polars.LazyFrame`, for a name too
    large to hold.

    Raises:
        LayoutError: As [`load_sweep`][] raises it.
    """
    under = Path(directory)
    opened = _opened_sweep(under)
    spill = Spill(under, opened.key_name, opened.records[opened.key_name].dtype)
    return replace(opened, _slices=spill.frames(eager=False), _spill=spill)


def _opened_sweep(under: Path) -> Sweep:
    """The sweep whose manifest is under *under*: its records and reasons, and none of its frames."""
    parquet.require('reading a saved sweep')
    manifest = under / MANIFEST_FILE
    if not manifest.is_file():
        raise LayoutError(
            f'{str(under)!r} holds no {MANIFEST_FILE!r}, so it is not a sweep save() or spill_to= wrote. A '
            f'single solve writes no manifest and is read by load_result.'
        )
    check_format(under)
    found = json.loads(manifest.read_text())
    stitch = found['stitch']
    no_duals, absent = read_reasons(under)
    key_name = found['key_name']
    keys = parquet.read(under / KEYS_FILE, columns=[key_name]).to_series()
    return Sweep(
        key_name=key_name,
        records=_rekeyed(consolidated(under, RECORD_FILE), keys, found['keys']),
        metrics=_rekeyed(consolidated(under, METRICS_FILE), keys, found['keys']),
        spec=read_spec(under),
        _no_duals=no_duals,
        _absent=absent,
        _outputs=read_outputs(under),
        _stitch=None
        if stitch is None
        else Stitch(stitch['dim'], parquet.read(under / OWNED_FILE).drop(RUN, strict=False)),
    )


def _rekeyed(table: pl.DataFrame, keys: pl.Series, texts: Sequence[str]) -> pl.DataFrame:
    """*table* with the typed key prepended, matched on the text each row's ``slice`` holds.

    Matched rather than placed by position, so a spill an interrupted fold
    left reads back the slices it finished. *texts* are the keys as the solve
    wrote them, in the order of *keys*: an archive holds a datetime key in
    UTC, whose own text is no longer the one a record names its slice by.
    """
    return table.select(pl.col('slice').replace_strict(texts, keys, return_dtype=keys.dtype).alias(keys.name), pl.all())
