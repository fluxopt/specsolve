"""What a sweep returns: the answer over the model's own coordinates, a record per slice, and its spill on disk.

[`load_sweep`][] and [`scan_sweep`][] read one back. The caller-facing rules
are [sweeps](https://specsolve.readthedocs.io/en/latest/reference/sweeps/).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import polars as pl

from specsolve.axes import Stitch, bulleted
from specsolve.errors import LayoutError, SpecsolveError
from specsolve.messages import no_model_behind_this_answer_message
from specsolve.relational.answer_layout import (
    BASES,
    KINDS,
    METRICS_FILE,
    METRICS_SCHEMA,
    NO_BASIS,
    OUTPUT_KINDS,
    PRICED,
    RECORD_FILE,
    RECORD_SCHEMA,
    RUN,
    Metrics,
    Record,
    check_format,
    checked_kind,
    consolidated,
    not_requested_message,
    read_outputs,
    read_reasons,
    row_of,
    write_format,
    write_reasons,
    write_whole,
)
from specsolve.relational.result import tidy_to_dataarray, tidy_to_dataset, tidy_to_pandas

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence

    import pandas as pd
    import xarray as xr

    from specsolve.inputs import Label
    from specsolve.relational.answer_layout import Output
    from specsolve.relational.result import Start


#: What each of the [`KINDS`][specsolve.relational.answer_layout.KINDS] is a frame of, as a message names it.
_LABELS: Mapping[str, str] = MappingProxyType(
    {
        'primal': 'variable',
        'dual': 'constraint',
        'expression': 'named expression',
        **{kind: carried.per for kind, carried in OUTPUT_KINDS.items()},
    }
)


#: A spilled sweep's manifest, its keys as their own type, and the coordinates
#: each window owns.
MANIFEST_FILE = 'sweep.json'
KEYS_FILE = 'keys.parquet'
OWNED_FILE = 'owned.parquet'


@dataclass(frozen=True)
class SliceAnswer:
    """One slice, solved and read out; plain data only, so it can cross a process."""

    meta: Record
    #: This slice's row of [`Sweep.metrics`][].
    metrics: Metrics
    #: ``{kind: {name: frame}}`` over [`KINDS`][specsolve.relational.answer_layout.KINDS]:
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
            meta=self.meta._replace(slice_axis=key_name, slice=text),
            metrics=self.metrics._replace(slice_axis=key_name, slice=text),
        )


def one_key_type(keys: Sequence[Label], key_name: str) -> pl.DataType:
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

    ``_rekeyed`` matches each row's ``slice`` text against the keys' text in
    the sweep's one type, when the fold ends and when a spill is scanned, so
    a key that type rewrites, or two keys of one text, would fail there,
    after every slice has solved.
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


def with_key(frame: pl.DataFrame, key_name: str, key: Label, dtype: pl.DataType) -> pl.DataFrame:
    """*frame* with the slice key prepended as *dtype*, the whole sweep's type rather than ``pl.lit``'s."""
    return frame.select(pl.lit(key, dtype=dtype).alias(key_name), pl.all())


@dataclass(frozen=True)
class Spill:
    """A sweep's answers on disk instead of in memory, one file per slice and name.

    ``<kind>/<name>/<position>.parquet`` holds the keyed frames, and
    ``record/`` and ``metrics/`` the rows, which name their slice as text in
    ``slice_axis`` and ``slice``; ``keys.parquet`` holds the keys as their own
    type. ``sweep.json`` names the key, the keys and the outputs, so a
    directory answers for one sweep. Every file lands whole. A slice's record file is written
    last and marks it done; ``sweep.json`` lands after the files a scan reads
    beside it and marks the directory stamped.
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
        stitch: Stitch | None,
        outputs: frozenset[Output],
    ) -> Spill:
        """The directory ready to take this sweep: stamped if it holds none, checked and never re-stamped if it does."""
        directory = Path(directory)
        manifest: dict[str, object] = {
            'key_name': key_name,
            'keys': [str(key) for key in keys],
            'stitch': None if stitch is None else {'local': stitch.local, 'dim': stitch.dim},
            'outputs': sorted(outputs),
        }
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
            write_whole(pl.DataFrame([pl.Series(key_name, keys, dtype=key_dtype)]), directory / KEYS_FILE)
            if stitch is not None:
                write_whole(stitch.owned, directory / OWNED_FILE)
            record.write_text(json.dumps(manifest))
        return cls(directory, key_name, key_dtype)

    def _file(self, kind: str, position: int, name: str | None = None) -> Path:
        under = self.directory / kind if name is None else self.directory / kind / name
        return under / f'{position:06d}.parquet'

    def done(self, position: int) -> bool:
        return self._file('record', position).exists()

    def write(self, position: int, key: Label, answer: SliceAnswer) -> SliceAnswer:
        """*answer*'s frames and record on disk, and the answer with the frames released."""
        answer = answer.sliced(self.key_name, key)
        for kind, produced in answer.frames.items():
            for name, frame in produced.items():
                write_whole(with_key(frame, self.key_name, key, self.key_dtype), self._file(kind, position, name))
        write_whole(pl.DataFrame([answer.metrics._asdict()], schema=METRICS_SCHEMA), self._file('metrics', position))
        write_whole(pl.DataFrame([answer.meta._asdict()], schema=RECORD_SCHEMA), self._file('record', position))
        return replace(answer, frames={})

    def read_back(self, position: int) -> SliceAnswer:
        """A done slice's record, with no frames."""
        row = pl.read_parquet(self._file('record', position)).row(0, named=True)
        held = pl.read_parquet(self._file('metrics', position)).row(0, named=True)
        return SliceAnswer(row_of(Record, row, self.directory), row_of(Metrics, held, self.directory))

    def written(self, kind: str, position: int, names: Iterable[str]) -> dict[str, pl.DataFrame]:
        """The named frames of *kind* a done slice wrote; a name it did not write is absent."""
        found = {name: self._file(kind, position, name) for name in names}
        return {name: pl.read_parquet(path).drop(self.key_name) for name, path in found.items() if path.exists()}

    def frames(self, *, whole: bool) -> dict[str, dict[str, pl.LazyFrame]]:
        """``{kind: {name: frame}}``, every slice's frame of a name as one, keyed and in slice order.

        A kind or a name no slice wrote is absent. *whole* reads the files into
        memory now, rather than as a `polars.scan_parquet` collected at the
        first read.
        """
        out: dict[str, dict[str, pl.LazyFrame]] = {}
        for kind in KINDS:
            under = self.directory / kind
            for named in sorted(under.iterdir()) if under.is_dir() else []:
                files = sorted(named.glob('*.parquet'))
                frame = pl.read_parquet(files).lazy() if whole else pl.scan_parquet(files)
                out.setdefault(kind, {})[named.name] = frame.drop(RUN, strict=False)
        return out


@dataclass(frozen=True)
class Sweep:
    """What a fold returned: the answer over the model's own coordinates, and a record per slice.

    [`Result`][specsolve.relational.result.Result]'s readers, same names and
    shapes, and each returns **the answer** by default. An [`EachWindow`][]
    sweep is read over the dimension it sliced: each coordinate comes from
    the window that owns it, and the lookahead rows are dropped. An
    [`EachCoordinate`][] or hand-built sweep is keyed by slice, the key
    prepended, each slice being a whole answer. Nothing is combined across
    slices.

    ``per_window=True`` reads an EachWindow sweep one window at a time:
    keyed by where each window started, over the index inside it, lookahead
    rows included.
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
    #: ``{kind: {name: frame}}``, every slice's frame of a name as one, keyed,
    #: in memory or scanned off disk; a kind no slice produced is absent.
    _slices: dict[str, dict[str, pl.LazyFrame]] = field(repr=False, default_factory=dict)
    _no_duals: str | None = field(repr=False, default=None)
    #: ``{kind: {name: reason}}`` for a name some slice could not produce, or
    #: one an archive holds no answer for.
    _absent: dict[str, dict[str, str]] = field(repr=False, default_factory=dict)
    _stitch: Stitch | None = field(repr=False, default=None)
    #: The directory the slices lie in, for a sweep solved with ``spill_to=``
    #: or scanned off one; what an archive of it is packed from.
    _spill: Spill | None = field(repr=False, default=None)
    #: ``{kind: {name: frame}}``, the answer as a sweep archive holds it, read
    #: instead of folding it from the slices; ``None`` everywhere else.
    _answer: dict[str, dict[str, pl.LazyFrame]] | None = field(repr=False, default=None)
    #: Whether the per-window frames are there to read: not on an archive
    #: written without ``keep_windows=True``.
    _windows: bool = field(repr=False, default=True)
    #: Each [`Output`][specsolve.relational.answer_layout.Output] the sweep was
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
                        frames[kind][name].append(with_key(frame, key_name, key, key_dtype))
        keyed = pl.Series(key_name, keys, dtype=key_dtype)
        if spill is not None:
            slices = spill.frames(whole=False)
        else:
            slices = {
                kind: {name: pl.concat(held).lazy() for name, held in named.items()} for kind, named in frames.items()
            }
        return cls(
            key_name=key_name,
            record=_rekeyed(pl.DataFrame([row._asdict() for row in rows], schema=RECORD_SCHEMA), keyed),
            metrics=_rekeyed(pl.DataFrame([row._asdict() for row in taken], schema=METRICS_SCHEMA), keyed),
            _slices=slices,
            _no_duals=no_duals,
            _absent={'expression': no_expressions} if no_expressions else {},
            _stitch=stitch,
            _spill=spill,
            _outputs=outputs,
        )

    @property
    def keys(self) -> list[Label]:
        return self.record[self.key_name].to_list()

    def _start(self) -> Start:
        """This sweep's answer as tables to start a sweep from, keyed as [`Start`][specsolve.types.Start] is.

        The primal, and the basis where the sweep carries one, each read as
        [`scan`][] reads it: keyed by slice, or over the dimension an
        EachWindow sweep cut.
        """
        given: dict[str, dict[str, pl.DataFrame]] = {}
        for kind in ('primal', *(sorted(BASES) if 'basis' in self._outputs else ())):
            held, _ = self._answerable(kind, per_window=False)
            if held:
                given[kind] = {name: self.scan(name, kind).collect() for name in held}
        return cast('Start', given)

    def _check_per_window(self) -> None:
        """Refuse ``per_window=True`` where there are no windows to read."""
        if self._stitch is None:
            raise SpecsolveError(
                f'per_window=True reads an EachWindow sweep one window at a time, and this sweep was not cut '
                f'into windows: its answer already is one frame per slice, keyed by {self.key_name!r}. Read '
                f'it without per_window.'
            )
        if not self._windows:
            raise SpecsolveError(NO_WINDOWS)

    def _check_requested(self, kind: str, name: str) -> None:
        """Refuse a kind of the [`OUTPUT_KINDS`][specsolve.relational.answer_layout.OUTPUT_KINDS] the sweep was not asked for."""
        if kind in OUTPUT_KINDS and OUTPUT_KINDS[kind].output not in self._outputs:
            raise SpecsolveError(not_requested_message(kind, name))

    def _named(self, kind: str, name: str, *, per_window: bool) -> pl.LazyFrame:
        """*name*'s frame of *kind*, lazily: the answer, or the frames per window.

        An archive's answer is read off its own files; any other answer is
        stitched from the slices.
        """
        self._check_requested(kind, name)
        if per_window:
            self._check_per_window()
        answer = None if per_window else self._answer
        held = answer[kind] if answer is not None else self._slices.get(kind, {})
        frame = held.get(name)
        if frame is None:
            absent = self._absent.get(kind, {}).get(name) or _whole_kind_absent(kind, held, self._no_duals, self.record)
            raise SpecsolveError(absent or _nothing_to_read(_LABELS[kind], name, held, self.record))
        return frame if answer is not None else self._answered(frame, per_window=per_window)

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
        [`scan_archive`][specsolve.archive.scan_archive] — nothing is read
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
        shorter than the sweep; [`record`][] is one row per slice always.

        Args:
            name: A variable the sweep's spec declares.
            per_window: Read an EachWindow sweep one window at a time instead
                of its answer.

        Raises:
            SpecsolveError: No slice of the sweep produced *name*; a variable
                that is not over an EachWindow sweep's windowed dimension,
                which has an answer only per window; ``per_window`` on a sweep
                that was not cut into windows, or on an archive written
                without them.
        """
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
        return self._named('dual', name, per_window=per_window).collect()

    def activity(self, name: str, *, per_window: bool = False) -> pl.DataFrame:
        """One constraint's left-hand side at every slice's solution.

        [`primal`][]'s shape and arguments, and carried only where the sweep
        was asked for it with ``outputs={'activity'}``. In the answer of an
        EachWindow sweep each coordinate carries the window that owns it.

        Raises:
            SpecsolveError: The sweep was not asked for its activity, or as
                [`primal`][] raises.
        """
        return self._named('activity', name, per_window=per_window).collect()

    def slack(self, name: str, *, per_window: bool = False) -> pl.DataFrame:
        """One constraint's distance to binding at every slice's solution.

        [`primal`][]'s shape and arguments, and
        [`Result.slack`][specsolve.relational.result.Result.slack]'s sign.
        Carried only where the sweep was asked for it with
        ``outputs={'slack'}``.

        Raises:
            SpecsolveError: The sweep was not asked for its slack, or as
                [`primal`][] raises.
        """
        return self._named('slack', name, per_window=per_window).collect()

    def reduced_cost(self, name: str, *, per_window: bool = False) -> pl.DataFrame:
        """One variable's reduced costs at every slice's solution.

        [`primal`][]'s shape and arguments, and
        [`Result.reduced_cost`][specsolve.relational.result.Result.reduced_cost]'s
        sign. Carried only where the sweep was asked for it with
        ``outputs={'reduced_cost'}``. A slice whose model had an integer
        variable contributes none, as it contributes no duals.

        Raises:
            SpecsolveError: The sweep was not asked for reduced costs, no
                slice produced them — the message says why — or as
                [`primal`][] raises.
        """
        return self._named('reduced_cost', name, per_window=per_window).collect()

    def variable_basis(self, name: str, *, per_window: bool = False) -> pl.DataFrame:
        """One variable's basis status where every slice's solve ended.

        [`primal`][]'s shape and arguments, and
        [`Result.variable_basis`][specsolve.relational.result.Result.variable_basis]'s
        words. Carried only where the sweep was asked for it with
        ``outputs={'basis'}``. A slice that ended on no basis contributes
        none.

        Raises:
            SpecsolveError: The sweep was not asked for its basis, no slice
                ended on one, or as [`primal`][] raises.
        """
        return self._named('variable_basis', name, per_window=per_window).collect()

    def constraint_basis(self, name: str, *, per_window: bool = False) -> pl.DataFrame:
        """One constraint's basis status where every slice's solve ended.

        [`primal`][]'s shape and arguments, and
        [`Result.constraint_basis`][specsolve.relational.result.Result.constraint_basis]'s
        words. Carried only where the sweep was asked for it with
        ``outputs={'basis'}``. A slice that ended on no basis contributes
        none.

        Raises:
            SpecsolveError: The sweep was not asked for its basis, no slice
                ended on one, or as [`primal`][] raises.
        """
        return self._named('constraint_basis', name, per_window=per_window).collect()

    def evaluate(self, expression: str | Mapping[str, object], *, per_window: bool = False) -> pl.DataFrame:
        """The value of *expression* at every slice's solution, as an answer.

        [`evaluate`][specsolve.relational.result.Result.evaluate] over the
        sweep, with [`primal`][]'s shape. A declared name is read from what
        the sweep holds, live or off disk. Anything else is valued at each
        slice's own solution with no re-solve, which needs the spec, sources
        and axis the sweep [`load_archive`][specsolve.archive.load_archive]
        hands back carries; a Sweep a live solve returned retains no model. An
        expression over a parameter the sweep **carried** is refused, that
        value being a previous slice's answer rather than stored data.

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
                an undeclared expression on a Sweep with no model behind it, or
                one that reads a parameter the sweep carried; a quantity
                reduced over an EachWindow sweep's windowed dimension, which
                has an answer only per window; or ``per_window`` as
                [`primal`][] raises it.
            LanguageError: A construct outside the language, or a name the spec
                does not declare.
        """
        if isinstance(expression, str) and self._holds_expression(expression):
            return self._named('expression', expression, per_window=per_window).collect()
        if self._evaluate is None:
            raise SpecsolveError(self._nothing_to_evaluate(expression))
        if per_window:
            self._check_per_window()
        return self._answered(self._evaluate(expression), per_window=per_window)

    def _expression_names(self) -> Mapping[str, object]:
        """The declared expressions the sweep holds."""
        return self._answer['expression'] if self._answer is not None else self._slices.get('expression', {})

    def _holds_expression(self, name: str) -> bool:
        """Whether *name* is a declared expression the sweep holds, or one it says why it lacks."""
        return name in self._expression_names() or name in self._absent.get('expression', {})

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
        kind = checked_kind(kind)
        if kind == 'expression':
            return self.evaluate(name, per_window=per_window)
        return self._named(kind, name, per_window=per_window).collect()

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

        Returns:
            The directory.

        Raises:
            SpecsolveError: The sweep was read off an archive written without
                its windows.
        """
        by_key = {kind: slice_index(self, kind) for kind in KINDS}
        spill = Spill.opened(
            directory, self.key_name, self.keys, self.record[self.key_name].dtype, self._stitch, self._outputs
        )
        write_reasons(spill.directory, self._no_duals, self._absent)
        for position, key in enumerate(self.keys):
            meta = Record(**self.record.drop(self.key_name).row(position, named=True))
            taken = Metrics(**self.metrics.select(Metrics._fields).row(position, named=True))
            frames = {
                kind: {name: keyed[key] for name, keyed in names.items() if key in keyed}
                for kind, names in by_key.items()
            }
            answer = SliceAnswer(meta, taken, frames)
            spill.write(position, key, answer)
        return spill.directory

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
            absent = _whole_kind_absent(kind, held, self._no_duals, self.record) or _none_answered(
                _LABELS[kind], left_out
            )
            raise SpecsolveError(absent or _nothing_to_read(_LABELS[kind], 'anything', held, self.record))
        return tuple(sorted(held))

    def _answerable(self, kind: str, *, per_window: bool) -> tuple[dict[str, object], dict[str, str]]:
        """Every name of *kind* there is an answer for, and why each name left out of the answer is."""
        left_out = dict(self._absent.get(kind, {}))
        if self._answer is not None and not per_window:
            return dict(self._answer[kind]), left_out
        held: dict[str, object] = {}
        for name, frame in self._slices.get(kind, {}).items():
            if not per_window and (why := self._unstitchable(frame)):
                left_out[name] = why
            else:
                held[name] = frame
        return held, left_out

    def __len__(self) -> int:
        return self.record.height


#: Why an archive's sweep has nothing to read per window.
NO_WINDOWS = (
    'this archive holds the answer only, because it was written without keep_windows=True, so it has no '
    'per-window frames to read. Solving again from the archived spec and sources restores them: '
    'load_archive gives both, with the axis and the carry, so '
    'sps.solve_over(archive.spec, archive.sources, archive.axis, carry=archive.carry) runs the sweep again.'
)


#: Where an archive keeps a windowed sweep's per-window frames, under its ``answer/``.
WINDOWS_DIR = 'windows'


def _whole_kind_absent(kind: str, held: Mapping[str, object], no_duals: str | None, record: pl.DataFrame) -> str | None:
    """Why no slice holds *kind*: the duals' reason for a priced kind, or [`NO_BASIS`][] where slices solved and none ended on a basis."""
    if kind in PRICED:
        return no_duals
    if kind in BASES and not held and record['has_primal'].any():
        return NO_BASIS
    return None


def _none_answered(kind: str, left_out: Mapping[str, str]) -> str | None:
    """The message for a sweep whose every *kind* was left out of its answer, or ``None`` where none was."""
    if not left_out:
        return None
    return f'no {kind} of this sweep has an answer, and each one says why:\n{bulleted(dict(sorted(left_out.items())))}'


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
    opened = opened_sweep(under, under / OWNED_FILE)
    spill = Spill(under, opened.key_name, opened.record[opened.key_name].dtype)
    return replace(opened, _slices=spill.frames(whole=True))


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
    opened = opened_sweep(under, under / OWNED_FILE)
    spill = Spill(under, opened.key_name, opened.record[opened.key_name].dtype)
    return replace(opened, _slices=spill.frames(whole=False), _spill=spill)


def opened_sweep(under: Path, owned: Path | None) -> Sweep:
    """The sweep whose manifest is under *under*: its record and reasons, and none of its frames.

    *owned* is where an EachWindow sweep keeps what each window owns, read
    only by a stitch from the windows; ``None`` for an archive that kept no
    windows to stitch.
    """
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
    keys = pl.read_parquet(under / KEYS_FILE, columns=[key_name]).to_series()
    return Sweep(
        key_name=key_name,
        record=_rekeyed(consolidated(under, RECORD_FILE), keys),
        metrics=_rekeyed(consolidated(under, METRICS_FILE), keys),
        _no_duals=no_duals,
        _absent=absent,
        _outputs=read_outputs(under),
        _stitch=None
        if stitch is None
        else Stitch(
            stitch['local'],
            stitch['dim'],
            pl.DataFrame() if owned is None else pl.read_parquet(owned).drop(RUN, strict=False),
        ),
    )


def _rekeyed(table: pl.DataFrame, keys: pl.Series) -> pl.DataFrame:
    """*table* with the typed key prepended, matched on the text each row's ``slice`` holds.

    Matched rather than placed by position, so a spill an interrupted fold
    left reads back the slices it finished.
    """
    texts = [str(key) for key in keys.to_list()]
    return table.select(pl.col('slice').replace_strict(texts, keys, return_dtype=keys.dtype).alias(keys.name), pl.all())


def slice_index(sweep: Sweep, kind: str) -> dict[str, dict[Label, pl.DataFrame]]:
    """``{name: {slice key: frame}}`` for *kind*, the key column dropped; a slice with no rows is absent.

    An archived sweep that was not cut into windows holds its slices as its
    answer, keyed already. One that was holds them only where the windows were
    kept.
    """
    if sweep._answer is not None and sweep._stitch is None:
        held = sweep._answer[kind]
    elif not sweep._windows:
        raise SpecsolveError(NO_WINDOWS)
    else:
        held = sweep._slices.get(kind, {})
    key = sweep.key_name
    return {
        name: {part[key][0]: part.drop(key) for part in frame.collect().partition_by(key, maintain_order=True)}
        for name, frame in held.items()
    }
