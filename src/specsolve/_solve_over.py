"""``solve_over``: one spec solved once per slice of an axis, and the answers folded into a sweep.

A plan cannot contain a loop; a process may loop over plans. This is the
driver above [`specsolve._api`][], built from the public verbs.

    scenario / sweep    ``EachLabel('scenario')``            independent
    myopic pathway      ``EachLabel('period', carry=...)``
    rolling horizon     ``EachWindow('snapshot', steps=24, lookahead=24, carry=...)``

The axes are [`specsolve._axes`][] and what a fold returns is
[`specsolve._sweep`][]. The caller-facing rules are [sweeps](https://specsolve.readthedocs.io/en/latest/reference/sweeps/).
"""

from __future__ import annotations

import io
from collections.abc import Iterator, Mapping
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from specsolve import _parquet as parquet
from specsolve._api import build, check
from specsolve._archive_layout import check_the_target, write_archive
from specsolve._axes import Axis, EachWindow, as_built, axis_from, carry_plan, sources_with_column
from specsolve._collect import collected
from specsolve._frames import as_frame
from specsolve._inputs import as_spec
from specsolve._names import VALUE, refuse_reserved
from specsolve._outputs import checked_outputs
from specsolve._records import Metrics, Record
from specsolve._sources import tidy_sources, with_positions
from specsolve._start import slice_starts
from specsolve._sweep import SliceAnswer, Spill, Sweep, one_key_type
from specsolve.errors import DataError, SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence

    import polars as pl
    from mathspec.program import Program
    from mathspec.spec import Spec

    from specsolve._api import Model
    from specsolve._axes import CarryRule, Slice
    from specsolve._inputs import Label, Source, SpecLike
    from specsolve._outputs import Output
    from specsolve._result import Diagnostics, Result
    from specsolve._start import Start


def _slice_metrics(after: Diagnostics, before: Diagnostics | None) -> Metrics:
    """One slice's row of [`Sweep.metrics`][], off the model's cumulative counters.

    A serial fold's model keeps summing across slices, so this slice's share is
    *after* minus *before*; a model built for one slice has no *before*.
    """
    now = after.metrics()
    return now if before is None else now.since(before.metrics())


def solve_over(
    spec: SpecLike,
    sources: Mapping[str, Source],
    over: Axis | Mapping[str, Any],  # pyrefly: ignore[explicit-any] — plain data, as JSON reads it
    *,
    executor: Executor | None = None,
    shared_filesystem: bool | None = None,
    solver_options: Mapping[str, object] | None = None,
    also_record: Sequence[str] | None = None,
    solver_name: str = 'highs',
    spill_to: str | Path | None = None,
    archive: str | Path | None = None,
    outputs: Iterable[Output] = (),
    start: Sweep | Result | Start | Literal['previous'] | None = 'previous',
) -> Sweep:
    """Solve *spec* once per slice of *over* and fold the answers together.

    The rules are [sweeps](https://specsolve.readthedocs.io/en/latest/reference/sweeps/).

    Args:
        spec: As [`check`][specsolve._api.check] takes it. Parsed once, whichever
            executor runs the slices.
        sources: As [`build`][specsolve._api.build] takes them, every shape
            included; the axis filters the parameters and relations that carry
            it and passes the rest through.
        over: The axis: [`EachLabel`][] or [`EachWindow`][], or the plain
            data its ``to_dict`` writes. An axis's ``carry=`` chains its
            slices, and its ``key_name=`` names the slice column.
        executor: Any `concurrent.futures.Executor`; ``None`` runs the
            slices in order on one model. A process pool must be ``spawn``
            or ``forkserver`` — a forked worker hangs.
        shared_filesystem: Whether the executor's workers can read this
            process's paths. Decided for the stdlib pools; anything else is
            assumed not to, and paths travel as bytes.
        solver_options: As [`solve`][specsolve._api.Model.solve] takes them.
        also_record: As [`solve`][specsolve._api.Model.solve] takes them,
            reaching every slice.
        solver_name: As [`solve`][specsolve._api.Model.solve] takes it.
        spill_to: A directory each slice's frames are written to as the fold
            goes, so the sweep holds one slice in memory however many there
            are; [`Sweep.scan`][] reads a name off it lazily. A directory
            holds one sweep: the same sweep run at it again does not solve the
            slices already there, which is how an interrupted sweep resumes.
        archive: Where to write the model, the sources the sweep was cut from,
            the axis that cut them and every slice's answer, so that
            ``sps.load_archive`` gives all four back and the sweep runs again
            from the file alone. A ``.zip`` suffix packs it into one file;
            anything else is a directory. The answer is in the layout
            *spill_to* writes, one file per slice of each name under
            ``answer/<kind>/<name>/``, so [`Sweep.save`][], a spill and an
            archive hold a sweep the same way. Beside *spill_to* the archive
            copies the spill file for file, so a sweep too large to hold is
            archived without being held; the archive is a second copy of the
            answers on disk.
            A sliced source is archived whole.
        outputs: As [`solve`][specsolve._api.Model.solve] takes them, reaching
            every slice. The sweep, its spill and its archive carry these and
            nothing else; a *spill_to* directory solved with others is
            refused rather than resumed.
        start: What each slice starts from, as
            [`solve`][specsolve._api.Model.solve] takes it: an earlier answer,
            an earlier sweep, or a [`Start`][specsolve.types.Start] of tables.
            Each table is cut by the axis as a source is: one that carries the
            sliced dimension gives each slice its own rows, and one without
            reaches every slice whole. An
            earlier sweep is its answer, so each slice starts from the earlier
            slice of its key, and each window from the hours it covers.
            ``'previous'``, the default, carries each slice on from the one
            before it on the model updated in place, as
            [`solve`][specsolve._api.Model.solve] does, and under an executor
            starts every slice from nothing. ``None`` starts every slice from
            nothing.

    Returns:
        The sweep, which reads its answer.

    Raises:
        SpecsolveError: Before a slice is taken: an *over* that is no axis,
            or a mapping whose keys name none; a carry that cannot line up,
            has no seed, collapses a dimension the axis does not advance
            along, or is asked together with an executor; a key that collides
            with a column the frames carry; an axis the program does not
            allow; a *spill_to* directory holding another sweep; *outputs* that [`solve`][specsolve._api.Model.solve]
            refuses; a *start* word other than ``'previous'`` — each answerable from the declarations alone; keys of more than one
            type, or two keys of one text; or a *start*
            [`solve`][specsolve._api.Model.solve] refuses.
        DataError: No source carries an EachLabel axis, an index
            carries it, an EachWindow axis's dimension has no index, or the
            axis produced no slices.

    Warns:
        SpecsolveWarning: A source carrying an EachLabel axis that is
            short of a label another has — that slice builds it empty —
            or a *start* that leaves a slice no row, which starts that slice
            from nothing.
    """
    axis = axis_from(over) if isinstance(over, Mapping) else over
    if not isinstance(axis, Axis):
        raise SpecsolveError(
            f"over= takes EachLabel or EachWindow, or the mapping an axis's to_dict writes (got {type(over).__name__})."
        )
    carry = axis.carry
    if carry and executor is not None:
        raise SpecsolveError(
            'carry and executor are mutually exclusive: a carried value makes slice i+1 depend on '
            "slice i's answer, so the slices cannot run concurrently. Drop the executor, or drop the carry."
        )
    if archive is not None or spill_to is not None:
        parquet.require('archive=' if archive is not None else 'spill_to=')
    document = as_spec(spec)
    sources = _materialised(sources)
    out = None if archive is None else Path(archive)
    if out is not None:
        check_the_target(out)
    asked = checked_outputs(outputs)
    program = check(document)
    plan = carry_plan(program, axis)
    key_name = _key_column(axis, program)

    _check_the_carry(plan, axis, sources, program)
    axis._check_the_program(program, sources)
    slices, stitch = axis._slice(program, sources, key_name)
    if not slices:
        raise DataError('the axis produced no slices')
    solving = {
        'solver_name': solver_name,
        'solver_options': dict(solver_options or {}) or None,
        'also_record': also_record,
        'outputs': asked,
    }
    keys = [current.key for current in slices]
    key_dtype = one_key_type(keys, key_name)
    starts = slice_starts(start, axis, slices, program)
    spill = None if spill_to is None else Spill.opened(spill_to, key_name, keys, key_dtype, stitch, asked, document)
    if executor is None:
        answered = _serially(document, slices, solving, plan, spill, starts)
    else:
        answered = _pooled(executor, shared_filesystem, document, slices, solving, spill, starts)
    folded = Sweep._folded(key_name, stitch, answered, spill, key_dtype, asked, document)
    if out is not None:
        _archive_the_sweep(out, document, program, axis, sources, folded, slices[0].sources)
    return folded


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
    sources: Mapping[str, Source],
    folded: Sweep,
    one_slice: Mapping[str, Source],
) -> None:
    """Write the sweep's question and its answer to *out*.

    A windowed sweep's sources are the ones a single solve attaches, so they
    are tidied whole. An [`EachLabel`][] sweep's are tidied from *one_slice*, except
    the ones the axis cut, which [`_uncut`][] gives.
    """
    if isinstance(axis, EachWindow):
        tables = with_positions(program, tidy_sources(program, sources))
    else:
        tidied = with_positions(program, tidy_sources(program, one_slice))
        cut = {name: _uncut(program, axis, name, t) for name, t in sources_with_column(sources, axis.dim).items()}
        held = {**tidied, **cut}
        tables = {name: held[name] for name in sources}
    write_archive(out, spec, tables, axis=axis.to_dict(), answer=folded._save)


def _uncut(program: Program, axis: Axis, name: str, table: pl.LazyFrame) -> pl.LazyFrame:
    """A source an [`EachLabel`][] sweep cuts, as its tidy columns with the axis column first."""
    declared = [*program.parameters[name].dims, VALUE] if name in program.parameters else program.relations[name].roles
    return table.select(axis.dim, *declared)


def _check_the_carry(
    plan: Mapping[str, CarryRule],
    axis: Axis,
    first: Mapping[str, Source],
    program: Program,
) -> None:
    """Refuse a carry with no seed, one that keeps the dimension an [`EachWindow`][] cuts, or one that collapses another.

    Reads no source: the seed is a key of *first*, and the owned dimension is
    the axis's own.
    """
    for parameter, rule in plan.items():
        if parameter not in first:
            raise SpecsolveError(
                f'carry writes {parameter!r} from the second slice on, and the first slice has nothing to '
                f'start from: supply {parameter!r} in sources as the seed.'
            )
        if isinstance(axis, EachWindow) and axis.dim in program.parameters[parameter].dims:
            raise SpecsolveError(
                f'carry {parameter!r} <- {rule.variable!r} keeps {axis.dim!r}, the dimension the sweep windows, '
                f'and no two windows hold the same labels: the rows one window solved are at labels the next '
                f'does not have. Carry into a parameter that is not over {axis.dim!r}, which takes the row at '
                f'the last label a window owns.'
            )
        if rule.dropped is None:
            continue
        if isinstance(axis, EachWindow) and rule.dropped == axis.dim:
            continue
        owned = f'this axis advances along {axis.dim!r}' if isinstance(axis, EachWindow) else 'this axis owns none'
        raise SpecsolveError(
            f'carry {parameter!r} <- {rule.variable!r} collapses {rule.dropped!r}, and {owned}: a carry hands '
            f'on the last label a slice owns, so it can only collapse the dimension the sweep advances '
            f'along. Reduce {rule.dropped!r} in the YAML — a derived variable is where the oracle can see the '
            f'math — so that {parameter!r} and {rule.variable!r} are over the same dimensions.'
        )


def _serially(
    document: Spec,
    slices: Sequence[Slice],
    solving: Mapping[str, Any],  # pyrefly: ignore[explicit-any] — the verb's own keywords, forwarded
    plan: Mapping[str, CarryRule],
    spill: Spill | None,
    starts: Sequence[Start | Literal['previous'] | None],
) -> Generator[tuple[Label, SliceAnswer], None, None]:
    """Each slice's answer, off one model updated in place.

    A generator, since a slice is built from what the one before it left;
    the caller closes it to release the model. A slice the spill holds is
    read back, and one solved here is written before it is yielded.
    """
    model: Model | None = None
    carried = {rule.variable for rule in plan.values()}
    last: tuple[int, SliceAnswer, Mapping[str, pl.DataFrame]] | None = None

    def left(before: Slice) -> Mapping[str, pl.DataFrame]:
        """The primal *before* left for the slice after it; a carried sweep stops at a slice that left none."""
        assert last is not None, 'a slice is asked for only after it was solved or read back'
        position, answer, primals = last
        if not primals:
            raise SpecsolveError(
                f'slice {before.key!r} ({position + 1} of {len(slices)}) terminated '
                f'{answer.record.termination_condition}, so slice {slices[position + 1].key!r} has no '
                f'{sorted(carried)} to start from. A carried sweep '
                f'stops at the first slice that leaves nothing to carry; the {position} before it solved.'
            )
        return primals

    try:
        for position, current in enumerate(as_built(slices, plan, left)):
            if spill is not None and spill.done(position):
                answer = spill.read_back(position)
                primals = spill.written('primal', position, carried)
            else:
                with _named_slice(current.key, position, len(slices)):
                    if model is None:
                        model, before = build(document, current.sources), None
                    else:
                        before = model.diagnostics()
                        model.update(current.sources)
                    result = model.solve(**solving, start=starts[position])
                    answer = _answers(result, _slice_metrics(model.diagnostics(), before))
                primals = answer.frames.get('primal', {})
                if spill is not None:
                    answer = spill.write(position, current.key, answer)
            last = (position, answer, primals)
            yield current.key, answer
    finally:
        if model is not None:
            model.close()


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
    shared_filesystem: bool | None,
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
    shared = _shares_filesystem(executor, shared_filesystem)
    memo: dict[str, tuple[Any, Any]] = {}  # pyrefly: ignore[explicit-any] — a source beside its encoding
    futures = [
        None
        if spill is not None and spill.done(position)
        else executor.submit(
            _run_slice,
            document,
            _encode(current.sources, memo, shared_filesystem=shared) if crosses else dict(current.sources),
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


def _answers(result: Result, metrics: Metrics) -> SliceAnswer:
    """One slice's answer as plain data, read out of *result*, every declared expression evaluated now.

    A slice with no primal, or with undefined duals, is not a failure: the
    reason ``Result.dual`` gives is carried.
    """
    if not result.has_primal:
        return SliceAnswer(result.record, metrics)
    held, no_expressions = result._all_frames()
    frames = {
        kind: {name: frame.lazy().pipe(collected) for name, frame in named.items()} for kind, named in held.items()
    }
    return SliceAnswer(result.record, metrics, frames, result._no_duals, no_expressions)


def _run_slice(
    document: Spec,
    encoded: dict[str, Any],  # pyrefly: ignore[explicit-any] — what crossed to the worker
    encode_out: bool,
    call: Mapping[str, Any],  # pyrefly: ignore[explicit-any] — the verb's own keywords, forwarded
    start: Start | Literal['previous'] | None,
) -> SliceAnswer:
    """One slice, start to finish, over plain data; module-level so a remote executor can pickle it."""
    with build(document, _decode(encoded)) as model, model.solve(**call, start=start) as result:
        answer = _answers(result, _slice_metrics(model.diagnostics(), None))
        if not encode_out:
            return answer
        return replace(answer, frames={kind: _encode(named, {}) for kind, named in answer.frames.items()})


def _key_column(axis: Axis, program: Program) -> str:
    """What *axis* calls the column holding the slice key; never a column the frames already carry."""
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
    shared_filesystem: bool = False,
) -> dict[str, Any]:  # pyrefly: ignore[explicit-any] — a frame crosses as parquet bytes
    """Sources in the shape a worker can be handed.

    A path the workers reach stays a path; one they cannot travels as its own
    bytes. A table becomes parquet ``bytes``; anything else crosses as itself.
    *memo* encodes a source no slice rewrote once.
    """
    out: dict[str, Any] = {}  # pyrefly: ignore[explicit-any] — a frame crosses as parquet bytes
    for name, obj in sources.items():
        if isinstance(obj, (str, Path)) and shared_filesystem:
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
            parquet.write(table.pipe(collected), buffer)
            out[name] = buffer.getvalue()
        memo[name] = (obj, out[name])
    return out


def _decode(encoded: Mapping[str, Any]) -> dict[str, Any]:  # pyrefly: ignore[explicit-any] — a frame crosses as parquet bytes
    """The inverse of [`_encode`][]; anything not ``bytes`` passes through."""
    return {name: parquet.read(io.BytesIO(v)) if isinstance(v, bytes) else v for name, v in encoded.items()}
