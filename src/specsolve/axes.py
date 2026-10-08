"""How a sweep cuts its sources into slices: one per coordinate, or one per window, or one per combination of several axes.

A partition filters the sources, rows and index together: the containment
check refuses parameter rows outside a narrowed index. An [`EachWindow`][]
axis also says how each window's frames are stitched back over the dimension
it cut. A tuple of axes cuts with each in turn, outer first, and its slices
fall into chains: the slices that share every outer key.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import polars as pl
import polars.selectors as cs
from mathspec import did_you_mean

from specsolve.errors import DataError, SpecsolveError, SpecsolveWarning
from specsolve.frames import as_frame
from specsolve.sources import in_microseconds, least_value

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mathspec.program import Program

    from specsolve.inputs import Label, Source


class Slice(NamedTuple):
    """One slice of a sweep: the key, the sources that build it, and what it owns.

    ``key`` holds one label per axis, outer first; every label but the last
    names the slice's chain. ``owns`` counts the coordinates of the re-indexed
    dimension this slice keeps, the rest being lookahead, or is ``None`` where
    the axis re-indexed nothing; a ``carry`` reads the seam off it.
    """

    key: tuple[Label, ...]
    sources: Mapping[str, Source]
    owns: int | None = None
    #: How this slice cuts a table as it cut its sources: each axis cuts the
    #: table where it carries that axis's column, and passes it through where
    #: it does not; ``None`` for a slice written by hand.
    cut: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None

    @property
    def chain(self) -> tuple[Label, ...]:
        """The outer keys, which the slices that run in order after one another share."""
        return self.key[:-1]


@dataclass(frozen=True)
class Stitch:
    """The way from a windowed sweep's frames to its answer over the dimension it sliced.

    ``owned`` is ``(*outer, key, local, dim)`` for the coordinates each window
    owns; the lookahead rows are not in it.
    """

    local: str
    dim: str
    owned: pl.DataFrame
    #: The key columns of the axes outside the windows, outer first, which
    #: the answer keeps.
    outer: tuple[str, ...] = ()

    def unstitchable(self, frame: pl.DataFrame | pl.LazyFrame) -> str | None:
        """Why *frame* has no answer over [`dim`][], or ``None`` where it has one.

        A frame with no [`local`][] column is over no coordinate a window owns.
        """
        if self.local in frame.collect_schema().names():
            return None
        return (
            f'this has no answer over {self.dim!r}: the frame has no {self.local!r} column, because the '
            f'quantity is not over the windowed dimension — each row covers a whole window, lookahead '
            f'included under an overlapping window. Read it with per_window=True for the value of each '
            f'window, or read a quantity that keeps {self.local!r} and aggregate its answer.'
        )

    def restore[F: (pl.DataFrame, pl.LazyFrame)](self, frame: F, key_name: str) -> F:
        """*frame* over the dimension the axis sliced, sorted on it; lazy in, lazy out.

        The inner join on ``owned`` drops the lookahead rows.

        Raises:
            SpecsolveError: *frame* is [`unstitchable`][].
        """
        if why := self.unstitchable(frame):
            raise SpecsolveError(why)
        columns = frame.collect_schema().names()
        keys = [key_name, self.local]
        rest = [column for column in columns if column not in (*self.outer, *keys, 'value')]
        restored = frame.lazy().join(self.owned.lazy(), on=[*self.outer, *keys], how='inner').drop(keys)
        stitched = restored.select(*self.outer, self.dim, *rest, 'value').sort(*self.outer, self.dim, *rest)
        return stitched if isinstance(frame, pl.LazyFrame) else stitched.collect()  # pyrefly: ignore[bad-return]  — the branch matches the frame's own kind


def bulleted(entries: Mapping[str, str]) -> str:
    return '\n'.join(f'  {label}: {reason}' for label, reason in entries.items())


@dataclass(frozen=True)
class EachCoordinate:
    """One slice per coordinate of *dim*, a column the sources carry: scenarios, draws, investment periods.

    Parameters and relations carrying *dim* are filtered to one coordinate and
    the column dropped, so the model never mentions it; a *dim* the spec
    declares is refused, and so is an index that carries it. Every other
    source passes through untouched. Slices run in sorted coordinate order,
    which is the order a ``carry`` chains them in.
    """

    dim: str

    def slices(self, sources: Mapping[str, Source]) -> list[tuple[Label, Mapping[str, Source]]]:
        """The ``(key, sources)`` list this axis would run — what ``axis=`` takes hand-built.

        For building one slice alone: ``sps.build(spec, axis.slices(sources)[3][1])``.
        """
        return [(current.key[-1], current.sources) for current in self._slice(sources, self._key_name())[0]]

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

    def _slice(self, sources: Mapping[str, Source], key_name: str) -> tuple[list[Slice], Stitch | None]:
        """One slice per coordinate, keyed by it, and no [`Stitch`][]: the slices are the answer."""
        del key_name
        carrying, coordinates = _coordinates(sources, self.dim, 'slice')
        out: list[Slice] = []
        for key in coordinates:
            cut = partial(_one_coordinate, self.dim, key)
            filtered = {name: cut(table) for name, table in carrying.items()}
            out.append(Slice((key,), {**sources, **filtered}, cut=partial(_where_carried, self.dim, cut)))
        return out, None


@dataclass(frozen=True)
class EachWindow:
    """One slice per window of consecutive coordinates of *dim*.

    Each window keeps ``steps`` coordinates and sees ``lookahead`` beyond them,
    so a ``lookahead`` above zero is overlap. An ``int`` keeps the same number
    every window; a sequence keeps those numbers in order, for a telescoping
    horizon or a month at a time. Both count coordinates, not values, so *dim*
    need only be orderable — datetimes, strings and gapped integers all work.
    *dim* is re-indexed into a dense ``0..n-1`` column named ``into``, which
    the spec has to declare.

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
        return [(current.key[-1], current.sources) for current in self._slice(sources, self._key_name())[0]]

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
        resolved from the parameter's least value.
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
                f'together, so no window holds every row whole:\n{bulleted(verdict.coupled)}\n'
                f'Each names the change that would lift it.'
            )
        if verdict.undecided:
            raise SpecsolveError(
                f"EachWindow('{self.dim}', …, into='{self.into}') slices '{self.into}', which the model reaches "
                f'along through a relation whose groups a window may split, and this driver does not resolve a '
                f'reach the relation decides:\n'
                f'{bulleted({r.label: f"through the relation {r.name!r}" for r in verdict.undecided})}\n'
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
                f"the model counts a position along '{self.into}':\n{bulleted(verdict.restarts)}\n"
                f'Every window restarts the count at its first row — what a rolling horizon seeding its '
                f'opening state means, and once per horizon otherwise.',
                SpecsolveWarning,
                stacklevel=3,
            )

    def _slice(self, sources: Mapping[str, Source], key_name: str) -> tuple[list[Slice], Stitch]:
        """One slice per window, keyed by its first coordinate; the [`Stitch`][] records what each owns."""
        carrying, coordinates = _coordinates(sources, self.dim, 'window')
        out: list[Slice] = []
        owned: list[dict[str, object]] = []
        start = 0
        for owns in self._blocks(len(coordinates)):
            window = coordinates[start : start + owns + self.lookahead]
            cut = partial(_one_window, self.dim, self.into, window)
            filtered = {name: cut(table) for name, table in carrying.items()}
            tolerant = partial(_where_carried, self.dim, cut)
            out.append(Slice((window[0],), {**sources, **filtered, self.into: range(len(window))}, owns, tolerant))
            owned.extend(
                {key_name: window[0], self.into: position, self.dim: coordinate}
                for position, coordinate in enumerate(window[:owns])
            )
            start += owns
        return out, Stitch(self.into, self.dim, pl.DataFrame(owned))

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

#: Several axes, outer first, which cut the sources with each in turn: every
#: [`EachCoordinate`][] but the last, which may also be an [`EachWindow`][].
type Axes = tuple[Axis, ...]


def checked_axes(axis: Axis | Axes) -> Axes:
    """*axis* as the axes it cuts with, outer first; one axis is a tuple of one.

    Raises:
        SpecsolveError: No axis, one that is not an axis, a window outside
            another axis, or two axes over one dimension.
    """
    axes = axis if isinstance(axis, tuple) else (axis,)
    if not axes:
        raise SpecsolveError('axis=() names no axis, so there is nothing to cut the sources by. Pass one or more axes.')
    if odd := [repr(each) for each in axes if not isinstance(each, (EachCoordinate, EachWindow))]:
        raise SpecsolveError(
            f'a tuple of axes takes EachCoordinate and EachWindow, outer first, and {", ".join(odd)} is neither. '
            f'A hand-built list of slices is passed alone, as axis=.'
        )
    if outside := [each.dim for each in axes[:-1] if isinstance(each, EachWindow)]:
        raise SpecsolveError(
            f'EachWindow({outside[0]!r}) is outside another axis, and windows can only be the last axis: a '
            f'window is stitched back over {outside[0]!r}, and an axis inside it would cut each window again. '
            f'Put the windows last, so each combination of the outer keys is one horizon.'
        )
    dims = [each.dim for each in axes]
    if repeated := sorted({dim for dim in dims if dims.count(dim) > 1}):
        raise SpecsolveError(
            f'two axes cut {repeated[0]!r}, so the inner one would find the column already dropped by the outer '
            f'one. Cut each dimension once.'
        )
    return axes


def cut_by(axes: Axes, sources: Mapping[str, Source], key_names: Sequence[str]) -> tuple[list[Slice], Stitch | None]:
    """Every slice *axes* cut *sources* into, outer first, and the [`Stitch`][] of a sweep whose last axis is windows.

    Each outer slice is cut again by the axes inside it, so the coordinates
    and the windows of an inner axis are the ones that outer slice holds. A
    slice's key prepends the outer keys to its own, and its cut applies each
    axis in turn.
    """
    first, *inner = axes
    if not inner:
        return first._slice(sources, key_names[0])
    out: list[Slice] = []
    owned: list[pl.DataFrame] = []
    stitch: Stitch | None = None
    for outer in first._slice(sources, key_names[0])[0]:
        slices, stitch = cut_by(tuple(inner), outer.sources, key_names[1:])
        out.extend(
            Slice((*outer.key, *each.key), each.sources, each.owns, partial(_in_turn, outer.cut, each.cut))
            for each in slices
        )
        if stitch is not None:
            owned.append(stitch.owned.select(pl.lit(outer.key[0]).alias(key_names[0]), pl.all()))
    if stitch is None:
        return out, None
    return out, Stitch(stitch.local, stitch.dim, pl.concat(owned), (key_names[0], *stitch.outer))


def keyed_slices(
    axis: Axis | Axes, sources: Mapping[str, Source]
) -> list[tuple[tuple[Label, ...], Mapping[str, Source]]]:
    """Each slice *axis* cuts *sources* into, as its key, one label per axis outer first, and its sources."""
    axes = axis if isinstance(axis, tuple) else (axis,)
    slices, _ = cut_by(axes, sources, [each._key_name() for each in axes])
    return [(current.key, current.sources) for current in slices]


def _in_turn(
    outer: Callable[[pl.LazyFrame], pl.LazyFrame] | None,
    inner: Callable[[pl.LazyFrame], pl.LazyFrame] | None,
    table: pl.LazyFrame,
) -> pl.LazyFrame:
    """*table* cut by the outer axis, then by the axes inside it; a class axis always has a cut."""
    assert outer is not None and inner is not None, 'every slice an axis cuts carries its cut'
    return inner(outer(table))


def _where_carried(dim: str, cut: Callable[[pl.LazyFrame], pl.LazyFrame], table: pl.LazyFrame) -> pl.LazyFrame:
    """*table* cut where it carries *dim*, and whole where it does not, as a slice takes a source that lacks the axis."""
    return cut(table) if dim in table.collect_schema().names() else table


def check_no_index_is_cut(program: Program, sources: Mapping[str, Source], axis: Axis) -> None:
    """Refuse an index of a dimension other than the axis's own that carries the axis column.

    An index says which labels the model has, and a sweep cuts the tables that
    carry the axis, so a carried index would make each slice a model over
    other labels. The axis refuses its own index's dimension in its own words.
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


def axis_manifest(axis: Axis | Axes) -> dict[str, Any]:  # pyrefly: ignore[explicit-any] — the archive's own JSON
    """*axis* as the JSON an archive carries, read back by [`axis_from`][]."""
    if isinstance(axis, tuple):
        return {'each': 'axes', 'axes': [axis_manifest(each) for each in axis]}
    if isinstance(axis, EachCoordinate):
        return {'each': 'coordinate', 'dim': axis.dim}
    steps = axis.steps if isinstance(axis.steps, int) else list(axis.steps)
    return {'each': 'window', 'dim': axis.dim, 'steps': steps, 'lookahead': axis.lookahead, 'into': axis.into}


def axis_from(manifest: Mapping[str, Any]) -> Axis | Axes:  # pyrefly: ignore[explicit-any] — the archive's own JSON
    """The axis [`axis_manifest`][] wrote."""
    if manifest['each'] == 'axes':
        return tuple(cast('Axis', axis_from(each)) for each in manifest['axes'])
    if manifest['each'] == 'coordinate':
        return EachCoordinate(manifest['dim'])
    return EachWindow(manifest['dim'], steps=manifest['steps'], lookahead=manifest['lookahead'], into=manifest['into'])


def _one_coordinate(dim: str, key: Label, table: pl.LazyFrame) -> pl.LazyFrame:
    """*table*'s rows at coordinate *key* of *dim*, without the column."""
    return _in_microseconds(table, dim).filter(pl.col(dim) == key).drop(dim)


def _one_window(dim: str, into: str, window: Sequence[Label], table: pl.LazyFrame) -> pl.LazyFrame:
    """*table*'s rows at the coordinates of *window*, each over its position in it as *into* rather than *dim*."""
    local = {coordinate: position for position, coordinate in enumerate(window)}
    return (
        _in_microseconds(table, dim)
        .filter(pl.col(dim).is_in(window))
        .with_columns(pl.col(dim).replace_strict(local, return_dtype=pl.Int64).alias(into))
        .drop(dim)
    )


def _in_microseconds(table: pl.LazyFrame, dim: str) -> pl.LazyFrame:
    """*table* with a datetime *dim* held in microseconds, as attach holds a label, so a coordinate matches its rows."""
    return table.with_columns((cs.by_name(dim) & cs.datetime()).dt.cast_time_unit('us'))


def sources_with_column(sources: Mapping[str, Source], dim: str) -> dict[str, pl.LazyFrame]:
    """The sources that carry a column called *dim*, by name; a sweep and its archive both cut these."""
    tables = {name: table for name, obj in sources.items() if (table := as_frame(obj)) is not None}
    return {name: table for name, table in tables.items() if dim in table.collect_schema().names()}


def _coordinates(sources: Mapping[str, Source], dim: str, verb: str) -> tuple[dict[str, pl.LazyFrame], list[Label]]:
    """The sources a slice has to filter, by name, and the coordinates to slice, sorted by value.

    A datetime axis is held in microseconds, as attach holds a label, so a
    coordinate read out as a python datetime matches its rows.
    """
    carrying = sources_with_column(sources, dim)
    if not carrying:
        raise DataError(
            f"no source carries a '{dim}' column, so there is nothing to {verb} over. "
            f'EachCoordinate names a column the data has; a span of consecutive coordinates is EachWindow.'
        )
    unique = {name: table.select(pl.col(dim).unique()).collect() for name, table in carrying.items()}
    held = {name: set(in_microseconds(labels, f"source '{name}'")[dim]) for name, labels in unique.items()}
    carrying = {name: _in_microseconds(table, dim) for name, table in carrying.items()}
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
