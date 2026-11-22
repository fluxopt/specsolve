"""How a sweep cuts its sources into slices: one per label, or one per window.

An [`EachLabel`][] axis squeezes a dimension the model does not declare: each
slice is the sources at one label, without its column, and a whole answer. An
[`EachWindow`][] axis keeps a dimension the model does declare and cuts it
into windows of its labels, and also says how each window's tables are
stitched back over that dimension. An axis's ``carry=`` is threaded here too:
[`as_built`][] gives each slice as it is built.
"""

from __future__ import annotations

import warnings
from dataclasses import MISSING, dataclass, field, fields
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

import polars as pl
import polars.selectors as cs
from mathspec.errors import did_you_mean

from specsolve._collect import collected
from specsolve._names import VALUE
from specsolve._sources import declared_dtype_of, dimension_labels, in_microseconds, least_value, parameter_table
from specsolve._tables import as_table
from specsolve.errors import DataError, SpecsolveError, SpecsolveWarning

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from mathspec.program import Program

    from specsolve._inputs import Label, Source


#: The key [`EachLabel.to_dict`][] and [`EachWindow.to_dict`][] write an axis's carry under.
CARRY = 'carry'


class Slice(NamedTuple):
    """One slice of a sweep: the key, the sources that build it, and where it ends.

    ``horizon_end`` is the last label of this slice's horizon, the labels after
    it being lookahead, or ``None`` where the axis windows nothing. A ``carry``
    hands on the row there.
    """

    key: Label
    sources: Mapping[str, Source]
    #: How this slice cuts a table that carries the sliced dimension, as it
    #: cut its sources.
    cut: Callable[[pl.LazyFrame], pl.LazyFrame]
    horizon_end: Label | None = None


@dataclass(frozen=True)
class Stitch:
    """The way from a windowed sweep's tables to its answer over the dimension it cut.

    ``horizon`` is ``(key, dim)`` for the labels of each window's horizon. The
    lookahead rows are not in it.
    """

    dim: str
    horizon: pl.DataFrame

    def unstitchable(self, table: pl.DataFrame | pl.LazyFrame) -> str | None:
        """Why *table* has no answer over [`dim`][], or ``None`` where it has one.

        A table with no [`dim`][] column is over no label of a window's horizon.
        """
        if self.dim in table.collect_schema().names():
            return None
        return (
            f'this has no answer over {self.dim!r}: the quantity is not over the windowed dimension, so '
            f'each row covers a whole window, lookahead included under an overlapping window. Read it '
            f'with per_window=True for the value of each window, or read a quantity that keeps '
            f'{self.dim!r} and aggregate its answer.'
        )

    def restore[F: (pl.DataFrame, pl.LazyFrame)](self, table: F, key_name: str) -> F:
        """*table* without its key and its lookahead rows, sorted on the dimension the axis cut; lazy in, lazy out.

        Raises:
            SpecsolveError: *table* is [`unstitchable`][].
        """
        if why := self.unstitchable(table):
            raise SpecsolveError(why)
        rest = [column for column in table.collect_schema().names() if column not in (key_name, self.dim, VALUE)]
        kept = table.lazy().join(self.horizon.lazy(), on=[key_name, self.dim], how='semi')
        stitched = kept.drop(key_name).sort(self.dim, *rest)
        return stitched if isinstance(table, pl.LazyFrame) else stitched.pipe(collected)  # pyrefly: ignore[bad-return]  — the branch matches the frame's own kind


def bulleted(reasons: Mapping[str, str]) -> str:
    return '\n'.join(f'  {label}: {reason}' for label, reason in reasons.items())


@dataclass(frozen=True)
class EachLabel:
    """One slice per label of *dim*, a dimension the sources carry and the spec does not declare.

    Scenarios, draws and investment periods are such dimensions.

    Each slice squeezes *dim*: the parameters and relations carrying it are
    cut to one label and the column dropped, so the model never mentions it.
    A *dim* the spec declares is refused, and so is a labels table that carries it.
    Every other source passes through untouched. Slices run in sorted label
    order, and the answer lifts each label back as the key column, named
    *dim* unless ``key_name=`` names it.

    ``carry={parameter: variable}`` chains them in that order: each slice's
    answer is copied into the next slice's data, and the first slice takes
    the parameter from the sources as its seed.
    """

    dim: str
    #: ``{parameter: variable}`` handed from each slice to the next; empty for
    #: slices that do not depend on one another.
    carry: Mapping[str, str] = field(default_factory=dict, kw_only=True, hash=False)
    #: What to call the key column in every table of the answer; ``None``
    #: takes the default the class documents.
    key_name: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'carry', dict(self.carry))

    def to_dict(self) -> dict[str, object]:
        """This axis as plain data, which [`solve_over`][specsolve._solve_over.solve_over] takes in its place.

        ``json.dumps`` writes it, and an archive's ``axis.json`` is this. It
        holds ``carry`` and ``key_name`` only where the axis sets them.
        """
        return {'each': 'label', 'dim': self.dim, **_where_set(self)}

    def _key_name(self) -> str:
        return self.dim if self.key_name is None else self.key_name

    def _check_the_program(self, program: Program, sources: Mapping[str, Source]) -> None:
        """Refuse a *dim* the spec declares, and a labels table that carries it.

        Every slice squeezes *dim* out of its sources, so nothing would supply
        a dimension of that name. A dimension's labels say which labels the
        model has, and a sweep cuts the tables that carry the axis, so labels
        carrying it would make each slice a model over other labels.
        """
        if self.dim in program.dimensions:
            raise SpecsolveError(
                f'EachLabel({self.dim!r}) squeezes {self.dim!r} out of every source, and the spec declares '
                f'it, so each slice would build a dimension nothing supplies. EachLabel cuts a dimension the '
                f'model does not have; EachWindow cuts one it does, and keeps it.'
            )
        for dim in program.dimensions:
            table = as_table(sources[dim]) if dim in sources else None
            if table is None or self.dim not in table.collect_schema().names():
                continue
            raise DataError(
                f"the labels table of dimension '{dim}' carries a '{self.dim}' column, and EachLabel"
                f"('{self.dim}') cuts every table that carries '{self.dim}'. A dimension's labels are not cut: "
                f"they are the labels every slice has. Pass the '{dim}' labels alone, and say which of them each "
                f"slice has in a parameter or a relation over ('{dim}', '{self.dim}'), where a missing row "
                f'already reads as absent.'
            )

    def _slice(
        self, program: Program, sources: Mapping[str, Source], key_name: str
    ) -> tuple[list[Slice], Stitch | None]:
        """One slice per label, keyed by it, and no [`Stitch`][]: the slices are the answer."""
        del program, key_name
        carrying, labels = _labels_carried(sources, self.dim)
        out: list[Slice] = []
        for key in labels:
            cut = partial(_one_label, self.dim, key)
            out.append(Slice(key, {**sources, **{name: cut(table) for name, table in carrying.items()}}, cut))
        return out, None


@dataclass(frozen=True)
class EachWindow:
    """One slice per window of consecutive labels of *dim*, an ordered dimension the spec declares.

    Each window keeps *dim* and is the model over its own labels: the labels
    of *dim* are narrowed to them, and every parameter and relation over *dim*
    is cut to the rows they hold. Nothing is squeezed or renamed, so the spec
    and the sources are the ones a single solve takes, and every table comes
    back over *dim*'s own labels.

    Each window keeps ``horizon`` labels and sees ``lookahead`` labels
    beyond them, so a ``lookahead`` above zero is overlap. An ``int``
    keeps the same number every window; a sequence keeps those numbers in
    order, for horizons that grow window by window or a month at a time. Both
    count labels in label order, not values, so datetimes, strings and gapped
    integers all work.

    A rule for a window's first row is written ``position(dim) == 0``: a
    position counts within the window, which is what a rolling horizon seeding
    its opening state means. A label written as a literal names that one label
    in whichever window holds it.

    Whether the model *can* be cut this way is asked before a slice is taken
    ([separability](https://mathspec.readthedocs.io/en/latest/reference/reading/#asking-whether-an-axis-can-be-cut)):
    a coupling along *dim* is refused, naming the entry and the change
    that would lift it, and ``lookahead`` has to cover what the rows read
    ahead. What the rows read *behind* is the rolling-horizon seed, met by the
    edge policy, and is not refused.

    ``carry={parameter: variable}`` chains the windows in order: each window's
    answer is copied into the next window's data. Where the two are over
    different dimensions, the dropped one is *dim*, and the row at the
    window's horizon end is handed on. The first window takes the parameter
    from the sources as its seed.

    ``per_window=True`` reads each window under the key column, named
    ``<dim>_start`` unless ``key_name=`` names it.
    """

    dim: str
    horizon: int | Sequence[int] = field(kw_only=True)
    lookahead: int = field(kw_only=True)
    #: ``{parameter: variable}`` handed from each window to the next; empty for
    #: windows that do not depend on one another.
    carry: Mapping[str, str] = field(default_factory=dict, kw_only=True, hash=False)
    #: What to call the key column in every table of the answer; ``None``
    #: takes the default the class documents.
    key_name: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        horizons = [self.horizon] if isinstance(self.horizon, int) else list(self.horizon)
        if not horizons:
            raise ValueError('horizon is empty, so no window would keep anything — pass an int, or one size per window')
        if short := [horizon for horizon in horizons if horizon < 1]:
            raise ValueError(f'every window must keep at least one label (got horizon with {short})')
        if self.lookahead < 0:
            raise ValueError(
                f'lookahead={self.lookahead} is negative; zero is contiguous windows, and above it overlaps'
            )
        if not isinstance(self.horizon, int):
            object.__setattr__(self, 'horizon', tuple(horizons))
        object.__setattr__(self, 'carry', dict(self.carry))

    def to_dict(self) -> dict[str, object]:
        """This axis as plain data, which [`solve_over`][specsolve._solve_over.solve_over] takes in its place.

        ``json.dumps`` writes it, and an archive's ``axis.json`` is this. It
        holds ``carry`` and ``key_name`` only where the axis sets them.
        """
        horizon = self.horizon if isinstance(self.horizon, int) else list(self.horizon)
        return {'each': 'window', 'dim': self.dim, 'horizon': horizon, 'lookahead': self.lookahead, **_where_set(self)}

    def _key_name(self) -> str:
        """Where the window started, never ``dim``, which every table of the window already carries."""
        return f'{self.dim}_start' if self.key_name is None else self.key_name

    def _check_the_program(self, program: Program, sources: Mapping[str, Source]) -> None:
        """Refuse a window the program's rows cannot be whole inside, before one is taken.

        The program's `separability` answers; a reach the data decides is
        resolved from the parameter's least value.
        """
        if self.dim not in program.dimensions:
            raise SpecsolveError(
                f'EachWindow({self.dim!r}) windows a dimension the spec declares, and it declares no '
                f'{self.dim!r}. {did_you_mean(self.dim, program.dimensions)} A dimension the model does '
                f'not declare is sliced with EachLabel.'
            )
        if not program.dimensions[self.dim].ordered:
            raise SpecsolveError(
                f'EachWindow({self.dim!r}) reads the order of {self.dim!r}, which is not declared ordered, so '
                f'the windows would come from the row order of the data. Declare '
                f"'{self.dim}: {{ordered: true}}' under dimensions:."
            )
        verdict = program.separability[self.dim]
        named = {reach.name for reach in verdict.undecided if reach.kind == 'offset'}
        verdict = verdict.resolved({name: least_value(program, sources, name) for name in named})
        if verdict.coupled:
            raise SpecsolveError(
                f"EachWindow('{self.dim}', …) windows '{self.dim}', which the model ties together, so no "
                f'window holds every row whole:\n{bulleted(verdict.coupled)}\n'
                f'Each names the change that would lift it.'
            )
        if verdict.undecided:
            raise SpecsolveError(
                f"EachWindow('{self.dim}', …) windows '{self.dim}', which the model reaches along through a "
                f'relation whose groups a window may split, and this driver does not resolve a reach the '
                f'relation decides:\n'
                f'{bulleted({r.label: f"through the relation {r.name!r}" for r in verdict.undecided})}\n'
                f'Cut a dimension the relation does not group.'
            )
        if self.lookahead < verdict.ahead:
            raise SpecsolveError(
                f'EachWindow(lookahead={self.lookahead}) looks ahead by {self.lookahead} label(s), and '
                f"the model reads {verdict.ahead} ahead along '{self.dim}' — a row near a window's end would "
                f'read past it. Raise lookahead to at least {verdict.ahead}.'
            )

    def _slice(self, program: Program, sources: Mapping[str, Source], key_name: str) -> tuple[list[Slice], Stitch]:
        """One slice per window, keyed by its first label; the [`Stitch`][] records each window's horizon."""
        labels = dimension_labels(program, sources, self.dim).to_list()
        tables = _tables_over(program, sources, self.dim, labels)
        out: list[Slice] = []
        horizons: list[dict[str, object]] = []
        start = 0
        for length in self._horizon_lengths(len(labels)):
            window = labels[start : start + length + self.lookahead]
            cut = {
                name: _within(table, columns, window, strangers) for name, (table, columns, strangers) in tables.items()
            }
            horizon_end = window[length - 1]
            out.append(
                Slice(
                    window[0], {**sources, **cut, self.dim: window}, partial(_one_window, self.dim, window), horizon_end
                )
            )
            horizons.extend({key_name: window[0], self.dim: label} for label in window[:length])
            start += length
        return out, Stitch(self.dim, pl.DataFrame(horizons))

    def _horizon_lengths(self, total: int) -> list[int]:
        """How many labels each window's horizon has, in order, summing to exactly *total*.

        An ``int`` repeats, the last window's horizon taking what is left. A sequence that
        stops short of the axis is refused.
        """
        if isinstance(self.horizon, int):
            horizons = [self.horizon] * -(-total // self.horizon)
        else:
            horizons = list(self.horizon)
            if sum(horizons) < total:
                raise DataError(
                    f'horizon keeps {sum(horizons)} label(s) across {len(horizons)} window(s), and '
                    f"'{self.dim}' has {total} — the last {total - sum(horizons)} would be solved by no window. "
                    f'List a horizon for them, or pass an int to repeat one size to the end.'
                )
        out: list[int] = []
        left = total
        for horizon in horizons:
            if left <= 0:
                break
            out.append(min(horizon, left))
            left -= horizon
        return out


#: An axis that cuts one set of sources into slices.
Axis = EachLabel | EachWindow


@dataclass(frozen=True)
class CarryRule:
    """One resolved carry: which variable moves into a parameter, and how.

    ``dropped`` is the one dimension the carry collapses, and ``None`` where the
    whole table moves forward.
    """

    variable: str
    dropped: str | None

    @classmethod
    def resolved(cls, program: Program, parameter: str, variable: str) -> CarryRule:
        """One carry checked against the program, without reading data."""
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
                f'advances along, so reduce the others in the YAML — a derived variable is where the linopy lane '
                f'can see the math.'
            )
        return cls(variable, dropped[0] if dropped else None)

    def value_from(
        self, tables: Mapping[str, pl.DataFrame], parameter: str, key: Label, horizon_end: Label | None
    ) -> pl.DataFrame:
        """What this rule hands the next slice, read out of one slice's primals.

        The row handed on is the one at *horizon_end*, the last label of the
        slice's horizon, never a lookahead row.
        """
        table = tables[self.variable]
        if self.dropped is None:
            return table
        assert horizon_end is not None, 'a carry that drops a dimension is refused unless the axis windows it'
        picked = table.filter(pl.col(self.dropped) == horizon_end).drop(self.dropped)
        if picked.is_empty():
            raise SpecsolveError(
                f'carry {parameter!r} <- {self.variable!r} has nothing to copy: slice {key!r} built no '
                f'{self.variable!r} at {self.dropped} == {horizon_end!r}, its horizon end, so there is no '
                f'value there to hand forward.'
            )
        return picked


def carry_plan(program: Program, axis: Axis) -> dict[str, CarryRule]:
    """Each carry of *axis* resolved against *program*, by the parameter it writes.

    Raises:
        SpecsolveError: As [`CarryRule.resolved`][] raises it.
    """
    return {parameter: CarryRule.resolved(program, parameter, variable) for parameter, variable in axis.carry.items()}


def as_built(
    slices: Sequence[Slice],
    plan: Mapping[str, CarryRule],
    primals_of: Callable[[Slice], Mapping[str, pl.DataFrame]],
) -> Iterator[Slice]:
    """Each slice as it is built: its sources are its cut, and what the slice before it handed on.

    This is the one place a carry is threaded, whether the slices are being
    solved, read back off a spill, or rebuilt from an archive. *primals_of*
    gives a slice's primal. It is asked for a slice only when the one after
    it is asked for, so a caller that solves as it goes has the primal by
    then, and it is never asked for the last slice, which hands nothing on.
    """
    handed: dict[str, pl.DataFrame] = {}
    before: Slice | None = None
    for current in slices:
        if before is not None and plan:
            primals = primals_of(before)
            handed = {
                parameter: rule.value_from(primals, parameter, before.key, before.horizon_end)
                for parameter, rule in plan.items()
            }
        yield current._replace(sources={**current.sources, **handed})
        before = current


#: Each axis by the name ``to_dict`` writes under ``each``.
_EACH: dict[str, type[EachLabel] | type[EachWindow]] = {'label': EachLabel, 'window': EachWindow}


def _where_set(axis: Axis) -> dict[str, object]:
    """The keys an axis's ``to_dict`` writes only where the axis sets them."""
    out: dict[str, object] = {}
    if axis.carry:
        out[CARRY] = dict(axis.carry)
    if axis.key_name is not None:
        out['key_name'] = axis.key_name
    return out


def axis_from(given: Mapping[str, Any]) -> Axis:  # pyrefly: ignore[explicit-any] — plain data, as JSON reads it
    """The axis *given* as plain data, as [`EachLabel.to_dict`][] and [`EachWindow.to_dict`][] write it.

    Raises:
        SpecsolveError: An ``each`` that names no axis, or a key the axis
            does not take or needs and is not given; the message names the
            keys it takes.
    """
    each = given.get('each')
    if each not in _EACH:
        raise SpecsolveError(
            f"an axis given as a mapping names its kind under 'each', one of {sorted(_EACH)} (got {each!r})"
        )
    kind = _EACH[each]
    taken = {f.name: f for f in fields(kind)}
    keys = {key: value for key, value in given.items() if key != 'each'}
    needed = [name for name, f in taken.items() if f.default is MISSING and f.default_factory is MISSING]
    if (unknown := sorted(set(keys) - set(taken))) or (missing := [name for name in needed if name not in keys]):
        wrong = f'{unknown} is not one of them' if unknown else f'{missing} is missing'
        raise SpecsolveError(
            f"{kind.__name__} given as a mapping takes 'each' and {list(taken)}, of which {needed} are "
            f'required: {wrong}.'
        )
    return kind(**keys)


def _one_label(dim: str, key: Label, table: pl.LazyFrame) -> pl.LazyFrame:
    """*table* squeezed at label *key* of *dim*: its rows there, without the column."""
    return _in_microseconds(table, [dim]).filter(pl.col(dim) == key).drop(dim)


def _one_window(dim: str, window: Sequence[Label], table: pl.LazyFrame) -> pl.LazyFrame:
    """*table*'s rows at the labels of *window* along its *dim* column."""
    return _as_the_labels_hold(table, [dim], window).filter(pl.col(dim).is_in(window))


def _tables_over(
    program: Program, sources: Mapping[str, Source], dim: str, labels: Sequence[Label]
) -> dict[str, tuple[pl.LazyFrame, tuple[str, ...], pl.DataFrame]]:
    """Each source a window cuts, by name: its table, its columns that hold labels of *dim*, and its strangers.

    A parameter over *dim* is cut on its *dim* column, and a relation on every
    column it declares over *dim*. One number is every window's number and is
    not cut. A mapping or a sequence is spread over all of *dim*'s labels first, as
    ``tidy_sources`` reads it, so a window cuts the same table a single solve attaches.
    The strangers are the rows holding a null or a label *dim* does not have. They
    are found once, and every window keeps them, so ``tidy_sources`` refuses them as
    it does for a single solve. A table short of a column it is cut on is left
    for ``tidy_sources`` to refuse.

    Raises:
        DataError: A label finer than a microsecond, which the cast that
            matches it to the labels would round onto another label.
    """
    listed = {dim: pl.LazyFrame({dim: labels})}
    found: dict[str, tuple[pl.LazyFrame, tuple[str, ...], str]] = {}
    for name, parameter in program.parameters.items():
        if dim in parameter.dims and name in sources and not isinstance(sources[name], (bool, int, float)):
            found[name] = (parameter_table(name, parameter, sources[name], listed), (dim,), 'parameter')
    for name, relation in program.relations.items():
        columns = tuple(role for role, over in relation.columns if over == dim)
        if columns and name in sources and (table := as_table(sources[name], relation.roles)) is not None:
            found[name] = (table, columns, 'relation')
    out = {}
    for name, (table, columns, kind) in found.items():
        schema = table.collect_schema()
        if set(columns) <= set(schema.names()):
            if finer := [column for column in columns if schema[column] == pl.Datetime('ns')]:
                in_microseconds(table.select(finer).unique().pipe(collected), f"{kind} '{name}'")
            held = _as_the_labels_hold(table, columns, labels)
            strange = pl.any_horizontal(~pl.col(column).is_in(labels).fill_null(False) for column in columns)
            out[name] = (held, columns, held.filter(strange).pipe(collected))
    return out


def _within(
    table: pl.LazyFrame, columns: Sequence[str], window: Sequence[Label], strangers: pl.DataFrame
) -> pl.LazyFrame:
    """*table*'s rows whose *columns* all hold a label of *window*, and its *strangers* for ``tidy_sources`` to refuse."""
    inside = table.filter(pl.all_horizontal(pl.col(column).is_in(window) for column in columns))
    return pl.concat([inside, strangers.lazy()]) if strangers.height else inside


def _as_the_labels_hold(table: pl.LazyFrame, columns: Sequence[str], labels: Sequence[Label]) -> pl.LazyFrame:
    """*table* with *columns* typed as *labels* are, so that a membership test compares like with like.

    A datetime is held in microseconds. Where the labels are datetimes, a
    date is the microsecond its day starts on, as attach reads one.
    """
    held = _in_microseconds(table, columns)
    if labels and isinstance(labels[0], datetime):
        held = held.with_columns((cs.by_name(columns) & cs.date()).cast(pl.Datetime('us')))
    return held


def _in_microseconds(table: pl.LazyFrame, columns: Sequence[str]) -> pl.LazyFrame:
    """*table* with each datetime column of *columns* held in microseconds, as attach holds a label, so a label matches its rows."""
    return table.with_columns((cs.by_name(columns) & cs.datetime()).dt.cast_time_unit('us'))


def sources_with_column(sources: Mapping[str, Source], dim: str) -> dict[str, pl.LazyFrame]:
    """The sources that carry a column called *dim*, by name; an [`EachLabel`][] sweep and its archive both cut these."""
    tables = {name: table for name, obj in sources.items() if (table := as_table(obj)) is not None}
    return {name: table for name, table in tables.items() if dim in table.collect_schema().names()}


def _kind(column: pl.DataType) -> str:
    """What a column of labels is, as a refusal names it; the labels of two columns of one kind sort as one list."""
    if isinstance(column, pl.Datetime):
        return 'datetime' if column.time_zone is None else f'datetime in {column.time_zone}'
    return declared_dtype_of(column) or str(column)


def _refuse_types_apart(carrying: Mapping[str, pl.LazyFrame], dim: str) -> None:
    """Refuse sources that carry *dim* as two kinds of column, from their schemas alone.

    The labels of every source become one sorted list of keys, written as one
    column type. Two kinds either cannot be sorted against each other, or
    sort into keys that one type then rewrites.
    """
    kinds = {name: _kind(table.collect_schema()[dim]) for name, table in carrying.items()}
    first, kind = next(iter(kinds.items()))
    for name, other in kinds.items():
        if other != kind:
            raise DataError(
                f"sources '{first}' and '{name}' carry '{dim}' as {kind} and {other}, and a sweep keys its "
                f"slices by one type. Cast '{dim}' in one of them to match the other."
            )


def _labels_carried(sources: Mapping[str, Source], dim: str) -> tuple[dict[str, pl.LazyFrame], list[Label]]:
    """The sources an [`EachLabel`][] sweep has to cut, by name, and the labels of *dim* they carry, sorted by value.

    A datetime column is held in microseconds, as attach holds a label, so a
    label read out as a python datetime matches its rows.
    """
    carrying = sources_with_column(sources, dim)
    if not carrying:
        raise DataError(
            f"no source carries a '{dim}' column, so there is nothing to slice over. "
            f'EachLabel names a dimension the data has and the spec does not; a span of consecutive labels '
            f'of a dimension the spec declares is EachWindow.'
        )
    _refuse_types_apart(carrying, dim)
    unique = {name: table.select(pl.col(dim).unique()).pipe(collected) for name, table in carrying.items()}
    if unlabelled := next((name for name, labels in unique.items() if labels[dim].null_count()), None):
        raise DataError(
            f"source '{unlabelled}' holds a null '{dim}'. A slice is keyed by a label, and a null names none. "
            f'Drop those rows: polars .drop_nulls({dim!r}), pandas .dropna(subset=[{dim!r}]).'
        )
    held = {name: set(in_microseconds(labels, f"source '{name}'")[dim]) for name, labels in unique.items()}
    carrying = {name: _in_microseconds(table, [dim]) for name, table in carrying.items()}
    labels = sorted(set().union(*held.values()))
    for name, mine in held.items():
        if missing := sorted(set(labels) - mine):
            other = next(o for o, theirs in held.items() if missing[0] in theirs)
            warnings.warn(
                f"'{name}' has no rows for {dim} {missing[0]!r}, which '{other}' has, so the slice at "
                f"{missing[0]!r} builds '{name}' empty and every row of it reads as absent. Supply the rows "
                f'if a value was meant; where the absence is, this is the record of it.',
                SpecsolveWarning,
                stacklevel=4,
            )
    return carrying, labels
