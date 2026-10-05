"""A manifest: the arguments of [`solve`][specsolve.api.solve] and [`solve_over`][specsolve.strategy.solve_over], written as a YAML file.

Each key is an argument of those two verbs or of an axis class, except the
six in ``OWN_KEYS``, which are the manifest's own. The rules are the
[manifest reference](https://specsolve.readthedocs.io/en/latest/reference/manifest/).

Example::

    import specsolve as sps

    manifest = sps.load_manifest('specsolve.yaml')
    sweep = manifest.runs['scenarios'].solve()
"""

from __future__ import annotations

import importlib.util
import inspect
import multiprocessing
import warnings
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
import yaml
from mathspec import did_you_mean
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

from specsolve.api import solve
from specsolve.axes import Axis, EachCoordinate, EachWindow
from specsolve.errors import DataError, SpecsolveError
from specsolve.strategy import solve_over

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from specsolve.inputs import Source
    from specsolve.relational.result import Result
    from specsolve.sweep import Sweep

__all__ = ['Manifest', 'Run', 'load_manifest']

#: The keys no verb takes: what they mean is the manifest's own.
OWN_KEYS = ('manifest', 'runs', 'from', 'archive', 'sources', 'workers')

#: What `pip install` adds the command line and the Excel reader with.
INSTALL_HINT = "pip install 'specsolve[cli]'"

#: The axis classes a run may name under ``axis:``.
AXES: dict[str, type[EachCoordinate] | type[EachWindow]] = {'EachCoordinate': EachCoordinate, 'EachWindow': EachWindow}

#: The file suffixes a source location may carry, besides a directory.
TABLE_SUFFIXES = ('.parquet', '.csv')


class _Settings(BaseModel):
    """What a run may set, and what the top level sets as every run's default."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)

    spec: str | None = None
    sources: list[str] | None = None
    solver_name: str | None = None
    solver_options: dict[str, object] | None = None
    record_options: list[str] | None = None
    archive: str | None = None
    axis: dict[str, dict[str, object]] | None = None
    carry: dict[str, str] | None = None
    key_name: str | None = None
    keep: Literal['nothing', 'solver', 'progress'] | None = None
    keep_windows: bool | None = None
    workers: PositiveInt | None = None


class _RunSettings(_Settings):
    from_: str | None = Field(default=None, alias='from')


class _Document(_Settings):
    manifest: Literal[1]
    runs: dict[str, _RunSettings]


def _keys(model: type[BaseModel]) -> list[str]:
    """The keys *model* takes, as the file spells them."""
    return sorted(held.alias or name for name, held in model.model_fields.items())


#: The keys a run takes, and the ones the top level takes.
RUN_KEYS = _keys(_RunSettings)
TOP_KEYS = _keys(_Document)


@dataclass(frozen=True)
class Run:
    """One run of a manifest: one call to [`solve`][specsolve.api.solve], or to `solve_over` where it has an axis.

    Every path is resolved against the manifest's directory.

    Attributes:
        name: The run's key under ``runs:``.
        spec: The spec file.
        sources: The locations read into ``sources``, in the order a later
            one replaces an earlier table of the same name.
        axis: The axis, or ``None`` for a single solve.
        archive: Where the run archives, ``<archive>/<name>``, or ``None``.
        workers: How many local processes solve the slices, or ``None`` for
            one at a time in this process.
        options: Every other argument the run sets, by the verb's own name.
    """

    name: str
    spec: Path
    sources: tuple[Path, ...]
    axis: Axis | None
    archive: Path | None
    workers: int | None
    options: Mapping[str, object]

    @property
    def is_sweep(self) -> bool:
        """Whether the run calls [`solve_over`][specsolve.strategy.solve_over]."""
        return self.axis is not None

    @property
    def arguments(self) -> dict[str, object]:
        """The keyword arguments of the call: ``sps.solve_over(**run.arguments)``.

        Reads every source location each time it is asked. With ``workers``
        set, ``executor`` is a new spawn-context process pool that nothing has
        started; the caller shuts it down.

        Raises:
            DataError: A location that does not exist, has a form no reader
                takes, or holds two tables of one name.
            SpecsolveError: An ``.xlsx`` location without ``fastexcel``
                installed.
        """
        arguments: dict[str, object] = {'spec': self.spec, 'sources': read_sources(self.sources), **self.options}
        if self.axis is not None:
            arguments['axis'] = self.axis
        if self.archive is not None:
            arguments['archive'] = str(self.archive)
        if self.workers is not None:
            arguments['executor'] = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context('spawn'))
        return arguments

    def solve(self, **overrides: object) -> Result | Sweep:
        """Call the run's verb with [`arguments`][], *overrides* replacing them.

        A pool that ``workers`` started is shut down before this returns.

        Raises:
            SpecsolveError: Whatever the verb raises, and what
                [`arguments`][] raises.
        """
        arguments = self.arguments
        pool = arguments.get('executor')
        try:
            verb = solve_over if self.axis is not None else solve
            return verb(**{**arguments, **overrides})  # pyrefly: ignore[bad-argument-type]  — the schema typed every value
        finally:
            if isinstance(pool, Executor):
                pool.shutdown()


@dataclass(frozen=True)
class Manifest:
    """A manifest file, read and checked.

    Attributes:
        path: The file.
        runs: Every run by name, in the order the file lists them.
    """

    path: Path
    runs: Mapping[str, Run]


def load_manifest(path: str | Path) -> Manifest:
    """Read a manifest and resolve every run; read no data.

    Args:
        path: The YAML file. Relative paths inside it are relative to its
            directory.

    Returns:
        The manifest, every run resolved: defaults applied, ``from``
        followed, the axis built and the paths resolved.

    Raises:
        SpecsolveError: Anything the file gets wrong, naming the file and the
            run — an unknown key with the keys that are valid there, a
            duplicate key, a ``from`` that names no run or makes a cycle, a
            run with no spec, an axis its class refuses, or a key only
            ``solve_over`` takes on a run with no axis.
        FileNotFoundError: No file at *path*.
    """
    file = Path(path)
    document = _validated(file, _parsed(file))
    defaults = _set(document, exclude={'manifest', 'runs'})
    resolved: dict[str, dict[str, object]] = {}

    def resolve(name: str, chain: tuple[str, ...]) -> dict[str, object]:
        if name in chain:
            cycle = ' -> '.join((*chain[chain.index(name) :], name))
            raise SpecsolveError(f'{file}: runs.{chain[-1]}: from makes a cycle, {cycle}. Remove one from.')
        if name not in resolved:
            own = _set(document.runs[name], exclude=set())
            parent = own.pop('from_', None)
            if parent is None:
                base = defaults
            elif isinstance(parent, str) and parent in document.runs:
                base = resolve(parent, (*chain, name))
            else:
                raise SpecsolveError(
                    f'{file}: runs.{name}: from: {parent!r} names no run. '
                    f'{did_you_mean(str(parent), document.runs, label="Runs")}'
                )
            resolved[name] = {**base, **own}
        return resolved[name]

    runs = {name: _run(file, name, dict(resolve(name, ()))) for name in document.runs}
    return Manifest(file, runs)


def _parsed(file: Path) -> object:
    """The YAML document in *file*, refusing a key written twice, which YAML would read as the last."""
    try:
        with file.open() as stream:
            return yaml.load(stream, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise SpecsolveError(f'{file}: {error}') from None


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe loader that refuses a mapping holding one key twice."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, object]:
        seen: set[object] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f'the key {key!r} is written twice; one of them would be dropped', key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _validated(file: Path, raw: object) -> _Document:
    """*raw* as the closed schema reads it, or every problem with it in one error."""
    try:
        return _Document.model_validate(raw)
    except ValidationError as error:
        problems = [_problem(entry['loc'], entry['type'], entry['msg']) for entry in error.errors()]
        raise SpecsolveError('\n'.join(f'{file}: {problem}' for problem in problems)) from None


def _problem(location: tuple[int | str, ...], kind: str, message: str) -> str:
    """One validation error as a sentence that names where it is and, for an unknown key, what is valid there."""
    where = '.'.join(str(part) for part in location)
    if kind != 'extra_forbidden':
        return f'{where}: {message}'
    key, place = str(location[-1]), location[:-1]
    run = len(place) == 2 and place[0] == 'runs'
    valid = RUN_KEYS if run else TOP_KEYS
    named = f'runs.{place[1]}' if run else 'the top level'
    near = did_you_mean(key, valid, listing=False)
    return f'unknown key {key!r} in {named}. {near + " " if near else ""}Valid keys: {", ".join(valid)}.'


def _set(model: BaseModel, *, exclude: set[str]) -> dict[str, object]:
    """The fields *model* was given, by field name; an absent key is not a ``None``."""
    return {name: getattr(model, name) for name in model.model_fields_set if name not in exclude}


def _strings(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _run(file: Path, name: str, settings: dict[str, object]) -> Run:
    """The run *settings* describe, every rule decided without data checked."""
    where = f'{file}: runs.{name}'
    if Path(name).name != name or name in {'.', '..'}:
        raise SpecsolveError(f'{where}: a run name is a directory under archive, so it cannot hold a path separator.')
    spec = settings.pop('spec', None)
    if not isinstance(spec, str):
        raise SpecsolveError(f'{where}: no spec. Set spec: at the top level, or in the run.')
    root = file.parent
    raw_axis = settings.pop('axis', None)
    axis = _axis(where, raw_axis) if isinstance(raw_axis, dict) else None
    workers = settings.pop('workers', None)
    if axis is None:
        only_sweeps = sorted(key for key in settings if key in SWEEP_ONLY)
        if workers is not None:
            only_sweeps.append('workers')
        if only_sweeps:
            raise SpecsolveError(
                f'{where}: {", ".join(only_sweeps)} only solve_over takes, and this run has no axis, so it calls '
                f'solve. Give the run an axis, or remove the key.'
            )
    archive = settings.pop('archive', None)
    locations = tuple(root / location for location in _strings(settings.pop('sources', None)))
    return Run(
        name=name,
        spec=root / spec,
        sources=locations,
        axis=axis,
        archive=root / archive / name if isinstance(archive, str) else None,
        workers=workers if isinstance(workers, int) else None,
        options=settings,
    )


def _sweep_only() -> frozenset[str]:
    """The arguments ``solve_over`` takes and ``solve`` does not, read off the two signatures."""
    return frozenset(inspect.signature(solve_over).parameters) - frozenset(inspect.signature(solve).parameters)


SWEEP_ONLY = _sweep_only()


def _axis(where: str, raw: Mapping[str, Mapping[str, object]]) -> Axis:
    """The axis ``{Class: {parameter: value}}`` names, built by the class, which refuses what it refuses."""
    if len(raw) != 1 or next(iter(raw)) not in AXES:
        raise SpecsolveError(
            f'{where}: axis takes one class and its arguments, {{EachCoordinate: {{dim: ...}}}} or '
            f'{{EachWindow: {{dim: ..., steps: ..., lookahead: ..., into: ...}}}}; got {sorted(raw)}.'
        )
    kind, arguments = next(iter(raw.items()))
    cls = AXES[kind]
    valid = [held.name for held in fields(cls)]
    unknown = sorted(set(arguments) - set(valid))
    if unknown:
        raise SpecsolveError(f'{where}: axis {kind} takes {", ".join(valid)}, not {", ".join(unknown)}.')
    missing = sorted(set(valid) - set(arguments))
    if missing:
        raise SpecsolveError(f'{where}: axis {kind} needs {", ".join(missing)}.')
    try:
        return cls(**arguments)  # pyrefly: ignore[bad-argument-type]  — the class checks its own values
    except (TypeError, ValueError) as error:
        raise SpecsolveError(f'{where}: axis {kind}: {error}') from None


def read_sources(locations: tuple[Path, ...]) -> dict[str, Source]:
    """Every table the locations hold, by name, a later location replacing an earlier table of the same name.

    A parquet file stays a path, which a worker can read itself; a CSV file
    and a workbook sheet are read here.
    """
    tables: dict[str, Source] = {}
    for location in locations:
        tables.update(_read(location))
    return tables


def _read(location: Path) -> dict[str, Source]:
    """The tables at one location: a workbook's sheets, a directory's files, or one file."""
    if location.is_dir():
        held = sorted(path for path in location.iterdir() if path.suffix in TABLE_SUFFIXES)
        stems = [path.stem for path in held]
        twice = sorted({stem for stem in stems if stems.count(stem) > 1})
        if twice:
            raise DataError(f'{location} holds {", ".join(twice)} twice, once per suffix. Keep one file per table.')
        return {path.stem: _table(path) for path in held}
    if not location.exists():
        raise DataError(f'{location} does not exist.')
    if location.suffix == '.xlsx':
        return _workbook(location)
    if location.suffix in TABLE_SUFFIXES:
        return {location.stem: _table(location)}
    raise DataError(
        f'{location} is not a source location: give an .xlsx workbook, a .parquet or .csv file, or a directory of them.'
    )


def _table(path: Path) -> Source:
    return path if path.suffix == '.parquet' else pl.read_csv(path)


def _workbook(path: Path) -> dict[str, Source]:
    """Every sheet of *path*, by sheet name; polars reads it through fastexcel, which the ``cli`` extra installs."""
    if importlib.util.find_spec('fastexcel') is None:
        raise SpecsolveError(f'reading {path} needs fastexcel, which the cli extra installs: {INSTALL_HINT}')
    with warnings.catch_warnings():
        # polars' own call into fastexcel warns where pyarrow is absent; the tables it returns are the same.
        warnings.filterwarnings('ignore', message=r'from_arrow\(', category=FutureWarning)
        return dict(pl.read_excel(path, sheet_id=0))
