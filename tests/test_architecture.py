"""docs/about/architecture.md, enforced.

Each test encodes one hard rule from the architecture document. Static checks
parse source with ``ast``, so they run on a bare install.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

REPO = Path(__file__).parent.parent
PKG = REPO / 'src' / 'specsolve'

#: What no module of the package may import at module level. pandas is absent:
#: the bridges out of a result import it lazily, and the bare-install job fences it.
FORBIDDEN_RUNTIME = {'linopy', 'xarray'}

#: The differential-test oracle: the same YAML built as a ``linopy.Model``.
ORACLE = REPO / 'tests' / 'linopy_lane'


def _module_level_imports(path: Path) -> set[str]:
    """Top-level (non-lazy, non-TYPE_CHECKING) imported root packages, ``try:`` blocks included."""
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    stmts = list(tree.body)
    while stmts:
        node = stmts.pop()
        if isinstance(node, ast.Import):
            found.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split('.')[0])
        elif isinstance(node, ast.Try):
            stmts.extend([*node.body, *node.orelse, *node.finalbody])
            for handler in node.handlers:
                stmts.extend(handler.body)
    return found


def _all_modules() -> list[Path]:
    return [p for p in PKG.rglob('*.py') if '__pycache__' not in p.parts]


def _imported(
    tree: ast.AST,
    *,
    nodes: Callable[[ast.AST], Iterator[ast.AST]] = ast.walk,
    relative: bool = False,
) -> list[str]:
    """Every imported name in *tree*, as written.

    ``import a.b`` yields ``a.b``; ``from a.b import c`` yields ``a.b``. With
    ``relative=True`` a relative import keeps its dots (``from ..x import y``
    yields ``..x``); otherwise a module-less relative import is dropped.
    *nodes* picks the walk — ``ast.walk`` sees everything,
    :func:`_runtime_nodes` prunes ``TYPE_CHECKING`` bodies.
    """
    names: list[str] = []
    for node in nodes(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if relative:
                names.append('.' * node.level + (node.module or ''))
            elif node.module:
                names.append(node.module)
    return names


def _reaches_past(
    package: str,
    allowed: tuple[str, ...],
    allowlist: set[str],
    *,
    third_party: frozenset[str],
    nodes: Callable[[ast.AST], Iterator[ast.AST]],
) -> dict[str, list[str]]:
    """Modules under *package* importing a name its fence forbids.

    Forbidden is a ``specsolve`` name outside *allowed* and *allowlist*, or a
    name whose root package is in *third_party*. Lazy imports count.
    """
    offenders = {}
    for path in (PKG / package).rglob('*.py'):
        if '__pycache__' in path.parts:
            continue
        bad = [
            n
            for n in _imported(ast.parse(path.read_text()), nodes=nodes)
            if n.split('.')[0] in third_party
            or (n.startswith('specsolve') and not n.startswith(allowed) and n not in allowlist)
        ]
        if bad:
            offenders[str(path.relative_to(PKG))] = sorted(set(bad))
    return offenders


def _runtime_nodes(tree: ast.AST) -> Iterator[ast.AST]:
    """Every node the interpreter can reach — ``if TYPE_CHECKING:`` bodies pruned, their ``else`` kept."""
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            stack.extend(node.orelse)
            continue
        stack.extend(ast.iter_child_nodes(node))


def _is_type_checking(test: ast.expr) -> bool:
    """``TYPE_CHECKING`` or ``typing.TYPE_CHECKING``, however it was spelled."""
    if isinstance(test, ast.Name):
        return test.id == 'TYPE_CHECKING'
    return isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING'


def test_the_lane_fences_see_running_code_and_only_running_code():
    """The pruner keeps what the interpreter reaches and drops only what it never runs."""
    erased, executed, otherwise = (
        'if TYPE_CHECKING:\n    import xarray\n',
        'def f():\n    import xarray\n',
        'if TYPE_CHECKING:\n    import xarray\nelse:\n    import linopy\n',
    )

    def imported(source: str) -> set[str]:
        return {
            alias.name
            for node in _runtime_nodes(ast.parse(source))
            if isinstance(node, ast.Import)
            for alias in node.names
        }

    assert imported(erased) == set(), 'an annotation-only import is not a dependency'
    assert imported(executed) == {'xarray'}, 'a lazy import inside a function still runs'
    assert imported(otherwise) == {'linopy'}, 'the else branch of a TYPE_CHECKING guard does run'


def test_runtime_lane_never_imports_linopy_or_xarray():
    """Hard rule 3: linopy is the test oracle only — never a runtime import."""
    offenders = {}
    for path in _all_modules():
        bad = _module_level_imports(path) & FORBIDDEN_RUNTIME
        if bad:
            offenders[str(path.relative_to(PKG))] = sorted(bad)
    assert not offenders, (
        f'runtime modules import linopy-lane packages at module level: {offenders} '
        f'— linopy belongs to the test oracle, and xarray is reached lazily'
    )


#: Modules that may reach linopy or xarray *lazily*, with the reason.
LAZY_ORACLE_ALLOWED: dict[str, str] = {
    'linopy.py': 'the linopy export, behind the [linopy] extra; it reads what the engine built and is never the engine',
}


def test_lazy_oracle_imports_stay_on_the_allowlist():
    """Hard rule 3, the half a module-level check cannot see: a lazy import is declared."""
    offenders = {}
    for path in _all_modules():
        if path.name in LAZY_ORACLE_ALLOWED:
            continue
        tree = ast.parse(path.read_text())
        bad = set()
        for node in _runtime_nodes(tree):
            if isinstance(node, ast.Import):
                bad |= {a.name for a in node.names if a.name.split('.')[0] in FORBIDDEN_RUNTIME}
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in FORBIDDEN_RUNTIME:
                bad.add(node.module)
        if bad:
            offenders[str(path.relative_to(PKG))] = sorted(bad)
    assert not offenders, (
        f'modules of the package reach the oracle lazily: {offenders} — '
        f'move the code to tests/linopy_lane, or add it to LAZY_ORACLE_ALLOWED with a reason'
    )


#: Package modules the engine may import: dependency-free leaves that carry no
#: YAML, schema or AST knowledge.
ENGINE_MAY_IMPORT = {'specsolve.errors', 'mathspec.program'}


def test_engine_is_isolated():
    """Hard rule 2: the engine knows nothing about linopy, xarray or YAML.

    Enforced as "imports nothing from the package bar ENGINE_MAY_IMPORT",
    which keeps the subpackage extractable.
    """
    offenders = _reaches_past(
        'relational',
        ('specsolve.relational',),
        ENGINE_MAY_IMPORT,
        third_party=FORBIDDEN_RUNTIME | {'yaml'},
        nodes=_runtime_nodes,
    )
    assert not offenders, f'engine reaches outside its subpackage: {offenders}'


def test_no_contract_module_names_an_engine():
    """``relational/__init__.py``'s own split: contract above, ``engines/`` below.

    A contract module naming a class out of ``engines/`` inverts the two.
    Type-only imports count here.
    """
    offenders = {}
    for path in (PKG / 'relational').rglob('*.py'):
        rel = path.relative_to(PKG / 'relational').as_posix()
        if '__pycache__' in path.parts or rel.startswith('engines/'):
            continue
        named = _imported(ast.parse(path.read_text()), relative=True)
        engines = sorted({m for m in named if 'engines' in m.split('.')})
        if engines:
            offenders[rel] = engines
    assert not offenders, (
        f'a contract module names an implementation: {offenders}. Either the fact belongs '
        f'under engines/, or what crosses the seam should be a type the contract already owns'
    )


#: Where python this repository owns lives.
SOURCE_DIRS = ('src', 'tests', 'tools', 'bench', 'examples')


def _repository_modules() -> list[Path]:
    return [p for d in SOURCE_DIRS for p in (REPO / d).rglob('*.py') if '__pycache__' not in p.parts]


def test_the_language_is_imported_as_one_package():
    """Hard rule 1: this repository depends on the ``__all__`` mathspec pins.

    A submodule that ``__all__`` exports, such as ``mathspec.program``, is
    inside the surface.
    """
    import mathspec

    exported = set(mathspec.__all__)

    offenders = {}
    for path in _repository_modules():
        inside = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('mathspec.'):
                reached = [f'{node.module}.{alias.name}' for alias in node.names]
            elif isinstance(node, ast.Import):
                reached = [alias.name for alias in node.names if alias.name.startswith('mathspec.')]
            else:
                continue
            inside += [name for name in reached if name.split('.')[1] not in exported]
        if inside:
            offenders[str(path.relative_to(REPO))] = sorted(inside)
    assert not offenders, (
        f'modules reach past the language package surface: {offenders} — import the name from '
        f'`mathspec` itself, or from a submodule its `__all__` exports'
    )


#: Directory prefixes a workflow can name that are files in this repository.
REPO_PREFIXES = ('examples/', 'tests/', 'src/', 'docs/', 'tools/', 'bench/')


#: A trigger a fork can fire, choosing the code the steps check out and run.
FORK_REACHABLE = ('pull_request', 'pull_request_target')

#: How a job asks for a machine somebody owns: the label itself, or the
#: variable that resolves to one.
OWN_MACHINE = ('self-hosted', 'BENCH_RUNNER')


def test_no_fork_can_reach_a_runner_we_own():
    """A public repository plus a self-hosted runner is arbitrary code execution.

    `pull_request` builds the contributor's branch, which on a machine
    registered to this repository is a shell on that machine.
    """
    guilty = []
    for workflow in sorted((REPO / '.github' / 'workflows').glob('*.y*ml')):
        text = workflow.read_text()
        if not any(marker in text for marker in OWN_MACHINE):
            continue
        triggers = re.search(r'^on:(.*?)^[a-z]', text, re.DOTALL | re.MULTILINE)
        fired_by = [
            trigger
            for trigger in FORK_REACHABLE
            if triggers and re.search(rf'^\s+{trigger}\s*:', triggers.group(1), re.MULTILINE)
        ]
        if fired_by:
            guilty.append(f'{workflow.name}: {fired_by}')
    assert not guilty, (
        f'{guilty} run on a machine we own and can be fired from a fork — a contributor '
        f'chooses the code, so this is a shell on that machine. Keep these workflows to '
        f'workflow_dispatch and schedule.'
    )


def test_every_repository_path_a_workflow_names_exists():
    """A workflow step reads files by path, and a move makes it read nothing.

    Globs are resolved: one matching nothing is the same hole as a missing file.
    """
    missing = []
    for workflow in sorted((REPO / '.github' / 'workflows').glob('*.y*ml')):
        for token in workflow.read_text().split():
            token = token.strip('\'"`,')
            if not token.startswith(REPO_PREFIXES):
                continue
            hits = list(REPO.glob(token)) if any(c in token for c in '*?[') else [REPO / token]
            if not any(path.exists() for path in hits):
                missing.append(f'{workflow.name}: {token}')
    assert not missing, (
        f'a workflow names paths that do not exist: {missing} — a step reading them '
        f'reads nothing, and no test outside CI would notice'
    )


#: The whole Python surface, by role (hard rule 5).
PUBLIC_API = {
    'run it': {'build', 'check', 'evaluate', 'solve', 'write'},
    'run it many times': {'solve_over', 'EachCoordinate', 'EachWindow'},
    'carry it': {
        'SolveArchive',
        'SweepArchive',
        'load_archive',
        'load_result',
        'load_sweep',
        'scan_archive',
        'scan_result',
        'scan_sweep',
    },
    'name what came back': {'Model', 'Result', 'Sweep'},
    'catch it': {
        'SpecsolveError',
        'LanguageError',
        'DataError',
        'DimensionError',
        'LayoutError',
        'SchemaError',
        'NoSolutionError',
        'SpecsolveWarning',
    },
}


def test_the_public_surface_is_exactly_what_is_declared():
    """Hard rule 5, in names: the Python surface is narrow, and stays narrow.

    Two directions: ``__all__`` matches the table, and no public non-module
    attribute exists outside it.
    """
    import inspect

    import specsolve

    unresolved = sorted(name for name in specsolve.__all__ if not hasattr(specsolve, name))
    assert not unresolved, (
        f'__all__ names what the package does not bind: {unresolved} — `from specsolve import *` '
        f'raises, and an annotation naming one is only silent because it is never evaluated'
    )

    declared = {name for names in PUBLIC_API.values() for name in names}
    assert set(specsolve.__all__) == declared, (
        f'specsolve.__all__ and PUBLIC_API disagree: only in __all__ '
        f'{sorted(set(specsolve.__all__) - declared)}, only in the table '
        f'{sorted(declared - set(specsolve.__all__))} — add the name to PUBLIC_API '
        f'with the role it plays, and to docs/about/architecture.md'
    )

    leaked = sorted(
        name
        for name in dir(specsolve)
        if not name.startswith('_') and name not in declared and not inspect.ismodule(getattr(specsolve, name))
    )
    assert not leaked, (
        f'public names outside __all__: {leaked} — a surface that grows by '
        f'accident is not narrow. Import it privately, or declare it.'
    )


#: The two sink families; the directory is the family.
SINKS = PKG / 'relational' / 'sinks'


def _family(name: str) -> set[str]:
    return {p.stem for p in (SINKS / name).glob('*.py') if p.stem != '__init__'}


def test_each_sink_family_is_its_directory_and_its_registry():
    """One shape per family, checked off the path.

    A solver is a module under ``solvers/`` named for it, a ``Solver``
    subclass defined in that module, a ``build_<name>`` seam for `bench/`, and
    the ``SOLVERS`` key holding the class. Writers are keyed by suffix.
    """
    import importlib

    from specsolve.relational.sinks import SOLVERS, WRITERS, Solver

    solvers = _family('solvers') - {'base'}
    assert set(SOLVERS) == solvers, f'solver modules and SOLVERS keys disagree: {solvers ^ set(SOLVERS)}'
    for name in sorted(solvers):
        module = importlib.import_module(f'specsolve.relational.sinks.solvers.{name}')
        held = SOLVERS[name]
        assert issubclass(held, Solver), f'SOLVERS[{name!r}] is not a Solver'
        assert held.__module__.rsplit('.', 1)[-1] == name, (
            f"{name}'s solver is defined in {held.__module__} — it belongs to its own module"
        )
        assert held.requires and all(isinstance(package, str) for package in held.requires), (
            f'{name} does not name the packages it needs, so is_available() cannot answer for it'
        )
        assert isinstance(held.is_available(), bool), (
            f'{name}.is_available() must answer without importing the solver or raising'
        )
        assert held.unavailable_message, f'{name} does not say what to do when is_available() is False'
        assert hasattr(module, f'build_{name}'), f'{name} has no build_{name}: the load-only seam `bench/` measures'

    assert {w.write.__module__.rsplit('.', 1)[-1] for w in WRITERS.values()} == _family('writers') - {'base'}
    assert all(s.startswith('.') for s in WRITERS), 'writers are keyed by file suffix'


def test_every_sink_declares_what_it_can_ingest():
    """Every family answers the capability axis, in one vocabulary."""

    from specsolve.relational.sinks import EXPORTS, SOLVERS, WRITERS
    from specsolve.relational.sinks.capabilities import (
        CAPABILITIES,
        Capabilities,
        Support,
    )

    described = {f'solver {name}': held.capabilities for name, held in SOLVERS.items()}
    described |= {f'writer {suffix}': found.capabilities for suffix, found in WRITERS.items()}
    described |= {f'export {name}': capabilities for name, capabilities in EXPORTS.items()}
    for sink, capabilities in described.items():
        assert isinstance(capabilities, Capabilities), f'{sink} declares no capabilities'
        strangers = sorted(set(capabilities.supports) - set(CAPABILITIES))
        assert not strangers, f'{sink} names capabilities the vocabulary has not got: {strangers}'
        answers = sorted(set(capabilities.supports.values()) - set(get_args(Support)))
        assert not answers, f'{sink} answers {answers}, which no comparison in the family reads as support'
        for combination in capabilities.excludes:
            unsupported = sorted(c for c in combination if capabilities.support(c) == 'absent')
            assert not unsupported, (
                f'{sink} excludes the combination {sorted(combination)} while lacking {unsupported} '
                f'outright — an exclusion is about a *pair* it has both halves of, and a capability '
                f'it simply does not have is already refused on its own'
            )


def test_the_door_gives_every_declared_dimension_dtype_a_column():
    """``sources._DECLARED`` spells the dtype set the language validates."""
    from mathspec.program import DimensionDtype

    from specsolve.sources import _DECLARED

    assert set(_DECLARED) == set(get_args(DimensionDtype)), 'the two homes of the dimension dtype vocabulary disagree'


def test_the_door_accepts_the_declared_parameter_dtype_vocabulary():
    """Every declared dtype has a column table entry, and ``int`` for ``float`` is the only widening."""
    from mathspec.program import ParameterDtype

    from specsolve.sources import _COLUMNS, ACCEPTED_VALUE_TYPES

    declared = set(get_args(ParameterDtype))
    assert set(_COLUMNS) == declared, 'the column table and the language disagree'
    assert set(ACCEPTED_VALUE_TYPES) == declared, 'the accepted table and the language disagree'

    widened = {name: set(types) - set(_COLUMNS[name]) for name, types in ACCEPTED_VALUE_TYPES.items()}
    assert widened == {'float': set(_COLUMNS['int']), 'int': set(), 'bool': set(), 'str': set()}, (
        'int-for-float is the only widening'
    )


def test_no_sink_reaches_a_sibling():
    """The fence that keeps an optional dependency optional.

    A leaf reads ``handoff.py``, its family's ``base``, ``capabilities``, and
    its own dependency — nothing else in the family.
    """
    shareable = ('.handoff', '.base', '.capabilities')
    offenders = {}
    for family in ('solvers', 'writers'):
        for path in sorted((SINKS / family).glob('*.py')):
            reached = {
                name
                for name in _imported(ast.parse(path.read_text()))
                if name.startswith('specsolve.relational.sinks.') and not name.endswith(shareable)
            }
            if reached and path.stem != '__init__':
                offenders[f'{family}/{path.name}'] = sorted(reached)
    assert not offenders, (
        f'sink modules reaching a sibling: {offenders} — a sink reads handoff.py, its family base '
        f'and its own dependency; anything else shared belongs on one of those two'
    )


def test_every_plan_node_is_handled_by_the_compiler():
    """A primitive is not done until every module that walks its union names it."""
    from typing import get_args

    from mathspec import program

    engine_dir = PKG / 'relational' / 'engines' / 'polars'
    walkers = [
        ('program', program.Expression, engine_dir / 'compiler.py'),
        ('program', program.Expression, ORACLE / 'builder.py'),
        ('program', program.Predicate, engine_dir / 'predicates.py'),
        ('program', program.Predicate, ORACLE / 'where.py'),
    ]
    for qualifier, union, module in walkers:
        source = module.read_text()
        unhandled = [c.__name__ for c in get_args(union) if f'{qualifier}.{c.__name__}' not in source]
        assert not unhandled, f'{qualifier} nodes unknown to {module.name}: {unhandled}'


def test_the_spec_argument_is_what_the_language_takes_minus_the_lowered_form():
    """Every verb here opens a spec the way ``to_spec`` does, and a lowered ``Program`` is not one of them.

    Textual, since neither annotation evaluates. Splitting on ``|`` holds
    while every member is a flat name or a subscript.
    """
    import inspect

    from mathspec import to_spec

    def members(annotation: str) -> set[str]:
        return {part.strip() for part in annotation.split('|')}

    upstream = members(str(inspect.signature(to_spec).parameters['spec'].annotation))
    ours = members(type_alias_value(PKG / 'lanes.py', 'Buildable'))
    assert upstream == ours and 'Program' not in ours, (
        f'the language takes {sorted(upstream)} and specsolve.lanes.Buildable takes {sorted(ours)} — '
        f'every shape the language reads a spec from, and not the lowered Program'
    )


def type_alias_value(path: Path, name: str) -> str:
    """The right-hand side of ``type <name> = ...`` in *path*, as source text."""
    module = ast.parse(path.read_text())
    for node in module.body:
        if isinstance(node, ast.TypeAlias) and node.name.id == name:
            return ast.unparse(node.value)
    raise AssertionError(f'{path} declares no `type {name} = ...`')


def test_the_sources_argument_is_one_type_at_every_door():
    """Every verb that takes data annotates it ``Mapping[str, Source]``.

    The linopy lane's two verbs are asked in ``tests/test_linopy_lane.py``.
    """
    import specsolve
    from specsolve.strategy import EachCoordinate, EachWindow, solve_over

    doors = {
        'build': specsolve.build,
        'solve': specsolve.solve,
        'write': specsolve.write,
        'SolveArchive': specsolve.SolveArchive.__init__,
        'SweepArchive': specsolve.SweepArchive.__init__,
        'Model': specsolve.Model.__init__,
        'Model.update': specsolve.Model.update,
        'solve_over': solve_over,
        'EachCoordinate.slices': EachCoordinate.slices,
        'EachWindow.slices': EachWindow.slices,
    }
    assert sources_annotations(doors) == {'Mapping[str, Source]'}, (
        f'every door takes sources as Mapping[str, Source], and these do not: {sources_annotations(doors)}'
    )


def test_both_lanes_lower_a_spec_through_one_function():
    """Neither lane accepts a file the other refuses, which is what ``lowered`` is for."""
    import ast

    reading = {
        path.relative_to(PKG).as_posix()
        for path in PKG.rglob('*.py')
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Attribute) and node.attr == 'program' and isinstance(node.value, ast.Call)
    }
    assert reading == {'lanes.py'}, (
        f'a program is read off a freshly opened model in {sorted(reading)}; every lane lowers through '
        f'lanes.lowered, which is what refuses a spec this package cannot build or write down'
    )


def sources_annotations(doors: dict[str, Any]) -> set[str]:
    """What each door annotates ``sources`` with — ``tests/test_linopy_lane.py`` asks the same of the lane's."""
    import inspect

    return {str(inspect.signature(door).parameters['sources'].annotation) for door in doors.values()}


def _gen_bus_direction(program: Any) -> Any:
    """One map read one way — the shape every operator below takes a `by=` in."""
    gen_bus = program.RelationDeclaration((('g', 'g'), ('bus', 'bus')), ('g',))
    return program.Direction('gen_bus', gen_bus, ('g',), ('bus',), ())


def test_every_shape_operator_declares_its_fan_in():
    """The absence pass asks :func:`fan_in`, so it has to answer for each shape operator.

    Pinned as a truth table: fan-in is a semantic claim about each operator (#1142).
    """
    from mathspec import program

    from specsolve.relational.engines.polars.fragments import fan_in

    x = program.Variable('x')
    declared = {
        type(node).__name__: fan_in(node)
        for node in (
            program.Sum(x, ('t',)),
            program.GroupSum(x, _gen_bus_direction(program)),
            program.Pullback(x, _gen_bus_direction(program)),
            program.Translate(x, 't', 1, wrap=False),
            program.WindowSum(x, 't', 3, wrap=False),
        )
    }
    assert declared == {
        'Sum': 'many-to-one',
        'GroupSum': 'many-to-one',
        'Pullback': 'one-to-one',
        'Translate': 'one-to-one',
        'WindowSum': 'one-to-many',
    }, 'a fan-in moved — the absence pass now treats that operator differently, which is a semantic change'


#: A kwarg no built-in declares, which makes :func:`call_shape_error` answer with its usage line.
_NOT_A_KEYWORD = '#no such keyword'


def _declared_keywords(usage: str) -> set[str]:
    """The keywords a built-in takes, read off the language's own usage line."""
    return set(re.findall(r'([a-z_]+)=', usage))


def _keywords_read(fn: ast.FunctionDef, helpers: Mapping[str, ast.FunctionDef]) -> set[str]:
    """Every literal key *fn* takes off a ``.kwargs`` mapping, one level of helpers included."""
    found: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Subscript) and _is_kwargs(node.value) and isinstance(node.slice, ast.Constant):
            found.add(node.slice.value)
        elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant):
            found |= {node.left.value for c in node.comparators if _is_kwargs(c)}
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'get' and _is_kwargs(node.func.value):
                found |= {a.value for a in node.args[:1] if isinstance(a, ast.Constant)}
            elif isinstance(node.func, ast.Name) and node.func.id in helpers:
                called.add(node.func.id)
    for name in called:
        found |= _keywords_read(helpers[name], {})
    return found


def _dispatched_by_name(fn: ast.FunctionDef) -> set[str]:
    """The operators *fn* spells out by name — ``if node.name == 'at'``."""
    return {
        comparator.value
        for node in ast.walk(fn)
        if isinstance(node, ast.Compare) and _is_name(node.left) and isinstance(node.ops[0], ast.Eq)
        for comparator in node.comparators[:1]
        if isinstance(comparator, ast.Constant)
    }


def _is_name(node: ast.expr) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == 'name'


def _is_kwargs(node: ast.expr) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == 'kwargs'


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """Every function in *tree* by name, methods included, last definition winning."""
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def test_both_lanes_dispatch_on_every_plan_node():
    """Hard rule 3: the same plan, dispatched on by both lanes.

    Node kinds, not their fields: ``isinstance(x, program.Foo)`` names the
    class and cannot collide. Read statically, so this runs on a bare install.
    """
    from typing import get_args

    from mathspec import program

    declared = {node.__name__ for union in (program.Expression, program.Predicate) for node in get_args(union)}
    assert declared, 'no plan node classes found — the census has nothing to run over'

    def dispatched_on(*paths: Path) -> set[str]:
        """Every ``program.X`` named in an isinstance test, however the tuple is written."""
        found: set[str] = set()
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text())):
                if not (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'isinstance'
                ):
                    continue
                second = node.args[1] if len(node.args) > 1 else None
                options = second.elts if isinstance(second, ast.Tuple) else [second]
                found |= {
                    o.attr
                    for o in options
                    if isinstance(o, ast.Attribute) and getattr(o.value, 'id', None) == 'program'
                }
        return found

    lanes = {
        'relational': dispatched_on(*(PKG / 'relational' / 'engines' / 'polars').glob('*.py')),
        'linopy': dispatched_on(*ORACLE.glob('*.py')),
    }
    for lane, handled in lanes.items():
        assert not declared - handled, (
            f'the {lane} lane dispatches on {sorted(declared - handled)} nowhere — a node kind one '
            f'lane builds and the other falls through on is the dialect split hard rule 3 refuses'
        )


def test_every_module_is_documented_somewhere():
    """No module is undocumented, in docs/about/architecture.md or a ``README.md`` in a directory above it."""
    architecture = (REPO / 'docs/about/architecture.md').read_text()
    missing = []
    for path in _all_modules():
        name = path.name
        if name.startswith('_'):
            continue
        if name == '__init__.py':
            continue
        readmes = [d / 'README.md' for d in path.parents if PKG in d.parents or d == PKG]
        documented = name in architecture or any(r.exists() and name in r.read_text() for r in readmes)
        if not documented:
            missing.append(str(path.relative_to(PKG)))
    assert not missing, (
        f'undocumented modules: {missing} — add each to docs/about/architecture.md, or to a '
        f'README.md in its own directory if it is one member of a family'
    )


#: Every in-function ``specsolve`` import in the package, with the cycle it breaks.
DELIBERATE_LAZY_IMPORTS: dict[tuple[str, str], str] = {
    ('relational/engines/polars/predicates.py', 'specsolve.relational.engines.polars.compiler'): (
        'the same comparison on the streaming lane, and the compiler reads this module for the mask '
        'walk and the carrier both of its walks join on'
    ),
}


def test_lazy_intra_package_imports_are_all_declared():
    """Hard rule 0: an undeclared in-function import is an unnoticed cycle or a leftover."""
    found = {}
    for path in _all_modules():
        tree = ast.parse(path.read_text())
        module_level = set()
        stack = list(tree.body)
        while stack:
            node = stack.pop()
            module_level.add(id(node))
            if isinstance(node, ast.Try):
                stack += [*node.body, *node.orelse, *node.finalbody]
                stack += [b for h in node.handlers for b in h.body]
            elif isinstance(node, ast.If):
                stack += [*node.body, *node.orelse]
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith('specsolve')
                and id(node) not in module_level
            ):
                found[(str(path.relative_to(PKG)), node.module)] = node.lineno

    undeclared = {k: v for k, v in found.items() if k not in DELIBERATE_LAZY_IMPORTS}
    assert not undeclared, (
        f'undeclared in-function imports {undeclared} — hoist them to module level, '
        f'or add them to DELIBERATE_LAZY_IMPORTS with the cycle they break'
    )
    stale = set(DELIBERATE_LAZY_IMPORTS) - set(found)
    assert not stale, f'DELIBERATE_LAZY_IMPORTS lists imports that no longer exist: {stale}'
