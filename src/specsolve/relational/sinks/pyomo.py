"""The ``pyomo`` export: the built model as a ``pyomo.environ.ConcreteModel``.

One ``Var`` per declared variable, one ``Constraint`` per declared constraint
and one ``SOSConstraint`` per declared set, each indexed by the coordinates the
build produced, so a coordinate a ``where`` removed has no entry. A row holds
the build's numbers: it is a flat sum of terms, not the formula the file wrote.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from specsolve.errors import SpecsolveError, unknown_name_message
from specsolve.relational.sinks.capabilities import Capabilities

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from specsolve.relational.sinks.handoff import Declared, Handoff, Run

__all__ = ['PYOMO_CAPABILITIES', 'component_names', 'to_pyomo']

#: A pyomo model holds every construct the language has; what it refuses is a name.
PYOMO_CAPABILITIES = Capabilities(
    supports={
        'integrality': 'native',
        'sos': 'native',
        'quadratic_objective': 'native',
        'nonconvex_quadratic_objective': 'native',
        'quadratic_constraint': 'native',
    }
)

#: What calling the export or its check without the extra says.
UNAVAILABLE = 'to_pyomo requires the [pyomo] extra: pip install "specsolve[pyomo]"'

#: The suffixes pyomo's solver interfaces look up on a model by name and load
#: results into, so a component of that name breaks every solve.
_RESULT_SUFFIXES = frozenset({'dual', 'rc', 'slack'})

#: The component the objective is added as.
_OBJECTIVE = 'objective'

#: The sections a ``rename`` is keyed by, as the file names them, beside the kind each holds.
_SECTIONS = {'variables': 'variable', 'constraints': 'constraint', 'sos': 'sos'}


def to_pyomo(handoff: Handoff, declared: Declared, rename: Mapping[str, Mapping[str, str]] | None = None) -> Any:
    """The built model as a ``ConcreteModel``, as [`Model.to_pyomo`][specsolve.api.Model.to_pyomo] describes it.

    Raises:
        SpecsolveError: What [`component_names`][] refuses.
    """
    names = component_names(declared, rename, objective=handoff.objective_sense is not None)
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    columns = _variables(pyo, m, handoff, declared.variables, names)
    _constraints(pyo, m, handoff, declared.constraints, columns, names)
    _sets(pyo, m, handoff, declared, columns, names)
    if handoff.objective_sense is not None:
        body = _linear(handoff.obj['coeff'].to_list(), [columns[c] for c in handoff.obj['col']])
        quadratic = _quadratic(handoff.quad, columns)
        sense = pyo.maximize if handoff.objective_sense == 'maximize' else pyo.minimize
        m.add_component(_OBJECTIVE, pyo.Objective(expr=body + quadratic + handoff.objective_constant, sense=sense))
    return m


def component_names(
    declared: Declared, rename: Mapping[str, Mapping[str, str]] | None, *, objective: bool
) -> dict[tuple[str, str], str]:
    """Each declaration's component name, keyed by ``(section, declaration)``: as declared, or as *rename* says.

    Every name is decided before any component is added, so a clash is
    refused whole: one error lists each one with the ``rename`` that names
    them apart. Nothing is renamed that the caller did not rename.

    Raises:
        SpecsolveError: pyomo is not installed, so the names every
            ``ConcreteModel`` holds are unknown; *rename* names a section or a
            declaration the spec does not have; or a name is taken twice, is a
            result suffix, or is an attribute every ``ConcreteModel`` has.
    """
    try:
        import pyomo.environ as pyo
    except ModuleNotFoundError as missing:
        raise SpecsolveError(UNAVAILABLE) from missing

    m = pyo.ConcreteModel()
    rename = rename or {}
    declarations = {
        'variables': [run.name for run in declared.variables],
        'constraints': [run.name for run in declared.constraints],
        'sos': [s.name for s in declared.sets],
    }
    for section, renamed in rename.items():
        if section not in _SECTIONS:
            raise SpecsolveError(unknown_name_message('rename section', section, _SECTIONS))
        for name in renamed:
            if name not in declarations[section]:
                raise SpecsolveError(unknown_name_message(_SECTIONS[section], name, declarations[section]))

    names = {
        (section, name): rename.get(section, {}).get(name, name)
        for section in _SECTIONS
        for name in declarations[section]
    }
    taken: dict[str, str] = {_OBJECTIVE: 'the objective'} if objective else {}
    clashes: list[tuple[str, str, str]] = []
    for (section, name), chosen in names.items():
        kind = _SECTIONS[section]
        why = _why_taken(m, chosen, taken)
        if why is not None:
            label = f"{kind} '{name}'" if chosen == name else f"{kind} '{name}' renamed to '{chosen}'"
            clashes.append((section, name, f'{label} {why}'))
        taken[chosen] = f"{kind} '{name}'"
    if clashes:
        raise SpecsolveError(
            f'a pyomo model has one namespace, and {len(clashes)} component name(s) cannot be used: '
            f'{"; ".join(why for _, _, why in clashes)}. '
            f'Pass rename={_suggestion(m, names, taken, rename, clashes)!r} to name them apart.'
        )
    return names


def _why_taken(m: Any, name: str, taken: Mapping[str, str]) -> str | None:
    """Why *name* cannot be a component, or ``None`` where it can."""
    if name in taken:
        return f'takes the name of {taken[name]}'
    if name in _RESULT_SUFFIXES:
        return "is a suffix pyomo's solvers load results into"
    if hasattr(m, name):
        return 'is an attribute every pyomo ConcreteModel has'
    return None


def _suggestion(
    m: Any,
    names: Mapping[tuple[str, str], str],
    taken: Mapping[str, str],
    rename: Mapping[str, Mapping[str, str]],
    clashes: Sequence[tuple[str, str, str]],
) -> dict[str, dict[str, str]]:
    """*rename* plus ``<name>_<kind>`` for each clash, or ``<name>_<kind>_2`` and up where that is taken too.

    A proposed name avoids every name the model will use, so passing the
    suggestion back resolves each clash. Two clashes never share a base, since
    a section's names are distinct and each section appends its own kind.
    """
    used = {**dict.fromkeys(names.values(), 'a declaration'), **taken}
    suggestion = {section: dict(renamed) for section, renamed in rename.items()}
    for section, name, _ in clashes:
        base = f'{name}_{_SECTIONS[section]}'
        candidate, n = base, 1
        while _why_taken(m, candidate, used) is not None:
            n += 1
            candidate = f'{base}_{n}'
        suggestion.setdefault(section, {})[name] = candidate
    return suggestion


def _variables(
    pyo: Any, m: Any, handoff: Handoff, runs: Sequence[Run], names: Mapping[tuple[str, str], str]
) -> list[Any]:
    """Add every variable, and return its column objects in solver order."""
    domains = {'continuous': pyo.Reals, 'binary': pyo.Binary, 'integer': pyo.Integers}
    lb, ub, vtype = (handoff.cols[c].to_list() for c in ('lb', 'ub', 'vtype'))
    columns: list[Any] = []
    for run in runs:
        index = [_index(c) for c in _coordinates(run)]
        bounds = {at: (lb[position], ub[position]) for position, at in enumerate(index, start=run.start)}
        domain = domains[vtype[run.start]] if run.height else pyo.Reals
        var = _component(pyo.Var, run.dims, bounds, domain=domain, bounds=_rule(bounds))
        m.add_component(names['variables', run.name], var)
        columns.extend(var[at] for at in index)
    return columns


def _constraints(
    pyo: Any, m: Any, handoff: Handoff, runs: Sequence[Run], columns: list[Any], names: Mapping[tuple[str, str], str]
) -> None:
    """Add every constraint, each row the matrix's terms plus its quadratic ones."""
    starts = handoff.row_starts
    col, coeff = handoff.matrix['col'].to_numpy(), handoff.matrix['coeff'].to_list()
    quadratic = {row: frame for (row,), frame in handoff.qmatrix.partition_by('row', as_dict=True).items()}
    sense, rhs = handoff.rows['sense'].cast(str).to_list(), handoff.rows['rhs'].to_list()
    for run in runs:
        rows = {}
        for row, coordinate in enumerate(_coordinates(run), start=run.start):
            span = slice(int(starts[row]), int(starts[row + 1]))
            body = _linear(coeff[span], [columns[c] for c in col[span]])
            if row in quadratic:
                body = body + _quadratic(quadratic[row], columns)
            rows[_index(coordinate)] = _compared(body, sense[row], rhs[row])
        component = _component(pyo.Constraint, run.dims, rows, rule=_rule(rows, pyo.Constraint.Skip))
        m.add_component(names['constraints', run.name], component)


def _sets(
    pyo: Any, m: Any, handoff: Handoff, declared: Declared, columns: list[Any], names: Mapping[tuple[str, str], str]
) -> None:
    """Add every ``sos:`` declaration, each set indexed by its members' coordinate minus the ordering dim.

    A variable carries at most one declaration's sets, so a declaration's sets
    are the ones over its variable's columns.
    """
    held = {run.name: run for run in declared.variables}
    for sets in declared.sets:
        variable = held[sets.variable]
        coordinates = _coordinates(variable)
        owned = handoff.sos.filter(pl.col('col').is_between(variable.start, variable.start + variable.height - 1))
        chosen = {}
        for frame in owned.partition_by('set', maintain_order=True):
            cols = frame['col'].to_list()
            coordinate = coordinates[cols[0] - variable.start]
            projected = coordinate[: sets.along] + coordinate[sets.along + 1 :]
            chosen[_index(projected)] = ([columns[c] for c in cols], frame['weight'].to_list())
        dims = variable.dims[: sets.along] + variable.dims[sets.along + 1 :]
        rule = _rule(chosen, pyo.SOSConstraint.Skip)
        m.add_component(
            names['sos', sets.name], _component(pyo.SOSConstraint, dims, chosen, rule=rule, sos=sets.sos_type)
        )


def _coordinates(run: Run) -> list[tuple[Any, ...]]:
    """*run*'s coordinates as label tuples, the empty tuple for each entry of a declaration with no dims."""
    return run.coordinates.rows() if run.dims else [()] * run.height


def _component(kind: Any, dims: tuple[str, ...], values: dict[Hashable, Any], **options: Any) -> Any:
    """*kind* over *values*' keys, or a scalar one where the declaration has no dims."""
    return kind(list(values), **options) if dims else kind(**options)


def _rule(values: dict[Hashable, Any], missing: Any = None) -> Any:
    """A rule that reads each entry off *values*, and *missing* where the build left none.

    Pyomo calls a rule with the index unpacked, and a scalar's with ``None``,
    so one rule serves both where *values* is keyed as ``_index`` spells it.
    A scalar the build left empty, such as a row it dropped for having no
    term, reads *missing*.
    """

    def rule(_m: Any, *at: Any) -> Any:
        return values.get(_index(at), missing)

    return rule


def _index(coordinate: tuple[Any, ...]) -> Hashable:
    """A coordinate as pyomo spells an index: ``None`` for no dims, the label for one, the tuple for more."""
    if not coordinate:
        return None
    return coordinate[0] if len(coordinate) == 1 else coordinate


def _linear(coefficients: Sequence[float], variables: Sequence[Any]) -> Any:
    """A linear sum pyomo keeps as a relation even with no terms, so ``0 >= 10`` stays infeasible."""
    from pyomo.core.expr.numeric_expr import LinearExpression, MonomialTermExpression

    return LinearExpression([MonomialTermExpression((c, v)) for c, v in zip(coefficients, variables, strict=True)])


def _quadratic(pairs: Any, columns: list[Any]) -> Any:
    """The ``(col_l, col_r, coeff)`` pairs as a sum of products, each pair whole."""
    return sum(
        (c * columns[left] * columns[right] for left, right, c in pairs.select('col_l', 'col_r', 'coeff').iter_rows()),
        start=0,
    )


def _compared(body: Any, sense: str, rhs: float) -> Any:
    if sense == '<=':
        return body <= rhs
    if sense == '>=':
        return body >= rhs
    return body == rhs
