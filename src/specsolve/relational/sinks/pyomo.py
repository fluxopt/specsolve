"""The ``pyomo`` export: the built model as a ``pyomo.environ.ConcreteModel``.

One ``Var`` per declared variable, one ``Constraint`` per declared constraint
and one ``SOSConstraint`` per declared set, each indexed by the coordinates the
build produced, so a coordinate a ``where`` removed has no entry. A row holds
the build's numbers: it is a flat sum of terms, not the formula the file wrote.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from specsolve.relational.sinks.handoff import Declared, Handoff, Run

__all__ = ['OBJECTIVE', 'RESULT_SUFFIXES', 'UNAVAILABLE', 'to_pyomo']

#: What calling the export without the extra says.
UNAVAILABLE = 'to_pyomo requires the [pyomo] extra: pip install "specsolve[pyomo]"'

#: The component the objective is added as.
OBJECTIVE = 'objective'

#: The suffixes pyomo's solver interfaces look up on a model by name and load
#: results into, so a component of that name breaks every solve.
RESULT_SUFFIXES = frozenset({'dual', 'rc', 'slack'})


def to_pyomo(handoff: Handoff, declared: Declared) -> Any:
    """The built model as a ``ConcreteModel``.

    Components are added variables first, then constraints, sets and the
    objective, and one whose name an earlier one took is suffixed with its
    kind: ``start_up_constraint`` beside the variable ``start_up``. So is one
    named like a result suffix pyomo's solvers read: ``slack_variable``.

    Raises:
        SpecsolveError: pyomo is not installed, or a declaration's name is
            already an attribute of a ``ConcreteModel``.
    """
    try:
        import pyomo.environ as pyo
    except ModuleNotFoundError as missing:
        raise SpecsolveError(UNAVAILABLE) from missing

    m = pyo.ConcreteModel()
    columns = _variables(pyo, m, handoff, declared.variables)
    _constraints(pyo, m, handoff, declared.constraints, columns)
    _sets(pyo, m, handoff, declared, columns)
    if handoff.objective_sense is not None:
        body = _linear(handoff.obj['coeff'].to_list(), [columns[c] for c in handoff.obj['col']])
        quadratic = _quadratic(handoff.quad, columns)
        sense = pyo.maximize if handoff.objective_sense == 'maximize' else pyo.minimize
        _add(m, OBJECTIVE, 'objective', pyo.Objective(expr=body + quadratic + handoff.objective_constant, sense=sense))
    return m


def _variables(pyo: Any, m: Any, handoff: Handoff, runs: Sequence[Run]) -> list[Any]:
    """Add every variable, and return its column objects in solver order."""
    domains = {'continuous': pyo.Reals, 'binary': pyo.Binary, 'integer': pyo.Integers}
    lb, ub, vtype = (handoff.cols[c].to_list() for c in ('lb', 'ub', 'vtype'))
    columns: list[Any] = [None] * handoff.column_count
    for run in runs:
        coordinates = _coordinates(run)
        domain = domains[vtype[run.start]] if run.height else pyo.Reals
        var = _component(pyo.Var, run.dims, [_index(c) for c in coordinates], domain=domain)
        _add(m, run.name, 'variable', var)
        for position, coordinate in enumerate(coordinates, start=run.start):
            column = var[_index(coordinate)] if run.dims else var
            column.setlb(lb[position])
            column.setub(ub[position])
            columns[position] = column
    return columns


def _constraints(pyo: Any, m: Any, handoff: Handoff, runs: Sequence[Run], columns: list[Any]) -> None:
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
        _add(m, run.name, 'constraint', _component(pyo.Constraint, run.dims, list(rows), rows))


def _sets(pyo: Any, m: Any, handoff: Handoff, declared: Declared, columns: list[Any]) -> None:
    """Add every ``sos:`` declaration, each set indexed by its members' coordinate minus the ordering dim.

    A declaration's sets are a contiguous run of set numbers, in declaration
    order, so the runs are read off one after another.
    """
    held = {run.name: (run, _coordinates(run)) for run in declared.variables}
    members = handoff.sos.partition_by('set', as_dict=True)
    first = 0
    for sets in declared.sets:
        variable, coordinates = held[sets.variable]
        chosen = {}
        for number in range(first, first + sets.count):
            frame = members[(number,)]
            cols = frame['col'].to_list()
            coordinate = coordinates[cols[0] - variable.start]
            projected = coordinate[: sets.along] + coordinate[sets.along + 1 :]
            chosen[_index(projected)] = ([columns[c] for c in cols], frame['weight'].to_list())
        first += sets.count
        dims = variable.dims[: sets.along] + variable.dims[sets.along + 1 :]
        _add(m, sets.name, 'sos', _component(pyo.SOSConstraint, dims, list(chosen), chosen, sos=sets.sos_type))


def _coordinates(run: Run) -> list[tuple[Any, ...]]:
    """*run*'s coordinates as label tuples, the empty tuple for each entry of a declaration with no dims."""
    return run.coordinates.rows() if run.dims else [()] * run.height


def _component(
    kind: Any, dims: tuple[str, ...], index: list[Hashable], values: dict[Hashable, Any] | None = None, **options: Any
) -> Any:
    """*kind* over *index*, or a scalar one where the declaration has no dims.

    With *values*, each entry is read off it by a rule. Pyomo unpacks a tuple
    index into the rule's arguments, and calls a scalar's rule with the model
    alone, so the two take different rules. A scalar the build left empty, such
    as a row it dropped for having no term, is skipped rather than built.
    """
    if values is None:
        return kind(index, **options) if dims else kind(**options)

    def scalar(_m: Any) -> Any:
        return values.get((), kind.Skip)

    def indexed(_m: Any, *at: Any) -> Any:
        return values[_index(at)]

    return kind(index, rule=indexed, **options) if dims else kind(rule=scalar, **options)


def _index(coordinate: tuple[Any, ...]) -> Hashable:
    """A coordinate as pyomo spells an index: the label itself for one dim, the tuple for more."""
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


def _add(m: Any, name: str, kind: str, component: Any) -> None:
    """Add *component* as *name*, or as ``<name>_<kind>`` where *name* is taken.

    An earlier component takes a name, since a spec may give a variable and a
    constraint one name and a model has one namespace; so does a result
    suffix. A name the model already answers to otherwise, such as ``write``,
    is refused.
    """
    if name in RESULT_SUFFIXES or m.component(name) is not None:
        name = f'{name}_{kind}'
    if hasattr(m, name):
        raise SpecsolveError(
            f"'{name}' is already an attribute of a pyomo ConcreteModel, so it cannot also be a "
            'component of one. Rename the declaration to export the model to pyomo.'
        )
    m.add_component(name, component)
