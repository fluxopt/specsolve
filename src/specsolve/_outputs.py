"""What an answer carries: the kinds of frame it holds, and the outputs a solve is asked for.

Every answer holds a ``primal``, a ``dual`` and an ``expression`` frame per
name. An [`Output`][] asked for with ``outputs=`` adds the kinds in
[`OUTPUT_KINDS`][]. A result, a sweep and the layout on disk all key their
frames by these [`KINDS`][].
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeGuard, get_args

from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

#: What an answer carries only where the solve asked for it with ``outputs=``,
#: beside the primal, the duals and the declared expressions it always
#: carries. ``activity`` is each constraint's left-hand side at the solution,
#: ``reduced_cost`` each variable's reduced cost, ``slack`` each
#: constraint's distance to binding, and ``basis`` the basis status the solve
#: ended on, of each variable and each constraint.
Output = Literal['activity', 'reduced_cost', 'slack', 'basis']

#: Every [`Output`][], in the order a message lists them.
OUTPUTS: tuple[Output, ...] = get_args(Output)

#: What a kind of frame holds one frame per.
Per = Literal['variable', 'constraint']


class OutputKind(NamedTuple):
    """What one kind of frame an [`Output`][] carries is."""

    #: The output that asks for it.
    output: Output
    #: What it holds one frame per.
    per: Per


#: Each kind of frame the [`OUTPUTS`][] carry, named after the reader it comes
#: back through: ``basis`` carries two, ``variable_basis`` and
#: ``constraint_basis``, and every other output one, of its own name.
OUTPUT_KINDS: Mapping[str, OutputKind] = MappingProxyType(
    {
        'activity': OutputKind('activity', 'constraint'),
        'reduced_cost': OutputKind('reduced_cost', 'variable'),
        'slack': OutputKind('slack', 'constraint'),
        'variable_basis': OutputKind('basis', 'variable'),
        'constraint_basis': OutputKind('basis', 'constraint'),
    }
)

#: The kinds that exist exactly where the duals do. Where a solve left none,
#: each carries the duals' reason.
PRICED = frozenset({'dual', 'reduced_cost'})

#: The kinds ``basis`` carries, which exist only where the solve ended on a
#: basis. Where it did not, each carries [`NO_BASIS`][].
BASES = frozenset(kind for kind, carried in OUTPUT_KINDS.items() if carried.output == 'basis')

NO_BASIS = (
    'the solve ended on no basis, so there is no basis status to read. Only an LP solved by simplex, or by '
    'an interior-point method followed by crossover, ends on one: a mixed-integer model, a model with a '
    'quadratic constraint, and an interior-point run with crossover off do not. Solve with crossover on — '
    "HiGHS's 'run_crossover', Gurobi's 'Crossover', Xpress's 'crossover' — or with a simplex method."
)

#: Every kind of frame an answer can hold: the three every solve answers with,
#: then the [`OUTPUT_KINDS`][]. Each is named after the reader it comes back
#: through.
KINDS = ('primal', 'dual', 'expression', *OUTPUT_KINDS)


def is_output(name: str) -> TypeGuard[Output]:
    """Whether *name* is one of the [`OUTPUTS`][]."""
    return name in OUTPUTS


def checked_outputs(outputs: Iterable[str]) -> frozenset[Output]:
    """*outputs* as a set, once each is checked to be one of [`OUTPUTS`][].

    Raises:
        SpecsolveError: One string rather than a collection of them, or a name
            that is not an output.
    """
    if isinstance(outputs, str):
        raise SpecsolveError(
            f'outputs={outputs!r} is one string, which would name each of its letters. '
            f'Pass a set: outputs={{{outputs!r}}}.'
        )
    asked = frozenset(outputs)
    if unknown := sorted(name for name in asked if not is_output(name)):
        raise SpecsolveError(
            f'outputs names {", ".join(map(repr, unknown))}, and an answer carries only '
            f'{", ".join(map(repr, OUTPUTS))} on request. The primal, the duals and the declared '
            f'expressions are carried always and are not named here.'
        )
    return frozenset(filter(is_output, asked))


def kinds_of(outputs: Iterable[Output]) -> tuple[str, ...]:
    """Each of the [`OUTPUT_KINDS`][] *outputs* carry."""
    return tuple(kind for kind, carried in OUTPUT_KINDS.items() if carried.output in outputs)


def outputs_of(kinds: Iterable[str]) -> frozenset[Output]:
    """The [`OUTPUTS`][] that carry *kinds*, each one of the [`OUTPUT_KINDS`][]."""
    return frozenset(OUTPUT_KINDS[kind].output for kind in kinds)


def not_requested_message(kind: str, name: str) -> str:
    """Why an answer refuses *kind*, one of the [`OUTPUT_KINDS`][], of *name*: the solve did not ask for it."""
    return (
        f"cannot read the {kind.replace('_', ' ')} of '{name}': the solve was not asked for it, so this "
        f'answer does not carry it. An answer carries the primal, the duals and the declared expressions, '
        f"and anything else only on request. Solve again with outputs={{'{OUTPUT_KINDS[kind].output}'}}."
    )


def checked_kind(kind: str) -> str:
    """*kind*, returned once it is checked to be one of [`KINDS`][].

    Raises:
        SpecsolveError: A *kind* that names no reader.
    """
    if kind not in KINDS:
        raise SpecsolveError(f'kind is one of {", ".join(KINDS)}, not {kind!r}')
    return kind
