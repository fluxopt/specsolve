"""Every expression up to a bounded depth, and the rewrites that must not move it.

The AST is closed, which is what makes this different from sampling: at depth
three over two dimensions the space of spellings is *finite*, so the sweep in
``test_expression_sweep.py`` makes the strong claim — every one of them means
the same on both lanes — rather than "a hundred did".

**Built well-formed, not filtered.** Each node carries the dimensions and the
degree it produces, and the constructors that can fail refuse what the language
would refuse: summing over a dimension the operand does not carry, a product
past degree two. Trial-loading the candidates instead cost twenty seconds of
collection, in every xdist worker, and turned a genuine load error into one more
rejected candidate. Here a load error is a failure.

**Rewrites are built, not found.** Depth bounds expression *size*, and the rules
worth checking need a *shape* — which gets rarer as the space grows, not
commoner. Enumerating to depth three and collecting the rules that happened to
fire yielded 3,824 commutativity pairs, which ``test_arithmetic_laws.py``
already states by hand, and ten of ``reduction-is-linear``, which is the one the
sweep exists for (#1203). So the rule-carrying shapes are constructed over an
operand pool instead — one builder per rule, below — and the two commute rules
are left to the curated file.

``reduction-is-linear`` is the rule with a side condition: ``sum(a + b)`` and
``sum(a) + sum(b)`` are equal only while every operand is *total*, and the
divergence when they are not went unnoticed for as long as the oracle shared it
(#311). The condition is read off the tree rather than assumed.

**A degree-two node is generated only as an operand, never emitted as a
case.** The linopy lane refuses a quadratic constraint outright (#942), and
every case here is a constraint row, so a degree-two expression would make the
sweep report a lane limit it already knows about.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import pytest

from tests.conftest import LAW_DIMS, law_spec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass(frozen=True)
class Node:
    """One expression, with what the language would say about it.

    Attributes:
        text: The expression as it is written in the model file.
        dims: The dimensions the expression carries.
        degree: 0 for data, 1 linear, 2 quadratic.
        total: Whether no masked variable appears anywhere inside.
    """

    text: str
    dims: frozenset[str]
    degree: int
    total: bool

    def __str__(self) -> str:
        return self.text


#: The leaves: a total variable, a masked one, a parameter, and a literal.
LEAVES = (
    Node('x', frozenset({'f', 't'}), 1, total=True),
    Node('y', frozenset({'f', 't'}), 1, total=False),
    Node('w', frozenset({'f'}), 0, total=True),
    Node('2', frozenset(), 0, total=True),
)


# ---------------------------------------------------------------------------
# the constructors — the two that cannot fail return a node, the three that can
# return None, which is how the enumeration drops what the language would refuse
# without ever writing it down
# ---------------------------------------------------------------------------


def negate(a: Node) -> Node:
    return Node(f'-({a})', a.dims, a.degree, a.total)


def add(a: Node, b: Node) -> Node:
    return Node(f'({a}) + ({b})', a.dims | b.dims, max(a.degree, b.degree), a.total and b.total)


def multiply(a: Node, b: Node) -> Node | None:
    """A product past degree two is not in the language, so it is not generated."""
    if a.degree + b.degree > 2:
        return None
    return Node(f'({a}) * ({b})', a.dims | b.dims, a.degree + b.degree, a.total and b.total)


def summed(a: Node, over: str) -> Node | None:
    """Summing over a dimension the operand does not carry is a load error."""
    if over not in a.dims:
        return None
    return Node(f'sum({a}, over={over})', a.dims - {over}, a.degree, a.total)


def shifted(a: Node, over: str) -> Node | None:
    """Likewise for shifting along one — and the result is never total.

    With no ``edge=``, "the vacated edge is absent"
    (docs/reference/language/operators.md), so a shift introduces absence
    wherever it lands however total its operand was. Propagating ``total``
    through it made the generator emit ``reduction-is-linear`` pairs whose side
    condition does not hold, and they duly disagreed (#1203) — the law is
    intact, the claim about the operand was not.
    """
    if over not in a.dims:
        return None
    return Node(f'shift({a}, along={over}, offset=1)', a.dims, a.degree, total=False)


#: Every way to grow an expression by one node. Ordered, so the enumeration is
#: the same list on every machine and a failing id can be found again.
UNARY: tuple[Callable[[Node], Node | None], ...] = (
    negate,
    partial(summed, over='f'),
    partial(summed, over='t'),
    partial(shifted, over='t'),
)
BINARY: tuple[Callable[[Node, Node], Node | None], ...] = (add, multiply)


def _grown(previous: tuple[Node, ...]) -> tuple[Node, ...]:
    grown = list(previous)
    grown += [made for grow in UNARY for a in previous if (made := grow(a)) is not None]
    grown += [made for grow in BINARY for a in previous for b in previous if (made := grow(a, b)) is not None]
    return tuple(dict.fromkeys(grown))


def space(depth: int) -> tuple[Node, ...]:
    """Every well-formed expression of at most *depth* nodes deep, deduplicated, at any degree."""
    grown: tuple[Node, ...] = LEAVES
    for _ in range(depth - 1):
        grown = _grown(grown)
    return grown


def expressions(depth: int) -> tuple[Node, ...]:
    """The expressions of :func:`space` a constraint row can be written from.

    One with no variable in it is data, which the language refuses in a
    constraint, and a quadratic one is a row the linopy lane cannot build at
    all (#942).
    """
    return tuple(node for node in space(depth) if node.degree == 1)


def stride(cases: tuple, spec: str) -> tuple:
    """The shard of *cases* that ``i/n`` asks for — every n-th, offset by i.

    A stride rather than a contiguous block: the space is ordered by shape, so
    consecutive cases cost about the same and a block would hand one leg all
    the cheap ones. Every case is in exactly one shard for any n, which is what
    lets the legs be compared with the unsharded run.

    Raises:
        pytest.UsageError: If *spec* is not ``i/n`` with ``0 <= i < n``. A shard
            nobody runs is coverage lost in a green job.
    """
    i, _, n = spec.partition('/')
    if not (i.isdigit() and n.isdigit() and 0 <= int(i) < int(n)):
        raise pytest.UsageError(f'--sweep-shard takes `i/n` with 0 <= i < n, not {spec!r}')
    return cases[int(i) :: int(n)]


def row_spec(node: Node) -> dict:
    """The shared fixture's model, with *node* as its one binding row.

    ``dims`` is the expression's own dimensions: anything else is a row
    repeated across a dimension the expression does not carry, which the
    language refuses — so it is computed rather than searched for.
    """
    return law_spec(f'{node} <= 10', dims=sorted(node.dims))


# ---------------------------------------------------------------------------
# the rewrites — one builder per rule, because the rule is the case
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rewrite:
    """A rule and the pair it produced, so a failure names both spellings."""

    rule: str
    before: Node
    after: Node


def _pool() -> tuple[Node, ...]:
    """The operands the rule-carrying shapes are built over.

    Depth two, and total — a masked operand is what ``reduction-is-linear`` is
    conditional on, so it belongs in the curated non-laws rather than here,
    where every pair must be an equality.
    """
    return tuple(node for node in _grown(LEAVES) if node.total)


def _negate_through_sum(pool: tuple[Node, ...], over: str) -> Iterator[Rewrite]:
    """A negation moves through a reduction: ``sum(-(a))`` is ``-(sum(a))``."""
    for a in pool:
        inner = summed(a, over)
        if a.degree == 1 and inner is not None:
            yield Rewrite('negate-through-sum', summed(negate(a), over), negate(inner))


def _reduction_is_linear(pool: tuple[Node, ...], over: str) -> Iterator[Rewrite]:
    """A reduction splits across a sum: ``sum(a + b)`` is ``sum(a) + sum(b)``.

    Only while both operands are total, which is what ``pool`` guarantees, and
    only where the whole is a row the linopy lane can build — a degree-two
    summand is #942's gap rather than a disagreement.
    """
    for a in pool:
        for b in pool:
            summand = add(a, b)
            parts = summed(summand, over), summed(a, over), summed(b, over)
            if summand.degree == 1 and all(part is not None for part in parts):
                whole, left, right = parts
                yield Rewrite('reduction-is-linear', whole, add(left, right))


def _double_negation(pool: tuple[Node, ...]) -> Iterator[Rewrite]:
    """Two negations are none: ``-(-(a))`` is ``a``."""
    for a in pool:
        if a.degree == 1:
            yield Rewrite('double-negation', negate(negate(a)), a)


def rewrites() -> tuple[Rewrite, ...]:
    """Every rule-carrying shape, built over the operand pool.

    Returns:
        The pairs, ordered by rule and then by operand, so a failing id can be
        found again and the census sampling a stride of them samples the same
        stride on every machine.
    """
    pool = _pool()
    over_a_dim = (_negate_through_sum, _reduction_is_linear)
    built = [pair for rule in over_a_dim for over in sorted(LAW_DIMS) for pair in rule(pool, over)]
    return tuple(built + list(_double_negation(pool)))
