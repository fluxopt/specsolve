"""Every expression up to a bounded depth, and the rewrites that must not move it.

At a bounded depth over two dimensions the space of spellings is finite, so
``test_expression_sweep.py`` claims every one of them means the same on both
lanes.

Each node carries the dimensions and degree it produces, and the constructors
that can fail refuse what the language would refuse, so a load error is a
failure. The rule-carrying rewrites are constructed over an operand pool, one
builder per rule. ``reduction-is-linear`` holds only while every operand is
total, which is read off the tree. A degree-two node is generated only as an
operand: the linopy lane refuses a quadratic constraint (#942).
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
# the constructors — None is what the language would refuse
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

    With no ``edge=`` the vacated edge is absent
    (docs/reference/language/operators.md).
    """
    if over not in a.dims:
        return None
    return Node(f'shift({a}, along={over}, offset=1)', a.dims, a.degree, total=False)


#: Every way to grow an expression by one node, ordered so a failing id can be found again.
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

    Data is refused in a constraint, and the linopy lane cannot build a
    quadratic row (#942).
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
    """The shared fixture's model, with *node* as its one binding row over its own dims."""
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
    """The operands the rule-carrying shapes are built over: depth two, and total."""
    return tuple(node for node in _grown(LEAVES) if node.total)


def _negate_through_sum(pool: tuple[Node, ...], over: str) -> Iterator[Rewrite]:
    """A negation moves through a reduction: ``sum(-(a))`` is ``-(sum(a))``."""
    for a in pool:
        inner = summed(a, over)
        if a.degree == 1 and inner is not None:
            yield Rewrite('negate-through-sum', summed(negate(a), over), negate(inner))


def _reduction_is_linear(pool: tuple[Node, ...], over: str) -> Iterator[Rewrite]:
    """A reduction splits across a sum: ``sum(a + b)`` is ``sum(a) + sum(b)``.

    Only while both operands are total, and only for a linear summand (#942).
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
        The pairs, ordered by rule and then by operand.
    """
    pool = _pool()
    over_a_dim = (_negate_through_sum, _reduction_is_linear)
    built = [pair for rule in over_a_dim for over in sorted(LAW_DIMS) for pair in rule(pool, over)]
    return tuple(built + list(_double_negation(pool)))
