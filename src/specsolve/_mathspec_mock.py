"""PROTOTYPE ONLY: the mathspec half of nonlinear support, patched onto mathspec 0.3.0 at import.

Nothing here belongs in specsolve. It stands in for what mathspec would ship,
so the specsolve half can be built and measured against a real `Program`:

- ``program.Apply``: ``function(operand)`` for the functions in
  [`FUNCTIONS`][]. It subclasses ``program.Power``, so every mathspec walk
  that recurses through a power (children, dims, signs, exclusivity) recurses
  through it with no further patch. Its ``exponent`` is a dummy constant
  nothing reads.
- The resolver reads ``exp(x)`` and the others as an ``Apply``.
- ``degree.check_binary`` admits a variable in a divisor, under a power, and in
  a product above degree 2. A bound stays affine (ceiling 1), and the rule
  against a product of two sums of terms stays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mathspec import _expression_resolver as resolver
from mathspec import degree, program
from mathspec.errors import LanguageError

#: The functions a file may call on an expression.
FUNCTIONS = ('exp', 'log', 'sqrt', 'sin', 'cos')


@dataclass(frozen=True)
class Apply(program.Power):
    """``function(operand)``, held as a power so the mathspec walks need no patch."""

    function: str = field(default='exp', kw_only=True)

    @property
    def operand(self) -> program.Expression:
        """The expression the function is applied to."""
        return self.base


def _call(self: Any, node: Any) -> program.Expression | None:  # pyrefly: ignore[explicit-any] — a patched method
    if node.name not in FUNCTIONS:
        return _resolver_call(self, node)
    if len(node.args) != 1 or node.kwargs:
        self.errors.append(f'{self.context}: {node.name}() expects {node.name}(<expr>)')
        return None
    operand = self.arith(node.args[0])
    return None if operand is None else Apply(operand, program.Constant(0.0), function=node.name)


def _check_binary(node: Any, context: str, *, ceiling: int) -> None:  # pyrefly: ignore[explicit-any] — a patched function
    nonlinear = isinstance(node, program.Divide | program.Power) or (
        isinstance(node, program.Multiply) and degree._degree(node) > 2
    )
    if not nonlinear:
        _degree_check_binary(node, context, ceiling=ceiling)
        return
    if ceiling < 2 and program.carries_variable(node):
        raise LanguageError(f'{context}: a nonlinear term may stand only in the objective or a constraint.')


_resolver_call = resolver.ExpressionResolver._call
_degree_check_binary = degree.check_binary
resolver.ExpressionResolver._call = _call
degree.check_binary = _check_binary
program.Apply = Apply  # pyrefly: ignore[missing-attribute]
