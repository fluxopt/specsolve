"""The assertion-message rule, checked rather than reviewed.

Only assertions whose claim is not visible in the expression need a message:

- **a literal collection** — ``== ['high', 'low', 'mid']`` claims an order and a
  completeness, and neither is on the line;
- **a count** — ``len(rows) == 4`` never says why four;
- **a tolerance** — ``approx(x, rel=1e-9)`` is a chosen precision;
- **an absence** — ``assert not offenders`` says what is empty, never why it
  must be.

The count in breach is a ratchet that falls as files are touched.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent

#: Assertions in breach. A ratchet: lower it in the PR that lowers the count.
IN_BREACH = 201


def _unwritten_claim(node: ast.Assert) -> str | None:
    """Why this assertion's claim is not visible in its expression, if it is not."""
    test = node.test
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return 'an absence'
    if not isinstance(test, ast.Compare):
        return None
    for side in (test.left, *test.comparators):
        if isinstance(side, (ast.List, ast.Set, ast.Tuple)) and side.elts:
            return 'a literal collection'
        if isinstance(side, ast.Dict) and side.keys:
            return 'a literal collection'
        if isinstance(side, ast.Call):
            if isinstance(side.func, ast.Name) and side.func.id == 'len':
                return 'a count'
            if 'approx' in ast.dump(side.func):
                return 'a tolerance'
    return None


def _in_breach() -> list[str]:
    """``path:line — why`` for every assertion the narrowed rule wants a message on."""
    found = []
    for path in sorted(TESTS.rglob('*.py')):
        tree = ast.walk(ast.parse(path.read_text()))
        found += [
            f'{path.relative_to(TESTS.parent)}:{node.lineno} — {why}'
            for node in tree
            if isinstance(node, ast.Assert) and node.msg is None and (why := _unwritten_claim(node))
        ]
    return found


def test_an_assertion_whose_claim_is_not_on_the_line_carries_a_message() -> None:
    """The ratchet holds in both directions: above it a claim went unwritten, below it the ceiling was not lowered."""
    breach = _in_breach()
    assert len(breach) <= IN_BREACH, (
        f'{len(breach)} assertions state a claim their expression does not carry, above the {IN_BREACH} '
        f'this ratchet was set at — put the claim in the message, since a message is what prints. '
        f'In breach, in full:\n  ' + '\n  '.join(breach)
    )
    assert len(breach) == IN_BREACH, (
        f'{len(breach)} assertions are in breach but IN_BREACH says {IN_BREACH} — set it to {len(breach)} '
        f'in this PR, so the headroom the check leaves is the headroom the tree actually has'
    )
