"""The run half of the exception hierarchy, and the whole of it re-exported.

The spec half — `LanguageError` and what derives from it, decidable at
load time with no data attached — belongs to ``mathspec`` and is re-exported here,
so one ``except`` clause covers the package. The run half is defined here:
[`DataError`][] is a fine file with the wrong thing attached to it, and
[`NoSolutionError`][] a solve with nothing to read back.

A message lives here only where the engine and the test oracle both raise it.
One raiser keeps its message beside itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mathspec import (
    DimensionError,
    LanguageError,
    MathSpecError,
    SchemaError,
    did_you_mean,
)

#: The root, under the name callers catch it by. An alias and not a subclass:
#: `except sps.SpecsolveError` has to catch a `LanguageError`.
SpecsolveError = MathSpecError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class SpecsolveWarning(UserWarning):
    """Advice from ``check``: the spec loads and solves, and reads wrong.

    Raised for a spec that is still part-written, where an expression has not
    yet reached what it declares.
    """


class DataError(SpecsolveError):
    """Data attached to a valid spec is missing or the wrong shape."""


class LayoutError(SpecsolveError):
    """What is on disk is not a layout this package reads.

    The target is a directory or archive that ``save`` wrote, or did not. The
    fix is which path was named, or re-solving a model whose layout has moved
    since it was written. The layout is the one
    [`save`][specsolve.relational.result.Result.save] stamps.
    """


class NoSolutionError(SpecsolveError):
    """The solve returned no values to read — infeasible, unbounded, errored.

    A scenario sweep catches this and records the outcome; a
    `LanguageError` instead means the file needs editing.
    """


__all__ = [
    'DataError',
    'DimensionError',
    'LanguageError',
    'NoSolutionError',
    'SchemaError',
    'SpecsolveError',
    'SpecsolveWarning',
    'did_you_mean',
]


def uncovered_constant_message(names: str, missing: int, subject: str) -> str:
    """The message for a constant side covering fewer coordinates than its rows."""
    return (
        f"{subject}: parameter '{names}' covers {missing} fewer coordinates than the rows "
        f'built here. A missing row is read as 0, and on the constant side that zero is a '
        f'bound rather than an absence — the row still exists, and it binds.\n'
        f'  Supply the missing rows, if the value is what was meant.\n'
        f'  Mask them out with a where, if the row should not exist there.\n'
        f'  Drop the declaration, if the spec has no such quantity at all.'
    )


def sparse_divisor_message(name: str, missing: int) -> str:
    """The message for a divisor parameter that is sparse over its index."""
    return (
        f"parameter '{name}' is used as a divisor but covers {missing} fewer "
        f'coordinates than it is indexed over. A missing row means a zero '
        f'coefficient everywhere else, and zero is not a divisor: the term '
        f'would drop and the constraint would silently stop constraining.\n'
        f'  Supply the missing rows, or mask the coordinates out with a where.'
    )


def reported_divisor_message(name: str, missing: int) -> str:
    """The message for a divisor parameter a reported expression reads short of a row."""
    return (
        f"parameter '{name}' is used as a divisor but has no row at {missing} of the coordinates "
        f'the expression divides at. A missing parameter row is not absence, so the quotient '
        f'is not dropped there, and there is no number to divide by.\n'
        f'  Supply the missing rows.\n'
        f'  Give the value 0 at a coordinate the quotient should skip: a quotient by zero has no value.'
    )


def null_bounds_message(name: str, rows: int) -> str:
    """The message for a bound parameter missing values at some coordinates."""
    return (
        f"variable '{name}': {rows} rows have NULL bounds — a bound parameter is missing "
        f'values for some coordinates. The two ways out build different models, so the '
        f'language will not pick one:\n'
        f'  supply the value           the variable exists there, bounded (`inf` is a value)\n'
        f'  where: "<the parameter>"   the variable does not exist there at all'
    )


def carried_parameter_message(carried: list[str]) -> str:
    """An expression to evaluate across a sweep that reads a carried parameter."""
    return (
        f'this expression reads {carried}, which the sweep carried from one slice into the next, and a '
        f"carried value is a previous slice's answer rather than stored data — so it cannot be put back "
        f'per slice from the archive. Re-run the sweep with sps.solve_over(spec, sources, axis, carry=...) '
        f'and evaluate on what comes back, or read a quantity over the sweep that reads no carried parameter.'
    )


def no_model_behind_this_answer_message() -> str:
    """An expression to read in an answer that has no model behind it."""
    return (
        'this answer has no model behind it, so a quantity the file never named cannot be read from '
        'it: an answer read back off disk carries the values without the model to splice the '
        'expression into. Re-ask with sps.solve(archive.spec, archive.sources), which reads any '
        'expression; a name the spec declares is readable either way.'
    )


def another_model_behind_this_answer_message(answered: str, rebuilt: str) -> str:
    """A saved answer read against a model its sources do not rebuild."""
    return (
        f'this answer came back from another model: it answered the model digesting to {answered} '
        f'and the spec and sources beside it build {rebuilt}. The document matched, so what differs '
        f'is the data — and reading a quantity the file never named against other numbers would '
        f'value it at an answer nobody solved for.\n'
        f'  Read the answer against the data the solve ran on. An archive holds that pair, so one '
        f'refused here has had a source replaced since it was written.'
    )


def position_out_of_range_message(name: str, op: str, position: int, at: int, cardinality: int) -> str:
    """A ``position(dim)`` boundary naming no coordinate of the dimension."""
    return (
        f'where: position({name}) {op} {position} names position {at} of '
        f"'{name}', which has {cardinality} coordinate(s). A boundary that "
        f'names no coordinate leaves the rows it was to seed unseeded.'
    )


def short_groups_message(name: str, by: str, op: str, position: int, short: Sequence[str]) -> str:
    """A grouped ``position(dim, by=)`` boundary that some group is too short to reach."""
    return (
        f'where: position({name}, by={by}) {op} {position} names position '
        f'{position} within each group, and {len(short)} of them are shorter than '
        f'that: {list(short[:5])}. A boundary that names no coordinate leaves the rows it '
        f'was to seed unseeded.'
    )


def unknown_name_message(kind: str, name: str, known: Iterable[str]) -> str:
    r"""``unknown <kind> '<name>'``, plus the near miss or the declared set.

    A name that is the prefix of one or more declarations lists the whole
    family instead of a near miss. The message stays single-line: it is raised
    as ``KeyError``, whose ``str`` reprs a newline as a literal ``\n``.
    """
    candidates = sorted(known)

    family = [c for c in candidates if c.startswith(f'{name}_')]
    if family:
        return (
            f"unknown {kind} '{name}': no declaration has that name, but "
            f'{len(family)} begin with it — {", ".join(family)}.'
        )

    return f"unknown {kind} '{name}'. {did_you_mean(name, candidates)}"
