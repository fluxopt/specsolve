"""What specsolve raises and warns: one tree, rooted at `SpecsolveError`.

The spec half, `LanguageError` and what derives from it, is the language's own,
re-exported from ``mathspec`` so that one ``except SpecsolveError`` covers both
packages.

Example::

    import specsolve as sps

    try:
        result = sps.solve('spec.yaml', sources)
    except sps.errors.NoSolutionError:
        ...
"""

from mathspec import DimensionError, LanguageError, SchemaError

# An alias, not a subclass, so ``except SpecsolveError`` catches a ``LanguageError``.
from mathspec import MathSpecError as SpecsolveError


class SpecsolveWarning(UserWarning):
    """Advice from ``check``: the spec loads and solves, and reads wrong.

    Raised for a spec that is still part-written, where an expression has not
    yet reached what it declares.
    """


class DataError(SpecsolveError):
    """Data attached to a valid spec is missing or the wrong shape."""


class LayoutError(SpecsolveError):
    """What is on disk is not a layout this package reads.

    The fix is which path was named, or re-solving a model whose layout has
    moved since it was written. The layout is the one
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
    'LayoutError',
    'NoSolutionError',
    'SchemaError',
    'SpecsolveError',
    'SpecsolveWarning',
]
