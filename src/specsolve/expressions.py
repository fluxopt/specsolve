"""Expressions the file never named, valued against a spec the language has already read.

An expression is spliced into the spec as a named one and the whole spec is
lowered again, so it passes every rule a declared one passes. This sits above
both lanes because nothing under ``relational/`` may see the spec as written
(docs/about/architecture.md, hard rule 2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from specsolve.lanes import lowered

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mathspec import Spec
    from mathspec.program import Expression

#: The name an unnamed expression is spliced under, stepped over where the
#: spec declares it.
_EVALUATED = '_evaluated'

#: The one section a caller may hand in.
_SECTION = 'expressions'


def lower(spec: Spec, expression: str | Mapping[str, object]) -> Expression:
    """One unnamed expression as a plan node, read in *spec*'s namespace.

    Args:
        spec: The spec the expression is written against. It supplies every
            name the expression may use; one it does not declare is refused.
        expression: What one ``expressions:`` entry takes — a string, or the
            mapping carrying ``cases:`` with ``dims:`` and ``otherwise:``.

    Returns:
        The node a declared named expression of *spec* lowers to.

    Raises:
        LanguageError: A construct outside the language, or a name *spec* does
            not declare.
        SchemaError: Something that is not an expression.
    """
    written = spec.to_dict()
    name = _free_name(written)
    return _splice(written, {name: expression})[name]


def _splice(written: dict[str, object], entries: Mapping[str, object]) -> dict[str, Expression]:
    """*entries* added to the spec *written* and lowered with it, as nodes."""
    section = written.get(_SECTION)
    written[_SECTION] = {**section, **entries} if isinstance(section, dict) else dict(entries)
    named = lowered(written).expressions
    return {name: named[name].expression for name in entries}


def _free_name(written: Mapping[str, object]) -> str:
    """A name no declaration in *written* holds — where an unnamed expression is spliced."""
    taken = _declared(written)
    name = _EVALUATED
    while name in taken:
        name += '_'
    return name


def _declared(written: Mapping[str, object]) -> set[str]:
    """Every name any section of *written* declares, so a section added later is covered."""
    return {name for section in written.values() if isinstance(section, dict) for name in section}
