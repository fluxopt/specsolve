"""The column names specsolve writes beside a spec's own, and the refusals that keep a spec from taking them.

There are two rules, and they differ in shape and in reach. Every declared
name is refused where it starts with [`RESERVED_PREFIX`][], which covers every
column specsolve adds under it. A dimension alone is refused where it is
[`VALUE`][], the one whole name, because only a dimension becomes a column
beside it: a parameter or a variable called ``value`` is accepted.
[`lowered`][specsolve.inputs.lowered] applies both, so
[`check`][specsolve.api.check] and every other verb name the collision before
any data is read.

Every module that names one of these columns imports it from here. It imports
nothing from the package but its errors, so every layer can import it.
"""

from __future__ import annotations

from specsolve.errors import SpecsolveError

#: The prefix reserved, in any letter case, for the columns specsolve adds, so
#: that no name a spec declares can collide with one.
RESERVED_PREFIX = 'specsolve_'


#: The column an archive adds to every table it holds, naming the run the
#: table came from. Read back, a frame comes without it.
RUN = f'{RESERVED_PREFIX}run'

#: The column a dimension's table numbers its labels in, from 0 in index order.
POSITION = f'{RESERVED_PREFIX}position'


def refuse_reserved(name: str, which: str) -> None:
    """Refuse *name*, which *which* describes, where it starts with [`RESERVED_PREFIX`][] in any letter case.

    Raises:
        SpecsolveError: A name a column specsolve adds could collide with.
    """
    if name.casefold().startswith(RESERVED_PREFIX):
        raise SpecsolveError(
            f'{which} starts with {RESERVED_PREFIX!r}, which is reserved in any letter case for the columns '
            f'specsolve adds, so it could collide with one. Rename it.'
        )


#: The column that holds the numbers beside a declaration's dimensions, in a
#: parameter's table and in every frame an answer comes back as. Besides the
#: names under [`RESERVED_PREFIX`][], it is the one name a dimension may not take.
VALUE = 'value'


def refuse_value(name: str, which: str) -> None:
    """Refuse *name*, which *which* describes, where it is [`VALUE`][] in any letter case.

    Raises:
        SpecsolveError: A name the ``value`` column would collide with.
    """
    if name.casefold() == VALUE:
        raise SpecsolveError(
            f'{which} has the name of the column {VALUE!r}, which holds the numbers beside the dimensions '
            f"in a parameter's table and in every frame an answer comes back as. Query engines read column "
            f'names without case, so the two columns would collide in any letter case. Rename it.'
        )
