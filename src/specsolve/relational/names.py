"""The column names specsolve writes beside a spec's own, and the refusals that keep a spec from taking them.

Every module that names one of these columns imports it from here, and
[`lowered`][specsolve.inputs.lowered] refuses a declaration that would collide
with one, so [`check`][specsolve.api.check] and every other verb name the
collision before any data is read. It imports nothing from the package but
its errors, so every layer can import it.
"""

from __future__ import annotations

from specsolve.errors import SpecsolveError

#: The prefix reserved, in any letter case, for the columns specsolve adds, so
#: that no name a spec declares can collide with one.
RESERVED = 'specsolve_'


#: The column an archive adds to every table it holds, naming the run the
#: table came from. Read back, a frame comes without it.
RUN = f'{RESERVED}run'

#: The column a dimension's table numbers its labels in, from 0 in index order.
POSITION = f'{RESERVED}position'


def refuse_reserved(name: str, which: str) -> None:
    """Refuse *name*, which *which* describes, where it starts with [`RESERVED`][] in any letter case.

    Raises:
        SpecsolveError: A name a column specsolve adds could collide with.
    """
    if name.casefold().startswith(RESERVED):
        raise SpecsolveError(
            f'{which} starts with {RESERVED!r}, which is reserved in any letter case for the columns '
            f'specsolve adds, so it could collide with one. Rename it.'
        )


#: The column that holds the numbers beside a declaration's dimensions, in a
#: parameter's table and in every frame an answer comes back as. Besides the
#: names under [`RESERVED`][], it is the one name a dimension may not take.
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
