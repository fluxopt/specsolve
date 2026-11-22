"""The column names specsolve writes beside a spec's own, and the refusals that keep a spec from taking them.

There are two rules, and they differ in shape and in reach. Every declared
name is refused where it starts with [`RESERVED_PREFIX`][], which covers every
column specsolve adds under it. A dimension alone is refused where it is
[`VALUE`][], the one whole name, because only a dimension becomes a column
beside it: a parameter or a variable called ``value`` is accepted.
[`as_program`][specsolve._inputs.as_program] applies both, so
[`check`][specsolve._api.check] and every other verb name the collision before
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
#: table came from. Read back, a table comes without it.
RUN = f'{RESERVED_PREFIX}run'

#: The column a dimension's table numbers its labels in, from 0 in label order.
POSITION = f'{RESERVED_PREFIX}position'

#: The column a dimension's table holds its labels in, beside [`POSITION`][].
LABEL = f'{RESERVED_PREFIX}label'

#: The column holding a constraint's row index, in its table and in the matrix a sink reads.
ROW = f'{RESERVED_PREFIX}row'

#: The column holding a variable's column index, in its table and in the matrix a sink reads.
COL = f'{RESERVED_PREFIX}col'

#: The second variable's column index of a quadratic term.
COL_2 = f'{RESERVED_PREFIX}col_2'

#: The smaller and the larger column index of a quadratic matrix entry.
COL_L = f'{RESERVED_PREFIX}col_l'
COL_R = f'{RESERVED_PREFIX}col_r'

#: The column holding a term's coefficient.
COEFF = f'{RESERVED_PREFIX}coeff'

#: The column holding a constant term's value at a coordinate.
CONSTANT = f'{RESERVED_PREFIX}constant'


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


#: The column that holds the numbers beside an entry's dimensions, in a
#: parameter's table and in every table an answer comes back as. Besides the
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
            f"in a parameter's table and in every table an answer comes back as. Query engines read column "
            f'names without case, so the two columns would collide in any letter case. Rename it.'
        )
