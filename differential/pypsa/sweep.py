# SPDX-FileCopyrightText: mathspec Contributors
#
# SPDX-License-Identifier: MIT

"""The coverage question the block-level one cannot reach, as a rule with no data in it.

Stamps and a program in, sentences out, so it is testable without pypsa.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def untested_conjuncts(name: str, program: Any, stamps: list[Mapping[str, Any]]) -> list[str]:
    """Every conjunct of every mask true somewhere on the ladder and false somewhere.

    A mask is exercised as a whole the moment one of its conjuncts varies, so
    `committable AND p_nom_mod > 0` passes block-level coverage while
    `committable` is true everywhere (mathspec#312).

    Each rung records one character per conjunct — ``t`` held everywhere, ``f``
    nowhere, ``b`` at some coordinates, ``-`` no frame at all. A conjunct is
    exercised when the ladder has it true somewhere (``t`` or ``b``) and false
    somewhere (``f`` or ``b``), which one ``b`` satisfies alone and two rungs
    disagreeing satisfy between them.

    A conjunct of a block no rung builds is not reported; the block is already
    a gap of its own.
    """
    gaps = []
    for block_name, block in {**program.constraints, **program.variables}.items():
        if getattr(block, 'where', None) is None:
            continue
        seen = [stamp['conjuncts'][block_name] for stamp in stamps if block_name in stamp.get('conjuncts', {})]
        for position, conjunct in enumerate(block.where.conjuncts):
            marks = {verdicts[position] for verdicts in seen if position < len(verdicts)} - {'-'}
            if not marks:
                continue
            if not marks & {'t', 'b'}:
                gaps.append(f'{name}: {block_name} is never true at conjunct {position} — {conjunct!r}')
            elif not marks & {'f', 'b'}:
                gaps.append(f'{name}: {block_name} is never false at conjunct {position} — {conjunct!r}')
    return gaps
