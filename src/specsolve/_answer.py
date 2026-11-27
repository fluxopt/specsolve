"""One solve's answer as plain data: how it ended, what it took, its tables, and why a table is not there.

A [`Result`][specsolve._result.Result] reads one, a sweep keeps one per slice,
and an archive's ``answer/`` holds one
([`write_answer`][specsolve._answer_layout.write_answer]). The tables are keyed
by the [`KINDS`][specsolve._outputs.KINDS] and then by name, and so are the
reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from specsolve._collect import collected
from specsolve._outputs import PRICED
from specsolve._status import SolveStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

    from specsolve._inputs import Label
    from specsolve._records import Metrics, Record

#: ``{kind: {name: table}}``: one ``(dims…, value)`` table per name, under each
#: of the [`KINDS`][specsolve._outputs.KINDS] an answer carries. A kind it
#: carries and has nothing of maps to nothing.
type Tables = Mapping[str, Mapping[str, pl.LazyFrame]]

#: ``{kind: {name: reason}}``: why a table is not there. [`WHOLE`][] names the
#: reason no table of the kind is.
type Reasons = Mapping[str, Mapping[str, str]]

#: The name a reason is kept under where it is the whole kind's, as the duals' is.
WHOLE = ''


def whole_kind_reason(reasons: Reasons, kind: str) -> str | None:
    """Why *reasons* say no table of *kind* is there: the duals' reason for a kind that exists only where they do."""
    return reasons.get('dual' if kind in PRICED else kind, {}).get(WHOLE)


def named_reasons(reasons: Reasons, kind: str) -> dict[str, str]:
    """Why each name of *kind* that *reasons* name is not there, by name."""
    return {name: why for name, why in reasons.get(kind, {}).items() if name != WHOLE}


def merged_reasons(reasons: Reasons, later: Reasons) -> dict[str, dict[str, str]]:
    """*reasons* with each of *later* it does not hold: the first reason given for a kind and a name is kept."""
    merged = {kind: dict(named) for kind, named in reasons.items()}
    for kind, named in later.items():
        for name, why in named.items():
            merged.setdefault(kind, {}).setdefault(name, why)
    return merged


@dataclass(frozen=True)
class Answer:
    """What one solve answered: its record, its metrics, its tables and its reasons."""

    record: Record
    #: What reaching this answer took, or ``None`` for one no solve produced.
    metrics: Metrics | None = None
    tables: Tables = field(default_factory=dict)
    reasons: Reasons = field(default_factory=dict)
    #: What the solver called the outcome, which the record does not hold.
    wording: str = ''

    @property
    def status(self) -> SolveStatus:
        """How the solve ended, with the solver's own wording where it is known."""
        return SolveStatus(self.record.termination_condition, self.wording, self.record.has_primal)

    @property
    def objective(self) -> float:
        """The objective value, or ``nan`` where the solve reached none."""
        return float('nan') if self.record.objective is None else self.record.objective

    def sliced(self, key_name: str, key: Label) -> Answer:
        """This answer with its record and metrics rows naming the sweep slice they are."""
        text = str(key)
        return replace(
            self,
            record=self.record._replace(slice_axis=key_name, slice=text),
            metrics=None if self.metrics is None else self.metrics._replace(slice_axis=key_name, slice=text),
        )

    def in_memory(self) -> Answer:
        """This answer with every table read now, so it holds no build and can outlive the model that answered."""
        tables = {
            kind: {name: table.pipe(collected).lazy() for name, table in named.items()}
            for kind, named in self.tables.items()
        }
        return replace(self, tables=tables)
