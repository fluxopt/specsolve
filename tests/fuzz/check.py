"""What one generated case is checked against, and what a failure looks like.

Each check is an oracle that needs no expected answer:

- **load**: the language accepts the file or refuses it with a
  ``MathSpecError``. Any other exception is a crash.
- **round trip**: the YAML a spec writes reads back as the same spec, and the
  canonical form is a fixed point.
- **typeset**: every format prints an accepted spec, written out and as stated.
- **build**: the two lanes accept the same data or refuse it with the same
  sentence. An oracle gap (``OracleCannotBuildError``) is not a finding.
- **read**: every expression the case names reads the same on both lanes at one
  point (:func:`tests.differential.at_a_point`'s claim, without its asserts).
- **solve**: for a linear case with an objective (linopy solves no feasibility problem), both lanes reach the same status and objective,
  build the same coefficients row for row, and the written LP file re-solves to
  that objective.
- **shuffle**: the rows of a table carry no order, so shuffling every
  parameter and relation table moves no read and no objective on the relational
  lane.

:func:`check` returns the findings rather than raising, so a campaign can count
and group them.
"""

from __future__ import annotations

import math
import random
import re
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from mathspec import advice, to_spec, typeset
from mathspec.errors import MathSpecError

import specsolve as sps
from specsolve._engine.engine import Engine
from specsolve._inputs import lowered
from specsolve._sources import tidy_sources
from specsolve.errors import DataError, SpecsolveError
from tests.differential import RTOL, _frame, _hold, by_coordinate
from tests.linopy_lane.loader import OracleCannotBuildError
from tests.oracle import specsolve_linopy

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mathspec.spec import Spec

    from tests.fuzz.generate import Case

#: A value the two lanes read apart by less than this is the same value.
ATOL = 1e-9


@dataclass(frozen=True)
class Finding:
    """One thing a case broke.

    Attributes:
        oracle: Which check found it.
        summary: One line, the same for every case that broke it the same way.
        detail: What to read to reproduce it: the exception or the two values.
    """

    oracle: str
    summary: str
    detail: str

    @property
    def signature(self) -> str:
        """What groups findings across seeds: the oracle and the summary with names and numbers blanked."""
        blanked = re.sub(r"'[^']*'|`[^`]*`|-?\d+(\.\d+)?(e-?\d+)?", '_', self.summary)
        return f'{self.oracle}: {blanked}'


@dataclass
class Outcome:
    """What :func:`check` concluded about one case."""

    status: str
    """``rejected`` (the language refused it), ``refused`` (both lanes refused the data), or ``checked``."""
    findings: list[Finding]
    solved: bool = False


def check(case: Case, *, solve: bool = True) -> Outcome:
    """Run every oracle on *case* and return what broke."""
    findings: list[Finding] = []
    try:
        spec = to_spec(case.spec)
    except MathSpecError:
        return Outcome('rejected', findings)
    except Exception as e:
        return Outcome('rejected', [_crash('load', e)])

    findings += _guarded('round-trip', lambda: _round_trip(spec))
    findings += _guarded('typeset', lambda: _typesets(spec))

    try:
        expanded = spec.expand()
    except Exception as e:
        return Outcome('checked', [*findings, _crash('expand', e)])

    built, refusal = _both_builds(expanded, case.sources)
    if refusal is not None:
        findings += refusal
    if built is None:
        return Outcome('refused' if refusal == [] else 'checked', findings)
    model, relational = built
    with relational:
        findings += _guarded('read', lambda: _reads(expanded, case, model, relational))
    solved = False
    if solve and case.linear and expanded.objective is not None:
        try:
            solve_findings, solved = _solve(expanded, case.sources)
        except Exception as e:
            solve_findings = [] if _is_gap(e) else [_crash('solve', e)]
        findings += solve_findings
        findings += _guarded('shuffle', lambda: _shuffled(expanded, case))
    return Outcome('checked', findings, solved=solved)


def _guarded(oracle: str, run: Callable[[], list[Finding]]) -> list[Finding]:
    """*run*'s findings, or the crash it raised: one oracle that raises does not hide the others' findings.

    An oracle gap is no finding here either, as everywhere else in this module.
    """
    try:
        return run()
    except Exception as e:
        return [] if _is_gap(e) else [_crash(oracle, e)]


# ---------------------------------------------------------------------------
# the language alone
# ---------------------------------------------------------------------------


def _round_trip(spec: Spec) -> list[Finding]:
    canonical = spec.to_yaml(canonical=True)
    findings = []
    for name, text in (('to_yaml', spec.to_yaml()), ('canonical', canonical)):
        try:
            again = to_spec(text).to_yaml(canonical=True)
        except Exception as e:
            findings.append(Finding('round-trip', f'the {name} form does not load back: {_first(e)}', _trace(e)))
            continue
        if again != canonical:
            findings.append(
                Finding(
                    'round-trip',
                    f'the {name} form loads back as a different spec',
                    f'--- before\n{canonical}\n--- after\n{again}',
                )
            )
    return findings


def _typesets(spec: Spec) -> list[Finding]:
    findings = []
    for form, stated in (('as written', spec), ('expanded', None)):
        for fmt in ('latex', 'typst', 'markdown'):
            try:
                typeset(stated if stated is not None else spec.expand(), fmt)
            except Exception as e:
                findings.append(_crash(f'typeset-{fmt}', e, f' ({form})'))
    try:
        advice(spec)
    except Exception as e:
        findings.append(_crash('advice', e))
    return findings


# ---------------------------------------------------------------------------
# both lanes
# ---------------------------------------------------------------------------


def _both_builds(spec: Spec, sources: Mapping[str, Any]) -> tuple[tuple[Any, Any] | None, list[Finding] | None]:
    """Build both lanes; return them, or why not.

    The second element is None where nothing is to say, ``[]`` where both lanes
    refused the data alike, and the findings otherwise.
    """
    model, theirs = _attempt(lambda: specsolve_linopy.build(spec, dict(sources)))
    relational, ours = _attempt(lambda: sps.build(spec, dict(sources)))
    if theirs is None and ours is None:
        return (model, relational), None
    if relational is not None:
        relational.close()
    if _is_gap(theirs) or _is_gap(ours):
        return None, None
    findings = [_crash(lane, e) for lane, e in (('linopy', theirs), ('relational', ours)) if _is_crash(e)]
    if findings:
        return None, findings
    if theirs is not None and ours is not None:
        if str(theirs) != str(ours):
            return None, [
                Finding(
                    'wording', 'the lanes refuse the data in different words', f'relational: {ours}\nlinopy: {theirs}'
                )
            ]
        return None, []
    lane, e = ('relational', ours) if ours is not None else ('linopy', theirs)
    return None, [Finding('build', f'only the {lane} lane refuses: {_first(e)}', _trace(e))]


def _reads(spec: Spec, case: Case, model: Any, relational: Any) -> list[Finding]:
    point = _hold(model)
    read, e = _attempt(lambda: relational._evaluator(point, None, 'a chosen point has no duals'))
    if e is not None:
        return [Finding('build', f'the lanes build a variable over different coordinates: {_first(e)}', _trace(e))]
    findings = []
    for expression in case.reads:
        ours, mine = _attempt(lambda e=expression: by_coordinate(read(e)))
        theirs, oracle = _attempt(
            lambda e=expression: by_coordinate(_frame(specsolve_linopy.evaluate(model, spec, e, dict(case.sources))))
        )
        if _is_gap(oracle) or _is_gap(mine):
            continue
        crashes = [
            _crash(f'read-{lane}', e, f' reading `{expression}`')
            for lane, e in (('linopy', oracle), ('relational', mine))
            if _is_crash(e)
        ]
        if crashes:
            findings += crashes
            continue
        if mine is not None or oracle is not None:
            if mine is None or oracle is None or str(mine) != str(oracle):
                lane = (
                    'both lanes'
                    if mine is not None and oracle is not None
                    else ('the relational lane' if mine is not None else 'the linopy lane')
                )
                findings.append(
                    Finding(
                        'read',
                        f'{lane} refuse(s) a read the other answers or words it differently: {_first(mine or oracle)}',
                        f'expression: {expression}\nrelational: {mine}\nlinopy: {oracle}',
                    )
                )
            continue
        difference = _difference(ours, theirs)
        if difference:
            findings.append(
                Finding(
                    'read',
                    'the lanes read an expression differently at one point',
                    f'expression: {expression}\npoint: { {k: by_coordinate(v) for k, v in point.items()} }\n{difference}',
                )
            )
    return findings


def _difference(
    ours: dict[tuple[str, ...], float],
    theirs: dict[tuple[str, ...], float],
    names: tuple[str, str] = ('relational', 'linopy'),
) -> str:
    lines = []
    for coordinate in sorted(ours.keys() | theirs.keys()):
        a, b = ours.get(coordinate), theirs.get(coordinate)
        if a is None or b is None or not _close(a, b):
            lines.append(f'  at {coordinate}: {names[0]} {a}, {names[1]} {b}')
    return '\n'.join(lines)


def _close(a: float, b: float) -> bool:
    if math.isinf(a) or math.isinf(b):
        return a == b
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)


def _solve(spec: Spec, sources: Mapping[str, Any]) -> tuple[list[Finding], bool]:
    m = specsolve_linopy.build(spec, dict(sources))
    status, condition = m.solve(solver_name='highs', output_flag=False)
    oracle = float(m.objective.value) if status == 'ok' else math.nan
    program = lowered(spec)
    findings = []
    with Engine() as engine:
        engine.build(program, tidy_sources(program, dict(sources)))
        result = engine.solve(solver_options={'output_flag': False})
        diagnostics = engine.diagnostics()
        if (diagnostics.columns, diagnostics.rows) != (m.nvars, m.ncons):
            findings.append(
                Finding(
                    'solve',
                    'the lanes build a different number of columns or rows',
                    f'relational {diagnostics.columns} columns, {diagnostics.rows} rows; linopy {m.nvars}, {m.ncons}',
                )
            )
        else:
            findings += _same_matrix(engine, m)
        if result.termination_condition != condition:
            if {result.termination_condition, condition} <= {'infeasible', 'infeasible_or_unbounded', 'unbounded'}:
                return findings, False
            findings.append(
                Finding(
                    'solve',
                    'the lanes end the solve differently',
                    f'relational {result.termination_condition}, linopy {condition}',
                )
            )
            return findings, False
        if not result.is_ok:
            return findings, False
        if not _close(result.objective, oracle):
            findings.append(
                Finding(
                    'solve', 'the lanes reach different objectives', f'relational {result.objective}, linopy {oracle}'
                )
            )
        if all(v.domain == 'continuous' for v in spec.variables.values()):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'model.lp'
                engine.write(path)
                written = _resolve(path)
            if written is None or not _close(written, oracle):
                findings.append(
                    Finding(
                        'solve',
                        'the written LP file re-solves to a different objective',
                        f'LP file {written}, both lanes {oracle}',
                    )
                )
    return findings, True


def _resolve(path: Path) -> float | None:
    import highspy

    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.readModel(str(path))
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None
    return h.getInfo().objective_function_value


def _same_matrix(engine: Engine, m: Any) -> list[Finding]:
    """Each constraint's rows as a sorted multiset of rows, each row its sorted coefficients up to sign.

    The lanes number rows and columns apart, and may move every term of a row
    to the other side, which negates the row and flips its sense: the objective
    check sees a sense that is wrong, this one the coefficients.
    """
    tables = engine._model.handoff
    findings = []
    for constraint, block in engine._model.constraints.items():
        got = _canonical(tables.matrix_block(block.start, block.start + block.height), block.height)
        want = _linopy_matrix(m, constraint)
        if len(got) != len(want) or any(
            len(a) != len(b) or not all(_close(x, y) for x, y in zip(a, b, strict=True))
            for a, b in zip(got, want, strict=False)
        ):
            findings.append(
                Finding(
                    'solve', 'the lanes write different coefficients', f'{constraint}:\nrelational {got}\nlinopy {want}'
                )
            )
    return findings


def _up_to_sign(row: list[float]) -> tuple[float, ...]:
    kept = [v for v in row if round(v, 12) != 0]
    plus, minus = tuple(sorted(kept)), tuple(sorted(-v for v in kept))
    return min(plus, minus, key=lambda r: tuple(round(v, 9) for v in r))


def _canonical(matrix: pl.DataFrame, height: int) -> list[tuple[float, ...]]:
    rows = matrix.group_by('row').agg(pl.col('coeff')).get_column('coeff').to_list()
    return sorted([_up_to_sign(r) for r in rows] + [()] * (height - len(rows)))


def _linopy_matrix(m: Any, constraint: str) -> list[tuple[float, ...]]:
    """The same off linopy's arrays, less the rows with no variable term, which the language does not build."""
    if constraint not in m.constraints:
        return []
    c = m.constraints[constraint]
    labels = np.asarray(c.labels).reshape(-1)
    variables = np.asarray(c.vars).reshape(len(labels), -1)
    coefficients = np.asarray(c.coeffs).reshape(len(labels), -1)
    rows = []
    for i, label in enumerate(labels):
        if label < 0 or not (variables[i] >= 0).any():
            continue
        collapsed: dict[int, float] = {}
        for column, coefficient in zip(variables[i], coefficients[i], strict=True):
            if column >= 0:
                collapsed[int(column)] = collapsed.get(int(column), 0.0) + float(coefficient)
        rows.append(_up_to_sign(list(collapsed.values())))
    return sorted(rows)


def _shuffled(spec: Spec, case: Case) -> list[Finding]:
    """The relational lane, its tables in another row order, reads and solves the same."""
    rng = random.Random(case.seed)
    shuffled = {
        name: source[rng.sample(range(source.height), source.height)]
        if isinstance(source, pl.DataFrame) and name not in case.spec['dimensions']
        else source
        for name, source in case.sources.items()
    }
    m = specsolve_linopy.build(spec, dict(case.sources))
    point = _hold(m)
    findings = []
    with sps.build(spec, dict(case.sources)) as before, sps.build(spec, shuffled) as after:
        try:
            first = before._evaluator(point, None, 'no duals')
            second = after._evaluator(point, None, 'no duals')
        except MathSpecError:
            return findings
        for expression in case.reads:
            a, e1 = _attempt(lambda e=expression: by_coordinate(first(e)))
            b, e2 = _attempt(lambda e=expression: by_coordinate(second(e)))
            if (e1 is None) != (e2 is None):
                findings.append(Finding('shuffle', 'a read refuses only in one row order', f'{expression}\n{e1}\n{e2}'))
            elif e1 is None and (difference := _difference(a, b, ('in data order', 'shuffled'))):
                findings.append(
                    Finding(
                        'shuffle',
                        'a read moves when the table rows are shuffled',
                        f'expression: {expression}\n{difference}',
                    )
                )
        one, two = (
            before.solve(solver_options={'output_flag': False}),
            after.solve(solver_options={'output_flag': False}),
        )
        if one.termination_condition != two.termination_condition or (
            one.has_primal and not _close(one.objective, two.objective)
        ):
            findings.append(
                Finding(
                    'shuffle',
                    'the solve moves when the table rows are shuffled',
                    f'{one.termination_condition} {one.objective} vs {two.termination_condition} {two.objective}',
                )
            )
    return findings


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _attempt(call: Callable[[], Any]) -> tuple[Any, BaseException | None]:
    try:
        return call(), None
    except Exception as e:
        return None, e


def _is_gap(e: BaseException | None) -> bool:
    """A refusal one lane makes by design: the oracle's ``OracleCannotBuildError``, or the relational lane's #1137."""
    return isinstance(e, OracleCannotBuildError) or (
        isinstance(e, SpecsolveError) and not isinstance(e, DataError) and 'specsolve cannot build that' in str(e)
    )


def _is_crash(e: BaseException | None) -> bool:
    """An exception that is neither the language's refusal nor a lane's."""
    return e is not None and not isinstance(e, (MathSpecError, SpecsolveError))


def _crash(where: str, e: BaseException, context: str = '') -> Finding:
    return Finding(f'crash-{where}', f'{type(e).__name__}{context}: {_first(e)}', _trace(e))


def _first(e: BaseException | None) -> str:
    return str(e).strip().splitlines()[0][:200] if e is not None and str(e).strip() else type(e).__name__


def _trace(e: BaseException) -> str:
    return ''.join(traceback.format_exception(e))
