"""Shrink a failing case to the smallest one that still fails the same way.

Greedy and one step at a time: drop an entry, a mask, a ``missing:``, a label
or a row, and keep the step if the case still produces a finding with the same
signature. The order runs from the steps that remove the most to those that
remove the least, and the passes repeat until none applies.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import polars as pl
from mathspec import to_spec
from mathspec.errors import MathSpecError

from tests.fuzz.check import check

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from tests.fuzz.generate import Case

SECTIONS = ('constraints', 'expressions', 'variables', 'parameters', 'relations')


def reduce(case: Case, signature: str, *, log: Callable[[str], None] = lambda _: None) -> Case:
    """The smallest case the steps reach that still has a finding of *signature*."""

    def fails(candidate: Case) -> bool:
        try:
            outcome = check(candidate)
        except Exception:
            return False
        return any(f.signature == signature for f in outcome.findings)

    assert fails(case), f'the case does not produce {signature!r} to begin with'
    changed = True
    while changed:
        changed = False
        for label, candidate in _steps(case):
            if fails(candidate):
                log(f'kept: {label}')
                case, changed = candidate, True
                break
    return case


def _steps(case: Case) -> Iterator[tuple[str, Case]]:
    spec, sources = case.spec, case.sources
    if len(case.reads) > 1:
        for read in case.reads:
            yield f'read only `{read}`', replace(case, reads=[read])
    if 'objective' in spec:
        yield 'drop the objective', _with(case, lambda s, _: s.pop('objective'))
    for section in SECTIONS:
        for name in list(spec.get(section, {})):
            yield f'drop {section}.{name}', _with(case, lambda s, d, sec=section, n=name: _drop(s, d, sec, n))
    for name in list(spec.get('dimensions', {})):
        yield f'drop dimension {name}', _with(case, lambda s, d, n=name: (s['dimensions'].pop(n), d.pop(n, None)))
    for name, entry in spec.get('constraints', {}).items():
        for smaller in _smaller(entry['expression']):
            yield f'{name}: {smaller}', _with(case, lambda s, _, n=name, e=smaller: _rewrite_constraint(s, n, e))
    for name, entry in spec.get('expressions', {}).items():
        if isinstance(entry, str):
            for smaller in _smaller(entry):
                yield (
                    f'{name}: {smaller}',
                    _with(case, lambda s, _, n=name, e=smaller: s['expressions'].__setitem__(n, e)),
                )
    if 'objective' in spec:
        for smaller in _smaller(spec['objective']['expression']):
            yield (
                f'objective: {smaller}',
                _with(case, lambda s, _, e=smaller: s['objective'].__setitem__('expression', e)),
            )
    for section in ('variables', 'constraints'):
        for name, entry in spec.get(section, {}).items():
            if 'where' in entry:
                yield f'drop the where of {name}', _with(case, lambda s, _, sec=section, n=name: s[sec][n].pop('where'))
            if 'missing' in entry:
                yield (
                    f'drop the missing of {name}',
                    _with(case, lambda s, _, sec=section, n=name: s[sec][n].pop('missing')),
                )
            if 'domain' in entry:
                yield (
                    f'drop the domain of {name}',
                    _with(case, lambda s, _, sec=section, n=name: s[sec][n].pop('domain')),
                )
    for name, dim in spec.get('dimensions', {}).items():
        if dim.get('ordered'):
            yield f'unorder {name}', _with(case, lambda s, _, n=name: s['dimensions'][n].pop('ordered'))
        labels = sources.get(name, [])
        labels = labels.get_column(name).to_list() if isinstance(labels, pl.DataFrame) else labels
        for label in labels if len(labels) > 1 else ():
            yield f'drop label {label!r} of {name}', _with(case, lambda _, d, n=name, x=label: _drop_label(d, n, x))
    for name, source in sources.items():
        if isinstance(source, pl.DataFrame) and source.height > 1:
            for i in range(source.height):
                yield (
                    f'drop row {i} of {name}',
                    _with(case, lambda _, d, n=name, k=i: d.__setitem__(n, _without_row(d[n], k))),
                )


def _with(case: Case, change: Callable[[dict[str, Any], dict[str, Any]], Any]) -> Case:
    spec, sources = copy.deepcopy(case.spec), dict(case.sources)
    change(spec, sources)
    return replace(case, spec=spec, sources=sources)


def _drop(spec: dict[str, Any], sources: dict[str, Any], section: str, name: str) -> None:
    del spec[section][name]
    if not spec[section]:
        del spec[section]
    sources.pop(name, None)


def _drop_label(sources: dict[str, Any], dim: str, label: Any) -> None:
    """*label* gone from *dim*'s index and from every table and dict keyed by it."""
    for name, source in sources.items():
        if isinstance(source, pl.DataFrame) and dim in source.columns:
            sources[name] = source.filter(pl.col(dim) != label)
        elif name == dim and isinstance(source, list):
            sources[name] = [x for x in source if x != label]
        elif isinstance(source, dict) and label in source:
            sources[name] = {k: v for k, v in source.items() if k != label}


def _without_row(frame: pl.DataFrame, i: int) -> pl.DataFrame:
    return pl.concat([frame.slice(0, i), frame.slice(i + 1)])


def _smaller(text: str) -> Iterator[str]:
    """Every expression one step smaller than *text*: a node replaced by an operand, or a keyword dropped.

    The grammar is close enough to Python's that ``ast`` parses it; a text it
    does not parse yields nothing.
    """
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError:
        return
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.Call, ast.BinOp, ast.UnaryOp))]
    for i in range(len(nodes)):
        for variant in _variants(nodes[i]):
            candidate = copy.deepcopy(tree)
            target = [n for n in ast.walk(candidate) if isinstance(n, (ast.Call, ast.BinOp, ast.UnaryOp))][i]
            _replace(candidate, target, variant(target))
            yield ast.unparse(candidate)


def _variants(node: ast.AST) -> list[Callable[[Any], ast.AST]]:
    if isinstance(node, ast.Call):
        drops = [
            (lambda n, k=k: ast.Call(n.func, n.args, [w for w in n.keywords if w.arg != k]))
            for k in (w.arg for w in node.keywords)
            if k in ('edge', 'within', 'by')
        ]
        return [lambda n: n.args[0], *drops] if node.args else drops
    if isinstance(node, ast.BinOp):
        return [lambda n: n.left, lambda n: n.right]
    return [lambda n: n.operand]


def _replace(tree: ast.AST, target: ast.AST, new: ast.AST) -> None:
    for parent in ast.walk(tree):
        for name, value in ast.iter_fields(parent):
            if value is target:
                setattr(parent, name, new)
            elif isinstance(value, list):
                setattr(parent, name, [new if v is target else v for v in value])


def _rewrite_constraint(spec: dict[str, Any], name: str, expression: str) -> None:
    """Set *expression* and the dims it carries, read off the language, or leave a spec it refuses."""
    spec['constraints'][name]['expression'] = expression
    probe = copy.deepcopy(spec)
    del probe['constraints'][name]
    sides = _sides(expression)
    probe['expressions'] = {**probe.get('expressions', {}), **{f'__side{i}': side for i, side in enumerate(sides)}}
    try:
        program = to_spec(probe).program
    except MathSpecError:
        return
    dims = set()
    for i in range(len(sides)):
        dims |= set(program.expressions[f'__side{i}'].dims)
    spec['constraints'][name]['dims'] = sorted(dims)


def _sides(expression: str) -> list[str]:
    for comparator in ('<=', '>=', '=='):
        if comparator in expression:
            return [side.strip() for side in expression.split(comparator, 1)]
    return [expression]
