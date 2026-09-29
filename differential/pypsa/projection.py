"""A rung's model: the one file, cut to what the rung builds.

The file states every row PyPSA can emit; a rung's network builds some of
them. Its projection keeps the constraints with rows built and the variables
with columns built, drops from every kept expression the additive terms over
variables the rung never builds or parameters and relations it never feeds —
they contribute nothing — and keeps the parameters, relations and dimensions
those blocks still name. The parity runner writes one per rung, solves it and
holds it to PyPSA's objective.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

NAME = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*\b')


def _relation_dims(relation: dict[str, Any]) -> set[str]:
    """The dimensions a raw relation block names, in every spelling of a side."""
    sides = (relation['key'], relation.get('values', ()))
    return {d for side in sides for d in _side_dims(side)}


def _side_dims(side: str | list[str] | dict[str, str]) -> set[str]:
    if isinstance(side, dict):
        return set(side.values())
    return {side} if isinstance(side, str) else set(side)


def terms(expression: str) -> list[str]:
    """The top-level additive terms of *expression*, each carrying its own sign."""
    out, depth, start = [], 0, 0
    text = expression.strip()
    for i, ch in enumerate(text):
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        elif ch in '+-' and depth == 0 and i > 0 and text[i - 1] in ' \n':
            out.append(text[start:i].strip())
            start = i
    out.append(text[start:].strip())
    return [t for t in out if t]


def names(text: str) -> set[str]:
    return set(NAME.findall(text or ''))


def _split_comparison(expression: str) -> tuple[str, str, str] | None:
    for op in ('<=', '>=', '=='):
        if op in expression:
            left, right = expression.split(op, 1)
            return left, op, right
    return None


def _cut(expression: str, dead: set[str]) -> str | None:
    """*expression* without the terms that name a dead variable; ``None`` when a side loses every term."""
    parts = _split_comparison(expression)
    sides = [parts[0], parts[2]] if parts else [expression]
    kept_sides = []
    for side in sides:
        kept = [t for t in terms(side) if not (names(t) & dead)]
        if not kept:
            return None
        joined = ' '.join(kept)
        kept_sides.append(joined.removeprefix('+ '))
    return f'{kept_sides[0]} {parts[1]} {kept_sides[1]}' if parts else kept_sides[0]


def _mentions(block: dict[str, Any]) -> set[str]:
    """Every declared name a block reads — through a ``cases:`` block's regions as well as its own keys."""
    found = names(str(block.get('expression', ''))) | names(str(block.get('where', '')))
    found |= names(' '.join(str(v) for v in (block.get('bounds') or {}).values()))
    if 'cases' in block:
        found |= names(str(block.get('otherwise', '')))
        for case in block['cases'].values():
            found |= names(str(case.get('when', ''))) | names(str(case.get('expression', '')))
    return found


def project(raw: dict[str, Any], parity: dict[str, Any]) -> dict[str, Any]:
    """The projection of *raw* (the file as a dict) onto what *parity* says the rung built.

    A cased quantity is kept whole, because only its regions' masks may say
    whether a region applies. Expressions keep the file's order, so the output
    is stable across processes. A variable a kept quantity names is declared
    as written, even where the rung builds no column for it.
    """
    variables = {n: v for n, v in raw['variables'].items() if parity['built_columns'].get(n)}
    fed = set(parity['attached_nonempty'])
    dead = (
        (set(raw['variables']) - set(variables))
        | (set(raw['parameters']) - fed)
        | (set(raw.get('relations', {})) - fed)
    )
    constraints = {}
    for name, block in raw['constraints'].items():
        if not parity['built_rows'].get(name):
            continue
        cut = _cut(block['expression'], dead)
        if cut is not None:
            constraints[name] = {**block, 'expression': cut}
    objective = {**raw['objective'], 'expression': _cut(raw['objective']['expression'], dead)}
    survived = {}
    for name, block in raw.get('expressions', {}).items():
        if 'cases' in block:
            survived[name] = block
            continue
        cut = _cut(block['expression'], dead)
        if cut is not None:
            survived[name] = {**block, 'expression': cut}

    mentioned: set[str] = set()
    for block in (*constraints.values(), *variables.values(), objective):
        mentioned |= _mentions(block)
    expressions: dict[str, Any] = {}
    while reached := [n for n in survived if n in mentioned and n not in expressions]:
        for name in reached:
            expressions[name] = survived[name]
            mentioned |= _mentions(survived[name])

    variables |= {n: v for n, v in raw['variables'].items() if n in mentioned}
    for block in variables.values():
        mentioned |= _mentions(block)
    parameters = {n: p for n, p in raw['parameters'].items() if n in mentioned}
    relations = {n: lk for n, lk in raw.get('relations', {}).items() if n in mentioned}
    dims: set[str] = set()
    for block in (*variables.values(), *constraints.values()):
        dims |= set(block.get('dims', []))
    for p in parameters.values():
        dims |= set(p.get('dims', []))
    for lk in relations.values():
        dims |= _relation_dims(lk)
    dimensions = {n: d for n, d in raw['dimensions'].items() if n in dims}
    out = {k: v for k, v in raw.items() if k in ('version', 'description')}
    out['dimensions'] = dimensions
    if relations:
        out['relations'] = relations
    out['parameters'] = parameters
    out['variables'] = variables
    out['constraints'] = constraints
    if expressions:
        out['expressions'] = expressions
    out['objective'] = objective
    return out


class _Compact(yaml.SafeDumper):
    """Lists and scalar-only mappings inline — ``dims: [snapshot, generator]``, ``bounds: {lower: 0}`` — as the file writes them."""


def _inline_list(dumper: yaml.SafeDumper, data: list) -> yaml.Node:
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


def _inline_scalar_mapping(dumper: yaml.SafeDumper, data: dict) -> yaml.Node:
    flat = all(isinstance(v, (str, int, float, bool)) or v is None for v in data.values())
    return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=flat and len(data) <= 3)


_Compact.add_representer(list, _inline_list)
_Compact.add_representer(dict, _inline_scalar_mapping)


def dump(projected: dict[str, Any]) -> str:
    return yaml.dump(projected, Dumper=_Compact, sort_keys=False, allow_unicode=True, width=100)
