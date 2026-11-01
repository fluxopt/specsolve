"""A random model and random data for it, from one seed.

The generator knows the declarations it made, so every expression it writes
reads names that exist, carries the dimensions it says it carries, and stays
inside degree two. It does not try to know every rule the loader holds a file
to: a spec the language refuses is counted as *rejected*, not as a finding, and
:mod:`tests.fuzz.run` reports the rate so a generator that drifts out of the
language shows. The data is written against the same declarations, dense where
a parameter refuses a missing row and sparse where its ``missing:`` allows one.

Everything is drawn from one ``random.Random(seed)``, so a seed is the whole
reproduction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import polars as pl

#: The dimension pool: name, dtype, whether its order counts.
DIMENSIONS = (
    ('t', 'int', True),
    ('g', 'str', False),
    ('b', 'str', False),
    ('h', 'int', True),
    ('k', 'int', False),
)

#: Coefficients and data values: zeros and negatives on purpose.
NUMBERS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
INTEGERS = (-1, 0, 1, 2, 3)
LITERALS = ('1', '2', '0.5', '3', '1.5')


@dataclass(frozen=True)
class Node:
    """One generated expression and what the language would say about it.

    Attributes:
        text: The expression as the file writes it.
        dims: The dimensions it carries.
        degree: 0 for data, 1 linear, 2 quadratic.
        reduced: Whether a sum of terms sits at the top, which a product of two
            variable factors refuses on both sides.
    """

    text: str
    dims: frozenset[str]
    degree: int
    reduced: bool = False

    def __str__(self) -> str:
        return self.text


@dataclass
class Dimension:
    dtype: str
    ordered: bool
    labels: list[Any]


@dataclass
class Relation:
    """A keyed map from *key* to *value*, or with ``bare`` a set of ``(key, value)`` pairs."""

    key: str
    value: str
    missing: str
    rows: list[tuple[Any, Any]]
    bare: bool = False


@dataclass
class Parameter:
    dims: tuple[str, ...]
    dtype: str
    missing: str | float | bool
    rows: dict[tuple[Any, ...], Any]


@dataclass
class Variable:
    dims: tuple[str, ...]
    entry: dict[str, Any]


@dataclass
class Case:
    """One generated model: the spec as a dict, the sources for it, and the reads to compare."""

    seed: int
    spec: dict[str, Any]
    sources: dict[str, Any]
    reads: list[str] = field(default_factory=list)
    linear: bool = True


class Generator:
    """Draws one model from *seed*; :meth:`case` is the whole interface."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.dims: dict[str, Dimension] = {}
        self.relations: dict[str, Relation] = {}
        self.params: dict[str, Parameter] = {}
        self.variables: dict[str, Variable] = {}
        self.named: dict[str, Node] = {}
        self.named_entries: dict[str, Any] = {}
        self.masks: dict[str, tuple[str, frozenset[str]]] = {}

    # ------------------------------------------------------------------
    # declarations
    # ------------------------------------------------------------------

    def case(self) -> Case:
        r = self.rng
        self._dimensions()
        self._relations()
        self._parameters()
        for i in range(r.choice((0, 0, 1))):
            self._named_mask(f'm{i}')
        self._variables()
        for i in range(r.choice((0, 0, 1, 2))):
            self._named_expression(f'e{i}')
        constraints = {}
        reads = list(self.named)
        linear = True
        for i in range(r.randint(1, 3)):
            entry, sides = self._constraint()
            if entry is None:
                continue
            constraints[f'c{i}'] = entry
            reads.extend(s.text for s in sides)
            linear &= all(s.degree <= 1 for s in sides)
        objective = self._objective()
        spec: dict[str, Any] = {
            'dimensions': {
                name: {'dtype': d.dtype, **({'ordered': True} if d.ordered else {})} for name, d in self.dims.items()
            },
            'variables': {name: v.entry for name, v in self.variables.items()},
        }
        if self.relations:
            spec['relations'] = {name: _relation_entry(rel) for name, rel in self.relations.items()}
        if self.masks:
            spec['conditions'] = {name: text for name, (text, _) in self.masks.items()}
        if self.params:
            spec['parameters'] = {name: self._parameter_entry(p) for name, p in self.params.items()}
        if self.named_entries:
            spec['expressions'] = dict(self.named_entries)
        if constraints:
            spec['constraints'] = constraints
        if objective is not None:
            spec['objective'] = {'sense': r.choice(('minimize', 'maximize')), 'expression': objective.text}
            reads.append(objective.text)
            linear &= objective.degree <= 1
        return Case(self.seed, spec, self._sources(), reads=list(dict.fromkeys(reads)), linear=linear)

    def _dimensions(self) -> None:
        r = self.rng
        pool = list(DIMENSIONS)
        r.shuffle(pool)
        for name, dtype, ordered in pool[: r.randint(1, 3)]:
            size = r.choice((1, 2, 2, 3, 3, 4))
            if dtype == 'int':
                labels: list[Any] = r.sample(range(-3, 12), size)
                if r.random() < 0.5:
                    labels.sort()
            else:
                labels = [f'{name}{i}' for i in r.sample(range(9), size)]
            self.dims[name] = Dimension(dtype, ordered, labels)

    def _relations(self) -> None:
        r = self.rng
        names = list(self.dims)
        for i in range(r.choice((0, 1, 1, 2))):
            if len(names) < 2:
                return
            key, value = r.sample(names, 2)
            if r.random() < 0.25:
                pairs = [(k, v) for k in self.dims[key].labels for v in self.dims[value].labels if r.random() < 0.5]
                self.relations[f'r{i}'] = Relation(key, value, 'absent', pairs, bare=True)
                continue
            missing = r.choice(('refused', 'refused', 'absent'))
            rows = [(k, r.choice(self.dims[value].labels)) for k in self.dims[key].labels]
            if missing == 'absent':
                rows = [row for row in rows if r.random() < 0.7]
            self.relations[f'r{i}'] = Relation(key, value, missing, rows)

    def _parameters(self) -> None:
        r = self.rng
        for i in range(r.randint(1, 5)):
            dims = self._some_dims()
            dtype = r.choice(('float', 'float', 'float', 'int', 'bool'))
            missing: str | float | bool = r.choice(('refused', 'refused', 'absent', 'neutral', 'value'))
            if missing == 'value':
                missing = {'float': r.choice((1.0, 0.0, 2.5)), 'int': r.choice((0, 1)), 'bool': r.random() < 0.5}[dtype]
            if dtype == 'bool' and missing == 'absent':
                missing = 'neutral'
            if not dims and missing != 'refused':
                missing = 'refused'
            rows = {}
            for coord in self._product(dims):
                if missing != 'refused' and r.random() < 0.3:
                    continue
                rows[coord] = self._value(dtype)
            self.params[f'p{i}'] = Parameter(dims, dtype, missing, rows)

    def _value(self, dtype: str) -> Any:
        r = self.rng
        if dtype == 'bool':
            return r.random() < 0.6
        if dtype == 'int':
            return r.choice(INTEGERS)
        return r.choice(NUMBERS)

    @staticmethod
    def _parameter_entry(p: Parameter) -> dict[str, Any]:
        entry: dict[str, Any] = {'dims': list(p.dims)}
        if p.dtype != 'float':
            entry['dtype'] = p.dtype
        if p.missing != 'refused':
            entry['missing'] = p.missing
        return entry

    def _variables(self) -> None:
        r = self.rng
        for i in range(r.randint(1, 3)):
            dims = self._some_dims(at_least=0 if i else 1)
            entry: dict[str, Any] = {'dims': list(dims)}
            lower = r.choice((-5, 0, 0, 'param'))
            upper = r.choice((5, 10, 'param'))
            bounds = {'lower': self._bound(lower, dims, -5), 'upper': self._bound(upper, dims, 5)}
            if bounds['lower'] is not None and bounds['upper'] is not None:
                entry['bounds'] = bounds
            else:
                entry['bounds'] = {'lower': -5, 'upper': 5}
            if dims and r.random() < 0.4:
                mask = self._mask(frozenset(dims))
                if mask is not None:
                    entry['where'] = mask
                    if r.random() < 0.3:
                        entry['missing'] = 'neutral'
            if r.random() < 0.1:
                entry['domain'] = 'integer'
            self.variables[f'v{i}'] = Variable(dims, entry)

    def _bound(self, choice: Any, dims: tuple[str, ...], fallback: float) -> Any:
        if choice != 'param':
            return choice
        fits = [
            name
            for name, p in self.params.items()
            if p.dtype in ('float', 'int') and set(p.dims) <= set(dims) and p.missing == 'refused'
        ]
        return self.rng.choice(fits) if fits else fallback

    def _some_dims(self, at_least: int = 0) -> tuple[str, ...]:
        names = list(self.dims)
        n = self.rng.randint(min(at_least, len(names)), min(2, len(names)))
        return tuple(self.rng.sample(names, n))

    def _product(self, dims: tuple[str, ...]) -> list[tuple[Any, ...]]:
        coords: list[tuple[Any, ...]] = [()]
        for d in dims:
            coords = [(*c, label) for c in coords for label in self.dims[d].labels]
        return coords

    # ------------------------------------------------------------------
    # masks
    # ------------------------------------------------------------------

    def _named_mask(self, name: str) -> None:
        frame = frozenset(self._some_dims(at_least=1))
        text = self._mask(frame)
        if text is not None:
            self.masks[name] = (text, frame)

    def _atoms(self, frame: frozenset[str]) -> list[str]:
        """Every predicate the generator writes whose dimensions fit inside *frame*."""
        r = self.rng
        atoms = []
        for name, p in self.params.items():
            if not p.dims:
                continue
            if set(p.dims) <= frame:
                if p.dtype == 'bool':
                    atoms += [name, f'NOT {name}']
                elif p.dtype in ('float', 'int'):
                    atoms.append(f'{name} {r.choice((">", ">=", "!=", "<"))} {r.choice((0, 1))}')
                    if p.dtype == 'float':
                        atoms.append(name)
            if p.dtype == 'bool':
                for d in p.dims:
                    if set(p.dims) - {d} <= frame:
                        atoms.append(f'count({name}, over={d}) {r.choice((">=", "==", "<="))} 1')
                    if self.dims[d].ordered and set(p.dims) <= frame:
                        atoms.append(f'shift({name}, along={d}, offset={r.choice((1, -1))})')
                for rel_name, rel in self.relations.items():
                    if (
                        not rel.bare
                        and rel.value in p.dims
                        and rel.key not in p.dims
                        and (set(p.dims) - {rel.value}) | {rel.key} <= frame
                    ):
                        atoms.append(f'at({name}, by={rel_name}[{rel.value}])')
        for d in sorted(frame):
            dim = self.dims[d]
            if dim.ordered:
                atoms.append(f'position({d}) {r.choice((">", "==", "<"))} {r.choice((0, 1, -1))}')
                for rel_name, rel in self.relations.items():
                    if not rel.bare and rel.key == d:
                        atoms.append(f'position({d}, within={rel_name}[{rel.value}]) == 0')
            atoms.append(f'{d} != {self._quoted(r.choice(dim.labels))}')
        for name, rel in self.relations.items():
            if rel.key in frame and (not rel.bare or rel.value in frame):
                atoms.append(name)
        for name, v in self.variables.items():
            if v.dims and set(v.dims) <= frame:
                atoms.append(name)
        for name, (_, dims) in self.masks.items():
            if dims <= frame:
                atoms += [name, f'NOT {name}']
        return atoms

    def _mask(self, frame: frozenset[str]) -> str | None:
        """A ``where`` string over no dimension outside *frame*, or None if none fits."""
        atoms = self._atoms(frame)
        if not atoms:
            return None
        mask = self.rng.choice(atoms)
        if self.rng.random() < 0.3:
            mask = f'{mask} {self.rng.choice(("AND", "OR"))} {self.rng.choice(atoms)}'
        return mask

    @staticmethod
    def _quoted(label: Any) -> str:
        return f"'{label}'" if isinstance(label, str) else str(label)

    # ------------------------------------------------------------------
    # expressions
    # ------------------------------------------------------------------

    def expression(self, depth: int, *, variable_free: bool = False) -> Node | None:
        """One expression up to *depth* deep, or None where a draw found nothing that fits."""
        for _ in range(20):
            node = self._draw(depth, variable_free)
            if node is not None:
                return node
        return None

    def _draw(self, depth: int, variable_free: bool) -> Node | None:
        r = self.rng
        if depth <= 0 or r.random() < 0.25:
            return self._leaf(variable_free)
        kind = r.choice(
            (
                'add', 'add', 'sub', 'neg', 'mul', 'mul', 'div', 'pow', 'sum', 'sum_over', 'sum_over',
                'sum_by', 'at', 'shift', 'shift', 'sum_back',
            )
        )  # fmt: skip
        return getattr(self, f'_{kind}')(depth - 1, variable_free)

    def _leaf(self, variable_free: bool) -> Node | None:
        r = self.rng
        choices: list[Node] = [Node(r.choice(LITERALS), frozenset(), 0)]
        choices += [Node(name, frozenset(p.dims), 0) for name, p in self.params.items() if p.dtype in ('float', 'int')]
        if not variable_free:
            choices += [Node(name, frozenset(v.dims), 1) for name, v in self.variables.items()] * 2
            choices += [node for node in self.named.values() if node.degree <= 1]
        else:
            choices += [node for node in self.named.values() if node.degree == 0]
        return r.choice(choices)

    def _add(self, depth: int, vf: bool, op: str = '+') -> Node | None:
        a, b = self.expression(depth, variable_free=vf), self.expression(depth, variable_free=vf)
        if a is None or b is None:
            return None
        return Node(f'{a} {op} {b}', a.dims | b.dims, max(a.degree, b.degree), reduced=True)

    def _sub(self, depth: int, vf: bool) -> Node | None:
        a, b = self.expression(depth, variable_free=vf), self.expression(depth, variable_free=vf)
        if a is None or b is None:
            return None
        return Node(f'{a} - ({b})', a.dims | b.dims, max(a.degree, b.degree), reduced=True)

    def _neg(self, depth: int, vf: bool) -> Node | None:
        a = self.expression(depth, variable_free=vf)
        return None if a is None else Node(f'-({a})', a.dims, a.degree, a.reduced)

    def _mul(self, depth: int, vf: bool) -> Node | None:
        a, b = self.expression(depth, variable_free=vf), self.expression(depth, variable_free=vf)
        if a is None or b is None or a.degree + b.degree > 2:
            return None
        if a.degree and b.degree and (a.reduced or b.reduced):
            return None
        return Node(f'({a}) * ({b})', a.dims | b.dims, a.degree + b.degree)

    def _div(self, depth: int, vf: bool) -> Node | None:
        a = self.expression(depth, variable_free=vf)
        b = self.expression(max(depth - 1, 0), variable_free=True)
        if a is None or b is None:
            return None
        return Node(f'({a}) / ({b})', a.dims | b.dims, a.degree)

    def _pow(self, depth: int, vf: bool) -> Node | None:
        base = self.expression(max(depth - 1, 0), variable_free=True)
        if base is None:
            return None
        return Node(f'({base}) ** {self.rng.choice((2, 1, 0, 3))}', base.dims, 0)

    def _sum(self, depth: int, vf: bool) -> Node | None:
        a = self.expression(depth, variable_free=vf)
        if a is None or not a.dims:
            return None
        return Node(f'sum({a})', frozenset(), a.degree, reduced=True)

    def _sum_over(self, depth: int, vf: bool) -> Node | None:
        a = self.expression(depth, variable_free=vf)
        if a is None or not a.dims:
            return None
        over = self.rng.sample(sorted(a.dims), self.rng.randint(1, len(a.dims)))
        written = over[0] if len(over) == 1 else f'[{", ".join(over)}]'
        return Node(f'sum({a}, over={written})', a.dims - set(over), a.degree, reduced=True)

    def _sum_by(self, depth: int, vf: bool) -> Node | None:
        """A sum through a relation: through a map from key to value, and through a bare relation either way."""
        if not self.relations:
            return None
        name = self.rng.choice(sorted(self.relations))
        rel = self.relations[name]
        over, by = (rel.value, rel.key) if rel.bare and self.rng.random() < 0.5 else (rel.key, rel.value)
        a = self.expression(depth, variable_free=vf)
        if a is None or over not in a.dims or by in a.dims:
            return None
        return Node(f'sum({a}, over={over}, by={name}[{by}])', (a.dims - {over}) | {by}, a.degree, reduced=True)

    def _at(self, depth: int, vf: bool) -> Node | None:
        if not self.relations:
            return None
        name = self.rng.choice(sorted(self.relations))
        rel = self.relations[name]
        a = self.expression(depth, variable_free=vf)
        if a is None or rel.bare or rel.value not in a.dims or rel.key in a.dims:
            return None
        return Node(f'at({a}, by={name}[{rel.value}])', (a.dims - {rel.value}) | {rel.key}, a.degree, a.reduced)

    def _ordered_in(self, dims: frozenset[str]) -> list[str]:
        return sorted(d for d in dims if self.dims[d].ordered)

    def _partition(self, along: str) -> str | None:
        fits = [f'{name}[{rel.value}]' for name, rel in self.relations.items() if rel.key == along and not rel.bare]
        return self.rng.choice(fits) if fits and self.rng.random() < 0.4 else None

    def _amount(self, dims: frozenset[str], along: str, literals: tuple[int, ...]) -> str:
        """A literal, or now and then an ``int`` parameter that does not vary along *along*."""
        fits = [
            name
            for name, p in self.params.items()
            if p.dtype == 'int' and p.missing == 'refused' and along not in p.dims and set(p.dims) <= dims
        ]
        if fits and self.rng.random() < 0.3:
            return self.rng.choice(fits)
        return str(self.rng.choice(literals))

    def _shift(self, depth: int, vf: bool) -> Node | None:
        a = self.expression(depth, variable_free=vf)
        if a is None:
            return None
        along = self._ordered_in(a.dims)
        if not along:
            return None
        d = self.rng.choice(along)
        offset = self._amount(a.dims, d, (1, 1, 2, -1))
        args = [f'along={d}', f'offset={offset}']
        named = not offset.lstrip('-').isdigit()
        edge = self.rng.choice(
            (None, "'wrap'", '0') if a.degree and not named else ("'wrap'", '0') if a.degree else ("'wrap'", '0', '1.5')
        )
        if edge is not None:
            args.append(f'edge={edge}')
        within = self._partition(d)
        if within is not None:
            args.append(f'within={within}')
        return Node(f'shift({a}, {", ".join(args)})', a.dims, a.degree, a.reduced)

    def _sum_back(self, depth: int, vf: bool) -> Node | None:
        a = self.expression(depth, variable_free=vf)
        if a is None:
            return None
        along = self._ordered_in(a.dims)
        if not along:
            return None
        d = self.rng.choice(along)
        args = [f'along={d}', f'window={self._amount(a.dims, d, (1, 2, 3))}']
        if self.rng.random() < 0.3:
            args.append("edge='wrap'")
        within = self._partition(d)
        if within is not None:
            args.append(f'within={within}')
        return Node(f'sum_back({a}, {", ".join(args)})', a.dims, a.degree, reduced=True)

    # ------------------------------------------------------------------
    # entries
    # ------------------------------------------------------------------

    def _named_expression(self, name: str) -> None:
        r = self.rng
        body = self.expression(r.randint(1, 3))
        if body is None:
            return
        if body.dims and r.random() < 0.3:
            other, third = self.expression(1), self.expression(1)
            when = self._mask(body.dims)
            if other is not None and when is not None and other.dims <= body.dims:
                cases = {'one': {'when': when, 'expression': other.text}}
                flags = [n for n, p in self.params.items() if p.dtype == 'bool' and p.dims and set(p.dims) <= body.dims]
                if flags and third is not None and third.dims <= body.dims and r.random() < 0.5:
                    flag = r.choice(flags)
                    cases = {
                        'one': {'when': flag, 'expression': other.text},
                        'two': {'when': f'NOT {flag}', 'expression': third.text},
                    }
                self.named_entries[name] = {'dims': sorted(body.dims), 'cases': cases, 'otherwise': body.text}
                degrees = [body.degree, other.degree] + ([third.degree] if 'two' in cases and third else [])
                self.named[name] = Node(name, body.dims, max(degrees))
                return
        self.named_entries[name] = body.text
        self.named[name] = Node(name, body.dims, body.degree, body.reduced)

    def _constraint(self) -> tuple[dict[str, Any] | None, tuple[Node, ...]]:
        r = self.rng
        lhs = self.expression(r.randint(1, 3))
        rhs = self.expression(r.randint(0, 2))
        if lhs is None or rhs is None or not (lhs.degree or rhs.degree):
            return None, ()
        entry: dict[str, Any] = {
            'dims': sorted(lhs.dims | rhs.dims),
            'expression': f'{lhs} {r.choice(("<=", ">=", "<=", "=="))} {rhs}',
        }
        if entry['dims'] and r.random() < 0.3:
            mask = self._mask(lhs.dims | rhs.dims)
            if mask is not None:
                entry['where'] = mask
        return entry, (lhs, rhs)

    def _objective(self) -> Node | None:
        """A body that carries a variable: the oracle has no slot for an objective constant (#894)."""
        for _ in range(10):
            body = self.expression(self.rng.randint(1, 3))
            if body is not None and body.degree:
                break
        else:
            return None
        if body.dims:
            body = Node(f'sum({body})', frozenset(), body.degree, reduced=True)
        return body

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------

    def _sources(self) -> dict[str, Any]:
        r = self.rng
        sources: dict[str, Any] = {}
        for name, d in self.dims.items():
            labels = list(d.labels)
            sources[name] = (
                pl.DataFrame({name: labels}, schema={name: _polars_dtype(d.dtype)}) if r.random() < 0.2 else labels
            )
        for name, p in self.params.items():
            dense = len(p.rows) == len(self._product(p.dims))
            if not p.dims and r.random() < 0.5:
                sources[name] = next(iter(p.rows.values()))
                continue
            if len(p.dims) == 1 and dense and r.random() < 0.2:
                sources[name] = {c[0]: v for c, v in p.rows.items()}
                continue
            columns = {d: [c[i] for c in p.rows] for i, d in enumerate(p.dims)}
            columns['value'] = list(p.rows.values())
            schema = {d: _polars_dtype(self.dims[d].dtype) for d in p.dims}
            integral = p.dtype == 'float' and all(float(v).is_integer() for v in p.rows.values())
            schema['value'] = pl.Int64 if integral and r.random() < 0.3 else _polars_dtype(p.dtype)
            if schema['value'] == pl.Int64 and p.dtype == 'float':
                columns['value'] = [int(v) for v in columns['value']]
            sources[name] = pl.DataFrame(columns, schema=schema)
        for name, rel in self.relations.items():
            sources[name] = pl.DataFrame(
                {rel.key: [k for k, _ in rel.rows], rel.value: [v for _, v in rel.rows]},
                schema={
                    rel.key: _polars_dtype(self.dims[rel.key].dtype),
                    rel.value: _polars_dtype(self.dims[rel.value].dtype),
                },
            )
        return sources


def _relation_entry(rel: Relation) -> dict[str, Any]:
    if rel.bare:
        return {'key': [rel.key, rel.value], 'missing': 'absent'}
    return {'key': rel.key, 'values': rel.value, **({'missing': 'absent'} if rel.missing == 'absent' else {})}


def _polars_dtype(dtype: str) -> Any:
    return {'int': pl.Int64, 'str': pl.String, 'float': pl.Float64, 'bool': pl.Boolean}[dtype]


def generate(seed: int) -> Case:
    """The model and data *seed* draws."""
    return Generator(seed).case()
