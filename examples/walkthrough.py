"""The whole architecture, one model, one stage at a time — run it and read.

docs/about/architecture.md describes the pipeline; this script executes it one
stage at a time and prints the artifact each stage produces, through the same
entry points ``sps.solve`` takes internally.

    pixi run python examples/walkthrough.py

Its output is committed as ``examples/walkthrough.out`` and asserted line for
line by ``tests/test_walkthrough.py``, so everything printed is deterministic.

A YAML math spec is a closed AST known before any data is touched: stages 1-3
happen with no data attached at all; only stage 4 sees a number.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
from mathspec import Spec, to_spec

import specsolve as sps
from specsolve.relational.engines.polars.engine import PolarsEngine
from specsolve.sources import tidy_sources

HERE = Path(__file__).parent
SPEC = HERE / 'walkthrough.yaml'

#: Six snapshots of demand against four generators. Small enough to print.
GENERATORS = ['wind', 'solar', 'gas', 'oil']
SOURCES = {
    'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [100.0, 60.0, 200.0, 0.0]}),
    'load': pl.DataFrame({'snapshot': range(6), 'value': [80.0, 120.0, 150.0, 180.0, 140.0, 100.0]}),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 0.0, 50.0, 80.0]}),
    'snapshot': range(6),
    'generator': GENERATORS,
}

#: Two ways out of the language, caught at two different stages (see stage 7).
_REFUSED = [
    (
        'an operator that is not in the closed built-in set',
        {
            'constraints': {
                'cumulative': {
                    'dims': ['snapshot'],
                    'expression': 'cumsum(total_supply) <= load',
                }
            }
        },
    ),
    (
        'variable x variable x variable — the ceiling is degree 2, not "nonlinear is fine"',
        {'objective': {'sense': 'minimize', 'expression': 'sum(p * p * p)'}},
    ),
]


def banner(n: int, title: str, module: str) -> None:
    print(f'\n{_bold(f"[{n}] {title}")}  {_dim(f"({module})")}')


def main() -> None:
    print(__doc__.split('\n\n')[0])
    print(f'\nspec: {SPEC.relative_to(HERE.parent)}')

    schema = validated_model()
    expanded_ast(schema)
    program = relational_ir(schema)
    with PolarsEngine() as engine:
        model_frames(engine, schema, program)
        lp_file(engine)
        solution(engine)
    refusals()

    print(f'\n{_dim("docs/about/architecture.md has the rules these stages enforce.")}')


def validated_model() -> Spec:
    """Stage 1 — YAML text to a validated spec.

    Parses the file, type-checks it against the pydantic schema, and
    name-checks every expression, where string, named expression and macro
    template, used or not. After this call the spec is known to be
    well-formed; no data has been touched.
    """
    banner(1, 'YAML text -> validated Spec', 'mathspec.to_spec')
    schema = to_spec(SPEC)
    print(f'    dimensions   {", ".join(schema.dimensions)}')
    print(f'    parameters   {", ".join(schema.parameters)}')
    print(f'    variables    {", ".join(schema.variables)}')
    print(f'    constraints  {", ".join(schema.constraints)}')
    print(f'    expressions  {", ".join(schema.expressions)}    <- tier 2, still present')
    print(f'    macros       {", ".join(schema.macros)}   <- tier 2, still present')
    return schema


def expanded_ast(schema: Spec) -> None:
    """Stage 2 — macros substituted away, named expressions kept as nodes.

    The core AST is the whole language. Everything above it is pure
    substitution, so no lane ever sees a macro. A named expression is not
    substituted: it stands as one node wherever a constraint uses it, its body
    under it, and stage 6 reads it back by name at the solution.
    """
    banner(2, 'expand macros -> core AST', 'Spec.program')
    objective_text = schema.objective.expression
    core = schema.program.objective.expression
    print(f'    written      {objective_text!r}')
    print(f'    core AST     {core}')
    print('                 ^ the macro is gone: sum(p * cost, over=generator)')


def relational_ir(schema: Spec) -> Any:
    """Stage 3 — the core AST lowered to the relational plan.

    The language's boundary is decided here, by attempting the lowering. It
    needs no data, so ``sps.check()`` can compile a spec with nothing attached.
    """
    banner(3, 'a spec -> the program both lanes build from', 'Spec.program')
    program = schema.program
    print('    Program(')
    for name, decl in (*program.variables.items(), *program.constraints.items()):
        print(f'      {name}: {decl}')
    print(f'      {program.objective}')
    print('    )')
    print('    ^ frozen dataclasses, no macro, no YAML, no linopy, no engine')
    return program


def model_frames(engine: PolarsEngine, schema: Spec, program: Any) -> None:
    """Stage 4 — plan plus data to the model frames, the first stage to see a number.

    Sources are adapted to tidy frames (dims..., value) and the engine
    assembles the model from them.

    The frames are engine-private, so this is the one place the script reads
    private attributes.
    """
    banner(4, 'plan + data -> the model frames', 'relational/engines/polars/engine.py')
    engine.build(program, tidy_sources(program, SOURCES))
    handoff = engine._model.handoff
    for name, frame in (
        ('cols', handoff.cols),
        ('obj', handoff.obj),
        ('rows', handoff.rows),
        ('A', handoff.matrix),
    ):
        print(f'    {name:<20} {frame.height:>4} rows')
    print('\n    cols/rows/A/obj = the LP itself, in COO form:')
    print('    a column is a bound and a cost, a row a sense and a rhs,')
    print('    and A is every nonzero coefficient as (row, col, coeff).')

    variables = engine._model.variables['p'].frame.collect()
    n_full = 6 * 4
    print('\n    where "p_max > 0" is not a mask array — it is row absence:')
    print(f'    p has {variables.height} rows, not {n_full}: retired oil never becomes a column.')
    print(_indent(variables.sort('var_label').head(4)))


def lp_file(engine: PolarsEngine) -> None:
    """Stage 5 — the same frames, second sink.

    The other one (the ``highs`` solver, stage 6) hands COO batches to highspy
    without ever forming a full CSR here.
    """
    banner(5, 'sink: stream the frames to an LP file', 'relational/sinks/writers/lp_file.py')
    with tempfile.TemporaryDirectory() as tmp:
        lp = Path(tmp) / 'model.lp'
        engine.write(lp)
        text = lp.read_text().splitlines()
        print(_indent('\n'.join(text[:12])))
        print(f'    ... ({len(text)} lines total)')


def solution(engine: PolarsEngine) -> None:
    """Stage 6 — batches to highspy, and the solution read back by label join.

    Never densified. ``primal()`` hands back a frame, printed unsorted: label
    order, row-major over the coordinate product, as ``Result.primal``
    documents.

    ``evaluate()`` reads the quantity stage 1 named, lowered on this call
    rather than at build.
    """
    banner(6, 'sink: batches -> highspy -> solution frames', 'relational/sinks/solvers/highs.py')
    result = engine.solve()
    print(f'    status     {result.status} ({result.termination_condition})')
    print(f'    objective  {result.objective:,.1f}')
    print(_indent(result.primal('p').head(6)))
    print(_indent(result.evaluate('total_supply').head(3)))
    print('                 ^ the named expression, read back at the solution')


def refusals() -> None:
    """Stage 7 — what the language refuses, and which stage catches it.

    The error names the construct and its rewrite. Each spec is run through
    ``sps.check()`` — stages 1-3, no data attached — and then, only if that
    passes, through a build, so the output says which stage caught it.

    ``LanguageError`` is a ``ValueError`` subclass, so that is what is caught.
    """
    banner(7, 'and what the language refuses', 'mathspec')
    for label, patch in _REFUSED:
        print(f'\n    {label}:')
        spec = {**_raw(SPEC), **patch}
        try:
            sps.check(spec)
        except ValueError as exc:
            _refusal('check()', exc)
            continue
        try:
            sps.build(spec, SOURCES).close()
        except ValueError as exc:
            _refusal('build()', exc)


def _refusal(verb: str, exc: Exception) -> None:
    print(f'    {_dim(f"caught by {verb} as {type(exc).__name__}")}')
    print(_indent(str(exc), '      '))


def _bold(text: str) -> str:
    return f'\033[1m{text}\033[0m' if sys.stdout.isatty() else text


def _dim(text: str) -> str:
    return f'\033[2m{text}\033[0m' if sys.stdout.isatty() else text


def _indent(obj: object, pad: str = '    ') -> str:
    return '\n'.join(pad + line for line in str(obj).splitlines())


def _raw(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


if __name__ == '__main__':
    main()
