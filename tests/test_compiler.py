"""The compiler is lazy: a plan node goes in and a query comes out, no row read.

This is the only place query shape is asserted. Every property here can
regress while every model still solves to the right answer:

- ``AGGREGATE`` absent from ``Sum`` and ``GroupSum``; duplicates collapse once,
  in the terminal ``SUM(coeff) GROUP BY row, col`` at assembly.
- ``OVER`` absent from a translation, which joins the dim table twice instead.
- the modulo appearing only when a translation wraps.
- a dimension comparison *filtering* a column the frame already carries rather
  than joining to find it, and a constant bound costing no join at all.
- ``SEMI JOIN`` present for a mask that reads some of the frame's dims and
  absent for one that reads them all.
- ``INNER`` rather than ``LEFT`` for a name the mask is certain of, and ``LEFT``
  again once that name sits under an ``Or``.

The frames are empty frames with the right schemas: a schema is all it takes
to compile. The assertions are about shape, not exact text.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import polars as pl
import pytest
from mathspec import program

from specsolve.errors import SpecsolveError
from specsolve.relational.engine.attaching import AttachedSources
from specsolve.relational.engine.compiler import Compiler
from specsolve.relational.engine.labels import Labelled
from specsolve.relational.engine.predicates import masked
from specsolve.relational.engine.scope import Scope

if TYPE_CHECKING:
    from collections.abc import Mapping

#: A generator's bus: the single-valued map, keyed by the generator.
GEN_BUS = program.RelationDeclaration((('generator', 'generator'), ('bus', 'bus')), ('generator',))

PROGRAM = program.Program(
    parameters={
        'cost': program.ParameterDeclaration(('generator',)),
        'load': program.ParameterDeclaration(('snapshot',)),
        'available': program.ParameterDeclaration(('generator',)),
    },
    variables={'p': program.VariableDeclaration(('snapshot', 'generator'))},
    constraints={},
    objective=program.ObjectiveDeclaration('minimize', program.Variable('p')),
    dimensions={
        'snapshot': program.DimensionDeclaration(),
        'generator': program.DimensionDeclaration(),
        'bus': program.DimensionDeclaration(),
    },
    relations={'gen_bus': GEN_BUS},
)

#: The direction every grouping case here reads: out of `generator`, into `bus`.
GEN_BUS_DIRECTION = program.Direction('gen_bus', GEN_BUS, ('generator',), ('bus',), ())

CARDINALITY = {'snapshot': 24, 'generator': 3, 'bus': 2}

DIMENSIONS = {
    'snapshot': pl.LazyFrame(schema={'val': pl.Int64, 'ord': pl.Int64}),
    'generator': pl.LazyFrame(schema={'val': pl.String, 'ord': pl.Int64}),
    'bus': pl.LazyFrame(schema={'val': pl.String, 'ord': pl.Int64}),
}
RELATIONS = {'gen_bus': pl.LazyFrame(schema={'generator': pl.String, 'bus': pl.String})}
PARAMETERS = {
    'cost': pl.LazyFrame(schema={'generator': pl.String, 'value': pl.Float64}),
    'load': pl.LazyFrame(schema={'snapshot': pl.Int64, 'value': pl.Float64}),
    'available': pl.LazyFrame(schema={'generator': pl.String, 'value': pl.Float64}),
}
VARIABLES = {
    'p': Labelled(pl.LazyFrame(schema={'snapshot': pl.Int64, 'generator': pl.String, 'var_label': pl.Int64}), 0, 0)
}


def attached() -> AttachedSources:
    """The data a query is written against — schemas only, no rows."""
    return AttachedSources(
        parameters=PARAMETERS,
        dimensions=DIMENSIONS,
        relations=RELATIONS,
        cardinality=CARDINALITY,
        parameter_rows={},
        consecutive={},
    )


def declared(dtypes: Mapping[str, str] = MappingProxyType({})) -> program.Program:
    """PROGRAM with the declared dtypes a case needs."""
    return replace(
        PROGRAM,
        parameters={n: replace(p, dtype=dtypes.get(n, p.dtype)) for n, p in PROGRAM.parameters.items()},
    )


def compiler(dtypes: Mapping[str, str] = MappingProxyType({})) -> Compiler:
    return Compiler(Scope(declared(dtypes), attached(), VARIABLES))


def columns(frame: pl.LazyFrame) -> list[str]:
    return frame.collect_schema().names()


def query(frame: pl.LazyFrame) -> str:
    """The query plan as text — what an admissibility judgement is read off."""
    return frame.explain(optimized=False)


def joins(frame: pl.LazyFrame) -> int:
    """How many joins the plan performs, counted on the ``… JOIN:`` headers polars prints."""
    return query(frame).count('JOIN:')


# ---------------------------------------------------------------------------
# expressions
# ---------------------------------------------------------------------------


def test_a_variable_compiles_to_one_term_piece_over_its_dims():
    compiled = compiler().expression(program.Variable('p'), 'test')
    assert len(compiled.terms) == 1
    assert not compiled.consts
    piece = compiled.terms[0]
    assert piece.dims == ('snapshot', 'generator')
    assert piece.kind == 'term'
    assert columns(piece.frame) == ['snapshot', 'generator', 'var_label', 'coeff']


def test_a_parameter_is_a_constant_piece_not_a_term():
    compiled = compiler().expression(program.Parameter('cost'), 'test')
    assert not compiled.terms
    assert compiled.consts[0].dims == ('generator',)
    assert compiled.consts[0].kind == 'const'
    assert columns(compiled.consts[0].frame) == ['generator', 'cval']


def test_addition_concatenates_pieces_rather_than_joining():
    """An LP row is a sum of terms, so ``+`` needs no query at all."""
    compiled = compiler().expression(program.Add(program.Variable('p'), program.Variable('p')), 'test')
    assert len(compiled.terms) == 2


def test_multiplying_a_variable_by_a_parameter_joins_on_the_shared_dim():
    compiled = compiler().expression(program.Multiply(program.Variable('p'), program.Parameter('cost')), 'test')
    piece = compiled.terms[0]
    assert piece.dims == ('snapshot', 'generator')
    assert columns(piece.frame) == ['snapshot', 'generator', 'var_label', 'coeff']
    assert 'JOIN' in query(piece.frame)


def test_a_quadratic_product_compiled_as_affine_is_an_invariant_not_a_refusal():
    """A quadratic product in a position compiled as affine is the lane contradicting itself.

    The assert stands in front of a term whose second variable would be
    silently dropped.
    """
    with pytest.raises(AssertionError, match='quadratic product in a position compiled as affine'):
        compiler().expression(program.Multiply(program.Variable('p'), program.Variable('p')), 'test')


# ---------------------------------------------------------------------------
# shape operators — each rewrites exactly one dim column
# ---------------------------------------------------------------------------


def test_sum_drops_the_dim_it_sums_over_without_aggregating():
    """The aggregate lives in the terminal assembly, not in the piece."""
    compiled = compiler().expression(program.Sum(program.Variable('p'), ('generator',)), 'test')
    piece = compiled.terms[0]
    assert piece.dims == ('snapshot',)
    assert columns(piece.frame) == ['snapshot', 'var_label', 'coeff']
    assert 'AGGREGATE' not in query(piece.frame)


def masked_compiler() -> Compiler:
    """A compiler over two masked variables, since a restriction only crosses between pieces."""
    over = ('snapshot', 'generator')
    where = program.Mask(program.ParameterComparison('available', '>', 0.0, ('generator',)))
    masked = program.Program(
        parameters=PROGRAM.parameters,
        variables={
            'p': program.VariableDeclaration(over, where=where),
            'q': program.VariableDeclaration(over, where=where),
        },
        constraints={},
        objective=PROGRAM.objective,
        dimensions=PROGRAM.dimensions,
    )
    frames = dict(VARIABLES, q=VARIABLES['p'])
    return Compiler(Scope(masked, attached(), frames))


def test_a_reduction_carries_absence_between_pieces_and_not_into_the_one_it_came_from():
    """`sum(p + q)` sums where each exists, and neither is checked against itself.

    Restricting a piece by its own coordinates returns the rows it was
    given, so a lone masked term costs no join.
    """
    both = masked_compiler().expression(
        program.Sum(program.Add(program.Variable('p'), program.Variable('q')), ('generator',)), 'test'
    )
    assert [joins(t.frame) for t in both.terms] == [1, 1]
    assert all('SEMI JOIN' in query(t.frame) for t in both.terms)

    alone = masked_compiler().expression(program.Sum(program.Variable('p'), ('generator',)), 'test')
    assert joins(alone.terms[0].frame) == 0


def test_a_reduction_restricts_by_existence_and_does_not_deduplicate():
    """A semi-join asks whether a key occurs, so a distinct on its right changes no row."""
    compiled = masked_compiler().expression(
        program.Sum(program.Add(program.Variable('p'), program.Variable('q')), ('generator',)), 'test'
    )
    assert 'UNIQUE' not in query(compiled.terms[0].frame)


def test_sum_over_an_absent_dim_scales_by_that_dims_cardinality():
    """Linopy parity: summing a snapshot-only term over `generator` repeats it."""
    inner = program.Sum(program.Variable('p'), ('generator',))
    compiled = compiler().expression(program.Sum(inner, ('generator',)), 'test')
    assert '3' in query(compiled.terms[0].frame)


def test_sum_swaps_the_source_dim_for_the_target_and_emits_no_aggregate():
    node = program.GroupSum(program.Variable('p'), GEN_BUS_DIRECTION)
    piece = compiler().expression(node, 'test').terms[0]
    assert piece.dims == ('snapshot', 'bus')
    assert columns(piece.frame) == ['snapshot', 'bus', 'var_label', 'coeff']
    assert 'AGGREGATE' not in query(piece.frame)
    assert joins(piece.frame) == 1


def test_translate_keeps_its_dims_and_joins_the_dim_table_twice():
    """Bounded halo: a row at ord *o* lands at ord *o + by*, no window."""
    piece = (
        compiler()
        .expression(program.Translate(program.Variable('p'), 'snapshot', offset=1, wrap=True), 'test')
        .terms[0]
    )
    assert piece.dims == ('snapshot', 'generator')
    assert columns(piece.frame) == ['generator', 'snapshot', 'var_label', 'coeff']
    assert joins(piece.frame) == 2
    assert 'OVER' not in query(piece.frame)


def test_wrapping_is_modulo_and_acyclic_is_not():
    cyclic = (
        compiler().expression(program.Translate(program.Variable('p'), 'snapshot', offset=1, wrap=True), 't').terms[0]
    )
    acyclic = (
        compiler().expression(program.Translate(program.Variable('p'), 'snapshot', offset=1, wrap=False), 't').terms[0]
    )
    assert '%' in query(cyclic.frame)
    assert '%' not in query(acyclic.frame)


def test_a_shape_operator_along_a_dim_the_expression_lacks_is_refused():
    with pytest.raises(SpecsolveError, match='shift'):
        compiler().expression(program.Translate(program.Parameter('cost'), 'snapshot', offset=1, wrap=True), 'test')


# ---------------------------------------------------------------------------
# predicates
# ---------------------------------------------------------------------------


def test_a_dimension_comparison_filters_a_column_already_in_the_frame():
    """Pointwise, and free: no table is read to decide it."""
    frame = masked(compiler().scope, ('snapshot',), program.Mask(program.DimensionComparison('snapshot', '>', 0)))
    text = query(frame)
    assert 'FILTER' in text
    assert 'JOIN' not in text


def test_a_parameter_predicate_needs_a_join():
    frame = masked(
        compiler().scope, ('generator',), program.Mask(program.ParameterDefined('available', ('generator',)))
    )
    text = query(frame)
    assert 'JOIN' in text
    assert 'FILTER' in text


def test_a_name_the_mask_is_certain_of_is_inner_joined():
    """The rows a left join would keep here are rows the filter then drops."""
    text = query(
        masked(compiler().scope, ('generator',), program.Mask(program.ParameterDefined('available', ('generator',))))
    )
    assert 'INNER JOIN' in text
    assert 'LEFT JOIN' not in text


def test_the_same_predicate_under_an_or_is_left_joined_again():
    """Under an ``Or`` a missing value can make the mask true, so no row is dropped early."""
    where = program.Mask(
        program.Or(
            program.ParameterDefined('available', ('generator',)),
            program.DimensionComparison('generator', '==', 'g'),
        )
    )
    text = query(masked(compiler().scope, ('generator',), where))
    assert 'LEFT JOIN' in text
    assert 'INNER JOIN' not in text


def test_what_a_bare_name_asks_is_decided_by_its_declaration():
    """Three readings, and the declared dtype picks — not the column that turned up."""
    numeric = query(
        masked(compiler().scope, ('generator',), program.Mask(program.ParameterDefined('available', ('generator',))))
    )
    boolean = query(
        masked(
            compiler({'available': 'bool'}).scope,
            ('generator',),
            program.Mask(program.ParameterDefined('available', ('generator',))),
        )
    )
    text = query(
        masked(
            compiler({'available': 'str'}).scope,
            ('generator',),
            program.Mask(program.ParameterDefined('available', ('generator',))),
        )
    )

    assert 'is_finite' in numeric, 'a number has to be finite as well as present'
    assert 'is_finite' not in boolean, 'a boolean is its own answer'
    assert 'is_finite' not in text, 'and a string is defined wherever it has a row'
    assert 'is_not_null' in text


# ---------------------------------------------------------------------------
# frames and bounds
# ---------------------------------------------------------------------------


def test_a_frame_cross_joins_its_dim_tables_and_carries_their_ordinals():
    frame = masked(compiler().scope, ('snapshot', 'generator'), None)
    assert columns(frame) == ['snapshot', '__ord snapshot__', 'generator', '__ord generator__']
    assert 'CROSS' in query(frame)


def test_an_unmasked_frame_has_nothing_to_filter():
    assert 'FILTER' not in query(masked(compiler().scope, ('snapshot', 'generator'), None))


def test_a_mask_reading_part_of_the_frame_restricts_by_semi_join():
    """The predicate is evaluated over the product of the dims it reads, and the full product semi-joined."""
    frame = masked(
        compiler().scope,
        ('snapshot', 'generator'),
        program.Mask(program.ParameterDefined('available', ('generator',))),
    )
    assert 'SEMI JOIN' in query(frame)


def test_a_mask_reading_every_dim_filters_instead():
    """A full-width truth set is as wide as the product itself, so a filter replaces the semi-join."""
    where = program.Mask(
        program.And(
            program.ParameterDefined('load', ('snapshot',)),
            program.ParameterDefined('available', ('generator',)),
        )
    )
    assert 'SEMI JOIN' not in query(masked(compiler().scope, ('snapshot', 'generator'), where))


def test_a_parameter_bound_joins_on_the_variable_frame():
    variable = program.VariableDeclaration(('snapshot', 'generator'), upper=program.Parameter('cost'))
    bounded = compiler().bounds(VARIABLES['p'].frame, 'p', variable)
    assert {'lb', 'ub'} <= set(columns(bounded))
    assert joins(bounded) == 1


def test_a_constant_bound_needs_no_join_at_all():
    bounded = compiler().bounds(VARIABLES['p'].frame, 'p', PROGRAM.variables['p'])
    assert {'lb', 'ub'} <= set(columns(bounded))
    assert joins(bounded) == 0


def test_a_side_the_program_leaves_open_is_an_infinite_column():
    """A program says an open side is `None`, and a relational sink reads an infinity in `lb` and `ub`."""
    variable = program.VariableDeclaration(('snapshot', 'generator'), lower=None, upper=None)
    frame = pl.LazyFrame({'snapshot': [0], 'generator': ['a'], 'var_label': [0]})
    bounded = compiler().bounds(frame, 'p', variable).select('lb', 'ub').collect()
    assert bounded.row(0) == (float('-inf'), float('inf')), 'an open side is the infinity on that side'


def test_a_zero_edge_writes_its_rows_like_any_other_fill():
    """`edge=0` over a constant leaves a row, not a gap.

    The presence branch counts a filled slot as present, so the frame has a
    row for it. Over a term there is nothing to write: the vacated slot
    contributes no term.
    """
    snapshots = pl.LazyFrame({'val': [0, 1, 2], 'ord': [0, 1, 2]})
    sources = AttachedSources(
        parameters={'load': pl.LazyFrame({'snapshot': [0, 1, 2], 'value': [10.0, 20.0, 30.0]})},
        dimensions={'snapshot': snapshots},
        relations={},
        cardinality={'snapshot': 3},
        parameter_rows={'load': 3},
        consecutive={'snapshot': 0},
    )
    q = Compiler(Scope(PROGRAM, sources, VARIABLES))
    shifted = program.Translate(program.Parameter('load'), 'snapshot', 1, wrap=False, fill=0.0)

    rows = q.expression(shifted, 'test').consts[0].frame.collect().sort('snapshot')
    assert rows['snapshot'].to_list() == [0, 1, 2], 'the vacated snapshot has no row, so its zero is invisible'
    assert rows['cval'].to_list() == [0.0, 10.0, 20.0], 'the fill is the value at the vacated slot'

    bare = program.Translate(program.Parameter('load'), 'snapshot', 1, wrap=False, fill=None)
    vacated = q.expression(bare, 'test').consts[0].frame.collect().sort('snapshot')
    assert vacated['snapshot'].to_list() == [1, 2], 'a bare shift vacates, and that is a gap on purpose'
