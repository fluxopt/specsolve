"""The relational lane as such: matrix assembly, labels, and the solver hand-off.

A repeated cell collapsing, a column being positional, a bound attached by
position rather than joined, a solution read back in label order: none is a
construct a modeller writes, and each is a way this lane can be wrong while
every objective still agrees.
"""

from __future__ import annotations

import contextlib
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from mathspec import Spec
from mathspec.program import (
    Add,
    Constant,
    ConstraintDeclaration,
    DimensionDeclaration,
    Direction,
    GroupSum,
    Mask,
    Multiply,
    Negate,
    ObjectiveDeclaration,
    Parameter,
    ParameterComparison,
    ParameterDeclaration,
    Program,
    RelationDeclaration,
    Sum,
    Variable,
    VariableDeclaration,
)

import specsolve as sps
from specsolve.errors import DataError, LanguageError, SpecsolveError
from specsolve.relational.engines.polars.compiler import PolarsCompiler
from specsolve.relational.engines.polars.engine import PolarsEngine
from specsolve.relational.engines.polars.scope import Scope
from specsolve.relational.sinks import SOLVERS
from specsolve.relational.sinks.handoff import ranges
from specsolve.relational.sinks.solvers.highs import Highs
from specsolve.sources import tidy_sources
from tests.conftest import SOLVER_VECTOR_LOAD, SOLVER_VECTOR_SPEC, by_coord, override, solve_written_file
from tests.differential import RTOL, differential
from tests.oracle import linopy, pd, specsolve_linopy, transport_linopy_objective, xr

# ---------------------------------------------------------------------------
# model 1: dispatch (the spec example)
# ---------------------------------------------------------------------------


@pytest.fixture
def dispatch_data():
    """The spec's dispatch instance, with one generator retired to `p_max = 0`.

    That row is what the model's `where` masks out.
    """
    rng = np.random.default_rng(7)
    n_s, n_g = 40, 6
    gens = pd.DataFrame(
        {
            'generator': [f'g{i}' for i in range(n_g)],
            'p_max': rng.uniform(50, 200, n_g).round(3),
            'cost': rng.uniform(5, 100, n_g).round(3),
        }
    )
    gens.loc[1, 'p_max'] = 0.0
    load = pd.DataFrame(
        {
            'snapshot': np.arange(n_s),
            'value': (rng.uniform(0.2, 0.7, n_s) * gens['p_max'].sum()).round(3),
        }
    )
    return gens, load


def dispatch_program() -> Program:
    return Program(
        parameters={
            'p_max': ParameterDeclaration(('generator',)),
            'cost': ParameterDeclaration(('generator',)),
            'load': ParameterDeclaration(('snapshot',)),
        },
        variables={
            'p': VariableDeclaration(
                ('snapshot', 'generator'),
                where=Mask(ParameterComparison('p_max', '>', 0, ('generator',))),
                lower=Constant(0.0),
                upper=Parameter('p_max'),
            )
        },
        constraints={
            'power_balance': ConstraintDeclaration(
                ('snapshot',),
                lhs=Sum(Variable('p'), over=('generator',)),
                sense='==',
                rhs=Parameter('load'),
            )
        },
        objective=ObjectiveDeclaration(
            'minimize', Sum(Multiply(Variable('p'), Parameter('cost')), over=('generator', 'snapshot'))
        ),
        dimensions={'snapshot': DimensionDeclaration(dtype='int'), 'generator': DimensionDeclaration()},
    )


def dispatch_sources(gens: pd.DataFrame, load: pd.DataFrame) -> dict:
    return {
        'p_max': gens[['generator', 'p_max']].rename(columns={'p_max': 'value'}),
        'cost': gens[['generator', 'cost']].rename(columns={'cost': 'value'}),
        'load': load,
        'snapshot': load[['snapshot']],
        'generator': gens[['generator']],
    }


#: The smallest model with a parameter on the constant side: `x >= rhs` over
#: two coordinates, `x` minimized and unbounded above.
RHS_SPEC = {
    'dimensions': {'i': {'dtype': 'int'}},
    'parameters': {'rhs': {'dims': ['i']}},
    'variables': {'x': {'dims': ['i'], 'bounds': {'lower': 0}}},  # no upper: +inf
    'constraints': {'c': {'dims': ['i'], 'expression': 'x >= rhs'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x, over=i)'},
}


#: The smallest string-dim model: `x` bounded by `cap` over `node`, forced to
#: it by `k`, minimized. The string-dimension tests below each override one
#: declaration of it.
NODE_CAP_SPEC = {
    'dimensions': {'node': {'dtype': 'str'}},
    'parameters': {'cap': {'dims': ['node']}},
    'variables': {'x': {'dims': ['node'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'k': {'dims': ['node'], 'expression': 'x >= cap'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x, over=node)'},
}


LABEL_SPEC = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'cost': {'dims': ['f']}, 'cap': {'dims': ['f']}},
    'variables': {'x': {'dims': ['f'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'k': {'dims': ['f'], 'expression': 'x <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost)'},
}


_CAP = pl.DataFrame({'f': ['a', 'b'], 'value': [5.0, 5.0]})


#: A constant part beside a term, under each operator that acts along a dim the
#: constant does not carry. `check` accepts every one, lowering passes it,
#: and the linopy lane builds and solves it — so the file is sayable and only this
#: lane is short (#1137).
CONSTANT_BESIDE_A_TERM = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'k': {'dims': ['t']}, 'd': {'dims': []}, 'load': {'dims': []}},
    'variables': {'x': {'dims': ['t'], 'where': 't != 2', 'bounds': {'lower': 0}}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x, over=t)'},
}


#: Every operator that acts along a dim, with the objective the linopy lane
#: reaches from the file *as written* — which is the number the rewrite owes.
CONSTANT_BESIDE_A_TERM_CASES = [
    ('sum(x * k + d, over=t) >= load', 8.0, 'sum-over'),
    ('sum(sum(x * k + d, by=r_of, over=t, into=r), over=r) >= load', 8.0, 'sum-by'),
    ('sum(shift(x * k + d, along=t, offset=1, edge=0), over=t) >= load', 8.0, 'shift'),
    ('sum(sum_back(x * k + d, along=t, window=2), over=t) >= load', 3.0, 'sum-back'),
]


#: Empty since #1142; kept as the place a future one-operator hole goes.
REWRITE_IS_BROKEN_ON: dict[str, pytest.MarkDecorator] = {}


def dispatch_linopy_objective(gens: pd.DataFrame, load: pd.DataFrame) -> float:
    gi = gens.set_index('generator')
    li = load.set_index('snapshot')['value']
    p_max = xr.DataArray.from_series(gi['p_max'])
    cost = xr.DataArray.from_series(gi['cost'])
    load_da = xr.DataArray.from_series(li)

    m = linopy.Model()
    mask = (p_max > 0).broadcast_like(load_da * p_max)
    p = m.add_variables(lower=0, upper=p_max, coords=[li.index, gi.index], name='p', mask=mask)
    m.add_constraints(p.sum('generator') == load_da, name='power_balance')
    m.add_objective((p * cost).sum())
    m.solve(solver_name='highs', output_flag=False)
    return float(m.objective.value)


#: The transport network's wiring, as the three maps a nodal balance reads.
WIRING = {
    'gen_bus': RelationDeclaration((('generator', 'generator'), ('bus', 'bus')), ('generator',)),
    'from': RelationDeclaration((('line', 'line'), ('bus', 'bus')), ('line',)),
    'to': RelationDeclaration((('line', 'line'), ('bus', 'bus')), ('line',)),
}


def _onto_bus(name: str) -> Direction:
    """Relation *name* read the way a nodal balance reads it: out of its key, onto the bus."""
    relation = WIRING[name]
    return Direction(name, relation, relation.key, relation.values, ())


def transport_program() -> Program:
    injection = Add(
        Add(
            GroupSum(Variable('p'), _onto_bus('gen_bus')),
            GroupSum(Variable('f'), _onto_bus('to')),
        ),
        Negate(GroupSum(Variable('f'), _onto_bus('from'))),
    )
    return Program(
        parameters={
            'p_max': ParameterDeclaration(('generator',)),
            'cost': ParameterDeclaration(('generator',)),
            'cap': ParameterDeclaration(('line',)),
            'cap_low': ParameterDeclaration(('line',)),
            'load': ParameterDeclaration(('snapshot', 'bus')),
        },
        variables={
            'p': VariableDeclaration(
                ('snapshot', 'generator'),
                lower=Constant(0.0),
                upper=Parameter('p_max'),
            ),
            'f': VariableDeclaration(
                ('snapshot', 'line'),
                lower=Parameter('cap_low'),
                upper=Parameter('cap'),
            ),
        },
        constraints={
            'balance': ConstraintDeclaration(
                ('snapshot', 'bus'),
                lhs=injection,
                sense='==',
                rhs=Parameter('load'),
            )
        },
        objective=ObjectiveDeclaration(
            'minimize', Sum(Multiply(Variable('p'), Parameter('cost')), over=('generator', 'snapshot'))
        ),
        dimensions={
            'snapshot': DimensionDeclaration(dtype='int'),
            'bus': DimensionDeclaration(),
            'generator': DimensionDeclaration(),
            'line': DimensionDeclaration(),
        },
        relations=WIRING,
    )


def transport_sources(gens, lines, load) -> dict:
    """The transport instance as tidy sources.

    Below the front door: `PolarsEngine.build` takes a *program* and the frames
    a built model reads, which is the same shape `tidy_sources` hands over — an
    index of labels, and each map as its own two-column relation.
    """
    return {
        'p_max': gens[['generator', 'p_max']].rename(columns={'p_max': 'value'}),
        'cost': gens[['generator', 'cost']].rename(columns={'cost': 'value'}),
        'cap': lines[['line', 'cap']].rename(columns={'cap': 'value'}),
        'cap_low': lines[['line']].assign(value=-lines['cap']),
        'load': load,
        'snapshot': load[['snapshot']],
        'bus': load[['bus']],
        'generator': gens[['generator']],
        'line': lines[['line']],
        'gen_bus': gens[['generator', 'bus']],
        'from': lines[['line', 'from_bus']].rename(columns={'from_bus': 'bus'}),
        'to': lines[['line', 'to_bus']].rename(columns={'to_bus': 'bus'}),
    }


class TestTwoModelsRoundTrip:
    """Two whole models, each built three ways, required to reach one objective.

    The engine into ``highs``, the engine into an LP file HiGHS re-reads, and
    the linopy build as the oracle.
    """

    def test_dispatch_roundtrip(self, dispatch_data, tmp_path):
        """Solver and LP file agree with the linopy oracle, down to the dispatch."""
        gens, load = dispatch_data
        oracle = dispatch_linopy_objective(gens, load)

        with PolarsEngine() as engine:
            engine.build(dispatch_program(), tidy_sources(dispatch_program(), dispatch_sources(gens, load)))

            result = engine.solve()
            assert result.is_ok
            assert result.objective == pytest.approx(oracle, rel=RTOL)

            lp = tmp_path / 'dispatch.lp'
            engine.write(lp)
            assert solve_written_file(lp) == pytest.approx(oracle, rel=RTOL)

            primal = result.to_pandas('p')
            n_active = int((gens['p_max'] > 0).sum())
            assert len(primal) == n_active * len(load), 'a masked variable row is absent, not zero'
            assert set(primal.columns) == {'snapshot', 'generator', 'value'}, 'primal joins back to the coords'
            balance = primal.groupby('snapshot')['value'].sum()
            expected = load.set_index('snapshot')['value']
            assert np.allclose(balance.sort_index(), expected.sort_index()), 'dispatch meets load in every snapshot'

    def test_transport_roundtrip(self, transport_data, tmp_path):
        """Solver and LP file agree with the linopy oracle."""
        gens, lines, load = transport_data
        oracle = transport_linopy_objective(gens, lines, load)
        assert np.isfinite(oracle), 'oracle model must be feasible'

        with PolarsEngine() as engine:
            engine.build(transport_program(), tidy_sources(transport_program(), transport_sources(gens, lines, load)))

            result = engine.solve()
            assert result.is_ok
            assert result.objective == pytest.approx(oracle, rel=RTOL)

            lp = tmp_path / 'transport.lp'
            engine.write(lp)
            assert solve_written_file(lp) == pytest.approx(oracle, rel=RTOL)

            primal_f = result.to_pandas('f')
            caps = lines.set_index('line')['cap']
            limits = primal_f['line'].map(caps)
            assert (primal_f['value'].abs() <= limits + 1e-6).all()


#: A scalar parameter used in a bound and in the objective — the two places a
#: silent row multiplication is least visible.
SCALAR_SPEC = {
    'dimensions': {'i': {'dtype': 'int'}},
    'parameters': {'s': {'dims': []}},
    'variables': {'x': {'dims': ['i'], 'bounds': {'lower': 0, 'upper': 's'}}},
    'constraints': {'floor': {'dims': ['i'], 'expression': 'x >= 1'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x * s, over=i)'},
}


class TestWhatBindRefusesAndWhatItTakes:
    """Settled before a solver is opened, so none of these reaches an objective.

    Each asserts a refusal or an acceptance at attach, where the only inputs are
    the model and the shape of the data.
    """

    def test_missing_source_rejected(self, dispatch_data):
        gens, load = dispatch_data
        sources = dispatch_sources(gens, load)
        del sources['cost']
        with pytest.raises(DataError, match="no data provided for parameter 'cost'"):
            tidy_sources(dispatch_program(), sources)

    def test_a_bound_parameter_short_of_a_coordinate_is_refused_with_the_count(self):
        """The refusal nothing else in the suite reaches.

        A bound whose *value* is null is refused at attach, so the only way to
        a null bound is a parameter with no row for a coordinate the variable
        has a column at — which attach cannot see, absence being ordinary
        everywhere else. The count is asserted because it is what the message
        exists to carry: it is read off the two bound columns before the frame
        that carries them is filtered, and a probe that looked at one of them
        would report every model as sound.
        """
        spec = {
            'dimensions': {'i': {'dtype': 'int'}},
            'parameters': {'cap': {'dims': ['i']}},
            'variables': {'x': {'dims': ['i'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
            'constraints': {'c': {'dims': ['i'], 'expression': 'x >= 0'}},
            'objective': {'sense': 'minimize', 'expression': 'sum(x, over=i)'},
        }
        sources = {'i': [0, 1, 2], 'cap': pl.DataFrame({'i': [0, 1], 'value': [5.0, 6.0]})}
        with pytest.raises(DataError, match="variable 'x': 1 rows have NULL bounds"):
            sps.build(spec, sources)

    @pytest.mark.parametrize('rows', [2, 0])
    def test_a_dimensionless_parameter_must_be_one_row(self, rows):
        """No dims means one value broadcast everywhere, so a second row is refused (#166).

        A dimensionless parameter is broadcast by joining on nothing, which is
        right for one row and a silent row multiplication for two — duplicate
        columns for one variable in a bound, duplicate mask rows in a where.
        """
        data = {'s': pl.DataFrame({'value': [1.0] * rows}, schema={'value': pl.Float64})}
        with pytest.raises(DataError, match=f"parameter 's' .* its source has {rows} rows"):
            sps.build(SCALAR_SPEC, data)

    def test_a_dimensionless_parameter_of_one_row_still_builds(self):
        """The control: the shape the check exists to let through."""
        data = {'i': [0, 1], 's': pl.DataFrame({'value': [10.0]})}
        with sps.solve(SCALAR_SPEC, data) as result:
            assert result.objective == pytest.approx(20.0), 'x == 1 at both coordinates of i, times s'

    def test_a_dimension_named_n_is_still_a_legal_dimension(self):
        """No check may claim a legal dim name for a column of its own.

        The duplicate-row check counts rows beside the dims it grouped by, so its
        count column must not collide with a dim named `n`.
        """
        spec = {
            'dimensions': {'n': {'dtype': 'int'}},
            'parameters': {'cost': {'dims': ['n']}},
            'variables': {'x': {'dims': ['n'], 'bounds': {'lower': 0, 'upper': 5}}},
            'constraints': {'c': {'dims': ['n'], 'expression': 'x <= 5'}},
            'objective': {'sense': 'maximize', 'expression': 'sum(x * cost)'},
        }
        with sps.solve(spec, {'n': [1, 2], 'cost': pl.DataFrame({'n': [1, 2], 'value': [1.0, 2.0]})}) as result:
            assert result.objective == pytest.approx(15.0)

        # the check the column belongs to still catches its own case
        doubled = {'cost': pl.DataFrame({'n': [1, 1, 2], 'value': [1.0, 9.0, 2.0]})}
        with pytest.raises(DataError, match="parameter 'cost'"):
            sps.build(spec, doubled)

    def test_an_awkward_path_is_a_value_not_syntax(self, tmp_path):
        """Paths come from the calling program, so no language rule constrains them.

        ``o'brien`` is a legal directory name, and a quote in one must be as
        uninteresting as a quote in a label. Every path-carrying sink and source is
        exercised here: a parquet source, an explicit index source, the LP writer,
        and the parquet sink.
        """
        odd = tmp_path / "o'brien"
        odd.mkdir()
        pl.DataFrame({'snapshot': [0, 1], 'value': [1.0, 2.0]}).write_parquet(odd / 'load.parquet')
        pl.DataFrame({'snapshot': [0, 1]}).write_parquet(odd / 'index.parquet')

        spec = {
            'dimensions': {'snapshot': {'dtype': 'int'}},
            'parameters': {'load': {'dims': ['snapshot']}},
            'variables': {'p': {'dims': ['snapshot'], 'bounds': {'lower': 0}}},
            'constraints': {'meet': {'dims': ['snapshot'], 'expression': 'p >= load'}},
            'objective': {'sense': 'minimize', 'expression': 'sum(p, over=snapshot)'},
        }
        sources = {'load': str(odd / 'load.parquet'), 'snapshot': str(odd / 'index.parquet')}

        sps.write(spec, sources, odd / 'model.lp')
        result = sps.solve(spec, sources)
        assert result.objective == pytest.approx(3.0)
        assert (result.save(odd / 'solution') / 'primal' / 'p.parquet').exists()

    def test_a_dictionary_encoded_source_column_binds_like_a_plain_one(self):
        """A `Categorical` dim column is a source encoding, not a different model.

        Each writer's dictionary is its own, and polars will not join dictionaries
        that disagree, so attaching re-encodes every dim column into the one
        dictionary the dimension declares.
        """
        encoded = pl.DataFrame({'node': ['a', 'b'], 'value': [3.0, 4.0]}).with_columns(
            pl.col('node').cast(pl.Categorical)
        )
        plain = pl.DataFrame({'node': ['a', 'b'], 'value': [3.0, 4.0]})

        with sps.build(NODE_CAP_SPEC, {'node': ['a', 'b'], 'cap': encoded}) as model:
            from_encoded = model.solve().objective
        with sps.build(NODE_CAP_SPEC, {'node': ['a', 'b'], 'cap': plain}) as model:
            from_plain = model.solve().objective

        assert from_encoded == pytest.approx(7.0)
        assert from_encoded == pytest.approx(from_plain), 'the encoding changed the model'

    def test_a_dimension_with_no_index_is_refused_rather_than_taking_one_from_the_data(self):
        """Which is what makes the stranger above a stranger.

        Labels read out of the parameters would *be* the definition, so a mistyped
        one could not be told from a new one — the check above would have nothing
        left to ask.
        """
        data = {'cost': pl.DataFrame({'f': ['a', 'b'], 'value': [1.0, 2.0]}), 'cap': _CAP}

        with pytest.raises(DataError, match="dimension 'f' has no index"):
            sps.solve(LABEL_SPEC, data)

        supplied = {**data, 'f': pl.DataFrame({'f': ['a', 'b']})}
        assert sps.solve(LABEL_SPEC, supplied).objective == pytest.approx(15.0)


#: A `line` whose two endpoints are *both* multi-valued for one label.
TWO_BAD_COORDS_SPEC = {
    'dimensions': {'bus': {'dtype': 'str'}, 'line': {}},
    'relations': {
        'from': {'key': 'line', 'values': 'bus'},
        'to': {'key': 'line', 'values': 'bus'},
    },
    'parameters': {'cap': {'dims': ['line']}},
    'variables': {'f': {'dims': ['line'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'k': {'dims': ['line'], 'expression': 'f <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(f)'},
}


class TestTheLabelSpace:
    """Which rows exist, and in what order — decided by the declaration, never the data.

    The shared claim is that order is a property of the *model*: a mask does not
    reorder what it keeps, a string dimension keeps its declared order rather
    than a bytewise one, and a label the dimension does not have is a stranger
    rather than a new member. A lane that read order off the rows would pass
    every objective assertion in this file and fail here.
    """

    def test_a_masked_variable_is_labelled_in_declaration_order(self):
        """A label is the solver's own column index, so its order is the contract.

        Labels are dense ``0..n-1`` over the coordinates the mask leaves, assigned
        row-major on the dims' declared ordinals. An off-by-one here is a different
        model rather than a slower one, so it is asserted against the order spelled
        out by hand rather than against whatever the engine produced.
        """
        spec = {
            'dimensions': {
                'snapshot': {'dtype': 'int'},
                'node': {'dtype': 'str'},
                'tech': {'dtype': 'str'},
            },
            'parameters': {'cap': {'dims': ['node', 'tech']}, 'load': {'dims': ['snapshot']}},
            'variables': {
                'p': {
                    'dims': ['snapshot', 'node', 'tech'],
                    'where': 'cap > 0',
                    'bounds': {'lower': 0, 'upper': 'cap'},
                }
            },
            'constraints': {
                'balance': {
                    'dims': ['snapshot', 'node'],
                    'expression': 'sum(p, over=tech) >= load',
                }
            },
            'objective': {
                'sense': 'minimize',
                'expression': 'sum(sum(sum(p, over=tech), over=node), over=snapshot)',
            },
        }
        caps = [
            {'node': n, 'tech': t, 'value': 0.0 if (n, t) in {('a', 'gas'), ('c', 'coal'), ('b', 'hydro')} else 10.0}
            for n in ['a', 'b', 'c']
            for t in ['wind', 'gas', 'coal', 'hydro']
        ]
        sources = {
            'snapshot': list(range(5)),
            'node': ['a', 'b', 'c'],
            'tech': ['wind', 'gas', 'coal', 'hydro'],
            'cap': pl.DataFrame(caps),
            'load': pl.DataFrame({'snapshot': list(range(5)), 'value': [1.0] * 5}),
        }

        zero = {('a', 'gas'), ('c', 'coal'), ('b', 'hydro')}
        expected = [
            (s, n, t)
            for s in range(5)
            for n in ['a', 'b', 'c']
            for t in ['wind', 'gas', 'coal', 'hydro']
            if (n, t) not in zero
        ]

        with sps.build(spec, sources) as model:
            labelled = model._engine._model.variables['p'].frame.collect()

        assert labelled['var_label'].to_list() == list(range(len(expected))), 'labels must be dense and ascending'
        assert list(labelled.select('snapshot', 'node', 'tech').iter_rows()) == expected

    def test_a_string_dimension_is_enum_encoded_up_to_the_read_back(self):
        """A string dim is an ``Enum`` over its labels in declaration order for the
        whole build — and plain ``String`` the moment it is handed back.

        A caller joins the result against their own data, which an ``Enum``
        refuses. Declaration order survives the cast because it is the row order,
        and `to_pandas` puts the ordered categories back for the bridge out.
        """
        cap = pl.DataFrame({'node': ['a', 'b', 'c'], 'value': [1.0, 2.0, 3.0]})
        declared = pl.Enum(['c', 'a', 'b'])

        with sps.build(NODE_CAP_SPEC, {'node': ['c', 'a', 'b'], 'cap': cap}) as model:
            assert model._engine._model.variables['x'].frame.collect_schema()['node'] == declared
            primal = model.solve().primal('x')

        assert primal.schema['node'] == pl.String, 'what leaves is what a caller can join against'
        assert primal['node'].to_list() == ['c', 'a', 'b'], 'read-back follows label order, not source order'
        assert primal.join(cap, on='node').height == 3, "the caller's own frame is String, and it joins"

    def test_a_where_orders_string_labels_bytewise_not_by_declaration(self):
        """`node >= 'b'` keeps {b, c} whatever order the labels were declared in
        (the where-string rules). Declared c, a, b so the Enum's declaration order would keep {b} alone."""
        spec = override(
            NODE_CAP_SPEC,
            **{'variables.x.where': "node >= 'b'", 'constraints.k.where': "node >= 'b'"},
        )
        cap = pl.DataFrame({'node': ['a', 'b', 'c'], 'value': [1.0, 2.0, 3.0]})

        with sps.build(spec, {'node': ['c', 'a', 'b'], 'cap': cap}) as model:
            assert sorted(model._engine._model.variables['x'].frame.collect()['node'].to_list()) == ['b', 'c']
            assert model.solve().objective == pytest.approx(5.0)

    def test_a_where_naming_an_undeclared_label_masks_nothing_in(self):
        """A quoted label the dimension does not carry masks everything out (the
        where-string rules); the comparison is in String space, since the Enum
        would refuse the stranger."""
        spec = override(NODE_CAP_SPEC, **{'constraints.k.where': "node == 'zzz'"})
        cap = pl.DataFrame({'node': ['a', 'b'], 'value': [1.0, 2.0]})

        with sps.build(spec, {'node': ['a', 'b'], 'cap': cap}) as model:
            assert model._engine._model.handoff.rows.height == 0, (
                'the mask holds nowhere, so no constraint row is built'
            )
            assert model.solve().objective == pytest.approx(0.0)

    def test_a_mask_that_removes_nothing_labels_exactly_like_no_mask(self, dispatch_data):
        """A vacuous `where` must not shift a single solver index.

        Labels are the solver's own column numbers, so "the mask removed nothing"
        and "there was no mask" have to produce identical frames rather than merely
        the same row count.
        """
        gens, load = dispatch_data
        gens = gens.assign(p_max=gens['p_max'].where(gens['p_max'] > 0, 1.0))  # nothing left to mask out

        labels = []
        for where in (None, Mask(ParameterComparison('p_max', '>', 0, ('generator',)))):
            base = dispatch_program()
            program = replace(base, variables={'p': replace(base.variables['p'], where=where)})
            with PolarsEngine() as engine:
                engine.build(program, tidy_sources(program, dispatch_sources(gens, load)))
                labels.append(engine._model.variables['p'].frame.collect().sort('var_label'))
        assert labels[0].equals(labels[1])

    def test_a_mask_a_missing_value_can_satisfy_keeps_the_rows_with_no_value(self):
        """`not` and `or` select rows the mask's own join must not have dropped.

        A mask's parameters are joined for, and under a plain conjunction that join
        can be an inner one: a coordinate the parameter has no row for fails the
        conjunct anyway, so keeping it only widens the frame the filter then
        narrows. Under `not` or `or` a missing value is what makes the mask *true*,
        and an inner join would have thrown the row away before the filter could
        say so.
        """
        spec = {
            'dimensions': {'i': {'dtype': 'int'}},
            'parameters': {'a': {'dims': ['i']}, 'b': {'dims': ['i']}},
            'variables': {
                'absent': {'dims': ['i'], 'where': 'not a', 'bounds': {'lower': 0, 'upper': 1}},
                'either': {'dims': ['i'], 'where': 'a > 0 or b > 0', 'bounds': {'lower': 0, 'upper': 1}},
                'both': {'dims': ['i'], 'where': 'a and a > 0', 'bounds': {'lower': 0, 'upper': 1}},
                'mixed': {'dims': ['i'], 'where': 'a and not b', 'bounds': {'lower': 0, 'upper': 1}},
            },
            'objective': {'sense': 'minimize', 'expression': 'sum(absent, over=i)'},
        }
        sources = {
            'i': [0, 1, 2, 3],
            'a': pl.DataFrame({'i': [0, 1], 'value': [5.0, -1.0]}),
            'b': pl.DataFrame({'i': [2], 'value': [7.0]}),
        }
        with sps.build(spec, sources) as model:
            surviving = {
                name: sorted(model._engine._model.variables[name].frame.select('i').collect().to_series().to_list())
                for name in ('absent', 'either', 'both', 'mixed')
            }
        assert surviving == {
            'absent': [2, 3],  # `a` is missing there, which is the whole condition
            'either': [0, 2],  # i=2 has no `a` at all and qualifies on `b`
            'both': [0],
            'mixed': [0, 1],  # `a` is certain, `b` is not
        }

    def test_every_declaration_owns_a_contiguous_run_of_labels(self):
        """What reading a solve back by position rests on.

        A solver vector is positional in the same index the labels are, so a
        declaration's share of it is a slice — but only if its labels are a dense
        run and the runs tile the whole index. Both are how `labels.frame` numbers
        a declaration, and neither is visible from the frames: a block
        that started one late would report a neighbour's numbers under this
        declaration's coordinates, with nothing out of range and nothing null.
        """
        spec = {
            'dimensions': {'i': {'dtype': 'int'}, 'j': {'dtype': 'str'}},
            'parameters': {'cap': {'dims': ['i']}},
            'variables': {
                'x': {'dims': ['i'], 'bounds': {'lower': 0, 'upper': 'cap'}},
                'y': {'dims': ['i', 'j'], 'bounds': {'lower': 0}},
                'z': {'dims': ['j'], 'bounds': {'lower': 0}},
            },
            'constraints': {
                'c1': {'dims': ['i'], 'expression': 'x >= cap'},
                'c2': {'dims': ['i', 'j'], 'expression': 'y >= 0'},
            },
            'objective': {'sense': 'minimize', 'expression': 'sum(x, over=i)'},
        }
        index = {'i': [0, 1, 2], 'j': ['a', 'b']}
        with sps.build(spec, index | {'cap': pl.DataFrame({'i': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})}) as model:
            engine = model._engine
            tables = engine._model.handoff
            for names, total, held, label in (
                (['x', 'y', 'z'], tables.column_count, engine._model.variables, 'var_label'),
                (['c1', 'c2'], tables.row_count, engine._model.constraints, 'row'),
            ):
                at = 0
                for name in names:
                    start, height = held[name].start, held[name].height
                    assert start == at, f'{name} does not start where the previous declaration ended'
                    labels = held[name].frame.select(label).collect().to_series()
                    assert sorted(labels) == list(range(start, start + height)), f'{name} is not a dense run'
                    at += height
                assert at == total, 'the runs do not tile the index'

    def test_a_variable_and_a_constraint_may_share_a_name(self):
        """Columns and rows are numbered independently, so the maps are separate.

        `pypsa_unit_commitment` names both `start_up`, and linopy — the oracle the
        whole corpus is checked against — keeps variables and constraints in
        separate namespaces, so the model is legal and the read-back has to tell
        the two apart. `pad` exists to push the variable's block off the
        constraint's: sharing a start, the wrong block returns plausible numbers
        from the wrong declaration rather than a length error.
        """
        spec = {
            'dimensions': {'i': {'dtype': 'int'}},
            'parameters': {'cap': {'dims': ['i']}},
            'variables': {
                'pad': {'dims': ['i'], 'bounds': {'lower': 0, 'upper': 0}},
                'x': {'dims': ['i'], 'bounds': {'lower': 0}},
            },
            'constraints': {'x': {'dims': ['i'], 'expression': 'x >= cap'}},
            'objective': {'sense': 'minimize', 'expression': 'sum(x, over=i)'},
        }
        with sps.solve(
            spec, {'i': [0, 1, 2], 'cap': pl.DataFrame({'i': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})}
        ) as result:
            assert result.primal('x')['value'].to_list() == pytest.approx([1.0, 2.0, 3.0]), (
                "variable 'x' read back over the constraint of the same name"
            )
            assert result.primal('pad')['value'].to_list() == pytest.approx([0.0, 0.0, 0.0])
            assert result.dual('x')['value'].to_list() == pytest.approx([1.0, 1.0, 1.0]), (
                "constraint 'x' read back over the variable of the same name"
            )

    def test_a_label_the_dimension_does_not_have_is_refused(self):
        """A label the dimension does not have is refused, not read as a zero (#350).

        `b` mistyped as `zz` would leave `b` with no cost row, which reads as a
        zero coefficient: the model would solve, report optimal, and return 5.0
        where 15.0 is right.
        """
        ok = {'cost': pl.DataFrame({'f': ['a', 'b'], 'value': [1.0, 2.0]}), 'cap': _CAP}
        assert sps.solve(LABEL_SPEC, {'f': ['a', 'b'], **ok}).objective == pytest.approx(15.0)

        typo = {'cost': pl.DataFrame({'f': ['a', 'zz'], 'value': [1.0, 2.0]}), 'cap': _CAP}
        with pytest.raises(DataError) as exc:
            sps.solve(LABEL_SPEC, {'f': ['a', 'b'], **typo})
        assert "'zz'" in str(exc.value), 'the refusal must name the offending label'
        assert 'typo' in str(exc.value)

    def test_a_missing_row_is_still_only_sparse(self):
        """The distinction the refusal above rests on. A row that is *absent* is
        ordinary — it reads as a zero coefficient (the data-attachment rules) — and
        only a row that is present and unaddressable is a typo.
        """
        sparse = {'cost': pl.DataFrame({'f': ['a'], 'value': [1.0]}), 'cap': _CAP}
        assert sps.solve(LABEL_SPEC, {'f': ['a', 'b'], **sparse}).objective == pytest.approx(5.0)

    def test_every_multi_valued_coordinate_is_named_at_once(self):
        """Every multi-valued coordinate is named in one refusal (#273).

        The pass is over the relation's rows, so both labels it maps twice arrive
        in one count and are named together.
        """
        data = {
            'cap': pl.DataFrame({'line': ['l1', 'l2'], 'value': [1.0, 1.0]}),
            'line': pl.DataFrame({'line': ['l1', 'l2']}),
            'from': pl.DataFrame({'line': ['l1', 'l1', 'l2', 'l2'], 'bus': ['b1', 'b2', 'b1', 'b2']}),
            'to': pl.DataFrame({'line': ['l1', 'l2'], 'bus': ['b2', 'b1']}),
        }

        with pytest.raises(DataError) as exc:
            sps.solve(TWO_BAD_COORDS_SPEC, data)

        message = str(exc.value)
        assert 'l1' in message and 'l2' in message, f'both offenders must be named; got: {message}'

    @pytest.mark.parametrize(
        ('declared', 'dtype', 'grown'),
        [
            pytest.param({'dtype': 'int'}, pl.Int64, [0, 1], id='a-declared-dtype'),
            pytest.param({}, pl.String, ['a', 'b'], id='the-default-dtype-str'),
        ],
    )
    def test_an_empty_index_keeps_the_dimension_s_declared_dtype(self, declared, dtype, grown):
        """polars infers `Null` from no labels, and a `Null` key joins nothing.

        A dimension whose members a caller appends to between solves starts empty —
        a Benders cut set is the motivating case — and the parameter attached to it
        carries the right dtype from the first call. Only the declaration knows
        what the column should be, and it always answers: the default `dtype`
        (`str`) carries the same weight as a declared one.
        """
        spec = {
            'dimensions': {'cut': declared},
            'parameters': {'c': {'dims': ['cut']}},
            'variables': {'x': {'dims': ['cut'], 'bounds': {'lower': 0}}},
            'objective': {'sense': 'minimize', 'expression': 'sum(x * c)'},
        }
        empty = pl.DataFrame(schema={'cut': dtype, 'value': pl.Float64})
        with sps.build(spec, {'c': empty} | {'cut': []}) as model:
            assert model._engine._model.handoff.column_count == 0

        with sps.build(spec, {'c': pl.DataFrame({'cut': grown, 'value': [1.0, 2.0]})} | {'cut': grown}) as model:
            assert model._engine._model.handoff.column_count == 2

    def test_two_solutions_over_different_members_concatenate(self):
        """An `Enum` column will not concatenate against different categories.

        A sweep holds one frame per slice and joins them on read, so slices that
        attached different members of a dimension still have to concatenate.
        """
        frames = []
        for members in (['a', 'b'], ['a', 'c']):
            cap = pl.DataFrame({'node': members, 'value': [1.0, 2.0]})
            with sps.build(NODE_CAP_SPEC, {'node': pl.DataFrame({'node': members}), 'cap': cap}) as model:
                frames.append(model.solve().primal('x'))

        assert pl.concat(frames).height == 4


def _objective_table(program, sources):
    """`obj` as `{col: coeff}`, plus whether the aggregate was skipped."""
    with PolarsEngine() as engine:
        engine.build(program, tidy_sources(program, sources))
        obj = engine._model.handoff.obj
        return dict(zip(obj['col'].to_list(), obj['coeff'].to_list(), strict=True)), obj.height


def _network(ends: tuple[str, str]) -> tuple[dict, dict]:
    """A balance of flows in minus flows out; `ends` is the second line's endpoints."""
    spec = {
        'dimensions': {
            'snapshot': {'dtype': 'int'},
            'bus': {'dtype': 'str'},
            'line': {},
        },
        'relations': {
            'from': {'key': 'line', 'values': 'bus'},
            'to': {'key': 'line', 'values': 'bus'},
        },
        'parameters': {'cap': {'dims': ['line']}, 'load': {'dims': ['snapshot', 'bus']}},
        'variables': {'f': {'dims': ['snapshot', 'line'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {
            'balance': {
                'dims': ['snapshot', 'bus'],
                'expression': 'sum(f, by=to, over=line, into=bus) - sum(f, by=from, over=line, into=bus) == load',
            }
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(sum(f, over=line), over=snapshot)'},
    }
    sources = {
        'snapshot': [0, 1],
        'bus': ['b0', 'b1'],
        'line': pl.DataFrame({'line': ['l0', 'l1']}),
        'from': pl.DataFrame({'line': ['l0', 'l1'], 'bus': ['b0', ends[0]]}),
        'to': pl.DataFrame({'line': ['l0', 'l1'], 'bus': ['b1', ends[1]]}),
        'cap': pl.DataFrame({'line': ['l0', 'l1'], 'value': [10.0, 10.0]}),
        'load': pl.DataFrame(
            {'snapshot': [0, 0, 1, 1], 'bus': ['b0', 'b1', 'b0', 'b1'], 'value': [0.0, 0.0, 0.0, 0.0]}
        ),
    }
    return spec, sources


PINNED_SPEC = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'relmax': {'dims': ['f']}, 'size_lb': {'dims': ['f']}, 'size_ub': {'dims': ['f']}},
    'variables': {
        'rate': {'dims': ['f'], 'bounds': {'lower': 0, 'upper': 1000}},
        'size': {'dims': ['f'], 'bounds': {'lower': 'size_lb', 'upper': 'size_ub'}},
    },
    'constraints': {'envelope': {'dims': ['f'], 'expression': 'rate - relmax * size <= 0'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(rate, over=f)'},
}


#: `a` spells its sparsity out as zeros rather than as absent rows, which is
#: what a caller handing over a dense table does. `i=1` is zero throughout, so
#: that row asserts `0 >= 10` and is infeasible.
SPELLED_ZEROS_INDEX = {'i': [0, 1], 'j': [0, 1, 2, 3]}

SPELLED_ZEROS_SPEC = {
    'dimensions': {'i': {'dtype': 'int'}, 'j': {'dtype': 'int'}},
    'parameters': {'a': {'dims': ['i', 'j']}},
    'variables': {'x': {'dims': ['j'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'c': {'dims': ['i'], 'expression': 'sum(a * x, over=j) >= 10'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x, over=j)'},
}


def _spelled_zeros(rows: list[list[float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            'i': [i for i, r in enumerate(rows) for _ in r],
            'j': [j for r in rows for j, _ in enumerate(r)],
            'value': [v for r in rows for v in r],
        }
    )


class TestWhatReachesTheSolverAsAnEntry:
    """One coordinate is one entry, however many terms wrote to it.

    These are the ways a matrix can be wrong while the model is right: a cell
    written twice and left duplicated, two reductions colliding, a zero handed
    over as though it were a coefficient. All of them survive an objective
    assertion on the models above.
    """

    def test_a_variable_appearing_twice_in_a_row_is_summed_not_duplicated(self):
        """The case the skipped aggregate must not break.

        `x + 2 * x` is two term fragments landing on one solver column, so the
        assembly has to add them. Its coefficient must be 3, and the row must hold
        one entry for that column rather than two — a solver handed the same
        column twice in one row is entitled to reject the model.
        """
        spec = override(RHS_SPEC, **{'constraints.c.expression': 'x + 2 * x >= rhs'})
        sources = {'i': [0, 1], 'rhs': pl.DataFrame({'i': [0, 1], 'value': [6.0, 9.0]})}
        with sps.build(spec, sources) as model:
            matrix = model._engine._model.handoff.matrix
            assert matrix.height == 2, 'one entry per row, not one per fragment'
            assert sorted(matrix['coeff'].to_list()) == [3.0, 3.0]
            result = model.solve()
        assert result.objective == pytest.approx(5.0), '6/3 + 9/3'

    def test_an_objective_naming_a_variable_twice_sums_its_coefficients(self):
        """Same argument, one dimension down: the objective is a column vector."""
        spec = {
            'dimensions': {'i': {'dtype': 'int'}},
            'parameters': {'lb': {'dims': ['i']}},
            'variables': {'x': {'dims': ['i'], 'bounds': {'lower': 'lb'}}},
            'constraints': {'c': {'dims': ['i'], 'expression': 'x >= lb'}},
            'objective': {'sense': 'minimize', 'expression': 'sum(x) + sum(4 * x)'},
        }
        with sps.build(spec, {'i': [0], 'lb': pl.DataFrame({'i': [0], 'value': [2.0]})}) as model:
            assert model._engine._model.handoff.obj.height == 1
            assert model._engine._model.handoff.obj['coeff'].to_list() == [5.0]
            assert model.solve().objective == pytest.approx(10.0)

    @pytest.mark.parametrize(
        ('expression', 'height', 'coeff'),
        [
            pytest.param('x + y >= rhs', 4, 1.0, id='disjoint-variables-nothing-to-collapse'),
            pytest.param('x + 3 * x >= rhs', 2, 4.0, id='one-variable-twice-summed-onto-one-cell'),
        ],
    )
    def test_the_matrix_collapses_a_repeated_cell_and_leaves_the_rest_alone(self, expression, height, coeff):
        """Both outcomes of the terminal aggregate, on models differing only in overlap.

        Two fragments over disjoint variables repeat nothing and must come out
        untouched; two over the same variable land on one cell and must be summed.
        A matrix holding the same `(row, col)` twice is not a slower model, it is
        one the sinks disagree about.
        """
        spec = override(
            RHS_SPEC,
            **{
                'variables.y': {'dims': ['i'], 'bounds': {'lower': 0}},
                'objective.expression': 'sum(x, over=i) + sum(y, over=i)',
                'constraints.c.expression': expression,
            },
        )
        sources = {'i': [0, 1], 'rhs': pl.DataFrame({'i': [0, 1], 'value': [4.0, 6.0]})}

        with sps.build(spec, sources) as model:
            matrix = model._engine._model.handoff.matrix
            assert matrix.height == height, 'one entry per (row, col) cell the expression reaches'
            assert set(matrix['coeff'].to_list()) == {coeff}

    @pytest.mark.parametrize(
        'ends',
        [
            pytest.param(('b0', 'b1'), id='two-lines-with-distinct-endpoints'),
            pytest.param(('b0', 'b0'), id='a-line-that-leaves-and-arrives-at-one-bus'),
        ],
    )
    def test_two_sums_of_one_variable_collide_only_where_the_coordinates_meet(self, ends):
        """`group_by=to` and `group_by=from` reach one cell exactly on a line to itself.

        Both fragments carry `f`, so counting variables says the aggregate is
        reachable and every nonzero in the model gets sorted to find out. Which
        labels they share is decided by the *line* table — two rows here, forty at
        the `l` rung, against 12.6M nonzeros — so it is asked there.

        The self-loop is the case the collapse exists for: `l1` leaves and arrives
        at `b0`, so `+f - f` lands twice on one cell. A matrix holding the same
        `(row, col)` twice is not a slower model, it is one the sinks disagree
        about — an LP reader sums the pair and a solver handed duplicate entries is
        entitled to do either.
        """
        spec, sources = _network(ends)
        with sps.build(spec, sources) as model:
            program = Spec(**spec).program
            built = model._engine._model
            compiler = PolarsCompiler(Scope(built.program, built.attached, built.variables))
            terms = compiler.expression(next(iter(program.constraints.values())).lhs, 'test').terms
            assert len(terms) == 2

            tables = model._engine._model.handoff
            cells = tables.matrix_block(0, tables.row_count).select('row', 'col')
            assert cells.height == cells.unique().height, 'a cell reached the sinks twice'

    @pytest.mark.parametrize(
        ('expression', 'expected'),
        [
            pytest.param('sum(p * cost)', {0: 2.0, 1: 3.0}, id='one-fragment-one-row-per-column'),
            pytest.param('sum(p * cost) + sum(p * cost)', {0: 4.0, 1: 6.0}, id='two-fragments-summed-onto-one-column'),
        ],
    )
    def test_the_objective_sums_the_coefficients_that_land_on_one_column(self, expression, expected):
        """`sum(p * cost)` is one row per column; the same twice over is two, summed."""
        base = {
            'dimensions': {'i': {'dtype': 'int'}},
            'parameters': {'cost': {'dims': ['i']}, 'lb': {'dims': ['i']}},
            'variables': {'p': {'dims': ['i'], 'bounds': {'lower': 'lb'}}},
            'constraints': {'c': {'dims': ['i'], 'expression': 'p >= lb'}},
            'objective': {'sense': 'minimize', 'expression': expression},
        }
        sources = {
            'i': pl.DataFrame({'i': [0, 1]}),
            'cost': pl.DataFrame({'i': [0, 1], 'value': [2.0, 3.0]}),
            'lb': pl.DataFrame({'i': [0, 1], 'value': [1.0, 1.0]}),
        }
        assert _objective_table(Spec(**base).program, sources) == (expected, 2)

    def test_the_objective_aggregate_survives_a_reduction_that_hides_extra_rows(self):
        """A fragment's dims can match the variable's while its rows do not.

        `sum(q * price, over=generator)` with `q` indexed by snapshot alone reduces
        to dims `('snapshot',)` — exactly `q`'s declaration — but `_sum_fragment`
        *projects*, so the fragment still carries one row per generator. Those rows
        all name one column, and `obj` must hold their sum: the LP file would
        quietly re-sum |generator| rows, while `cols` joined to `obj` in the HiGHS
        sink would hand the solver more columns than the model has.
        """
        spec = {
            'dimensions': {'snapshot': {'dtype': 'int'}, 'generator': {'dtype': 'str'}},
            'parameters': {'price': {'dims': ['snapshot', 'generator']}, 'load': {'dims': ['snapshot']}},
            'variables': {'q': {'dims': ['snapshot'], 'bounds': {'lower': 0, 'upper': 10}}},
            'constraints': {'floor': {'dims': ['snapshot'], 'expression': 'q >= load'}},
            'objective': {'sense': 'minimize', 'expression': 'sum(q * price)'},
        }
        sources = {
            'snapshot': pl.DataFrame({'snapshot': [0, 1]}),
            'generator': pl.DataFrame({'generator': ['g0', 'g1', 'g2']}),
            'price': pl.DataFrame(
                {'snapshot': [0, 0, 0, 1, 1, 1], 'generator': ['g0', 'g1', 'g2'] * 2, 'value': [1.0, 2.0, 3.0] * 2}
            ),
            'load': pl.DataFrame({'snapshot': [0, 1], 'value': [5.0, 5.0]}),
        }
        assert _objective_table(Spec(**spec).program, sources) == ({0: 6.0, 1: 6.0}, 2), (
            'one row per column, each carrying the summed price — not three rows of one'
        )

    def test_a_coefficient_of_zero_is_not_handed_to_the_solver(self):
        """Absence and a spelled-out zero say the same thing, so they cost the same.

        A sparse parameter never builds a term, and one that spells its zeros out
        builds none either.
        """
        a = _spelled_zeros([[1.0, 0.0, 0.0, 2.0], [0.0, 3.0, 0.0, 0.0]])
        with sps.build(SPELLED_ZEROS_SPEC, SPELLED_ZEROS_INDEX | {'a': a}) as model:
            tables = model._engine._model.handoff
            assert tables.matrix.height == 3, 'the five zero coefficients reached the matrix'
            assert list(tables.matrix['coeff']) == [1.0, 2.0, 3.0], 'the surviving coefficients are the nonzero ones'

    def test_a_row_of_only_zeros_stays_the_infeasibility_it_is(self):
        """Pruning may not turn `0 >= 10` into no constraint at all.

        A row whose every coefficient is zero owns no matrix entries once they are
        pruned, which is exactly the shape `_drop_termless_rows` removes — so the
        prune has to run after that rule rather than before it. Removing the row
        would report `optimal` for a model with no feasible point.
        """
        a = _spelled_zeros([[1.0, 1.0], [0.0, 0.0]])
        with sps.build(SPELLED_ZEROS_SPEC, {'i': [0, 1], 'j': [0, 1], 'a': a}) as model:
            tables = model._engine._model.handoff
            assert tables.row_count == 2, 'the all-zero row was dropped instead of kept'
            assert list(np.diff(tables.row_starts)) == [2, 0], 'the all-zero row should own no entries'
            assert model.solve().termination_condition == 'infeasible', 'a row asserting 0 >= 10 came back feasible'

    def test_a_zero_objective_coefficient_is_not_handed_to_the_solver(self):
        """A cost of zero and no cost at all are the same instruction."""
        a = _spelled_zeros([[1.0, 1.0]])
        spec = override(SPELLED_ZEROS_SPEC, objective={'sense': 'minimize', 'expression': 'sum(x * cost)'})
        spec['parameters'] = {**spec['parameters'], 'cost': {'dims': ['j']}}
        cost = pl.DataFrame({'j': [0, 1], 'value': [0.0, 5.0]})
        with sps.build(spec, {'i': [0], 'j': [0, 1], 'a': a, 'cost': cost}) as model:
            tables = model._engine._model.handoff
            assert tables.obj.height == 1, 'the zero-cost column reached the objective frame'
            assert list(tables.obj['coeff']) == [5.0]

    def test_infinite_bounds_survive_the_handoff(self, dispatch_data):
        """An absent upper bound must reach HiGHS as infinity, not as a number."""
        gens, load = dispatch_data
        base = dispatch_program()
        unbounded = replace(base, variables={'p': replace(base.variables['p'], upper=None)})
        with PolarsEngine() as engine:
            engine.build(unbounded, tidy_sources(unbounded, dispatch_sources(gens, load)))
            assert engine._model.handoff.cols['ub'].is_infinite().all()
            assert engine.solve().is_ok

    def test_equal_bounds_pin_a_variable_so_one_equation_covers_both_regimes(self):
        """A capacity that is data in one model and a decision in another.

        Pinning with equal bounds writes the row form once and lets presolve
        substitute the fixed column, so both regimes come out of the single
        equation.
        """
        data = {
            'f': ['fixed', 'sized'],
            'relmax': pd.Series({'fixed': 0.8, 'sized': 0.8}),
            'size_lb': pd.Series({'fixed': 10.0, 'sized': 0.0}),
            'size_ub': pd.Series({'fixed': 10.0, 'sized': 50.0}),
        }
        with differential(PINNED_SPEC, data, lp=True) as run:
            rate = by_coord(run.result, 'rate', 'f')
            assert rate['fixed'] == pytest.approx(8.0, rel=RTOL), 'pinned at 10, so the envelope is 0.8 * 10'
            assert rate['sized'] == pytest.approx(40.0, rel=RTOL), 'free to 50, so the envelope is 0.8 * 50'


#: Two dims on purpose (see the test below): a single-dim label frame is a scan
#: and cannot come out of order.
POSITIONAL_COLS_SPEC = {
    'dimensions': {'i': {'dtype': 'int'}, 'j': {'dtype': 'str'}},
    'parameters': {'cap': {'dims': ['i', 'j']}},
    'variables': {'x': {'dims': ['i', 'j'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'c': {'dims': ['i', 'j'], 'expression': 'x <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(sum(x, over=j), over=i)'},
}


#: A bound parameter dense over the whole variable product — a profile per
#: node, per hour. `p`'s upper bound spans exactly `p`'s dims, so alignment
#: is positional rather than a join (compiler `_aligned_bound`).
DENSE_BOUND_SPEC = {
    'dimensions': {'t': {'dtype': 'int'}, 'n': {'dtype': 'str'}},
    'parameters': {'avail': {'dims': ['t', 'n']}, 'cost': {'dims': ['n']}},
    'variables': {'p': {'dims': ['t', 'n'], 'bounds': {'lower': 0, 'upper': 'avail'}}},
    'constraints': {'cap': {'dims': ['t'], 'expression': 'sum(p, over=n) <= 100'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(p * cost)'},
}


#: The same numbers the oracle sees, with the coordinates deliberately out of
#: order — a positional attach that skipped the sort would pair every
#: coordinate with another one's bound.
SHUFFLED_BOUND = pl.DataFrame(
    {
        't': [2, 0, 1, 2, 0, 1],
        'n': ['b', 'b', 'a', 'a', 'a', 'b'],
        'value': [6.0, 2.0, 3.0, 5.0, 1.0, 4.0],
    }
)


FLAT_COST = pl.DataFrame({'n': ['a', 'b'], 'value': [1.0, 1.0]})

#: The indices those two carry, which every build of them has to be handed.
DENSE_BOUND_INDEX = {'t': [0, 1, 2], 'n': ['a', 'b']}
FLAT_INDEX = {'n': ['a', 'b', 'c']}


def _aligned_for(spec, data, monkeypatch):
    """Which bound parameters took the positional path building *spec*."""
    from specsolve.relational.engines.polars import compiler as compiler_module

    real = compiler_module.PolarsCompiler._aligned_bound
    seen = {}

    def spy(self, frame, param, v, alias):
        out = real(self, frame, param, v, alias)
        seen[param] = out is not None
        return out

    monkeypatch.setattr(compiler_module.PolarsCompiler, '_aligned_bound', spy)
    # the decision is recorded while the plan is built, before it runs
    with contextlib.suppress(SpecsolveError):
        sps.build(spec, data).close()
    return seen


#: One dimension, so the oracle can take it: the linopy loader wants a
#: `pd.Series` for a 1-D parameter and refuses a polars frame for a 2-D one.
FLAT_SPEC = {
    'dimensions': {'n': {'dtype': 'str'}},
    'parameters': {'avail': {'dims': ['n']}, 'cost': {'dims': ['n']}},
    'variables': {'p': {'dims': ['n'], 'bounds': {'lower': 0, 'upper': 'avail'}}},
    'constraints': {'cap': {'dims': [], 'expression': 'sum(p, over=n) <= 100'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(p * cost)'},
}


class TestThePositionalHandoff:
    """The sink is handed positions, not labels, and position is the whole contract.

    A column index *is* the solver column, a bound is attached by position
    rather than joined, and a chunk is a range of entries. Every test here
    would still pass if the labels were wrong and only the arithmetic right,
    and every one fails if the order the sink is handed disagrees with the
    order the model believes.
    """

    def test_a_solution_is_read_back_in_label_order_without_sorting_for_it(self):
        """The order is produced by the labeller, so the read-back only reads it.

        Seeding the vector with an ``arange`` makes every value its own label, so a
        read-back in label order comes out ascending, which is the contract
        `sol.primal` states. The plan holds no sort, which would move a full copy
        of the coordinates while the solver's own model is still resident.
        """
        spec = {
            'dimensions': {'t': {'dtype': 'int'}, 'g': {'dtype': 'str'}},
            'parameters': {'cap': {'dims': ['g']}, 'load': {'dims': ['t']}},
            'variables': {'p': {'dims': ['t', 'g'], 'where': 'cap > 0', 'bounds': {'lower': 0, 'upper': 'cap'}}},
            'constraints': {'meet': {'dims': ['t'], 'where': 'load > 0', 'expression': 'sum(p, over=g) >= load'}},
            'objective': {'sense': 'minimize', 'expression': 'sum(sum(p, over=g), over=t)'},
        }
        sources = {
            't': [0, 1, 2, 3],
            'g': ['a', 'b', 'c'],
            'cap': pl.DataFrame({'g': ['a', 'b', 'c'], 'value': [5.0, 0.0, 7.0]}),
            'load': pl.DataFrame({'t': [0, 1, 2, 3], 'value': [1.0, 0.0, 2.0, 3.0]}),
        }
        with sps.build(spec, sources) as model:
            primal = pl.Series('value', np.arange(model._engine._model.handoff.column_count, dtype=np.float64))
            dual = pl.Series('value', np.arange(model._engine._model.handoff.row_count, dtype=np.float64))
            primals, duals, activities, _ = model._engine._read_back(primal, dual, dual, None)
            assert 'SORT' not in primals['p'].explain(optimized=False), 'the labeller already ordered this'
            assert primals['p'].collect()['value'].to_list() == list(range(len(primal))), 'primal not in label order'
            assert duals['meet'].collect()['value'].to_list() == list(range(len(dual))), 'dual not in label order'
            assert activities['meet'].collect()['value'].to_list() == list(range(len(dual))), (
                'activity not in label order'
            )

    @pytest.mark.parametrize('length', [2, 5], ids=['short', 'long'])
    def test_a_solver_vector_that_does_not_span_the_model_is_refused(self, monkeypatch, length):
        """A wrong length is a different model's answer, not a short one.

        Reading a solution back is positional. A short vector leaves the trailing
        declarations reading past the end; a long one leaves every declaration
        reading the right slice of the wrong vector. Neither is recoverable, so
        both are refused where the solver hands them over — not where they are
        read: the objective comes straight from the solver, so a `Result` built on
        a broken vector reports a plausible number and fails only if someone asks
        for a coordinate.

        The double overrides `_run`, the half a sink writes; the guard is the
        base's `run` around it, so every sink is checked. Everything else it goes
        through is the real one.
        """

        class Crooked(Highs):
            def _run(self, tables):
                answer = super()._run(tables)
                stretched = pl.Series('value', [*answer.primal, *([0.0] * length)])
                return replace(answer, primal=stretched.head(length))

        monkeypatch.setitem(SOLVERS, 'highs', Crooked)
        with (
            sps.build(SOLVER_VECTOR_SPEC, SOLVER_VECTOR_LOAD) as model,
            pytest.raises(SpecsolveError, match=f'returned {length} primal values for a model with 3'),
        ):
            model.solve()

    @pytest.mark.parametrize('solver_name', sorted(SOLVERS))
    def test_a_solver_hands_back_a_vector_and_not_an_index(self, solver_name):
        """A solution is positional, so there is nothing to key it by.

        Solver output is indexed by the solver's own index, which *is* our label —
        so a ``(label, value)`` frame carries an ``arange`` beside every value that
        the read-back never reads, 8 bytes a column for as long as the result is
        held.

        Read off the hand-off rather than off the `Result`, which lays these
        vectors into its frames and keeps no second copy of them.
        """
        with sps.build(SOLVER_VECTOR_SPEC, SOLVER_VECTOR_LOAD) as model:
            assert model.solve(solver_name=solver_name).is_ok
            engine = model._engine
            assert engine._solver is not None, 'a solve leaves the solver holding the model'
            tables = engine._model.handoff
            answer = engine._solver.run(tables)
            for values, count in ((answer.primal, tables.column_count), (answer.dual, tables.row_count)):
                assert isinstance(values, pl.Series), 'a frame here is an index column nothing reads'
                assert values.name == 'value'
                assert len(values) == count, 'the read-back slices it positionally, so it spans the model'

    def test_row_chunks_are_bounded_by_nonzeros_not_by_rows(self):
        """A chunk of rows is a chunk of *entries*, and only entries are residency.

        The solver hand-off reads ``matrix`` a range at a time and holds that range
        while it works it. Sizing the range in rows bounds the wrong quantity: the
        same 100k-row range is 900k entries in ``transport`` and 10M in
        ``dispatch``, so what the sink holds would be set by the model's shape
        rather than by the budget.

        Wide rows are the case that separates the two, so this builds them: 50
        generators summed into each of 4 snapshots is 50 entries per row.
        """
        n_g, n_s = 50, 4
        gens = pd.DataFrame(
            {
                'generator': [f'g{i}' for i in range(n_g)],
                'p_max': np.full(n_g, 10.0),
                'cost': np.arange(1.0, n_g + 1.0),
            }
        )
        load = pd.DataFrame({'snapshot': np.arange(n_s), 'value': np.full(n_s, 100.0)})

        with PolarsEngine() as engine:
            engine.build(dispatch_program(), tidy_sources(dispatch_program(), dispatch_sources(gens, load)))
            tables = engine._model.handoff
            assert tables.matrix.height == n_g * n_s

            def widest(ranges):
                return max(int(tables.row_starts[hi] - tables.row_starts[lo]) for lo, hi in ranges)

            budget = 100
            assert widest(tables._spans(budget)) <= budget

            assert widest(ranges(tables.row_count, budget, 1.0)) == n_g * n_s, (
                'the same budget spent as if a row cost one element puts every entry in one chunk'
            )

    @pytest.mark.parametrize(
        ('total', 'budget', 'width'),
        [(0, 100, 1.0), (1, 100, 1.0), (10, 3, 1.0), (10, 100, 1.0), (10, 3, 7.0), (10, 3, 0.25), (10, 1, 1e9)],
    )
    def test_chunk_ranges_are_contiguous_gapless_and_cover_everything(self, total, budget, width):
        """The property the whole hand-off rests on, at every awkward size.

        ``addCols`` and ``addRows`` append: column *k* must be the *k*-th row handed
        over. That holds only if the ranges are ordered, consecutive and gapless —
        a dropped range silently shortens the model, an overlapping one relabels
        it, and neither shows up as an error. The widths here include one below 1
        and one far above the budget, the two ends where a ``//`` can produce a
        zero step or a chunk wider than asked for.
        """
        got = list(ranges(total, budget, width))

        assert all(lo < hi for lo, hi in got), 'an empty range means a wasted pass'
        assert [lo for lo, _ in got] == sorted(lo for lo, _ in got), 'ranges must ascend'
        assert [i for lo, hi in got for i in range(lo, hi)] == list(range(total)), 'gap, overlap, or short'
        if total:
            widest = max(hi - lo for lo, hi in got)
            assert widest * max(1.0, width) <= max(budget, max(1.0, width)), 'a chunk exceeded the budget'

    def test_dense_columns_does_not_edit_the_model_it_projects(self):
        """Two solvers, two spellings of infinity, one model — and it survives both.

        `cols` has no `col` column: it is one row per column in label order, so the
        vectors a solver sink is handed are *views* of the frame rather than the
        scatter's fresh arrays. Replacing an infinity in place through one would
        rewrite the built model to suit whichever solver asked last, and the second
        ask would read bounds the first had already edited.
        """
        with sps.build(RHS_SPEC, {'i': [0, 1], 'rhs': pl.DataFrame({'i': [0, 1], 'value': [1.0, 2.0]})}) as model:
            tables = model._engine._model.handoff
            first = tables.dense_columns(1e30).lb
            ub_after_first = tables.cols['ub'].to_list()

            second_ub = tables.dense_columns(1e100).ub

            assert ub_after_first == tables.cols['ub'].to_list(), 'the projection edited the model'
            assert all(v == float('inf') for v in tables.cols['ub'].to_list()), 'the frame lost its infinities'
            assert list(second_ub) == [1e100, 1e100], "the second solver got the first solver's infinity"
            assert list(first) == [0.0, 0.0]

    @pytest.mark.parametrize('where', [None, 'cap > 0'])
    def test_cols_is_positional_so_a_row_index_is_its_solver_column(self, where):
        """Every row of `cols` sits at its own label, masked or not.

        `cols` carries no `col`: it is one row per column in label order, so a
        row's position *is* the solver's index. Both arithmetic label paths get
        that order from the emission order of a **cross join**, which is a property
        of polars rather than of this package — so it is checked here against the
        label frame, which knows the answer independently.

        Two dims on purpose: one dim is a scan and cannot be out of order, so a
        single-dim model would pass whatever the product did. Bounds are distinct
        per coordinate for the same reason — a frame permuted by a mask, a join, or
        a change in someone else's engine still has the right *multiset* of bounds
        and the wrong row for every one of them.
        """
        caps = [
            {'i': i, 'j': j, 'value': 0.0 if where and (i, j) == (1, 'b') else float(10 * i + ord(j))}
            for i in range(4)
            for j in ('a', 'b', 'c')
        ]
        spec = override(POSITIONAL_COLS_SPEC, **{'variables.x.where': where}) if where else POSITIONAL_COLS_SPEC

        with sps.build(spec, {'i': list(range(4)), 'j': ['a', 'b', 'c'], 'cap': pl.DataFrame(caps)}) as model:
            tables = model._engine._model.handoff
            assert 'col' not in tables.cols.columns, 'cols carries an index it does not need'
            assert tables.cols.height == tables.column_count

            labels = model._engine._model.variables['x'].frame.collect().sort('var_label')
            raw = pl.DataFrame(caps).with_columns(pl.col('j').cast(labels['j'].dtype))
            expected = labels.join(raw, on=['i', 'j'], how='left')['value'].to_list()
            assert tables.cols['ub'].to_list() == expected, 'a bound is attached to the wrong column'

    def test_a_bound_dense_over_the_product_is_attached_by_position_not_joined(self, monkeypatch):
        """The shape xarray gets for free: position *is* the coordinate.

        `avail` spans exactly `p`'s dims and has a row per coordinate, so the
        label frame and the parameter describe the same product — and the label
        frame is in label order by construction. Sorting the parameter by the
        dimension ordinals reproduces that order, so the value column is attached
        rather than joined, which on `profiled/l` is most of a 0.58 s join.

        Checked against the oracle, because a misaligned bound is a different model
        rather than an error.
        """
        data = FLAT_INDEX | {
            'avail': pd.Series({'a': 1.0, 'b': 2.0, 'c': 3.0}),
            'cost': pd.Series({'a': 1.0, 'b': 1.0, 'c': 1.0}),
        }

        assert _aligned_for(FLAT_SPEC, data, monkeypatch) == {'avail': True}
        with differential(FLAT_SPEC, data, lp=True) as run:
            assert run.result.objective == pytest.approx(1 + 2 + 3, rel=RTOL), (
                'every p at its own upper bound; the cap of 100 never binds'
            )

    def test_the_positional_attach_sorts_rather_than_trusting_the_source_order(self, monkeypatch):
        """The same numbers arriving shuffled must give the same model.

        A parameter's rows come in whatever order its source had. Attaching without
        sorting would be right only for a file that happened to be written in label
        order and silently wrong for every other one.
        """
        shuffled = DENSE_BOUND_INDEX | {'avail': SHUFFLED_BOUND, 'cost': FLAT_COST}
        assert _aligned_for(DENSE_BOUND_SPEC, shuffled, monkeypatch) == {'avail': True}

        with sps.solve(DENSE_BOUND_SPEC, shuffled) as result:
            assert result.objective == pytest.approx(1 + 2 + 3 + 4 + 5 + 6, rel=RTOL)
            got = result.primal('p').sort('t', 'n')['value'].to_list()
            assert got == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 'each coordinate got its own bound'

    def test_a_mask_or_a_sparse_bound_keeps_the_join(self, monkeypatch):
        """Both ways position stops meaning the coordinate, refused by the gate.

        A `where` makes the label frame a subset of the product, and a parameter
        with a missing row makes the *parameter* one — either way the two sides no
        longer line up row for row, and attaching would pair a coordinate with
        another coordinate's bound. Neither is a shape the join cannot handle, so
        the gate declines and nothing else changes.
        """
        masked = override(DENSE_BOUND_SPEC, **{'variables.p.where': 't > 0'})
        shuffled = DENSE_BOUND_INDEX | {'avail': SHUFFLED_BOUND, 'cost': FLAT_COST}
        assert _aligned_for(masked, shuffled, monkeypatch) == {'avail': False}

        holey = DENSE_BOUND_INDEX | {'avail': SHUFFLED_BOUND.head(5), 'cost': FLAT_COST}
        assert _aligned_for(DENSE_BOUND_SPEC, holey, monkeypatch) == {'avail': False}


def _constant_beside_a_term_sources(expression: str, *, over_the_dim: bool = False) -> dict:
    """Data for the model above, carrying only what that model declares.

    The variable is absent at ``t = 2``, so a masked reduction counts two slots
    and not three — which is the whole reason cardinality cannot stand in for
    the count.
    """
    sources: dict = {'k': [1.0, 1.0, 1.0], 'load': 10.0, 't': pl.DataFrame({'t': [0, 1, 2]})}
    sources['d'] = [1.0, 1.0, 1.0] if over_the_dim else 1.0
    if 'r_of' in expression:
        sources['r'] = ['n', 's']
        sources['r_of'] = pl.DataFrame({'t': [0, 1, 2], 'r': ['n', 'n', 's']})
    return sources


REFUSAL_CASES = [pytest.param(expression, id=name) for expression, _, name in CONSTANT_BESIDE_A_TERM_CASES]


REWRITE_CASES = [
    pytest.param(
        expression, answer, id=name, marks=[REWRITE_IS_BROKEN_ON[name]] if name in REWRITE_IS_BROKEN_ON else []
    )
    for expression, answer, name in CONSTANT_BESIDE_A_TERM_CASES
]


def _constant_beside_a_term(expression: str, *, over_the_dim: bool = False) -> dict:
    """The model, optionally with the rewrite the refusal names applied.

    `r` and its relation are added only for the case that groups into them:
    `check` advises on a dimension nothing uses as an axis, and the suite turns
    warnings into errors.
    """
    spec = {**CONSTANT_BESIDE_A_TERM}
    parameters = dict(spec['parameters'])
    if over_the_dim:
        parameters['d'] = {'dims': ['t']}
    if 'r_of' in expression:
        spec['dimensions'] = {**spec['dimensions'], 'r': {'dtype': 'str'}}
        spec['relations'] = {'r_of': {'key': 't', 'values': 'r'}}
    return {
        **spec,
        'parameters': parameters,
        'constraints': {'bal': {'dims': [], 'expression': expression}},
    }


#: A constant beside a **masked** term, under every operator that moves or
#: reduces along a dimension. The variable is absent at ``t = 2`` and the
#: constant is not, so each operator has to decide whether the constant survives
#: that slot; the powers of ten make a disagreement name the slot it kept.
ABSENT_SLOT_CASES = [
    pytest.param('sum(x * k + d, over=t) >= load', id='sum-over'),
    pytest.param('sum(sum(x * k + d, by=r_of, over=t, into=r), over=r) >= load', id='sum-by'),
    pytest.param('sum(shift(x * k + d, along=t, offset=1, edge=0), over=t) >= load', id='shift-forward'),
    pytest.param('sum(shift(x * k + d, along=t, offset=-1, edge=0), over=t) >= load', id='shift-back'),
    pytest.param("sum(shift(x * k + d, along=t, offset=1, edge='wrap'), over=t) >= load", id='shift-wrap'),
    pytest.param('sum(at(sum(x * k + d, by=r_of, over=t, into=r), by=r_of, over=r, into=t), over=t) >= load', id='at'),
    pytest.param('sum(sum_back(x * k + d, along=t, window=2), over=t) >= load', id='sum-back'),
]


ABSENT_SLOT_SOURCES = {
    't': pl.DataFrame({'t': [0, 1, 2]}),
    'r_of': pl.DataFrame({'t': [0, 1, 2], 'r': ['n', 'n', 's']}),
    'r': ['n', 's'],
    'k': [1.0, 1.0, 1.0],
    'd': [1.0, 10.0, 100.0],
    'load': 1000.0,
}


def _absent_slot_spec(expression: str) -> dict:
    return {
        'dimensions': {'t': {'dtype': 'int'}, 'r': {'dtype': 'str'}},
        'relations': {'r_of': {'key': 't', 'values': 'r'}},
        'parameters': {'k': {'dims': ['t']}, 'd': {'dims': ['t']}, 'load': {'dims': []}},
        'variables': {'x': {'dims': ['t'], 'where': 't != 2', 'bounds': {'lower': 0}}},
        'constraints': {'bal': {'dims': [], 'expression': expression}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x, over=t)'},
    }


class TestWhereTheLanesDifferByDesign:
    """Not a bug in either lane: the linopy one builds what this one refuses by name.

    The shared claim is that the refusal is deliberate and carries its own
    rewrite (#1137). A test here going green without the message changing means
    the boundary moved, which is a decision rather than a fix.
    """

    @pytest.mark.parametrize('expression', REFUSAL_CASES)
    def test_a_constant_beside_a_term_is_a_lane_gap_not_a_language_error(self, expression):
        """#1137: the refusal is this lane's, so it may not wear the language's class.

        `LanguageError` here says the file is unsayable, which is false twice over:
        `check` passes with no data, and the linopy oracle builds the model and
        reaches *answer*. What is true is that this lane cannot represent a constant
        fragment with no rows for the dim the operator acts along, so the refusal is
        a `SpecsolveError` that names the rewrite.
        """
        spec = _constant_beside_a_term(expression)
        sps.check(spec)

        with pytest.raises(SpecsolveError) as refusal:
            sps.solve(spec, _constant_beside_a_term_sources(expression))
        text = str(refusal.value)
        assert "constraint 'bal'" in text, 'a refusal names where in the file it happened'
        assert 'Declare the parameter over' in text, 'and the rewrite that reaches the same number'
        assert not isinstance(refusal.value, LanguageError), (
            'a lane gap is not a language error — the file is sayable and the oracle builds it'
        )

    @pytest.mark.parametrize(('expression', 'answer'), REWRITE_CASES)
    def test_the_rewrite_the_lane_gap_names_reaches_the_answer(self, expression, answer):
        """The rewrite the refusal names reaches the linopy lane's answer.

        Declaring the constant over the dim gives the fragment the rows this lane
        needs, and the number it then reaches is the one the linopy lane reaches from
        the unrewritten file.
        """
        rewritten = _constant_beside_a_term(expression, over_the_dim=True)
        sources = _constant_beside_a_term_sources(expression, over_the_dim=True)
        assert sps.solve(rewritten, sources).objective == pytest.approx(answer), (
            'the rewrite answers what the linopy lane answers for the file as written'
        )

    @pytest.mark.parametrize('expression', ABSENT_SLOT_CASES)
    def test_a_constant_at_an_absent_slot_reads_the_same_on_both_lanes(self, expression):
        """#1142: every operator that moves along a dim drops the constant with the term.

        Differential rather than a pinned number, because the question is not what
        the answer is but whether the two lanes give the same one — and the linopy
        lane is the oracle for exactly this reading of v1 §13.
        """
        spec = _absent_slot_spec(expression)
        relational = sps.solve(spec, ABSENT_SLOT_SOURCES).objective

        linopy_lane = specsolve_linopy.build(spec, ABSENT_SLOT_SOURCES)
        linopy_lane.solve(solver_name='highs')

        assert relational == pytest.approx(float(linopy_lane.objective.value), rel=RTOL), (
            'the lanes disagree about whether a constant survives a slot its term is absent from'
        )


# ---------------------------------------------------------------------------
# attaching an index to a dim: by name where there is one, by position otherwise
# ---------------------------------------------------------------------------

NETWORK = {
    'dimensions': {'from_bus': {'dtype': 'str'}, 'to_bus': {'dtype': 'str'}},
    'parameters': {'cap': {'dims': ['from_bus', 'to_bus']}},
    'variables': {'f': {'dims': ['from_bus', 'to_bus'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'objective': {'sense': 'maximize', 'expression': 'sum(f)'},
}

#: Asymmetric, so a transposition changes the answer rather than hiding in it.
CAPS = {('n1', 'n1'): 1.0, ('n2', 'n1'): 5.0, ('n1', 'n2'): 500.0, ('n2', 'n2'): 1.0}


def _tidy_cap(names):
    """`cap` keyed by (from_bus, to_bus), read back off the normalised frame.

    The columns come back by name — which is the point: a transposition shows
    up as swapped values rather than hiding in the order they were written.
    """
    import pandas as pd

    wide = pd.DataFrame([(a, b, v) for (a, b), v in CAPS.items()], columns=[*names, 'value'])
    schema = Spec(**NETWORK)
    buses = {'from_bus': ['n1', 'n2'], 'to_bus': ['n1', 'n2']}
    frame = tidy_sources(schema.program, {'cap': wide, **buses})['cap'].collect()
    table = frame.to_dict(as_series=False)
    return dict(zip(zip(table['from_bus'], table['to_bus'], strict=True), table['value'], strict=True))


def test_a_named_column_attaches_by_name_not_position():
    """Two dims over the same label space make a transposed source type-check
    and cover every coordinate, so nothing downstream can catch it. Attaching by
    name is what makes the transposition visible instead.
    """
    assert _tidy_cap(['from_bus', 'to_bus']) == CAPS
    assert _tidy_cap(['to_bus', 'from_bus']) == {(f, t): v for (t, f), v in CAPS.items()}


def test_a_column_name_outside_the_declared_dims_is_an_error():
    """Refused by attaching, which asks it of a parquet path as well as a frame.

    ``tidy_sources`` only ever sees the in-memory half, so asking there too
    would be a second wording of one defect covering fewer sources.
    """
    import pandas as pd

    wide = pd.DataFrame([(a, b, v) for (a, b), v in CAPS.items()], columns=['banana', 'to_bus', 'value'])
    with pytest.raises(DataError, match='is missing columns'):
        sps.build(Spec(**NETWORK), {'cap': wide})
