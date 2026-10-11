# Coming from linopy

For a reader who writes models in [linopy](https://github.com/PyPSA/linopy).
specsolve solves the same models, and borrows linopy's names where it can. The
difference is where the math lives: in a YAML file you check and diff, not in
the Python that builds it.

## The same model

The dispatch model of [Your first model](../run.md), written both ways. Both
reach the same objective and the same prices, and `tests/test_ports.py` holds
them to it.

=== "linopy"

    <!-- The fence includes the model from examples/ports/references/linopy/dispatch.py. -->
    <!-- doctest: skip -->
    ```python
    --8<-- "examples/ports/references/linopy/dispatch.py:model"
    ```

=== "specsolve"

    <!-- The fence includes examples/dispatch.yaml, which the suite solves. -->
    <!-- doctest: skip -->
    ```yaml
    --8<-- "examples/dispatch.yaml"
    ```

    ```python
    result = sps.solve('dispatch.yaml', sources)
    ```

## What maps to what

| linopy | specsolve |
|---|---|
| `add_variables(lower, upper, coords, mask)` | `variables:` with `dims`, `bounds` and `where` |
| `add_constraints(lhs == rhs)` | `constraints:` with an `expression` |
| `add_objective(expr)` | `objective:` |
| `coords` from pandas indexes | `sources`: one table per parameter, and the labels of each dimension ([the data contract](../reference/data.md)) |
| `m.solve()` | [`sps.solve`](../reference/api.md#specsolve.solve), or `sps.build` and `model.solve()` to solve again |
| `p.solution`, `con.dual` | `result.primal('p')`, `result.dual('power_balance')`, as polars tables |
| an xarray answer | `result.to_dataarray('p')` ([pandas or xarray](../howto/pandas-xarray.md)) |
| `fix`, `relax`, `remove_constraints` | data and spec edits ([fixing, relaxing and removing](../howto/fix-relax-remove.md)) |
| a constraint's rows | [`model.row`](../reference/api.md#specsolve.types.Model.row) |
| `compute_infeasibilities` | [`model.infeasible_subsystem`](../reference/api.md#specsolve.types.Model.infeasible_subsystem) |

Solve statuses take linopy's names: `status` and `termination_condition`, with
`is_ok` as the rollup.

## What is different

- **The model is a file.** There is no Python API that builds math.
  `sps.check` validates the file with no data, so a model can be reviewed and
  checked in CI before any data exists.
- **Data is attached by name, and a missing row is a missing coordinate.**
  A model holds only the rows that exist, which is what keeps a sparse model
  small.
- **linopy is not a dependency.** No install of specsolve brings it. The test
  suite builds every file a second time in linopy and compares the two
  ([verification](../development/verification.md#the-oracle)).
- **Some of linopy is not taken**: array operations such as `merge` and
  `reindex` are data preparation, and the solver layer is specsolve's own
  ([#106](https://github.com/fluxopt/specsolve/issues/106)).

## Where linopy is ahead

- **Solver breadth**: twelve solvers against specsolve's three, plus the LP
  and MPS files.
- **Verbs on the model**: `fix` and `relax` are methods there, and edits to the
  data or the spec here.
- **Labelled answers**: linopy answers in xarray; specsolve answers in tables,
  and `to_dataarray` bridges.
