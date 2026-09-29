# Debugging a wrong answer

What to read when a model solves to the wrong number, or does not solve, in
the order that finds the fault soonest. Each step needs the file, or the file
and its data, and none needs a solver run you have not already paid for.

## 1. Check the file against the sink

```python
import specsolve as sps

sps.check('dispatch.yaml', sink='highs')
```

`check` raises on a construct outside the language, and with `sink=` on one
the solver cannot take: an `sos:` set on `highs`, which has no such concept,
until `Spec.expand()` writes it out as binaries
([what each sink takes](../reference/api.md#what-each-sink-takes)).

## 2. Read the shape the build produced

```python
model = sps.build('dispatch.yaml', sources)
report = model.diagnostics()
report.columns, report.rows, report.nonzeros  # 8, 4, 8
report.omissions  # rows a constraint declared and did not build
report.sparse_parameters  # parameters whose table is short of their coordinates
```

A count smaller than you expected is a mask, or a table with a row missing.
`omissions` names the constraint, `sparse_parameters` the parameter; which one
you have is the difference between a `where:` you wrote and a row you lost
([diagnostics](../reference/api.md#specsolve.relational.result.Diagnostics)).

## 3. Read the row that is wrong

```python
print(model.row('power_balance', snapshot=2))
# power_balance[snapshot=2]: +1 p[2, wind] +1 p[2, gas] == 180
```

The line is the row as the solver got it: every coefficient the data
produced, and no term for a variable a `where:` removed. Here `solar` is
absent because its `p_max` is `0.0`. A term you expected and do not see is a
mask; a coefficient you did not expect is the data
([`Model.row`](../reference/api.md#specsolve.Model.row)).

## 4. When the solve is infeasible

```python
result = model.solve()
result.status, result.termination_condition  # 'warning', 'infeasible'
result.has_primal  # False: nothing to read, and every reader raises
```

There is no IIS (irreducible infeasible subsystem) read-back. Locate the
fault instead with a slack: add a variable to the balance row, minimise it,
and read where it is nonzero. The
[feasibility model](../about/decomposition.md#when-the-subproblem-is-infeasible)
is that file for a dispatch. Then read the row at a snapshot the slack lands
on, as in step 3.

## 5. When the number is wrong and the rows look right

Rows that read right and a number you still believe is wrong mean the file
says something other than what you meant. Render it as math and read the
constraint as written:
[typeset](https://mathspec.readthedocs.io/en/latest/reference/typeset/).

## 6. When a loop of re-solves is slow

Compare `keep='solver'` with `keep='progress'` on the loop, as
[warm-starting a re-solve](warm-start.md#check-that-it-pays) shows. A `kept`
of `'nothing'` on every iteration means each update moved a mask, so the loop
pays for a rebuild, not for the solve.
