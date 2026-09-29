# Run a model

Five steps from `pip install` to an answer read back, on the dispatch model
of [the home page](index.md): three generators meet a load over four
snapshots at least cost. What a spec may contain is
[the language's](https://mathspec.readthedocs.io/en/latest/reference/language/)
to say.

## 1. Install

```bash
pip install specsolve
```

That brings polars, HiGHS and the language.

## 2. Save the spec

Copy the YAML from [the home page](index.md#a-spec-is-one-file)
into `dispatch.yaml`. It is also
[`examples/dispatch.yaml`](https://github.com/fluxopt/specsolve/blob/main/examples/dispatch.yaml)
in the repository.

## 3. Check the file

```python
import specsolve as sps

program = sps.check('dispatch.yaml')
```

`check` reads the file against the language, and needs no data to do it. To
see it refuse one, take the sum out of `power_balance`: change its line to
`expression: p == load`, and check again:

```python
sps.check('dispatch.yaml')
```

```text
DimensionError: Constraint 'power_balance': the expression carries dims ['generator'] that are not in its dims: ['snapshot'] — every stray dim multiplies the rows this constraint builds; add it to dims: if that is intended, or sum it out.
```

The message names the fix. Put `sum(p, over=generator)` back before the next
step.

## 4. Attach the numbers and solve

The file declares three parameters and two dimensions. `sources` supplies
each by name. A parameter over one dimension is a table with that dimension
and a `value` column; a bare sequence supplies a dimension's labels:

```python
import polars as pl

generators = ['wind', 'solar', 'gas']
sources = {
    'p_max': pl.DataFrame({'generator': generators, 'value': [80.0, 0.0, 200.0]}),
    'cost': pl.DataFrame({'generator': generators, 'value': [10.0, 25.0, 50.0]}),
    'load': pl.DataFrame({'snapshot': range(4), 'value': [60.0, 120.0, 180.0, 90.0]}),
    'snapshot': range(4),
    'generator': generators,
}

result = sps.solve('dispatch.yaml', sources)
print(result.objective)  # 10500.0
```

Wind at 10 runs first, and gas at 50 covers what is left. Solar has no
capacity, so the `where: "p_max > 0"` on `p` built no column for it.

## 5. Read the answer back

```python
print(result.primal('p'))  # (snapshot, generator, value): eight rows, wind and gas at each snapshot
print(result.dual('power_balance'))  # (snapshot, value): the price of one more unit of load
```

Each answer is a polars table keyed by the declaration's labels. The dual is
the cost of the last generator on: 10 at snapshot 0, where wind alone covers
the load, and 50 at the other three.

## Where next

| | |
|---|---|
| [Tables in, tables out](tables.md) | the next lesson: parquet in, tables out, and an archive to query |
| [Preparing the data](howto/data.md) | from files to the tables above |
| [The verbs](reference/api.md) · [The data contract](reference/data.md) | what every call takes, returns and refuses |
| [Choosing a solver](howto/solvers.md) | another solver, or a file for a tool specsolve has no solver for |
| [Language reference](https://mathspec.readthedocs.io/en/latest/reference/language/) · [the limits of the language](https://mathspec.readthedocs.io/en/latest/about/limits/) | what a file may contain, and where it stops |
| [Debug a wrong answer](howto/debug.md) | when it solves and the number is wrong, or it does not solve |
| [Examples](examples/index.md) | every model in the repository |
| [Roadmap](about/roadmap.md) | what is refused on purpose |
