# The data contract

This page lists what `sources` may hold and what attaching refuses and accepts,
for anyone putting data on a [spec](glossary.md). `sources` supplies the
numbers, keyed by the names the spec declares:

```python
import specsolve as sps

result = sps.solve(
    'dispatch.yaml',
    {'load': 'load.parquet', 'cost': cost_frame, 'p_max': p_max_frame},
)
```

Getting from the files an instance arrives in to these shapes is
[preparing the data](../howto/data.md).

## What a parameter accepts

For a parameter declared `dims: [d1, d2]`, the value under its key is one of:

- a **parquet path**;
- **a table exposing the Arrow PyCapsule protocol** (polars, pandas, pyarrow,
  duckdb) with columns `d1, d2, value`;
- an **`int` or `float`**, one value for every
  [coordinate](glossary.md#the-data) the parameter covers;
- a **`dict`** of label to value, for a parameter over one dimension;
- a **sequence** (list, tuple, `np.ndarray`) for a parameter over one
  dimension, positional against that dimension's index.

**The last three shapes serve models written out in Python.** Each is dense,
and attach materialises it: one number over `(snapshot, generator)` becomes one
row per pair. Declare a constant as `dims: []` instead. A sequence carries no
labels, so its dimension's index comes from one of the three sources under
[where coordinates come from](#where-coordinates-come-from).

**A `pd.Series` is unwrapped first.** Its one dimension is its index rather
than a column. Attach unwraps it only if pandas is already imported. An unnamed
index attaches to the declared dimension. A named index attaches by that name,
and a name outside the declared dimensions raises.

**A `MultiIndex` is refused.** A parameter over two dimensions arrives as a
table with both as columns. `series.reset_index()` is the whole change.

**An `xr.DataArray` is refused.** Pass `array.to_series().reset_index()`.
`Result.to_dataarray()` is the way back out.

**Nothing on this path imports pandas, xarray or linopy on your behalf.**

## Where coordinates come from

**Each dimension's index is resolved before any parameter loads**, from a key
in `sources` named after the dimension. That key holds a table with a column of
that name, a parquet path, or a bare sequence of the labels. Attach reads only
the column named after the dimension, and other columns of the table are
ignored. **An index lists each label once.** A label's row is its position,
and that order is what
[`shift`](https://mathspec.readthedocs.io/en/latest/reference/language/operators/#shift)
reads positionally. A label that is on two rows is refused:
`table.select('generator').unique(maintain_order=True)` keeps the first row of
each label of a table, and `list(dict.fromkeys(labels))` the first of a bare
sequence.

**A datetime label is held in microseconds**, in the time zone it arrives in.
A column must be in its index's time zone.

**A dimension nothing supplies raises.** Attach never reads labels out of the
parameters. Which labels an axis has is data's to say, and that rule is
[the language's](https://mathspec.readthedocs.io/en/latest/reference/language/dimensions/).

**A relation goes under
[its own name](https://mathspec.readthedocs.io/en/latest/reference/language/relations/#the-data-contract)**,
as a table of the rows it has, one column per column it declares. Attach reads
every column against the labels its dimension's index supplied: a key no row
mentions is unmapped, and a value matching no label is refused as a typo. A
keyed relation holds one row per key tuple, however many columns the key
names; a bare relation holds each row at most once.

## What attaching refuses and accepts

**A coordinate has a value, or it has no row.** Attach refuses a row whose
value is null or NaN. Polars and parquet write a hole as a null, pandas has
only NaN, and `None` in a pandas column is NaN by the time either lane sees it.

### Refused

| What arrives | What the message says |
|---|---|
| a declared parameter with no data | names the parameter |
| a source nothing can be read as a table from | names the shapes that are read |
| an `xr.DataArray` | names `to_series().reset_index()` |
| a `pd.Series` with a `MultiIndex` | names the table and the `reset_index()` that gets there |
| a `dims: []` parameter whose source has more than one row | one value broadcast everywhere has one row |
| a dict or a sequence for a parameter over more than one dimension | each runs along one dimension |
| a sequence whose length is not the dimension's | positional, so one entry per label |
| a sequence for a dimension nothing else supplies labels for | names the three ways to supply them |
| a key naming neither a parameter, a dimension nor a relation | names the near miss |
| a relation table short of a column it declares | names them, and what each is |
| a relation table with a null in any column | a relation is partial by omitting a row |
| a relation table mapping one key twice, or relating one tuple twice | a keyed relation holds one row per key, a bare one each row once |
| a map with both authors, or neither | names them, and says which way out |
| an index carrying a column named after a relation with a column over it | names the key it belongs under |
| a datetime label finer than a microsecond | names a label, and the rounding |
| a datetime column in another time zone than its index | names both, and the conversion |
| an index holding a label twice | names the dimension and the labels, and the rewrite that keeps the first of each, for a table or a bare sequence |
| a table missing a declared dimension column, or `value` | names the columns needed |
| a `value` column carrying a null or a NaN | names the parameter and the coordinates |
| a label outside the dimension's index | names the parameter and the strays |
| two rows for one coordinate | |
| a relation with two values for one key | |
| a relation value that is not a label of its column's dimension | one wording, checked once for both lanes |
| a dimension carrying relations with no index | |
| a dimension nothing can supply labels for | names both ways to fix it |
| a dimension the spec declares and the caller also supplies | names the declaration and the colliding key |
| a relation whose map the spec declares and the caller also supplies | names the map and the colliding column |
| a declared map whose labels nothing supplies | names the map, and asks only for the labels |
| a declared map keyed by something the labels do not carry | names the relation and the strays |
| a column that is not the declared `dtype` | names both, and the declaration the data would satisfy |
| a divisor parameter with no row where the spec divides by it | names the parameter and how many rows ([absence](https://mathspec.readthedocs.io/en/latest/reference/language/absence/)) |
| a comparison's whole constant side with no value where the row is built | the same, naming the constraint |
| a bound parameter with no value where the variable exists | names both models the two repairs build |

### Accepted

| What arrives | What happens |
|---|---|
| an undeclared column in a table | ignored |
| a datetime label in nanoseconds or milliseconds | held in microseconds |
| a coordinate with no row | sparse variables; what a missing row means where it is read is [absence](https://mathspec.readthedocs.io/en/latest/reference/language/absence/). `diagnostics().sparse_parameters` names the parameters that arrived short of their dims ([`Diagnostics`](api.md#specsolve.types.Diagnostics)) |
| a value that is readable and wrong | bound as given |

### Stray labels

**The index is what makes a stray label a stray.**

```python
cost = {'wind': 1.0, 'gsa': 2.0}  # 'gas' misspelled — refused by name
```

A dimension whose labels came from the parameters would read `gsa` as a third
generator.

## The tables a solve reads

**[`sps.tidy(spec, sources)`](api.md#specsolve.tidy) returns the tables a solve
attaches**, one per name the spec declares, after every check above:

| Declared | Columns |
|---|---|
| a dimension | `<dim>, specsolve_position`: each label once, and its position as an `Int64` from 0 |
| a parameter | its dims, then `value` |
| a relation | the columns it declares |

An [archive](../howto/archiving.md) of one solve holds these tables under
`sources/`, each with `specsolve_run` added. A build reads only the columns
above, so each one goes back into `sources` as it is. An archive of a
sweep holds what the axis cuts its slices from, so each slice attaches from it
the tables it attached.

### Column types in an archive

**An archive writes each column as a type that parquet readers agree on**, in
`sources/` and in `answer/`. Power BI and Spark, among others, read these
three differently:

| A column that arrives as | is written as |
|---|---|
| `UInt8`, `UInt16` or `UInt32` | `Int64` (a `UInt64` stays, as `Int64` cannot hold it) |
| a timestamp in nanoseconds or milliseconds | microseconds |
| a timestamp in a time zone other than UTC | the same instant in UTC |

A table read back from an archive has the written type, and it attaches as the
original did.

## Growing or replacing the data

**A built model takes new numbers with
[`update`](api.md#specsolve.types.Model.update).** A sweep over slices of one
dimension is [`solve_over`](sweeps.md). Both attach through the rules above.

**The [linopy lane](../about/linopy.md#the-same-language-and-the-same-data)
attaches by these same rules, refusals included**, held to them by
`tests/test_data_parity.py`. The same malformed source gets the same verdict,
and where one defect has one repair, the same message.
