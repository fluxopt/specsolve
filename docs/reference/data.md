# The data contract

What `sources` may hold, and what attaching refuses and accepts. `sources`
maps each name a [spec](glossary.md) declares to its numbers.
[Preparing the data](../howto/data.md) gets an instance into these shapes.

## What a parameter accepts

For a parameter declared `dims: [d1, d2]`, the value under its key is one of:

- a **parquet path**;
- a **polars `LazyFrame`**, collected when the data attaches;
- **a table exposing the Arrow PyCapsule protocol** (polars, pandas, pyarrow,
  duckdb) with columns `d1, d2, value`;
- an **`int` or `float`**, one value for every
  [coordinate](glossary.md#the-data) the parameter covers;
- a **`dict`** of label to value, over one dimension;
- a **sequence** (list, tuple, `np.ndarray`) over one dimension, positional
  against its index.

**A number, a dict and a sequence are dense**, so declare a constant as
`dims: []`. A sequence carries no labels, so its dimension's index comes from
[another source](#where-coordinates-come-from).

**A `pd.Series` attaches by its index.** An unnamed index attaches to the
declared dimension, and a named one by its name. A name outside the declared
dimensions raises.

**A `MultiIndex` and an `xr.DataArray` are refused.** Pass
`series.reset_index()` or `array.to_series().reset_index()`.

## Where coordinates come from

**Each dimension's index is a key in `sources` named after the dimension**: a
table with a column of that name, a parquet path, or a bare sequence of labels.
Other columns of the table are ignored.

**An index lists each label once, in order.** On a dimension the spec declares
`ordered: true`,
[`shift`](https://mathspec.readthedocs.io/en/latest/reference/language/operators/#shift),
`sum_back`, `position`, a curve and an SOS read that order. A label on two rows is refused:
`table.select('generator').unique(maintain_order=True)` keeps the first row of
each label of a table, and `list(dict.fromkeys(labels))` the first of a bare
sequence.

**A datetime label is held in microseconds**, in the time zone it arrives in.
A column must be in its index's time zone. Under a `dtype: datetime`
dimension, a date is the microsecond its day starts on.

**A dimension nothing supplies raises.** Attach never reads labels out of the
parameters, so a misspelled label, such as `gsa` for `gas`, is refused by name.

**A [set](https://mathspec.readthedocs.io/en/latest/reference/language/sets/)
goes under its own name**, as a table with one column per member and one row
per combination it holds. **A property goes under `subject.property`**, as a
table of the subject's members and one column named after the property, with
one row per row of the subject that has it: `sources['line.bus0']` holds the
columns `line` and `bus0`. A value matching no label is refused. A set holds
each row at most once, and a property one row per row of its subject. A
property of a set holds rows of that set only.

**A property may also arrive as a column of its subject's table.** The table
under `sources['line']`, which is also the index of `line`, may carry `bus0`
and `bus1` columns; a null there is a line without that property. A set's
table may carry its properties the same way. A property given both ways is
refused.

## What attaching refuses and accepts

**A coordinate has a value, or it has no row.** A null or NaN value is
refused.

**A number the model computes is finite:**

| The model reads | The build |
|---|---|
| `x / d` or `h / d`, `d` zero | refuses it: there is no quotient |
| `x * p`, `p` infinite | refuses it: there is no coefficient |
| `h` on a constant side, `h` infinite | keeps it: `x <= inf` never binds |

An infinite bound never binds either. A reported quotient by zero is absent
([absence](https://mathspec.readthedocs.io/en/latest/reference/language/absence/)).

### Refused

| What arrives | What the message says |
|---|---|
| a declared parameter with no data | names the parameter |
| a source nothing can be read as a table from | names the shapes that are read |
| a `dims: []` parameter whose source has more than one row | |
| a dict or a sequence for a parameter over more than one dimension | |
| a sequence whose length is not the dimension's | |
| a sequence for a dimension nothing else supplies labels for | names the three ways to supply them |
| a key naming neither a parameter, a dimension, a set nor a property table | names the near miss |
| a set or a property table short of a column it declares | names them, and what each is |
| a set or a property table with a null in any column | |
| a property given under its own key and as a column of its subject's table | names both |
| a map with both authors, or neither | names them, and says which way out |
| an index carrying a column named after a set with a member over it | names the key it belongs under |
| a datetime label finer than a microsecond | names a label, and the rounding |
| a table missing a declared dimension column, or `value` | names the columns needed |
| a label outside the dimension's index | names the parameter and the strays |
| two rows for one coordinate | |
| a set with a row twice, or a property with two values for one row of its subject | |
| a property of a set at a row the set does not hold | names the strays |
| a dimension with sets or properties over it and no index | |
| a dimension nothing can supply labels for | names both ways to fix it |
| a dimension the spec declares and the caller also supplies | names the declaration and the colliding key |
| a set whose table the spec declares and the caller also supplies | names the set and the colliding column |
| a declared map whose labels nothing supplies | names the map, and asks only for the labels |
| a declared map keyed by something the labels do not carry | names the table and the strays |
| a column that is not the declared `dtype` | names both, and the declaration the data would satisfy |
| a parameter, a set or a property with no row at a coordinate, or a property missing at a row of its subject, under the default `missing: refused` | names the first such coordinate and how many more have no row ([a missing row](https://mathspec.readthedocs.io/en/latest/reference/language/declarations/#a-missing-row)) |

### Accepted

| What arrives | What happens |
|---|---|
| an undeclared column in a table | ignored |
| a datetime label in nanoseconds or milliseconds | held in microseconds |
| a pandas `datetime64[s]` column, which `pd.to_datetime` gives a date | held in microseconds |
| a coordinate with no row, of a parameter declared `missing: absent` or a value | read as its `missing:` says ([absence](https://mathspec.readthedocs.io/en/latest/reference/language/absence/)); `diagnostics().sparse_parameters` names the short parameters ([`Diagnostics`](api.md#specsolve.types.Diagnostics)) |
| a value that is readable and wrong | bound as given |

## The tables a solve reads

**[`sps.tidy(spec, sources)`](api.md#specsolve.tidy) returns the tables a solve
attaches**, after every check above:

| Declared | Columns |
|---|---|
| a dimension | `<dim>, specsolve_position`: each label once, and its position as an `Int64` from 0 |
| a parameter | its dims, then `value` |
| a set | one column per member |
| a property, under `subject.property` | the subject's members, then the property |

An [archive](../howto/archiving.md) holds these tables under `sources/`, each
with `specsolve_run` added, and each goes back into `sources` as it is.

### Column types in an archive

**An archive writes each column as a type that parquet readers agree on**, in
`sources/` and in `answer/`:

| A column that arrives as | is written as |
|---|---|
| `UInt8`, `UInt16` or `UInt32` | `Int64` (a `UInt64` stays, as `Int64` cannot hold it) |
| a timestamp in a time zone other than UTC | the same instant in UTC |

A table read back from an archive attaches as the original did.

## Growing or replacing the data

**[`update`](api.md#specsolve.types.Model.update) and
[`solve_over`](sweeps.md) attach through the rules above.**

**Do not write into data while a model or its answers are in use.** A model
keeps tables without a copy, so writing into the numpy array under a polars
frame changes what [`evaluate`](api.md#specsolve.types.Result.evaluate) reads
and what a save records. A parquet path is read once. Change data through
`update`, or pass a copy.
