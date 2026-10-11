# About

Why specsolve exists, what it will not become, and where to read more. Nothing
here is needed to write a spec or run a model.

## Why

An optimisation model usually arrives as Python that *builds* math. The
equations are tangled with the loops and the library that assembled them, so a
diff shows scaffolding rather than constraints, and reviewing the model means
reviewing a program.

specsolve makes the math the artifact. A YAML file says what the variables,
constraints and objective are. It is checked before any data exists, and
someone who knows the math can review it without knowing the builder.

## What it will not become

The closed language has a price: structure that needs the solver's answer to
decide the next row, and imperative modelling of any kind. That price buys
validation before data, two builds of one language, and a build that streams.
The specific refusals, each with its rewrite, are
[the limits of the language](https://mathspec.readthedocs.io/en/latest/about/limits/#deliberate-non-primitives).
Parity with another tool is not by itself a reason to add anything.

The work is in issues, under three tracks:
[primitives](https://github.com/fluxopt/specsolve/issues/470),
[the operational surface](https://github.com/fluxopt/specsolve/issues/471) and
[capabilities](https://github.com/fluxopt/specsolve/issues/472).

## More

| | |
|---|---|
| [Benchmarks](benchmarks.md) | what a build costs against four other ways of writing the same model |
| [Coming from linopy](linopy.md) | the same model in linopy and in specsolve, and what maps to what |
| [Prior art and credit](prior-art.md) | the work this is derived from, and how to cite it |
| [Changelog](changelog.md) | every release |
