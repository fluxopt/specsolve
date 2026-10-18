# About

Why specsolve exists and what it will not become. Nothing here is needed to
run a model.

## Why

An optimisation model usually arrives as Python that *builds* math, so a diff
shows scaffolding rather than constraints.

specsolve makes the math the artifact: a YAML file of variables, constraints
and objective, checked before any data exists, and reviewable by someone who
knows only the math.

## What it will not become

The closed language cannot say structure that needs the solver's answer to
decide the next row, or imperative modelling of any kind. In return it gets
validation before data, two builds of one language, and a build that streams.
[The limits of the language](https://mathspec.readthedocs.io/en/latest/about/limits/#deliberate-non-primitives)
list each refusal with its rewrite. Parity with another tool is not by itself
a reason to add anything.

## More

| | |
|---|---|
| [Coming from linopy](linopy.md) | the same model in linopy and in specsolve, and what maps to what |
| [Benchmarks](benchmarks.md) | what a build costs against four other ways of writing the same model |
| [Prior art and credit](prior-art.md) | the work this is derived from, and how to cite it |
