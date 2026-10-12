---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# specsolve

--8<-- "README.md:tagline"

--8<-- "README.md:badges"

[Your first model](run.md){ .md-button .md-button--primary }
[Browse the examples](examples/index.md){ .md-button }
[Benchmarks](about/benchmarks.md){ .md-button }

</div>

---

<div class="landing" markdown>

--8<-- "README.md:intro"

## What it is for

<div class="grid cards" markdown>

--8<-- "README.md:benefits"

</div>

## Built on mathspec

specsolve and [mathspec](https://mathspec.readthedocs.io/) split one job, and both
read the same model file.

<div class="grid cards" markdown>

- **mathspec states the model.** A YAML file of dimensions, parameters,
  variables, constraints and an objective. mathspec checks the file with no
  data, and prints it as math for a paper.
  [The language →](https://mathspec.readthedocs.io/en/latest/reference/language/)
- **specsolve solves it.** specsolve attaches your tables to the file, builds
  the model on polars, solves it, and keeps the answer as parquet.
  [Your first model →](run.md)

</div>

## A model is one file

--8<-- "README.md:model"

The file holds no data, so you can check and diff it before any data exists,
and [print it as math](https://mathspec.readthedocs.io/en/latest/howto/print/).

## Solve it

--8<-- "README.md:solve"

## Install it

```bash
pip install specsolve
```

[Installation](https://github.com/fluxopt/specsolve#installation) lists the
extras.

!!! warning "Alpha, pre-1.0"

    --8<-- "README.md:status"

</div>
