---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# specsolve

**Solve an optimisation problem written in YAML. Attach your data as tables, and
keep the solver loaded for quick updates and warm starts.**

--8<-- "README.md:badges"

[Your first model](run.md){ .md-button .md-button--primary }
[Browse the examples](examples/index.md){ .md-button }
[The language](https://mathspec.readthedocs.io/en/latest/reference/language/){ .md-button }

</div>

---

<div class="landing" markdown>

--8<-- "README.md:intro"

## What it is for

<div class="grid cards" markdown>

--8<-- "README.md:benefits"

</div>

## A model is one file

The file is written in [mathspec](https://mathspec.readthedocs.io/), the
language specsolve solves.
[Its reference](https://mathspec.readthedocs.io/en/latest/reference/language/)
lists everything a file may hold.

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
