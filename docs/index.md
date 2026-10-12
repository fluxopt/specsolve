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

</div>

---

<div class="landing" markdown>

--8<-- "README.md:intro"

## What it is for

<div class="grid cards" markdown>

--8<-- "README.md:benefits"

</div>

## A model is one file

--8<-- "README.md:model"

The file holds no data, so it can be checked, reviewed and diffed before any
data exists. mathspec prints it as LaTeX, Typst or Markdown
([print a spec as math](https://mathspec.readthedocs.io/en/latest/howto/print/)).

## Solve it

--8<-- "README.md:solve"

## Install it

```bash
pip install specsolve
```

[Installation](https://github.com/fluxopt/specsolve#installation) lists the
extras. Then start with [your first model](run.md).

!!! warning "Alpha, pre-1.0"

    --8<-- "README.md:status"

</div>
