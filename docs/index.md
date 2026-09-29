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

[Run a model](run.md){ .md-button .md-button--primary }
[Browse the examples](examples/index.md){ .md-button }

</div>

---

<div class="landing" markdown>

--8<-- "README.md:intro"

## What it is for

<div class="grid cards" markdown>

--8<-- "README.md:benefits"

</div>

## A spec is one file

--8<-- "README.md:model"

## The math it states

Printed from the file above, with no data and no solver. **How** shows the
call.

<!-- home-math:begin -->
=== "The math"

    Least-cost dispatch of a generator fleet against an hourly load.

    #### Sets

    | Symbol | Meaning |
    |---|---|
    | $`\mathcal{S}`$ | index $`s`$ — `snapshot` — dispatch periods |
    | $`\mathcal{G}`$ | index $`g`$ — `generator` — generating units |

    #### Parameters

    | Symbol | Meaning |
    |---|---|
    | $`\bar p`$ | `p_max` over $`\mathcal{G}`$ — installed capacity |
    | $`\ell`$ | `load` over $`\mathcal{S}`$ — demand to be met |
    | $`c`$ | `cost` over $`\mathcal{G}`$ — marginal cost |

    #### Variables

    | Symbol | Meaning |
    |---|---|
    | $`p`$ | `p` over $`\mathcal{S} \times \mathcal{G}`$ — output of a generator in a snapshot |

    #### Objective

    ```math
    \min \sum_{s \in \mathcal{S},\ g \in \mathcal{G}} p_{s,g} \cdot c_{g}
    ```

    #### Subject to

    **`power_balance`**

    ```math
    \sum_{g \in \mathcal{G}} p_{s,g} = \ell_{s} \qquad \forall\, s \in \mathcal{S}
    ```

    #### Variable domains

    **`p`**

    ```math
    0 \le p_{s,g} \le \bar p_{g} \qquad \forall\, s \in \mathcal{S},\ g \in \mathcal{G} \,:\, \bar p_{g} > 0
    ```

=== "LaTeX"

    ```latex
    \noindent Least-cost dispatch of a generator fleet against an hourly load.

    \paragraph{Sets}
    \begin{description}
    \item[{$\mathcal{S}$}] index $s$ --- \texttt{snapshot} --- dispatch periods
    \item[{$\mathcal{G}$}] index $g$ --- \texttt{generator} --- generating units
    \end{description}

    \paragraph{Parameters}
    \begin{description}
    \item[{$\bar p$}] \texttt{p\_max} over $\mathcal{G}$ --- installed capacity
    \item[{$\ell$}] \texttt{load} over $\mathcal{S}$ --- demand to be met
    \item[{$c$}] \texttt{cost} over $\mathcal{G}$ --- marginal cost
    \end{description}

    \paragraph{Variables}
    \begin{description}
    \item[{$p$}] \texttt{p} over $\mathcal{S} \times \mathcal{G}$ --- output of a generator in a snapshot
    \end{description}

    \paragraph{Objective}
    \begin{align}
     && \min & \sum_{s \in \mathcal{S},\ g \in \mathcal{G}} p_{s,g} \cdot c_{g}
    \end{align}

    \paragraph{Subject to}
    \begin{align}
    \text{power\_balance} && \sum_{g \in \mathcal{G}} p_{s,g} & = \ell_{s} && \forall\, s \in \mathcal{S}
    \end{align}

    \paragraph{Variable domains}
    \begin{align}
    \text{p} && 0 \le p_{s,g} & \le \bar p_{g} && \forall\, s \in \mathcal{S},\ g \in \mathcal{G} \,:\, \bar p_{g} > 0
    \end{align}
    ```

=== "How"

    ```python
    import mathspec as ms

    symbols = {
        'notation': 'latex',
        'dimensions': {
            'snapshot': {'index': 's', 'set': '\\mathcal{S}'},
            'generator': {'index': 'g', 'set': '\\mathcal{G}'},
        },
        'names': {
            'cost': 'c',
            'load': '\\ell',
            'p_max': '\\bar p',
        },
    }

    ms.to_latex('dispatch.yaml', symbols=symbols)  # amsmath align
    ms.to_typst('dispatch.yaml')  # compiles without a TeX toolchain
    ms.to_markdown('dispatch.yaml')  # renders as-is on GitHub
    ```

    `symbols` is optional — drop it and the same model prints as
    $\mathit{load}_t$, $p^{\mathrm{max}}_g$. A dict, a YAML path or a
    `SymbolTable`; a key naming nothing in the model is an error, not a symbol that
    silently never applies. Every spelling is printed verbatim — `notation` says
    which language they are, and a render in the other one refuses.

    Or from a shell, where the table is that same YAML on disk and `--standalone`
    emits a document that compiles rather than a fragment to `\input`:

    ```bash
    python -m mathspec latex dispatch.yaml --symbols dispatch.symbols.yaml
    python -m mathspec typst dispatch.yaml --standalone -o dispatch.typ
    ```

    The renderer is [mathspec](https://mathspec.readthedocs.io/en/latest/reference/typeset/)'s,
    and reads the same file this page solves.
<!-- home-math:end -->

## Solve it

--8<-- "README.md:solve"

## Where to next

- [Run a model](run.md): a file and your tables to an answer, in five steps.
- [Your data](howto/data.md): from the files an instance arrives in to one
  table per parameter, and [what attaching refuses](reference/data.md).
- [Python API](reference/api.md): attach, build, solve and read back, and
  [sweep](reference/sweeps.md) one spec over scenarios.
- [The language](https://mathspec.readthedocs.io/en/latest/reference/language/):
  what a file may contain, on mathspec's site.
- [About](about/index.md): the architecture, the measured cost, and what will
  never be built.

## Install it

```bash
pip install specsolve
```

[Installation](https://github.com/fluxopt/specsolve#installation) lists the
extras.

!!! warning "Alpha, pre-1.0"

    --8<-- "README.md:status"

</div>
