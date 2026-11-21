<!-- Copied verbatim from energy-models/mathspec@5f1464c, .claude/skills/prose-writing/SKILL.md, section "Words this project owns". Do not edit: copy it again. tests/test_words.py checks it against the pinned mathspec. -->

### Words this project owns

A word that names an operator cannot also name a
concept, so `ceiling` rounds up and the limit of what the language can express
is _the limit_. _dims_ is a key; in prose the word is _dimensions_.

| Project word  | Define it at first use as                                                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| a tool        | a piece of software that reads a spec: an engine, a renderer, a checker. Never "a program", because `Program` is the object `to_program` returns |
| an engine     | a tool that attaches data to a spec, builds the model and hands it to a solver                                                                   |
| an operator   | a construct built into the language, which no file can add; never "primitive"                                                                    |
| a macro       | a template that takes arguments and is substituted into an expression before anything reads it                                                   |
| a formulation | an entry that expands into ordinary entries before the model is built                                                                            |
| to load       | to run `to_spec` on a file. "Refused at load" means `to_spec` raises before any data exists                                                      |
| to compile    | to turn a spec into its program, which `to_spec` does in the same call. Never "lower" in prose                                                   |
| a fragment    | one file of a spec that `merge` composes                                                                                                         |
| a patch       | a file that `override` lays over a base                                                                                                          |
| advice        | what `check` prints about a spec without refusing it                                                                                             |
| a rung        | one step of the PyPSA ladder, which is one `n.optimize()` keyword stated in full                                                                 |

Words to avoid, and what to write instead: consumer (a tool), bind data
(attach data), mask or predicate (a condition), reject or error out (refuse),
normal form (the canonical form), unbuilt or masked out (absent), summand
(a term of a sum), seam,
surface, lane, tax, closure, sayable (say the plain thing). The coupling
surface of a component library is a glossary term and is not this fault.

