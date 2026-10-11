"""An A/B of two git refs over a grid of models, to prove a change makes the code faster.

``grid`` holds the models, ``worker`` takes one measurement in a fresh process,
and ``compare`` runs both refs and writes the verdicts. See ``bench/README.md``.
"""
