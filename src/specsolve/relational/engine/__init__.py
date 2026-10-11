"""The engine: plan → frames → `sinks.Handoff`.

Everything here is engine-private. The contract is either side of it —
`mathspec.program` going in, `relational/sinks/handoff.py` coming out — and
nothing outside this package may reach past those two.
"""
