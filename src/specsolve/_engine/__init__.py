"""The engine: plan → frames → `_sinks.Handoff`.

Everything here is engine-private. The contract is either side of it —
`mathspec.program` going in, `_sinks/handoff.py` coming out. No module
of the lane outside this package names one inside it.
"""
