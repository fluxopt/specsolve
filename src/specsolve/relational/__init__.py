"""Internal: relational LP construction — the streaming lane.

The public interface of the package is YAML (see ``specsolve.api``); constructing
programs in Python is not supported API.

``sinks/``, ``status.py`` and ``result.py`` are the contract: what an engine
answers to and what a sink reads. ``engines/`` holds implementations of that
contract, one per directory. A solver's own package is imported inside the
function that calls it, so one a caller has not installed never reaches their
import path.

Nothing is re-exported here.
"""
