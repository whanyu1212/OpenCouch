"""Therapeutic response subgraph for the OpenCouch agent.

This package owns the non-crisis response branch of the main workflow.
It is packaged as a LangGraph subgraph so that therapeutic logic stays
isolated from crisis handling, memory bookkeeping, and runtime concerns.

Phase 1 starting point: this package only contains a design sketch
(``nodes_sketch.py``). The actual implementation lands alongside the
memory-layer phase-1 rebuild, once the mode set and dispatch policy
have been validated against real content.
"""
