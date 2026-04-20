"""Therapeutic response subgraph for the OpenCouch agent.

This package owns the non-crisis response branch of the main workflow.
It is packaged as a LangGraph subgraph so that therapeutic logic stays
isolated from crisis handling, memory bookkeeping, and runtime concerns.
"""
