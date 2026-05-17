"""Therapeutic response services for the OpenCouch agent.

This package owns the non-crisis response branch of the main workflow.
Therapeutic logic stays isolated from crisis handling, memory bookkeeping, and
runtime concerns through plain services consumed by the OpenAI text adapter.
"""
