"""OpenAI text agent definitions grouped by owner.

Import concrete agent definitions from their owner modules directly. Keeping
this package initializer passive avoids import cycles between SDK tools,
skills, and agent-owned prompt helpers.
"""
