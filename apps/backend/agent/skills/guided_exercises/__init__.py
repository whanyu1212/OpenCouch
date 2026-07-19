"""Guided exercise internals split by responsibility.

Public subpackages:
- ``catalog``: runtime source-of-truth exercise definitions, types, and filters.
- ``lifecycle``: app-owned exercise selection and step progression service.
- ``rendering``: prompt-local skill/directive rendering helpers.

Exercise ids, state schema, tool names, and text/voice behavior are owned by
these runtime services; package names should describe those responsibilities
without changing guided-exercise execution semantics.
"""
