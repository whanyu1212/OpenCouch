"""Shared storage primitives for app-owned persistence backends.

This package holds the SQL-dialect shim that lets the SQLite and PostgreSQL
implementations of small key/value audit backends (crisis log, session
feedback) share one logic body. The dialect bundles the genuinely
backend-specific pieces — placeholder token, connection factory, schema-DDL
atomicity, JSON encode/decode, cursor ceremony, and commit policy — so the
shared store code reads identically across drivers.
"""
