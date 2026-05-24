"""Compatibility shim for prompt input helpers during CLI deprecation."""

from opencouch_tui import input as _tui_input
from opencouch_tui.input import *  # noqa: F401,F403

_insert_slash_and_maybe_complete = _tui_input._insert_slash_and_maybe_complete
