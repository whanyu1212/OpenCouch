from __future__ import annotations

import importlib.util


def test_greenlet_available_for_sqlalchemy_async_persistence() -> None:
    assert importlib.util.find_spec("greenlet") is not None
