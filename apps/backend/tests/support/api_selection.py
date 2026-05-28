from __future__ import annotations

from types import SimpleNamespace

from api.models import ApiMemoryMode


def runtime_selection(
    runtime: object,
    mode: ApiMemoryMode | str | None = None,
) -> SimpleNamespace:
    memory_mode = ApiMemoryMode(mode) if mode is not None else ApiMemoryMode.PERSISTENT
    return SimpleNamespace(memory_mode=memory_mode, runtime=runtime)
