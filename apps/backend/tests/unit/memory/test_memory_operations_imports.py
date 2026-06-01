"""Import smoke tests for memory operations compatibility paths."""

from agent.memory.dedup import find_near_duplicate as shim_find_near_duplicate
from agent.memory.operations.dedup import (
    find_near_duplicate as canonical_find_near_duplicate,
)


def test_dedup_canonical_and_shim_exports_match() -> None:
    assert canonical_find_near_duplicate is shim_find_near_duplicate
