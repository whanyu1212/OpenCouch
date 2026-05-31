# Backward-compatibility shim — canonical location: agent.memory.operations.semantic_writes
import sys
import agent.memory.operations.semantic_writes as _mod

sys.modules[__name__] = _mod
