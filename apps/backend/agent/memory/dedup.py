# Backward-compatibility shim — canonical location: agent.memory.operations.dedup
import sys
import agent.memory.operations.dedup as _mod

sys.modules[__name__] = _mod
