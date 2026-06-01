# Backward-compatibility shim — canonical location: agent.memory.operations.procedural_profile
import sys
import agent.memory.operations.procedural_profile as _mod

sys.modules[__name__] = _mod
