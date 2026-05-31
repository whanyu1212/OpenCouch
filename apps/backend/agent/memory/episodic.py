# Backward-compatibility shim — canonical location: agent.memory.operations.episodic
import sys
import agent.memory.operations.episodic as _mod

sys.modules[__name__] = _mod
