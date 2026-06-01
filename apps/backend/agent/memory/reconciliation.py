# Backward-compatibility shim — canonical location: agent.memory.operations.reconciliation
import sys
import agent.memory.operations.reconciliation as _mod

sys.modules[__name__] = _mod
