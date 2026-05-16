# Backend Test Layout

The backend pytest suite is organized by test type first, then by feature area.

- `unit/`: pure or narrowly scoped tests for helpers, services, prompt builders,
  schemas, and storage primitives.
- `integration/`: tests that exercise graph/runtime/API/channel/persistence
  wiring with fakes or local stores.
- `live/`: opt-in provider tests that require API keys and explicit environment
  flags.
- `support/`: shared test-only helpers. Test modules should import shared fakes
  from here instead of importing private helpers from another test module.

Run the default CI-safe suite from `apps/backend`:

```bash
.venv/bin/python -m pytest -q tests/unit tests/integration
```

Run live provider checks only when the relevant API keys and opt-in flags are
configured:

```bash
.venv/bin/python -m pytest -q tests/live
```
