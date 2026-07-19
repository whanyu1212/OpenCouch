# Realtime Voice Dogfood Checklist

## Backend

Postgres-backed persistent mode:

```bash
cd /Volumes/ORICO/OpenCouch
docker compose -f compose.yml up -d postgres --wait

cd apps/backend
OPENCOUCH_PERSISTENCE_BACKEND=postgres \
OPENCOUCH_MEMORY_DATABASE_URL=postgresql://opencouch:opencouch@localhost:5432/opencouch \
.venv/bin/python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Web

Run from `apps/web`:

```bash
pnpm dev
```

Open `http://localhost:3000/voice/realtime-dev`.

## Scenarios

- Incognito supportive turn: verify no durable memory claims.
- Incognito persistence guard: disconnect, reload the same thread, and verify the backend did not save the turn.
- Persistent memory status: ask "is memory on?" and verify `show_memory_status`.
- Persistent restart check: use Postgres mode, restart the backend, and verify memory status/history survives through the shared runtime stores.
- Grounded lookup: ask for current/official guidance and verify `answer_grounded_lookup`.
- Crisis resource: ask for a local hotline and verify `lookup_crisis_resources`.
- Guided exercise: ask for a grounding or breathing exercise and verify skill tools.
- End session: disconnect and verify session finalization result.
