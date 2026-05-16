"""Memory context loading and turn-time injection for voice sessions."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from livekit.agents import Agent, ChatContext

from agent.memory.modes import MemoryMode
from agent.memory.procedural_profile import aget_procedural_profile
from agent.memory.reconciliation import filter_active_semantic_records
from agent.memory.store import MemoryStore

logger = logging.getLogger(__name__)

_MAX_MEMORY_ITEMS = 6
_MID_SESSION_MEMORY_ITEMS = 3


@dataclass(frozen=True)
class VoiceStartupMemoryContext:
    """Prompt-ready memory context loaded before a voice session starts."""

    semantic_facts: list[str]
    episodic_arcs: list[str]
    procedural_rules: list[str]
    proactive_recall_enabled: bool


class VoiceMemoryContextService:
    """Load and inject voice-session memory without owning LiveKit policy."""

    async def load_startup_context(
        self,
        store: MemoryStore | None,
        *,
        user_id: str,
        mode: MemoryMode,
    ) -> VoiceStartupMemoryContext:
        """Load compact startup memory for a voice prompt."""

        if store is None or mode == MemoryMode.INCOGNITO:
            return VoiceStartupMemoryContext([], [], [], False)

        facts = await self.load_semantic_facts(store, user_id=user_id, mode=mode)
        arcs = await self.load_episodic_arcs(store, user_id=user_id, mode=mode)
        rules, proactive_recall_enabled = await self.load_procedural_memory(
            store,
            user_id=user_id,
            mode=mode,
        )
        return VoiceStartupMemoryContext(
            semantic_facts=facts,
            episodic_arcs=arcs,
            procedural_rules=rules,
            proactive_recall_enabled=proactive_recall_enabled,
        )

    async def load_semantic_facts(
        self,
        store: MemoryStore,
        *,
        user_id: str,
        mode: MemoryMode,
    ) -> list[str]:
        """Return active semantic facts suitable for startup prompt context."""

        if mode == MemoryMode.INCOGNITO:
            return []
        try:
            records = await store.asearch(
                (user_id, "semantic"),
                query=None,
                limit=_MAX_MEMORY_ITEMS,
            )
        except Exception:
            logger.warning("failed to load semantic facts", exc_info=True)
            return []
        records = filter_active_semantic_records(records)
        return [
            f"Previously noted: {record.value.get('evidence_quote', '')}"
            for record in records
            if record.value.get("evidence_quote")
        ]

    async def load_episodic_arcs(
        self,
        store: MemoryStore,
        *,
        user_id: str,
        mode: MemoryMode,
    ) -> list[str]:
        """Return active episodic summaries suitable for startup prompt context."""

        if mode == MemoryMode.INCOGNITO:
            return []
        try:
            records = await store.asearch((user_id, "episodic"), query=None, limit=3)
        except Exception:
            logger.warning("failed to load episodic arcs", exc_info=True)
            return []
        arcs = []
        for record in records:
            summary = record.value.get("summary", "")
            if summary:
                themes = ", ".join(record.value.get("primary_themes", [])) or "untagged"
                arcs.append(f"Past session ({themes}): {summary}")
        return arcs

    async def load_procedural_memory(
        self,
        store: MemoryStore,
        *,
        user_id: str,
        mode: MemoryMode,
    ) -> tuple[list[str], bool]:
        """Return procedural rules and proactive-recall setting."""

        if mode == MemoryMode.INCOGNITO:
            return [], False
        try:
            profile = await aget_procedural_profile(store, user_id=user_id)
        except Exception:
            logger.warning("failed to load procedural rules", exc_info=True)
            return [], False
        return [rule.rule for rule in profile.rules[:_MAX_MEMORY_ITEMS]], (
            profile.proactive_recall_enabled
        )

    async def load_turn_relevant_semantic_facts(
        self,
        store: MemoryStore,
        *,
        user_id: str,
        mode: MemoryMode,
        query: str,
    ) -> list[tuple[str, str]]:
        """Fetch turn-relevant semantic facts for optional context injection."""

        if mode == MemoryMode.INCOGNITO or not query.strip():
            return []

        try:
            records = await store.asearch(
                (user_id, "semantic"),
                query=query,
                limit=_MID_SESSION_MEMORY_ITEMS,
            )
        except Exception:
            logger.warning("failed to load turn-relevant semantic facts", exc_info=True)
            return []

        facts: list[tuple[str, str]] = []
        for record in filter_active_semantic_records(records):
            quote = (record.value.get("evidence_quote") or "").strip()
            if quote:
                facts.append((record.key, f"Previously noted: {quote}"))

        return facts

    async def inject_turn_relevant_memory(
        self,
        agent: Agent,
        turn_ctx: ChatContext,
        *,
        user_id: str,
        mode: MemoryMode,
        query: str,
        store: MemoryStore | None,
        already_injected_keys: set[str],
        proactive_recall_enabled: bool,
    ) -> set[str]:
        """Inject unseen relevant memory into the current LiveKit turn context."""

        if store is None or not proactive_recall_enabled:
            return set()

        facts = await self.load_turn_relevant_semantic_facts(
            store,
            user_id=user_id,
            mode=mode,
            query=query,
        )
        unseen_facts = [
            (key, fact) for key, fact in facts if key not in already_injected_keys
        ]
        if not unseen_facts:
            return set()

        memory_keys = {key for key, _ in unseen_facts}
        memory_lines = "\n".join(f"- {fact}" for _, fact in unseen_facts)
        turn_ctx.add_message(
            role="system",
            content=(
                "Relevant background from prior sessions. "
                "Use it only if it fits naturally:\n"
                f"{memory_lines}"
            ),
            extra={
                "source": "semantic_memory_injection",
                "memory_keys": sorted(memory_keys),
            },
        )

        try:
            await agent.update_chat_ctx(turn_ctx)
        except Exception:
            logger.warning(
                "failed to persist turn-relevant semantic memory injection",
                exc_info=True,
            )
            return set()

        logger.info(
            "livekit session: injected semantic memory facts=%d", len(memory_keys)
        )
        return memory_keys
