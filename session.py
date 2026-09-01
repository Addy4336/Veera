"""
Per-Session State Management for Vera Voice AI Agent.

Each concurrent call / conversation gets its own:
- FlowManager (chat history, flow node, context)
- ResilientTTSManager (Cartesia -> Kokoro fallback state)
- STT engine handle, VAD segmenter, and turn lock (barge-in safety)

This removes the shared-global-state bottleneck and makes concurrent
calls safe. Global `metrics` remains a process-wide telemetry aggregator
for the live dashboard.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Optional

from config import config
from flows import FlowManager
from stt import GroqSTT
from tts_fallback import ResilientTTSManager
from vad import UtteranceSegmenter, create_vad

logger = logging.getLogger(__name__)

MAX_ACTIVE_SESSIONS = 50


class ConversationSession:
    """Isolated state for a single call / conversation."""

    def __init__(self, transport: str = "websocket"):
        self.id = uuid.uuid4().hex[:12]
        self.transport = transport
        self.created_at = time.time()
        self.last_active_at = time.time()

        # Per-session conversational state
        self.flow = FlowManager()
        # Per-session TTS resilience (fallback state must not leak across calls)
        self.tts = ResilientTTSManager(sample_rate=config.SAMPLE_RATE)
        # Per-session STT engine handle
        self.stt = GroqSTT(sample_rate=config.SAMPLE_RATE)
        # Per-session VAD / utterance segmentation
        self.segmenter = UtteranceSegmenter(vad=create_vad(config.SAMPLE_RATE), sample_rate=config.SAMPLE_RATE)
        # Serializes turns; cancelled on barge-in
        self.turn_task: Optional[asyncio.Task] = None

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def touch(self) -> None:
        self.last_active_at = time.time()

    def cancel_turn(self) -> bool:
        """Cancel any in-flight turn (barge-in). Returns True if one was running."""
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
            return True
        return False

    async def run_turn(self, coro_factory):
        """
        Run a turn as a managed task so it can be cancelled by barge-in.
        coro_factory: zero-arg callable returning the awaitable turn coroutine.
        """
        self.cancel_turn()  # previous turn should be done; defensive cleanup
        self.turn_task = asyncio.create_task(coro_factory())
        try:
            return await self.turn_task
        finally:
            self.turn_task = None

    def start_turn(self, coro_factory) -> Optional[asyncio.Task]:
        """
        Start a turn as a background task so the receive loop stays free to
        handle barge-in / control frames. Returns None if a turn is active.
        """
        if self.turn_task and not self.turn_task.done():
            logger.debug(f"Session {self.id}: new utterance dropped (turn still active).")
            return None
        self.turn_task = asyncio.create_task(coro_factory())
        return self.turn_task

    async def close(self) -> None:
        """Cleanup on session end."""
        self.cancel_turn()
        logger.info(f"Session {self.id} closed (age {self.age_seconds:.1f}s, transport={self.transport}).")


class SessionManager:
    """Registry of active conversation sessions."""

    def __init__(self):
        self._sessions: Dict[str, ConversationSession] = {}
        self._default_session_id: Optional[str] = None

    def create_session(self, transport: str = "websocket") -> ConversationSession:
        if len(self._sessions) >= MAX_ACTIVE_SESSIONS:
            self._evict_oldest()
        session = ConversationSession(transport=transport)
        self._sessions[session.id] = session
        logger.info(f"Session created: {session.id} (active={len(self._sessions)}, transport={transport})")
        return session

    def get_default_session(self) -> ConversationSession:
        """Sticky session for REST endpoints / browser tab without explicit ID."""
        if self._default_session_id not in self._sessions:
            session = self.create_session(transport="rest")
            self._default_session_id = session.id
        session = self._sessions[self._default_session_id]
        session.touch()
        return session

    def get(self, session_id: str) -> Optional[ConversationSession]:
        session = self._sessions.get(session_id)
        if session:
            session.touch()
        return session

    async def end_session(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session:
            await session.close()
            if self._default_session_id == session_id:
                self._default_session_id = None
            return True
        return False

    def _evict_oldest(self) -> None:
        oldest_id = min(self._sessions, key=lambda sid: self._sessions[sid].last_active_at)
        logger.warning(f"Session cap reached; evicting idle session {oldest_id}")
        # Synchronous eviction path (cannot await here); close is best-effort
        session = self._sessions.pop(oldest_id)
        session.cancel_turn()

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def stats(self) -> Dict[str, int]:
        return {
            "active_sessions": self._sessions.__len__(),
            "max_sessions": MAX_ACTIVE_SESSIONS,
        }


# Global session registry
sessions = SessionManager()
