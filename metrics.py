"""
Per-Turn Observability and Latency Instrumentation for Vera Voice AI Agent.
Tracks:
- STT Latency: End of user speech -> Transcript received
- LLM TTFT: Transcript sent to LLM -> First token generated
- TTS TTFA: First token -> First audio packet emitted
- Total Roundtrip: End of user speech -> First audio packet
- Interruption / Barge-in occurrences
- Active provider status (Primary Cartesia vs Kokoro Fallback)
"""

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnMetrics:
    """Latency metrics and metadata for a single conversational turn."""

    turn_id: int
    timestamp: float = field(default_factory=time.time)

    # Milestones (epoch seconds with millisecond precision)
    speech_start_time: Optional[float] = None
    speech_end_time: Optional[float] = None
    transcript_received_time: Optional[float] = None
    llm_prompt_sent_time: Optional[float] = None
    llm_first_token_time: Optional[float] = None
    tts_first_audio_time: Optional[float] = None

    # Computed Latencies (in milliseconds)
    stt_latency_ms: Optional[float] = None
    llm_ttft_ms: Optional[float] = None
    tts_ttfa_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None

    # Context & State
    transcript: str = ""
    response_text: str = ""
    current_flow_node: str = "Greeting"
    active_tts_provider: str = "cartesia"
    was_interrupted: bool = False
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def compute_latencies(self) -> None:
        """Compute end-to-end and component latencies in milliseconds."""
        if self.speech_end_time and self.transcript_received_time:
            self.stt_latency_ms = max(0.0, round((self.transcript_received_time - self.speech_end_time) * 1000, 1))

        if self.llm_prompt_sent_time and self.llm_first_token_time:
            self.llm_ttft_ms = max(0.0, round((self.llm_first_token_time - self.llm_prompt_sent_time) * 1000, 1))

        if self.llm_first_token_time and self.tts_first_audio_time:
            self.tts_ttfa_ms = max(0.0, round((self.tts_first_audio_time - self.llm_first_token_time) * 1000, 1))

        if self.speech_end_time and self.tts_first_audio_time:
            self.total_latency_ms = max(0.0, round((self.tts_first_audio_time - self.speech_end_time) * 1000, 1))
        elif self.stt_latency_ms and self.llm_ttft_ms and self.tts_ttfa_ms:
            self.total_latency_ms = round(self.stt_latency_ms + self.llm_ttft_ms + self.tts_ttfa_ms, 1)


class MetricsCollector:
    """Manages telemetry collection, logging, and UI broadcasting."""

    def __init__(self):
        self.turn_counter: int = 0
        self.barge_in_count: int = 0
        self.current_turn: Optional[TurnMetrics] = None
        self.history: List[TurnMetrics] = []
        self.subscribers: List[asyncio.Queue] = []
        self._current_flow_node: str = "Greeting"
        self._active_tts_provider: str = "cartesia"

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to real-time telemetry stream."""
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unsubscribe from real-time telemetry stream."""
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    async def broadcast_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast an observability event to all connected dashboard clients."""
        payload = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data,
        }
        for q in list(self.subscribers):
            try:
                q.put_nowait(payload)
            except Exception as e:
                logger.debug(f"Failed to queue event for subscriber: {e}")

    def _safe_broadcast(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast an event if an asyncio event loop is active."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcast_event(event_type, data))
        except RuntimeError:
            pass

    def set_flow_node(self, node_name: str) -> None:
        """Update current conversation state machine node."""
        self._current_flow_node = node_name
        if self.current_turn:
            self.current_turn.current_flow_node = node_name
        self._safe_broadcast("flow_node_change", {"node": node_name})

    def set_tts_provider(self, provider_name: str) -> None:
        """Update active TTS provider (cartesia vs kokoro_fallback)."""
        self._active_tts_provider = provider_name
        if self.current_turn:
            self.current_turn.active_tts_provider = provider_name
        self._safe_broadcast("tts_provider_change", {"provider": provider_name})

    def on_user_speech_start(self) -> TurnMetrics:
        """Record user began speaking."""
        self.turn_counter += 1
        self.current_turn = TurnMetrics(
            turn_id=self.turn_counter,
            speech_start_time=time.time(),
            current_flow_node=self._current_flow_node,
            active_tts_provider=self._active_tts_provider,
        )
        self._safe_broadcast("user_speech_start", {"turn_id": self.turn_counter, "node": self._current_flow_node})
        return self.current_turn

    def on_user_speech_end(self) -> None:
        """Record user finished speaking (VAD / Smart Turn confirmation)."""
        if not self.current_turn:
            self.on_user_speech_start()
        if self.current_turn:
            self.current_turn.speech_end_time = time.time()
            self._safe_broadcast("user_speech_end", {"turn_id": self.current_turn.turn_id})

    def on_transcript_received(self, text: str) -> None:
        """Record STT transcript received from Groq Whisper."""
        now = time.time()
        if not self.current_turn:
            self.on_user_speech_start()
        if self.current_turn:
            self.current_turn.transcript = text
            self.current_turn.transcript_received_time = now
            if self.current_turn.speech_end_time:
                self.current_turn.stt_latency_ms = round((now - self.current_turn.speech_end_time) * 1000, 1)

            self._safe_broadcast(
                "transcription",
                {
                    "turn_id": self.current_turn.turn_id,
                    "text": text,
                    "stt_latency_ms": self.current_turn.stt_latency_ms,
                },
            )

    def on_llm_prompt_sent(self) -> None:
        """Record LLM prompt dispatched to Groq Llama 3.3 70B."""
        if self.current_turn:
            self.current_turn.llm_prompt_sent_time = time.time()

    def on_llm_first_token(self, token: str) -> None:
        """Record first token generated by Groq LLM."""
        now = time.time()
        if self.current_turn and not self.current_turn.llm_first_token_time:
            self.current_turn.llm_first_token_time = now
            if self.current_turn.llm_prompt_sent_time:
                self.current_turn.llm_ttft_ms = round((now - self.current_turn.llm_prompt_sent_time) * 1000, 1)

            self._safe_broadcast(
                "llm_first_token",
                {
                    "turn_id": self.current_turn.turn_id,
                    "token": token,
                    "llm_ttft_ms": self.current_turn.llm_ttft_ms,
                },
            )

    def on_tts_first_audio(self) -> None:
        """Record first audio packet produced by TTS."""
        now = time.time()
        if self.current_turn and not self.current_turn.tts_first_audio_time:
            self.current_turn.tts_first_audio_time = now
            self.current_turn.compute_latencies()

            # Log metrics to server console
            self.log_turn_summary(self.current_turn)

            self._safe_broadcast("latency_summary", asdict(self.current_turn))
            self.history.append(self.current_turn)

    def on_barge_in(self) -> None:
        """Record user interrupted Vera mid-sentence (barge-in cancellation)."""
        self.barge_in_count += 1
        if self.current_turn:
            self.current_turn.was_interrupted = True
        logger.info(f"[INTERRUPT] Barge-in detected! Total interruptions: {self.barge_in_count}")
        self._safe_broadcast(
            "barge_in",
            {
                "total_barge_ins": self.barge_in_count,
                "turn_id": self.current_turn.turn_id if self.current_turn else 0,
            },
        )

    def on_tool_executed(self, tool_name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Record tool execution event."""
        event_payload = {
            "name": tool_name,
            "args": args,
            "result": result,
            "timestamp": time.time(),
        }
        if self.current_turn:
            self.current_turn.tool_calls.append(event_payload)
        logger.info(f"[TOOL CALL] {tool_name}({args}) => {result}")
        self._safe_broadcast("tool_call", event_payload)

    def log_turn_summary(self, turn: TurnMetrics) -> None:
        """Format and print a clean latency summary to terminal."""
        logger.info(
            f"📊 Turn #{turn.turn_id} | Node: {turn.current_flow_node} | "
            f"STT: {turn.stt_latency_ms or 0}ms | "
            f"LLM TTFT: {turn.llm_ttft_ms or 0}ms | "
            f"TTS TTFA: {turn.tts_ttfa_ms or 0}ms | "
            f"Total: {turn.total_latency_ms or 0}ms | "
            f"TTS: {turn.active_tts_provider}"
        )


# Global metrics instance
metrics = MetricsCollector()
