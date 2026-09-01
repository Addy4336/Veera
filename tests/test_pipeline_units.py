"""
Unit Tests for Conversational Flows, Speech Sanitization, TTS Fallback, and Metrics Tracker.
"""

try:
    import pytest
except ImportError:
    class DummyMark:
        def __getattr__(self, name):
            return lambda f: f
    class DummyPytest:
        mark = DummyMark()
    pytest = DummyPytest()

import asyncio
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flows import clean_text_for_speech, flow_manager
from metrics import MetricsCollector, TurnMetrics
from tts_fallback import ResilientTTSManager, LocalKokoroFallbackTTS


def test_clean_text_for_speech():
    """Verify markdown symbols, URLs, and formatting are stripped before TTS."""
    raw = "**Hello**! Check [order](https://tracking.com/123) for details #1 *now*."
    cleaned = clean_text_for_speech(raw)
    assert "*" not in cleaned
    assert "#" not in cleaned
    assert "https://" not in cleaned
    assert cleaned == "Hello! Check order for details 1 now."


def test_metrics_latency_computation():
    """Verify turn latency calculations."""
    turn = TurnMetrics(turn_id=1)
    turn.speech_end_time = 1000.0
    turn.transcript_received_time = 1000.180  # 180ms STT
    turn.llm_prompt_sent_time = 1000.185
    turn.llm_first_token_time = 1000.395      # 210ms LLM TTFT
    turn.tts_first_audio_time = 1000.490      # 95ms TTS TTFA

    turn.compute_latencies()
    assert turn.stt_latency_ms == 180.0
    assert turn.llm_ttft_ms == 210.0
    assert turn.tts_ttfa_ms == 95.0
    assert turn.total_latency_ms == 490.0


@pytest.mark.asyncio
async def test_flow_node_transitions():
    """Verify flow state transitions."""
    flow_manager.reset()
    assert flow_manager.current_node == "Greeting"

    # User asks for order
    tokens = []
    async for token in flow_manager.process_user_turn("Where is my order ORD-12345?"):
        tokens.append(token)
    assert len(tokens) > 0
    assert flow_manager.current_node in ["OrderStatus", "Resolution"]


@pytest.mark.asyncio
async def test_tts_fallback_local_synthesis():
    """Verify local Kokoro fallback synthesis produces valid PCM chunks."""
    fallback_engine = LocalKokoroFallbackTTS(sample_rate=16000)
    await fallback_engine.initialize()

    chunks = []
    async for chunk in fallback_engine.synthesize("Hello, your order is on the way."):
        chunks.append(chunk)

    assert len(chunks) > 0
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 0
    # Chunks should be multiples of 16-bit 2-byte samples
    assert total_bytes % 2 == 0


if __name__ == "__main__":
    test_clean_text_for_speech()
    test_metrics_latency_computation()
    asyncio.run(test_flow_node_transitions())
    asyncio.run(test_tts_fallback_local_synthesis())
    print("[SUCCESS] All pipeline unit tests passed successfully!")
