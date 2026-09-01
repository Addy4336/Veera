"""
Live provider verification with real API keys (.env):
1. Cartesia Sonic TTS (primary) - explicit direct streaming test
2. Groq Llama 3.3 70B - streaming + tool calling
3. Groq Whisper STT - round-trip on Cartesia-generated audio
Run: python tests/test_live_providers.py
"""

import asyncio
import io
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")


async def test_cartesia_direct():
    """Direct Cartesia streaming test - bypasses fallback to catch SDK issues."""
    from tts_fallback import ResilientTTSManager

    m = ResilientTTSManager(sample_rate=16000)
    report("Cartesia client init", m.cartesia_client is not None)
    if not m.cartesia_client:
        return None

    start = time.time()
    try:
        audio = b""
        async for chunk in m._stream_cartesia("Hello! This is Vera testing Cartesia Sonic voice."):
            audio += chunk
        elapsed = time.time() - start
        report("Cartesia direct TTS", len(audio) > 1000,
               f"{len(audio)} bytes ({len(audio)/(2*16000):.2f}s audio) in {elapsed:.2f}s")
        return audio
    except Exception as e:
        report("Cartesia direct TTS", False, f"{type(e).__name__}: {str(e)[:300]}")
        return None


async def test_tts_manager_provider():
    """Full manager path - confirm it picks Cartesia as active provider."""
    from tts_fallback import ResilientTTSManager
    from metrics import metrics

    m = ResilientTTSManager(sample_rate=16000)
    total = 0
    async for c in m.synthesize("Testing the resilient manager provider selection."):
        total += len(c)
    report("TTS manager provider", metrics._active_tts_provider == "cartesia",
           f"active={metrics._active_tts_provider}, {total} bytes, fallback_active={m._fallback_active}")


async def test_groq_llm():
    """Groq LLM streaming + tool-calling verification."""
    from flows import FlowManager

    flow = FlowManager()
    report("Groq LLM client init", flow.groq_client is not None)
    if not flow.groq_client:
        return

    start = time.time()
    # Simulate a proper turn lifecycle so tool-call metrics attach to current_turn
    from metrics import metrics
    metrics.on_user_speech_start()
    metrics.on_user_speech_end()
    tokens = []
    async for tok in flow.process_user_turn("Where is my order ORD-12345?"):
        tokens.append(tok)
    reply = "".join(tokens)
    elapsed = time.time() - start
    used_tool = any(tc["name"] == "order_status" for tc in metrics_tool_calls())
    report("Groq LLM + tool call", len(reply) > 20 and used_tool,
           f"{len(reply)} chars in {elapsed:.2f}s | tool_used={used_tool}")
    print(f"       reply: {reply[:120]}")


def metrics_tool_calls():
    from metrics import metrics
    if metrics.current_turn and metrics.current_turn.tool_calls:
        return metrics.current_turn.tool_calls
    return []


async def test_groq_stt_roundtrip(cartesia_audio):
    """Groq Whisper transcription of real speech audio (from Cartesia)."""
    from stt import GroqSTT

    stt = GroqSTT(sample_rate=16000)
    report("Groq STT client init", stt.available)
    if not stt.available or not cartesia_audio:
        return

    start = time.time()
    text = await stt.transcribe(cartesia_audio)
    elapsed = time.time() - start
    ok = len(text) > 5
    report("Groq Whisper STT round-trip", ok,
           f"'{text[:80]}' in {elapsed:.2f}s")


async def main():
    print("=" * 60)
    print("VERA LIVE PROVIDER TESTS (real keys from .env)")
    print("=" * 60)

    cartesia_audio = await test_cartesia_direct()
    await test_tts_manager_provider()
    await test_groq_llm()
    await test_groq_stt_roundtrip(cartesia_audio)

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
