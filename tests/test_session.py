"""
Unit Tests for Session Management, VAD Segmentation, and STT WAV Packaging.
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
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from session import ConversationSession, SessionManager
from stt import pcm_to_wav_bytes, GroqSTT
from vad import EnergyVAD, UtteranceSegmenter, pcm_bytes_to_samples


def make_speech_pcm(duration_s=1.0, sample_rate=16000, freq=440, amplitude=8000):
    """Generate loud sine-wave PCM simulating speech."""
    n = int(sample_rate * duration_s)
    samples = [int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(n)]
    import struct
    return struct.pack(f"<{n}h", *samples)


def make_silence_pcm(duration_s=0.6, sample_rate=16000):
    return b"\x00\x00" * int(sample_rate * duration_s)


def test_energy_vad_speech_detection():
    """Loud sine frames should register as speech; silence should not."""
    vad = EnergyVAD(sample_rate=16000)
    speech_frame = make_speech_pcm(duration_s=0.02)  # exactly one 20ms frame
    silence_frame = make_silence_pcm(duration_s=0.02)
    assert vad.frame_is_speech(pcm_bytes_to_samples(speech_frame)) is True
    assert vad.frame_is_speech(pcm_bytes_to_samples(silence_frame)) is False


def test_energy_vad_events():
    """VAD should emit speech_start after sustained speech and speech_end after sustained silence."""
    vad = EnergyVAD(sample_rate=16000, min_speech_ms=200, min_silence_ms=350)
    events = []
    # 10 speech frames (200ms)
    for _ in range(10):
        ev = vad.process_frame(make_speech_pcm(duration_s=0.02))
        if ev:
            events.append(ev)
    # 20 silence frames (400ms)
    for _ in range(20):
        ev = vad.process_frame(make_silence_pcm(duration_s=0.02))
        if ev:
            events.append(ev)
    assert events == ["speech_start", "speech_end"]


def test_utterance_segmenter_completes_on_silence():
    """Segmenter should emit a complete utterance once speech ends."""
    seg = UtteranceSegmenter(sample_rate=16000)
    speech = make_speech_pcm(duration_s=1.0)
    silence = make_silence_pcm(duration_s=0.6)

    result = seg.process(speech)  # speech still open
    assert result is None

    result = seg.process(silence)  # trailing silence closes the utterance
    assert result is not None
    assert len(result) >= len(speech)  # includes pre-roll
    assert seg.process(silence) is None  # nothing more until new speech


def test_pcm_to_wav_packaging():
    """PCM should be wrapped into a valid 16kHz mono 16-bit WAV container."""
    pcm = make_speech_pcm(duration_s=0.5)
    wav = pcm_to_wav_bytes(pcm, sample_rate=16000)
    assert len(wav) > len(pcm)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


@pytest.mark.asyncio
async def test_session_isolation():
    """Two sessions must have independent flow state and history."""
    sm = SessionManager()
    s1 = sm.create_session()
    s2 = sm.create_session()
    assert s1.id != s2.id
    assert s1.flow is not s2.flow
    assert s1.tts is not s2.tts

    s1.flow.transition_to("OrderStatus")
    assert s2.flow.current_node == "Greeting"

    await sm.end_session(s1.id)
    await sm.end_session(s2.id)
    assert sm.active_count == 0


@pytest.mark.asyncio
async def test_session_barge_in_cancels_turn():
    """Barge-in should cancel a long-running in-flight turn task."""
    session = ConversationSession()

    async def slow_turn():
        await asyncio.sleep(10)
        return "never"

    task = session.start_turn(slow_turn)
    assert task is not None
    await asyncio.sleep(0.05)
    assert session.cancel_turn() is True
    await asyncio.sleep(0.05)
    assert task.cancelled() or task.done()
    assert session.cancel_turn() is False


@pytest.mark.asyncio
async def test_session_start_turn_drops_concurrent():
    """A second start_turn while one is active should be dropped (None)."""
    session = ConversationSession()

    async def turn():
        await asyncio.sleep(0.2)

    t1 = session.start_turn(turn)
    t2 = session.start_turn(turn)
    assert t1 is not None
    assert t2 is None
    session.cancel_turn()


@pytest.mark.asyncio
async def test_stt_simulation_mode():
    """Without API keys the STT engine returns deterministic simulation text."""
    stt = GroqSTT(sample_rate=16000)
    if stt.available:
        return  # real key present; skip simulation assertions
    text = await stt.transcribe(make_speech_pcm(duration_s=1.0))
    assert isinstance(text, str) and len(text) > 0
    empty = await stt.transcribe(make_speech_pcm(duration_s=0.05))
    assert empty == ""


if __name__ == "__main__":
    test_energy_vad_speech_detection()
    test_energy_vad_events()
    test_utterance_segmenter_completes_on_silence()
    test_pcm_to_wav_packaging()
    asyncio.run(test_session_isolation())
    asyncio.run(test_session_barge_in_cancels_turn())
    asyncio.run(test_session_start_turn_drops_concurrent())
    asyncio.run(test_stt_simulation_mode())
    print("[SUCCESS] All session/pipeline unit tests passed successfully!")
