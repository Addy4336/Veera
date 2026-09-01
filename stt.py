"""
Speech-to-Text Engine for Vera Voice AI Agent.
Primary: Groq Whisper Large v3 Turbo (cloud, ultra-fast).
Fallback: deterministic simulation transcript when no API key is configured.
"""

import asyncio
import io
import logging
import wave
from typing import Optional

from config import config

logger = logging.getLogger(__name__)


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw 16-bit little-endian PCM in a WAV container for STT APIs."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


class GroqSTT:
    """
    Groq Whisper Large v3 Turbo transcription of complete user utterances.
    Accepts raw PCM 16-bit mono frames; returns the recognized text.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.model = config.GROQ_STT_MODEL
        self.client = None

        if config.GROQ_API_KEY and not config.GROQ_API_KEY.startswith("your_"):
            try:
                from groq import AsyncGroq  # type: ignore

                self.client = AsyncGroq(api_key=config.GROQ_API_KEY)
                logger.info(f"Groq STT initialized ({self.model}).")
            except Exception as e:
                logger.warning(f"Could not initialize Groq STT client: {e}")
        else:
            logger.info("No Groq key; STT running in simulation mode.")

    @property
    def available(self) -> bool:
        return self.client is not None

    async def transcribe(self, pcm: bytes) -> str:
        """
        Transcribe a complete utterance of raw PCM bytes.
        Returns empty string on silence/unrecognizable audio.
        """
        if not pcm or len(pcm) < self.sample_rate // 10:  # <100ms ignore
            return ""

        if not self.client:
            return self._simulate_transcription(pcm)

        wav_bytes = pcm_to_wav_bytes(pcm, sample_rate=self.sample_rate)

        try:
            transcript = await self._transcribe_async(wav_bytes)
            return transcript.strip()
        except Exception as e:
            logger.error(f"Groq STT transcription failed: {e}")
            return ""

    async def _transcribe_async(self, wav_bytes: bytes) -> str:
        buffer = io.BytesIO(wav_bytes)
        buffer.name = "utterance.wav"  # groq SDK requires a filename attribute
        response = await self.client.audio.transcriptions.create(
            model=self.model,
            file=buffer,
            language="en",
            temperature=0.0,
        )
        return response.text or ""

    def _simulate_transcription(self, pcm: bytes) -> str:
        """
        Offline simulation: derive a plausible demo utterance deterministically
        from utterance duration so the pipeline can be exercised without keys.
        """
        duration_s = len(pcm) / (2 * self.sample_rate)
        logger.info(f"[STT Simulation] Utterance duration: {duration_s:.2f}s")
        if duration_s < 0.4:
            return ""
        # Cycle through demo phrases so all flows are testable offline
        phrases = [
            "Where is my order ORD-12345?",
            "I would like to schedule a callback for tomorrow at 2 PM",
            "What is your return policy?",
            "No that is all, have a good day",
        ]
        idx = int(duration_s * 10) % len(phrases)
        return phrases[idx]
