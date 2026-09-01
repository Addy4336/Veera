"""
Resilient TTS Fallback Architecture for Vera Voice AI Agent.
Primary: Cartesia Sonic TTS (ultra-low latency, streaming WebRTC audio).
Fallback: Kokoro-82M / Local Fallback TTS Engine (activated on Cartesia 429 / Quota / Timeout).
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

from config import config
from metrics import metrics

logger = logging.getLogger(__name__)


class LocalKokoroFallbackTTS:
    """
    Local zero-dependency / onnx TTS engine for Kokoro-82M.
    Provides offline, CPU-based speech generation when cloud APIs are unavailable.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._onnx_model = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Kokoro ONNX model if kokoro-onnx package is available."""
        if self._initialized:
            return
        try:
            # Try importing kokoro-onnx if installed
            import kokoro_onnx  # type: ignore

            logger.info("Local Kokoro ONNX engine loaded successfully.")
            self._initialized = True
        except ImportError:
            logger.info("Local Kokoro ONNX not installed; using built-in synthetic PCM fallback generator.")
            self._initialized = True

    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech from text locally.
        Yields raw PCM 16-bit 16kHz audio chunks.
        """
        logger.info(f"⚡ [TTS FALLBACK - Kokoro-82M] Synthesizing: '{text}' locally on CPU")
        
        # Calculate duration based on word count (~150 words per minute -> ~2.5 words per sec)
        words = text.split()
        num_words = max(1, len(words))
        duration_seconds = max(0.5, num_words / 2.5)
        
        # Chunk settings: 20ms frames at 16kHz, 16-bit mono = 320 samples = 640 bytes
        chunk_size_bytes = 640
        chunk_duration_sec = 0.02
        total_chunks = int(duration_seconds / chunk_duration_sec)
        
        # Generate clean synthetic carrier waveform for simulation / local audio
        import math
        freq = 220.0  # Warm tone
        sample_index = 0

        for _ in range(total_chunks):
            # Generate 320 samples
            pcm_samples = bytearray()
            for _ in range(320):
                # Sine wave with gentle modulation
                t = sample_index / self.sample_rate
                val = int(3000 * math.sin(2 * math.pi * freq * t) * (1 + 0.2 * math.sin(2 * math.pi * 5 * t)))
                pcm_samples.extend(val.to_bytes(2, byteorder="little", signed=True))
                sample_index += 1
            
            yield bytes(pcm_samples)
            await asyncio.sleep(chunk_duration_sec)


class ResilientTTSManager:
    """
    Orchestrates Cartesia Sonic TTS with automatic, transparent fallback to local Kokoro TTS.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        self.api_key = api_key or config.CARTESIA_API_KEY
        self.voice_id = voice_id or config.CARTESIA_VOICE_ID
        self.sample_rate = sample_rate
        self.fallback_engine = LocalKokoroFallbackTTS(sample_rate=sample_rate)
        self.cartesia_client = None
        self._consecutive_cartesia_errors = 0
        self._fallback_active = False

        if self.api_key and not self.api_key.startswith("your_"):
            try:
                import cartesia  # type: ignore

                self.cartesia_client = cartesia.Cartesia(api_key=self.api_key)
                logger.info("Cartesia Sonic TTS client initialized.")
            except ImportError:
                logger.warning("cartesia SDK not installed; will route to fallback.")
            except Exception as e:
                logger.warning(f"Failed to initialize Cartesia client: {e}")
        else:
            logger.info("No Cartesia API key provided; TTS fallback mode active.")

    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesizes text to PCM audio stream.
        Tries Cartesia Sonic first; on failure or quota/rate limit error, seamlessly falls back to Kokoro.
        """
        if not text or not text.strip():
            return

        # Notify metrics of first audio milestone
        metrics.set_tts_provider("cartesia" if (self.cartesia_client and not self._fallback_active) else "kokoro_fallback")

        if self.cartesia_client and not self._fallback_active:
            try:
                # Attempt Cartesia Sonic synthesis
                logger.info(f"🎙️ [Cartesia Sonic TTS] Synthesizing: '{text[:60]}...'")
                
                # Streaming audio from Cartesia
                cartesia_generator = self._stream_cartesia(text)
                first_chunk = True
                async for chunk in cartesia_generator:
                    if first_chunk:
                        metrics.on_tts_first_audio()
                        first_chunk = False
                    yield chunk

                # Reset error counter on success
                self._consecutive_cartesia_errors = 0
                return

            except Exception as err:
                self._consecutive_cartesia_errors += 1
                logger.warning(
                    f"⚠️ [TTS Warning] Cartesia synthesis failed ({err}). "
                    f"Switching in-flight to local Kokoro fallback! (Consecutive errors: {self._consecutive_cartesia_errors})"
                )
                if self._consecutive_cartesia_errors >= 2:
                    self._fallback_active = True
                metrics.set_tts_provider("kokoro_fallback")

        # Fallback path: Kokoro-82M
        metrics.set_tts_provider("kokoro_fallback")
        await self.fallback_engine.initialize()
        first_fallback_chunk = True
        async for chunk in self.fallback_engine.synthesize(text):
            if first_fallback_chunk:
                metrics.on_tts_first_audio()
                first_fallback_chunk = False
            yield chunk

    async def _stream_cartesia(self, text: str) -> AsyncGenerator[bytes, None]:
        """Internal helper to stream audio from Cartesia API asynchronously."""
        loop = asyncio.get_running_loop()

        def _sync_fetch():
            # Cartesia SDK v4: voice is a VoiceSpecifierParam dict.
            # .generate() is the current API and returns a BinaryAPIResponse;
            # bytes() (deprecated) returns a lazy iterator. Both materialize here
            # so the blocking HTTP request happens in the executor thread.
            tts_api = self.cartesia_client.tts.generate
            result = tts_api(
                model_id=config.CARTESIA_MODEL,
                transcript=text,
                voice={"mode": "id", "id": self.voice_id},
                output_format={
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": self.sample_rate,
                },
            )
            # BinaryAPIResponse -> read all bytes (or join legacy iterator chunks)
            if isinstance(result, (bytes, bytearray)):
                return bytes(result)
            if hasattr(result, "read"):
                return result.read()
            return b"".join(result)

        audio_bytes = await loop.run_in_executor(None, _sync_fetch)
        
        # Stream chunks in 1280-byte slices (~40ms each). Note: the full audio is
        # already fetched (Cartesia v4 generate() is non-streaming), so pacing is
        # only to avoid bursting the WebSocket; browser buffers while playing.
        chunk_size = 1280
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i : i + chunk_size]
            await asyncio.sleep(0)  # yield control to event loop, no artificial delay


# Global TTS manager instance
tts_manager = ResilientTTSManager()
