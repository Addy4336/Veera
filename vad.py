"""
Voice Activity Detection and Utterance Segmentation for Vera Voice AI Agent.

Provides:
1. EnergyVAD - lightweight, zero-dependency energy-based endpointing (default)
2. UtteranceSegmenter - accumulates raw PCM and emits complete speech utterances

Optional: if pipecat-ai (with Silero) is importable, `create_vad()` returns a
Silero-backed analyzer wrapper for higher-accuracy turn detection.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def pcm_bytes_to_samples(pcm: bytes) -> list[int]:
    """Convert little-endian 16-bit PCM bytes to a list of integer samples."""
    if len(pcm) < 2:
        return []
    return list(memoryview(pcm).cast("h"))


class EnergyVAD:
    """
    Frame-based energy Voice Activity Detector.
    Decides speech/silence per 20ms frame using RMS threshold.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        energy_threshold: int = 550,
        min_speech_ms: int = 200,   # speech must persist this long to open
        min_silence_ms: int = 350,  # silence must persist this long to close
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_ms / 1000)  # samples per frame
        self.energy_threshold = energy_threshold
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.min_silence_frames = max(1, min_silence_ms // frame_ms)

        self._in_speech = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0

    def reset(self) -> None:
        self._in_speech = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0

    def frame_is_speech(self, samples: list[int]) -> bool:
        """Compute RMS energy of a frame and compare against threshold."""
        if not samples:
            return False
        total = 0
        for s in samples:
            total += s * s
        rms = (total // len(samples)) ** 0.5
        return rms > self.energy_threshold

    def process_frame(self, frame_pcm: bytes) -> Optional[str]:
        """
        Process one fixed-size PCM frame.
        Returns 'speech_start', 'speech_end', or None.
        """
        samples = pcm_bytes_to_samples(frame_pcm)
        is_speech = self.frame_is_speech(samples)

        if is_speech:
            self._consecutive_speech += 1
            self._consecutive_silence = 0
            if not self._in_speech and self._consecutive_speech >= self.min_speech_frames:
                self._in_speech = True
                return "speech_start"
        else:
            self._consecutive_silence += 1
            self._consecutive_speech = 0
            if self._in_speech and self._consecutive_silence >= self.min_silence_frames:
                self._in_speech = False
                self._consecutive_silence = 0
                return "speech_end"
        return None


class UtteranceSegmenter:
    """
    Streams raw PCM audio in, emits complete user utterances (bytes) out.
    Uses a VAD for endpointing; keeps a pre-roll buffer so speech onsets
    are not clipped.
    """

    def __init__(self, vad: Optional[EnergyVAD] = None, sample_rate: int = 16000, max_utterance_ms: int = 15000):
        self.vad = vad or EnergyVAD(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.frame_size_bytes = self.vad.frame_size * 2  # 16-bit mono
        self.max_utterance_bytes = int(sample_rate / 1000 * max_utterance_ms) * 2

        self._buffer = bytearray()
        self._preroll = bytearray(max(1, 250 // 20) * self.frame_size_bytes)  # ~250ms
        self._collecting = False

    def reset(self) -> None:
        self.vad.reset()
        self._buffer.clear()
        self._collecting = False

    def process(self, pcm: bytes) -> Optional[bytes]:
        """
        Feed an arbitrary-size PCM chunk.
        Returns the complete utterance bytes when speech ends (or max length hit), else None.
        """
        result: Optional[bytes] = None
        offset = 0
        while offset < len(pcm):
            frame = pcm[offset : offset + self.frame_size_bytes]
            offset += self.frame_size_bytes
            if len(frame) < self.frame_size_bytes:
                break  # drop tiny trailing partial frame (client sends aligned chunks)

            event = self.vad.process_frame(frame)

            if self._collecting:
                self._buffer.extend(frame)
                if event == "speech_end" or len(self._buffer) >= self.max_utterance_bytes:
                    result = bytes(self._buffer)
                    self._buffer.clear()
                    self._collecting = False
            else:
                self._preroll = self._preroll[self.frame_size_bytes :] + frame
                if event == "speech_start":
                    self._collecting = True
                    self._buffer.extend(self._preroll)

        return result

    def flush(self) -> Optional[bytes]:
        """Return any in-progress utterance (used when the stream ends)."""
        if self._collecting and len(self._buffer) > self.frame_size_bytes * 5:
            result = bytes(self._buffer)
            self._buffer.clear()
            self._collecting = False
            return result
        self._buffer.clear()
        self._collecting = False
        return None


def create_vad(sample_rate: int = 16000):
    """
    Factory: returns SileroVAD-backed wrapper if pipecat is available,
    otherwise the zero-dependency EnergyVAD.
    """
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer  # type: ignore

        analyzer = SileroVADAnalyzer(sample_rate=sample_rate)
        logger.info("Silero VAD analyzer loaded (pipecat detected).")
        return SileroVADWrapper(analyzer, sample_rate=sample_rate)
    except Exception:
        logger.info("Silero/pipecat unavailable; using built-in Energy VAD.")
        return EnergyVAD(sample_rate=sample_rate)


class SileroVADWrapper:
    """Minimal adapter so Silero can be swapped in wherever EnergyVAD is used."""

    def __init__(self, analyzer, sample_rate: int = 16000):
        self._analyzer = analyzer
        self.sample_rate = sample_rate
        try:
            self.frame_size = analyzer._framesize  # pipecat internal
        except AttributeError:
            self.frame_size = int(sample_rate * 0.03)

    def reset(self) -> None:
        pass

    def frame_is_speech(self, samples: list[int]) -> bool:
        import struct

        pcm = struct.pack(f"<{len(samples)}h", *samples)
        try:
            result = self._analyzer.analyze_audio(pcm)
            state = str(getattr(result, "state", result)).lower()
            return "speech" in state and "off" not in state
        except Exception:
            return False

    def process_frame(self, frame_pcm: bytes) -> Optional[str]:
        samples = pcm_bytes_to_samples(frame_pcm)
        is_speech = self.frame_is_speech(samples)
        if not hasattr(self, "_was_speech"):
            self._was_speech = False
        prev = self._was_speech
        self._was_speech = is_speech
        if is_speech and not prev:
            return "speech_start"
        if not is_speech and prev:
            return "speech_end"
        return None
