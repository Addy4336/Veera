"""
Centralized Configuration and Environment Management for Vera Voice AI Agent.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root if it exists
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class Config:
    """Application configuration and provider settings."""

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")

    # Groq Settings (Whisper STT + Llama LLM)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_LLM_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_STT_MODEL: str = "whisper-large-v3-turbo"

    # Cartesia TTS Settings
    CARTESIA_API_KEY: str = os.getenv("CARTESIA_API_KEY", "")
    CARTESIA_VOICE_ID: str = os.getenv("CARTESIA_VOICE_ID", "79a125e8-cd45-4c13-8a67-188112f4dd22")
    CARTESIA_MODEL: str = os.getenv("CARTESIA_MODEL", "sonic-2")

    # Daily.co WebRTC Settings
    DAILY_API_KEY: str = os.getenv("DAILY_API_KEY", "")
    DAILY_SAMPLE_ROOM_URL: str = os.getenv("DAILY_SAMPLE_ROOM_URL", "")

    # Telephony Settings (Plivo)
    PLIVO_AUTH_ID: str = os.getenv("PLIVO_AUTH_ID", "")
    PLIVO_AUTH_TOKEN: str = os.getenv("PLIVO_AUTH_TOKEN", "")
    PLIVO_PHONE_NUMBER: str = os.getenv("PLIVO_PHONE_NUMBER", "")

    # Audio Pipeline Settings
    SAMPLE_RATE: int = 16000  # 16kHz for WebRTC & Smart Turn v2
    TELEPHONY_SAMPLE_RATE: int = 8000  # 8kHz for PSTN calls
    AUDIO_CHANNELS: int = 1

    # Fallback and Resilience Configuration
    ENABLE_TTS_FALLBACK: bool = os.getenv("ENABLE_TTS_FALLBACK", "true").lower() in ("true", "1", "yes")
    TTS_FALLBACK_PROVIDER: str = os.getenv("TTS_FALLBACK_PROVIDER", "kokoro")

    @classmethod
    def validate_web_keys(cls) -> dict[str, bool]:
        """Check availability of credentials needed for WebRTC voice pipeline."""
        return {
            "groq": bool(cls.GROQ_API_KEY and not cls.GROQ_API_KEY.startswith("your_")),
            "cartesia": bool(cls.CARTESIA_API_KEY and not cls.CARTESIA_API_KEY.startswith("your_")),
            "daily": bool(cls.DAILY_API_KEY and not cls.DAILY_API_KEY.startswith("your_")),
        }

    @classmethod
    def validate_telephony_keys(cls) -> bool:
        """Check availability of credentials for Plivo telephony path."""
        return bool(
            cls.PLIVO_AUTH_ID
            and not cls.PLIVO_AUTH_ID.startswith("your_")
            and cls.PLIVO_AUTH_TOKEN
            and not cls.PLIVO_AUTH_TOKEN.startswith("your_")
        )


config = Config()


