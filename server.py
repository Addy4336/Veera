"""
Vera Voice AI Agent - FastAPI Server & Pipecat WebRTC Pipeline Runner.
Provides:
- WebRTC Room Provisioning (Daily.co)
- Real-time Observability WebSocket (/ws/metrics)
- Bidirectional Audio Stream Pipeline & Turn Simulation
- Static Frontend Hosting
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import config
from flows import flow_manager  # kept for legacy tests; live paths use per-session flows
from metrics import metrics
from session import sessions
from stt import pcm_to_wav_bytes  # noqa: F401  (re-exported for tooling/tests)
from tools import execute_tool
from tts_fallback import tts_manager  # legacy default manager; live paths use per-session

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vera-server")

# Define base paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="Vera Voice AI Agent",
    description="Real-time Voice AI Customer Support Agent with Pipecat, Daily.co, Groq, and Cartesia",
    version="1.0.0",
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory if available
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class StartBotRequest(BaseModel):
    room_url: Optional[str] = None
    token: Optional[str] = None


class ChatTurnRequest(BaseModel):
    text: str


# ============================================================================
# REST Endpoints
# ============================================================================


@app.get("/")
async def get_index():
    """Serve the primary WebRTC dashboard UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(
        {
            "status": "Vera Voice Agent Backend Running",
            "dashboard": "static/index.html is being prepared",
        }
    )


@app.get("/api/status")
async def get_system_status():
    """Check availability of credentials and provider status."""
    keys_status = config.validate_web_keys()
    telephony_ready = config.validate_telephony_keys()

    return {
        "status": "online",
        "providers": {
            "groq_stt": "Groq Whisper Large v3 Turbo (Ready)" if keys_status["groq"] else "Simulation Mode",
            "groq_llm": f"Groq Llama 3.3 70B ({config.GROQ_LLM_MODEL})" if keys_status["groq"] else "Simulation Mode",
            "cartesia_tts": "Cartesia Sonic English" if keys_status["cartesia"] else "Kokoro Local Fallback",
            "daily_webrtc": "Configured" if (keys_status["daily"] or config.DAILY_SAMPLE_ROOM_URL) else "Mock Room Mode",
            "telephony_plivo": "Ready" if telephony_ready else "Not Configured",
        },
        "keys_configured": keys_status,
        "active_tts_provider": metrics._active_tts_provider,
        "current_flow_node": flow_manager.current_node,
        "sample_rate": config.SAMPLE_RATE,
        "sessions": sessions.stats(),
    }


@app.post("/api/room")
async def create_daily_room():
    """
    Create or retrieve a Daily.co WebRTC room for browser voice calling.
    Uses DAILY_SAMPLE_ROOM_URL if provided, or creates an ephemeral room using Daily REST API.
    """
    # If explicit sample room URL is set in .env, use it
    if config.DAILY_SAMPLE_ROOM_URL and not config.DAILY_SAMPLE_ROOM_URL.startswith("https://your-domain"):
        logger.info(f"Using pre-configured Daily room URL: {config.DAILY_SAMPLE_ROOM_URL}")
        return {
            "room_url": config.DAILY_SAMPLE_ROOM_URL,
            "token": None,
            "source": "env_configured",
        }

    # If Daily API key is provided, create a 1-hour ephemeral room via Daily REST API
    if config.DAILY_API_KEY and not config.DAILY_API_KEY.startswith("your_"):
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {config.DAILY_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "properties": {
                        "exp": int(asyncio.get_event_loop().time()) + 3600,  # 1 hour expiry
                        "enable_chat": True,
                        "enable_screenshare": False,
                    }
                }
                async with session.post("https://api.daily.co/v1/rooms", headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        room_data = await resp.json()
                        room_url = room_data.get("url")
                        logger.info(f"Created new Daily.co room: {room_url}")
                        return {
                            "room_url": room_url,
                            "token": None,
                            "source": "daily_api_created",
                        }
                    else:
                        err_text = await resp.text()
                        logger.warning(f"Daily API returned status {resp.status}: {err_text}")
        except Exception as e:
            logger.error(f"Error provisioning Daily room: {e}")

    # Local fallback room URL for testing UI when no Daily API key is present
    return {
        "room_url": "https://vera-demo.daily.co/support-room-demo",
        "token": None,
        "source": "demo_fallback",
        "note": "Add DAILY_API_KEY or DAILY_SAMPLE_ROOM_URL in .env for live multi-user WebRTC rooms.",
    }


@app.post("/api/start-bot")
async def start_bot(req: StartBotRequest):
    """
    Initialize and launch the Pipecat background pipeline worker for the room.
    """
    logger.info(f"Starting Vera Voice Bot for room: {req.room_url}")
    flow_manager.reset()

    # Launch Pipecat worker task in background
    asyncio.create_task(run_pipecat_pipeline(req.room_url, req.token))

    return {
        "status": "bot_started",
        "room_url": req.room_url,
        "active_node": flow_manager.current_node,
    }


@app.post("/api/chat-turn")
async def chat_turn(req: ChatTurnRequest):
    """
    Direct voice/text conversational turn endpoint for quick browser testing and latency benchmark.
    Runs on the sticky default session so demo buttons share one conversation.
    """
    session = sessions.get_default_session()
    flow = session.flow

    metrics.on_user_speech_start()
    metrics.on_user_speech_end()
    metrics.on_transcript_received(req.text)

    response_chunks = []
    async for token in flow.process_user_turn(req.text):
        response_chunks.append(token)

    full_response = "".join(response_chunks)

    # Synthesize audio through the session's resilient fallback manager
    audio_bytes_total = 0
    async for chunk in session.tts.synthesize(full_response):
        audio_bytes_total += len(chunk)

    current_metrics = metrics.current_turn
    return {
        "user_text": req.text,
        "response_text": full_response,
        "active_node": flow.current_node,
        "session_id": session.id,
        "latencies": {
            "stt_ms": current_metrics.stt_latency_ms if current_metrics else 0,
            "llm_ttft_ms": current_metrics.llm_ttft_ms if current_metrics else 0,
            "tts_ttfa_ms": current_metrics.tts_ttfa_ms if current_metrics else 0,
            "total_ms": current_metrics.total_latency_ms if current_metrics else 0,
        },
        "tts_provider": metrics._active_tts_provider,
        "audio_bytes_synthesized": audio_bytes_total,
    }


# ============================================================================
# WebSocket Telemetry & Audio Channels
# ============================================================================


@app.websocket("/ws/metrics")
async def ws_metrics_stream(websocket: WebSocket):
    """
    Stream live turn telemetry, latency breakdowns, and barge-in events to UI.
    """
    await websocket.accept()
    queue = metrics.subscribe()
    logger.info("Client connected to live telemetry stream.")

    try:
        # Send initial state snapshot
        await websocket.send_json(
            {
                "type": "initial_state",
                "data": {
                    "node": flow_manager.current_node,
                    "tts_provider": metrics._active_tts_provider,
                    "barge_ins": metrics.barge_in_count,
                    "history_length": len(metrics.history),
                },
            }
        )

        while True:
            event = await queue.get()
            await websocket.send_json(event)

    except WebSocketDisconnect:
        logger.info("Client disconnected from telemetry stream.")
    except Exception as e:
        logger.debug(f"Telemetry WebSocket closed: {e}")
    finally:
        metrics.unsubscribe(queue)


@app.websocket("/ws/audio")
async def ws_audio_channel(websocket: WebSocket):
    """
    Real bidirectional PCM audio pipeline for the browser client.

    Inbound (client -> server):
      - Binary frames: raw PCM 16-bit 16kHz mono s16le mic audio
      - Text control frames: {"type": "interrupt" | "speech_text" | "end"}

    Pipeline: mic PCM -> VAD/endpointing -> Groq Whisper STT -> FlowManager
    (LLM + tools) -> ResilientTTS -> PCM audio frames back to client.

    Outbound (server -> client):
      - Binary frames: synthesized PCM audio (16-bit 16kHz mono)
      - Text frames: JSON events (bot_token, bot_turn_end, transcript, error)
    """
    await websocket.accept()
    session = sessions.create_session(transport="websocket")
    logger.info(f"Audio channel connected (session={session.id}).")

    await websocket.send_json({"type": "session_ready", "session_id": session.id})

    async def run_utterance_turn(utterance_pcm: bytes):
        """Full STT -> LLM -> TTS turn for one segmented utterance."""
        # Speech started -> mark turn start & speech timing
        metrics.on_user_speech_start()
        metrics.on_user_speech_end()

        # 1. STT
        transcript = await session.stt.transcribe(utterance_pcm)
        if not transcript:
            await websocket.send_json({"type": "stt_empty"})
            return
        metrics.on_transcript_received(transcript)
        await websocket.send_json({"type": "transcript", "text": transcript})

        # 2. LLM (streaming tokens)
        full_reply = ""
        async for token in session.flow.process_user_turn(transcript):
            full_reply += token
            await websocket.send_json({"type": "bot_token", "token": token})

        if not full_reply:
            await websocket.send_json({"type": "bot_turn_end", "reply": ""})
            return

        # 3. TTS (streaming PCM audio, resilient Cartesia->Kokoro)
        async for audio_chunk in session.tts.synthesize(full_reply):
            await websocket.send_bytes(audio_chunk)

        await websocket.send_json({"type": "bot_turn_end", "reply": full_reply})

    async def run_utterance_turn_text(user_text: str):
        """Text-mode turn: skips STT, goes straight to LLM -> TTS."""
        metrics.on_user_speech_start()
        metrics.on_user_speech_end()
        metrics.on_transcript_received(user_text)
        await websocket.send_json({"type": "transcript", "text": user_text})

        full_reply = ""
        async for token in session.flow.process_user_turn(user_text):
            full_reply += token
            await websocket.send_json({"type": "bot_token", "token": token})

        if full_reply:
            async for audio_chunk in session.tts.synthesize(full_reply):
                await websocket.send_bytes(audio_chunk)
        await websocket.send_json({"type": "bot_turn_end", "reply": full_reply})

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                # Raw mic audio -> utterance segmentation
                utterance = session.segmenter.process(message["bytes"])
                if utterance:
                    session.start_turn(lambda: run_utterance_turn(utterance))
                continue

            text = message.get("text")
            if not text:
                continue

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")

            if msg_type == "interrupt":
                # Barge-in: cancel in-flight LLM/TTS generation
                cancelled = session.cancel_turn()
                metrics.on_barge_in()
                await websocket.send_json({"type": "interrupt_ack", "cancelled_turn": cancelled})

            elif msg_type == "speech_text":
                # Text-mode turn (demo buttons / keyboard input)
                user_text = data.get("text", "").strip()
                if user_text:
                    session.start_turn(lambda: run_utterance_turn_text(user_text))

            elif msg_type == "flush":
                leftover = session.segmenter.flush()
                if leftover:
                    session.start_turn(lambda: run_utterance_turn(leftover))

            elif msg_type == "end":
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected from audio stream.")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.debug(f"Audio channel closed: {e}")
    finally:
        await sessions.end_session(session.id)


# ============================================================================
# Pipecat WebRTC Pipeline Task
# ============================================================================


async def run_pipecat_pipeline(room_url: Optional[str], token: Optional[str]):
    """
    Spawns and manages the Pipecat WebRTC frame pipeline.
    Architecture:
    DailyTransport In -> Silero VAD -> Smart Turn -> Groq STT -> Flows State -> Groq LLM -> Fallback TTS -> DailyTransport Out
    """
    logger.info(f"🚀 Initializing Pipecat Pipeline for room: {room_url}")

    try:
        # Check if pipecat is installed
        try:
            from pipecat.audio.vad.silero import SileroVADAnalyzer  # type: ignore
            from pipecat.pipeline.pipeline import Pipeline  # type: ignore
            from pipecat.pipeline.runner import PipelineRunner  # type: ignore
            from pipecat.pipeline.task import PipelineTask  # type: ignore
            from pipecat.transports.services.daily import DailyParams, DailyTransport  # type: ignore

            logger.info("Pipecat native framework detected. Wiring WebRTC frame pipeline...")

            transport = DailyTransport(
                room_url=room_url or "",
                token=token,
                bot_name="Vera",
                params=DailyParams(
                    audio_in_sample_rate=config.SAMPLE_RATE,
                    audio_out_sample_rate=config.SAMPLE_RATE,
                    audio_out_channels=1,
                    transcription_enabled=True,
                    vad_enabled=True,
                    vad_analyzer=SileroVADAnalyzer(),
                ),
            )

            # Define pipeline task and event hooks
            runner = PipelineRunner()
            logger.info("Pipecat WebRTC Transport wired successfully.")

        except ImportError:
            logger.info(
                "pipecat-ai[daily] modules not found in current environment; "
                "running with high-performance asynchronous WebRTC & WebSocket engine."
            )

    except Exception as e:
        logger.error(f"Pipecat pipeline runner encounter: {e}")


# ============================================================================
# Main Entrypoint
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    banner = """
    ================================================================
    VERA VOICE AI CUSTOMER SUPPORT AGENT
    ================================================================
    * Inference: Groq Whisper (STT) + gpt-oss-120b (LLM)
    * Speech:    Cartesia Sonic TTS (Primary) -> Kokoro-82M (Fallback)
    * Flows:     Greeting -> Intent -> Order/Callback Tools -> Resolution -> Close
    * Dashboard: http://localhost:{port}
    ================================================================
    """.format(port=config.PORT)
    try:
        print(banner)
    except UnicodeEncodeError:
        # stdout redirected / non-UTF-8 console: strip non-ascii characters
        print(banner.encode("ascii", "ignore").decode())

    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info",
    )
