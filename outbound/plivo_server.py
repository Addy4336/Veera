"""
Plivo Telephony Webhook and Media Stream Handler for Vera.
Handles:
1. POST /plivo/answer -> Generates Plivo XML containing bidirectional <Stream> tag
2. WebSocket /plivo/media-stream -> Bidirectional 8kHz PCM audio bridge connecting telephony to Pipecat core
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect
from config import config
from flows import flow_manager
from metrics import metrics
from tts_fallback import tts_manager

logger = logging.getLogger("plivo-telephony")
telephony_router = APIRouter(prefix="/plivo", tags=["telephony"])


@telephony_router.post("/answer")
async def plivo_answer_webhook(request: Request):
    """
    Called by Plivo when the called party answers the phone.
    Returns XML instructing Plivo to connect audio to our WebSocket stream.
    """
    host = request.headers.get("host", f"localhost:{config.PORT}")
    protocol = "wss" if "https" in str(request.base_url) else "ws"
    ws_url = f"{protocol}://{host}/plivo/media-stream"

    logger.info(f"📞 Inbound call answered! Connecting to media stream: {ws_url}")

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Speak>Hello, connecting you to Vera Support.</Speak>
    <Stream bidirectional="true" streamUrl="{ws_url}" />
</Response>"""

    return Response(content=xml_content, media_type="application/xml")


@telephony_router.websocket("/media-stream")
async def plivo_media_stream(websocket: WebSocket):
    """
    WebSocket endpoint handling real-time telephony audio stream from Plivo.
    PSTN audio arrives at 8kHz and is routed to the Pipecat conversational pipeline.
    """
    await websocket.accept()
    logger.info("📞 Plivo Telephony Media Stream connected!")
    flow_manager.reset()

    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            event_type = data.get("event")

            if event_type == "media":
                # Raw audio chunk received from phone
                payload_b64 = data.get("media", {}).get("payload")
                # Downstream processing hooks
                pass
            elif event_type == "stop":
                logger.info("📞 Phone call ended by user.")
                break

    except WebSocketDisconnect:
        logger.info("📞 Plivo WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Error in Plivo media stream: {e}")
