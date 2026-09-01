"""
Plivo Outbound Telephony Dialer for Vera Voice AI Agent (Stretch Goal).
Initiates an outbound PSTN phone call and connects the audio stream to the Pipecat server.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("plivo-dialer")


def dial_outbound_call(to_number: str, webhook_host: str):
    """
    Trigger an outbound call using the Plivo REST API.

    Args:
        to_number: Recipient phone number in E.164 format (e.g. +14155552671 or +919876543210).
        webhook_host: Publicly accessible URL (e.g., ngrok tunnel or production server domain).
    """
    if not config.validate_telephony_keys():
        logger.error(
            "Plivo credentials missing! Please configure PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, "
            "and PLIVO_PHONE_NUMBER in your .env file."
        )
        return False

    try:
        import plivo  # type: ignore

        client = plivo.RestClient(
            auth_id=config.PLIVO_AUTH_ID,
            auth_token=config.PLIVO_AUTH_TOKEN,
        )

        answer_url = f"https://{webhook_host.strip('/')}/plivo/answer"
        logger.info(f"📞 Initiating outbound call to {to_number} from {config.PLIVO_PHONE_NUMBER}")
        logger.info(f"Answer Webhook: {answer_url}")

        response = client.calls.create(
            from_=config.PLIVO_PHONE_NUMBER,
            to_=to_number,
            answer_url=answer_url,
            answer_method="POST",
        )

        logger.info(f"✅ Call dispatched successfully! Plivo Call UUID: {response.get('request_uuid')}")
        return True

    except ImportError:
        logger.error("The 'plivo' Python package is not installed. Install via: pip install plivo")
        return False
    except Exception as e:
        logger.error(f"Failed to place Plivo outbound call: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vera Voice Agent - Outbound Phone Call Trigger")
    parser.add_argument("--to", required=True, help="Destination phone number (E.164 format)")
    parser.add_argument("--host", required=True, help="Public server host (e.g. my-app.ngrok-free.app)")

    args = parser.parse_args()
    dial_outbound_call(to_number=args.to, webhook_host=args.host)
