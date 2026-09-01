"""
Mock Business Tools for Vera Customer Support Voice Agent.
Implements:
1. order_status(order_id: str) - E-commerce package tracking and order state
2. schedule_callback(time_slot: str, phone_number: str, reason: str) - Support callback reservation
"""

import datetime
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Realistic Mock Database for Orders
MOCK_ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD-12345": {
        "order_id": "ORD-12345",
        "customer_name": "Alex Johnson",
        "status": "Out for Delivery",
        "carrier": "FedEx Priority",
        "tracking_number": "FX-889102391",
        "estimated_delivery": "Today by 4:30 PM",
        "delivery_address": "742 Evergreen Terrace, Springfield",
        "items": ["1x Wireless Noise-Cancelling Headphones (Midnight Black)"],
        "total_amount": "$149.99",
        "last_update": "Loaded on delivery vehicle at local distribution hub (8:15 AM)",
    },
    "ORD-9921": {
        "order_id": "ORD-9921",
        "customer_name": "Samantha Miller",
        "status": "In Transit",
        "carrier": "UPS Ground",
        "tracking_number": "1Z9999999999999999",
        "estimated_delivery": "Tomorrow by end of day",
        "delivery_address": "1204 Pine Street, Seattle, WA",
        "items": ["2x Ergonomic Desk Mounts", "1x USB-C Multiport Dock"],
        "total_amount": "$219.50",
        "last_update": "Departed sorting facility in Chicago, IL (Yesterday 11:20 PM)",
    },
    "ORD-8810": {
        "order_id": "ORD-8810",
        "customer_name": "Marcus Vance",
        "status": "Delivered",
        "carrier": "DHL Express",
        "tracking_number": "DHL-44019283",
        "estimated_delivery": "Delivered on August 14, 2026 at 2:15 PM",
        "delivery_address": "450 Ocean Drive, Miami, FL",
        "items": ["1x Smart Fitness Watch Series 5"],
        "total_amount": "$299.00",
        "last_update": "Delivered to front porch / parcel locker",
    },
    "ORD-7740": {
        "order_id": "ORD-7740",
        "customer_name": "Elena Rostova",
        "status": "Processing in Warehouse",
        "carrier": "USPS Priority",
        "tracking_number": "Pending generation",
        "estimated_delivery": "August 20, 2026",
        "delivery_address": "88 Market St, Austin, TX",
        "items": ["1x Mechanical Keyboard (Brown Switches)"],
        "total_amount": "$89.00",
        "last_update": "Order confirmed, payment verified, awaiting packing",
    },
}

# In-memory storage for booked callbacks
BOOKED_CALLBACKS: list[Dict[str, Any]] = []


async def order_status(order_id: str) -> Dict[str, Any]:
    """
    Look up the real-time shipping and delivery status of an order.

    Args:
        order_id: Order identifier (e.g. 'ORD-12345', 'ORD-9921', or numeric string '12345').

    Returns:
        Structured order information or a polite not-found response.
    """
    # Normalize ID: e.g. "12345" -> "ORD-12345"
    cleaned_id = order_id.strip().upper()
    if not cleaned_id.startswith("ORD-"):
        cleaned_id = f"ORD-{cleaned_id}"

    logger.info(f"[Tool: order_status] Looking up order: {cleaned_id}")

    if cleaned_id in MOCK_ORDERS:
        order = MOCK_ORDERS[cleaned_id]
        return {
            "found": True,
            "order_id": order["order_id"],
            "status": order["status"],
            "carrier": order["carrier"],
            "estimated_delivery": order["estimated_delivery"],
            "items_summary": ", ".join(order["items"]),
            "tracking_number": order["tracking_number"],
            "last_update": order["last_update"],
            "spoken_summary": (
                f"Order {order['order_id']} is currently {order['status'].lower()}. "
                f"It is being shipped via {order['carrier']} with estimated delivery {order['estimated_delivery']}."
            ),
        }
    else:
        # Dynamic fallback for generic IDs
        return {
            "found": False,
            "order_id": cleaned_id,
            "error": "Order not found in database",
            "spoken_summary": (
                f"I couldn't locate an active order for {cleaned_id}. "
                "Please double check the order number from your confirmation email, or I can connect you with a live specialist."
            ),
        }


async def schedule_callback(
    time_slot: str,
    phone_number: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Schedule a priority callback from a senior support specialist.

    Args:
        time_slot: Requested callback time or date (e.g. 'Tomorrow at 2 PM', 'in 30 minutes', 'Friday morning').
        phone_number: Contact phone number for the callback.
        reason: Brief reason or topic for the callback.

    Returns:
        Confirmation payload with booking reference ID.
    """
    logger.info(f"[Tool: schedule_callback] Booking callback for: {time_slot} (phone: {phone_number}, reason: {reason})")

    # Generate a unique confirmation ID
    confirmation_id = f"CB-{datetime.datetime.now().strftime('%M%S')}"
    specialist_name = "Senior Tech Support Team"

    record = {
        "confirmation_id": confirmation_id,
        "time_slot": time_slot,
        "phone_number": phone_number or "Calling number on file",
        "reason": reason or "General inquiry follow-up",
        "specialist": specialist_name,
        "status": "Confirmed",
        "booked_at": datetime.datetime.now().isoformat(),
        "spoken_summary": (
            f"I have scheduled your callback for {time_slot}. "
            f"Your reference number is {confirmation_id}. A member of our {specialist_name} will reach out to you."
        ),
    }

    BOOKED_CALLBACKS.append(record)
    return record


# OpenAI / Groq Compatible Tool Definitions Schema
VERA_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "order_status",
            "description": "Look up real-time shipping status, carrier tracking, ETA, and package location for a customer order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order tracking number or ID provided by customer (e.g. 'ORD-12345', '9921').",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_callback",
            "description": "Schedule a telephone callback with a senior support specialist at a requested date or time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_slot": {
                        "type": "string",
                        "description": "The preferred day and time for the callback (e.g. 'Tomorrow at 2 PM', 'in 1 hour', 'Monday 10 AM').",
                    },
                    "phone_number": {
                        "type": "string",
                        "description": "Optional contact telephone number if different from caller.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "The reason or issue for the callback request.",
                    },
                },
                "required": ["time_slot"],
            },
        },
    },
]


async def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatcher for tool execution."""
    if name == "order_status":
        order_id = arguments.get("order_id", "")
        return await order_status(order_id)
    elif name == "schedule_callback":
        time_slot = arguments.get("time_slot", "Next available slot")
        phone_number = arguments.get("phone_number")
        reason = arguments.get("reason")
        return await schedule_callback(time_slot=time_slot, phone_number=phone_number, reason=reason)
    else:
        logger.warning(f"Unknown tool requested: {name}")
        return {"error": f"Unknown tool: {name}"}
