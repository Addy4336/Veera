"""
Unit Tests for Vera Mock Tools (order_status and schedule_callback).
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
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import order_status, schedule_callback, execute_tool, MOCK_ORDERS


@pytest.mark.asyncio
async def test_order_status_valid_known():
    """Test order lookup for an existing order ID."""
    result = await order_status("ORD-12345")
    assert result["found"] is True
    assert result["order_id"] == "ORD-12345"
    assert result["status"] == "Out for Delivery"
    assert "FedEx" in result["carrier"]
    assert "spoken_summary" in result
    assert "out for delivery" in result["spoken_summary"].lower()


@pytest.mark.asyncio
async def test_order_status_raw_number():
    """Test order lookup with unformatted number like '9921' -> normalized to ORD-9921."""
    result = await order_status("9921")
    assert result["found"] is True
    assert result["order_id"] == "ORD-9921"
    assert result["status"] == "In Transit"


@pytest.mark.asyncio
async def test_order_status_not_found():
    """Test order lookup for non-existent order."""
    result = await order_status("ORD-00000")
    assert result["found"] is False
    assert "couldn't locate" in result["spoken_summary"].lower()


@pytest.mark.asyncio
async def test_schedule_callback_success():
    """Test callback scheduling tool."""
    result = await schedule_callback(
        time_slot="Tomorrow at 3 PM",
        phone_number="+1-555-0192",
        reason="Billing dispute",
    )
    assert result["status"] == "Confirmed"
    assert "CB-" in result["confirmation_id"]
    assert result["time_slot"] == "Tomorrow at 3 PM"
    assert "spoken_summary" in result
    assert "scheduled your callback" in result["spoken_summary"].lower()


@pytest.mark.asyncio
async def test_execute_tool_dispatcher():
    """Test generic tool execution dispatcher."""
    res1 = await execute_tool("order_status", {"order_id": "ORD-8810"})
    assert res1["found"] is True
    assert res1["status"] == "Delivered"

    res2 = await execute_tool("schedule_callback", {"time_slot": "Friday 10 AM"})
    assert res2["status"] == "Confirmed"

    res3 = await execute_tool("invalid_tool", {})
    assert "error" in res3


if __name__ == "__main__":
    asyncio.run(test_order_status_valid_known())
    asyncio.run(test_order_status_raw_number())
    asyncio.run(test_order_status_not_found())
    asyncio.run(test_schedule_callback_success())
    asyncio.run(test_execute_tool_dispatcher())
    print("[SUCCESS] All tool unit tests passed successfully!")
