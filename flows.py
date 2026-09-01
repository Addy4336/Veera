"""
Conversational State Engine and Flow Definitions for Vera Voice AI Agent.
Implements structured conversational nodes:
[Greeting] -> [Intent Capture] -> [Order Status | Schedule Callback | General FAQ] -> [Resolution] -> [Close]
"""

import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from config import config
from metrics import metrics
from tools import VERA_TOOLS_SCHEMA, execute_tool

logger = logging.getLogger(__name__)

# Curated Support FAQ Knowledge Base for Grounded Answers
FAQ_KNOWLEDGE_BASE = """
- Returns & Exchanges: Customers can return any unused item within 30 days of delivery for a full refund. Pre-paid return shipping labels are generated automatically on the website.
- Refund Processing: Once received at our fulfillment center, refunds take 3 to 5 business days to reflect on the original payment method.
- Shipping Windows: Standard ground shipping takes 3-5 business days. Express overnight shipping is available at checkout.
- Support Hours: Vera automated assistance is available 24/7. Live senior specialists are available Monday through Friday, 8 AM to 8 PM Eastern Time.
- Warranty: All electronics and hardware come with a complimentary 1-year limited manufacturer warranty.
"""

SYSTEM_BASE_PROMPT = f"""You are Vera, a friendly, ultra-fast, and professional customer support voice assistant for an e-commerce platform.

IMPORTANT VOICE CONVERSATION RULES:
1. Speak in short, concise, natural conversational sentences (1-3 sentences per turn maximum).
2. NEVER use markdown, asterisks, bullet points, or emoji. Your text is directly converted to speech.
3. Pronounce numbers and IDs clearly (e.g., say 'Order O R D 1 2 3 4 5' or 'Order one two three four five').
4. Always remain warm, empathetic, and solution-focused.

Company Knowledge & Policies:
{FAQ_KNOWLEDGE_BASE}
"""


def clean_text_for_speech(text: str) -> str:
    """Strip markdown symbols, brackets, and emojis for clean speech synthesis."""
    # Convert markdown links [text](url) -> text
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove standalone brackets
    cleaned = re.sub(r"[\[\]]", "", cleaned)
    # Remove markdown headers, bold, italics, code delimiters
    cleaned = re.sub(r"[*#_`~]", "", cleaned)
    # Replace remaining raw urls
    cleaned = re.sub(r"https?://\S+", "the tracking website", cleaned)
    # Normalize excessive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class FlowNode:
    """Base definition of a conversational state machine node."""

    def __init__(self, name: str, prompt_instruction: str):
        self.name = name
        self.prompt_instruction = prompt_instruction

    def get_node_prompt(self, context: Dict[str, Any]) -> str:
        return f"{SYSTEM_BASE_PROMPT}\n\nCurrent Conversation Stage: [{self.name}]\nGoal: {self.prompt_instruction}"


class FlowManager:
    """Manages conversational state, context memory, and LLM turn generation."""

    def __init__(self):
        self.current_node: str = "Greeting"
        self.messages: List[Dict[str, str]] = []
        self.context: Dict[str, Any] = {}
        self.groq_client = None

        if config.GROQ_API_KEY and not config.GROQ_API_KEY.startswith("your_"):
            try:
                from groq import AsyncGroq  # type: ignore

                self.groq_client = AsyncGroq(api_key=config.GROQ_API_KEY)
                logger.info("Groq LLM client initialized successfully.")
            except ImportError:
                logger.warning("groq package not found; running with simulation fallback.")
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}")

        # Initialize conversation with greeting
        self.reset()

    def reset(self) -> None:
        """Reset conversation state to initial Greeting node."""
        self.current_node = "Greeting"
        self.messages = [
            {
                "role": "system",
                "content": f"{SYSTEM_BASE_PROMPT}\nCurrent Stage: [Greeting]. Welcome the caller warmly to Vera Support.",
            }
        ]
        self.context = {}
        metrics.set_flow_node(self.current_node)

    def transition_to(self, new_node: str) -> None:
        """Transition conversation state to a new node."""
        logger.info(f"🔄 State Transition: [{self.current_node}] -> [{new_node}]")
        self.current_node = new_node
        metrics.set_flow_node(new_node)

    def _trim_history(self, max_turns: int = 20) -> None:
        """
        Bound stored conversation history (excluding the system message) so
        long calls don't grow unbounded. Keeps the most recent turns.
        """
        non_system = [m for m in self.messages if m["role"] != "system"]
        system = [m for m in self.messages if m["role"] == "system"]
        if len(non_system) > max_turns * 2:
            dropped = len(non_system) - max_turns * 2
            self.messages = system + non_system[dropped:]
            logger.debug(f"Trimmed {dropped} old conversation messages (bounded history).")

    def _determine_intent(self, text: str) -> str:
        """Heuristic intent classification to route nodes accurately."""
        lower = text.lower()
        if any(w in lower for w in ["order", "package", "tracking", "track", "delivery", "shipped", "where is my"]):
            return "OrderStatus"
        elif any(w in lower for w in ["call me", "callback", "schedule", "speak to human", "agent", "representative", "appointment"]):
            return "ScheduleCallback"
        elif any(w in lower for w in ["bye", "goodbye", "that's all", "nothing else", "no that is all", "have a good day"]):
            return "Close"
        elif any(w in lower for w in ["return", "refund", "hours", "warranty", "shipping time", "policy", "cost", "how long"]):
            return "GeneralFAQ"
        return "IntentCapture"

    async def process_user_turn(self, user_text: str) -> AsyncGenerator[str, None]:
        """
        Process user speech turn, manage state transitions, execute tools, and stream response tokens.
        """
        metrics.on_llm_prompt_sent()
        user_clean = clean_text_for_speech(user_text)
        self.messages.append({"role": "user", "content": user_clean})

        # Update node according to user intent
        next_intent = self._determine_intent(user_clean)
        if self.current_node in ["Greeting", "IntentCapture", "Resolution"]:
            if next_intent != "IntentCapture":
                self.transition_to(next_intent)
        elif self.current_node in ["OrderStatus", "ScheduleCallback", "GeneralFAQ"]:
            if next_intent == "Close":
                self.transition_to("Close")
            elif next_intent in ["OrderStatus", "ScheduleCallback", "GeneralFAQ"]:
                self.transition_to(next_intent)

        # Build dynamic node instruction
        node_instruction = (
            f"Current Node: [{self.current_node}]. Respond conversationally to the user. "
            f"If an order lookup or callback is requested, use the available tools."
        )
        current_messages = [
            {"role": "system", "content": f"{SYSTEM_BASE_PROMPT}\n{node_instruction}"}
        ] + self.messages[-6:]  # Keep last 6 turns for tight conversational context

        full_reply = ""
        first_token = True

        if self.groq_client:
            try:
                # Dispatched to Groq Llama 3.3 70B
                response = await self.groq_client.chat.completions.create(
                    model=config.GROQ_LLM_MODEL,
                    messages=current_messages,
                    tools=VERA_TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=0.4,
                    max_tokens=250,
                    stream=True,
                )

                tool_calls_to_run = []
                async for chunk in response:
                    delta = chunk.choices[0].delta

                    # Check for tool call streaming
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            if tc.function:
                                tool_calls_to_run.append(tc)

                    # Stream text tokens
                    if delta.content:
                        token = delta.content
                        if first_token:
                            metrics.on_llm_first_token(token)
                            first_token = False
                        full_reply += token
                        yield token

                # If tool calls were initiated
                if tool_calls_to_run:
                    for tc in tool_calls_to_run:
                        fn_name = tc.function.name or ""
                        fn_args = {}
                        if tc.function.arguments:
                            try:
                                fn_args = json.loads(tc.function.arguments)
                            except Exception:
                                fn_args = {}

                        # Execute tool
                        tool_result = await execute_tool(fn_name, fn_args)
                        metrics.on_tool_executed(fn_name, fn_args, tool_result)

                        # Generate conversational follow-up from tool output
                        spoken_summary = tool_result.get("spoken_summary", json.dumps(tool_result))
                        self.messages.append({"role": "assistant", "content": spoken_summary})
                        
                        if first_token:
                            metrics.on_llm_first_token(spoken_summary.split()[0])
                            first_token = False
                        
                        yield spoken_summary
                        full_reply = spoken_summary
                        self.transition_to("Resolution")
                        return

            except Exception as e:
                logger.error(f"Groq LLM streaming error: {e}")
                fallback_reply = await self._generate_simulated_reply(user_clean)
                for word in fallback_reply.split():
                    token = f"{word} "
                    if first_token:
                        metrics.on_llm_first_token(token)
                        first_token = False
                    full_reply += token
                    yield token
        else:
            # Simulation fallback when GROQ_API_KEY is not configured
            fallback_reply = await self._generate_simulated_reply(user_clean)
            for word in fallback_reply.split():
                token = f"{word} "
                if first_token:
                    metrics.on_llm_first_token(token)
                    first_token = False
                full_reply += token
                yield token

        self.messages.append({"role": "assistant", "content": clean_text_for_speech(full_reply)})
        self._trim_history()

        # Check if we should move to Resolution or Close
        if self.current_node in ["OrderStatus", "ScheduleCallback", "GeneralFAQ"]:
            self.transition_to("Resolution")
        elif self.current_node == "Close":
            logger.info("Conversation completed gracefully.")

    async def _generate_simulated_reply(self, user_text: str) -> str:
        """Provide intelligent deterministic simulation when offline."""
        lower = user_text.lower()
        if "order" in lower or "tracking" in lower or "ord-" in lower:
            self.transition_to("OrderStatus")
            match = re.search(r"(ord-?\d+|\d{4,5})", lower)
            order_id = match.group(1) if match else "ORD-12345"
            result = await execute_tool("order_status", {"order_id": order_id})
            metrics.on_tool_executed("order_status", {"order_id": order_id}, result)
            return result.get("spoken_summary", f"Your order {order_id} is out for delivery today.")

        elif "callback" in lower or "call me" in lower or "appointment" in lower:
            self.transition_to("ScheduleCallback")
            result = await execute_tool("schedule_callback", {"time_slot": "Tomorrow at 2 PM", "reason": "Support inquiry"})
            metrics.on_tool_executed("schedule_callback", {"time_slot": "Tomorrow at 2 PM"}, result)
            return result.get("spoken_summary", "I have scheduled a callback for tomorrow at 2 PM.")

        elif "return" in lower or "refund" in lower:
            self.transition_to("GeneralFAQ")
            return "Our return policy allows 30 days for a full refund with free return shipping. Is there anything else I can help with?"

        elif "bye" in lower or "thank" in lower:
            self.transition_to("Close")
            return "Thank you for contacting Vera Support. Have a wonderful day!"

        else:
            self.transition_to("IntentCapture")
            return "I can look up your order status, schedule a specialist callback, or answer questions about our returns. How can I help?"


# Global flow manager instance
flow_manager = FlowManager()
