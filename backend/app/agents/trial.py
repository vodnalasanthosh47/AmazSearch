"""
trial.py – Minimal Google ADK agent run via function call (no adk web UI needed).

Run with:
    /home/santhosh/pythonvenvs/amazsearch_venv/bin/python backend/app/agents/trial.py
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_env_path)

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# ── Tool ───────────────────────────────────────────────────────────────────────

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city: The name of the city.

    Returns:
        A dict with keys 'status', 'city', and 'time'.
    """
    # Mock implementation – swap in a real timezone lookup later.
    time_map = {
        "tokyo": "10:30 AM",
        "london": "02:30 AM",
        "new york": "09:30 PM",
    }
    time = time_map.get(city.lower(), "unknown")
    return {"status": "success", "city": city, "time": time}

# ── Agent definition ───────────────────────────────────────────────────────────

root_agent = Agent(
    model="gemini-2.5-flash",   # any valid Gemini model
    name="root_agent",
    description="Tells the current time in a specified city.",
    instruction=(
        "You are a helpful assistant that tells the current time in cities. "
        "Use the 'get_current_time' tool for this purpose."
    ),
    tools=[get_current_time],
)

# ── Runner constants ───────────────────────────────────────────────────────────

APP_NAME   = "trial_app"
USER_ID    = "user_001"
SESSION_ID = "session_001"

# ── Main async runner ──────────────────────────────────────────────────────────

async def call_agent(user_message: str) -> None:
    """Send a single user message to the agent and print the response."""

    # 1. Session service (in-memory; no DB needed for local testing)
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    # 2. Runner wires agent + session together
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # 3. Wrap the user message in ADK's Content/Part envelope
    message = types.Content(
        role="user",
        parts=[types.Part(text=user_message)],
    )

    print(f"\n>>> User: {user_message}")

    # 4. Stream events; only print the final text response
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=message,
    ):
        # is_final_response() marks the agent's last turn output
        if event.is_final_response():
            # Guard: content may be None on error/cancel events
            if event.content and event.content.parts:
                response_text = event.content.parts[0].text
                print(f"<<< Agent: {response_text}")
            else:
                print("<<< Agent: (no text content in final response)")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(call_agent("What time is it in Tokyo?"))