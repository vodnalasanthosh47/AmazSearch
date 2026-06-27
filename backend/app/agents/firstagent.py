import asyncio
import os
from typing import List
from pydantic import BaseModel, Field
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import os
from pathlib import Path

from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_env_path)

# ── Load API key ───────────────────────────────────────────────────────────────
# trial.py lives at  backend/app/agents/trial.py
# .env lives at      backend/.env  (3 levels up from this file)
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_env_path)


# 1. Define the Nested Product Structure
class ProductItem(BaseModel):
    product: str = Field(
        description="The guessed product category name required to fulfill the user's intent (e.g., 'Ergonomic Office Chair')."
    )
    constraint_list: List[str] = Field(
        default=[],
        description="Logical constraints or features inferred for this item based on the user's input."
    )


# 2. Define the Complete Array Object Schema that the UI will read
class InferredSetupSpecs(BaseModel):
    items: List[ProductItem] = Field(
        default=[],
        description="The complete broken-down list of guessed products and their associated constraints."
    )


# 3. System Instructions focused entirely on Intelligent Extrapolation
SYSTEM_INSTRUCTION = """
You are an expert product curation engine. Your job is to take broad, user inputs (e.g., 'set up my desk', 'build a home streaming studio', 'prepare for a college hostel room') and intelligently guess the specific products they will need.

Rules:
1. Break down the macro request into separate, logical individual product objects.
2. Constraints should only be either logisitics or price, only add constraints if user provides some specific information.
3. Do not ask clarifying questions, show conversational filler, or chat. Return strictly the structured format.
4. If the user is refining or correcting a previous request, treat it as an update to the earlier result, not a brand new unrelated request.
"""

APP_NAME = "amazesearch_app"
USER_ID = "demo_user"
SESSION_ID = "demo_session"

# 4. Instantiate the Agent with native Pydantic enforcement
inference_agent = Agent(
    name="first_agent",
    model="gemini-2.5-flash",
    instruction=SYSTEM_INSTRUCTION,
    output_schema=InferredSetupSpecs,
)


async def infer_products_from_vague_input(
    runner: Runner, user_prompt: str
) -> InferredSetupSpecs:
    """Sends one turn to the agent (within the existing session) and returns the validated Pydantic model."""
    content = types.Content(role="user", parts=[types.Part(text=user_prompt)])

    final_text = None
    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text

    if final_text is None:
        raise RuntimeError("Agent did not return a final response.")

    return InferredSetupSpecs.model_validate_json(final_text)


async def main():
    # Set up the session + runner ONCE, outside the loop, so the agent
    # keeps conversational memory across turns (e.g. "make it for home use"
    # refines the previous answer instead of starting from scratch).
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    runner = Runner(
        app_name=APP_NAME,
        agent=inference_agent,
        session_service=session_service,
    )

    print("Describe what you're setting up (e.g. 'beginner gym equipment').")
    print("Type 'exit' or 'quit' when you're satisfied with the result.\n")

    while True:
        user_prompt = input("You: ").strip()

        if user_prompt.lower() in {"exit", "quit", "q"}:
            print("\nDone. Final result was printed above.")
            break

        if not user_prompt:
            continue

        try:
            predicted_specs = await infer_products_from_vague_input(
                runner, user_prompt
            )
        except Exception as e:
            print(f"\n[Error generating specs: {e}]\n")
            continue

        print("\nGuessed Setup Structure:")
        print(predicted_specs.model_dump_json(indent=2))
        print()  # blank line for readability before next prompt


if __name__ == "__main__":
    # Ensure your environment variable is set: export GOOGLE_API_KEY="your-key"
    asyncio.run(main())