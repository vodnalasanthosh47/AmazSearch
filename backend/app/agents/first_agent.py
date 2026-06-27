import os
from typing import List
from pydantic import BaseModel, Field
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.genai import types

LLM_USED = os.environ.get("LLM_TO_USE", "gemini-2.5-flash")

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

# 4. Instantiate the Agent with native Pydantic enforcement
first_agent = Agent(
    name="first_agent",
    model=LLM_USED,
    instruction=SYSTEM_INSTRUCTION,
    output_schema=InferredSetupSpecs,
)


async def infer_products_from_vague_input(
    runner: Runner, user_id: str, session_id: str, user_prompt: str
) -> InferredSetupSpecs:
    """Sends one turn to first_agent (within an existing session/runner) and
    returns the validated Pydantic model."""
    content = types.Content(role="user", parts=[types.Part(text=user_prompt)])

    final_text = None
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text

    if final_text is None:
        raise RuntimeError("first_agent did not return a final response.")

    return InferredSetupSpecs.model_validate_json(final_text)