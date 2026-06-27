from typing import List
from pydantic import BaseModel, Field
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.genai import types

# pyrefly: ignore [missing-import]
from first_agent import InferredSetupSpecs


# ---------------------------------------------------------------------------
# Output schema for this agent
# ---------------------------------------------------------------------------

class ResolvedProduct(BaseModel):
    product: str = Field(description="The same product name passed in, unchanged.")
    specs_list: List[str] = Field(
        default=[],
        description=(
            "Concrete, real-world specs/standards that satisfy the input "
            "constraints for this product — and ONLY those constraints. "
            "No extra specs beyond what was asked for."
        ),
    )


class ResolvedSetupSpecs(BaseModel):
    items: List[ResolvedProduct] = Field(default=[])


# ---------------------------------------------------------------------------
# Sub-agent 2a: SEARCHES the web for real specs.
# Has tools -> must NOT have output_schema (ADK constraint).
# ---------------------------------------------------------------------------

SEARCH_INSTRUCTION = """
The user's message will contain a JSON object describing products and constraints,
shaped like:
{"items": [{"product": "...", "constraint_list": ["..."]}, ...]}

For EACH item in that JSON:
1. Use google_search to research what real-world specs, standards, or features
   satisfy ONLY the constraints listed for that product. Do not research or invent
   anything beyond what each constraint asks for.
2. Note concrete, verifiable specs/standards (e.g. "30W USB-C PD"),
   not vague marketing language.
3. Do not add unnecessary seller brands in your spec_list, unless the constraints specifies the prefered seller/sellers.
4. Make sure user's preferences are included in the specs_list. (e.g. for men/women, colour)

Write your findings as a clear plain-text summary, one section per product. For each
product, restate its product name and original constraint_list EXACTLY as given in the
input JSON, followed by the specific specs/standards you found that map directly to
those constraints. Do not output JSON. Do not add specs unrelated to the given
constraints.
"""

spec_search_agent = Agent(
    name="spec_search_agent",
    model="gemini-2.5-flash",
    instruction=SEARCH_INSTRUCTION,
    tools=[google_search],
    output_key="spec_research_notes",
)


# ---------------------------------------------------------------------------
# Sub-agent 2b: STRUCTURES the research notes into the strict schema.
# output_schema -> must NOT have tools (ADK constraint).
# ---------------------------------------------------------------------------

STRUCTURE_INSTRUCTION = """
You will receive plain-text research notes under the key `spec_research_notes`. These
notes restate, for each product, its original product name and constraint_list, followed
by researched specs that satisfy those constraints. Convert these notes into the
required structured format:

- One object per product.
- `product` must match the original product name exactly as stated in the notes.
- `specs_list` must contain ONLY specs that map to a constraint actually listed for
  that product in the notes. Do not include extra specs the notes mention that go
  beyond the stated constraints. Do not omit a constraint that was clearly resolved
  in the notes.

Research notes:
{spec_research_notes}
"""

spec_structurer_agent = Agent(
    name="spec_structurer_agent",
    model="gemini-2.5-flash",
    instruction=STRUCTURE_INSTRUCTION,
    output_schema=ResolvedSetupSpecs,
    output_key="resolved_specs",
)


# ---------------------------------------------------------------------------
# Combine into one pipeline = "spec_resolution_agent"
# ---------------------------------------------------------------------------

spec_resolution_agent = SequentialAgent(
    name="spec_resolution_agent",
    sub_agents=[spec_search_agent, spec_structurer_agent],
)


async def resolve_specs(
    runner: Runner,
    user_id: str,
    session_id: str,
    inferred_specs: InferredSetupSpecs,
) -> ResolvedSetupSpecs:
    """Feeds first_agent's output into the spec_resolution_agent pipeline
    (running within a session that belongs to spec_resolution_agent's own
    runner) and returns the validated, structured result.

    The session/runner passed in here must be wired to spec_resolution_agent
    (not first_agent) and the session_id should be one that has only ever been
    driven by spec_resolution_agent, to avoid ADK's "event from an unknown
    agent" cross-session issue.
    """
    content = types.Content(
        role="user",
        parts=[types.Part(text=inferred_specs.model_dump_json())],
    )

    async for _event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        pass

    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    # pyrefly: ignore [missing-attribute]
    resolved_json = session.state.get("resolved_specs")
    if resolved_json is None:
        raise RuntimeError(
            "spec_resolution_agent did not produce 'resolved_specs' in session state."
        )

    if isinstance(resolved_json, str):
        return ResolvedSetupSpecs.model_validate_json(resolved_json)
    return ResolvedSetupSpecs.model_validate(resolved_json)
