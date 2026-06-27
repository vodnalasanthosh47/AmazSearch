"""
compliance_check_agent.py — Agent 4 in the pipeline.

Takes all products from product_search_agent (Agent 3), inspects each one using
amazon_product (for detailed specs/features) and google_search (for external
compliance/standard verification), then filters to return only products that
are compliant with the resolved specs_list.
"""

import sys
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import Runner
from google.adk.tools import google_search
from google.genai import types

# Ensure the tools package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.amazon_product import amazon_product

# pyrefly: ignore [missing-import]
from product_search_agent import ProductSearchResults, SearchResultProduct
# pyrefly: ignore [missing-import]
from first_agent import LLM_USED


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class CompliantProduct(BaseModel):
    """A single product that passed compliance checks."""
    asin: str = Field(description="Amazon Standard Identification Number.")
    title: str = Field(description="Product title.")
    brand: str | None = Field(default=None, description="Brand name.")
    price: str | None = Field(default=None, description="Formatted price.")
    rating: float | None = Field(default=None, description="Average rating 0.0–5.0.")
    ratings_count: int | None = Field(default=None, description="Total number of ratings.")
    image: str | None = Field(default=None, description="Primary thumbnail URL.")
    url: str | None = Field(default=None, description="Canonical Amazon product URL.")
    is_prime: bool = Field(default=False, description="Whether Prime delivery is available.")
    compliance_summary: str = Field(
        default="",
        description="Brief explanation of how the product meets each spec.",
    )


class CompliantProductGroup(BaseModel):
    """Compliant products for a single product category."""
    product: str = Field(description="The original product category name.")
    specs_list: List[str] = Field(
        default=[],
        description="The specs/constraints the products were checked against.",
    )
    compliant_products: List[CompliantProduct] = Field(
        default=[],
        description="Only the products that satisfy ALL specs in specs_list.",
    )


class ComplianceResults(BaseModel):
    """Final output: all compliant products across every category."""
    items: List[CompliantProductGroup] = Field(default=[])


# ---------------------------------------------------------------------------
# Sub-agent 4a: CHECKS each product for spec compliance.
# Has tools → must NOT have output_schema (ADK constraint).
# ---------------------------------------------------------------------------

COMPLIANCE_INSTRUCTION = """\
You receive a JSON object with Amazon search results grouped by product category,
shaped like:
{
  "items": [
    {
      "product": "...",
      "specs_list": ["spec1", "spec2", ...],
      "search_query": "...",
      "products": [
        {"asin": "...", "title": "...", "brand": "...", "price": "...", ...},
        ...
      ]
    },
    ...
  ]
}

Your job is to verify which products ACTUALLY comply with ALL specs in the
specs_list for their category. For EACH product category:

1. Go through each product in the "products" list.
2. For each product, call amazon_product with its ASIN to fetch detailed
   specifications, feature bullets, and description.
3. Cross-reference the detailed product info against the specs_list.
   If certain specs cannot be confirmed from Amazon data alone, use
   google_search to look up "<product title> <spec>" to verify.
4. A product is COMPLIANT only if it satisfies ALL specs in specs_list.
   Be reasonably flexible — e.g. a spec of "Bluetooth 5.0" should pass for
   a product advertising "Bluetooth 5.2" (which is backward compatible).
   But do NOT mark a product compliant if a key spec is clearly missing.
5. If a spec relates to price (e.g. "under ₹2000"), check the product's
   price field directly.

After checking all products, write a structured plain-text summary:

--- PRODUCT CATEGORY: <product name> ---
SPECS REQUIRED: <comma-separated specs_list>
COMPLIANT PRODUCTS:
1. ASIN: <asin> | Title: <title> | Brand: <brand> | Price: <price> | Rating: <rating> | Ratings Count: <ratings_count> | Image: <image_url> | URL: <url> | Prime: <yes/no> | Compliance: <brief summary of how each spec is met>
2. ...
NON-COMPLIANT (skipped): <comma-separated ASINs with brief reason>

IMPORTANT:
- Do NOT skip the amazon_product check — the search results alone do not
  contain enough detail to verify compliance.
- Do NOT invent specs. Only report what the tool returns.
- If NO products in a category pass, say "No compliant products found."
"""

compliance_check_subagent = Agent(
    name="compliance_check_subagent",
    model=LLM_USED,
    instruction=COMPLIANCE_INSTRUCTION,
    tools=[amazon_product, google_search],
    output_key="compliance_notes",
)


# ---------------------------------------------------------------------------
# Sub-agent 4b: STRUCTURES the compliance notes into strict schema.
# output_schema → must NOT have tools (ADK constraint).
# ---------------------------------------------------------------------------

STRUCTURE_INSTRUCTION = """\
You will receive plain-text compliance check results under the key
`compliance_notes`.

Convert these notes into the required structured format:
- One CompliantProductGroup per product category.
- `product` must match the original product category name exactly.
- `specs_list` must match the specs listed in the notes.
- `compliant_products` must include ONLY the products explicitly marked as
  compliant in the notes. Do NOT include non-compliant/skipped products.
- For each compliant product, populate all fields from the notes (use null
  for "N/A" values) and set `compliance_summary` to the compliance
  explanation given in the notes.
- If a category has no compliant products, set `compliant_products` to an
  empty list.

Compliance notes:
{compliance_notes}
"""

compliance_structurer_agent = Agent(
    name="compliance_structurer_agent",
    model=LLM_USED,
    instruction=STRUCTURE_INSTRUCTION,
    output_schema=ComplianceResults,
    output_key="compliance_results",
)


# ---------------------------------------------------------------------------
# Combined pipeline = "compliance_check_agent"
# ---------------------------------------------------------------------------

compliance_check_agent = SequentialAgent(
    name="compliance_check_agent",
    sub_agents=[compliance_check_subagent, compliance_structurer_agent],
)


# ---------------------------------------------------------------------------
# Helper to drive the agent from main_agent.py
# ---------------------------------------------------------------------------

async def check_compliance(
    runner: Runner,
    user_id: str,
    session_id: str,
    search_results: ProductSearchResults,
) -> ComplianceResults:
    """Feeds product_search_agent's output into compliance_check_agent and
    returns only the products that pass spec compliance for each category.

    The runner/session passed here must be wired to compliance_check_agent.
    """
    content = types.Content(
        role="user",
        parts=[types.Part(text=search_results.model_dump_json())],
    )

    async for _event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        pass

    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    # pyrefly: ignore [missing-attribute]
    results_json = session.state.get("compliance_results")
    if results_json is None:
        raise RuntimeError(
            "compliance_check_agent did not produce 'compliance_results' "
            "in session state."
        )

    if isinstance(results_json, str):
        return ComplianceResults.model_validate_json(results_json)
    return ComplianceResults.model_validate(results_json)
