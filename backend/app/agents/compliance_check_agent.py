"""
compliance_check_agent.py — Agent 4 in the pipeline.

Takes all products from product_search_agent (Agent 3), enriches each one
with detailed specs via amazon_product (called programmatically), then uses
an LLM with google_search to verify compliance against the specs_list.
Returns only products that are compliant.

Architecture:
  - Python code calls amazon_product for each ASIN (no LLM needed for data
    fetching → avoids the built-in-tool + function-calling mixing error).
  - LLM agent with ONLY google_search verifies compliance using the enriched
    product data.
  - Structurer agent converts notes to strict schema.
"""

import json
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
    rating: float | None = Field(default=None, description="Average rating 0.0-5.0.")
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
# Sub-agent 4a: VERIFIES compliance using google_search ONLY.
# amazon_product details are already embedded in the input by the helper.
# Has tools (google_search only) → must NOT have output_schema.
# ---------------------------------------------------------------------------

COMPLIANCE_INSTRUCTION = """\
You receive a JSON object with Amazon products grouped by category. Each product
has already been enriched with detailed information (description, feature_bullets,
specifications) fetched from Amazon. The data is shaped like:

{
  "items": [
    {
      "product": "...",
      "specs_list": ["spec1", "spec2", ...],
      "products": [
        {
          "asin": "...",
          "title": "...",
          "brand": "...",
          "price": "...",
          "rating": ...,
          "ratings_count": ...,
          "image": "...",
          "url": "...",
          "is_prime": true/false,
          "description": "...",
          "feature_bullets": ["...", ...],
          "specifications": {"key": "value", ...}
        },
        ...
      ]
    },
    ...
  ]
}

Your job is to determine which products COMPLY with ALL specs in specs_list
for their category:

1. For each product, cross-reference the provided description, feature_bullets,
   and specifications against the specs_list.
2. If a spec cannot be confirmed from the provided Amazon data alone, use
   google_search to look up "<product title> <spec>" to verify.
3. A product is COMPLIANT only if it satisfies ALL specs in specs_list.
   Be reasonably flexible — e.g. "Bluetooth 5.0" should pass for "Bluetooth 5.2"
   (backward compatible). But do NOT mark compliant if a key spec is clearly missing.
4. If a spec relates to price (e.g. "under ₹2000"), check the price field directly.

After checking all products, write a structured plain-text summary:

--- PRODUCT CATEGORY: <product name> ---
SPECS REQUIRED: <comma-separated specs_list>
COMPLIANT PRODUCTS:
1. ASIN: <asin> | Title: <title> | Brand: <brand> | Price: <price> | Rating: <rating> | Ratings Count: <ratings_count> | Image: <image_url> | URL: <url> | Prime: <yes/no> | Compliance: <brief summary of how each spec is met>
2. ...
NON-COMPLIANT (skipped): <comma-separated ASINs with brief reason>

IMPORTANT:
- The product details are ALREADY provided — do NOT call any product detail tool.
- Use google_search ONLY when Amazon data is insufficient to confirm a spec.
- Do NOT invent specs. Only report what the data contains.
- If NO products in a category pass, say "No compliant products found."
"""

compliance_check_subagent = Agent(
    name="compliance_check_subagent",
    model=LLM_USED,
    instruction=COMPLIANCE_INSTRUCTION,
    tools=[google_search],
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

def _enrich_product(product: SearchResultProduct) -> dict:
    """Call amazon_product programmatically and merge details with search data.

    Only keeps fields the LLM needs for compliance checking — strips images,
    full URLs, and other bulky data that would waste context tokens.
    """
    detail = amazon_product(product.asin)

    # Start with only the fields relevant for compliance
    base = {
        "asin": product.asin,
        "title": product.title,
        "brand": product.brand,
        "price": product.price,
        "rating": product.rating,
        "ratings_count": product.ratings_count,
        "is_prime": product.is_prime,
        # image & url are preserved for the structurer to pass through
        "image": product.image,
        "url": product.url,
    }

    if detail.get("status") == "success":
        desc = detail.get("description") or ""
        base["description"] = desc[:500] if len(desc) > 500 else desc
        base["feature_bullets"] = (detail.get("feature_bullets") or [])[:10]
        base["specifications"] = detail.get("specifications") or {}
    else:
        base["description"] = None
        base["feature_bullets"] = []
        base["specifications"] = {}

    return base


async def check_compliance(
    runner: Runner,
    user_id: str,
    session_id: str,
    search_results: ProductSearchResults,
) -> ComplianceResults:
    """Enriches each product with amazon_product details (programmatically),
    then feeds the enriched data to the compliance LLM agent for verification.

    The runner/session passed here must be wired to compliance_check_agent.
    """
    # ── Step 1: Programmatic amazon_product enrichment ─────────────────────
    enriched_items = []
    for group in search_results.items:
        enriched_products = []
        for p in group.products:
            label = (p.title or p.asin)[:60]
            print(f"  📦 Fetching details: {p.asin} — {label}...")
            enriched_products.append(_enrich_product(p))

        enriched_items.append({
            "product": group.product,
            "specs_list": group.specs_list,
            "products": enriched_products,
        })

    enriched_payload = json.dumps({"items": enriched_items}, ensure_ascii=False)

    # ── Step 2: LLM compliance check (google_search only) ──────────────────
    content = types.Content(
        role="user",
        parts=[types.Part(text=enriched_payload)],
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
