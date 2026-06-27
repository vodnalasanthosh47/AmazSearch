"""
product_search_agent.py — Agent 3 in the pipeline.

Takes the resolved product + specs_list from spec_resolution_agent and
uses amazon_search to find relevant products on Amazon India. Returns a
structured list of all discovered products (same JSON format as amazon_search
output), grouped by the original product name.

Architecture:
  - LLM agent builds optimised Amazon search queries from product + specs.
  - Python code then calls amazon_search programmatically (no LLM in the
    data-copy path → no token-limit truncation even for large result sets).
"""

import sys
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.genai import types

# Ensure the tools package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.amazon_search import amazon_search

# pyrefly: ignore [missing-import]
from spec_resolution_agent import ResolvedSetupSpecs


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class SearchResultProduct(BaseModel):
    """A single product from amazon_search results."""
    asin: str = Field(description="Amazon Standard Identification Number.")
    title: str = Field(description="Product title.")
    brand: str | None = Field(default=None, description="Brand name.")
    price: str | None = Field(default=None, description="Formatted price, e.g. '₹1,299'.")
    rating: float | None = Field(default=None, description="Average rating 0.0-5.0.")
    ratings_count: int | None = Field(default=None, description="Total number of ratings.")
    image: str | None = Field(default=None, description="Primary thumbnail URL.")
    url: str | None = Field(default=None, description="Canonical Amazon product URL.")
    is_prime: bool = Field(default=False, description="Whether Prime delivery is available.")
    sponsored: bool = Field(default=False, description="Whether the listing is sponsored.")


class ProductSearchGroup(BaseModel):
    """Search results for a single product category."""
    product: str = Field(description="The product name that was searched for.")
    specs_list: List[str] = Field(
        default=[],
        description="The specs/constraints used to build the search query.",
    )
    search_query: str = Field(
        default="",
        description="The actual search query sent to Amazon.",
    )
    products: List[SearchResultProduct] = Field(
        default=[],
        description="List of Amazon products returned for this query.",
    )


class ProductSearchResults(BaseModel):
    """Aggregated search results across all product categories."""
    items: List[ProductSearchGroup] = Field(default=[])


# ---------------------------------------------------------------------------
# Schema for the LLM query-builder step
# ---------------------------------------------------------------------------

class SearchQueryItem(BaseModel):
    """A single search query built by the LLM."""
    product: str = Field(description="The original product name, unchanged.")
    specs_list: List[str] = Field(
        default=[],
        description="The specs_list passed in, preserved as-is.",
    )
    search_query: str = Field(
        description=(
            "An optimised Amazon keyword search query built from the product "
            "name and specs. Price constraints should be dropped as they "
            "don't help keyword search."
        ),
    )


class SearchQueries(BaseModel):
    """Collection of search queries for all products."""
    items: List[SearchQueryItem] = Field(default=[])


# ---------------------------------------------------------------------------
# LLM Agent: builds smart Amazon search queries (no tools needed)
# ---------------------------------------------------------------------------

QUERY_BUILDER_INSTRUCTION = """\
You receive a JSON object describing products and their required specs, shaped like:
{"items": [{"product": "...", "specs_list": ["...", ...]}, ...]}

For EACH item, build ONE concise but effective Amazon search query by combining
the product name with the key specs from specs_list.

Rules:
- Drop price constraints from the search string (they don't help keyword search).
- Keep brand names if specified in specs.
- Focus on searchable attributes: brand, type, key features, size, gender.
- The query should be natural and similar to what a human would type into Amazon.
- If an item in specs_list is itself a full product name (e.g. "Nike Revolution 8
  Men's Road Running Shoes, size 8"), use that directly as the search query instead
  of combining — it's already a good query.

Example:
  Product: "Wireless Mouse"
  Specs: ["Bluetooth 5.0", "ergonomic design", "under ₹2000"]
  → search_query: "Bluetooth 5.0 ergonomic wireless mouse"

Return the structured output with one item per product. If a product has multiple
specs that each name a distinct product/model, produce ONE search query item
PER distinct model/spec entry (duplicating the product name).
"""

query_builder_agent = Agent(
    name="query_builder_agent",
    model="gemini-2.5-flash",
    instruction=QUERY_BUILDER_INSTRUCTION,
    output_schema=SearchQueries,
    output_key="search_queries",
)


# ---------------------------------------------------------------------------
# Helper to drive the agent from main_agent.py
# ---------------------------------------------------------------------------

MAX_RESULTS_PER_QUERY = 10


async def search_products(
    runner: Runner,
    user_id: str,
    session_id: str,
    resolved_specs: ResolvedSetupSpecs,
) -> ProductSearchResults:
    """Feeds spec_resolution_agent's output through the query-builder LLM,
    then programmatically calls amazon_search for each query.

    The runner/session passed here must be wired to query_builder_agent.
    """
    # ── Step 1: LLM builds optimised search queries ────────────────────────
    content = types.Content(
        role="user",
        parts=[types.Part(text=resolved_specs.model_dump_json())],
    )

    async for _event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        pass

    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    # pyrefly: ignore [missing-attribute]
    queries_json = session.state.get("search_queries")
    if queries_json is None:
        raise RuntimeError(
            "query_builder_agent did not produce 'search_queries' "
            "in session state."
        )

    if isinstance(queries_json, str):
        queries = SearchQueries.model_validate_json(queries_json)
    else:
        queries = SearchQueries.model_validate(queries_json)

    # ── Step 2: Programmatic Amazon search (no LLM in the data path) ───────
    result_groups: List[ProductSearchGroup] = []

    for q in queries.items:
        print(f"  🔍 Searching Amazon: \"{q.search_query}\"")
        raw = amazon_search(q.search_query, max_results=MAX_RESULTS_PER_QUERY)

        products: List[SearchResultProduct] = []
        if raw.get("status") == "success":
            for p in raw.get("products", []):
                products.append(SearchResultProduct(
                    asin=p.get("asin", ""),
                    title=p.get("title", ""),
                    brand=p.get("brand"),
                    price=p.get("price"),
                    rating=p.get("rating"),
                    ratings_count=p.get("ratings_count"),
                    image=p.get("image"),
                    url=p.get("url"),
                    is_prime=bool(p.get("is_prime", False)),
                    sponsored=bool(p.get("sponsored", False)),
                ))
        else:
            print(f"    ⚠️  Search failed: {raw.get('message', 'unknown error')}")

        result_groups.append(ProductSearchGroup(
            product=q.product,
            specs_list=q.specs_list,
            search_query=q.search_query,
            products=products,
        ))

    return ProductSearchResults(items=result_groups)
