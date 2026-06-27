"""
review_analysis_agent.py — Agent 5 in the pipeline.

Takes the ComplianceResults from compliance_check_agent (Agent 4), fetches
customer reviews for each compliant product via amazon_reviews
(programmatically), then feeds ONLY the product name + review texts to an
LLM that extracts:
  1. Good points (pros mentioned across reviews)
  2. Bad points  (cons / complaints mentioned across reviews)
  3. Trust score  (0-100) based on review quality, consistency, and sentiment

The full product metadata (price, image, url, etc.) is kept in Python and
stitched back onto the LLM's analysis output at the end.

Architecture:
  - Python code calls amazon_reviews for each ASIN (no LLM needed).
  - LLM agent receives a lightweight payload (name + reviews only).
  - Structurer agent converts notes to strict Pydantic schema.
  - Python merges the structured analysis back onto product metadata.
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
from tools.amazon_reviews import amazon_reviews

# pyrefly: ignore [missing-import]
from compliance_check_agent import ComplianceResults, CompliantProduct
# pyrefly: ignore [missing-import]
from first_agent import LLM_USED


# ---------------------------------------------------------------------------
# Lightweight LLM output schemas (no product metadata — just the analysis)
# ---------------------------------------------------------------------------

class ProductAnalysis(BaseModel):
    """LLM-produced analysis for a single product."""
    asin: str = Field(description="ASIN of the product that was analysed.")
    good_points: List[str] = Field(
        default=[],
        description="Specific pros/positives extracted from customer reviews.",
    )
    bad_points: List[str] = Field(
        default=[],
        description="Specific cons/negatives extracted from customer reviews.",
    )
    trust_score: int = Field(
        default=0,
        description="Trust score 0-100 based on review quality and consistency.",
    )
    trust_reasoning: str = Field(
        default="",
        description="Brief explanation of how the trust score was determined.",
    )
    reviews_analyzed: int = Field(
        default=0,
        description="Number of reviews that were analyzed.",
    )


class AnalysisGroup(BaseModel):
    """LLM analysis results for one product category."""
    product: str = Field(description="The product category name.")
    analyses: List[ProductAnalysis] = Field(default=[])


class LLMAnalysisResults(BaseModel):
    """Raw LLM output — analysis only, no product metadata."""
    items: List[AnalysisGroup] = Field(default=[])


# ---------------------------------------------------------------------------
# Final output schemas (analysis merged with full product metadata)
# ---------------------------------------------------------------------------

class ProductReviewAnalysis(BaseModel):
    """Full review analysis for a single compliant product."""
    asin: str = Field(description="Amazon Standard Identification Number.")
    title: str = Field(description="Product title.")
    brand: str | None = Field(default=None, description="Brand name.")
    price: str | None = Field(default=None, description="Formatted price.")
    rating: float | None = Field(default=None, description="Average rating 0.0-5.0.")
    ratings_count: int | None = Field(default=None, description="Total number of ratings.")
    image: str | None = Field(default=None, description="Primary thumbnail URL.")
    url: str | None = Field(default=None, description="Canonical Amazon product URL.")
    is_prime: bool = Field(default=False, description="Whether Prime delivery is available.")
    compliance_summary: str = Field(default="", description="Compliance summary from Stage 4.")
    good_points: List[str] = Field(default=[], description="Pros from reviews.")
    bad_points: List[str] = Field(default=[], description="Cons from reviews.")
    trust_score: int = Field(default=0, description="Trust score 0-100.")
    trust_reasoning: str = Field(default="", description="Trust score explanation.")
    reviews_analyzed: int = Field(default=0, description="Number of reviews analysed.")


class ReviewAnalysisGroup(BaseModel):
    """Review analysis results for a single product category."""
    product: str = Field(description="The original product category name.")
    specs_list: List[str] = Field(default=[], description="Specs from compliance stage.")
    analyzed_products: List[ProductReviewAnalysis] = Field(default=[])


class ReviewAnalysisResults(BaseModel):
    """Final output: full product info + review analysis."""
    items: List[ReviewAnalysisGroup] = Field(default=[])


# ---------------------------------------------------------------------------
# Sub-agent 5a: ANALYSES reviews — receives ONLY product name + reviews.
# Has tools (google_search) → must NOT have output_schema.
# ---------------------------------------------------------------------------

REVIEW_ANALYSIS_INSTRUCTION = """\
You receive a JSON object containing products and their customer reviews.
The input is deliberately minimal — just product names and review texts:

{
  "items": [
    {
      "product": "<category name>",
      "products": [
        {
          "asin": "<asin>",
          "name": "<product title>",
          "reviews": [
            {"title": "...", "body": "...", "rating": 4, "verified": true},
            ...
          ]
        },
        ...
      ]
    },
    ...
  ]
}

For EACH product, read its reviews and extract:

1. **GOOD POINTS** — Specific pros mentioned by reviewers. Be concrete.
   Example: "Battery lasts 2+ days on a single charge"

2. **BAD POINTS** — Specific cons/complaints mentioned by reviewers.
   Example: "Bluetooth drops connection after 30 minutes"

3. **TRUST SCORE (0-100)** — Based on:
   - Do reviews agree with each other? (consistency)
   - Are they verified purchases?
   - Are reviews detailed or generic one-liners?
   - Any red flags (identical phrasing, all same date, etc.)?
   Score guide: 90-100 = very trustworthy, 70-89 = good, 50-69 = moderate,
   30-49 = low trust, 0-29 = suspicious/manipulated.

Write output as plain text:

--- CATEGORY: <category> ---
PRODUCT: <name> (ASIN: <asin>)
REVIEWS ANALYZED: <count>
GOOD POINTS:
+ <point 1>
+ <point 2>
BAD POINTS:
- <point 1>
- <point 2>
TRUST SCORE: <0-100>
TRUST REASONING: <brief explanation>
---

Rules:
- Analyse ALL products. Do not skip any.
- If a product has NO reviews, trust score = 0, note "No reviews available".
- Do NOT invent content. Only report what reviews actually say.
- Be specific, not generic.
"""

review_analysis_subagent = Agent(
    name="review_analysis_subagent",
    model=LLM_USED,
    instruction=REVIEW_ANALYSIS_INSTRUCTION,
    tools=[google_search],
    output_key="review_analysis_notes",
)


# ---------------------------------------------------------------------------
# Sub-agent 5b: STRUCTURES the notes into strict schema.
# output_schema → must NOT have tools (ADK constraint).
# ---------------------------------------------------------------------------

REVIEW_STRUCTURER_INSTRUCTION = """\
You will receive plain-text review analysis results under the key
`review_analysis_notes`.

Convert them into the required structured format:
- One AnalysisGroup per product category.
- `product` = the category name from the notes.
- `analyses` = one ProductAnalysis per product, containing:
  - asin: from the notes
  - good_points: list of "+" items
  - bad_points: list of "-" items
  - trust_score: the number 0-100
  - trust_reasoning: the explanation
  - reviews_analyzed: the count

Review analysis notes:
{review_analysis_notes}
"""

review_structurer_agent = Agent(
    name="review_structurer_agent",
    model=LLM_USED,
    instruction=REVIEW_STRUCTURER_INSTRUCTION,
    output_schema=LLMAnalysisResults,
    output_key="review_analysis_results",
)


# ---------------------------------------------------------------------------
# Combined pipeline = "review_analysis_agent"
# ---------------------------------------------------------------------------

review_analysis_agent = SequentialAgent(
    name="review_analysis_agent",
    sub_agents=[review_analysis_subagent, review_structurer_agent],
)


# ---------------------------------------------------------------------------
# Helper to drive the agent from main_agent.py
# ---------------------------------------------------------------------------

MAX_REVIEWS_PER_PRODUCT = 20


def _fetch_reviews_only(asin: str) -> list[dict]:
    """Fetch reviews for an ASIN and return just the review texts.

    Returns a list of lightweight review dicts (title, body, rating, verified).
    """
    reviews_data = amazon_reviews(asin, max_reviews=MAX_REVIEWS_PER_PRODUCT)

    if reviews_data.get("status") != "success":
        return []

    reviews = []
    for r in reviews_data.get("reviews", []):
        body = r.get("body") or ""
        reviews.append({
            "title": r.get("title"),
            "body": body[:600] if len(body) > 600 else body,
            "rating": r.get("rating"),
            "verified": bool(r.get("verified_purchase", False)),
        })
    return reviews


def _merge_analysis_with_product(
    product: CompliantProduct,
    analysis: ProductAnalysis | None,
) -> ProductReviewAnalysis:
    """Merge the LLM analysis back onto the full product metadata."""
    return ProductReviewAnalysis(
        asin=product.asin,
        title=product.title,
        brand=product.brand,
        price=product.price,
        rating=product.rating,
        ratings_count=product.ratings_count,
        image=product.image,
        url=product.url,
        is_prime=product.is_prime,
        compliance_summary=product.compliance_summary,
        good_points=analysis.good_points if analysis else [],
        bad_points=analysis.bad_points if analysis else [],
        trust_score=analysis.trust_score if analysis else 0,
        trust_reasoning=analysis.trust_reasoning if analysis else "Analysis unavailable.",
        reviews_analyzed=analysis.reviews_analyzed if analysis else 0,
    )


def _normalize_category(name: str) -> str:
    """Lowercase + strip whitespace for fuzzy category matching."""
    return name.strip().lower()


async def analyse_reviews(
    runner: Runner,
    user_id: str,
    session_id: str,
    compliance_results: ComplianceResults,
) -> ReviewAnalysisResults:
    """Fetches reviews for each compliant product, sends ONLY product name +
    reviews to the LLM, then merges the analysis back onto full product data.

    The runner/session passed here must be wired to review_analysis_agent.
    """
    # ── Short-circuit: nothing to analyse if no compliant products ──────────
    total_compliant = sum(
        len(g.compliant_products) for g in compliance_results.items
    )
    if total_compliant == 0:
        print("  ⚠️  No compliant products to analyse. Skipping review stage.")
        return ReviewAnalysisResults(
            items=[
                ReviewAnalysisGroup(
                    product=g.product,
                    specs_list=g.specs_list,
                    analyzed_products=[],
                )
                for g in compliance_results.items
            ]
        )

    # ── Step 1: Fetch reviews and build lightweight LLM payload ────────────
    llm_items = []

    for group in compliance_results.items:
        llm_products = []

        for p in group.compliant_products:
            label = (p.title or p.asin)[:60]
            print(f"  📝 Fetching reviews: {p.asin} — {label}...")

            reviews = _fetch_reviews_only(p.asin)
            llm_products.append({
                "asin": p.asin,
                "name": p.title,
                "reviews": reviews,
            })

        llm_items.append({
            "product": group.product,
            "products": llm_products,
        })

    llm_payload = json.dumps({"items": llm_items}, ensure_ascii=False)

    # ── Step 2: Send to LLM for analysis ───────────────────────────────────
    content = types.Content(
        role="user",
        parts=[types.Part(text=llm_payload)],
    )

    async for _event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        pass

    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    # pyrefly: ignore [missing-attribute]
    results_json = session.state.get("review_analysis_results")
    if results_json is None:
        raise RuntimeError(
            "review_analysis_agent did not produce 'review_analysis_results' "
            "in session state."
        )

    if isinstance(results_json, str):
        llm_results = LLMAnalysisResults.model_validate_json(results_json)
    else:
        llm_results = LLMAnalysisResults.model_validate(results_json)

    # ── Step 3: Merge LLM analysis back onto full product metadata ─────────
    # Build a normalized-name → LLM analysis group lookup so that minor
    # LLM rephrasing (whitespace, case) doesn't break the merge.
    llm_group_lookup: dict[str, AnalysisGroup] = {
        _normalize_category(g.product): g for g in llm_results.items
    }

    final_items: list[ReviewAnalysisGroup] = []

    for group in compliance_results.items:
        # Find matching LLM analysis group (case-insensitive)
        llm_group = llm_group_lookup.get(_normalize_category(group.product))

        # Build ASIN → analysis lookup
        analysis_by_asin: dict[str, ProductAnalysis] = {}
        if llm_group:
            for a in llm_group.analyses:
                analysis_by_asin[a.asin] = a

        # Merge each compliant product with its analysis
        merged_products = []
        for p in group.compliant_products:
            analysis = analysis_by_asin.get(p.asin)
            merged_products.append(_merge_analysis_with_product(p, analysis))

        final_items.append(ReviewAnalysisGroup(
            product=group.product,
            specs_list=group.specs_list,
            analyzed_products=merged_products,
        ))

    return ReviewAnalysisResults(items=final_items)
