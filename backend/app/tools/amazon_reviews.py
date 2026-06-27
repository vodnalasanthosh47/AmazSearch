"""
amazon_reviews.py - Google ADK-compatible tool: customer reviews for an ASIN.

Tries type=reviews first; falls back to top_reviews embedded in type=product
when the reviews endpoint is temporarily unavailable (e.g. 503).
"""

import time
from ._rainforest_client import rainforest_get, INTER_PAGE_DELAY_S

_REVIEWS_PER_PAGE = 10


def _parse_review(r: dict) -> dict:
    """Normalize a single review object from any Rainforest source."""
    images_raw = r.get("images") or []
    reviewer = (
        (r.get("profile") or {}).get("name")
        or r.get("author")
        or r.get("reviewer")
    )
    return {
        "id":                r.get("id"),
        "title":             r.get("title"),
        "body":              r.get("body"),
        "rating":            r.get("rating"),
        "date":              r.get("date"),
        "verified_purchase": bool(r.get("verified_purchase", False)),
        "helpful_votes":     r.get("helpful_votes"),
        "reviewer":          reviewer,
        "images":            [img.get("link") for img in images_raw if img.get("link")],
    }


def amazon_reviews(asin: str, max_reviews: int = 20, star_filter: str = "all_stars") -> dict:
    """Fetch customer reviews for a product on Amazon India.

    Use this tool when the user wants to understand sentiment, common complaints,
    praise, or detailed opinions about a specific product. Reviews are returned
    newest-first by default.

    Note: If the dedicated reviews endpoint is temporarily unavailable, this tool
    automatically falls back to the top reviews embedded in the product page
    (typically 8-10 reviews). The star_filter may be ignored in fallback mode.

    Args:
        asin: The ASIN of the product to fetch reviews for
            (e.g. "B09G9FPHY6"). Obtain from amazon_search or amazon_product.
        max_reviews: Total number of reviews to return (1-100). Defaults to 20.
            Values above 10 trigger multiple API calls (one per page of ~10).
        star_filter: Filter reviews by star rating. One of:
            "all_stars" (default), "five_star", "four_star", "three_star",
            "two_star", "one_star", "all_positive" (4-5 stars),
            "all_critical" (1-3 stars).

    Returns:
        A dict with the following structure:
        {
            "status": "success" | "error",
            "source": "reviews_api" | "product_page_fallback",
            "asin": str,
            "star_filter": str,
            "pages_fetched": int,
            "total_returned": int,
            "overall_rating": float | None,
            "rating_breakdown": {
                "five_star": int | None,
                "four_star": int | None,
                "three_star": int | None,
                "two_star": int | None,
                "one_star": int | None,
            },
            "reviews": [
                {
                    "id": str | None,
                    "title": str | None,
                    "body": str | None,
                    "rating": float | None,
                    "date": str | None,
                    "verified_purchase": bool,
                    "helpful_votes": int | None,
                    "reviewer": str | None,
                    "images": list[str]
                },
                ...
            ],
            "message": str  # present only on error or when fallback is used
        }
    """
    max_reviews = max(1, min(int(max_reviews), 100))
    pages_needed = -(-max_reviews // _REVIEWS_PER_PAGE)

    all_reviews: list[dict] = []
    overall_rating = None
    rating_breakdown: dict = {}
    pages_fetched = 0
    reviews_api_unavailable = False

    # Primary: type=reviews endpoint
    for page_num in range(1, pages_needed + 1):
        if page_num > 1:
            time.sleep(INTER_PAGE_DELAY_S)

        raw = rainforest_get({
            "type":         "reviews",
            "asin":         asin,
            "review_stars": star_filter,
            "page":         page_num,
        })

        # Check for service-level errors (e.g. 503 temporarily unavailable)
        req_info = raw.get("request_info") or {}
        if not req_info.get("success", True) or ("status" in raw and raw["status"] == "error"):
            reviews_api_unavailable = True
            break

        pages_fetched += 1

        if page_num == 1:
            summary = raw.get("summary") or {}
            overall_rating = summary.get("rating")
            rb = summary.get("rating_breakdown") or {}
            rating_breakdown = {
                "five_star":  (rb.get("five_star")  or {}).get("count"),
                "four_star":  (rb.get("four_star")  or {}).get("count"),
                "three_star": (rb.get("three_star") or {}).get("count"),
                "two_star":   (rb.get("two_star")   or {}).get("count"),
                "one_star":   (rb.get("one_star")   or {}).get("count"),
            }

        page_reviews = raw.get("reviews") or []
        if not page_reviews:
            break

        for r in page_reviews:
            all_reviews.append(_parse_review(r))
            if len(all_reviews) >= max_reviews:
                break

        if len(all_reviews) >= max_reviews:
            break

    # Fallback: pull top_reviews from type=product
    if reviews_api_unavailable or (pages_fetched == 0 and not all_reviews):
        prod_raw = rainforest_get({"type": "product", "asin": asin})
        req_info = prod_raw.get("request_info") or {}
        if not req_info.get("success", True) or ("status" in prod_raw and prod_raw["status"] == "error"):
            return {
                "status":  "error",
                "asin":    asin,
                "message": "Reviews endpoint unavailable and product fallback also failed.",
            }

        p = prod_raw.get("product") or {}
        overall_rating = p.get("rating") or overall_rating

        trb = p.get("top_reviews_summary") or p.get("ratings_breakdown") or {}
        if trb:
            rating_breakdown = {
                "five_star":  (trb.get("five_star")  or {}).get("count"),
                "four_star":  (trb.get("four_star")  or {}).get("count"),
                "three_star": (trb.get("three_star") or {}).get("count"),
                "two_star":   (trb.get("two_star")   or {}).get("count"),
                "one_star":   (trb.get("one_star")   or {}).get("count"),
            }

        for r in (p.get("top_reviews") or [])[:max_reviews]:
            all_reviews.append(_parse_review(r))

        return {
            "status":           "success",
            "source":           "product_page_fallback",
            "asin":             asin,
            "star_filter":      star_filter,
            "pages_fetched":    0,
            "total_returned":   len(all_reviews),
            "overall_rating":   overall_rating,
            "rating_breakdown": rating_breakdown,
            "reviews":          all_reviews,
            "message":          "type=reviews endpoint temporarily unavailable; showing top reviews from product page.",
        }

    return {
        "status":           "success",
        "source":           "reviews_api",
        "asin":             asin,
        "star_filter":      star_filter,
        "pages_fetched":    pages_fetched,
        "total_returned":   len(all_reviews),
        "overall_rating":   overall_rating,
        "rating_breakdown": rating_breakdown,
        "reviews":          all_reviews,
    }
