"""
amazon_seller.py - Google ADK-compatible tool: seller profile for a product ASIN.

Two-step process:
  1. type=offers  → discover buy-box seller_id and all competing sellers.
  2. type=seller_profile → fetch full seller profile, feedback summary, recent feedback.
Total cost: 2 API credits.
"""

import time
from ._rainforest_client import rainforest_get, fmt_price, INTER_PAGE_DELAY_S


def amazon_seller(asin: str) -> dict:
    """Fetch the seller profile for the buy-box seller of a given product ASIN.

    Use this tool when the user wants to know who is selling a product, check
    seller feedback ratings, business information, or verify seller legitimacy.
    Internally calls type=offers to discover the buy-box seller_id, then calls
    type=seller_profile to retrieve their profile (2 API credits total).

    Args:
        asin: The ASIN of the product whose buy-box seller you want details on
            (e.g. "B09G9FPHY6"). Obtain this from amazon_search or amazon_product.

    Returns:
        A dict with the following structure:
        {
            "status": "success" | "partial" | "error",
            "asin": str,
            "seller_id": str | None,
            "seller_name": str | None,
            "seller_url": str | None,
            "logo": str | None,              # seller logo image URL
            "about": str | None,             # seller self-description
            "rating": float | None,          # overall seller rating (0-5)
            "ratings_count": int | None,     # lifetime total feedback count
            "positive_feedback_pct": int | None,  # lifetime % positive
            "storefront_url": str | None,
            "feedback_30d": {                # 30-day feedback summary
                "positive_pct": int,
                "neutral_pct": int,
                "negative_pct": int,
                "count": int
            } | None,
            "feedback_90d": dict | None,
            "feedback_12m": dict | None,
            "feedback_lifetime": dict | None,
            "recent_feedback": [             # most recent individual feedback entries
                {
                    "rating": int,
                    "body": str,
                    "rater": str             # e.g. "By Name on 27 June, 2026."
                },
                ...
            ],
            "all_sellers": [                 # all sellers currently offering this ASIN
                {
                    "seller_name": str | None,
                    "seller_id": str | None,
                    "price": str | None,
                    "is_prime": bool,
                    "condition": str | None,
                    "fulfilled_by_amazon": bool,
                    "ratings_positive_pct": int | None,
                    "ratings_total": int | None
                },
                ...
            ],
            "message": str  # present on error or partial results
        }
    """
    # Step 1: Get all offers to find the buy-box seller_id
    offers_raw = rainforest_get({"type": "offers", "asin": asin})

    req_info = offers_raw.get("request_info") or {}
    if not req_info.get("success", True) or ("status" in offers_raw and offers_raw["status"] == "error"):
        return {"status": "error", "asin": asin, "message": offers_raw.get("message", "Offers request failed.")}

    offers_list = offers_raw.get("offers") or []

    # Gather all sellers for reference
    all_sellers = []
    for offer in offers_list:
        seller_info = offer.get("seller") or {}
        price_info  = offer.get("price") or {}
        delivery    = offer.get("delivery") or {}
        all_sellers.append({
            "seller_name":          seller_info.get("name"),
            "seller_id":            seller_info.get("id"),
            "price":                fmt_price(price_info),
            "is_prime":             bool(offer.get("is_prime", False)),
            "condition":            (offer.get("condition") or {}).get("title"),
            "fulfilled_by_amazon":  bool(delivery.get("fulfilled_by_amazon", False)),
            "ratings_positive_pct": seller_info.get("ratings_percentage_positive"),
            "ratings_total":        seller_info.get("ratings_total"),
        })

    # Buy-box seller = first offer (Rainforest returns buy-box winner first)
    first_offer   = offers_list[0] if offers_list else {}
    buybox_seller = first_offer.get("seller") or {}
    seller_id     = buybox_seller.get("id")

    if not seller_id:
        return {
            "status":                "success",
            "asin":                  asin,
            "seller_id":             None,
            "seller_name":           buybox_seller.get("name"),
            "seller_url":            buybox_seller.get("link"),
            "logo":                  None,
            "about":                 None,
            "rating":                None,
            "ratings_count":         None,
            "positive_feedback_pct": None,
            "storefront_url":        None,
            "feedback_30d":          None,
            "feedback_90d":          None,
            "feedback_12m":          None,
            "feedback_lifetime":     None,
            "recent_feedback":       [],
            "all_sellers":           all_sellers,
            "message":               "Seller ID not found in offers; profile unavailable.",
        }

    # Step 2: Fetch seller profile
    time.sleep(INTER_PAGE_DELAY_S)
    profile_raw = rainforest_get({"type": "seller_profile", "seller_id": seller_id})

    req_info = profile_raw.get("request_info") or {}
    if not req_info.get("success", True) or ("status" in profile_raw and profile_raw["status"] == "error"):
        return {
            "status":                "partial",
            "asin":                  asin,
            "seller_id":             seller_id,
            "seller_name":           buybox_seller.get("name"),
            "seller_url":            buybox_seller.get("link"),
            "logo":                  None,
            "about":                 None,
            "rating":                buybox_seller.get("rating"),
            "ratings_count":         buybox_seller.get("ratings_total"),
            "positive_feedback_pct": buybox_seller.get("ratings_percentage_positive"),
            "storefront_url":        None,
            "feedback_30d":          None,
            "feedback_90d":          None,
            "feedback_12m":          None,
            "feedback_lifetime":     None,
            "recent_feedback":       [],
            "all_sellers":           all_sellers,
            "message":               f"Profile fetch failed: {profile_raw.get('message')}",
        }

    # Real field paths confirmed from live API response
    profile  = profile_raw.get("seller_details") or {}
    feedback = profile_raw.get("feedback_summary") or {}
    recent   = profile_raw.get("feedback") or []

    def _fb(period: str) -> dict | None:
        fb = feedback.get(period) or {}
        if not fb:
            return None
        return {
            "positive_pct": fb.get("positive_percent"),
            "neutral_pct":  fb.get("neutral_percent"),
            "negative_pct": fb.get("negative_percent"),
            "count":        fb.get("count"),
        }

    lifetime = feedback.get("lifetime") or {}

    return {
        "status":                "success",
        "asin":                  asin,
        "seller_id":             seller_id,
        "seller_name":           profile.get("name") or buybox_seller.get("name"),
        "seller_url":            profile.get("store_link") or buybox_seller.get("link"),
        "logo":                  profile.get("logo"),
        "about":                 profile.get("about_this_seller"),
        "rating":                profile.get("rating"),
        "ratings_count":         profile.get("ratings_total"),
        "positive_feedback_pct": lifetime.get("positive_percent"),
        "storefront_url":        profile.get("store_link"),
        "feedback_30d":          _fb("thirty_days"),
        "feedback_90d":          _fb("ninety_days"),
        "feedback_12m":          _fb("twelve_months"),
        "feedback_lifetime":     _fb("lifetime"),
        "recent_feedback": [
            {"rating": fb.get("rating"), "body": fb.get("body"), "rater": fb.get("rater")}
            for fb in recent
        ],
        "all_sellers": all_sellers,
    }
