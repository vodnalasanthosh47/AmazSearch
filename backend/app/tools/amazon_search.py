"""
amazon_search.py - Google ADK-compatible tools for the Rainforest Amazon API.

Provides four tools:
  - amazon_search        : keyword search on amazon.in, supports up to 100 results
                           via automatic pagination (Amazon returns ~20 per page).
  - amazon_product       : full product detail for a known ASIN.
  - amazon_reviews       : paginated customer reviews for a known ASIN.
  - amazon_seller        : seller profile + storefront for the buy-box seller of an ASIN.

All tools return plain dicts (status + data) so the ADK agent can read and
reason over the results directly.

Environment variable required:
    RAINFOREST_API_KEY - Your Rainforest API key (loaded from backend/.env)

Pagination note on amazon_search:
    Amazon's search pages return ~20 products each. There is no "results per page"
    parameter on Rainforest API for search — to exceed 20 results the API must be
    called once per page. This module handles that automatically: requesting
    max_results=60 will fire 3 API calls (pages 1-3) and merge the results.
    Each call costs 1 API credit, so keep max_results proportionate to your need.
"""

import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv


_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_env_path)

_RAINFOREST_BASE    = "https://api.rainforestapi.com/request"
_AMAZON_DOMAIN      = "amazon.in"
_LANGUAGE           = "en_GB"
_CURRENCY           = "inr"
_RESULTS_PER_PAGE   = 20   # Amazon's fixed page size for search results
_INTER_PAGE_DELAY_S = 0.3  # seconds between paginated requests (be polite)



def _rainforest_get(params: dict) -> dict:
    """Internal helper - makes one GET request and handles top-level errors."""
    params.setdefault("api_key", os.environ.get("RAINFOREST_API_KEY", ""))
    params.setdefault("amazon_domain", _AMAZON_DOMAIN)
    params.setdefault("language", _LANGUAGE)
    params.setdefault("currency", _CURRENCY)

    try:
        response = requests.get(_RAINFOREST_BASE, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Request to Rainforest API timed out."}
    except requests.exceptions.HTTPError as exc:
        return {"status": "error", "message": f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"}
    except requests.exceptions.RequestException as exc:
        return {"status": "error", "message": str(exc)}


def _fmt_price(info: dict) -> str | None:
    """Format a Rainforest price object into a human-readable string."""
    if not info:
        return None
    return info.get("symbol", "₹") + str(info.get("value", ""))



def amazon_search(search_term: str, max_results: int = 10) -> dict:
    """Search Amazon India for products matching a keyword query.

    Use this tool when the user wants to find or compare products by name,
    category, or description (e.g. "gaming chair under 20000", "wireless
    earbuds"). Returns a ranked list of matching products with key details.

    IMPORTANT - pagination note: Amazon returns at most 20 results per search
    page. To fetch more than 20 results this tool automatically fires multiple
    API requests (one per page, each costs 1 API credit). Use max_results
    proportionately - prefer 10-20 for quick lookups and up to 100 only when
    a broad catalogue comparison is genuinely needed.

    Args:
        search_term: The keyword or phrase to search for on Amazon
            (e.g. "mechanical keyboard", "boAt headphones").
        max_results: Total number of products to return (1-100). Defaults to 10.
            Values above 20 trigger multiple API calls (one per page of 20).

    Returns:
        A dict with the following structure:
        {
            "status": "success" | "error",
            "search_term": str,
            "pages_fetched": int,        # number of API calls made
            "total_returned": int,       # actual number of products in list
            "products": [
                {
                    "position": int,
                    "asin": str,
                    "title": str,
                    "brand": str | None,
                    "price": str | None,       # formatted, e.g. "₹1,299"
                    "rating": float | None,    # 0.0-5.0
                    "ratings_count": int | None,
                    "image": str | None,       # URL of the primary thumbnail
                    "url": str | None,         # canonical Amazon product URL
                    "is_prime": bool,
                    "sponsored": bool
                },
                ...
            ],
            "message": str  # present only on error
        }
    """
    max_results = max(1, min(int(max_results), 100))
    pages_needed = -(-max_results // _RESULTS_PER_PAGE)  # ceiling division

    all_products = []
    pages_fetched = 0

    for page_num in range(1, pages_needed + 1):
        if page_num > 1:
            time.sleep(_INTER_PAGE_DELAY_S)

        raw = _rainforest_get({
            "type": "search",
            "search_term": search_term,
            "page": page_num,
        })

        if "status" in raw and raw["status"] == "error":
            if page_num == 1:
                return raw  # fail fast on first page
            break  # partial results are still useful

        pages_fetched += 1
        page_items = raw.get("search_results", [])

        if not page_items:
            break  # Amazon returned an empty page - no more results

        for item in page_items:
            price_info = item.get("price") or {}
            all_products.append({
                "asin":          item.get("asin"),
                "title":         item.get("title"),
                "brand":         item.get("brand"),
                "price":         _fmt_price(price_info),
                "rating":        item.get("rating"),
                "ratings_count": item.get("ratings_total"),
                "image":         item.get("image"),
                "url":           item.get("link"),
                "is_prime":      bool(item.get("is_prime", False)),
                "sponsored":     bool(item.get("is_sponsored", False)),
            })

            if len(all_products) >= max_results:
                break

        if len(all_products) >= max_results:
            break

    return {
        "status":         "success",
        "search_term":    search_term,
        "pages_fetched":  pages_fetched,
        "total_returned": len(all_products),
        "products":       all_products,
    }



def amazon_product(asin: str) -> dict:
    """Fetch detailed information about a specific Amazon India product by ASIN.

    Use this tool when you already know a product's ASIN (e.g. from a previous
    amazon_search call) and need deeper details such as full description,
    specifications, feature bullets, seller info, or variant options.

    Args:
        asin: The Amazon Standard Identification Number (ASIN) for the product
            (e.g. "B09G9FPHY6"). ASINs are 10-character alphanumeric strings.

    Returns:
        A dict with the following structure:
        {
            "status": "success" | "error",
            "asin": str,
            "title": str | None,
            "brand": str | None,
            "price": str | None,            # formatted, e.g. "₹1,299"
            "original_price": str | None,   # before discount
            "discount": str | None,         # e.g. "20%"
            "rating": float | None,
            "ratings_count": int | None,
            "availability": str | None,     # e.g. "In Stock"
            "description": str | None,
            "feature_bullets": list[str],   # key product highlights
            "images": list[str],            # URLs of all product images
            "specifications": dict,         # tech specs as key-value pairs
            "variants": list[dict],         # colour/size/style variants
            "seller": str | None,           # buy-box seller name
            "seller_id": str | None,        # buy-box seller ID (use with amazon_seller)
            "url": str | None,
            "message": str                  # present only on error
        }
    """
    raw = _rainforest_get({"type": "product", "asin": asin})

    if "status" in raw and raw["status"] == "error":
        return raw

    p = raw.get("product", {})

    buy_box = p.get("buybox_winner") or {}
    price_info = buy_box.get("price") or p.get("price") or {}
    orig_price_info = buy_box.get("rrp") or p.get("rrp") or {}

    # Specifications
    specs_raw = p.get("specifications") or p.get("specifications_flat") or []
    specs = {}
    if isinstance(specs_raw, list):
        for entry in specs_raw:
            name = entry.get("name") or entry.get("key")
            value = entry.get("value")
            if name:
                specs[name] = value
    elif isinstance(specs_raw, dict):
        specs = specs_raw

    # Images
    images_raw = p.get("images") or p.get("images_flat") or []
    images = [img.get("link") for img in images_raw if img.get("link")]

    # Variants
    variants_raw = p.get("variants") or []
    variants = [
        {
            "asin":       v.get("asin"),
            "title":      v.get("title"),
            "dimensions": v.get("dimensions"),
        }
        for v in variants_raw
    ]

    # Seller info from buy-box
    fulfillment = buy_box.get("fulfillment") or {}
    third_party  = fulfillment.get("third_party_seller") or {}

    return {
        "status":          "success",
        "asin":            asin,
        "title":           p.get("title"),
        "brand":           p.get("brand"),
        "price":           _fmt_price(price_info),
        "original_price":  _fmt_price(orig_price_info),
        "discount":        (buy_box.get("saving") or {}).get("percentage"),
        "rating":          p.get("rating"),
        "ratings_count":   p.get("ratings_total"),
        "availability":    (buy_box.get("availability") or {}).get("raw"),
        "description":     p.get("description"),
        "feature_bullets": p.get("feature_bullets") or p.get("feature_bullets_flat") or [],
        "images":          images,
        "specifications":  specs,
        "variants":        variants,
        "seller":          third_party.get("name"),
        "seller_id":       third_party.get("id"),
        "url":             p.get("link"),
    }



def _parse_review(r: dict) -> dict:
    """Normalize a single review object from any Rainforest source."""
    images_raw = r.get("images") or []
    # profile name may be at r["profile"]["name"] or r["author"] depending on source
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
    _REVIEWS_PER_PAGE = 10
    max_reviews = max(1, min(int(max_reviews), 100))
    pages_needed = -(-max_reviews // _REVIEWS_PER_PAGE)

    all_reviews = []
    overall_rating = None
    rating_breakdown = {}
    pages_fetched = 0
    reviews_api_unavailable = False

    for page_num in range(1, pages_needed + 1):
        if page_num > 1:
            time.sleep(_INTER_PAGE_DELAY_S)

        raw = _rainforest_get({
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


    if reviews_api_unavailable or (pages_fetched == 0 and not all_reviews):
        prod_raw = _rainforest_get({"type": "product", "asin": asin})
        req_info = prod_raw.get("request_info") or {}
        if not req_info.get("success", True) or ("status" in prod_raw and prod_raw["status"] == "error"):
            return {
                "status":  "error",
                "asin":    asin,
                "message": "Reviews endpoint unavailable and product fallback also failed.",
            }

        p = prod_raw.get("product") or {}
        overall_rating = p.get("rating") or overall_rating

        # rating_breakdown from top_reviews_summary if present
        trb = p.get("top_reviews_summary") or p.get("ratings_breakdown") or {}
        if trb:
            rating_breakdown = {
                "five_star":  (trb.get("five_star")  or {}).get("count"),
                "four_star":  (trb.get("four_star")  or {}).get("count"),
                "three_star": (trb.get("three_star") or {}).get("count"),
                "two_star":   (trb.get("two_star")   or {}).get("count"),
                "one_star":   (trb.get("one_star")   or {}).get("count"),
            }

        fallback_reviews = p.get("top_reviews") or []
        for r in fallback_reviews[:max_reviews]:
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
            "status": "success" | "error",
            "asin": str,
            "seller_id": str | None,
            "seller_name": str | None,
            "seller_url": str | None,
            "rating": float | None,          # seller feedback rating (0-5)
            "ratings_count": int | None,     # total feedback count
            "positive_feedback_pct": str | None,    # e.g. "97%"
            "business_name": str | None,     # legal/registered business name
            "business_address": str | None,
            "started_selling": str | None,   # e.g. "January 2019"
            "storefront_url": str | None,
            "all_sellers": [                 # all sellers offering this ASIN
                {
                    "seller_name": str | None,
                    "seller_id": str | None,
                    "price": str | None,
                    "is_prime": bool,
                    "condition": str | None,
                    "is_amazon": bool,
                },
                ...
            ],
            "message": str  # present only on error
        }
    """
    # Step 1: Get all offers for the ASIN to find the buy-box seller_id
    offers_raw = _rainforest_get({"type": "offers", "asin": asin})

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
            "seller_name": seller_info.get("name"),
            "seller_id":   seller_info.get("id"),
            "price":       _fmt_price(price_info),
            "is_prime":    bool(offer.get("is_prime", False)),
            "condition":   (offer.get("condition") or {}).get("title"),
            "fulfilled_by_amazon": bool(delivery.get("fulfilled_by_amazon", False)),
            "ratings_positive_pct": seller_info.get("ratings_percentage_positive"),
            "ratings_total": seller_info.get("ratings_total"),
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
            "rating":                None,
            "ratings_count":         None,
            "positive_feedback_pct": None,
            "about":                 None,
            "storefront_url":        None,
            "feedback_30d":          None,
            "feedback_90d":          None,
            "feedback_12m":          None,
            "feedback_lifetime":     None,
            "recent_feedback":       [],
            "all_sellers":           all_sellers,
            "message":               "Seller ID not found in offers; profile unavailable.",
        }

    # Step 2: Fetch seller profile using the seller_id
    time.sleep(_INTER_PAGE_DELAY_S)
    profile_raw = _rainforest_get({"type": "seller_profile", "seller_id": seller_id})

    req_info = profile_raw.get("request_info") or {}
    if not req_info.get("success", True) or ("status" in profile_raw and profile_raw["status"] == "error"):
        return {
            "status":                "partial",
            "asin":                  asin,
            "seller_id":             seller_id,
            "seller_name":           buybox_seller.get("name"),
            "seller_url":            buybox_seller.get("link"),
            "rating":                buybox_seller.get("rating"),
            "ratings_count":         buybox_seller.get("ratings_total"),
            "positive_feedback_pct": buybox_seller.get("ratings_percentage_positive"),
            "about":                 None,
            "storefront_url":        None,
            "feedback_30d":          None,
            "feedback_90d":          None,
            "feedback_12m":          None,
            "feedback_lifetime":     None,
            "recent_feedback":       [],
            "all_sellers":           all_sellers,
            "message":               f"Profile fetch failed: {profile_raw.get('message')}",
        }

    # Real field path is profile_raw["seller_details"] (confirmed from API response)
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
        "recent_feedback":       [
            {"rating": fb.get("rating"), "body": fb.get("body"), "rater": fb.get("rater")}
            for fb in recent
        ],
        "all_sellers":           all_sellers,
    }



def _test():
    print("=== Search: gaming chair (top 5) ===")
    result = amazon_search("gaming chair", max_results=5)
    print(f"  Pages fetched: {result.get('pages_fetched')}  |  Results: {result.get('total_returned')}")
    for product in result.get("products", []):
        print(f"  [{product['position']}] {product['title'][:60]}... — {product['price']} ★{product['rating']}")
        print(f"       ASIN: {product['asin']}")

    if result.get("products"):
        first_asin = result["products"][0]["asin"]

        print(f"\n=== Product detail: {first_asin} ===")
        detail = amazon_product(first_asin)
        print(f"  Title:     {detail.get('title')}")
        print(f"  Price:     {detail.get('price')}  (was {detail.get('original_price')})")
        print(f"  Rating:    {detail.get('rating')} ({detail.get('ratings_count')} ratings)")
        print(f"  In stock:  {detail.get('availability')}")
        print(f"  Seller:    {detail.get('seller')} (ID: {detail.get('seller_id')})")

        print(f"\n=== Reviews: {first_asin} (top 5, all stars) ===")
        reviews = amazon_reviews(first_asin, max_reviews=5, star_filter="all_stars")
        print(f"  Source:    {reviews.get('source')}  |  Overall: ★{reviews.get('overall_rating')}")
        print(f"  Message:   {reviews.get('message', 'n/a')}")
        for rev in reviews.get("reviews", []):
            print(f"  ★{rev['rating']} | {rev['title']} — {rev['reviewer']} ({rev['date']})")
            print(f"    {(rev['body'] or '')[:120]}...")

        print(f"\n=== Seller details for ASIN: {first_asin} ===")
        seller = amazon_seller(first_asin)
        print(f"  Seller:    {seller.get('seller_name')}")
        print(f"  About:     {(seller.get('about') or '')[:100]}")
        print(f"  Rating:    ★{seller.get('rating')} ({seller.get('ratings_count')} total)")
        print(f"  Lifetime:  {seller.get('positive_feedback_pct')}% positive  |  feedback_lifetime: {seller.get('feedback_lifetime')}")
        print(f"  30-day:    {seller.get('feedback_30d')}")
        print(f"  Storefront:{seller.get('storefront_url')}")
        print(f"  All sellers on this listing: {len(seller.get('all_sellers', []))}")
        recent = seller.get("recent_feedback") or []
        if recent:
            print("  Recent seller feedback:")
            for fb in recent[:3]:
                print(f"    ★{fb['rating']} {fb['rater']}: {(fb['body'] or '')[:80]}")


if __name__ == "__main__":
    _test()