"""
amazon_search.py - Google ADK-compatible tool: keyword search on Amazon India.

Supports up to 100 results via automatic pagination.
Amazon returns ~20 products per page — values above 20 trigger multiple API
calls (one per page, each costs 1 credit).
"""

import time
from ._rainforest_client import rainforest_get, fmt_price, RESULTS_PER_PAGE, INTER_PAGE_DELAY_S


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
    max_results = max(1, min(max_results, 100))
    pages_needed = -(-max_results // RESULTS_PER_PAGE)  # ceiling division

    all_products = []
    pages_fetched = 0

    for page_num in range(1, pages_needed + 1):
        if page_num > 1:
            time.sleep(INTER_PAGE_DELAY_S)

        raw = rainforest_get({
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
                "price":         fmt_price(price_info),
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