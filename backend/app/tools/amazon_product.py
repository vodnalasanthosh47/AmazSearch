"""
amazon_product.py - Google ADK-compatible tool: full product detail by ASIN.
"""

from ._rainforest_client import rainforest_get, fmt_price


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
    raw = rainforest_get({"type": "product", "asin": asin})

    if "status" in raw and raw["status"] == "error":
        return raw

    p = raw.get("product", {})

    buy_box = p.get("buybox_winner") or {}
    price_info = buy_box.get("price") or p.get("price") or {}
    orig_price_info = buy_box.get("rrp") or p.get("rrp") or {}

    # Specifications
    specs_raw = p.get("specifications") or p.get("specifications_flat") or []
    specs: dict = {}
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
        "price":           fmt_price(price_info),
        "original_price":  fmt_price(orig_price_info),
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
