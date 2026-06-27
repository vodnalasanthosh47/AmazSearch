"""
_rainforest_client.py - Shared Rainforest API client and helpers.

All amazon_* tool modules import from here to avoid duplication.

Environment variable required:
    RAINFOREST_API_KEY - Your Rainforest API key (loaded from backend/.env)
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_env_path)

RAINFOREST_BASE    = "https://api.rainforestapi.com/request"
AMAZON_DOMAIN      = "amazon.in"
LANGUAGE           = "en_GB"
CURRENCY           = "inr"
RESULTS_PER_PAGE   = 20    # Amazon's fixed page size for search results
INTER_PAGE_DELAY_S = 0.3   # seconds between paginated requests


def rainforest_get(params: dict) -> dict:
    """Make a single GET request to the Rainforest API and handle top-level errors.

    Injects api_key, amazon_domain, language, and currency defaults automatically.

    Args:
        params: Query parameters to send (must include at least "type").

    Returns:
        Parsed JSON response dict, or an error dict with keys
        {"status": "error", "message": str} on failure.
    """
    params.setdefault("api_key", os.environ.get("RAINFOREST_API_KEY", ""))
    params.setdefault("amazon_domain", AMAZON_DOMAIN)
    params.setdefault("language", LANGUAGE)
    params.setdefault("currency", CURRENCY)

    try:
        response = requests.get(RAINFOREST_BASE, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Request to Rainforest API timed out."}
    except requests.exceptions.HTTPError as exc:
        return {"status": "error", "message": f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"}
    except requests.exceptions.RequestException as exc:
        return {"status": "error", "message": str(exc)}


def fmt_price(info: dict) -> str | None:
    """Format a Rainforest price object into a human-readable string (e.g. '₹1,299')."""
    if not info:
        return None
    return info.get("symbol", "₹") + str(info.get("value", ""))
