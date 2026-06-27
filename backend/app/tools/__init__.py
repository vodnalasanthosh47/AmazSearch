"""
tools/__init__.py - Re-exports all Google ADK-compatible Amazon tools.

Usage:
    from app.tools import amazon_search, amazon_product, amazon_reviews, amazon_seller

    agent = Agent(
        ...
        tools=[amazon_search, amazon_product, amazon_reviews, amazon_seller],
    )
"""

from .amazon_search import amazon_search
from .amazon_product import amazon_product
from .amazon_reviews import amazon_reviews
from .amazon_seller import amazon_seller

__all__ = [
    "amazon_search",
    "amazon_product",
    "amazon_reviews",
    "amazon_seller",
]
