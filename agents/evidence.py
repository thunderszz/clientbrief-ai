"""Web search results turned into prompt snippets for the three research agents."""

from __future__ import annotations

import logging

from services.tavily_client import TavilySearchClient

logger = logging.getLogger(__name__)


def gather_evidence(queries: list[str], *, cite_urls: bool = True) -> tuple[list[str], list[str]]:
    """Run *queries* and return one snippet per hit, plus the distinct source URLs in order.

    With *cite_urls* every snippet starts with its URL, so the model can name its source.
    Without a Tavily key there is nothing to search, and both lists come back empty.
    """
    try:
        client = TavilySearchClient()
    except OSError as exc:
        logger.warning("Tavily client init failed: %s", exc)
        return [], []

    hits = [hit for response in client.search_batch(queries) for hit in response.results]
    snippets = [
        (f"[source: {hit.url}] " if cite_urls else "") + f"[{hit.title}] {hit.content}" for hit in hits
    ]
    urls = list(dict.fromkeys(hit.url for hit in hits if hit.url))
    return snippets, urls


def evidence_block(snippets: list[str], limit: int) -> str:
    """The first *limit* snippets, fenced so the prompt can refer to "the evidence above"."""
    joined = "\n\n".join(snippets[:limit])
    return f"--- Evidence ---\n{joined}\n--- End evidence ---"
