"""Tavily search client for ClientBrief AI.

Thin wrapper around the Tavily Python SDK that:
- loads the API key from the environment
- exposes a single ``search()`` entry point returning structured results
- handles errors gracefully so callers never see raw exceptions
- keeps the interface minimal and reusable across all research agents
- runs batches of queries concurrently
- accepts a per-request key via ``use_tavily_key()`` ("bring your own key")
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from tavily import TavilyClient as _TavilyClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchResult:
    """A single search hit returned by Tavily."""

    title: str
    url: str
    content: str
    score: float = 0.0
    raw_content: str | None = None


@dataclass
class SearchResponse:
    """Aggregated response from a single Tavily search call."""

    query: str
    results: list[SearchResult] = field(default_factory=list)
    answer: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def sources(self) -> list[str]:
        """Return deduplicated list of source URLs."""
        seen: set[str] = set()
        urls: list[str] = []
        for r in self.results:
            if r.url not in seen:
                seen.add(r.url)
                urls.append(r.url)
        return urls


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RESULTS = 5
_DEFAULT_SEARCH_DEPTH = "basic"          # "basic" or "advanced"
_DEFAULT_INCLUDE_RAW_CONTENT = False
_MAX_PARALLEL_SEARCHES = 4

_key_override: ContextVar[str | None] = ContextVar("tavily_key_override", default=None)


@contextmanager
def use_tavily_key(api_key: str | None) -> Iterator[None]:
    """Use *api_key* for every Tavily client built in this context. ``None`` is a no-op."""
    if not api_key:
        yield
        return
    token = _key_override.set(api_key)
    try:
        yield
    finally:
        _key_override.reset(token)


class TavilySearchClient:
    """Reusable Tavily search client.

    Usage::

        client = TavilySearchClient()              # reads TAVILY_API_KEY
        resp = client.search("Acme Corp overview")
        for hit in resp.results:
            print(hit.title, hit.url)
    """

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or _key_override.get() or os.environ.get("TAVILY_API_KEY", "")
        if not key:
            raise OSError(
                "TAVILY_API_KEY is not set. "
                "Export it as an environment variable or pass it explicitly."
            )
        self._client = _TavilyClient(api_key=key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        max_results: int = _DEFAULT_MAX_RESULTS,
        search_depth: str = _DEFAULT_SEARCH_DEPTH,
        include_answer: bool = False,
        include_raw_content: bool = _DEFAULT_INCLUDE_RAW_CONTENT,
    ) -> SearchResponse:
        """Execute a single search query and return structured results.

        Parameters
        ----------
        query:
            The search query string (should already be templated by the caller).
        max_results:
            Maximum number of results to return (default 5).
        search_depth:
            ``"basic"`` (faster/cheaper) or ``"advanced"`` (deeper).
        include_answer:
            If ``True`` Tavily returns a short AI-generated answer alongside hits.
        include_raw_content:
            If ``True`` each result includes the full page text.

        Returns
        -------
        SearchResponse
            Always returns a response object.  On failure the ``.error`` field
            is populated and ``.results`` is empty.
        """
        try:
            raw: dict[str, Any] = self._client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
            )
        except Exception as exc:
            logger.warning("Tavily search failed for query=%r: %s", query, exc)
            return SearchResponse(query=query, error=str(exc))

        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                raw_content=r.get("raw_content"),
            )
            for r in raw.get("results", [])
        ]

        return SearchResponse(
            query=query,
            results=results,
            answer=raw.get("answer"),
        )

    def search_batch(
        self,
        queries: list[str],
        *,
        max_results: int = _DEFAULT_MAX_RESULTS,
        search_depth: str = _DEFAULT_SEARCH_DEPTH,
        include_answer: bool = False,
        include_raw_content: bool = _DEFAULT_INCLUDE_RAW_CONTENT,
    ) -> list[SearchResponse]:
        """Run multiple queries concurrently and return responses in query order.

        Each query is independent — a failure in one does not block the others.
        """
        if not queries:
            return []

        def run_one(query: str) -> SearchResponse:
            return self.search(
                query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
            )

        workers = min(_MAX_PARALLEL_SEARCHES, len(queries))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tavily") as pool:
            return list(pool.map(run_one, queries))
