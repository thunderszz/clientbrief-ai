"""Web search through Tavily.

A failed search never raises: it comes back as a response with ``error`` set and no
results, so one bad query cannot sink a research step. Batches run concurrently. A visitor's
own key can be set for the current context with ``use_tavily_key()``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from tavily import TavilyClient as _TavilyClient

logger = logging.getLogger(__name__)

_MAX_RESULTS = 5
_MAX_PARALLEL_SEARCHES = 4

_key_override: ContextVar[str | None] = ContextVar("tavily_key_override", default=None)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    content: str


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


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
    def __init__(self, api_key: str | None = None) -> None:
        """Use *api_key*, else the context override, else ``TAVILY_API_KEY``; raise if none is set."""
        key = api_key or _key_override.get() or os.environ.get("TAVILY_API_KEY", "")
        if not key:
            raise OSError(
                "TAVILY_API_KEY is not set. Export it as an environment variable or pass it explicitly."
            )
        self._client = _TavilyClient(api_key=key)

    def search(self, query: str) -> SearchResponse:
        try:
            raw = self._client.search(
                query=query,
                max_results=_MAX_RESULTS,
                search_depth="basic",
                include_answer=False,
                include_raw_content=False,
            )
        except Exception as exc:
            logger.warning("Tavily search failed for query=%r: %s", query, exc)
            return SearchResponse(query=query, error=str(exc))

        results = [
            SearchResult(title=r.get("title", ""), url=r.get("url", ""), content=r.get("content", ""))
            for r in raw.get("results", [])
        ]
        return SearchResponse(query=query, results=results)

    def search_batch(self, queries: list[str]) -> list[SearchResponse]:
        """Run the queries concurrently; responses come back in query order."""
        if not queries:
            return []
        workers = min(_MAX_PARALLEL_SEARCHES, len(queries))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tavily") as pool:
            return list(pool.map(self.search, queries))
