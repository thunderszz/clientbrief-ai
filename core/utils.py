"""Deterministic utility helpers for ClientBrief AI.

Every function in this module is pure Python with no LLM calls.
Covers: domain normalization, role/function normalization, query templating,
signal deduplication, date normalization, and input validation.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from core.state import (
    VALID_FUNCTIONS,
    VALID_MEETING_TYPES,
    VALID_MEETING_STAGES,
    SENIORITY_LEVELS,
)


# =============================================================================
# Domain / URL normalization
# =============================================================================

def normalize_domain(raw: str) -> str:
    """Extract a clean root domain from a URL or domain string.

    Examples
    --------
    >>> normalize_domain("https://www.example.com/about")
    'example.com'
    >>> normalize_domain("WWW.EXAMPLE.COM")
    'example.com'
    >>> normalize_domain("example.com")
    'example.com'
    """
    raw = raw.strip()
    # Add scheme if missing so urlparse can handle it
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    # Strip leading www.
    if host.startswith("www."):
        host = host[4:]
    return host


def validate_domain(domain: str) -> bool:
    """Return True if *domain* looks like a plausible internet domain."""
    return bool(domain) and "." in domain and " " not in domain


# =============================================================================
# Role / function normalization
# =============================================================================

# Mapping from keyword patterns to canonical functions.
# Each entry is (pattern, function, word_boundary).  When word_boundary is True
# the pattern is matched as a whole word (via \b regex) to avoid false positives
# like "cto" matching inside "director".
# Order matters: first match wins, so more-specific keywords come first.
_FUNCTION_KEYWORDS: list[tuple[str, str, bool]] = [
    # transformation must precede operations (COO vs. ERP Transformation Lead)
    ("transformation", "transformation", False),
    ("change management", "transformation", False),
    ("erp", "transformation", True),
    ("digital transformation", "transformation", False),
    # accounting before finance (Controller ≠ CFO)
    ("accounting", "accounting", False),
    ("controller", "accounting", False),
    ("audit", "accounting", False),
    ("bookkeep", "accounting", False),
    # finance
    ("cfo", "finance", True),
    ("chief financial", "finance", False),
    ("finance", "finance", False),
    ("treasury", "finance", False),
    ("fp&a", "finance", False),
    ("financial planning", "finance", False),
    # engineering
    ("engineer", "engineering", False),
    ("developer", "engineering", False),
    ("architect", "engineering", False),
    ("cto", "engineering", True),
    ("chief technology", "engineering", False),
    ("devops", "engineering", False),
    ("sre", "engineering", True),
    ("software", "engineering", False),
    # product
    ("product", "product", False),
    ("cpo", "product", True),
    # sales
    ("sales", "sales", False),
    ("revenue", "sales", False),
    ("revops", "sales", False),
    ("business development", "sales", False),
    ("account executive", "sales", False),
    ("crm", "sales", True),
    # strategy
    ("strategy", "strategy", False),
    ("cso", "strategy", True),
    ("chief strategy", "strategy", False),
    ("corporate development", "strategy", False),
    ("m&a", "strategy", False),
    # it / data
    ("cio", "it_data", True),
    ("chief information", "it_data", False),
    ("information technology", "it_data", False),
    ("it manager", "it_data", False),
    ("it director", "it_data", False),
    ("it infrastructure", "it_data", False),
    ("data", "it_data", False),
    ("analytics", "it_data", False),
    ("security", "it_data", False),
    ("ciso", "it_data", True),
    ("infrastructure", "it_data", False),
    # operations
    ("coo", "operations", True),
    ("chief operating", "operations", False),
    ("operations", "operations", False),
    ("supply chain", "operations", False),
    ("logistics", "operations", False),
    ("procurement", "operations", False),
]

# Keywords that hint at seniority level.
# (pattern, level, word_boundary) — same convention as _FUNCTION_KEYWORDS.
_SENIORITY_KEYWORDS: list[tuple[str, str, bool]] = [
    ("chief", "c_level", False),
    ("ceo", "c_level", True),
    ("cfo", "c_level", True),
    ("coo", "c_level", True),
    ("cto", "c_level", True),
    ("cio", "c_level", True),
    ("ciso", "c_level", True),
    ("cpo", "c_level", True),
    ("cso", "c_level", True),
    ("cmo", "c_level", True),
    ("vp", "vp", True),
    ("vice president", "vp", False),
    ("svp", "vp", True),
    ("evp", "vp", True),
    ("director", "director", False),
    ("head of", "head", False),
    ("head,", "head", False),
    ("manager", "manager", False),
    ("lead", "manager", True),
    ("senior", "individual_contributor", False),
    ("analyst", "individual_contributor", False),
    ("specialist", "individual_contributor", False),
    ("coordinator", "individual_contributor", False),
]


def normalize_function(role: str) -> str:
    """Map a free-text role/title to one of the canonical business functions.

    Returns the canonical function string, or ``"general"`` if no match.
    """
    lower = role.lower().strip()
    for keyword, function, word_boundary in _FUNCTION_KEYWORDS:
        if word_boundary:
            if re.search(rf"\b{re.escape(keyword)}\b", lower):
                return function
        else:
            if keyword in lower:
                return function
    return "general"


def normalize_seniority(role: str) -> Optional[str]:
    """Infer seniority level from a free-text role/title.

    Returns a canonical seniority string, or ``None`` if indeterminate.
    """
    lower = role.lower().strip()
    for keyword, level, word_boundary in _SENIORITY_KEYWORDS:
        if word_boundary:
            if re.search(rf"\b{re.escape(keyword)}\b", lower):
                return level
        else:
            if keyword in lower:
                return level
    return None


# =============================================================================
# Input validation
# =============================================================================

def validate_inputs(
    company_name: str,
    domain: str,
    meeting_type: str,
    stakeholder_role: str,
    meeting_stage: str | None = None,
) -> list[str]:
    """Validate required input fields and return a list of error messages.

    An empty list means all validations passed.

    ``meeting_stage`` is optional.  When provided it must be one of the
    canonical ``VALID_MEETING_STAGES`` values; when omitted the orchestrator
    defaults to ``"first_intro"``.
    """
    errors: list[str] = []
    if not company_name or not company_name.strip():
        errors.append("company_name is required and must be non-empty.")
    if not validate_domain(domain):
        errors.append(
            f"domain '{domain}' is invalid — must contain at least one dot and no spaces."
        )
    if meeting_type not in VALID_MEETING_TYPES:
        allowed = ", ".join(sorted(VALID_MEETING_TYPES))
        errors.append(
            f"meeting_type '{meeting_type}' is not recognized. Allowed: {allowed}."
        )
    if not stakeholder_role or not stakeholder_role.strip():
        errors.append("stakeholder_role is required and must be non-empty.")
    if meeting_stage is not None and meeting_stage not in VALID_MEETING_STAGES:
        allowed = ", ".join(sorted(VALID_MEETING_STAGES))
        errors.append(
            f"meeting_stage '{meeting_stage}' is not recognized. Allowed: {allowed}."
        )
    return errors


# =============================================================================
# Tavily query templating
# =============================================================================

def build_company_queries(company_name: str, domain: str) -> list[str]:
    """Return the standard set of Tavily queries for company research."""
    return [
        f'"{company_name}" company overview business model',
        f'"{company_name}" products customers industry',
        f"{domain}",
    ]


def build_signal_queries(
    company_name: str,
    extra_queries: list[str] | None = None,
) -> list[str]:
    """Return Tavily queries for external-signals research.

    Always includes the always-on queries from the spec.
    ``extra_queries`` are appended (typically from the research plan).
    """
    current_year = datetime.now().year
    prev_year = current_year - 1
    queries = [
        f'"{company_name}" news {prev_year} {current_year}',
        f'"{company_name}" acquisition partnership announcement',
        f'"{company_name}" layoffs restructuring expansion',
    ]
    if extra_queries:
        queries.extend(extra_queries)
    return queries


def build_person_queries(
    contact_name: str,
    company_name: str,
    contact_title: str | None = None,
) -> list[str]:
    """Return Tavily queries for person research.

    Follows the candidate search patterns from the spec.
    """
    queries = [
        f'"{contact_name}" "{company_name}"',
        f'"{contact_name}" "{company_name}" speaker bio',
        f'"{contact_name}" "{company_name}" interview',
    ]
    if contact_title:
        queries.append(f'"{contact_name}" "{company_name}" "{contact_title}"')
    return queries


# =============================================================================
# Signal deduplication
# =============================================================================

def _signal_fingerprint(signal: dict) -> str:
    """Create a coarse fingerprint for a signal dict.

    Two signals are considered duplicates if their normalized descriptions
    share enough overlap. We use a simple lowered, stripped, punctuation-removed
    description as the key.
    """
    desc = signal.get("description", "")
    desc = re.sub(r"[^\w\s]", "", desc.lower()).strip()
    # collapse whitespace
    desc = re.sub(r"\s+", " ", desc)
    return desc


def deduplicate_signals(signals: list[dict]) -> list[dict]:
    """Remove near-duplicate signals based on description similarity.

    Keeps the first occurrence (which preserves the original ordering from
    the retrieval step).

    NOTE: This currently deduplicates on exact normalised-description match
    only (lowered, punctuation-stripped, whitespace-collapsed).  It does NOT
    use fuzzy/semantic similarity.  A fuzzy approach (e.g. token-overlap or
    embedding cosine) can be added later if near-miss duplicates become a
    quality problem in practice.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for s in signals:
        fp = _signal_fingerprint(s)
        if not fp:
            continue
        if fp not in seen:
            seen.add(fp)
            unique.append(s)
    return unique


# =============================================================================
# Date normalization
# =============================================================================

# Common date patterns we try to parse, ordered from most to least specific.
_DATE_FORMATS: list[str] = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%B %d, %Y",       # January 15, 2025
    "%b %d, %Y",       # Jan 15, 2025
    "%d %B %Y",        # 15 January 2025
    "%d %b %Y",        # 15 Jan 2025
    "%B %Y",           # January 2025
    "%b %Y",           # Jan 2025
    "%Y-%m",
    "%Y/%m",
    "%m/%d/%Y",
    "%d/%m/%Y",
]


def normalize_date(raw: str | None) -> str | None:
    """Normalize a free-text date string to ``YYYY-MM`` format.

    Returns ``None`` if the input is empty or unparseable.
    """
    if not raw or not raw.strip():
        return None

    raw = raw.strip()

    # Fast path: already in YYYY-MM
    if re.fullmatch(r"\d{4}-\d{2}", raw):
        return raw

    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y-%m")
        except ValueError:
            continue

    # A bare year (e.g. "sometime in 2024") is not precise enough to
    # normalise to YYYY-MM — fabricating a month would be semantically
    # incorrect.  Return None and let the caller decide how to handle it.
    return None


# =============================================================================
# Reliability scoring
# =============================================================================

def compute_reliability(external_signals: list) -> int:
    """Return a reliability score (0–5) based on the number of external signals."""
    return min(5, len(external_signals))
