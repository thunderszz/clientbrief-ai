"""Deterministic helpers: domains, roles, input validation, search queries, signals and dates."""

from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urlparse

from core.state import (
    VALID_MEETING_STAGES,
    VALID_MEETING_TYPES,
)


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


# (keyword, function, whole_word). Whole-word keywords are matched with \b, so "cto" does not
# match inside "director". The first match wins, so more specific keywords come first.
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

# (keyword, seniority, whole_word), same convention as above.
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


def _first_match(role: str, keywords: list[tuple[str, str, bool]]) -> str | None:
    text = role.lower().strip()
    for keyword, value, whole_word in keywords:
        found = re.search(rf"\b{re.escape(keyword)}\b", text) if whole_word else keyword in text
        if found:
            return value
    return None


def normalize_function(role: str) -> str:
    """Map a free-text title to a business function, or ``"general"`` if none matches."""
    return _first_match(role, _FUNCTION_KEYWORDS) or "general"


def normalize_seniority(role: str) -> str | None:
    """Map a free-text title to a seniority level, or None if it does not say."""
    return _first_match(role, _SENIORITY_KEYWORDS)


def validate_inputs(
    company_name: str,
    domain: str,
    meeting_type: str,
    stakeholder_role: str,
    meeting_stage: str | None = None,
) -> list[str]:
    """Return one message per invalid field; an empty list means the input is valid.

    ``meeting_stage`` is optional, but when given it must be a known stage.
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


def build_company_queries(company_name: str, domain: str) -> list[str]:
    """Search queries for an overview of the company."""
    return [
        f'"{company_name}" company overview business model',
        f'"{company_name}" products customers industry',
        f"{domain}",
    ]


def build_signal_queries(
    company_name: str,
    extra_queries: list[str] | None = None,
) -> list[str]:
    """Search queries for recent news: the standard three plus *extra_queries* from the plan."""
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
    """Search queries that tend to surface team pages, speaker bios and interviews."""
    queries = [
        f'"{contact_name}" "{company_name}"',
        f'"{contact_name}" "{company_name}" speaker bio',
        f'"{contact_name}" "{company_name}" interview',
    ]
    if contact_title:
        queries.append(f'"{contact_name}" "{company_name}" "{contact_title}"')
    return queries


def _signal_fingerprint(signal: dict) -> str:
    desc = re.sub(r"[^\w\s]", "", signal.get("description", "").lower()).strip()
    return re.sub(r"\s+", " ", desc)


def deduplicate_signals(signals: list[dict]) -> list[dict]:
    """Drop signals whose description matches an earlier one once case, punctuation and
    spacing are ignored. Signals with an empty description are dropped too."""
    seen: set[str] = set()
    unique: list[dict] = []
    for signal in signals:
        fingerprint = _signal_fingerprint(signal)
        if fingerprint and fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(signal)
    return unique


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
