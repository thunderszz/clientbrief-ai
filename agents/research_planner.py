"""Research Planner — fully deterministic, no LLM calls.

Implementation: lookup table keyed by ``(meeting_type, normalized_function)``,
with **meeting-stage overlays** that add stage-specific research objectives
and queries when the engagement stage goes beyond a first introduction.

For every combination the planner produces:
- ``research_objectives``  — what we need to learn
- ``focus_areas``          — topical angles to prioritise
- ``evidence_needed``      — concrete evidence the brief should contain
- ``tavily_queries``       — search strings fed to the Tavily client

The ``tavily_queries`` are *supplementary* — each research agent also has its
own always-on queries (company overview, signals, person).  The planner's
queries sharpen the search toward the specific meeting context.

Ownership:  ``research_plan`` only.
"""

from __future__ import annotations

from typing import Any

from core.utils import build_signal_queries


# =============================================================================
# Lookup table
# =============================================================================
# Keys: (meeting_type, normalized_function)
# Values: partial plan dicts that get merged with the company-name-templated
# queries at runtime.
#
# When a specific combo is missing the planner falls through to:
#   1. (meeting_type, "general")
#   2. ("_default", "general")

_PLAN_TABLE: dict[tuple[str, str], dict[str, list[str]]] = {

    # -------------------------------------------------------------------------
    # consulting_intro
    # -------------------------------------------------------------------------
    ("consulting_intro", "finance"): {
        "research_objectives": [
            "Understand the company's financial health and reporting maturity",
            "Identify CFO-level priorities and recent financial disclosures",
        ],
        "focus_areas": [
            "financial reporting", "margin trends", "cost optimisation",
            "budgeting and forecasting", "audit and compliance",
        ],
        "evidence_needed": [
            "Recent earnings or annual-report highlights",
            "Public statements from finance leadership",
            "Known ERP or financial-system landscape",
        ],
    },
    ("consulting_intro", "accounting"): {
        "research_objectives": [
            "Assess the company's close-process maturity and compliance posture",
            "Identify accounting team structure signals",
        ],
        "focus_areas": [
            "month-end close", "reconciliation", "internal controls",
            "regulatory compliance", "accounting systems",
        ],
        "evidence_needed": [
            "Regulatory filings or compliance mentions",
            "Accounting technology or outsourcing signals",
        ],
    },
    ("consulting_intro", "engineering"): {
        "research_objectives": [
            "Understand the engineering organisation and tech stack",
            "Identify scalability or modernisation challenges",
        ],
        "focus_areas": [
            "tech stack", "platform scalability", "developer productivity",
            "technical debt", "cloud migration",
        ],
        "evidence_needed": [
            "Job postings or blog posts revealing tech stack",
            "Engineering-leadership public talks or blog content",
        ],
    },
    ("consulting_intro", "product"): {
        "research_objectives": [
            "Map the product portfolio and roadmap signals",
            "Identify product-market fit challenges or pivots",
        ],
        "focus_areas": [
            "product roadmap", "feature prioritisation", "user analytics",
            "competitive product positioning",
        ],
        "evidence_needed": [
            "Recent product launches or sunset announcements",
            "User or analyst reviews",
        ],
    },
    ("consulting_intro", "sales"): {
        "research_objectives": [
            "Understand the go-to-market motion and sales org structure",
            "Identify pipeline or conversion pain points",
        ],
        "focus_areas": [
            "sales pipeline", "CRM and RevOps tooling", "conversion rates",
            "sales enablement", "channel strategy",
        ],
        "evidence_needed": [
            "Public revenue or growth figures",
            "Sales-leadership commentary in press or events",
        ],
    },
    ("consulting_intro", "strategy"): {
        "research_objectives": [
            "Map the company's strategic direction and competitive landscape",
            "Identify M&A activity or expansion signals",
        ],
        "focus_areas": [
            "corporate strategy", "market expansion", "competitive positioning",
            "M&A and partnerships", "growth vectors",
        ],
        "evidence_needed": [
            "Strategic announcements or investor presentations",
            "Industry analyst commentary on the company",
        ],
    },
    ("consulting_intro", "it_data"): {
        "research_objectives": [
            "Understand the IT and data landscape",
            "Identify integration, governance, or security challenges",
        ],
        "focus_areas": [
            "data governance", "system integration", "data quality",
            "cybersecurity posture", "cloud and infrastructure",
        ],
        "evidence_needed": [
            "Known technology partnerships or platform choices",
            "Data-breach or security-incident history",
        ],
    },
    ("consulting_intro", "operations"): {
        "research_objectives": [
            "Understand operational complexity and process maturity",
            "Identify efficiency or visibility gaps",
        ],
        "focus_areas": [
            "process efficiency", "supply chain", "execution visibility",
            "operational KPIs", "cross-functional coordination",
        ],
        "evidence_needed": [
            "Operational footprint (plants, warehouses, regions)",
            "Public statements on operational initiatives",
        ],
    },
    ("consulting_intro", "transformation"): {
        "research_objectives": [
            "Assess in-flight or planned transformation programmes",
            "Understand change-management maturity",
        ],
        "focus_areas": [
            "ERP transformation", "change management", "adoption metrics",
            "programme governance", "digital transformation roadmap",
        ],
        "evidence_needed": [
            "Public references to transformation projects",
            "Technology vendor partnerships (SAP, Oracle, Workday, etc.)",
        ],
    },
    ("consulting_intro", "general"): {
        "research_objectives": [
            "Build a broad understanding of the company's business and challenges",
            "Identify conversation-relevant angles for a first meeting",
        ],
        "focus_areas": [
            "business model", "competitive landscape", "recent news",
            "leadership changes", "industry trends",
        ],
        "evidence_needed": [
            "Company overview from credible sources",
            "Recent leadership or strategy announcements",
        ],
    },

    # -------------------------------------------------------------------------
    # sales_discovery
    # -------------------------------------------------------------------------
    ("sales_discovery", "finance"): {
        "research_objectives": [
            "Identify buying signals related to finance tooling or advisory",
            "Understand current financial-process pain points",
        ],
        "focus_areas": [
            "budgeting tools", "financial consolidation", "reporting automation",
            "treasury management", "cost reduction initiatives",
        ],
        "evidence_needed": [
            "Technology spend signals or vendor mentions",
            "Finance-team job postings hinting at gaps",
        ],
    },
    ("sales_discovery", "accounting"): {
        "research_objectives": [
            "Identify buying signals related to accounting automation",
            "Understand close-process bottlenecks",
        ],
        "focus_areas": [
            "close automation", "reconciliation tools", "compliance requirements",
            "audit readiness",
        ],
        "evidence_needed": [
            "Mentions of accounting software or migration",
            "Regulatory pressure signals",
        ],
    },
    ("sales_discovery", "engineering"): {
        "research_objectives": [
            "Identify buying signals for developer tools or platform services",
            "Understand current engineering bottlenecks",
        ],
        "focus_areas": [
            "developer experience", "CI/CD pipeline", "observability",
            "platform engineering", "cloud costs",
        ],
        "evidence_needed": [
            "Tech-blog posts on infrastructure challenges",
            "Engineering hiring patterns",
        ],
    },
    ("sales_discovery", "product"): {
        "research_objectives": [
            "Identify buying signals for product-management tooling",
            "Understand product-team maturity and process gaps",
        ],
        "focus_areas": [
            "product analytics", "experimentation", "roadmap visibility",
            "feature-flagging", "customer feedback loops",
        ],
        "evidence_needed": [
            "Product-team job postings or conference talks",
            "Public product changelog or community feedback",
        ],
    },
    ("sales_discovery", "sales"): {
        "research_objectives": [
            "Identify sales-tooling gaps or CRM modernisation signals",
            "Understand quota and pipeline challenges",
        ],
        "focus_areas": [
            "CRM effectiveness", "pipeline analytics", "sales automation",
            "lead scoring", "revenue operations",
        ],
        "evidence_needed": [
            "Mentions of CRM vendors or RevOps tools",
            "Sales-leadership commentary on growth targets",
        ],
    },
    ("sales_discovery", "strategy"): {
        "research_objectives": [
            "Identify strategic initiatives that create budget for new solutions",
            "Understand competitive threats driving urgency",
        ],
        "focus_areas": [
            "strategic investments", "market entry", "competitive response",
            "innovation initiatives", "board-level priorities",
        ],
        "evidence_needed": [
            "Investor presentations or earnings-call transcripts",
            "Press coverage of strategic moves",
        ],
    },
    ("sales_discovery", "it_data"): {
        "research_objectives": [
            "Identify IT spending signals and modernisation priorities",
            "Understand data-platform maturity and gaps",
        ],
        "focus_areas": [
            "IT modernisation", "data platform", "vendor consolidation",
            "security compliance", "integration backlog",
        ],
        "evidence_needed": [
            "Technology partnership announcements",
            "IT-leadership conference presentations",
        ],
    },
    ("sales_discovery", "operations"): {
        "research_objectives": [
            "Identify operational-tooling gaps or process-improvement budgets",
            "Understand supply-chain or logistics pain points",
        ],
        "focus_areas": [
            "process automation", "supply-chain visibility", "inventory optimisation",
            "operational analytics", "vendor management",
        ],
        "evidence_needed": [
            "Operational KPIs in public filings",
            "Mentions of operational-improvement programmes",
        ],
    },
    ("sales_discovery", "transformation"): {
        "research_objectives": [
            "Identify active transformation budgets and programme status",
            "Understand adoption and change-management readiness",
        ],
        "focus_areas": [
            "programme status", "vendor selection signals", "rollout challenges",
            "user adoption", "training and enablement",
        ],
        "evidence_needed": [
            "RFP or vendor-selection press mentions",
            "Employee commentary on transformation progress",
        ],
    },
    ("sales_discovery", "general"): {
        "research_objectives": [
            "Identify potential buying triggers across the organisation",
            "Build a business-context map to position the offering",
        ],
        "focus_areas": [
            "company growth trajectory", "recent challenges", "technology landscape",
            "competitive pressure", "hiring trends",
        ],
        "evidence_needed": [
            "Recent funding, revenue, or headcount signals",
            "Job postings indicating new initiatives",
        ],
    },

    # -------------------------------------------------------------------------
    # account_review
    # -------------------------------------------------------------------------
    ("account_review", "finance"): {
        "research_objectives": [
            "Track recent developments in the finance function since last engagement",
            "Surface new pain points or expansion opportunities",
        ],
        "focus_areas": [
            "new regulatory requirements", "financial system changes",
            "leadership turnover", "budget cycle timing",
        ],
        "evidence_needed": [
            "Quarterly or annual report updates",
            "Finance-leadership changes or announcements",
        ],
    },
    ("account_review", "accounting"): {
        "research_objectives": [
            "Identify changes in accounting requirements or team structure",
            "Surface compliance or audit-related news",
        ],
        "focus_areas": [
            "regulatory changes", "restatement or audit findings",
            "system migrations", "outsourcing shifts",
        ],
        "evidence_needed": [
            "Regulatory-body notices or filings",
            "Accounting-team job changes",
        ],
    },
    ("account_review", "engineering"): {
        "research_objectives": [
            "Track engineering-team evolution and tech-stack changes",
            "Identify new technical initiatives since last meeting",
        ],
        "focus_areas": [
            "new tech adoption", "platform migration", "team scaling",
            "reliability incidents", "open-source contributions",
        ],
        "evidence_needed": [
            "Engineering blog updates",
            "Conference talks or open-source releases",
        ],
    },
    ("account_review", "product"): {
        "research_objectives": [
            "Track product launches and roadmap shifts",
            "Identify new user feedback or competitive pressure",
        ],
        "focus_areas": [
            "new feature releases", "product-strategy pivots",
            "user sentiment", "competitive product moves",
        ],
        "evidence_needed": [
            "Product changelog or release notes",
            "App-store or community reviews",
        ],
    },
    ("account_review", "sales"): {
        "research_objectives": [
            "Track GTM changes and sales-org restructuring",
            "Identify new revenue targets or channel shifts",
        ],
        "focus_areas": [
            "GTM strategy changes", "sales-team restructuring",
            "new market entry", "partnership announcements",
        ],
        "evidence_needed": [
            "Revenue or ARR growth figures",
            "Sales-leadership changes",
        ],
    },
    ("account_review", "strategy"): {
        "research_objectives": [
            "Track strategic pivots, M&A, or market-expansion moves",
            "Refresh competitive-landscape context",
        ],
        "focus_areas": [
            "M&A activity", "new market entry", "competitive response",
            "board or investor changes", "strategic partnerships",
        ],
        "evidence_needed": [
            "Investor updates or SEC filings",
            "Press coverage of strategic moves",
        ],
    },
    ("account_review", "it_data"): {
        "research_objectives": [
            "Track IT-landscape changes and new vendor relationships",
            "Identify data-governance or security incidents",
        ],
        "focus_areas": [
            "new vendor implementations", "security incidents",
            "data-governance initiatives", "cloud-migration progress",
        ],
        "evidence_needed": [
            "Technology partnership updates",
            "Security-incident disclosures",
        ],
    },
    ("account_review", "operations"): {
        "research_objectives": [
            "Track operational changes and efficiency initiatives",
            "Identify supply-chain or process disruptions",
        ],
        "focus_areas": [
            "operational restructuring", "supply-chain disruptions",
            "new facility or expansion", "process-improvement outcomes",
        ],
        "evidence_needed": [
            "Operational KPI changes in filings",
            "News on facility openings or closures",
        ],
    },
    ("account_review", "transformation"): {
        "research_objectives": [
            "Track transformation programme progress and roadblocks",
            "Identify phase-two or expansion opportunities",
        ],
        "focus_areas": [
            "programme milestones", "adoption metrics", "change fatigue signals",
            "vendor satisfaction", "scope changes",
        ],
        "evidence_needed": [
            "Programme-status references in press or analyst reports",
            "Employee sentiment on transformation",
        ],
    },
    ("account_review", "general"): {
        "research_objectives": [
            "Refresh overall company context since last meeting",
            "Surface any new strategic, financial, or organisational signals",
        ],
        "focus_areas": [
            "leadership changes", "financial performance",
            "strategic announcements", "market sentiment",
        ],
        "evidence_needed": [
            "Latest press releases or earnings highlights",
            "Employee or customer sentiment signals",
        ],
    },
}

# Ultimate fallback when meeting_type itself is unrecognised (shouldn't
# happen after orchestrator validation, but defensive).
_DEFAULT_PLAN: dict[str, list[str]] = {
    "research_objectives": [
        "Build a foundational understanding of the company and its context",
    ],
    "focus_areas": [
        "business model", "recent news", "industry trends",
    ],
    "evidence_needed": [
        "Company overview from credible sources",
    ],
}


# =============================================================================
# Meeting-stage overlays
# =============================================================================
# These add *supplementary* objectives, focus areas, and query hints when the
# meeting stage is beyond a first introduction.  They are merged into the
# base plan at runtime.  ``first_intro`` has no overlay — it is the default.

_STAGE_OVERLAYS: dict[str, dict[str, list[str]]] = {
    # first_intro — no overlay needed; base plan is already intro-oriented
    "discovery_followup": {
        "extra_objectives": [
            "Identify developments since the previous meeting",
            "Deepen understanding of specific pain points surfaced earlier",
        ],
        "extra_focus": ["follow-up topics", "updated competitive moves"],
        "extra_query_hints": ["recent developments", "updated strategy"],
    },
    "solution_discussion": {
        "extra_objectives": [
            "Understand the evaluation criteria and decision process",
            "Identify technical or organisational constraints for the solution",
        ],
        "extra_focus": ["vendor evaluation", "implementation requirements"],
        "extra_query_hints": ["technology evaluation", "implementation challenges"],
    },
    "proposal_review": {
        "extra_objectives": [
            "Identify budget cycle timing and procurement process",
            "Surface any competitive alternatives under consideration",
        ],
        "extra_focus": ["budget and procurement", "competitive alternatives"],
        "extra_query_hints": ["procurement process", "budget approval"],
    },
    "negotiation": {
        "extra_objectives": [
            "Understand pricing sensitivity and deal-blocking risks",
            "Identify internal champions and detractors",
        ],
        "extra_focus": ["pricing benchmarks", "decision-maker dynamics"],
        "extra_query_hints": ["pricing", "contract negotiation"],
    },
    "client_kickoff": {
        "extra_objectives": [
            "Understand the client's expected outcomes and success criteria",
            "Map key stakeholders for the engagement",
        ],
        "extra_focus": ["success criteria", "stakeholder map", "onboarding"],
        "extra_query_hints": ["implementation kickoff", "success metrics"],
    },
    "account_expansion": {
        "extra_objectives": [
            "Identify adjacent business units or functions with unmet needs",
            "Surface new strategic initiatives that create expansion opportunities",
        ],
        "extra_focus": ["cross-sell opportunities", "new initiatives", "org changes"],
        "extra_query_hints": ["new initiatives", "expansion plans", "organisational changes"],
    },
}


# =============================================================================
# Query builder (meeting-type + function → Tavily queries)
# =============================================================================

def _build_contextual_queries(
    company_name: str,
    meeting_type: str,
    normalized_function: str,
    focus_areas: list[str],
) -> list[str]:
    """Generate Tavily queries that combine company name with focus areas.

    We take the top focus areas and turn them into targeted search strings.
    These are *supplementary* to the always-on queries in ``build_signal_queries``.
    """
    queries: list[str] = []
    # Use up to 3 focus areas to keep query volume reasonable.
    for area in focus_areas[:3]:
        queries.append(f'"{company_name}" {area}')

    # Add a meeting-type-flavoured query.
    _MEETING_FLAVOUR = {
        "consulting_intro": "challenges priorities strategy",
        "sales_discovery": "technology tools vendor evaluation",
        "account_review": "recent news updates changes",
    }
    flavour = _MEETING_FLAVOUR.get(meeting_type, "")
    if flavour:
        queries.append(f'"{company_name}" {normalized_function} {flavour}')

    return queries


# =============================================================================
# Research-mode query modifiers
# =============================================================================
# These control how research_mode affects query volume and focus.
#
# ``full``        — run all baseline + contextual + stage-overlay queries.
#                   This is the default for first-intro / low-context meetings.
#
# ``light``       — reduce baseline queries (keep only the domain query for
#                   company overview), but preserve all signal queries and
#                   stage overlays.  Suitable for follow-ups where baseline
#                   context is partially known.
#
# ``update_only`` — strip baseline company queries entirely; keep only
#                   signal-focused and stage-overlay queries.  The assumption
#                   is that the company is well-known from prior interactions.
#                   Recent-signal awareness is NEVER eliminated.

def _trim_queries_for_mode(
    research_mode: str,
    focus_areas: list[str],
    contextual_queries: list[str],
    company_name: str,
    domain: str,
) -> list[str]:
    """Trim the contextual Tavily queries based on research_mode.

    The always-on signal queries (from ``build_signal_queries``) are handled
    by the external-signals agent and are NOT affected by this function.
    This only trims the *planner-generated* contextual queries.
    """
    if research_mode == "full":
        # No trimming — return all contextual queries.
        return contextual_queries

    if research_mode == "light":
        # Keep up to 2 contextual queries (the most targeted ones).
        # Drop the generic meeting-type-flavoured query (typically last).
        return contextual_queries[:2]

    if research_mode == "update_only":
        # Keep only 1 contextual query (the most specific focus area).
        # Add an explicit update/recent-changes query instead.
        trimmed = contextual_queries[:1]
        trimmed.append(f'"{company_name}" recent changes updates news')
        return trimmed

    # Unknown mode — fall back to full.
    return contextual_queries


# =============================================================================
# Public entry-point
# =============================================================================

def run(state: dict[str, Any]) -> dict[str, Any]:
    """Produce the research plan from meeting context and engagement context.

    Now engagement-aware: reads ``engagement_context`` (including
    ``research_mode``) to calibrate research depth.

    Writes only ``research_plan``.
    """
    company_name: str = state.get("company_name", "")
    domain: str = state.get("domain", "")
    meeting_type: str = state.get("meeting_type", "")
    normalized_function: str = state.get("normalized_function", "general")

    # Read engagement context — available because the workflow now runs
    # engagement_inference → engagement_context → plan (sequential).
    ec: dict[str, Any] = state.get("engagement_context", {})
    meeting_stage: str = ec.get("interaction_stage", "first_intro")
    research_mode: str = ec.get("research_mode", "full")

    # --- Base lookup ----------------------------------------------------------
    plan_data = (
        _PLAN_TABLE.get((meeting_type, normalized_function))
        or _PLAN_TABLE.get((meeting_type, "general"))
        or _DEFAULT_PLAN
    )

    research_objectives = list(plan_data.get("research_objectives", []))
    focus_areas = list(plan_data.get("focus_areas", []))
    evidence_needed = list(plan_data.get("evidence_needed", []))

    # --- Meeting-stage overlay ------------------------------------------------
    overlay = _STAGE_OVERLAYS.get(meeting_stage)
    if overlay:
        research_objectives.extend(overlay.get("extra_objectives", []))
        focus_areas.extend(overlay.get("extra_focus", []))

    # --- Build base contextual queries ----------------------------------------
    contextual = _build_contextual_queries(
        company_name, meeting_type, normalized_function, focus_areas,
    )

    # Add stage-specific query hints if available
    if overlay:
        for hint in overlay.get("extra_query_hints", [])[:2]:
            contextual.append(f'"{company_name}" {hint}')

    # --- Apply research_mode trimming -----------------------------------------
    tavily_queries = _trim_queries_for_mode(
        research_mode, focus_areas, contextual, company_name, domain,
    )

    # --- Annotate plan with research_mode for transparency -------------------
    research_plan: dict[str, Any] = {
        "research_objectives": research_objectives,
        "focus_areas": focus_areas,
        "evidence_needed": evidence_needed,
        "tavily_queries": tavily_queries,
        "research_mode": research_mode,
    }

    return {"research_plan": research_plan}
