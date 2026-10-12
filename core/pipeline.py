"""Single source of truth for the pipeline's shape.

``core/workflow.py`` builds the LangGraph graph from these definitions, and the
API serves them to the UI so it can draw the graph without hard-coding it.
"""

from __future__ import annotations

PIPELINE_NODES: list[dict[str, str]] = [
    {
        "id": "orchestrate",
        "label": "Validate inputs",
        "kind": "deterministic",
        "phase": "setup",
        "description": "Validates the meeting fields and normalises role, function and seniority.",
    },
    {
        "id": "infer_engagement",
        "label": "Infer engagement",
        "kind": "llm",
        "phase": "setup",
        "description": "Infers meeting stage and relationship when the user did not state them.",
    },
    {
        "id": "engagement_context",
        "label": "Engagement context",
        "kind": "deterministic",
        "phase": "setup",
        "description": "Resolves stage, tone, risks and how deep the research should go.",
    },
    {
        "id": "plan",
        "label": "Plan research",
        "kind": "deterministic",
        "phase": "setup",
        "description": "Builds research objectives and search queries for the research agents.",
    },
    {
        "id": "research_company",
        "label": "Company research",
        "kind": "search+llm",
        "phase": "research",
        "description": "Searches the web and extracts a structured company profile.",
    },
    {
        "id": "research_signals",
        "label": "Strategic signals",
        "kind": "search+llm",
        "phase": "research",
        "description": "Finds recent funding, launches, hires and partnerships.",
    },
    {
        "id": "research_person",
        "label": "Contact lookup",
        "kind": "search+llm",
        "phase": "research",
        "description": "Checks whether public sources reliably identify the contact.",
    },
    {
        "id": "audience_context",
        "label": "Audience context",
        "kind": "llm",
        "phase": "reasoning",
        "description": "Works out what this stakeholder likely cares about.",
    },
    {
        "id": "hypothesize",
        "label": "Pain-point hypotheses",
        "kind": "llm",
        "phase": "reasoning",
        "description": "Separates facts, inferences and hypotheses about likely pain points.",
    },
    {
        "id": "strategize",
        "label": "Meeting strategy",
        "kind": "llm",
        "phase": "reasoning",
        "description": "Drafts the opening, questions, objections and next step.",
    },
    {
        "id": "compose",
        "label": "Compose brief",
        "kind": "deterministic",
        "phase": "output",
        "description": "Assembles the brief and scores how well-grounded it is.",
    },
]

PIPELINE_EDGES: list[tuple[str, str]] = [
    ("orchestrate", "infer_engagement"),
    ("infer_engagement", "engagement_context"),
    ("engagement_context", "plan"),
    ("plan", "research_company"),
    ("plan", "research_signals"),
    ("plan", "research_person"),
    ("research_company", "audience_context"),
    ("research_signals", "audience_context"),
    ("research_person", "audience_context"),
    ("audience_context", "hypothesize"),
    ("hypothesize", "strategize"),
    ("strategize", "compose"),
]

NODE_LABELS: dict[str, str] = {node["id"]: node["label"] for node in PIPELINE_NODES}
