# ClientBrief AI v4 — Project Specification

## Overview
ClientBrief AI is a multi-agent pre-meeting intelligence system for consultants, account managers, and business development teams.

It generates a structured briefing by combining:
- optional free-text meeting intake (LLM-powered pre-processing layer)
- company-level research
- recent strategic signals
- optional person/contact research
- audience/function context
- hybrid engagement understanding (explicit user input + LLM-inferred context)
- pain point hypotheses
- meeting strategy

The purpose is to reduce repetitive desk research and produce a practical briefing before a client or prospect meeting.

Users can either provide structured fields directly or describe the meeting in free text — the system handles both paths.

---

## Product goal
Given a company, meeting context, and optionally a contact name/title, the system should output a markdown briefing that helps the user:
- understand the company quickly
- identify recent strategic developments
- understand the likely audience priorities
- understand where we are in the relationship lifecycle (inferred if not stated)
- anticipate pain points calibrated to the engagement stage
- ask better questions
- enter the meeting with a credible conversation angle

---

## Recommended MVP stack
For the initial MVP, use:
- **LLM provider:** Google Gemini API (`google-genai` SDK)
- **Initial model:** Gemini 3.1 Flash Lite Preview (`gemini-3.1-flash-lite-preview`)
- **Search provider:** Tavily
- **Workflow orchestration:** LangGraph
- **Demo UI later:** Streamlit

Rationale:
- Gemini 3.1 Flash Lite is cost-efficient ($0.25/1M input, $1.50/1M output) with fast inference
- Native structured output support via JSON schema + Pydantic integration
- Tavily is well-suited for source-attributed web research
- Streamlit is the fastest path to a demoable interface

Only if quality is insufficient, test **Claude Sonnet 4.6** on:
- `pain_point_agent.py`
- `meeting_strategy_agent.py`

Do not start with self-hosted open-source models unless there is a hard requirement.

---

## Inputs
Required inputs:

```python
company_name: str
company_domain: str
meeting_type: str        # consulting_intro | sales_discovery | account_review
stakeholder_role: str    # free-text role/title
meeting_goal: str
```

Optional inputs:

```python
contact_name: str | None
contact_title: str | None

# Engagement / relationship context (all optional — hybrid inference fills gaps)
meeting_stage: str | None          # first_intro | discovery_followup | solution_discussion
                                   # | proposal_review | negotiation | client_kickoff
                                   # | account_expansion   (default: first_intro)
previous_meeting_outcome: str | None   # positive | neutral | unclear | no_previous_meeting
relationship_status: str | None    # prospect | active_opportunity | client | former_client
meeting_notes_summary: str | None  # free-text notes from prior interactions

# User-side context (all optional — who the user is and what they want)
user_role: str | None              # user's own role/title
user_company: str | None           # user's company/organisation
user_function: str | None          # user's functional area (sales, consulting, etc.)
engagement_type: str | None        # interaction lens (sales, consulting, interview, etc.)
desired_outcome: str | None        # what the user wants from this meeting
success_definition: str | None     # what a successful meeting looks like
```

When engagement fields are omitted, the system infers them from context
(meeting_goal, meeting_type, stakeholder_role, notes) using an LLM.
Explicit values always take precedence over inferred ones.

### Free-text input (via Meeting Intake Agent)

Alternatively, the user can provide a single free-text description:

```python
raw_meeting_context: str   # "Second meeting with Stripe's CFO about pricing…"
```

The **Meeting Intake Agent** (a pre-workflow LLM layer, NOT a LangGraph node) parses
this into the structured fields above, plus metadata:

```python
# Intake metadata (stored in state for audit / UI confirmation)
raw_meeting_context: str              # original text preserved
intake_missing_fields: list[str]      # important fields still null
intake_low_confidence_fields: list[dict]  # {field, value, reason}
intake_confirmation_summary: str      # human-readable summary
```

The main workflow always accepts structured input. The intake layer is an optional
pre-processing step — both paths (direct structured input and intake-parsed input)
are fully supported.

---

## Core principles
1. Deterministic logic for rule-based tasks
2. LLM reasoning only where interpretation adds value
3. Facts, inferences, and hypotheses must be clearly separated
4. All extracted information must preserve sources
5. Person-level research requires confidence-aware identity matching
6. Function/department context is more important than raw title labels
7. The final brief should be concise, readable, and practical
8. Engagement understanding is hybrid: explicit input when present, inferred context when missing
9. Research depth adapts to engagement stage via `research_mode`
10. Free-text input is supported via an optional pre-workflow intake layer; the main pipeline never depends on it

---

## Architecture

### Pre-workflow layer

```text
Free-text meeting context (optional)
    |
Meeting Intake Agent          (LLM — standalone, NOT a LangGraph node)
    |
    → structured fields + missing fields + low-confidence flags + confirmation summary
    |
(future: CLI/UI confirmation — user fills gaps, corrects values)
    |
Structured input dict → app.invoke({…})
```

### Main workflow

```text
Structured Input (direct or from intake)
    |
Orchestrator
    |
Engagement Inference Agent      (LLM — infers missing engagement fields)
    |
Engagement Context Agent        (deterministic — resolves final context + research_mode)
    |
Research Planner                (deterministic — engagement-aware query planning)
    |
    +-- Company Research Agent
    +-- External Signals Agent
    +-- Person Research Agent
    |
Audience Context Agent
    |
Pain Point Hypothesis Agent
    |
Meeting Strategy Agent
    |
Brief Composer Agent
    |
Structured Markdown Brief
```

### Why this architecture
- `Meeting Intake Agent` (pre-workflow) converts free text to structured fields; fully optional — the workflow works without it
- `Orchestrator` validates and normalizes all inputs; passes through engagement fields and intake metadata as-is (None if not provided)
- `Engagement Inference Agent` uses LLM to infer missing engagement fields from meeting context; short-circuits when all fields are provided
- `Engagement Context Agent` merges explicit > inferred > default values deterministically; derives `research_mode`
- `Research Planner` converts meeting context + engagement context into a search plan, adapting depth via `research_mode`
- Three research agents gather independent evidence in parallel
- `Audience Context Agent` converts title/person/context into audience priorities (reads engagement_context from shared state)
- `Pain Point Hypothesis Agent` synthesizes research into business challenges, calibrated to engagement stage; uses user-side context as a framing lens (not evidence)
- `Meeting Strategy Agent` turns insights into actionable meeting preparation, driven by engagement context and user-side context (engagement_type, desired_outcome, success_definition)
- `Brief Composer` assembles the final markdown output deterministically

---

## Shared state

```python
from typing import TypedDict, Optional

class BriefingState(TypedDict):
    # Pre-intake metadata (optional — populated only when intake agent was used)
    raw_meeting_context: Optional[str]
    intake_missing_fields: list
    intake_low_confidence_fields: list
    intake_confirmation_summary: Optional[str]

    # Input
    company_name: str
    domain: str
    meeting_type: str
    stakeholder_role: str
    meeting_goal: str
    contact_name: Optional[str]
    contact_title: Optional[str]

    # Engagement inputs (optional — None means "not provided by user")
    meeting_stage: Optional[str]
    previous_meeting_outcome: Optional[str]
    relationship_status: Optional[str]
    meeting_notes_summary: Optional[str]

    # User-side context (optional — extracted by intake or provided directly)
    user_role: Optional[str]
    user_company: Optional[str]
    user_function: Optional[str]
    engagement_type: Optional[str]
    desired_outcome: Optional[str]
    success_definition: Optional[str]

    # Deterministic derived fields
    normalized_function: str
    normalized_seniority: Optional[str]

    # Engagement inference (LLM-inferred gap-fill)
    engagement_inference: dict

    # Agent outputs
    research_plan: dict
    company_profile: dict
    external_signals: list
    person_profile: dict
    audience_context: dict
    engagement_context: dict      # includes research_mode
    pain_point_hypotheses: dict
    meeting_strategy: dict
    final_brief: str

    # Metadata
    sources: list
    brief_reliability: int
    cache_hit: bool
    errors: list
```

---

## Node specifications

### 0. Meeting Intake Agent (pre-workflow)
**Type:** LLM-assisted
**File:** `agents/meeting_intake_agent.py`
**NOT a LangGraph node** — standalone function called before `app.invoke()`

Purpose:
- convert free-text meeting descriptions into structured workflow fields
- report missing fields and low-confidence extractions
- produce a human-readable confirmation summary for UI/CLI review

Input:
- `raw_meeting_context: str` — free-text meeting description

Output:
```python
{
    # All workflow input fields (each str or None)
    "company_name": str,
    "domain": str,
    "contact_name": str | None,
    "contact_title": str | None,
    "stakeholder_role": str,
    "meeting_goal": str,
    "meeting_type": str,
    "meeting_stage": str | None,
    "relationship_status": str | None,
    "previous_meeting_outcome": str | None,
    "meeting_notes_summary": str | None,

    # User-side context
    "user_role": str | None,
    "user_company": str | None,
    "user_function": str | None,
    "engagement_type": str | None,
    "desired_outcome": str | None,
    "success_definition": str | None,

    # Intake metadata
    "raw_meeting_context": str,            # original text preserved
    "intake_missing_fields": list[str],    # e.g. ["Company domain", "Meeting stage (recommended)"]
    "intake_low_confidence_fields": list[dict],  # [{field, value, reason}]
    "intake_confirmation_summary": str     # human-readable summary
}
```

Behavior:
- Extracts fields from free text using structured LLM output (Pydantic schema)
- Sanitises all enum fields against known valid values; nulls out invalid values
- **Low-confidence domain suppression:** if `domain` is flagged low-confidence (guessed from company name), it is nulled out and reported as missing — prevents poisoned research queries
- **Low-confidence engagement suppression:** if `meeting_stage`, `relationship_status`, or `previous_meeting_outcome` are flagged low-confidence, they are nulled out so `engagement_inference_agent` handles them with richer context (prevents shaky intake guesses from masquerading as explicit user input)
- Reports important missing fields (company_name, domain, meeting_type, stakeholder_role, meeting_goal)
- Reports recommended missing fields (meeting_stage, relationship_status) with "(recommended)" suffix
- On LLM failure or empty input, returns valid empty structure with all important fields marked missing
- Falls back stakeholder_role to contact_title if role is empty but title exists

### 1. Orchestrator
**Type:** deterministic

Responsibilities:
- validate input fields
- normalize company name and domain
- validate meeting type
- normalize title/role into broader function and seniority
- pass through engagement inputs as-is (None if not provided)
- pass through intake metadata as-is (empty defaults when absent)
- initialize state with safe defaults
- check local cache

### 2. Engagement Inference Agent
**Type:** LLM-assisted (conditional)

Purpose:
- infer missing engagement fields from available meeting context
- short-circuit when all engagement fields are already provided

Inputs:
- `meeting_stage`, `relationship_status`, `previous_meeting_outcome` (may be None)
- `meeting_goal`, `meeting_type`, `stakeholder_role`, `contact_title`
- `meeting_notes_summary`

Behavior:
- If ALL of (meeting_stage, relationship_status, previous_meeting_outcome) are provided → skip LLM, return `confidence: "not_needed"`
- If ANY are missing → use LLM to infer the missing ones from context
- Never overwrite explicitly provided values
- Sanitise inferred values against known enums
- Return sparse/null values when evidence is weak

Output:

```python
{
    "inferred_meeting_stage": str | None,
    "inferred_relationship_status": str | None,
    "inferred_previous_meeting_outcome": str | None,
    "confidence": str,          # high | medium | low | not_needed
    "reasoning_summary": str
}
```

The inference object is stored separately from raw inputs and final context
to enable future UI confirmation flows.

### 3. Engagement Context Agent
**Type:** deterministic

Purpose:
- resolve final normalised engagement values with strict precedence:
  1. Explicit user-provided values (highest priority)
  2. Inferred values from `engagement_inference`
  3. Safe defaults (lowest priority)
- derive `research_mode` for the planner
- produce structured guidance for downstream agents

Output:

```python
{
    "interaction_stage": str,
    "relationship_status": str | None,
    "conversation_mode": str,        # exploratory | consultative | solution_oriented | commercial | partnership
    "stage_objective": str,
    "recommended_tone": str,
    "decision_risks": list[str],
    "next_step_pressure": str,       # low | moderate | high
    "continuity_context": str | None,
    "research_mode": str             # full | light | update_only
}
```

Research mode rules:
- `full`: first_intro, or former_client (any stage), or low inference confidence
- `light`: discovery_followup, solution_discussion, account_expansion
- `update_only`: proposal_review, negotiation, client_kickoff

This agent does not call an LLM.

### 4. Research Planner
**Type:** deterministic

Implementation: lookup table keyed by `(meeting_type, normalized_function)`,
with meeting-stage overlays and `research_mode` query trimming.

Now engagement-aware:
- reads `engagement_context.interaction_stage` for stage overlays
- reads `engagement_context.research_mode` to calibrate query volume:
  - `full` → all baseline + contextual + stage-overlay queries
  - `light` → reduced baseline (top 2 contextual queries), preserves stage overlays
  - `update_only` → minimal baseline (1 contextual query + recent-changes query)

Output:

```python
{
    "research_objectives": [],
    "focus_areas": [],
    "evidence_needed": [],
    "tavily_queries": [],
    "research_mode": str
}
```

### 5. Company Research Agent
**Type:** LLM-assisted extraction

Purpose:
- identify company overview, industry, business model, products, customers, geographies, positioning, complexity signals

### 6. External Signals Agent
**Type:** LLM-assisted extraction

Purpose:
- find recent strategic developments
- extract signals with factual descriptions and inferential implications

Always-on signal queries are NOT affected by research_mode — recent-signal
awareness is never eliminated.

### 7. Person Research Agent
**Type:** LLM-assisted extraction

Purpose:
- attempt to identify the meeting contact from public professional sources

### 8. Audience Context Agent
**Type:** LLM reasoning plus deterministic normalized role context

Purpose:
- build an audience profile grounded in function, seniority, meeting context, engagement context, and person profile if reliable

### 9. Pain Point Hypothesis Agent
**Type:** LLM reasoning

Purpose:
- synthesize all research into likely business challenges
- calibrate pain-point framing to the engagement stage
- use user-side context (engagement_type, user_function, desired_outcome) as a **framing lens** — not evidence. Consulting → diagnostic framing; sales → commercial framing; interview → role-context framing

### 10. Meeting Strategy Agent
**Type:** LLM reasoning

Purpose:
- turn the analysis into practical meeting prep
- engagement context is a primary driver of the strategy
- user-side context (user_role, user_company, engagement_type, desired_outcome, success_definition) is a **primary strategy driver** — shapes executive summary emphasis, opening angle, talking points, questions, objections, and suggested next step

### 11. Brief Composer
**Type:** deterministic

Purpose:
- format all structured outputs into final markdown
- compute reliability score

---

## Final brief structure
The final markdown brief includes:

```markdown
# Pre-Meeting Brief: {company_name}
**Meeting type:** {meeting_type} | **Stage:** {meeting_stage} | **Audience:** {stakeholder_role}
**Brief reliability:** {brief_reliability}/5
*Generated: {timestamp}*

## Executive Summary
...

## Company Snapshot
...

## Recent Strategic Signals
...

## Meeting Contact
...

## Audience Context
...

## Engagement Context
...

## Likely Pain Points
...

## Suggested Meeting Questions
...

## Items to Validate During Meeting
...

## Meeting Prep Notes
...

*Sources: ...*
```

---

## Deterministic tasks
These must be implemented in plain Python:
- input validation
- URL normalization
- domain extraction
- role/function normalization
- Tavily query templating
- signal deduplication
- date normalization to `YYYY-MM`
- reliability scoring
- markdown assembly
- cache read/write with TTL
- engagement context resolution (explicit > inferred > default)
- research_mode derivation
- intake enum sanitisation (post-LLM)
- intake missing-field detection (post-LLM)

---

## Caching layer
Implement file-based cache:

```python
# Cache key: SHA256(domain + meeting_type + normalized_function + stakeholder_role)
# Path: .clientbrief_cache/{hash}.json
# TTL: 24 hours
# Store: full BriefingState minus final_brief
```

---

## File structure

```text
clientbrief_ai/
+-- agents/
|   +-- meeting_intake_agent.py     # pre-workflow LLM intake (NOT a LangGraph node)
|   +-- orchestrator.py
|   +-- engagement_inference_agent.py
|   +-- engagement_context_agent.py
|   +-- research_planner.py
|   +-- company_research.py
|   +-- external_signals.py
|   +-- person_research.py
|   +-- audience_context_agent.py
|   +-- pain_point_agent.py
|   +-- meeting_strategy_agent.py
|   +-- brief_composer.py
+-- core/
|   +-- state.py
|   +-- workflow.py
|   +-- cache.py
|   +-- utils.py
+-- services/
|   +-- tavily_client.py
|   +-- llm_client.py
+-- app/
|   +-- main.py
+-- .env.example
+-- requirements.txt
+-- README.md
```

---

## LangGraph workflow

```python
from langgraph.graph import StateGraph, END
from core.state import BriefingState

graph = StateGraph(BriefingState)

# 11 nodes
graph.add_node("orchestrate", orchestrator.run)
graph.add_node("infer_engagement", engagement_inference_agent.run)
graph.add_node("engagement_context", engagement_context_agent.run)
graph.add_node("plan", research_planner.run)
graph.add_node("research_company", company_research.run)
graph.add_node("research_signals", external_signals.run)
graph.add_node("research_person", person_research.run)
graph.add_node("audience_context", audience_context_agent.run)
graph.add_node("hypothesize", pain_point_agent.run)
graph.add_node("strategize", meeting_strategy_agent.run)
graph.add_node("compose", brief_composer.run)

graph.set_entry_point("orchestrate")

# Engagement-first chain (before planning)
graph.add_edge("orchestrate", "infer_engagement")
graph.add_edge("infer_engagement", "engagement_context")
graph.add_edge("engagement_context", "plan")

# Fan-out: plan → 3 research agents
graph.add_edge("plan", "research_company")
graph.add_edge("plan", "research_signals")
graph.add_edge("plan", "research_person")

# Fan-in: 3 research agents → audience_context
graph.add_edge("research_company", "audience_context")
graph.add_edge("research_signals", "audience_context")
graph.add_edge("research_person", "audience_context")

# Sequential reasoning chain
graph.add_edge("audience_context", "hypothesize")
graph.add_edge("hypothesize", "strategize")
graph.add_edge("strategize", "compose")
graph.add_edge("compose", END)

app = graph.compile()
```

---

## CLI requirements
`app/main.py` should:
- use `rich`
- support two input modes:
  - **structured mode:** ask interactively for all fields
  - **free-text mode:** accept a free-text meeting description, run intake agent, show confirmation summary with missing/low-confidence fields, let user fill gaps or accept
- show progress while workflow runs
- optionally show inferred engagement values and ask for confirmation
- render markdown output in terminal
- save final brief to `./output/{company_name}_{timestamp}.md`
- display the reliability score prominently

---

## Build order
Implement in this order:
1. `core/state.py`
2. `core/utils.py`
3. `core/cache.py`
4. `services/tavily_client.py`
5. `agents/meeting_intake_agent.py`
6. `agents/orchestrator.py`
7. `agents/engagement_inference_agent.py`
8. `agents/engagement_context_agent.py`
9. `agents/research_planner.py`
10. `agents/company_research.py`
11. `agents/external_signals.py`
12. `agents/person_research.py`
13. `agents/audience_context_agent.py`
14. `agents/pain_point_agent.py`
15. `agents/meeting_strategy_agent.py`
16. `agents/brief_composer.py`
17. `core/workflow.py`
18. `app/main.py`

---

## Output quality rules
- no hallucinated facts
- hypotheses labeled `[HYPOTHESIS]`
- inferences labeled `[INFERENCE]`
- dates normalized to `YYYY-MM`
- duplicate signals removed
- person-level claims only used with medium/high match confidence
- final questions must be company-specific and audience-specific
- engagement inference transparent: stored separately for auditability
