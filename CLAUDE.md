# ClientBrief AI — Claude Code Working Instructions

## Project goal
Build **ClientBrief AI**, a multi-agent pre-meeting intelligence system that prepares a structured briefing for meetings with companies.

The system should:
- accept free-text meeting context and extract structured fields (pre-intake layer)
- gather public company information
- detect recent strategic signals
- optionally research the meeting contact if identifiable from reliable public sources
- build audience/function context
- infer and interpret relationship and meeting-stage context (hybrid: explicit + inferred)
- infer likely pain points
- generate meeting strategy and questions
- compose a final markdown brief

## Recommended MVP stack
Use this stack for the first implementation:
- **LLM provider:** Google Gemini API (`google-genai` SDK)
- **Initial model:** Gemini 3.1 Flash Lite Preview (`gemini-3.1-flash-lite-preview`)
- **Search provider:** Tavily
- **Workflow orchestration:** LangGraph
- **UI later:** Streamlit

Do **not** start with self-hosted open-source models unless there is a hard requirement to avoid commercial APIs.
Prioritize reliability, structured outputs, and speed of iteration.

If higher-quality reasoning is needed later, test **Claude Sonnet 4.6** only on:
- `pain_point_agent.py`
- `meeting_strategy_agent.py`

## Core architecture
The system has two layers:

1. **Pre-intake layer** (`meeting_intake_agent.py`) — an optional LLM-powered parser that converts free-text meeting descriptions into structured fields. Called *before* the main workflow. Not a LangGraph node.
2. **Main LangGraph pipeline** — the stateful multi-agent workflow that produces the final brief.

The main workflow accepts structured input directly (no intake needed) or structured input produced by the intake layer. Both paths are fully supported.

### Pre-intake layer

```text
Free-text meeting context
    |
Meeting Intake Agent          (LLM — extracts structured fields from text)
    |
    → extracted fields + missing fields + low-confidence fields + confirmation summary
    |
(future: CLI/UI confirmation step — user fills gaps, confirms/corrects)
    |
Structured workflow input → app.invoke({…})
```

The intake agent:
- extracts all workflow input fields (company_name, domain, meeting_type, stakeholder_role, meeting_goal, contact_name, contact_title, meeting_stage, relationship_status, previous_meeting_outcome, meeting_notes_summary)
- sanitises enum fields against known valid values
- reports which important fields are still missing
- flags fields extracted with low confidence (with reason)
- produces a human-readable confirmation summary
- preserves the raw text in `raw_meeting_context` for auditability

### Main workflow graph

```text
User Input (structured — direct or from intake)
    |
Orchestrator
    |
Engagement Inference Agent     (LLM — infers missing engagement fields)
    |
Engagement Context Agent       (deterministic — resolves final context + research_mode)
    |
Research Planner               (deterministic — engagement-aware query planning)
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
Final Markdown Brief
```

## Design rules
1. Use **deterministic Python logic** for:
   - input validation
   - URL/domain normalization
   - role/function normalization
   - research-plan lookup
   - query templating
   - deduplication
   - date normalization
   - reliability scoring
   - markdown formatting
   - cache read/write
   - engagement context resolution (merge explicit + inferred + defaults)
   - research_mode derivation

2. Use **LLMs only** for:
   - free-text meeting intake parsing (pre-workflow)
   - fact extraction from retrieved text
   - implication generation
   - audience reasoning
   - pain point synthesis
   - meeting strategy generation
   - engagement field inference (when explicit inputs are missing)

3a. Use **hybrid approach** for engagement understanding:
   - engagement inputs are optional — user may provide all, some, or none
   - `engagement_inference_agent` uses LLM to infer missing fields from context
   - `engagement_context_agent` merges with strict precedence: **explicit > inferred > default**
   - final engagement context is always deterministic
   - `research_mode` (full / light / update_only) is derived deterministically

3b. Use **deterministic logic** for engagement context resolution:
   - meeting-stage → conversation mode, tone, pressure
   - relationship-status resolution
   - stage-appropriate risk seeds
   - continuity context formatting
   - research_mode derivation

3c. Always separate:
   - **facts**
   - **inferences**
   - **hypotheses**

4. Person-level information must only be used if match confidence is **medium** or **high**.

5. Prefer **function/department reasoning** over title-only reasoning.

6. **User-side context** (user_role, user_company, user_function, engagement_type, desired_outcome, success_definition) is a **framing lens** for pain_point_agent and a **primary driver** for meeting_strategy_agent. It does NOT provide evidence — facts must still come from research. The `audience_context_agent` intentionally does NOT consume user-side context — audience context is about understanding the counterpart, not the user. The `brief_composer` renders user-side context in a "Your Position" section when present.

## State model
Use a shared state object similar to:

```python
from typing import TypedDict, Optional

class BriefingState(TypedDict):
    # Pre-intake metadata (optional — populated only when intake agent was used)
    raw_meeting_context: Optional[str]
    intake_missing_fields: list
    intake_low_confidence_fields: list
    intake_confirmation_summary: Optional[str]

    # User inputs
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

    sources: list
    brief_reliability: int
    cache_hit: bool
    errors: list
```

### Pre-intake metadata fields
These fields are populated only when the meeting intake agent is used before the workflow.
When using direct structured input, they are absent or empty.

- `raw_meeting_context`: the original free-text input (preserved for audit)
- `intake_missing_fields`: list of important field labels still null after intake
- `intake_low_confidence_fields`: list of `{field, value, reason}` dicts
- `intake_confirmation_summary`: human-readable summary for user confirmation

### `engagement_inference` schema
```python
{
    "inferred_meeting_stage": str | None,
    "inferred_relationship_status": str | None,
    "inferred_previous_meeting_outcome": str | None,
    "confidence": str,          # high | medium | low | not_needed
    "reasoning_summary": str
}
```

### `engagement_context` schema (updated)
```python
{
    "interaction_stage": str,
    "relationship_status": str | None,
    "conversation_mode": str,
    "stage_objective": str,
    "recommended_tone": str,
    "decision_risks": list[str],
    "next_step_pressure": str,
    "continuity_context": str | None,
    "research_mode": str        # full | light | update_only
}
```

## Agent ownership
Each node/layer owns only its slice of state:
- `meeting_intake_agent` → pre-workflow; returns structured fields + user-side context + intake metadata (not a LangGraph node)
- `orchestrator` → normalized inputs, engagement inputs (raw), user-side context pass-through, cache flags, intake metadata pass-through
- `engagement_inference_agent` → `engagement_inference`
- `engagement_context_agent` → `engagement_context` (including `research_mode`)
- `research_planner` → `research_plan`
- `company_research` → `company_profile`
- `external_signals` → `external_signals`
- `person_research` → `person_profile`
- `audience_context_agent` → `audience_context`
- `pain_point_agent` → `pain_point_hypotheses` (reads user-side context as framing lens)
- `meeting_strategy_agent` → `meeting_strategy` (reads user-side context as primary strategy driver)
- `brief_composer` → `final_brief`, `brief_reliability`

Do not let nodes overwrite fields they do not own.

## Person research rules
- Only attempt person matching if `contact_name` is provided.
- Prefer reliable public professional sources:
  - company team pages
  - speaker bios
  - press releases
  - interviews/articles
  - public professional profiles if accessible
- Do not fabricate identities.
- If confidence is low, keep output sparse and explicitly say no reliable profile was found.

## Audience context rules
Do not rely on executive titles only.
Map roles to broader functions such as:
- finance
- accounting
- engineering
- product
- sales
- strategy
- it_data
- operations
- transformation

Then reason using:
- function
- seniority
- company context
- person profile if reliable
- meeting goal

## Final brief requirements
The final brief must include:
- Executive Summary
- Company Snapshot
- Recent Strategic Signals
- Meeting Contact
- Audience Context
- Engagement Context
- Likely Pain Points
- Suggested Meeting Questions
- Items to Validate During Meeting
- Meeting Prep Notes
- Sources

The composer is **deterministic** and must not call an LLM.
`meeting_strategy["executive_summary"]` must be inserted into the brief.

## Meeting intake agent rules
- The intake agent is **NOT** a LangGraph node — it is a standalone function called before `app.invoke()`
- The main workflow must always work with direct structured input (no intake required)
- Intake metadata fields (`raw_meeting_context`, `intake_missing_fields`, etc.) are passed through by the orchestrator but do not affect workflow logic
- Enum fields extracted by intake must be sanitised against `VALID_MEETING_TYPES`, `VALID_MEETING_STAGES`, `VALID_RELATIONSHIP_STATUSES`, `VALID_PREVIOUS_OUTCOMES`
- Missing-field detection distinguishes **important** fields (company_name, domain, meeting_type, stakeholder_role, meeting_goal) from **recommended** fields (meeting_stage, relationship_status)
- Low-confidence fields include the field name, extracted value, and reason for uncertainty
- On LLM failure, the intake agent returns a valid but empty structure with all important fields marked as missing
- **Low-confidence domain suppression:** if `domain` is flagged low-confidence (e.g. guessed from company name), it is nulled out and reported as missing — a wrong domain would poison research queries
- **Low-confidence engagement suppression:** if `meeting_stage`, `relationship_status`, or `previous_meeting_outcome` are flagged low-confidence, they are nulled out so the `engagement_inference_agent` can handle them with richer context instead of treating a shaky intake guess as explicit user input

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

## Working style
- Build incrementally
- Keep modules small and testable
- Use Pydantic or explicit structured schemas for LLM outputs
- Make assumptions explicit
- Prefer simple deterministic logic over cleverness
- When something is ambiguous, flag it clearly before overengineering
