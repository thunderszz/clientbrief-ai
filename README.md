Pre-meeting intelligence powered by multi-agent AI.

ClientBrief AI takes a plain-language description of an upcoming business meeting and produces a structured, research-backed briefing — in minutes.

What it does
You describe your meeting in natural language. The system:

Parses your input into structured fields using an LLM intake agent
Researches the company, recent strategic signals, and the meeting contact via live web search
Reasons about the audience, likely pain points, and optimal meeting strategy
Composes a full markdown brief ready to read before you walk in the door
The final brief includes: company snapshot, recent signals, audience context, engagement context, pain point hypotheses, suggested questions, and a meeting strategy tailored to your role and goals.

Stack
Google Gemini · Tavily Search · LangGraph · Streamlit · Python

Architecture
Built as an 11-node LangGraph pipeline with a pre-workflow LLM intake layer. Design principles: deterministic logic for validation and composition, LLMs only where reasoning is needed, facts and hypotheses always labelled separately.

API keys needed
GEMINI_API_KEY — Google AI Studio (free tier available)
TAVILY_API_KEY — app.tavily.com (free tier available)
