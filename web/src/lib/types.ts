// Shapes mirrored from the Python API (api/runner.py, core/result.py, api/main.py).

export type ProviderId = "gemini" | "groq" | "deepseek" | "openrouter";

export interface PipelineNode {
  id: string;
  label: string;
  kind: "deterministic" | "llm" | "search+llm";
  phase: "setup" | "research" | "reasoning" | "output";
  description: string;
}

export interface ProviderPreset {
  id: ProviderId;
  label: string;
  model_fast: string | null;
  model_reasoning: string | null;
  key_url: string;
  free_tier: boolean;
  needs_model: boolean;
}

export interface AppConfig {
  pipeline: { nodes: PipelineNode[]; edges: [string, string][] };
  provider: string | null;
  models: { fast: string; reasoning: string } | null;
  server_runs: { available: boolean; remaining: number | null };
  providers: ProviderPreset[];
}

export interface DemoSummary {
  slug: string;
  title: string;
  subtitle: string;
  company: string;
  recorded_at: string;
  models: { fast: string; reasoning: string };
  duration_s: number;
}

export interface MeetingFields {
  company_name: string;
  domain: string;
  meeting_type: string;
  stakeholder_role: string;
  meeting_goal: string;
  contact_name?: string | null;
  contact_title?: string | null;
  meeting_stage?: string | null;
  relationship_status?: string | null;
  previous_meeting_outcome?: string | null;
  meeting_notes_summary?: string | null;
  user_role?: string | null;
  user_company?: string | null;
  user_function?: string | null;
  engagement_type?: string | null;
  desired_outcome?: string | null;
  success_definition?: string | null;
}

export interface LowConfidenceField {
  field: string;
  value: string;
  reason: string;
}

export interface IntakeResponse {
  ok: boolean;
  fields: Record<keyof MeetingFields, string | null>;
  missing: string[];
  low_confidence: LowConfidenceField[];
  summary: string;
}

export interface LlmCall {
  model: string;
  tier: "fast" | "reasoning";
  ok: boolean;
  latency_s: number;
  fallback: boolean;
  attempts: number;
  node: string;
}

export interface ReliabilityComponent {
  key: string;
  label: string;
  points: number;
  max: number;
  detail: string;
  applicable: boolean;
}

export interface ReliabilityReport {
  score: number;
  rating: number;
  components: ReliabilityComponent[];
}

export interface Signal {
  signal_type: string;
  description: string;
  implication: string;
  confidence: string;
  source: string;
  date: string | null;
}

export interface Hypothesis {
  hypothesis: string;
  supporting_evidence: string[];
  confidence: string;
}

export interface BriefResult {
  input: MeetingFields;
  engagement_inference: {
    inferred_meeting_stage?: string | null;
    inferred_relationship_status?: string | null;
    inferred_previous_meeting_outcome?: string | null;
    confidence?: string;
    reasoning_summary?: string;
  };
  engagement_context: {
    interaction_stage?: string;
    relationship_status?: string;
    conversation_mode?: string;
    stage_objective?: string;
    recommended_tone?: string;
    decision_risks?: string[];
    next_step_pressure?: string;
    continuity_context?: string;
    research_mode?: string;
  };
  research_plan: Record<string, unknown>;
  company_profile: {
    industry?: string | null;
    business_model?: string | null;
    products?: string[];
    customer_segments?: string[];
    geographies?: string[];
    company_positioning?: string | null;
    complexity_signals?: string[];
    sources?: string[];
  };
  external_signals: Signal[];
  person_profile: {
    matched_person?: boolean;
    match_confidence?: string;
    name?: string | null;
    title?: string | null;
    department?: string | null;
    tenure_hint?: string | null;
    background_summary?: string | null;
    public_signals?: string[];
    sources?: string[];
  };
  audience_context: {
    department?: string;
    role_seniority?: string | null;
    likely_priorities?: string[];
    relevant_topics?: string[];
    communication_style?: string;
    potential_sensitivities?: string[];
  };
  pain_point_hypotheses: { facts?: string[]; inferences?: string[]; hypotheses?: Hypothesis[] };
  meeting_strategy: {
    executive_summary?: string;
    opening_angle?: string;
    talking_points?: string[];
    recommended_questions?: string[];
    possible_objections?: string[];
    suggested_next_step?: string;
  };
  sources: string[];
  errors: string[];
  llm_calls: LlmCall[];
  reliability: ReliabilityReport;
  final_brief: string;
  generated_at: string;
}

export type PipelineEvent =
  | { type: "run_started"; t: number }
  | { type: "node_started"; t: number; node: string; label: string }
  | {
      type: "node_finished";
      t: number;
      node: string;
      label: string;
      duration: number;
      summary: string;
      llm_calls: LlmCall[];
      warnings: string[];
    }
  | { type: "node_failed"; t: number; node: string; label: string }
  | { type: "result"; t: number; result: BriefResult }
  | { type: "error"; t: number; message: string };
