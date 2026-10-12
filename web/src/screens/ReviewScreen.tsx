import { ArrowLeft } from "lucide-react";
import { useState } from "react";
import { Button, Notice } from "../components/ui";
import { labelFor } from "../lib/format";
import type { IntakeResponse, MeetingFields } from "../lib/types";

type Key = keyof MeetingFields;

interface FieldDef {
  key: Key;
  label: string;
  required?: boolean;
  hint?: string;
  options?: string[];
  emptyOption?: string;
  multiline?: boolean;
  wide?: boolean;
}

const MEETING: FieldDef[] = [
  { key: "company_name", label: "Company", required: true },
  { key: "domain", label: "Company website", required: true, hint: "For example satispay.com" },
  {
    key: "meeting_type",
    label: "Meeting type",
    required: true,
    options: ["consulting_intro", "sales_discovery", "account_review"],
    emptyOption: "Choose one",
  },
  { key: "stakeholder_role", label: "Role of the person you're meeting", required: true },
  { key: "contact_name", label: "Their name", hint: "Optional. Used to look up their public profile." },
  { key: "contact_title", label: "Their job title" },
  {
    key: "meeting_stage",
    label: "Stage",
    options: [
      "first_intro",
      "discovery_followup",
      "solution_discussion",
      "proposal_review",
      "negotiation",
      "client_kickoff",
      "account_expansion",
    ],
    emptyOption: "Let the agents infer it",
  },
  {
    key: "relationship_status",
    label: "Relationship",
    options: ["prospect", "active_opportunity", "client", "former_client"],
    emptyOption: "Let the agents infer it",
  },
  {
    key: "previous_meeting_outcome",
    label: "How the last meeting went",
    options: ["positive", "neutral", "unclear", "no_previous_meeting"],
    emptyOption: "Let the agents infer it",
  },
  { key: "meeting_goal", label: "What you want from this meeting", required: true, multiline: true, wide: true },
  { key: "meeting_notes_summary", label: "Notes from earlier conversations", multiline: true, wide: true },
];

const YOU: FieldDef[] = [
  { key: "user_role", label: "Your role" },
  { key: "user_company", label: "Your company" },
  { key: "user_function", label: "Your field" },
  {
    key: "engagement_type",
    label: "Kind of engagement",
    options: ["sales", "consulting", "partnership", "investing", "recruiting", "interview"],
    emptyOption: "Not specified",
  },
  { key: "desired_outcome", label: "Best outcome for you", multiline: true, wide: true },
  { key: "success_definition", label: "How you'll judge success", multiline: true, wide: true },
];

const REQUIRED = [...MEETING, ...YOU].filter((f) => f.required).map((f) => f.key);

const control =
  "mt-1 block w-full rounded-[3px] border bg-surface px-3 py-2 text-[15px] text-ink placeholder:text-pencil focus:border-ballpoint focus:outline-none";

interface Props {
  intake: IntakeResponse;
  busy: boolean;
  error: string | null;
  onBack: () => void;
  onSubmit: (fields: MeetingFields) => void;
}

export function ReviewScreen({ intake, busy, error, onBack, onSubmit }: Props) {
  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries(Object.entries(intake.fields).map(([k, v]) => [k, v ?? ""])),
  );
  const [attempted, setAttempted] = useState(false);
  const unsure = new Map(intake.low_confidence.map((lc) => [lc.field, lc]));
  const missing = REQUIRED.filter((key) => !values[key]?.trim());

  function renderField(def: FieldDef) {
    const value = values[def.key] ?? "";
    const invalid = def.required && !value.trim() && (attempted || intake.fields[def.key] === null || intake.fields[def.key] === "");
    const note = unsure.get(def.key);
    const id = `field-${def.key}`;
    const border = invalid ? "border-alert" : "border-rule";
    const common = {
      id,
      value,
      "aria-invalid": invalid || undefined,
      "aria-describedby": `${id}-help`,
      onChange: (e: { target: { value: string } }) => setValues((v) => ({ ...v, [def.key]: e.target.value })),
    };
    return (
      <div key={def.key} className={def.wide ? "md:col-span-2" : ""}>
        <label htmlFor={id} className="text-[14px] font-medium text-ink">
          {def.label}
          {def.required && <span className="text-alert"> *</span>}
        </label>
        {def.options ? (
          <select {...common} className={`${control} ${border}`}>
            <option value="">{def.emptyOption}</option>
            {def.options.map((o) => (
              <option key={o} value={o}>
                {labelFor(o)}
              </option>
            ))}
          </select>
        ) : def.multiline ? (
          <textarea {...common} rows={3} className={`${control} ${border} resize-y`} />
        ) : (
          <input {...common} className={`${control} ${border}`} />
        )}
        <p id={`${id}-help`} className="mt-1 text-[13px]">
          {invalid && <span className="block text-alert">Needed before the research can start.</span>}
          {note && (
            <span className="block italic text-ink-soft">
              Not sure about this one{note.value ? ` (read as "${note.value}")` : ""}: {note.reason}
            </span>
          )}
          {!invalid && !note && def.hint && <span className="text-pencil">{def.hint}</span>}
        </p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-10 md:px-8">
      <button type="button" onClick={onBack} className="flex items-center gap-1.5 text-[15px] text-ink-soft hover:text-ink">
        <ArrowLeft size={16} aria-hidden="true" /> Edit the description
      </button>
      <h1 className="mt-6 text-[2rem] font-semibold tracking-[-0.02em] text-ink">Check the details</h1>
      {intake.summary && (
        <blockquote className="mt-4 border-l-2 border-ballpoint pl-4 font-serif text-[17px] leading-relaxed text-ink-soft">
          {intake.summary}
        </blockquote>
      )}
      {!intake.ok && (
        <div className="mt-4">
          <Notice tone="error">The details couldn't be read automatically. Fill in the fields below.</Notice>
        </div>
      )}

      <form
        className="mt-8"
        onSubmit={(e) => {
          e.preventDefault();
          setAttempted(true);
          if (missing.length || busy) return;
          const fields = Object.fromEntries(
            Object.entries(values).map(([k, v]) => [k, v.trim() || null]),
          ) as unknown as MeetingFields;
          onSubmit(fields);
        }}
      >
        <fieldset>
          <legend className="text-[19px] font-semibold text-ink">The meeting</legend>
          <div className="mt-4 grid gap-x-6 gap-y-5 md:grid-cols-2">{MEETING.map(renderField)}</div>
        </fieldset>
        <fieldset className="mt-10">
          <legend className="text-[19px] font-semibold text-ink">You</legend>
          <p className="mt-1 text-[14px] text-ink-soft">Optional. Helps the strategy fit your position.</p>
          <div className="mt-4 grid gap-x-6 gap-y-5 md:grid-cols-2">{YOU.map(renderField)}</div>
        </fieldset>

        {error && (
          <div className="mt-8">
            <Notice tone="error">{error}</Notice>
          </div>
        )}
        <div className="mt-8 flex flex-wrap items-center gap-4 border-t border-rule pt-6">
          <Button type="submit" disabled={busy}>
            Build the brief
          </Button>
          {attempted && missing.length > 0 && (
            <p className="text-[14px] text-alert">Fill in the fields marked with * first.</p>
          )}
        </div>
      </form>
    </div>
  );
}
