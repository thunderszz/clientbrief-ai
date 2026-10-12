import type { ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "secondary" | "quiet";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-ink text-paper hover:bg-ink-soft disabled:bg-pencil disabled:cursor-not-allowed px-4 py-2.5 rounded-[3px] font-medium",
  secondary:
    "border border-rule text-ink hover:border-ink disabled:opacity-50 px-4 py-2.5 rounded-[3px] font-medium",
  quiet: "text-ballpoint hover:underline underline-offset-4 disabled:opacity-50",
};

export function Button({
  variant = "primary",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      type="button"
      className={`inline-flex items-center gap-2 text-[15px] transition-colors ${VARIANTS[variant]} ${className}`}
      {...props}
    />
  );
}

export function Notice({ tone = "info", children }: { tone?: "info" | "error"; children: ReactNode }) {
  const color = tone === "error" ? "border-alert text-ink" : "border-ballpoint text-ink";
  return (
    <div role={tone === "error" ? "alert" : "status"} className={`border-l-2 ${color} bg-surface px-4 py-3 text-[15px]`}>
      {children}
    </div>
  );
}

export function ModelChip({ model, fallback }: { model: string; fallback?: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1 whitespace-nowrap rounded-[3px] border px-1.5 py-px text-[12px] ${
        fallback ? "border-alert/60 text-alert" : "border-rule text-ink-soft"
      }`}
      title={fallback ? "The first-choice model was busy, so this one answered instead." : "Model that answered"}
    >
      {model}
      {fallback && <span>fallback</span>}
    </span>
  );
}
