import { KeyRound, Moon, Sun } from "lucide-react";
import type { Theme } from "../lib/theme";

interface Props {
  theme: Theme;
  hasKeys: boolean;
  onHome: () => void;
  onToggleTheme: () => void;
  onOpenKeys: () => void;
}

function Logo() {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" className="h-6 w-6">
      <rect x="5" y="3" width="22" height="26" rx="2" fill="currentColor" />
      <rect x="9" y="9" width="14" height="4" fill="var(--highlight)" />
      <rect x="9" y="16" width="14" height="2" fill="var(--paper)" />
      <rect x="9" y="21" width="9" height="2" fill="var(--pencil)" />
    </svg>
  );
}

export function Header({ theme, hasKeys, onHome, onToggleTheme, onOpenKeys }: Props) {
  return (
    <header className="no-print border-b border-rule">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 md:px-8">
        <button type="button" onClick={onHome} className="flex items-center gap-2 text-ink">
          <Logo />
          <span className="text-[17px] font-semibold tracking-[-0.01em]">ClientBrief</span>
        </button>
        <nav className="flex items-center gap-1 text-[15px]">
          <button
            type="button"
            onClick={onOpenKeys}
            className="flex items-center gap-2 rounded-[3px] px-3 py-2 text-ink-soft hover:text-ink"
          >
            <KeyRound size={16} aria-hidden="true" />
            <span>{hasKeys ? "Your keys" : "Add keys"}</span>
            {hasKeys && <span className="h-1.5 w-1.5 rounded-full bg-ok" aria-label="keys saved" />}
          </button>
          <button
            type="button"
            onClick={onToggleTheme}
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            className="rounded-[3px] p-2 text-ink-soft hover:text-ink"
          >
            {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
          </button>
          <a
            href="https://github.com/thunderszz/clientbrief-ai"
            className="hidden rounded-[3px] px-3 py-2 text-ink-soft hover:text-ink sm:inline"
          >
            Source code
          </a>
        </nav>
      </div>
    </header>
  );
}
