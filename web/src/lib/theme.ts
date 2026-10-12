// Light/dark preference: follows the OS unless the viewer picks one.

export type Theme = "light" | "dark";
const KEY = "clientbrief.theme";

export function storedTheme(): Theme | null {
  try {
    const value = window.localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

export function systemTheme(): Theme {
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  } catch {
    return "light";
  }
}

export function applyTheme(theme: Theme, persist: boolean): void {
  document.documentElement.dataset.theme = theme;
  if (!persist) return;
  try {
    window.localStorage.setItem(KEY, theme);
  } catch {
    // Storage blocked: the choice lasts for this page view only.
  }
}
