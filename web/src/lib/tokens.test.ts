// Guards the palette against falling below WCAG AA contrast for small text.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const css = readFileSync(new URL("../styles.css", import.meta.url), "utf8");

function block(selector: string): string {
  const start = css.indexOf(selector);
  return css.slice(start, css.indexOf("}", start));
}

function token(scope: string, name: string): string {
  const match = new RegExp(`--${name}:\\s*(#[0-9a-f]{6})`, "i").exec(scope);
  if (!match) throw new Error(`token --${name} not found`);
  return match[1];
}

function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

describe("palette contrast", () => {
  for (const [name, selector] of [
    ["light", ":root {"],
    ["dark", ':root[data-theme="dark"] {'],
  ] as const) {
    it(`${name}: secondary text colours reach 4.5:1 on paper and surface`, () => {
      const scope = block(selector);
      for (const fg of ["ink", "ink-soft", "pencil", "ballpoint", "alert"]) {
        for (const bg of ["paper", "surface"]) {
          expect(contrast(token(scope, fg), token(scope, bg)), `${fg} on ${bg}`).toBeGreaterThanOrEqual(4.5);
        }
      }
    });
  }
});
