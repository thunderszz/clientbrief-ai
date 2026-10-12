// The visual vocabulary of certainty, explained where it is first seen.

export function MarksLegend({ compact = false }: { compact?: boolean }) {
  const items = [
    { mark: "mark-fact", word: "Highlighted", text: "a fact found in a cited source" },
    { mark: "mark-inference", word: "Underlined", text: "an inference drawn from those facts" },
    { mark: "mark-hypothesis", word: "Italic grey", text: "a hypothesis to test in the meeting" },
  ];
  return (
    <ul className={compact ? "space-y-1.5 text-[14px]" : "grid gap-4 text-[15px] md:grid-cols-3"}>
      {items.map((item) => (
        <li key={item.mark} className="text-ink-soft">
          <span className={`font-serif text-ink ${item.mark}`}>{item.word}</span> text is {item.text}.
        </li>
      ))}
    </ul>
  );
}
