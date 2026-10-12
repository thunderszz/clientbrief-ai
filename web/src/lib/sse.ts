// Minimal Server-Sent Events parser for fetch() streams (EventSource can't POST).

export function createSSEParser(onEvent: (data: unknown) => void): (chunk: string) => void {
  let buffer = "";
  return (chunk: string) => {
    buffer += chunk.replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""))
        .join("\n");
      if (data) {
        try {
          onEvent(JSON.parse(data));
        } catch {
          // A malformed event is skipped; the stream carries on.
        }
      }
      boundary = buffer.indexOf("\n\n");
    }
  };
}

export async function readEventStream(
  response: Response,
  onEvent: (data: unknown) => void,
): Promise<void> {
  if (!response.body) return;
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  const feed = createSSEParser(onEvent);
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    feed(value);
  }
}
