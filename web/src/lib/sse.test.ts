import { describe, expect, it } from "vitest";
import { createSSEParser } from "./sse";

describe("createSSEParser", () => {
  it("parses complete events", () => {
    const seen: unknown[] = [];
    const feed = createSSEParser((e) => seen.push(e));

    feed('data: {"type":"run_started","t":0}\n\ndata: {"type":"result","t":1}\n\n');

    expect(seen).toEqual([
      { type: "run_started", t: 0 },
      { type: "result", t: 1 },
    ]);
  });

  it("buffers events split across chunks", () => {
    const seen: unknown[] = [];
    const feed = createSSEParser((e) => seen.push(e));

    feed('data: {"type":"no');
    expect(seen).toEqual([]);
    feed('de_started","t":0.5}\n');
    expect(seen).toEqual([]);
    feed("\n");

    expect(seen).toEqual([{ type: "node_started", t: 0.5 }]);
  });

  it("ignores heartbeat comments and handles CRLF", () => {
    const seen: unknown[] = [];
    const feed = createSSEParser((e) => seen.push(e));

    feed(': ping\r\n\r\ndata: {"type":"x","t":2}\r\n\r\n');

    expect(seen).toEqual([{ type: "x", t: 2 }]);
  });

  it("skips malformed JSON instead of throwing", () => {
    const seen: unknown[] = [];
    const feed = createSSEParser((e) => seen.push(e));

    feed("data: {nope\n\ndata: {\"type\":\"ok\",\"t\":0}\n\n");

    expect(seen).toEqual([{ type: "ok", t: 0 }]);
  });
});
