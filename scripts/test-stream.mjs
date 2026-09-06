import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../app/history_agent/web/static/stream.js", import.meta.url), "utf8");
const {consumeEventStream} = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

test("SSE handles split UTF-8, CRLF, comments, and multiple records", async () => {
  const bytes = new TextEncoder().encode(': ping\r\nevent: delta\r\ndata: {"text":"毛泽东\\n周恩来"}\r\n\r\nevent: done\ndata: {"answer":"完成"}\n\n');
  const stream = new ReadableStream({
    start(controller) {
      for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
      controller.close();
    }
  });
  const events = [];
  await consumeEventStream(new Response(stream), (event, data) => events.push([event, data]));
  assert.deepEqual(events, [["delta", {text: "毛泽东\n周恩来"}], ["done", {answer: "完成"}]]);
});

test("deltas arrive while the stream remains open; done cancels the reader", async () => {
  let controller;
  let cancelled = false;
  const stream = new ReadableStream({start(c) { controller = c; }, cancel() { cancelled = true; }});
  let first;
  const received = new Promise(resolve => { first = resolve; });
  const reading = consumeEventStream(new Response(stream), event => {
    if (event === "delta") first();
    return event !== "done";
  });
  controller.enqueue(new TextEncoder().encode('event: delta\ndata: {"text":"先到"}\n\n'));
  await received;
  assert.equal(cancelled, false);
  controller.enqueue(new TextEncoder().encode('event: done\ndata: {}\n\n'));
  await reading;
  assert.equal(cancelled, true);
});

test("malformed JSON rejects and releases the stream", async () => {
  let cancelled = false;
  const stream = new ReadableStream({
    start(c) { c.enqueue(new TextEncoder().encode("event: delta\ndata: broken\n\n")); },
    cancel() { cancelled = true; }
  });
  await assert.rejects(consumeEventStream(new Response(stream), () => {}));
  assert.equal(cancelled, true);
});
