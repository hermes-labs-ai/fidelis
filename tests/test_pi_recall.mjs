import assert from "node:assert/strict";
import { createServer } from "node:http";
import test from "node:test";
import register from "../extensions/fidelis-recall.ts";

function hook() {
  const handlers = new Map();
  const notices = [];
  register({ on(name, callback) {
    handlers.set(name, callback);
  } });
  return {
    notices,
    call(prompt) {
      return handlers.get("before_agent_start")({ prompt }, { ui: { notify: (...args) => notices.push(args) } });
    },
    context(messages) {
      return handlers.get("context")({ messages }).messages;
    },
    compact(turnPrefixMessages, messagesToSummarize) {
      const preparation = { turnPrefixMessages, messagesToSummarize };
      handlers.get("session_before_compact")({ preparation });
      return preparation;
    },
  };
}

async function server(t, respond) {
  const requests = [];
  const http = createServer(async (req, res) => {
    let body = "";
    for await (const chunk of req) body += chunk;
    requests.push({ method: req.method, path: req.url, body: JSON.parse(body) });
    respond(req, res);
  });
  await new Promise((resolve) => http.listen(0, "127.0.0.1", resolve));
  const prior = process.env.FIDELIS_PORT;
  process.env.FIDELIS_PORT = String(http.address().port);
  t.after(async () => {
    if (prior === undefined) delete process.env.FIDELIS_PORT;
    else process.env.FIDELIS_PORT = prior;
    await new Promise((resolve) => http.close(resolve));
  });
  return requests;
}

test("retrieves at most three verbatim local notes before a Pi turn", async (t) => {
  const text = "Retry Atlas only after the billing ledger is reconciled.";
  const requests = await server(t, (_req, res) => {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ memories: [
      { id: "a", text }, { id: "a", text },
      { id: "b", text: "Second exact note", supersession: { status: "superseded", note: "Use the later decision" } },
      { id: "c", text: "Third exact note" },
      { id: "d", text: "Fourth should be omitted" },
    ] }));
  });
  const pi = hook();
  const prompt = "What must happen before Atlas retry?";
  const result = await pi.call(prompt);
  assert.deepEqual(requests, [{
    method: "POST", path: "/recall_b", body: { text: prompt, limit: 3 },
  }]);
  assert.equal(result.message.customType, "fidelis-pi-recall");
  assert.equal(result.message.display, true);
  assert.equal(result.message.content.match(/Retry Atlas only after the billing ledger is reconciled\./g).length, 1);
  assert.match(result.message.content, /Second exact note/);
  assert.match(result.message.content, /superseded/);
  assert.match(result.message.content, /Third exact note/);
  assert.doesNotMatch(result.message.content, /Fourth should be omitted/);
  assert.deepEqual(pi.notices, []);
});

test("two non-whitespace Unicode characters do not trigger recall", async (t) => {
  const requests = await server(t, (_req, res) => {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ memories: [{ text: "Exact note" }] }));
  });
  const pi = hook();
  assert.equal(await pi.call("😀a"), undefined);
  assert.equal(await pi.call("😀 a"), undefined);
  assert.deepEqual(requests, []);
  assert.deepEqual(pi.notices, []);
  assert.equal((await pi.call("😀ab")).message.customType, "fidelis-pi-recall");
  assert.equal(requests.length, 1);
});

test("equal text with different record IDs remains distinct", async (t) => {
  await server(t, (_req, res) => {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ memories: [
      { id: "old", text: "Same decision text" },
      { id: "new", text: "Same decision text" },
    ] }));
  });
  const pi = hook();
  const result = await pi.call("Decision query");
  const notes = JSON.parse(result.message.content.split("\n").slice(1).join("\n"));
  assert.deepEqual(notes.map((note) => note.id), ["old", "new"]);
});

test("older recall messages leave model context on a new or failed turn", async (t) => {
  let fail = false;
  await server(t, (_req, res) => {
    res.setHeader("Content-Type", "application/json");
    res.end(fail ? JSON.stringify({ memories: [] }) : JSON.stringify({ memories: [{ text: "Current exact note" }] }));
  });
  const pi = hook();
  const previous = { role: "custom", customType: "fidelis-pi-recall", content: "Old note" };
  const current = (await pi.call("Current query")).message;
  const user = { role: "user", content: "Current query" };
  assert.deepEqual(pi.context([previous, current, user]), [current, user]);
  assert.deepEqual(pi.compact([previous], [current, user]), {
    turnPrefixMessages: [], messagesToSummarize: [user],
  });
  fail = true;
  assert.equal(await pi.call("Next query"), undefined);
  assert.deepEqual(pi.context([previous, current, user]), [user]);
  assert.deepEqual(pi.compact([previous], [current, user]), {
    turnPrefixMessages: [], messagesToSummarize: [user],
  });
});

test("long source notes remain as marked exact excerpts with IDs", async (t) => {
  const longText = "A".repeat(2100);
  await server(t, (_req, res) => {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ memories: [{ id: "long-id", text: longText }] }));
  });
  const pi = hook();
  const result = await pi.call("Long note query");
  const notes = JSON.parse(result.message.content.split("\n").slice(1).join("\n"));
  assert.deepEqual(notes, [{ id: "long-id", text: "A".repeat(2000), truncated: true }]);
});

test("an empty or failed recall passes through without model context", async (t) => {
  const requests = await server(t, (_req, res) => {
    res.writeHead(503);
    res.end("unavailable");
  });
  const pi = hook();
  assert.equal(await pi.call("hello"), undefined);
  assert.equal(await pi.call("hello again"), undefined);
  assert.equal(requests.length, 2);
  assert.equal(pi.notices.length, 1);
  assert.match(pi.notices[0][0], /continuing without memory/);
});

test("a stalled local service times out and leaves the turn usable", async (t) => {
  await server(t, () => {});
  const pi = hook();
  const start = performance.now();
  assert.equal(await pi.call("hello"), undefined);
  assert.ok(performance.now() - start < 1800);
  assert.match(pi.notices[0][0], /timed out/);
});

test("an oversized local response is rejected before parsing and leaves the turn usable", async (t) => {
  await server(t, (_req, res) => {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ memories: [{ id: "oversize", text: "A".repeat(70 * 1024) }] }));
  });
  const pi = hook();
  assert.equal(await pi.call("oversize query"), undefined);
  assert.equal(pi.notices.length, 1);
  assert.match(pi.notices[0][0], /continuing without memory/);
});

test("invalid port skips network and reports it", async (t) => {
  const prior = process.env.FIDELIS_PORT;
  process.env.FIDELIS_PORT = "https://example.com";
  t.after(() => {
    if (prior === undefined) delete process.env.FIDELIS_PORT;
    else process.env.FIDELIS_PORT = prior;
  });
  const pi = hook();
  assert.equal(await pi.call("hello"), undefined);
  assert.match(pi.notices[0][0], /invalid local port/);
});
