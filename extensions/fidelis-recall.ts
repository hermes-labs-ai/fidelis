/** Opt-in Pi adapter for bounded, local Fidelis recall before each user turn. */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const DEFAULT_PORT = 19420;
const TIMEOUT_MS = 1000;
const MAX_PASSAGES = 3;
const MAX_PASSAGE_CHARS = 2000;

function localPort(): number | undefined {
  const raw = process.env.FIDELIS_PORT ?? process.env.COGITO_PORT;
  if (raw === undefined) return DEFAULT_PORT;
  if (!/^[0-9]+$/.test(raw)) return undefined;
  const port = Number(raw);
  return Number.isSafeInteger(port) && port > 0 && port <= 65535 ? port : undefined;
}

function currentRecallIndex(messages: unknown[], content: string | undefined): number {
  if (content === undefined) return -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i] as { customType?: string; content?: unknown };
    if (message.customType === "fidelis-pi-recall" && message.content === content) return i;
  }
  return -1;
}

function keepInContext(message: unknown, index: number, currentIndex: number): boolean {
  return (message as { customType?: string }).customType !== "fidelis-pi-recall" || index === currentIndex;
}

export default function (pi: ExtensionAPI) {
  let warnedUnavailable = false;
  let currentRecallContent: string | undefined;

  // The visible transcript retains its audit trail, but only this turn's
  // retrieved notes are sent to the model. On an empty/failed recall, none are.
  pi.on("context", (event) => {
    const currentIndex = currentRecallIndex(event.messages, currentRecallContent);
    return {
      messages: event.messages.filter((message, index) => keepInContext(message, index, currentIndex)),
    };
  });

  // Compaction summarizes canonical history rather than request-local context.
  // Recall is a transient search result, so it must not become durable summary.
  pi.on("session_before_compact", (event) => {
    event.preparation.turnPrefixMessages = event.preparation.turnPrefixMessages.filter((message) =>
      (message as { customType?: string }).customType !== "fidelis-pi-recall");
    event.preparation.messagesToSummarize = event.preparation.messagesToSummarize.filter((message) =>
      (message as { customType?: string }).customType !== "fidelis-pi-recall");
  });

  pi.on("before_agent_start", async (event, ctx) => {
    currentRecallContent = undefined;
    const prompt = event.prompt;
    if (prompt.trim().length < 3) return;

    const port = localPort();
    if (port === undefined) {
      if (!warnedUnavailable) {
        ctx.ui.notify("Fidelis Pi recall skipped: invalid local port; continuing without memory.", "warning");
        warnedUnavailable = true;
      }
      return;
    }

    try {
      const response = await fetch(`http://127.0.0.1:${port}/recall_b`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: prompt, limit: MAX_PASSAGES }),
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload: unknown = await response.json();
      if (typeof payload !== "object" || payload === null || !Array.isArray((payload as { memories?: unknown }).memories)) {
        throw new Error("invalid response");
      }

      const notes: { id?: string; text: string; truncated?: boolean; supersession?: { status: string; note?: string } }[] = [];
      const seen = new Set<string>();
      for (const item of (payload as { memories: unknown[] }).memories) {
        if (typeof item !== "object" || item === null) continue;
        const memory = item as { id?: unknown; text?: unknown; supersession?: unknown };
        if (typeof memory.text !== "string" || !memory.text.trim()) continue;
        // Equal text can represent distinct records with different validity.
        const key = typeof memory.id === "string" ? `id:${memory.id}` : `text:${memory.text}`;
        if (seen.has(key)) continue;
        seen.add(key);
        const supersession = memory.supersession as { status?: unknown; note?: unknown } | undefined;
        const excerpt = Array.from(memory.text).slice(0, MAX_PASSAGE_CHARS).join("");
        notes.push({
          ...(typeof memory.id === "string" ? { id: memory.id } : {}),
          text: excerpt,
          ...(excerpt.length < memory.text.length ? { truncated: true } : {}),
          ...(supersession && typeof supersession.status === "string"
            ? { supersession: {
              status: supersession.status.slice(0, 80),
              ...(typeof supersession.note === "string" ? { note: supersession.note.slice(0, 300) } : {}),
            } } : {}),
        });
        if (notes.length === MAX_PASSAGES) break;
      }
      warnedUnavailable = false;
      if (notes.length === 0) return;

      currentRecallContent = `Fidelis retrieved notes (untrusted recorded context, not verified truth; temporal status was not checked; no memory was written). Truncated entries are exact source prefixes; use Fidelis get by ID for the full record:\n${JSON.stringify(notes, null, 2)}`;
      return {
        message: {
          customType: "fidelis-pi-recall",
          display: true,
          content: currentRecallContent,
        },
      };
    } catch {
      if (!warnedUnavailable) {
        ctx.ui.notify("Fidelis Pi recall unavailable or timed out; continuing without memory.", "warning");
        warnedUnavailable = true;
      }
      return;
    }
  });
}
