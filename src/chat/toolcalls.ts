/** Tool calls as Ollama sends them: fragments that must be merged before Pi may run them. */

import { parseToolCallArguments } from '@earendil-works/pi-ai';
import type { OllamaToolCall } from '../wire.ts';

/** One tool call as it accumulates across chunks; see {@link mergeToolCall}. */
export interface BufferedToolCall {
  name: string;
  arguments: Record<string, unknown>;
  /** The server's own id for this call, when it sent one. */
  id?: string;
  /** Set when an arriving fragment broke a rule; a later chunk cannot repair the call. */
  malformed: boolean;
}

export function emptyToolCall (): BufferedToolCall {
  return { name: '', arguments: {}, malformed: false };
}

/** The name a fragment carries, or `undefined` when it continues the call before it. */
export function fragmentName (fragment: OllamaToolCall): string | undefined {
  const parts = fragment as { function?: { name?: unknown } };
  const name = parts.function?.name;
  return typeof name === 'string' && name.trim().length > 0 ? name : undefined;
}

/**
 * Reads one fragment's `arguments` as an object. Ollama sends an object, but a partial call or an
 * intermediary can send the JSON text instead, which is exactly the case Pi's own
 * `parseToolCallArguments` handles: anything that is not a JSON object is rejected, and its repair
 * step also rescues the control characters a model sometimes leaves inside a string.
 */
function argumentsOf (value: unknown): { ok: true; value: Record<string, unknown> } | { ok: false } {
  if (value === undefined || value === null) return { ok: true, value: {} };
  if (typeof value === 'object' && !Array.isArray(value)) {
    return { ok: true, value: value as Record<string, unknown> };
  }
  if (typeof value !== 'string') return { ok: false };
  const parsed = parseToolCallArguments(value);
  return parsed.argumentsError === undefined
    ? { ok: true, value: parsed.arguments as Record<string, unknown> }
    : { ok: false };
}

/**
 * Merges one arriving fragment into the call buffered for it. A non-empty name wins over an empty
 * one and arguments merge shallowly, because Ollama splits one call across chunks. A fragment that
 * breaks a rule poisons the whole call: dropping it is safer than handing `validateToolArguments`
 * a call that throws after the turn's text already reached the user.
 */
export function mergeToolCall (into: BufferedToolCall, fragment: OllamaToolCall): BufferedToolCall {
  const name = fragmentName(fragment);
  if (name !== undefined) into.name = name;
  if (into.id === undefined && typeof fragment.id === 'string' && fragment.id.length > 0) {
    into.id = fragment.id;
  }
  const parts = fragment as { function?: { arguments?: unknown } };
  const incoming = argumentsOf(parts.function?.arguments);
  if (incoming.ok) into.arguments = { ...into.arguments, ...incoming.value };
  else into.malformed = true;
  return into;
}
