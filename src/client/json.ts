/** One JSON endpoint exchange, with a deadline that covers send, headers and body. */

import { httpError, messageOf, OllamaError, providerMessage } from '../errors.ts';
import { isAbort } from './abort.ts';

/** Everything one JSON request needs; the client builds it, this module performs it. */
export interface JsonExchange {
  /** Normalized base URL, used when naming the failure. */
  baseUrl: string;
  path: string;
  init: RequestInit;
  /** Headers already merged by the client: its defaults, the key and Pi's overrides. */
  headers: Record<string, string>;
  fetch: typeof globalThis.fetch;
  signal?: AbortSignal;
  timeoutMs: number | undefined;
}

/**
 * One deadline for the whole exchange of a JSON endpoint: send, headers and body. Unlike a chat
 * stream there is no progress to re-arm the window on, and aborting the fetch signal also aborts
 * the body read, so a server that sends headers and never finishes the payload is covered without
 * a second mechanism.
 */
export async function requestJson<T> (exchange: JsonExchange): Promise<T> {
  const { baseUrl, path, init, headers, fetch, signal, timeoutMs } = exchange;
  const url = `${baseUrl}${path}`;
  // A non-positive or absent budget means no deadline, which is what the chat idle window also
  // decides, so `timeoutMs: 0` disables both.
  const deadline = timeoutMs !== undefined && timeoutMs > 0 ? AbortSignal.timeout(timeoutMs) : undefined;
  const requestSignal =
    deadline === undefined ? signal : signal === undefined ? deadline : AbortSignal.any([signal, deadline]);
  // The caller's own abort wins over the deadline that may fire alongside it, so a cancelled
  // request keeps reporting `AbortError` instead of a timeout the caller caused.
  const timedOut = (): boolean => deadline?.aborted === true && signal?.aborted !== true;
  const deadlineError = (): OllamaError => new OllamaError('network', `Ollama ${url} timed out after ${timeoutMs}ms`);

  let response: Response;
  try {
    response = await fetch(url, { ...init, headers, signal: requestSignal });
  } catch (error) {
    if (timedOut()) throw deadlineError();
    if (isAbort(error, signal)) throw error;
    throw new OllamaError('network', `Could not reach Ollama at ${baseUrl}: ${messageOf(error)}`);
  }

  if (!response.ok) {
    const text = await response.text().catch((error: unknown) => {
      if (timedOut()) throw deadlineError();
      if (isAbort(error, signal)) throw error;
      throw httpError(response, `the error body could not be read (${messageOf(error)})`);
    });
    throw httpError(response, providerMessage(text));
  }

  try {
    return (await response.json()) as T;
  } catch (error) {
    if (timedOut()) throw deadlineError();
    if (isAbort(error, signal)) throw error;
    throw new OllamaError('protocol', `Ollama returned an invalid JSON body: ${messageOf(error)}`);
  }
}
