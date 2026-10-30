/**
 * Establishing the chat response: the only part of a chat request a retry may replay, because a
 * second request after the body starts arriving would duplicate text the caller already saw.
 */

import { retryProviderRequest } from '../../vendor/pi-ai/utils/provider-retry.js';
import { httpError, messageOf, OllamaError, providerMessage } from '../errors.ts';
import { abortReason } from './abort.ts';
import { IDLE_TIMEOUT_STATUS, type IdleWindow } from './idle-window.ts';

/**
 * The delay a failed response asked for, as a readable suffix. Display only: the retry policy
 * belongs to the vendored helper, and a delay it refuses to wait out is reported, never silently
 * shortened. An HTTP-date `retry-after` is left out instead of being parsed a second time.
 */
function retryDelayHint (headers: Headers | undefined): string {
  const milliseconds = headers?.get('retry-after-ms');
  if (milliseconds !== null && milliseconds !== undefined) {
    const value = Number.parseFloat(milliseconds);
    if (Number.isFinite(value)) return ` (the server asked to retry after ${Math.ceil(value / 1000)}s)`;
  }
  const seconds = headers?.get('retry-after');
  if (seconds !== null && seconds !== undefined) {
    const value = Number.parseFloat(seconds);
    if (Number.isFinite(value)) return ` (the server asked to retry after ${Math.ceil(value)}s)`;
  }
  return '';
}

/**
 * What to throw when establishing the response failed. The helper substitutes its own error for
 * the provider's in exactly two cases: the caller aborted, whose outcome stays the caller's, and a
 * server-requested delay above the retry budget. In the second the provider's own error is more
 * truthful — it carries the status, the headers Pi's policy reads and the server's own words — so
 * the adapter reports that one instead, naming the delay the server asked for.
 */
function retryFailure (error: unknown, lastAttemptError: unknown, signal: AbortSignal | undefined): unknown {
  if (signal?.aborted || !(lastAttemptError instanceof OllamaError) || error === lastAttemptError) {
    return error;
  }
  return new OllamaError(
    lastAttemptError.kind,
    `${lastAttemptError.message}${retryDelayHint(lastAttemptError.headers)}`,
    lastAttemptError.status,
    lastAttemptError.headers
  );
}

/** The pieces one establishment needs; the client owns every value here. */
export interface ChatAttempt {
  /** Normalized base URL: the request goes to `${baseUrl}/chat`. */
  baseUrl: string;
  /** Method, headers and body of the chat request, already built by the client. */
  init: RequestInit;
  fetch: typeof globalThis.fetch;
  /** The caller's signal: it interrupts a backoff and outranks the attempt's own abort. */
  signal: AbortSignal | undefined;
  /** The attempt's abort and timeout state; the fetch is sent with its signal. */
  window: IdleWindow;
  maxRetries: number;
  maxRetryDelayMs: number | undefined;
}

/**
 * Sends the chat request and returns the response once its headers arrive, retrying only this
 * step under Pi's retry policy (`maxRetries`, `maxRetryDelayMs`, `retry-after`, `x-should-retry`).
 * A connection failure is retried: it is the typical transient case. An idle timeout is not,
 * because `timeoutMs` already bounds it and retrying would multiply that bounded wait.
 */
export async function establishChatResponse (attempt: ChatAttempt): Promise<Response> {
  /** The failure of the most recent attempt, which the helper may replace with its own. */
  let lastAttemptError: unknown;

  const send = async (): Promise<Response> => {
    // Start the window before the fetch: a server that accepts the connection but never answers
    // (hung proxy, model still loading before its first byte) would otherwise stall the request
    // forever, because the read loop that re-arms the timer only starts once the headers arrive.
    attempt.window.arm();
    try {
      const response = await attempt.fetch(`${attempt.baseUrl}/chat`, {
        ...attempt.init,
        signal: attempt.window.signal,
      });
      // Headers are progress too: restart the window so a stalled error body keeps its own budget.
      attempt.window.arm();
      if (response.ok) return response;
      // A stalled error body surfaces the timeout that stopped it instead of an empty status line;
      // a body that cannot be read at all names that, so the caller always gets a description next
      // to the status.
      let readFailure: unknown;
      const text = await response.text().catch((error: unknown) => {
        if (attempt.window.signal.aborted) throw abortReason(error, attempt.signal, attempt.window.signal);
        readFailure = error;
        return undefined;
      });
      throw httpError(
        response,
        text === undefined
          ? `the error body could not be read (${messageOf(readFailure)})`
          : providerMessage(text)
      );
    } catch (error) {
      // One rule for an interrupted attempt, whichever side stopped it: the caller's own reason
      // wins, then the idle timeout that aborted this attempt, then the raw failure. Reading the
      // two signals separately let the result depend on which abort happened to land first.
      let failure: unknown;
      if (attempt.signal?.aborted || attempt.window.signal.aborted) {
        failure = abortReason(error, attempt.signal, attempt.window.signal);
      } else if (error instanceof OllamaError) {
        failure = error;
      } else {
        // No response at all: the transient failure the retry policy exists to ride out.
        failure = new OllamaError('network', `Could not reach Ollama at ${attempt.baseUrl}: ${messageOf(error)}`);
      }
      // Exactly what the helper saw, for the failures it may replace with its own.
      lastAttemptError = failure;
      throw failure;
    } finally {
      // The window covers one attempt: the wait between attempts is backoff, not the server's
      // silence, so a stale timer must not fire during the sleep.
      attempt.window.clear();
    }
  };

  try {
    return await retryProviderRequest(send, {
      maxRetries: attempt.maxRetries,
      maxRetryDelayMs: attempt.maxRetryDelayMs,
      // The caller's abort interrupts a backoff; the attempt signal belongs to one attempt.
      signal: attempt.signal,
      noRetryStatuses: [IDLE_TIMEOUT_STATUS],
    });
  } catch (error) {
    throw retryFailure(error, lastAttemptError, attempt.signal);
  }
}
