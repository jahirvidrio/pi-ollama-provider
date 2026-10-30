/** HTTP client for the native Ollama API (`/api/*`): endpoints, deadlines, retries and the chat stream. */

import { retryProviderRequest } from '../vendor/pi-ai/utils/provider-retry.js';
import { capErrorBody, OllamaError, providerMessage } from './errors.ts';
import { readLines } from './ndjson.ts';
import {
  type OllamaChatChunk,
  type OllamaChatRequest,
  type OllamaModelSummary,
  type OllamaShowResponse,
  normalizeOllamaBaseUrl,
} from './wire.ts';

function isAbort (error: unknown, signal: AbortSignal | undefined): boolean {
  if (signal?.aborted) return true;
  return error instanceof Error && error.name === 'AbortError';
}

/** The text of a thrown value, for the messages that carry it. */
function messageOf (error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Chooses what to throw when a request or read stopped early: the caller's own abort keeps its
 * `AbortError`, while a stream aborted by the idle timer keeps the `OllamaError` that names it.
 */
function abortReason (
  error: unknown,
  signal: AbortSignal | undefined,
  controllerSignal: AbortSignal
): unknown {
  if (signal?.aborted) return signal.reason ?? error;
  const reason = controllerSignal.reason;
  return reason instanceof OllamaError ? reason : error;
}

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

/** Status and headers of an Ollama HTTP response, mirroring Pi's `ProviderResponse`. */
export interface OllamaResponseInfo {
  status: number;
  headers: Record<string, string>;
}

function responseHeaders (headers: Headers): Record<string, string> {
  const record: Record<string, string> = {};
  headers.forEach((value, key) => {
    record[key] = value;
  });
  return record;
}

export interface OllamaClientOptions {
  /** Origin or `/api` base URL; normalized before use. */
  baseUrl: string;
  apiKey?: string;
  /** Fetch implementation supplied by Pi for the current request. */
  fetch?: typeof globalThis.fetch;
  /**
   * Abort a request when the server goes this long without progress. A chat request has an idle
   * window: from the send until the response headers arrive, then while awaiting each body
   * chunk, and caller work never counts against it. The JSON endpoints (`/tags`, `/show`) have no
   * progress to observe, so they get this as a deadline for the whole exchange, body included; the
   * catalog and the login probes pass their own budgets. Pi passes its own request timeout here.
   */
  timeoutMs?: number;
  /**
   * Headers Pi computed for this request (`ProviderRequestOptions.headers`): session, auth,
   * caller and `transformHeaders` values, already merged. They win over the adapter's own
   * defaults and a `null` value removes one, which is the rule Pi applies to its defaults.
   * Names are matched case-insensitively. Every request of the client honors them, the
   * catalog included.
   */
  headers?: Record<string, string | null>;
  /**
   * Pi's client-side retry budget for establishing a chat response. Defaults to
   * {@link DEFAULT_MAX_RETRIES}, the same budget the OpenAI and Anthropic SDKs use.
   */
  maxRetries?: number;
  /** Pi's cap on a server-requested retry delay; `0` disables the cap, absent keeps the helper's 60s. */
  maxRetryDelayMs?: number;
  /**
   * Called once the chat response headers arrive and before its body is consumed, so
   * callers can observe the request the way Pi's `onResponse` hook expects. Only the
   * chat request reports a response: the other endpoints return their body at once.
   */
  onResponse?: (response: OllamaResponseInfo) => void | Promise<void>;
}

/** Same budget the OpenAI and Anthropic SDKs retry with, which Pi's helper mirrors. */
const DEFAULT_MAX_RETRIES = 2;

/**
 * Status that marks an idle timeout. It is a provider-shaped error so Pi's retry policy sees
 * it, but `chat` lists it in `noRetryStatuses`: the wait is already bounded by `timeoutMs`, and
 * retrying would turn a five minute stall into fifteen. A server that genuinely answers 408 is
 * therefore not retried either; Ollama does not send one.
 */
const IDLE_TIMEOUT_STATUS = 408;

export class OllamaClient {
  readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly fetchImpl: typeof globalThis.fetch;
  private readonly onResponse: OllamaClientOptions['onResponse'];
  private readonly timeoutMs: number | undefined;
  private readonly requestHeaderOverrides: Record<string, string | null> | undefined;
  private readonly maxRetries: number;
  private readonly maxRetryDelayMs: number | undefined;

  constructor (options: OllamaClientOptions) {
    this.baseUrl = normalizeOllamaBaseUrl(options.baseUrl);
    this.apiKey = options.apiKey;
    this.fetchImpl = options.fetch ?? globalThis.fetch;
    this.onResponse = options.onResponse;
    this.timeoutMs = options.timeoutMs;
    this.requestHeaderOverrides = options.headers;
    this.maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
    this.maxRetryDelayMs = options.maxRetryDelayMs;
  }

  /**
   * The headers of one request: the adapter's defaults, then Pi's request headers, which win
   * and where `null` removes a default — the same rule `providerHeadersToRecord` applies.
   * Keys are deduplicated case-insensitively, keeping the last spelling.
   */
  private requestHeaders (extra?: Record<string, string | null>): Record<string, string> {
    const merged = new Map<string, [string, string]>();
    const apply = (source: Record<string, string | null> | undefined): void => {
      for (const [name, value] of Object.entries(source ?? {})) {
        merged.delete(name.toLowerCase());
        if (value !== null) merged.set(name.toLowerCase(), [name, value]);
      }
    };
    apply(extra);
    if (this.apiKey) apply({ authorization: `Bearer ${this.apiKey}` });
    apply(this.requestHeaderOverrides);
    return Object.fromEntries(merged.values());
  }

  /** The error that closes one idle window; see {@link IDLE_TIMEOUT_STATUS}. */
  private timeoutError (): OllamaError {
    return new OllamaError(
      'network',
      `Ollama stream timed out after ${this.timeoutMs}ms without data`,
      IDLE_TIMEOUT_STATUS
    );
  }

  /** The error that closes one JSON request whose whole budget ran out. */
  private deadlineError (path: string): OllamaError {
    return new OllamaError('network', `Ollama ${this.baseUrl}${path} timed out after ${this.timeoutMs}ms`);
  }

  /**
   * One non-OK response as an error. The status and its headers are what Pi's retry policy reads,
   * and the description is what the user reads, so it never ends at a separator with nothing
   * after it: a body that arrived empty and one that could not be read each say which they are.
   */
  private httpError (response: Response, description: string): OllamaError {
    const detail = description.trim().length > 0 ? description : 'the server sent no error body';
    return new OllamaError('http', `${response.status}: ${detail}`, response.status, response.headers);
  }

  /**
   * One deadline for the whole exchange of a JSON endpoint: send, headers and body. Unlike a
   * chat stream there is no progress to re-arm the window on, and aborting the fetch signal also
   * aborts the body read, so a server that sends headers and never finishes the payload is
   * covered without a second mechanism.
   */
  private async json<T>(path: string, init: RequestInit, signal?: AbortSignal): Promise<T> {
    // A non-positive or absent budget means no deadline, which is what the chat idle window
    // also decides, so `timeoutMs: 0` disables both.
    const deadline =
      this.timeoutMs !== undefined && this.timeoutMs > 0 ? AbortSignal.timeout(this.timeoutMs) : undefined;
    const requestSignal =
      deadline === undefined ? signal : signal === undefined ? deadline : AbortSignal.any([signal, deadline]);
    // The caller's own abort wins over the deadline that may fire alongside it, so a cancelled
    // request keeps reporting `AbortError` instead of a timeout the caller caused.
    const timedOut = (): boolean => deadline?.aborted === true && signal?.aborted !== true;
    let response: Response;
    try {
      response = await this.fetchImpl(`${this.baseUrl}${path}`, {
        ...init,
        headers: this.requestHeaders(init.headers as Record<string, string | null> | undefined),
        signal: requestSignal,
      });
    } catch (error) {
      if (timedOut()) throw this.deadlineError(path);
      if (isAbort(error, signal)) throw error;
      throw new OllamaError('network', `Could not reach Ollama at ${this.baseUrl}: ${messageOf(error)}`);
    }
    if (!response.ok) {
      const text = await response.text().catch((error: unknown) => {
        if (timedOut()) throw this.deadlineError(path);
        if (isAbort(error, signal)) throw error;
        throw this.httpError(response, `the error body could not be read (${messageOf(error)})`);
      });
      throw this.httpError(response, providerMessage(text));
    }
    try {
      return (await response.json()) as T;
    } catch (error) {
      if (timedOut()) throw this.deadlineError(path);
      if (isAbort(error, signal)) throw error;
      throw new OllamaError(
        'protocol',
        `Ollama returned an invalid JSON body: ${messageOf(error)}`
      );
    }
  }

  async listModels (signal?: AbortSignal): Promise<OllamaModelSummary[]> {
    const payload = await this.json<{ models?: unknown }>('/tags', { method: 'GET' }, signal);
    if (!Array.isArray(payload.models)) {
      throw new OllamaError('protocol', 'Ollama /api/tags response has no models array');
    }
    return payload.models as OllamaModelSummary[];
  }

  async showModel (model: string, signal?: AbortSignal): Promise<OllamaShowResponse> {
    return this.json<OllamaShowResponse>(
      '/show',
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model }),
      },
      signal
    );
  }

  /**
   * What to throw when establishing the response failed. The helper substitutes its own error for
   * the provider's in exactly two cases: the caller aborted, whose outcome stays the caller's, and a
   * server-requested delay above the retry budget. In the second the provider's own error is more
   * truthful — it carries the status, the headers Pi's policy reads and the server's own words — so
   * the adapter reports that one instead, naming the delay the server asked for.
   */
  private retryFailure (error: unknown, lastAttemptError: unknown, signal: AbortSignal | undefined): unknown {
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

  /**
   * Streams one chat completion as decoded NDJSON chunks. Chunks are yielded as soon
   * as their line arrives; an `error` chunk or a malformed line throws `OllamaError`
   * after the chunks that already arrived, because Ollama reports mid-stream failures
   * inside the stream and never changes the HTTP status.
   *
   * Establishing the response follows Pi's retry policy (`maxRetries`, `maxRetryDelayMs`,
   * `retry-after`, `x-should-retry`). A connection failure is retried: it is the typical
   * transient case. An idle timeout is not, because `timeoutMs` already bounds it and
   * retrying would multiply that bounded wait by the retry budget. Once the body is being
   * consumed nothing is retried, which is the same rule `openai-node` applies to a
   * streaming call.
   */
  async * chat (request: OllamaChatRequest, signal?: AbortSignal): AsyncGenerator<OllamaChatChunk> {
    const controller = new AbortController();
    const forwardAbort = (): void => controller.abort(signal?.reason);
    signal?.addEventListener('abort', forwardAbort, { once: true });
    // A signal that is already aborted never fires the listener, so mirror the caller's state
    // before the fetch instead of letting a cancelled request start.
    if (signal?.aborted) forwardAbort();
    let idleTimer: ReturnType<typeof setTimeout> | undefined;
    const clearIdleTimer = (): void => {
      if (idleTimer) clearTimeout(idleTimer);
      idleTimer = undefined;
    };
    const armIdleTimer = (): void => {
      if (this.timeoutMs === undefined || this.timeoutMs <= 0) return;
      clearIdleTimer();
      idleTimer = setTimeout(() => {
        controller.abort(this.timeoutError());
      }, this.timeoutMs);
    };
    /** The failure of the most recent attempt, which the helper may replace with its own. */
    let lastAttemptError: unknown;
    /**
     * One attempt at establishing the response, the only part a retry can replay: once the body
     * is being consumed a second request would duplicate the text already yielded.
     */
    const attempt = async (): Promise<Response> => {
      // Start the window before the fetch: a server that accepts the connection but never
      // answers (hung proxy, model still loading before its first byte) would otherwise stall
      // the request forever, because the read loop that arms the timer only starts once the
      // headers arrive.
      armIdleTimer();
      try {
        const response = await this.fetchImpl(`${this.baseUrl}/chat`, {
          method: 'POST',
          headers: this.requestHeaders({ 'content-type': 'application/json' }),
          body: JSON.stringify({ ...request, stream: true }),
          signal: controller.signal,
        });
        // Headers are progress too: restart the window so a stalled error body keeps its own
        // budget. The read loop re-arms it on each awaited chunk.
        armIdleTimer();
        if (response.ok) return response;
        // A stalled error body surfaces the timeout that stopped it instead of an empty
        // status line; a body that cannot be read at all names that, so the caller always gets
        // a description next to the status.
        let readFailure: unknown;
        const text = await response.text().catch((error: unknown) => {
          if (controller.signal.aborted) throw abortReason(error, signal, controller.signal);
          readFailure = error;
          return undefined;
        });
        throw this.httpError(
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
        if (signal?.aborted || controller.signal.aborted) {
          failure = abortReason(error, signal, controller.signal);
        } else if (error instanceof OllamaError) {
          failure = error;
        } else {
          // No response at all: the transient failure the retry policy exists to ride out.
          failure = new OllamaError('network', `Could not reach Ollama at ${this.baseUrl}: ${messageOf(error)}`);
        }
        // Exactly what the helper saw, for the failures it may replace with its own.
        lastAttemptError = failure;
        throw failure;
      } finally {
        // The window covers one attempt: the wait between attempts is backoff, not the
        // server's silence, so a stale timer must not fire during the sleep.
        clearIdleTimer();
      }
    };
    let response: Response;
    try {
      try {
        response = await retryProviderRequest(attempt, {
          maxRetries: this.maxRetries,
          maxRetryDelayMs: this.maxRetryDelayMs,
          // The caller's abort interrupts a backoff; the attempt controller belongs to one attempt.
          signal,
          noRetryStatuses: [IDLE_TIMEOUT_STATUS],
        });
      } catch (error) {
        throw this.retryFailure(error, lastAttemptError, signal);
      }
      if (this.onResponse) {
        // Caller work never counts against the server: pause the window while the hook runs
        // and let the read loop start it again right before the first chunk.
        clearIdleTimer();
        await this.onResponse({ status: response.status, headers: responseHeaders(response.headers) });
      }
      if (!response.body) {
        throw new OllamaError('protocol', 'Ollama /api/chat returned no response body');
      }

      try {
        for await (const line of readLines(response.body, { arm: armIdleTimer, clear: clearIdleTimer })) {
          yield this.parseChunk(line);
        }
      } catch (error) {
        if (isAbort(error, signal) || controller.signal.aborted) throw abortReason(error, signal, controller.signal);
        if (error instanceof OllamaError) throw error;
        throw new OllamaError(
          'stream',
          `Ollama chat stream failed: ${messageOf(error)}`
        );
      } finally {
        clearIdleTimer();
      }
    } finally {
      // The window can be armed before the fetch and the caller abort listener is attached
      // even earlier, so every exit path (failed fetch, HTTP error, missing body, early
      // generator return) must release both.
      clearIdleTimer();
      signal?.removeEventListener('abort', forwardAbort);
    }
  }

  private parseChunk (line: string): OllamaChatChunk {
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      throw new OllamaError('stream', `Ollama sent a malformed stream line: ${capErrorBody(line)}`);
    }
    if (parsed === null || typeof parsed !== 'object') {
      throw new OllamaError('stream', `Ollama sent a non-object stream line: ${capErrorBody(line)}`);
    }
    const chunk = parsed as OllamaChatChunk;
    if (typeof chunk.error === 'string') {
      throw new OllamaError('stream', capErrorBody(chunk.error));
    }
    return chunk;
  }
}

/** The part of a resolved credential the client needs: the key Pi picked for this request, if any. */
export interface OllamaCredential {
  apiKey?: string;
}

/** Every client option `createClient` does not take positionally. */
export type OllamaClientOverrides = Omit<OllamaClientOptions, 'baseUrl' | 'apiKey'>;

/**
 * The single place an `OllamaClient` is built, so a new option is honored everywhere by
 * construction instead of by memory. Defaults live in the class: `fetch` falls back to the
 * global implementation and `maxRetries` to the SDK budget. The logins, the catalog refresh and
 * the chat stream each used to build the client by hand, which is how `timeoutMs` ended up
 * missing from three of the four sites.
 */
export function createClient (
  baseUrl: string,
  credential?: OllamaCredential,
  overrides?: OllamaClientOverrides
): OllamaClient {
  return new OllamaClient({ baseUrl, apiKey: credential?.apiKey, ...overrides });
}
