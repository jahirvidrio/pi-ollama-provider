/**
 * HTTP client for the native Ollama API (`/api/*`). It owns the request shape — URL, headers,
 * credential — and delegates the two hard parts: one JSON exchange with its deadline
 * (`json.ts`) and the retried establishment of a chat response (`establish.ts`).
 */

import { capErrorBody, messageOf, OllamaError } from '../errors.ts';
import {
  type OllamaChatChunk,
  type OllamaChatRequest,
  type OllamaModelSummary,
  type OllamaShowResponse,
  normalizeOllamaBaseUrl,
} from '../wire.ts';
import { abortReason, isAbort } from './abort.ts';
import { establishChatResponse } from './establish.ts';
import { createIdleWindow } from './idle-window.ts';
import { requestJson } from './json.ts';
import { readLines } from './ndjson.ts';
import { DEFAULT_MAX_RETRIES, type OllamaClientOptions, type OllamaResponseInfo } from './options.ts';

/** Headers of an HTTP response as a plain record, for `onResponse`. */
function responseHeaders (headers: Headers): Record<string, string> {
  const record: Record<string, string> = {};
  headers.forEach((value, key) => {
    record[key] = value;
  });
  return record;
}

function parseChunk (line: string): OllamaChatChunk {
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
   * Streams one chat completion as decoded NDJSON chunks. Chunks are yielded as soon
   * as their line arrives; an `error` chunk or a malformed line throws `OllamaError`
   * after the chunks that already arrived, because Ollama reports mid-stream failures
   * inside the stream and never changes the HTTP status.
   *
   * Establishing the response is the only step a retry may replay; once the body is being
   * consumed nothing is retried, which is the same rule `openai-node` applies to a streaming
   * call. See `establish.ts`.
   */
  async * chat (request: OllamaChatRequest, signal?: AbortSignal): AsyncGenerator<OllamaChatChunk> {
    // The attempt's abort state: the caller's signal is forwarded into it, so one signal answers
    // both "the caller cancelled" and "the server went silent".
    const window = createIdleWindow(this.timeoutMs);
    const forwardAbort = (): void => window.abort(signal?.reason);
    signal?.addEventListener('abort', forwardAbort, { once: true });
    // A signal that is already aborted never fires the listener, so mirror the caller's state
    // before the fetch instead of letting a cancelled request start.
    if (signal?.aborted) forwardAbort();

    try {
      const response = await establishChatResponse({
        baseUrl: this.baseUrl,
        init: {
          method: 'POST',
          headers: this.requestHeaders({ 'content-type': 'application/json' }),
          body: JSON.stringify({ ...request, stream: true }),
        },
        fetch: this.fetchImpl,
        signal,
        window,
        maxRetries: this.maxRetries,
        maxRetryDelayMs: this.maxRetryDelayMs,
      });

      if (this.onResponse) {
        // Caller work never counts against the server: pause the window while the hook runs and
        // let the read loop start it again right before the first chunk.
        window.clear();
        const info: OllamaResponseInfo = { status: response.status, headers: responseHeaders(response.headers) };
        await this.onResponse(info);
      }
      if (!response.body) {
        throw new OllamaError('protocol', 'Ollama /api/chat returned no response body');
      }

      try {
        for await (const line of readLines(response.body, window)) {
          yield parseChunk(line);
        }
      } catch (error) {
        if (isAbort(error, signal) || window.signal.aborted) throw abortReason(error, signal, window.signal);
        if (error instanceof OllamaError) throw error;
        throw new OllamaError('stream', `Ollama chat stream failed: ${messageOf(error)}`);
      }
    } finally {
      // The window can be armed before the fetch and the caller abort listener is attached even
      // earlier, so every exit path (failed fetch, HTTP error, missing body, early generator
      // return) must release both.
      window.clear();
      signal?.removeEventListener('abort', forwardAbort);
    }
  }

  private json<T> (path: string, init: RequestInit, signal?: AbortSignal): Promise<T> {
    return requestJson<T>({
      baseUrl: this.baseUrl,
      path,
      init,
      headers: this.requestHeaders(init.headers as Record<string, string | null> | undefined),
      fetch: this.fetchImpl,
      signal,
      timeoutMs: this.timeoutMs,
    });
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
