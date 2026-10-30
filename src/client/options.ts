/** The client's options, and the small shapes its contract exposes. */

/** Status and headers of an Ollama HTTP response, mirroring Pi's `ProviderResponse`. */
export interface OllamaResponseInfo {
  status: number;
  headers: Record<string, string>;
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
export const DEFAULT_MAX_RETRIES = 2;
