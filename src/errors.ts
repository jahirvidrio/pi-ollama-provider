/**
 * The adapter's error model: one error type, plus the bounded text that describes a failure.
 * Everything a caller needs to classify or display an Ollama failure lives here, so the HTTP
 * client and the stream adapter cannot drift on what an error says or how long it may be.
 */

export type OllamaErrorKind = 'http' | 'stream' | 'network' | 'protocol';

export class OllamaError extends Error {
  readonly kind: OllamaErrorKind;
  /**
   * HTTP status and response headers, when a response arrived at all. Both are always own
   * properties, even when undefined: Pi's `isProviderError` probe requires the keys to be
   * present before it reads `status`, and its retry policy takes `retry-after` and
   * `x-should-retry` from these headers.
   */
  readonly status: number | undefined;
  readonly headers: Headers | undefined;

  constructor (kind: OllamaErrorKind, message: string, status?: number, headers?: Headers) {
    super(message);
    this.name = 'OllamaError';
    this.kind = kind;
    this.status = status;
    this.headers = headers;
  }
}

/**
 * One non-OK response as an error. The status and its headers are what Pi's retry policy reads,
 * and the description is what the user reads, so it never ends at a separator with nothing
 * after it: a body that arrived empty and one that could not be read each say which they are.
 */
export function httpError (response: Response, description: string): OllamaError {
  const detail = description.trim().length > 0 ? description : 'the server sent no error body';
  return new OllamaError('http', `${response.status}: ${detail}`, response.status, response.headers);
}

/** The text of a thrown value, for the messages that carry it. */
export function messageOf (error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Same cap Pi's `error-body` helper applies, so a hostile or buggy server cannot inflate the
 * transcript. Reused for an error chunk and for a malformed stream line: both end up in
 * `errorMessage`, which the transcript persists, the TUI renders and Pi matches its error
 * patterns against.
 */
const MAX_ERROR_BODY_CHARS = 4000;

/**
 * Largest JSON error body that is parsed whole. Cap-then-parse keeps the envelope intact up to
 * here, so a server that explains itself in a long JSON payload still contributes its message;
 * past it the capped raw body is shown instead of paying for a parse that cannot fit the
 * transcript either. Anything at or below the 4 KB display cap parses in both designs.
 */
const MAX_PARSED_ERROR_BODY_CHARS = 64 * 1024;

export function capErrorBody (text: string): string {
  const trimmed = text.trim();
  if (trimmed.length <= MAX_ERROR_BODY_CHARS) return trimmed;
  return `${trimmed.slice(0, MAX_ERROR_BODY_CHARS)}… [truncated ${trimmed.length - MAX_ERROR_BODY_CHARS} chars]`;
}

/** The message a non-OK response carries: its JSON `error` field when it has one, capped either way. */
export function providerMessage (text: string): string {
  if (text.length > MAX_PARSED_ERROR_BODY_CHARS) return capErrorBody(text);
  try {
    const parsed = JSON.parse(text) as { error?: unknown };
    if (typeof parsed.error === 'string') return capErrorBody(parsed.error);
  } catch {
    // Not JSON: the capped raw body is the best description available.
  }
  return capErrorBody(text);
}
