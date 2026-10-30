/** Abort semantics shared by the JSON endpoints and the chat stream. */

import { OllamaError } from '../errors.ts';

/** Whether a failure came from an abort: the caller's signal already fired, or the error says so. */
export function isAbort (error: unknown, signal: AbortSignal | undefined): boolean {
  if (signal?.aborted) return true;
  return error instanceof Error && error.name === 'AbortError';
}

/**
 * Chooses what to throw when a request or read stopped early: the caller's own abort keeps its
 * `AbortError`, while a stream aborted by the idle timer keeps the `OllamaError` that names it.
 */
export function abortReason (
  error: unknown,
  signal: AbortSignal | undefined,
  attemptSignal: AbortSignal
): unknown {
  if (signal?.aborted) return signal.reason ?? error;
  const reason = attemptSignal.reason;
  return reason instanceof OllamaError ? reason : error;
}
