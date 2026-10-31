/** The idle window of one chat attempt: the timer that aborts it when the server goes silent. */

import { OllamaError } from '../errors.ts';

/**
 * Status that marks an idle timeout. It is a provider-shaped error so Pi's retry policy sees
 * it, but the chat establishment lists it in `noRetryStatuses`: the wait is already bounded by
 * `timeoutMs`, and retrying would turn a five minute stall into fifteen. A server that genuinely
 * answers 408 is therefore not retried either; Ollama does not send one.
 */
export const IDLE_TIMEOUT_STATUS = 408;

export function idleTimeoutError (timeoutMs: number | undefined): OllamaError {
  return new OllamaError('network', `Ollama stream timed out after ${timeoutMs}ms without data`, IDLE_TIMEOUT_STATUS);
}

/**
 * The abort state of one attempt. `arm` starts the timer and `clear` stops it, so the window covers
 * only the waits the client spends on the server: decoding, parsing and consumer time never count
 * against it. `abort` forwards the caller's own signal into the same attempt, which lets
 * `abortReason` answer with a single signal instead of racing two.
 */
export interface IdleWindow {
  /** The signal each attempt sends with its fetch; aborted by the caller or by the timer. */
  readonly signal: AbortSignal;
  arm (): void;
  clear (): void;
  abort (reason: unknown): void;
}

/** A non-positive or absent budget means no timer, which is what the JSON deadline also decides. */
export function createIdleWindow (timeoutMs: number | undefined): IdleWindow {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const clear = (): void => {
    if (timer) clearTimeout(timer);
    timer = undefined;
  };
  const arm = (): void => {
    if (timeoutMs === undefined || timeoutMs <= 0) return;
    clear();
    timer = setTimeout(() => {
      controller.abort(idleTimeoutError(timeoutMs));
    }, timeoutMs);
  };
  return {
    signal: controller.signal,
    arm,
    clear,
    abort: (reason) => controller.abort(reason),
  };
}
