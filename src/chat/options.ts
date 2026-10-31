/** What a chat stream call may be told, and which deployment it targets. */

import type { ModelThinkingLevel, SimpleStreamOptions } from '@earendil-works/pi-ai';

/** Which Ollama server a request targets; only the local server allocates a context window. */
export type OllamaChatConfig = { kind: 'local' | 'cloud' };

/**
 * `SimpleStreamOptions` with the thinking level widened to `ModelThinkingLevel`: unlike the
 * built-in adapters, this one does not clamp the level, so it also honors an explicit
 * `off` and omits `think` for a level the model does not offer.
 */
export type OllamaStreamOptions = Omit<SimpleStreamOptions, 'reasoning'> & {
  reasoning?: ModelThinkingLevel;
};
