/** One `/api/tags` summary plus its `/api/show` metadata translated into a Pi model. */

import type { Model, ModelInputModality } from '@earendil-works/pi-ai';
import { OLLAMA_CHAT_API, type OllamaModelSummary, type OllamaShowResponse } from '../wire.ts';
import { thinkingLevelMapOf } from './thinking.ts';

/**
 * Context window used when `/api/show` reports no `*.context_length`, so a model
 * without metadata is still usable with a conservative budget instead of a guess.
 */
const DEFAULT_CONTEXT_WINDOW = 4096;

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } as const;

export interface ToChatModelInput {
  summary: OllamaModelSummary;
  show: OllamaShowResponse;
  providerId: string;
  baseUrl: string;
}

/** The model id Ollama itself uses: `/api/tags` repeats it in `name` and `model`. */
export function modelName (summary: OllamaModelSummary): string {
  return summary.name ?? summary.model;
}

/**
 * `/api/show` nests the context window under an architecture prefix that differs per
 * model (`gptoss.context_length`, `gemma3.context_length`, ...), so the key is matched
 * by suffix instead of by an architecture allowlist.
 */
function contextWindowOf (show: OllamaShowResponse): number {
  for (const [key, value] of Object.entries(show.model_info ?? {})) {
    if (!key.endsWith('.context_length')) continue;
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) return Math.floor(value);
  }
  return DEFAULT_CONTEXT_WINDOW;
}

/**
 * Union of the capabilities `/api/tags` and `/api/show` report. Cloud lists them inline
 * on `/api/tags`, local models report them from `/api/show`, and either may omit one, so
 * a modality is only dropped when both endpoints agree it is missing.
 */
function capabilitiesOf (summary: OllamaModelSummary, show: OllamaShowResponse): Set<string> {
  return new Set([...(summary.capabilities ?? []), ...(show.capabilities ?? [])]);
}

/**
 * Builds one Pi model from the tags summary and the `/api/show` payload. Everything a
 * request needs (context window, thinking levels) lives on the model, so a catalog
 * restored from disk keeps it without a network round trip. `show` may be empty: a model
 * whose metadata request failed keeps conservative defaults instead of disappearing from
 * the catalog.
 */
export function toChatModel (input: ToChatModelInput): Model<'ollama-chat'> {
  const { summary, show, providerId, baseUrl } = input;
  const id = modelName(summary);
  const contextWindow = contextWindowOf(show);
  const capabilities = capabilitiesOf(summary, show);
  const thinkingLevelMap = thinkingLevelMapOf(show.thinking);

  const modalities: ModelInputModality[] = ['text'];
  if (capabilities.has('vision')) modalities.push('image');

  const model: Model<'ollama-chat'> = {
    id,
    name: id,
    api: OLLAMA_CHAT_API,
    provider: providerId,
    baseUrl,
    reasoning: thinkingLevelMap !== undefined,
    input: modalities,
    cost: { ...ZERO_COST },
    contextWindow,
    maxTokens: contextWindow,
  };
  if (thinkingLevelMap !== undefined) model.thinkingLevelMap = thinkingLevelMap;
  return model;
}
