/** Ollama's final chunk turned into the usage Pi accounts for. */

import { type Model, type Usage, calculateCost } from '@earendil-works/pi-ai';
import type { OllamaChatChunk } from '../wire.ts';

export function usageOf (chunk: OllamaChatChunk | undefined, model: Model<'ollama-chat'>): Usage {
  // Pi's contract: `input` counts only the uncached prompt tokens and is disjoint from
  // `cacheRead`. `prompt_eval_count` includes the cached prefix, so subtracting here keeps
  // `overflow` (`input + cacheRead`) and `calculateCost` from counting that prefix twice.
  const promptTotal = chunk?.prompt_eval_count ?? 0;
  const cacheRead = chunk?.prompt_eval_cached_count ?? 0;
  const input = Math.max(0, promptTotal - cacheRead);
  const output = chunk?.eval_count ?? 0;
  const usage: Usage = {
    input,
    output,
    cacheRead,
    cacheWrite: 0,
    totalTokens: promptTotal + output,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
  usage.cost = calculateCost(model, usage);
  return usage;
}
