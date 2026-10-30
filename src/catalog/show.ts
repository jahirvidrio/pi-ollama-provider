/** The `/api/show` fan-out: metadata for every summary, a few requests at a time. */

import type { OllamaClient } from '../client/index.ts';
import type { OllamaModelSummary, OllamaShowResponse } from '../wire.ts';
import { modelName } from './model.ts';

/** How many `/api/show` requests are allowed in flight while loading one catalog. */
const SHOW_CONCURRENCY = 4;

/** Enriches every summary with `/api/show`, at most {@link SHOW_CONCURRENCY} at a time. */
export async function showAll (
  client: OllamaClient,
  summaries: readonly OllamaModelSummary[],
  signal: AbortSignal | undefined
): Promise<(OllamaShowResponse | undefined)[]> {
  const shows = new Array<OllamaShowResponse | undefined>(summaries.length);
  let next = 0;
  const worker = async (): Promise<void> => {
    while (true) {
      const index = next;
      next += 1;
      const summary = summaries[index];
      if (summary === undefined) return;
      try {
        shows[index] = await client.showModel(modelName(summary), signal);
      } catch (error) {
        // One model without metadata degrades to defaults; it must not fail the catalog.
        if (signal?.aborted) throw error;
        shows[index] = undefined;
      }
    }
  };
  const workers = Math.min(SHOW_CONCURRENCY, summaries.length);
  await Promise.all(Array.from({ length: workers }, worker));
  return shows;
}
