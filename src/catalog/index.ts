/** Model catalog: `GET /api/tags` plus `POST /api/show` translated into Pi models. */

import type { Model } from '@earendil-works/pi-ai';
import type { OllamaClient } from '../client/index.ts';
import type { OllamaModelSummary } from '../wire.ts';
import { toChatModel } from './model.ts';
import { showAll } from './show.ts';

export interface LoadCatalogOptions {
  client: OllamaClient;
  providerId: string;
  /** `false` keeps the cached catalog and never opens a connection. */
  allowNetwork: boolean;
  stored?: readonly Model<'ollama-chat'>[];
  signal?: AbortSignal;
}

/**
 * A model is dropped only when `/api/tags` explicitly says it cannot chat: a summary with
 * no `capabilities` at all is kept, because older servers omit the field.
 */
function isChatModel (summary: OllamaModelSummary): boolean {
  if (summary.capabilities === undefined) return true;
  return summary.capabilities.includes('completion');
}

/**
 * Loads the catalog for one provider. Offline and on a failed `/api/tags` the stored
 * catalog is returned unchanged, so a provider always has models to offer.
 */
export async function loadCatalog (options: LoadCatalogOptions): Promise<Model<'ollama-chat'>[]> {
  const stored: Model<'ollama-chat'>[] = [...(options.stored ?? [])];
  if (!options.allowNetwork) {
    return stored;
  }

  let summaries: OllamaModelSummary[];
  try {
    summaries = await options.client.listModels(options.signal);
  } catch (error) {
    if (options.signal?.aborted) throw error;
    return stored;
  }

  const chatModels = summaries.filter(isChatModel);
  const shows = await showAll(options.client, chatModels, options.signal);

  return chatModels.map((summary, index) =>
    toChatModel({
      summary,
      show: shows[index] ?? {},
      providerId: options.providerId,
      baseUrl: options.client.baseUrl,
    })
  );
}
