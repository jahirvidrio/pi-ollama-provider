/** Model catalog: `GET /api/tags` plus `POST /api/show` translated into Pi models. */

import type { Model, ModelInputModality, ThinkingLevel, ThinkingLevelMap } from "@earendil-works/pi-ai";
import type { OllamaClient, OllamaModelSummary, OllamaShowResponse } from "./client.ts";
import { OLLAMA_CHAT_API } from "./stream.ts";

/**
 * Context window used when `/api/show` reports no `*.context_length`, so a model
 * without metadata is still usable with a conservative budget instead of a guess.
 */
const DEFAULT_CONTEXT_WINDOW = 4096;

/** How many `/api/show` requests are allowed in flight while loading one catalog. */
const SHOW_CONCURRENCY = 4;

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } as const;

export interface ToChatModelInput {
	summary: OllamaModelSummary;
	show: OllamaShowResponse;
	providerId: string;
	baseUrl: string;
}

/** The model id Ollama itself uses: `/api/tags` repeats it in `name` and `model`. */
function modelName(summary: OllamaModelSummary): string {
	return summary.name ?? summary.model;
}

/**
 * `/api/show` nests the context window under an architecture prefix that differs per
 * model (`gptoss.context_length`, `gemma3.context_length`, ...), so the key is matched
 * by suffix instead of by an architecture allowlist.
 */
function contextWindowOf(show: OllamaShowResponse): number {
	for (const [key, value] of Object.entries(show.model_info ?? {})) {
		if (!key.endsWith(".context_length")) continue;
		if (typeof value === "number" && Number.isFinite(value) && value > 0) return Math.floor(value);
	}
	return DEFAULT_CONTEXT_WINDOW;
}

/**
 * Union of the capabilities `/api/tags` and `/api/show` report. Cloud lists them inline
 * on `/api/tags`, local models report them from `/api/show`, and either may omit one, so
 * a modality is only dropped when both endpoints agree it is missing.
 */
function capabilitiesOf(summary: OllamaModelSummary, show: OllamaShowResponse): Set<string> {
	return new Set([...(summary.capabilities ?? []), ...(show.capabilities ?? [])]);
}

/**
 * Pi thinking levels mapped onto the level names the model itself accepts. Named values
 * are matched case-insensitively; a level the model does not offer is `null` so Pi never
 * sends an unsupported named level. A model that only reports booleans offers one on
 * level, exposed as `medium` and mapped to the string `"true"` so the request sends the
 * boolean the model accepts.
 */
function thinkingLevelMapOf(thinking: OllamaShowResponse["thinking"]): ThinkingLevelMap | undefined {
	const values = thinking?.values;
	if (!values || values.length === 0) return undefined;
	const named = values.filter((value): value is string => typeof value === "string");
	if (named.length === 0) {
		if (!values.includes(true)) return undefined;
		return {
			off: "off",
			minimal: null,
			low: null,
			medium: "true",
			high: null,
			xhigh: null,
			max: null,
		};
	}
	const byLevel = new Map(named.map((value) => [value.toLowerCase(), value]));
	const mapped = (level: ThinkingLevel): string | null => byLevel.get(level) ?? null;
	return {
		off: "off",
		minimal: mapped("minimal"),
		low: mapped("low"),
		medium: mapped("medium"),
		high: mapped("high"),
		xhigh: mapped("xhigh"),
		max: mapped("max"),
	};
}

/**
 * Builds one Pi model from the tags summary and the `/api/show` payload. Everything a
 * request needs (context window, thinking levels) lives on the model, so a catalog
 * restored from disk keeps it without a network round trip. `show` may be empty: a model
 * whose metadata request failed keeps conservative defaults instead of disappearing from
 * the catalog.
 */
export function toChatModel(input: ToChatModelInput): Model<"ollama-chat"> {
	const { summary, show, providerId, baseUrl } = input;
	const id = modelName(summary);
	const contextWindow = contextWindowOf(show);
	const capabilities = capabilitiesOf(summary, show);
	const thinkingLevelMap = thinkingLevelMapOf(show.thinking);

	const modalities: ModelInputModality[] = ["text"];
	if (capabilities.has("vision")) modalities.push("image");

	return {
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
		...(thinkingLevelMap !== undefined && { thinkingLevelMap }),
	};
}

export interface LoadCatalogOptions {
	client: OllamaClient;
	providerId: string;
	/** `false` keeps the cached catalog and never opens a connection. */
	allowNetwork: boolean;
	stored?: readonly Model<"ollama-chat">[];
	signal?: AbortSignal;
}

/**
 * A model is dropped only when `/api/tags` explicitly says it cannot chat: a summary with
 * no `capabilities` at all is kept, because older servers omit the field.
 */
function isChatModel(summary: OllamaModelSummary): boolean {
	if (summary.capabilities === undefined) return true;
	return summary.capabilities.includes("completion");
}

/** Enriches every summary with `/api/show`, at most {@link SHOW_CONCURRENCY} at a time. */
async function showAll(
	client: OllamaClient,
	summaries: readonly OllamaModelSummary[],
	signal: AbortSignal | undefined,
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

/**
 * Loads the catalog for one provider. Offline and on a failed `/api/tags` the stored
 * catalog is returned unchanged, so a provider always has models to offer.
 */
export async function loadCatalog(options: LoadCatalogOptions): Promise<Model<"ollama-chat">[]> {
	const stored: Model<"ollama-chat">[] = [...(options.stored ?? [])];
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
		}),
	);
}
