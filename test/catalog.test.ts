import { describe, expect, it } from "vitest";
import type { Model } from "@earendil-works/pi-ai";
import { OllamaClient } from "../src/client.ts";
import { loadCatalog, toChatModel } from "../src/catalog.ts";
import { startFakeOllama } from "./helpers/fake-ollama.ts";

const TAGS = {
	models: [
		{ name: "gemma4:31b", model: "gemma4:31b", capabilities: ["completion", "vision"] },
		{ name: "gpt-oss:20b", model: "gpt-oss:20b", capabilities: ["completion", "tools", "thinking"] },
		{ name: "nomic-embed-text", model: "nomic-embed-text", capabilities: ["embedding"] },
		{ name: "legacy:1b", model: "legacy:1b" },
	],
};

function shows(): unknown {
	return {
		capabilities: ["completion"],
		model_info: { "gptoss.context_length": 131072 },
		thinking: { values: ["low", "medium", "high"], default: "medium" },
	};
}

describe("toChatModel", () => {
	it("maps context window, capabilities and named thinking levels", () => {
		const { model, metadata } = toChatModel({
			summary: { name: "gpt-oss:20b", model: "gpt-oss:20b", capabilities: ["completion", "vision", "thinking"] },
			show: {
				model_info: { "gptoss.context_length": 131072 },
				thinking: { values: ["low", "medium", "high"], default: "medium" },
			},
			providerId: "ollama",
			baseUrl: "http://localhost:11434/api",
			kind: "local",
		});

		expect(model).toMatchObject({
			id: "gpt-oss:20b",
			name: "gpt-oss:20b",
			api: "ollama-chat",
			provider: "ollama",
			baseUrl: "http://localhost:11434/api",
			reasoning: true,
			input: ["text", "image"],
			contextWindow: 131072,
			maxTokens: 131072,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		});
		expect(model.thinkingLevelMap).toEqual({
			off: "off",
			minimal: null,
			low: "low",
			medium: "medium",
			high: "high",
			xhigh: null,
			max: null,
		});
		expect(metadata.thinking?.values).toEqual(["low", "medium", "high"]);
		expect(metadata.numCtx).toBe(131072);
	});

	it("offers off and one on level for a boolean-only thinking model", () => {
		const { model, metadata } = toChatModel({
			summary: { name: "gpt-oss:20b", model: "gpt-oss:20b", capabilities: ["completion", "thinking"] },
			show: { thinking: { values: [false, true], default: true } },
			providerId: "ollama-cloud",
			baseUrl: "https://ollama.com/api",
			kind: "cloud",
		});

		expect(model.reasoning).toBe(true);
		expect(model.thinkingLevelMap).toEqual({
			off: "off",
			minimal: null,
			low: null,
			medium: "medium",
			high: null,
			xhigh: null,
			max: null,
		});
		expect(metadata.thinking?.values).toEqual([false, true]);
	});

	it("falls back to conservative defaults without show metadata", () => {
		const { model, metadata } = toChatModel({
			summary: { name: "gemma4:31b", model: "gemma4:31b" },
			show: {},
			providerId: "ollama",
			baseUrl: "http://localhost:11434/api",
			kind: "local",
		});

		expect(model.contextWindow).toBe(4096);
		expect(model.maxTokens).toBe(4096);
		expect(model.reasoning).toBe(false);
		expect(model.thinkingLevelMap).toBeUndefined();
		expect(model.input).toEqual(["text"]);
		expect(metadata.thinking).toBeUndefined();
		expect(metadata.numCtx).toBe(4096);
	});

	it("does not send num_ctx or keep_alive for cloud models", () => {
		const { metadata } = toChatModel({
			summary: { name: "gpt-oss:20b", model: "gpt-oss:20b" },
			show: { model_info: { "gptoss.context_length": 131072 } },
			providerId: "ollama-cloud",
			baseUrl: "https://ollama.com/api",
			kind: "cloud",
		});

		expect(metadata.numCtx).toBeUndefined();
	});
});

describe("loadCatalog", () => {
	it("skips non-chat models when capabilities are reported and keeps unnamed-capability models", async () => {
		const server = await startFakeOllama({
			tags: TAGS,
			show: (body) => (body.model === "gpt-oss:20b" ? shows() : { capabilities: ["completion"] }),
		});
		try {
			const { models } = await loadCatalog({
				client: new OllamaClient({ baseUrl: server.baseUrl }),
				providerId: "ollama",
				kind: "local",
				allowNetwork: true,
			});

			expect(models.map((model) => model.id)).toEqual(["gemma4:31b", "gpt-oss:20b", "legacy:1b"]);
		} finally {
			await server.stop();
		}
	});

	it("returns stored models without touching the network when offline", async () => {
		const server = await startFakeOllama({ tags: TAGS, show: shows });
		const stored: Model<"ollama-chat">[] = [
			{
				id: "cached:7b",
				name: "cached:7b",
				api: "ollama-chat",
				provider: "ollama",
				baseUrl: server.baseUrl,
				reasoning: false,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 8192,
				maxTokens: 8192,
			},
		];
		try {
			const { models } = await loadCatalog({
				client: new OllamaClient({ baseUrl: server.baseUrl }),
				providerId: "ollama",
				kind: "local",
				allowNetwork: false,
				stored,
			});

			expect(models).toEqual(stored);
			expect(server.requests).toHaveLength(0);
		} finally {
			await server.stop();
		}
	});
});
