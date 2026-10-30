import { describe, expect, it } from "vitest";
import { OllamaClient, OllamaError, normalizeOllamaBaseUrl } from "../src/client.ts";
import { startFakeOllama } from "./helpers/fake-ollama.ts";

describe("normalizeOllamaBaseUrl", () => {
	it("appends /api to an origin", () => {
		expect(normalizeOllamaBaseUrl("http://localhost:11434")).toBe("http://localhost:11434/api");
		expect(normalizeOllamaBaseUrl("https://ollama.com")).toBe("https://ollama.com/api");
	});

	it("keeps an existing /api suffix and trims trailing slashes", () => {
		expect(normalizeOllamaBaseUrl("http://localhost:11434/api/")).toBe("http://localhost:11434/api");
		expect(normalizeOllamaBaseUrl("  https://ollama.com/api  ")).toBe("https://ollama.com/api");
	});
});

describe("OllamaClient auth", () => {
	it("sends a bearer key only when one is configured", async () => {
		const server = await startFakeOllama();
		try {
			await new OllamaClient({ baseUrl: server.baseUrl, apiKey: "secret" }).listModels();
			await new OllamaClient({ baseUrl: server.baseUrl }).listModels();

			expect(server.requests[0]?.headers.authorization).toBe("Bearer secret");
			expect(server.requests[1]?.headers.authorization).toBeUndefined();
		} finally {
			await server.stop();
		}
	});
});

describe("OllamaClient endpoints", () => {
	it("lists models from /api/tags", async () => {
		const server = await startFakeOllama({
			tags: {
				models: [
					{ name: "gemma4:31b", model: "gemma4:31b", capabilities: ["completion", "vision"] },
					{ name: "gpt-oss:20b", model: "gpt-oss:20b", capabilities: ["completion", "tools", "thinking"] },
				],
			},
		});
		try {
			const models = await new OllamaClient({ baseUrl: server.baseUrl }).listModels();
			expect(models).toEqual([
				{ name: "gemma4:31b", model: "gemma4:31b", capabilities: ["completion", "vision"] },
				{ name: "gpt-oss:20b", model: "gpt-oss:20b", capabilities: ["completion", "tools", "thinking"] },
			]);
			expect(server.requests[0]?.path).toBe("/api/tags");
		} finally {
			await server.stop();
		}
	});

	it("shows one model through POST /api/show", async () => {
		const server = await startFakeOllama({
			show: {
				capabilities: ["completion", "tools", "thinking"],
				model_info: { "gptoss.context_length": 131072 },
				thinking: { values: ["low", "medium", "high"], default: "medium" },
			},
		});
		try {
			const shown = await new OllamaClient({ baseUrl: server.baseUrl }).showModel("gpt-oss:20b");
			expect(shown.model_info?.["gptoss.context_length"]).toBe(131072);
			expect(shown.thinking).toEqual({ values: ["low", "medium", "high"], default: "medium" });
			expect(server.requests[0]?.body).toEqual({ model: "gpt-oss:20b" });
		} finally {
			await server.stop();
		}
	});
});

describe("OllamaClient chat", () => {
	it("delivers NDJSON chunks in order", async () => {
		const server = await startFakeOllama({
			chat: {
				chunks: [
					{ message: { role: "assistant", content: "Hel" }, done: false },
					{ message: { role: "assistant", content: "lo" }, done: false },
					{ message: { role: "assistant", content: "" }, done: true, done_reason: "stop", eval_count: 2 },
				],
			},
		});
		try {
			const chunks: unknown[] = [];
			for await (const chunk of new OllamaClient({ baseUrl: server.baseUrl }).chat({
				model: "gemma4:31b",
				messages: [{ role: "user", content: "hi" }],
			})) {
				chunks.push(chunk);
			}

			expect(chunks).toEqual([
				{ message: { role: "assistant", content: "Hel" }, done: false },
				{ message: { role: "assistant", content: "lo" }, done: false },
				{ message: { role: "assistant", content: "" }, done: true, done_reason: "stop", eval_count: 2 },
			]);
			expect(server.requests[0]?.body).toEqual({
				model: "gemma4:31b",
				messages: [{ role: "user", content: "hi" }],
				stream: true,
			});
		} finally {
			await server.stop();
		}
	});

	it("rejects an HTTP error with the provider message and status", async () => {
		const server = await startFakeOllama({ chat: { status: 401, body: { error: "Unauthorized" } } });
		try {
			const client = new OllamaClient({ baseUrl: server.baseUrl, apiKey: "bad" });
			const error = await client
				.chat({ model: "gpt-oss:20b", messages: [{ role: "user", content: "hi" }] })
				.next()
				.catch((cause: unknown) => cause);

			expect(error).toBeInstanceOf(OllamaError);
			expect((error as OllamaError).kind).toBe("http");
			expect((error as OllamaError).status).toBe(401);
			expect((error as OllamaError).message).toContain("Unauthorized");
		} finally {
			await server.stop();
		}
	});

	it("rejects a mid-stream error chunk after yielding earlier chunks", async () => {
		const server = await startFakeOllama({
			chat: {
				chunks: [
					{ message: { role: "assistant", content: "Yes" }, done: false },
					{ error: "an error was encountered while running the model" },
				],
			},
		});
		try {
			const iterator = new OllamaClient({ baseUrl: server.baseUrl }).chat({
				model: "gemma4:31b",
				messages: [{ role: "user", content: "hi" }],
			});

			const first = await iterator.next();
			expect(first.value).toEqual({ message: { role: "assistant", content: "Yes" }, done: false });

			const error = await iterator.next().catch((cause: unknown) => cause);
			expect(error).toBeInstanceOf(OllamaError);
			expect((error as OllamaError).kind).toBe("stream");
			expect((error as OllamaError).message).toContain("while running the model");
		} finally {
			await server.stop();
		}
	});

	it("rejects a malformed NDJSON line instead of skipping it", async () => {
		const server = await startFakeOllama({
			chat: { rawChunks: ['{"message":{"content":"ok"},"done":false}', "not-json"] },
		});
		try {
			const iterator = new OllamaClient({ baseUrl: server.baseUrl }).chat({
				model: "gemma4:31b",
				messages: [{ role: "user", content: "hi" }],
			});
			await iterator.next();
			const error = await iterator.next().catch((cause: unknown) => cause);

			expect(error).toBeInstanceOf(OllamaError);
			expect((error as OllamaError).kind).toBe("stream");
			expect((error as OllamaError).message).toContain("not-json");
		} finally {
			await server.stop();
		}
	});

	it("rejects with an AbortError when the caller aborts", async () => {
		const server = await startFakeOllama({
			chat: { chunks: [{ message: { role: "assistant", content: "partial" }, done: false }], keepOpen: true },
		});
		try {
			const controller = new AbortController();
			const iterator = new OllamaClient({ baseUrl: server.baseUrl }).chat(
				{ model: "gemma4:31b", messages: [{ role: "user", content: "hi" }] },
				controller.signal,
			);

			await iterator.next();
			controller.abort();
			const error = await iterator.next().catch((cause: unknown) => cause);

			expect((error as Error).name).toBe("AbortError");
		} finally {
			await server.stop();
		}
	});
});
