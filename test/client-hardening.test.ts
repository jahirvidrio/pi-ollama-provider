import { getEventListeners } from "node:events";
import { describe, expect, it } from "vitest";
import { OllamaClient, OllamaError } from "../src/client.ts";
import { startFakeOllama } from "./helpers/fake-ollama.ts";

const REQUEST = { model: "gemma4:31b", messages: [{ role: "user" as const, content: "hi" }] };

describe("request timeout", () => {
	it("aborts a stream that stalls after the headers, and names the timeout", async () => {
		const server = await startFakeOllama({
			chat: { chunks: [{ message: { role: "assistant", content: "partial" }, done: false }], keepOpen: true },
		});
		try {
			const client = new OllamaClient({ baseUrl: server.baseUrl, timeoutMs: 150 });
			const iterator = client.chat(REQUEST);

			const first = await iterator.next();
			expect(first.value).toMatchObject({ message: { content: "partial" } });

			const startedAt = Date.now();
			const error = await iterator.next().catch((cause: unknown) => cause);
			const elapsed = Date.now() - startedAt;

			expect(error).toBeInstanceOf(OllamaError);
			expect((error as OllamaError).message).toContain("timed out");
			expect(elapsed).toBeLessThan(2_000);
		} finally {
			await server.stop();
		}
	}, 10_000);

	it("aborts a request whose headers never arrive, and names the timeout", async () => {
		const server = await startFakeOllama({ chat: { hangHeaders: true } });
		try {
			const client = new OllamaClient({ baseUrl: server.baseUrl, timeoutMs: 150 });
			const startedAt = Date.now();
			const error = await client.chat(REQUEST).next().catch((cause: unknown) => cause);
			const elapsed = Date.now() - startedAt;

			expect(error).toBeInstanceOf(OllamaError);
			expect((error as OllamaError).message).toContain("timed out");
			expect(elapsed).toBeLessThan(2_000);
		} finally {
			await server.stop();
		}
	}, 10_000);

	it("lets a stream finish when every chunk arrives inside the window", async () => {
		const server = await startFakeOllama({
			chat: {
				chunks: [
					{ message: { role: "assistant", content: "a" }, done: false },
					{ message: { role: "assistant", content: "b" }, done: false },
					{ message: { role: "assistant", content: "" }, done: true, done_reason: "stop" },
				],
				chunkDelayMs: 50,
			},
		});
		try {
			const contents: string[] = [];
			const client = new OllamaClient({ baseUrl: server.baseUrl, timeoutMs: 150 });
			for await (const chunk of client.chat(REQUEST)) {
				contents.push(chunk.message?.content ?? "");
			}
			expect(contents).toEqual(["a", "b", ""]);
		} finally {
			await server.stop();
		}
	}, 10_000);

	it("does not charge consumer time between chunks against the timeout", async () => {
		const server = await startFakeOllama({
			chat: {
				chunks: [
					{ message: { role: "assistant", content: "a" }, done: false },
					{ message: { role: "assistant", content: "" }, done: true, done_reason: "stop" },
				],
				chunkDelayMs: 250,
			},
		});
		try {
			const client = new OllamaClient({ baseUrl: server.baseUrl, timeoutMs: 150 });
			const iterator = client.chat(REQUEST);
			const first = await iterator.next();
			expect(first.value).toMatchObject({ message: { content: "a" } });

			await new Promise((resolve) => setTimeout(resolve, 400));

			const second = await iterator.next();
			expect(second.value).toMatchObject({ done: true });
		} finally {
			await server.stop();
		}
	}, 10_000);

	it("does not charge the onResponse hook against the timeout", async () => {
		const server = await startFakeOllama({
			chat: {
				chunks: [
					{ message: { role: "assistant", content: "a" }, done: false },
					{ message: { role: "assistant", content: "" }, done: true, done_reason: "stop" },
				],
			},
		});
		try {
			const client = new OllamaClient({
				baseUrl: server.baseUrl,
				timeoutMs: 150,
				onResponse: async () => {
					await new Promise((resolve) => setTimeout(resolve, 400));
				},
			});
			const contents: string[] = [];
			for await (const chunk of client.chat(REQUEST)) {
				contents.push(chunk.message?.content ?? "");
			}
			expect(contents).toEqual(["a", ""]);
		} finally {
			await server.stop();
		}
	}, 10_000);

	it("surfaces the timeout reason when an error body stalls", async () => {
		const server = await startFakeOllama({ chat: { status: 500, hangBody: true } });
		try {
			const client = new OllamaClient({ baseUrl: server.baseUrl, timeoutMs: 150 });
			const error = await client.chat(REQUEST).next().catch((cause: unknown) => cause);

			expect(error).toBeInstanceOf(OllamaError);
			expect((error as OllamaError).kind).toBe("network");
			expect((error as OllamaError).message).toContain("timed out");
		} finally {
			await server.stop();
		}
	}, 10_000);
});

describe("caller cancellation", () => {
	it("rejects without sending when the caller signal is already aborted", async () => {
		const server = await startFakeOllama();
		try {
			const controller = new AbortController();
			controller.abort();

			const error = await new OllamaClient({ baseUrl: server.baseUrl })
				.chat(REQUEST, controller.signal)
				.next()
				.catch((cause: unknown) => cause);

			expect((error as Error).name).toBe("AbortError");
			expect(server.requests).toHaveLength(0);
		} finally {
			await server.stop();
		}
	});

	it("releases the caller abort listener after a completed chat", async () => {
		const server = await startFakeOllama({
			chat: {
				chunks: [
					{ message: { role: "assistant", content: "a" }, done: false },
					{ message: { role: "assistant", content: "" }, done: true, done_reason: "stop" },
				],
			},
		});
		try {
			const controller = new AbortController();
			for await (const chunk of new OllamaClient({ baseUrl: server.baseUrl }).chat(REQUEST, controller.signal)) {
				void chunk;
			}
			expect(getEventListeners(controller.signal, "abort")).toEqual([]);
		} finally {
			await server.stop();
		}
	});

	it("releases the caller abort listener when the fetch fails", async () => {
		const controller = new AbortController();
		const client = new OllamaClient({
			baseUrl: "http://127.0.0.1:1/api",
			fetch: async () => {
				throw new Error("connection failed");
			},
		});
		const error = await client.chat(REQUEST, controller.signal).next().catch((cause: unknown) => cause);

		expect(error).toBeInstanceOf(OllamaError);
		expect((error as OllamaError).kind).toBe("network");
		expect(getEventListeners(controller.signal, "abort")).toEqual([]);
	});

	it("releases the caller abort listener after an HTTP error", async () => {
		const controller = new AbortController();
		const client = new OllamaClient({
			baseUrl: "http://127.0.0.1:1/api",
			fetch: async () => new Response("boom", { status: 500 }),
		});
		const error = await client.chat(REQUEST, controller.signal).next().catch((cause: unknown) => cause);

		expect(error).toBeInstanceOf(OllamaError);
		expect((error as OllamaError).kind).toBe("http");
		expect(getEventListeners(controller.signal, "abort")).toEqual([]);
	});

	it("releases the caller abort listener when the response has no body", async () => {
		const controller = new AbortController();
		const client = new OllamaClient({
			baseUrl: "http://127.0.0.1:1/api",
			fetch: async () => new Response(null, { status: 200 }),
		});
		const error = await client.chat(REQUEST, controller.signal).next().catch((cause: unknown) => cause);

		expect(error).toBeInstanceOf(OllamaError);
		expect((error as OllamaError).kind).toBe("protocol");
		expect(getEventListeners(controller.signal, "abort")).toEqual([]);
	});
});

describe("error bodies", () => {
	it("caps a huge error body instead of keeping all of it", async () => {
		const server = await startFakeOllama({ chat: { status: 500, body: { error: "x".repeat(2_000_000) } } });
		try {
			const error = await new OllamaClient({ baseUrl: server.baseUrl })
				.chat(REQUEST)
				.next()
				.catch((cause: unknown) => cause);

			expect(error).toBeInstanceOf(OllamaError);
			expect((error as OllamaError).status).toBe(500);
			expect((error as Error).message.length).toBeLessThan(4_200);
			expect((error as Error).message).toContain("truncated");
		} finally {
			await server.stop();
		}
	});
});

describe("stream boundaries", () => {
	it("parses a final line that has no trailing newline", async () => {
		const server = await startFakeOllama({
			chat: {
				rawBody:
					'{"message":{"role":"assistant","content":"a"},"done":false}\n' +
					'{"message":{"role":"assistant","content":"b"},"done":true,"done_reason":"stop"}',
			},
		});
		try {
			const contents: string[] = [];
			for await (const chunk of new OllamaClient({ baseUrl: server.baseUrl }).chat(REQUEST)) {
				contents.push(chunk.message?.content ?? "");
			}
			expect(contents).toEqual(["a", "b"]);
		} finally {
			await server.stop();
		}
	});
});
