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
