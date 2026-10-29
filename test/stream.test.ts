import { describe, expect, it } from "vitest";
import {
	type AssistantMessageEvent,
	type Model,
	type Tool,
	isContextOverflow,
	normalizeContext,
} from "@earendil-works/pi-ai";
import { streamOllamaChat, type OllamaStreamMetadata } from "../src/stream.ts";
import { startFakeOllama } from "./helpers/fake-ollama.ts";

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

function model(overrides: Partial<Model<"ollama-chat">> = {}): Model<"ollama-chat"> {
	return {
		id: "gemma4:31b",
		name: "gemma4:31b",
		api: "ollama-chat",
		provider: "ollama",
		baseUrl: "http://127.0.0.1:1/api",
		reasoning: false,
		input: ["text"],
		cost: ZERO_COST,
		contextWindow: 131072,
		maxTokens: 8192,
		...overrides,
	};
}

function context(tools?: Tool[]) {
	return normalizeContext({ systemPrompt: "You are Pi.", messages: [{ role: "user", content: "hi", timestamp: 0 }], tools });
}

async function collect(stream: AsyncIterable<AssistantMessageEvent>): Promise<AssistantMessageEvent[]> {
	const events: AssistantMessageEvent[] = [];
	for await (const event of stream) events.push(event);
	return events;
}

function types(events: AssistantMessageEvent[]): string[] {
	return events.map((event) => event.type);
}

function lastMessage(events: AssistantMessageEvent[]) {
	const terminal = events.at(-1);
	if (terminal?.type === "done") return terminal.message;
	if (terminal?.type === "error") return terminal.error;
	throw new Error("stream did not terminate");
}

describe("streamOllamaChat", () => {
	it("streams text and finishes with a stop reason and usage", async () => {
		const server = await startFakeOllama({
			chat: {
				chunks: [
					{ message: { role: "assistant", content: "Hel" }, done: false },
					{ message: { role: "assistant", content: "lo" }, done: false },
					{
						message: { role: "assistant", content: "" },
						done: true,
						done_reason: "stop",
						prompt_eval_count: 11,
						prompt_eval_cached_count: 8,
						eval_count: 2,
					},
				],
			},
		});
		try {
			const events = await collect(
				streamOllamaChat(model({ baseUrl: server.baseUrl }), context(), { apiKey: undefined }),
			);

			expect(types(events)).toEqual(["start", "text_start", "text_delta", "text_delta", "text_end", "done"]);
			expect(events.at(-1)).toMatchObject({ type: "done", reason: "stop" });
			const message = lastMessage(events);
			expect(message.content).toEqual([{ type: "text", text: "Hello" }]);
			expect(message.usage).toMatchObject({
				input: 11,
				output: 2,
				cacheRead: 8,
				cacheWrite: 0,
				totalTokens: 13,
			});
			expect(message.stopReason).toBe("stop");
			expect(server.requests[0]?.body).toEqual({
				model: "gemma4:31b",
				messages: [
					{ role: "system", content: "You are Pi." },
					{ role: "user", content: "hi" },
				],
				stream: true,
			});
		} finally {
			await server.stop();
		}
	});

	it("streams thinking as its own block", async () => {
		const server = await startFakeOllama({
			chat: {
				chunks: [
					{ message: { role: "assistant", thinking: "hm" }, done: false },
					{ message: { role: "assistant", content: "ok" }, done: false },
					{ message: { role: "assistant", content: "" }, done: true, done_reason: "stop" },
				],
			},
		});
		try {
			const events = await collect(streamOllamaChat(model({ baseUrl: server.baseUrl }), context()));
			expect(types(events)).toEqual([
				"start",
				"thinking_start",
				"thinking_delta",
				"thinking_end",
				"text_start",
				"text_delta",
				"text_end",
				"done",
			]);
			expect((events[2] as { delta: string }).delta).toBe("hm");
			expect(lastMessage(events).content).toEqual([
				{ type: "thinking", thinking: "hm" },
				{ type: "text", text: "ok" },
			]);
		} finally {
			await server.stop();
		}
	});

	it("reports tool calls and stops with toolUse", async () => {
		const server = await startFakeOllama({
			chat: {
				chunks: [
					{
						message: {
							role: "assistant",
							content: "",
							tool_calls: [{ function: { name: "read", arguments: { path: "a.ts" } } }],
						},
						done: true,
						done_reason: "stop",
					},
				],
			},
		});
		try {
			const events = await collect(streamOllamaChat(model({ baseUrl: server.baseUrl }), context()));
			expect(types(events)).toEqual(["start", "toolcall_start", "toolcall_end", "done"]);
			expect(events.at(-1)).toMatchObject({ type: "done", reason: "toolUse" });
			expect(lastMessage(events).content).toEqual([
				{ type: "toolCall", id: "ollama_call_0", name: "read", arguments: { path: "a.ts" } },
			]);
		} finally {
			await server.stop();
		}
	});

	it("maps a length done reason to a length stop", async () => {
		const server = await startFakeOllama({
			chat: { chunks: [{ message: { role: "assistant", content: "partial" }, done: true, done_reason: "length" }] },
		});
		try {
			const events = await collect(streamOllamaChat(model({ baseUrl: server.baseUrl }), context()));
			expect(events.at(-1)).toMatchObject({ type: "done", reason: "length" });
			expect(lastMessage(events).stopReason).toBe("length");
		} finally {
			await server.stop();
		}
	});

	it("turns a mid-stream error into an error event that Pi detects as overflow", async () => {
		const server = await startFakeOllama({
			chat: {
				chunks: [
					{ message: { role: "assistant", content: "start" }, done: false },
					{ error: "prompt too long; exceeded max context length by 12 tokens" },
				],
			},
		});
		try {
			const events = await collect(streamOllamaChat(model({ baseUrl: server.baseUrl }), context()));
			expect(events.at(-1)?.type).toBe("error");
			const message = lastMessage(events);
			expect(message.stopReason).toBe("error");
			expect(message.errorMessage).toContain("exceeded max context length");
			expect(isContextOverflow(message)).toBe(true);
		} finally {
			await server.stop();
		}
	});

	it("turns an HTTP failure into an error event instead of throwing", async () => {
		const server = await startFakeOllama({ chat: { status: 401, body: { error: "Unauthorized" } } });
		try {
			const events = await collect(
				streamOllamaChat(model({ baseUrl: server.baseUrl }), context(), { apiKey: "bad" }),
			);
			expect(events.at(-1)?.type).toBe("error");
			expect(lastMessage(events).errorMessage).toContain("Unauthorized");
		} finally {
			await server.stop();
		}
	});

	it("finishes with an aborted error when the caller aborts", async () => {
		const server = await startFakeOllama({
			chat: { chunks: [{ message: { role: "assistant", content: "partial" }, done: false }], keepOpen: true },
		});
		try {
			const controller = new AbortController();
			const stream = streamOllamaChat(model({ baseUrl: server.baseUrl }), context(), { signal: controller.signal });
			const events: AssistantMessageEvent[] = [];
			for await (const event of stream) {
				events.push(event);
				if (event.type === "text_delta") controller.abort();
			}

			expect(events.at(-1)?.type).toBe("error");
			expect(lastMessage(events).stopReason).toBe("aborted");
		} finally {
			await server.stop();
		}
	});

	it("sends think, num_ctx, num_predict and tools when configured", async () => {
		const server = await startFakeOllama({
			chat: { chunks: [{ message: { role: "assistant", content: "ok" }, done: true, done_reason: "stop" }] },
		});
		const metadata: OllamaStreamMetadata = {
			numCtx: 131072,
			keepAlive: "5m",
			thinking: { values: ["low", "medium", "high"], default: "medium" },
		};
		const tools: Tool[] = [
			{ name: "read", description: "Read a file", parameters: { type: "object", properties: {} } as Tool["parameters"] },
		];
		try {
			await collect(
				streamOllamaChat(model({ baseUrl: server.baseUrl, reasoning: true }), context(tools), {
					reasoning: "medium",
					maxTokens: 512,
					samplingParams: { top_p: 0.9, top_k: 40 },
				}, metadata),
			);

			expect(server.requests[0]?.body).toMatchObject({
				think: "medium",
				keep_alive: "5m",
				options: { num_ctx: 131072, num_predict: 512, top_p: 0.9, top_k: 40 },
				tools: [{ type: "function", function: { name: "read", description: "Read a file" } }],
			});
		} finally {
			await server.stop();
		}
	});

	it("lets onPayload replace the request body before sending", async () => {
		const server = await startFakeOllama({
			chat: { chunks: [{ message: { role: "assistant", content: "ok" }, done: true, done_reason: "stop" }] },
		});
		let respondedStatus: number | undefined;
		try {
			await collect(
				streamOllamaChat(model({ baseUrl: server.baseUrl }), context(), {
					onPayload: (payload) => ({ ...(payload as object), think: "high" }),
					onResponse: (response) => {
						respondedStatus = response.status;
					},
				}),
			);

			expect(server.requests[0]?.body).toMatchObject({ think: "high" });
			expect(respondedStatus).toBe(200);
		} finally {
			await server.stop();
		}
	});

	it("keeps an empty response terminal", async () => {
		const server = await startFakeOllama({
			chat: { chunks: [{ message: { role: "assistant", content: "" }, done: true, done_reason: "stop" }] },
		});
		try {
			const events = await collect(streamOllamaChat(model({ baseUrl: server.baseUrl }), context()));
			expect(types(events)).toEqual(["start", "done"]);
			expect(lastMessage(events).content).toEqual([]);
			expect(lastMessage(events).usage.totalTokens).toBe(0);
		} finally {
			await server.stop();
		}
	});
});
