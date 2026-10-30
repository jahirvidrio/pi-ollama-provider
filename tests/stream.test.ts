import { describe, expect, it } from "vitest";
import {
	type AssistantMessage,
	type AssistantMessageEvent,
	type Message,
	type Model,
	type ThinkingLevelMap,
	type Tool,
	type Usage,
	isContextOverflow,
	isRetryableAssistantError,
	normalizeContext,
} from "@earendil-works/pi-ai";
import { streamOllamaChat, type OllamaStreamOptions } from "../src/stream.ts";
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

/** What `/api/show` reports for `gpt-oss`: real level names the model accepts. */
const NAMED_LEVELS: ThinkingLevelMap = {
	off: "off",
	minimal: null,
	low: "low",
	medium: "medium",
	high: "high",
	xhigh: null,
	max: null,
};

/** What `catalog.ts` derives for a model that only reports `thinking.values: [false, true]`. */
const BOOLEAN_ONLY: ThinkingLevelMap = {
	off: "off",
	minimal: null,
	low: null,
	medium: "true",
	high: null,
	xhigh: null,
	max: null,
};

/** Streams one request against a fake server and returns the JSON body it received. */
async function sentRequest(
	input: {
		model?: Partial<Model<"ollama-chat">>;
		options?: OllamaStreamOptions;
		config?: { kind: "local" | "cloud" };
		tools?: Tool[];
	} = {},
): Promise<Record<string, unknown>> {
	const server = await startFakeOllama({
		chat: { chunks: [{ message: { role: "assistant", content: "ok" }, done: true, done_reason: "stop" }] },
	});
	try {
		await collect(
			streamOllamaChat(
				model({ baseUrl: server.baseUrl, ...input.model }),
				context(input.tools),
				input.options,
				input.config,
			),
		);
		return (server.requests[0]?.body ?? {}) as Record<string, unknown>;
	} finally {
		await server.stop();
	}
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
			// Pi's usage contract: 11 prompt tokens include the 8 cached ones, so `input` is the
			// uncached remainder and the total still counts each token once.
			expect(message.usage).toMatchObject({
				input: 3,
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

	it("buffers a tool call split across chunks into a single call", async () => {
		const server = await startFakeOllama({
			chat: {
				chunks: [
					{
						message: { role: "assistant", content: "", tool_calls: [{ function: { name: "read", arguments: {} } }] },
						done: false,
					},
					{
						message: { role: "assistant", content: "", tool_calls: [{ function: { arguments: { path: "a.ts" } } }] },
						done: false,
					},
					{ message: { role: "assistant", content: "" }, done: true, done_reason: "stop" },
				],
			},
		});
		try {
			const events = await collect(streamOllamaChat(model({ baseUrl: server.baseUrl }), context()));

			// One call, emitted once: a partial call must never reach Pi as a callable tool.
			expect(types(events)).toEqual(["start", "toolcall_start", "toolcall_end", "done"]);
			expect(lastMessage(events).content).toEqual([
				{ type: "toolCall", id: "ollama_call_0", name: "read", arguments: { path: "a.ts" } },
			]);
			expect(lastMessage(events).stopReason).toBe("toolUse");
		} finally {
			await server.stop();
		}
	});

	it("parses tool call arguments delivered as a JSON string", async () => {
		const server = await startFakeOllama({
			chat: {
				chunks: [
					{
						message: { role: "assistant", content: "", tool_calls: [{ function: { name: "read", arguments: '{"path":"a.ts"}' } }] },
						done: true,
						done_reason: "stop",
					},
				],
			},
		});
		try {
			const events = await collect(streamOllamaChat(model({ baseUrl: server.baseUrl }), context()));

			expect(lastMessage(events).content).toEqual([
				{ type: "toolCall", id: "ollama_call_0", name: "read", arguments: { path: "a.ts" } },
			]);
		} finally {
			await server.stop();
		}
	});

	it("drops a tool call with no name and keeps the turn's text visible", async () => {
		const server = await startFakeOllama({
			chat: {
				chunks: [
					{ message: { role: "assistant", content: "hello" }, done: false },
					{
						message: { role: "assistant", content: "", tool_calls: [{ function: { arguments: { path: "a.ts" } } }] },
						done: true,
						done_reason: "stop",
					},
				],
			},
		});
		try {
			const events = await collect(streamOllamaChat(model({ baseUrl: server.baseUrl }), context()));

			expect(types(events)).toEqual(["start", "text_start", "text_delta", "text_delta", "text_end", "done"]);
			expect(lastMessage(events).stopReason).toBe("stop");
			expect(lastMessage(events).content).toEqual([
				{ type: "text", text: "hello\n[dropped a malformed tool call from the server]" },
			]);
		} finally {
			await server.stop();
		}
	});

	it("drops a tool call whose arguments are not a JSON object", async () => {
		const server = await startFakeOllama({
			chat: {
				chunks: [
					{ message: { role: "assistant", content: "hello" }, done: false },
					{
						message: { role: "assistant", content: "", tool_calls: [{ function: { name: "read", arguments: "{not json" } }] },
						done: true,
						done_reason: "stop",
					},
				],
			},
		});
		try {
			const events = await collect(streamOllamaChat(model({ baseUrl: server.baseUrl }), context()));

			expect(types(events).filter((type) => type.startsWith("toolcall"))).toEqual([]);
			expect(lastMessage(events).content).toEqual([
				{ type: "text", text: "hello\n[dropped a malformed tool call from the server]" },
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

	it("sends think from the model map, num_ctx for local, num_predict and tools", async () => {
		const tools: Tool[] = [
			{ name: "read", description: "Read a file", parameters: { type: "object", properties: {} } as Tool["parameters"] },
		];
		const body = await sentRequest({
			model: { reasoning: true, thinkingLevelMap: NAMED_LEVELS },
			options: { reasoning: "medium", maxTokens: 512, samplingParams: { top_p: 0.9, top_k: 40 } },
			config: { kind: "local" },
			tools,
		});

		expect(body).toMatchObject({
			think: "medium",
			options: { num_ctx: 131072, num_predict: 512, top_p: 0.9, top_k: 40 },
			tools: [{ type: "function", function: { name: "read", description: "Read a file" } }],
		});
		expect(body).toHaveProperty("options.num_ctx", 131072);
		expect(body).not.toHaveProperty("keep_alive");
	});

	it("omits num_ctx and keep_alive for a cloud request", async () => {
		const body = await sentRequest({ config: { kind: "cloud" } });
		expect(body).not.toHaveProperty("options");
		expect(body).not.toHaveProperty("keep_alive");
	});

	it("sends the model's own level name verbatim for an enabled level", async () => {
		const renamed = { ...NAMED_LEVELS, high: "thinking_hard" };
		expect((await sentRequest({ model: { reasoning: true, thinkingLevelMap: NAMED_LEVELS }, options: { reasoning: "high" } })).think).toBe(
			"high",
		);
		expect((await sentRequest({ model: { reasoning: true, thinkingLevelMap: renamed }, options: { reasoning: "high" } })).think).toBe(
			"thinking_hard",
		);
	});

	it("sends boolean false for off only when the model supports turning thinking off", async () => {
		expect((await sentRequest({ model: { reasoning: true, thinkingLevelMap: NAMED_LEVELS }, options: { reasoning: "off" } })).think).toBe(
			false,
		);
		const offUnsupported = { ...NAMED_LEVELS, off: null };
		expect(await sentRequest({ model: { reasoning: true, thinkingLevelMap: offUnsupported }, options: { reasoning: "off" } })).not.toHaveProperty(
			"think",
		);
	});

	it("sends boolean true for a boolean-only thinking model", async () => {
		expect(
			(await sentRequest({ model: { reasoning: true, thinkingLevelMap: BOOLEAN_ONLY }, options: { reasoning: "medium" } })).think,
		).toBe(true);
	});

	it("omits think for an unsupported level, an unset level or a model without thinking", async () => {
		expect(await sentRequest({ model: { reasoning: true, thinkingLevelMap: NAMED_LEVELS }, options: { reasoning: "max" } })).not.toHaveProperty(
			"think",
		);
		expect(await sentRequest({ model: {}, options: { reasoning: "medium" } })).not.toHaveProperty("think");
		expect(await sentRequest({ model: { reasoning: true, thinkingLevelMap: NAMED_LEVELS } })).not.toHaveProperty("think");
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

	it("reports a stream that ends without a completion chunk as an error", async () => {
		const server = await startFakeOllama({
			chat: { chunks: [{ message: { role: "assistant", content: "partial" }, done: false }] },
		});
		try {
			const events = await collect(streamOllamaChat(model({ baseUrl: server.baseUrl }), context()));

			expect(events.at(-1)?.type).toBe("error");
			const message = lastMessage(events);
			expect(message.stopReason).toBe("error");
			expect(message.errorMessage).toContain("completion chunk");
			expect(message.content).toEqual([{ type: "text", text: "partial" }]);
		} finally {
			await server.stop();
		}
	});

	// The end-of-stream message is a cross-package contract, not just prose: Pi's classifier decides
	// whether a failed turn is retried by matching the phrasing, so losing that match turns a
	// dropped connection into a dead turn with no retry. These two tests pin both halves — this
	// adapter still emits the sentence, and Pi still classifies it as retryable.
	it("emits an end-of-stream failure Pi classifies as retryable", async () => {
		const server = await startFakeOllama({
			chat: { chunks: [{ message: { role: "assistant", content: "partial" }, done: false }] },
		});
		try {
			const events = await collect(streamOllamaChat(model({ baseUrl: server.baseUrl }), context()));
			const message = lastMessage(events);

			expect(message.errorMessage).toBe("Ollama stream ended without a completion chunk");
			expect(isRetryableAssistantError(message)).toBe(true);
		} finally {
			await server.stop();
		}
	});

	it("keeps the end-of-stream sentence retryable for Pi's classifier", () => {
		// Spelled out on purpose: this is the exact string the adapter relies on, so a classifier
		// that stops recognizing it fails here before a user sees a turn that is never retried.
		const message: AssistantMessage = {
			role: "assistant",
			content: [],
			api: "ollama-chat",
			provider: "ollama",
			model: "gemma4:31b",
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { ...ZERO_COST, total: 0 },
			},
			stopReason: "error",
			errorMessage: "Ollama stream ended without a completion chunk",
			timestamp: 0,
		};

		expect(isRetryableAssistantError(message)).toBe(true);
	});
});

describe("streamOllamaChat transcript transform", () => {
	const zeroUsage: Usage = {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { ...ZERO_COST, total: 0 },
	};

	function assistant(overrides: Partial<AssistantMessage> = {}): AssistantMessage {
		return {
			role: "assistant",
			content: [],
			api: "ollama-chat",
			provider: "ollama",
			model: "gemma4:31b",
			usage: zeroUsage,
			stopReason: "stop",
			timestamp: 0,
			...overrides,
		};
	}

	/** Streams one request over the given transcript and returns the JSON body the server saw. */
	async function chatBody(
		messages: Message[],
		modelOverrides: Partial<Model<"ollama-chat">> = {},
	): Promise<{ messages: Record<string, unknown>[] }> {
		const server = await startFakeOllama({
			chat: { chunks: [{ message: { role: "assistant", content: "ok" }, done: true, done_reason: "stop" }] },
		});
		try {
			await collect(
				streamOllamaChat(
					model({ baseUrl: server.baseUrl, ...modelOverrides }),
					normalizeContext({ systemPrompt: "You are Pi.", messages }),
				),
			);
			return (server.requests[0]?.body ?? { messages: [] }) as { messages: Record<string, unknown>[] };
		} finally {
			await server.stop();
		}
	}

	it("drops an aborted turn instead of replaying a tool call with no result", async () => {
		const body = await chatBody([
			{ role: "user", content: "hi", timestamp: 0 },
			assistant({
				stopReason: "aborted",
				content: [{ type: "toolCall", id: "ollama_call_0", name: "read", arguments: { path: "a" } }],
			}),
		]);

		expect(body.messages.map((message) => message.role)).toEqual(["system", "user"]);
	});

	it("synthesizes a result for an orphaned tool call", async () => {
		const body = await chatBody([
			{ role: "user", content: "hi", timestamp: 0 },
			assistant({ content: [{ type: "toolCall", id: "ollama_call_0", name: "read", arguments: {} }] }),
		]);

		expect(body.messages.map((message) => message.role)).toEqual(["system", "user", "assistant", "tool"]);
		expect(body.messages.at(-1)).toMatchObject({ role: "tool", content: "No result provided" });
	});

	it("downgrades an image for a model that cannot read it", async () => {
		const body = await chatBody(
			[
				{
					role: "user",
					content: [
						{ type: "text", text: "what is this?" },
						{ type: "image", data: "aGk=", mimeType: "image/png" },
					],
					timestamp: 0,
				},
			],
			{ input: ["text"] },
		);

		expect(body.messages.at(-1)).toMatchObject({
			role: "user",
			content: "what is this?(image omitted: model does not support images)",
		});
		expect(body.messages.at(-1)).not.toHaveProperty("images");
	});
});
