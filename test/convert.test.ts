import { describe, expect, it } from "vitest";
import type { AssistantMessage, Message, Tool, ToolResultMessage, UserMessage, Usage } from "@earendil-works/pi-ai";
import { buildOllamaMessages, ollamaToolCallToPi, toOllamaTools } from "../src/convert.ts";

const ZERO_USAGE: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function assistant(content: AssistantMessage["content"]): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "ollama-chat",
		provider: "ollama",
		model: "gemma4:31b",
		usage: ZERO_USAGE,
		stopReason: "stop",
		timestamp: 0,
	};
}

function user(content: UserMessage["content"]): UserMessage {
	return { role: "user", content, timestamp: 0 };
}

function toolResult(content: ToolResultMessage["content"], isError = false): ToolResultMessage {
	return { role: "toolResult", toolCallId: "call_0", toolName: "read", content, isError, timestamp: 0 };
}

describe("buildOllamaMessages", () => {
	it("puts the system prompt first and omits it when empty", () => {
		const messages: Message[] = [user("hi")];

		expect(buildOllamaMessages({ systemPrompt: "You are Pi.", messages })).toEqual([
			{ role: "system", content: "You are Pi." },
			{ role: "user", content: "hi" },
		]);
		expect(buildOllamaMessages({ systemPrompt: "   ", messages })).toEqual([{ role: "user", content: "hi" }]);
	});

	it("sends user text and base64 images", () => {
		const converted = buildOllamaMessages({
			messages: [
				user([
					{ type: "text", text: "What is this?" },
					{ type: "image", data: "aGVsbG8=", mimeType: "image/png" },
				]),
			],
		});

		expect(converted).toEqual([{ role: "user", content: "What is this?", images: ["aGVsbG8="] }]);
	});

	it("concatenates assistant text, keeps thinking, and drops tool call ids", () => {
		const converted = buildOllamaMessages({
			messages: [
				assistant([
					{ type: "thinking", thinking: "considering" },
					{ type: "text", text: "Let me " },
					{ type: "text", text: "look." },
					{ type: "toolCall", id: "pi-generated", name: "read", arguments: { path: "a.ts" } },
				]),
			],
		});

		expect(converted).toEqual([
			{
				role: "assistant",
				content: "Let me look.",
				thinking: "considering",
				tool_calls: [{ function: { name: "read", arguments: { path: "a.ts" } } }],
			},
		]);
	});

	it("keeps an empty content string for a tool-call-only assistant message", () => {
		const converted = buildOllamaMessages({
			messages: [assistant([{ type: "toolCall", id: "x", name: "read", arguments: {} }])],
		});

		expect(converted).toEqual([
			{ role: "assistant", content: "", tool_calls: [{ function: { name: "read", arguments: {} } }] },
		]);
	});

	it("sends tool results as a tool message without ids, marking omitted images", () => {
		const converted = buildOllamaMessages({
			messages: [
				toolResult([
					{ type: "text", text: "line 1" },
					{ type: "image", data: "aGVsbG8=", mimeType: "image/png" },
				]),
			],
		});

		expect(converted).toEqual([
			{ role: "tool", content: "line 1\n[image: image/png omitted, Ollama tool messages do not accept images]" },
		]);
	});

	it("folds later system messages into the leading one instead of repeating them", () => {
		const converted = buildOllamaMessages({
			systemPrompt: "base",
			messages: [
				user("hi"),
				{ role: "system", content: "mid-conversation instruction", timestamp: 0 },
				user("again"),
			],
		});

		expect(converted).toEqual([
			{ role: "system", content: "base\n\nmid-conversation instruction" },
			{ role: "user", content: "hi" },
			{ role: "user", content: "again" },
		]);
	});
});

describe("toOllamaTools", () => {
	it("maps tool declarations to Ollama function definitions", () => {
		const tools: Tool[] = [
			{
				name: "read",
				description: "Read a file",
				parameters: {
					type: "object",
					properties: { path: { type: "string" } },
					required: ["path"],
				} as Tool["parameters"],
			},
		];

		expect(toOllamaTools(tools)).toEqual([
			{
				type: "function",
				function: {
					name: "read",
					description: "Read a file",
					parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
				},
			},
		]);
	});

	it("returns undefined when there are no tools", () => {
		expect(toOllamaTools([])).toBeUndefined();
	});
});

describe("ollamaToolCallToPi", () => {
	it("synthesizes unique ids and keeps arguments", () => {
		expect(ollamaToolCallToPi({ function: { name: "read", arguments: { path: "a.ts" } } }, 0)).toEqual({
			type: "toolCall",
			id: "ollama_call_0",
			name: "read",
			arguments: { path: "a.ts" },
		});
		expect(ollamaToolCallToPi({ function: { name: "read" } }, 1)).toEqual({
			type: "toolCall",
			id: "ollama_call_1",
			name: "read",
			arguments: {},
		});
	});
});
