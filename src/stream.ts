/** Native `/api/chat` stream: Ollama NDJSON chunks translated into Pi assistant events. */

import {
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Model,
	type ModelThinkingLevel,
	type SimpleStreamOptions,
	type TextContent,
	type ThinkingContent,
	type ToolCall,
	type TranscriptContext,
	type Usage,
	calculateCost,
	createAssistantMessageEventStream,
	getCurrentSystemPrompt,
	getCurrentTools,
	parseToolCallArguments,
} from "@earendil-works/pi-ai";
import { createClient } from "./client.ts";
import { OllamaError } from "./errors.ts";
import type { OllamaChatChunk, OllamaChatRequest, OllamaToolCall } from "./wire.ts";
import { buildOllamaMessages, ollamaToolCallToPi, toOllamaTools } from "./convert.ts";
import { transformMessages } from "../vendor/pi-ai/api/transform-messages.js";

/** The custom API id this adapter implements, used by the registered providers. */
export const OLLAMA_CHAT_API = "ollama-chat";

/** Which Ollama server a request targets; only the local server allocates a context window. */
export type OllamaChatConfig = { kind: "local" | "cloud" };

/**
 * `SimpleStreamOptions` with the thinking level widened to `ModelThinkingLevel`: unlike the
 * built-in adapters, this one does not clamp the level, so it also honors an explicit
 * `off` and omits `think` for a level the model does not offer.
 */
export type OllamaStreamOptions = Omit<SimpleStreamOptions, "reasoning"> & {
	reasoning?: ModelThinkingLevel;
};

function createAssistantMessage(model: Model<"ollama-chat">): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: OLLAMA_CHAT_API,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "pending",
		timestamp: Date.now(),
	};
}

/**
 * Maps a Pi thinking level to Ollama's `think` field through the model's own
 * `thinkingLevelMap`, so the value survives a restart from the persisted catalog:
 * `off` becomes boolean `false` when the model supports it, an enabled level becomes the
 * mapped name verbatim, and the string `"true"` (a boolean-only model) becomes boolean
 * `true`. A `null`, missing or absent map entry omits the field.
 */
function resolveThink(
	model: Model<"ollama-chat">,
	reasoning: ModelThinkingLevel | undefined,
): boolean | string | undefined {
	const map = model.thinkingLevelMap;
	if (!map || reasoning === undefined) return undefined;
	if (reasoning === "off") return map.off != null ? false : undefined;
	const mapped = map[reasoning];
	if (mapped == null) return undefined;
	return mapped === "true" ? true : mapped;
}

function buildChatRequest(
	model: Model<"ollama-chat">,
	context: TranscriptContext,
	options: OllamaStreamOptions | undefined,
	config: OllamaChatConfig | undefined,
): OllamaChatRequest {
	const chatOptions: Record<string, unknown> = {};
	// Cloud serves its own fixed context, so only the local server is asked for one.
	if (config?.kind === "local") chatOptions.num_ctx = model.contextWindow;
	if (options?.maxTokens !== undefined) chatOptions.num_predict = options.maxTokens;
	if (options?.temperature !== undefined) chatOptions.temperature = options.temperature;
	// `samplingParams` is merged last, so a caller can override any named option above.
	Object.assign(chatOptions, options?.samplingParams ?? {});

	const think = resolveThink(model, options?.reasoning);
	const tools = toOllamaTools(getCurrentTools(context.messages));

	// Every Pi adapter runs the transcript through `transformMessages` before converting: errored
	// and aborted turns are dropped, orphaned tool calls get a synthetic result, and images a
	// model cannot read are downgraded to placeholders. Without it, an Esc after a tool call
	// replays an assistant `tool_calls` with no `tool` behind it — an invalid sequence — and a
	// text-only model receives `images` it rejects.
	const messages = transformMessages([...context.messages], model);

	return {
		model: model.id,
		messages: buildOllamaMessages({
			systemPrompt: getCurrentSystemPrompt(messages),
			// The prompt above already replays every system message, so the conversion only
			// receives the conversation: handing it the system messages again would send the
			// prompt twice.
			messages: messages.filter((message) => message.role !== "system"),
		}),
		...(think !== undefined && { think }),
		...(Object.keys(chatOptions).length > 0 && { options: chatOptions }),
		...(tools !== undefined && { tools }),
	};
}

function usageOf(chunk: OllamaChatChunk | undefined, model: Model<"ollama-chat">): Usage {
	// Pi's contract: `input` counts only the uncached prompt tokens and is disjoint from
	// `cacheRead`. `prompt_eval_count` includes the cached prefix, so subtracting here keeps
	// `overflow` (`input + cacheRead`) and `calculateCost` from counting that prefix twice.
	const promptTotal = chunk?.prompt_eval_count ?? 0;
	const cacheRead = chunk?.prompt_eval_cached_count ?? 0;
	const input = Math.max(0, promptTotal - cacheRead);
	const output = chunk?.eval_count ?? 0;
	const usage: Usage = {
		input,
		output,
		cacheRead,
		cacheWrite: 0,
		totalTokens: promptTotal + output,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
	usage.cost = calculateCost(model, usage);
	return usage;
}

/**
 * Ollama reports failures as `{"error": ...}` inside a 200 stream, so the provider text is
 * kept verbatim: Pi matches its context-overflow patterns against `errorMessage`.
 */
function errorText(error: unknown, aborted: boolean): string {
	if (aborted) return "Request was aborted";
	if (error instanceof OllamaError) return error.message;
	if (error instanceof Error) return error.message;
	return String(error);
}

/** Notice appended to the assistant text when the server sends a tool call Pi cannot run. */
const DROPPED_TOOL_CALL_NOTICE = "[dropped a malformed tool call from the server]";

/** One tool call as it accumulates across chunks; see {@link mergeToolCall}. */
interface BufferedToolCall {
	name: string;
	arguments: Record<string, unknown>;
	/** Set when an arriving fragment broke a rule; a later chunk cannot repair the call. */
	malformed: boolean;
}

/**
 * Reads one fragment's `arguments` as an object. Ollama sends an object, but a partial call or an
 * intermediary can send the JSON text instead, which is exactly the case Pi's own
 * `parseToolCallArguments` handles: anything that is not a JSON object is rejected, and its repair
 * step also rescues the control characters a model sometimes leaves inside a string.
 */
function argumentsOf(value: unknown): { ok: true; value: Record<string, unknown> } | { ok: false } {
	if (value === undefined || value === null) return { ok: true, value: {} };
	if (typeof value === "object" && !Array.isArray(value)) {
		return { ok: true, value: value as Record<string, unknown> };
	}
	if (typeof value !== "string") return { ok: false };
	const parsed = parseToolCallArguments(value);
	return parsed.argumentsError === undefined
		? { ok: true, value: parsed.arguments as Record<string, unknown> }
		: { ok: false };
}

/**
 * Merges one arriving fragment into the call buffered at the same position. A non-empty name wins
 * over an empty one and arguments merge shallowly, because Ollama may split one call across
 * chunks. A fragment that breaks a rule poisons the whole call: dropping it is safer than handing
 * `validateToolArguments` a call that throws after the turn's text already reached the user.
 */
function mergeToolCall(into: BufferedToolCall | undefined, fragment: OllamaToolCall): BufferedToolCall {
	const merged = into ?? { name: "", arguments: {}, malformed: false };
	const parts = fragment as { function?: { name?: unknown; arguments?: unknown } };
	const name = parts.function?.name;
	if (typeof name === "string" && name.trim().length > 0) merged.name = name;
	const incoming = argumentsOf(parts.function?.arguments);
	if (incoming.ok) merged.arguments = { ...merged.arguments, ...incoming.value };
	else merged.malformed = true;
	return merged;
}

/**
 * Streams one Ollama chat completion as Pi assistant events.
 *
 * Text and thinking are streamed as they arrive. Tool calls are buffered until the stream's
 * final chunk, because Ollama documents `message.tool_calls` as "partial tool calls": a name
 * can arrive in one chunk and its arguments in another, and emitting per chunk would hand Pi
 * two broken calls instead of one. Each runnable call then emits `toolcall_start` immediately
 * followed by `toolcall_end` and no deltas, since Ollama never streams argument fragments.
 * Every request-shaping fact (context window, thinking levels) is read from the model itself.
 * Failures never throw: they terminate the stream with one `error` event.
 */
export function streamOllamaChat(
	model: Model<"ollama-chat">,
	context: TranscriptContext,
	options?: OllamaStreamOptions,
	config?: OllamaChatConfig,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();
	const output = createAssistantMessage(model);
	let textBlock: TextContent | undefined;
	let thinkingBlock: ThinkingContent | undefined;

	const closeText = (): void => {
		if (!textBlock) return;
		stream.push({
			type: "text_end",
			contentIndex: output.content.indexOf(textBlock),
			content: textBlock.text,
			partial: output,
		});
		textBlock = undefined;
	};
	const closeThinking = (): void => {
		if (!thinkingBlock) return;
		stream.push({
			type: "thinking_end",
			contentIndex: output.content.indexOf(thinkingBlock),
			content: thinkingBlock.thinking,
			partial: output,
		});
		thinkingBlock = undefined;
	};
	const closeBlocks = (): void => {
		closeThinking();
		closeText();
	};

	void (async () => {
		try {
			const payload = buildChatRequest(model, context, options, config);
			const replacement = await options?.onPayload?.(payload, model);
			const request = (replacement ?? payload) as OllamaChatRequest;
			const client = createClient(
				model.baseUrl,
				{ apiKey: options?.apiKey },
				{
					fetch: options?.fetch,
					// Pi passes its provider request timeout here; the client applies it from the
					// request send through the headers and then as the idle gap between chunks, so a
					// hung or stalled server cannot hold the stream forever.
					timeoutMs: options?.timeoutMs,
					// The headers Pi computed for this request: session, auth, caller and
					// `transformHeaders` values. Ollama ignores most of them, but a corporate proxy and a
					// caller's `transformHeaders` do not, and `null` suppresses an adapter default.
					headers: options?.headers,
					maxRetries: options?.maxRetries,
					maxRetryDelayMs: options?.maxRetryDelayMs,
					onResponse: options?.onResponse
						? (response) => options.onResponse?.(response, model)
						: undefined,
				},
			);
			// `options.identity` (Pi's `User-Agent` and attribution headers) is deliberately not
			// forwarded: building those headers needs Pi's `utils/identity.js` and `utils/version.js`,
			// neither exported from the package root, so honoring it would open a second vendoring
			// surface for a header Ollama ignores.

			stream.push({ type: "start", partial: output });

			let finalChunk: OllamaChatChunk | undefined;
			let toolCallCount = 0;
			/** Calls accumulate here by arrival position and are emitted together at `done`. */
			const toolCalls = new Map<number, BufferedToolCall>();
			const flushToolCalls = (): void => {
				const runnable: ToolCall[] = [];
				let dropped = 0;
				for (const call of toolCalls.values()) {
					if (call.malformed || call.name.length === 0) {
						dropped += 1;
						continue;
					}
					runnable.push(
						ollamaToolCallToPi({ function: { name: call.name, arguments: call.arguments } }, toolCallCount),
					);
					toolCallCount += 1;
				}
				if (dropped > 0) {
					// Degrade visibly instead of failing the turn: the text above may already be the answer
					// the user is reading, and a call Pi cannot validate would throw mid-turn.
					if (!textBlock) {
						closeThinking();
						textBlock = { type: "text", text: "" };
						output.content.push(textBlock);
						stream.push({ type: "text_start", contentIndex: output.content.length - 1, partial: output });
					}
					const notice = Array.from({ length: dropped }, () => DROPPED_TOOL_CALL_NOTICE).join("\n");
					const delta = textBlock.text.length > 0 ? `\n${notice}` : notice;
					textBlock.text += delta;
					stream.push({
						type: "text_delta",
						contentIndex: output.content.indexOf(textBlock),
						delta,
						partial: output,
					});
				}
				closeBlocks();
				for (const toolCall of runnable) {
					output.content.push(toolCall);
					const contentIndex = output.content.length - 1;
					stream.push({ type: "toolcall_start", contentIndex, partial: output });
					stream.push({ type: "toolcall_end", contentIndex, toolCall, partial: output });
				}
			};
			for await (const chunk of client.chat(request, options?.signal)) {
				const incoming = chunk.message?.thinking;
				if (incoming) {
					if (!thinkingBlock) {
						closeText();
						thinkingBlock = { type: "thinking", thinking: "" };
						output.content.push(thinkingBlock);
						stream.push({
							type: "thinking_start",
							contentIndex: output.content.length - 1,
							partial: output,
						});
					}
					thinkingBlock.thinking += incoming;
					stream.push({
						type: "thinking_delta",
						contentIndex: output.content.length - 1,
						delta: incoming,
						partial: output,
					});
				}

				const text = chunk.message?.content;
				if (text) {
					if (!textBlock) {
						closeThinking();
						textBlock = { type: "text", text: "" };
						output.content.push(textBlock);
						stream.push({ type: "text_start", contentIndex: output.content.length - 1, partial: output });
					}
					textBlock.text += text;
					stream.push({
						type: "text_delta",
						contentIndex: output.content.length - 1,
						delta: text,
						partial: output,
					});
				}

				const calls = chunk.message?.tool_calls;
				if (calls && calls.length > 0) {
					// Buffer by position, never emit here: a name and its arguments can arrive in different
					// chunks, and emitting per chunk would give Pi two calls it cannot run.
					calls.forEach((call, index) => {
						toolCalls.set(index, mergeToolCall(toolCalls.get(index), call));
					});
				}

				if (chunk.done) {
					finalChunk = chunk;
					break;
				}
			}

			if (!finalChunk) {
				// A stream that closes without `done` is a dropped connection, not a completion.
				// Phrased to match Pi's retryable "ended without" pattern, so a dropped connection
				// is retried instead of surfacing as a dead turn.
				throw new OllamaError("stream", "Ollama stream ended without a completion chunk");
			}

			// Flushes the buffered calls (and closes the open blocks) before the reason is computed.
			flushToolCalls();
			const hasToolCall = output.content.some((block) => block.type === "toolCall");
			const reason: "stop" | "length" | "toolUse" =
				finalChunk?.done_reason === "length" ? "length" : hasToolCall ? "toolUse" : "stop";
			output.usage = usageOf(finalChunk, model);
			output.stopReason = reason;
			stream.push({ type: "done", reason, message: output });
			stream.end();
		} catch (error) {
			// A failed turn keeps the text it already streamed but drops any buffered tool call: Pi
			// discards an errored assistant turn, and a call from a broken stream must never be runnable.
			closeBlocks();
			const aborted =
				options?.signal?.aborted === true || (error instanceof Error && error.name === "AbortError");
			output.stopReason = aborted ? "aborted" : "error";
			output.errorMessage = errorText(error, aborted);
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end();
		}
	})();

	return stream;
}
