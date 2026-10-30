/** Native `/api/chat` stream: Ollama NDJSON chunks translated into Pi assistant events. */

import {
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Model,
	type ModelThinkingLevel,
	type SimpleStreamOptions,
	type TextContent,
	type ThinkingContent,
	type TranscriptContext,
	type Usage,
	calculateCost,
	createAssistantMessageEventStream,
	getCurrentSystemPrompt,
	getCurrentTools,
} from "@earendil-works/pi-ai";
import { OllamaClient, OllamaError, type OllamaChatChunk, type OllamaChatRequest } from "./client.ts";
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

/**
 * Streams one Ollama chat completion as Pi assistant events.
 *
 * Text and thinking are streamed as they arrive; Ollama delivers tool calls already
 * complete (arguments are an object), so each call emits `toolcall_start` immediately
 * followed by `toolcall_end` and no deltas. Every request-shaping fact (context window,
 * thinking levels) is read from the model itself. Failures never throw: they terminate
 * the stream with one `error` event.
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
			const client = new OllamaClient({
				baseUrl: model.baseUrl,
				apiKey: options?.apiKey,
				fetch: options?.fetch,
				// Pi passes its provider request timeout here; the client applies it from the
				// request send through the headers and then as the idle gap between chunks, so a
				// hung or stalled server cannot hold the stream forever.
				timeoutMs: options?.timeoutMs,
				onResponse: options?.onResponse
					? (response) => options.onResponse?.(response, model)
					: undefined,
			});

			stream.push({ type: "start", partial: output });

			let finalChunk: OllamaChatChunk | undefined;
			let toolCallCount = 0;
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
					closeBlocks();
					for (const call of calls) {
						const toolCall = ollamaToolCallToPi(call, toolCallCount);
						toolCallCount += 1;
						output.content.push(toolCall);
						const contentIndex = output.content.length - 1;
						stream.push({ type: "toolcall_start", contentIndex, partial: output });
						stream.push({ type: "toolcall_end", contentIndex, toolCall, partial: output });
					}
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

			closeBlocks();
			const hasToolCall = output.content.some((block) => block.type === "toolCall");
			const reason: "stop" | "length" | "toolUse" =
				finalChunk?.done_reason === "length" ? "length" : hasToolCall ? "toolUse" : "stop";
			output.usage = usageOf(finalChunk, model);
			output.stopReason = reason;
			stream.push({ type: "done", reason, message: output });
			stream.end();
		} catch (error) {
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
