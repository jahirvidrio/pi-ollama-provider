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

	return {
		model: model.id,
		messages: buildOllamaMessages({
			systemPrompt: getCurrentSystemPrompt(context.messages),
			// The prompt above already replays every system message, so the conversion only
			// receives the conversation: handing it the system messages again would send the
			// prompt twice.
			messages: context.messages.filter((message) => message.role !== "system"),
		}),
		...(think !== undefined && { think }),
		...(Object.keys(chatOptions).length > 0 && { options: chatOptions }),
		...(tools !== undefined && { tools }),
	};
}

function usageOf(chunk: OllamaChatChunk | undefined, model: Model<"ollama-chat">): Usage {
	const input = chunk?.prompt_eval_count ?? 0;
	const output = chunk?.eval_count ?? 0;
	const cacheRead = chunk?.prompt_eval_cached_count ?? 0;
	const usage: Usage = {
		input,
		output,
		cacheRead,
		cacheWrite: 0,
		// `prompt_eval_count` already counts the cached prompt tokens, so `cacheRead` is a
		// subset of `input` and must not be added to the total a second time.
		totalTokens: input + output,
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
				throw new OllamaError("stream", "Ollama closed the chat stream before sending a completion chunk");
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
