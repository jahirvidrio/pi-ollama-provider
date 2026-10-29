/** Converts between Pi's normalized transcript and Ollama's chat wire format. */

import type {
	AssistantMessage,
	ImageContent,
	JsonObject,
	Message,
	SystemMessage,
	TextContent,
	Tool,
	ToolCall,
} from "@earendil-works/pi-ai";
import type { OllamaChatMessage, OllamaToolCall, OllamaToolDefinition } from "./client.ts";

function textOf(blocks: readonly (TextContent | ImageContent)[]): string {
	return blocks
		.filter((block): block is TextContent => block.type === "text")
		.map((block) => block.text)
		.join("");
}

function imagesOf(blocks: readonly (TextContent | ImageContent)[]): string[] {
	return blocks.filter((block): block is ImageContent => block.type === "image").map((block) => block.data);
}

function systemText(message: SystemMessage): string {
	if (typeof message.content === "string") return message.content;
	return message.content.map((block) => block.text).join("");
}

/**
 * Builds the Ollama message list. Ollama has no documented mid-conversation system
 * support, so every system message is folded into the leading one, the same shape Pi's
 * own `collapseSystemMessages()` produces.
 */
export function buildOllamaMessages(input: {
	systemPrompt?: string;
	messages: readonly Message[];
}): OllamaChatMessage[] {
	const systemParts: string[] = [];
	const prompt = input.systemPrompt?.trim();
	if (prompt) systemParts.push(prompt);

	const converted: OllamaChatMessage[] = [];
	for (const message of input.messages) {
		switch (message.role) {
			case "system": {
				const text = systemText(message).trim();
				if (text) systemParts.push(text);
				break;
			}
			case "user": {
				let text: string;
				let images: string[] = [];
				if (typeof message.content === "string") {
					text = message.content;
				} else {
					text = textOf(message.content);
					images = imagesOf(message.content);
				}
				converted.push({ role: "user", content: text, ...(images.length > 0 && { images }) });
				break;
			}
			case "assistant": {
				converted.push(toAssistantMessage(message));
				break;
			}
			case "toolResult": {
				const parts = [textOf(message.content)];
				for (const block of message.content) {
					if (block.type === "image") {
						parts.push(`[image: ${block.mimeType} omitted, Ollama tool messages do not accept images]`);
					}
				}
				converted.push({ role: "tool", content: parts.join("\n") });
				break;
			}
		}
	}

	if (systemParts.length > 0) {
		converted.unshift({ role: "system", content: systemParts.join("\n\n") });
	}
	return converted;
}

function toAssistantMessage(message: AssistantMessage): OllamaChatMessage {
	const text = message.content
		.filter((block): block is TextContent => block.type === "text")
		.map((block) => block.text)
		.join("");
	const thinking = message.content
		.filter((block) => block.type === "thinking")
		.map((block) => block.thinking)
		.join("");
	const toolCalls = message.content
		.filter((block): block is ToolCall => block.type === "toolCall")
		.map((call) => ({ function: { name: call.name, arguments: call.arguments } }));

	return {
		role: "assistant",
		content: text,
		...(thinking.length > 0 && { thinking }),
		...(toolCalls.length > 0 && { tool_calls: toolCalls }),
	};
}

/** Ollama declares tools as JSON Schema functions. */
export function toOllamaTools(tools: readonly Tool[]): OllamaToolDefinition[] | undefined {
	if (tools.length === 0) return undefined;
	return tools.map((tool) => ({
		type: "function",
		function: {
			name: tool.name,
			description: tool.description,
			parameters: tool.parameters as unknown as Record<string, unknown>,
		},
	}));
}

/**
 * Ollama returns tool calls without ids and expects tool results positionally, so Pi's
 * ids are synthesized. `index` must be the call's position within the assistant message
 * to keep the ids stable and unique.
 */
export function ollamaToolCallToPi(call: OllamaToolCall, index: number): ToolCall {
	return {
		type: "toolCall",
		id: `ollama_call_${index}`,
		name: call.function.name,
		arguments: (call.function.arguments ?? {}) as JsonObject,
	};
}
