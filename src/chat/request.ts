/** One Pi transcript turned into an `/api/chat` request body. */

import {
  type Model,
  type ModelThinkingLevel,
  type TranscriptContext,
  getCurrentSystemPrompt,
  getCurrentTools,
} from '@earendil-works/pi-ai';
import { transformMessages } from '../../vendor/pi-ai/api/transform-messages.js';
import type { OllamaChatRequest } from '../wire.ts';
import { buildOllamaMessages, toOllamaTools } from './convert.ts';
import type { OllamaChatConfig, OllamaStreamOptions } from './options.ts';

/**
 * Maps a Pi thinking level to Ollama's `think` field through the model's own
 * `thinkingLevelMap`, so the value survives a restart from the persisted catalog:
 * `off` becomes boolean `false` when the model supports it, an enabled level becomes the
 * mapped name verbatim, and the string `"true"` (a boolean-only model) becomes boolean
 * `true`. A `null`, missing or absent map entry omits the field.
 */
function resolveThink (
  model: Model<'ollama-chat'>,
  reasoning: ModelThinkingLevel | undefined
): boolean | string | undefined {
  const map = model.thinkingLevelMap;
  if (!map || reasoning === undefined) return undefined;
  if (reasoning === 'off') return map.off != null ? false : undefined;
  const mapped = map[reasoning];
  if (mapped == null) return undefined;
  return mapped === 'true' ? true : mapped;
}

export function buildChatRequest (
  model: Model<'ollama-chat'>,
  context: TranscriptContext,
  options: OllamaStreamOptions | undefined,
  config: OllamaChatConfig | undefined
): OllamaChatRequest {
  const chatOptions: Record<string, unknown> = {};
  // Cloud serves its own fixed context, so only the local server is asked for one.
  if (config?.kind === 'local') chatOptions.num_ctx = model.contextWindow;
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

  const request: OllamaChatRequest = {
    model: model.id,
    messages: buildOllamaMessages({
      systemPrompt: getCurrentSystemPrompt(messages),
      // The prompt above already replays every system message, so the conversion only
      // receives the conversation: handing it the system messages again would send the
      // prompt twice.
      messages: messages.filter((message) => message.role !== 'system'),
    }),
  };
  if (think !== undefined) request.think = think;
  if (Object.keys(chatOptions).length > 0) request.options = chatOptions;
  if (tools !== undefined) request.tools = tools;
  return request;
}
