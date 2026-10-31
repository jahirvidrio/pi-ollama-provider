/** Native `/api/chat` stream: Ollama NDJSON chunks translated into Pi assistant events. */

import {
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Model,
  type TranscriptContext,
  createAssistantMessageEventStream,
} from '@earendil-works/pi-ai';
import { createClient } from '../client/index.ts';
import { OllamaError } from '../errors.ts';
import { OLLAMA_CHAT_API, type OllamaChatChunk, type OllamaChatRequest } from '../wire.ts';
import type { OllamaChatConfig, OllamaStreamOptions } from './options.ts';
import { buildChatRequest } from './request.ts';
import { TurnWriter } from './writer.ts';

function createAssistantMessage (model: Model<'ollama-chat'>): AssistantMessage {
  return {
    role: 'assistant',
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
    stopReason: 'pending',
    timestamp: Date.now(),
  };
}

/**
 * Streams one Ollama chat completion as Pi assistant events.
 *
 * Text and thinking are streamed as they arrive. Tool calls are buffered until the stream's
 * final chunk, because Ollama documents `message.tool_calls` as "partial tool calls": a name
 * can arrive in one chunk and its arguments in another, and emitting per chunk would hand Pi
 * two broken calls instead of one. Every request-shaping fact (context window, thinking levels)
 * is read from the model itself. Failures never throw: they terminate the stream with one
 * `error` event.
 */
export function streamOllamaChat (
  model: Model<'ollama-chat'>,
  context: TranscriptContext,
  options?: OllamaStreamOptions,
  config?: OllamaChatConfig
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();
  const writer = new TurnWriter(stream, createAssistantMessage(model));

  // eslint-disable-next-line no-void -- the task reports failures through the stream, never by rejecting
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
          // Pi passes its provider request timeout here; the client applies it from the request
          // send through the headers and then as the idle gap between chunks, so a hung or stalled
          // server cannot hold the stream forever.
          timeoutMs: options?.timeoutMs,
          // The headers Pi computed for this request: session, auth, caller and `transformHeaders`
          // values. Ollama ignores most of them, but a corporate proxy and a caller's
          // `transformHeaders` do not, and `null` suppresses an adapter default.
          headers: options?.headers,
          maxRetries: options?.maxRetries,
          maxRetryDelayMs: options?.maxRetryDelayMs,
          onResponse: options?.onResponse
            ? (response) => options.onResponse?.(response, model)
            : undefined,
        }
      );
      // `options.identity` (Pi's `User-Agent` and attribution headers) is deliberately not
      // forwarded: building those headers needs Pi's `utils/identity.js` and `utils/version.js`,
      // neither exported from the package root, so honoring it would open a second vendoring
      // surface for a header Ollama ignores.

      writer.start();

      let finalChunk: OllamaChatChunk | undefined;
      for await (const chunk of client.chat(request, options?.signal)) {
        const thinking = chunk.message?.thinking;
        if (thinking) writer.thinking(thinking);

        const text = chunk.message?.content;
        if (text) writer.text(text);

        const calls = chunk.message?.tool_calls;
        if (calls && calls.length > 0) writer.buffer(calls);

        if (chunk.done) {
          finalChunk = chunk;
          break;
        }
      }

      if (!finalChunk) {
        // A stream that closes without `done` is a dropped connection, not a completion. Phrased to
        // match Pi's retryable "ended without" pattern, so a dropped connection is retried instead
        // of surfacing as a dead turn.
        throw new OllamaError('stream', 'Ollama stream ended without a completion chunk');
      }

      writer.finish(finalChunk, model);
    } catch (error) {
      // A failed turn keeps the text it already streamed but drops any buffered tool call: Pi
      // discards an errored assistant turn, and a call from a broken stream must never be runnable.
      const aborted =
        options?.signal?.aborted === true || (error instanceof Error && error.name === 'AbortError');
      writer.fail(error, aborted);
    }
  })();

  return stream;
}
