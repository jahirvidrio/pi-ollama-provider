import { describe, expect, it } from 'vitest';
import {
  type AssistantMessageEvent,
  type Model,
  normalizeContext,
} from '@earendil-works/pi-ai';
import { OllamaClient } from '../src/client/index.ts';
import { OllamaError } from '../src/errors.ts';
import { streamOllamaChat } from '../src/chat/index.ts';
import { startFakeOllama } from './helpers/fake-ollama.ts';

const REQUEST = { model: 'gemma4:31b', messages: [{ role: 'user' as const, content: 'hi' }] };

const DONE_CHUNK = [{ message: { role: 'assistant', content: 'ok' }, done: true, done_reason: 'stop' }];

function model (overrides: Partial<Model<'ollama-chat'>> = {}): Model<'ollama-chat'> {
  return {
    id: 'gemma4:31b',
    name: 'gemma4:31b',
    api: 'ollama-chat',
    provider: 'ollama',
    baseUrl: 'http://127.0.0.1:1/api',
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 131072,
    maxTokens: 8192,
    ...overrides,
  };
}

function context () {
  return normalizeContext({
    systemPrompt: 'You are Pi.',
    messages: [{ role: 'user', content: 'hi', timestamp: 0 }],
  });
}

async function collect (stream: AsyncIterable<AssistantMessageEvent>): Promise<AssistantMessageEvent[]> {
  const events: AssistantMessageEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

async function failure (promise: Promise<unknown>): Promise<unknown> {
  return promise.catch((cause: unknown) => cause);
}

describe('provider retries', () => {
  it('retries a rate-limited request and reports the response once', async () => {
    const server = await startFakeOllama({
      chat: { status: 429, body: { error: 'too many requests' }, headers: { 'retry-after': '0' } },
    });
    server.respond({ chunks: DONE_CHUNK });
    try {
      let responses = 0;
      const client = new OllamaClient({
        baseUrl: server.baseUrl,
        onResponse: () => {
          responses += 1;
        },
      });

      const contents: string[] = [];
      for await (const chunk of client.chat(REQUEST)) contents.push(chunk.message?.content ?? '');

      expect(contents).toEqual(['ok']);
      expect(server.requests).toHaveLength(2);
      expect(responses).toBe(1);
    } finally {
      await server.stop();
    }
  });

  it("retries up to Pi's default budget of two attempts", async () => {
    const server = await startFakeOllama({
      chat: { status: 503, body: { error: 'unavailable' }, headers: { 'retry-after': '0' } },
    });
    server.respond({ status: 503, body: { error: 'unavailable' }, headers: { 'retry-after': '0' } });
    server.respond({ chunks: DONE_CHUNK });
    try {
      const contents: string[] = [];
      for await (const chunk of new OllamaClient({ baseUrl: server.baseUrl }).chat(REQUEST)) {
        contents.push(chunk.message?.content ?? '');
      }

      expect(contents).toEqual(['ok']);
      expect(server.requests).toHaveLength(3);
    } finally {
      await server.stop();
    }
  });

  it('fails at once with maxRetries: 0 and keeps the response headers on the error', async () => {
    const server = await startFakeOllama({
      chat: { status: 503, body: { error: 'unavailable' }, headers: { 'retry-after': '7' } },
    });
    try {
      const error = await failure(
        new OllamaClient({ baseUrl: server.baseUrl, maxRetries: 0 }).chat(REQUEST).next()
      );

      expect(server.requests).toHaveLength(1);
      expect(error).toBeInstanceOf(OllamaError);
      expect(error).toMatchObject({ kind: 'http', status: 503 });
      expect((error as OllamaError).headers?.get('retry-after')).toBe('7');
      expect((error as Error).message).toContain('unavailable');
    } finally {
      await server.stop();
    }
  });

  it('does not retry an idle timeout, which is already bounded by timeoutMs', async () => {
    const server = await startFakeOllama({ chat: { hangHeaders: true } });
    try {
      const error = await failure(
        new OllamaClient({ baseUrl: server.baseUrl, timeoutMs: 120 }).chat(REQUEST).next()
      );

      expect(server.requests).toHaveLength(1);
      expect(error).toBeInstanceOf(OllamaError);
      expect((error as OllamaError).status).toBe(408);
      expect((error as Error).message).toContain('timed out');
    } finally {
      await server.stop();
    }
  }, 10_000);

  it('retries a connection failure and reports it as a network error once exhausted', async () => {
    let attempts = 0;
    const client = new OllamaClient({
      baseUrl: 'http://127.0.0.1:1/api',
      maxRetries: 1,
      fetch: async () => {
        attempts += 1;
        throw new Error('connection failed');
      },
    });

    const error = await failure(client.chat(REQUEST).next());

    expect(attempts).toBe(2);
    expect(error).toBeInstanceOf(OllamaError);
    expect(error).toMatchObject({ kind: 'network' });
    expect((error as Error).message).toContain('connection failed');
  });
});

describe('request headers', () => {
  it("sends Pi's headers and lets null suppress a default one", async () => {
    const server = await startFakeOllama({ chat: { chunks: DONE_CHUNK } });
    try {
      await collect(
        streamOllamaChat(model({ baseUrl: server.baseUrl }), context(), {
          apiKey: 'secret',
          headers: { 'x-trace': '1', authorization: null },
        })
      );

      const headers = server.requests[0]?.headers ?? {};
      expect(headers['x-trace']).toBe('1');
      expect(headers.authorization).toBeUndefined();
      expect(headers['content-type']).toBe('application/json');
    } finally {
      await server.stop();
    }
  });

  it("honors Pi's retry budget for a stream, including zero", async () => {
    const noRetries = await startFakeOllama({ chat: { status: 503, body: { error: 'unavailable' } } });
    try {
      const events = await collect(
        streamOllamaChat(model({ baseUrl: noRetries.baseUrl }), context(), { maxRetries: 0 })
      );
      expect(events.at(-1)?.type).toBe('error');
      expect(noRetries.requests).toHaveLength(1);
    } finally {
      await noRetries.stop();
    }

    const retried = await startFakeOllama({
      chat: { status: 503, body: { error: 'unavailable' }, headers: { 'retry-after': '0' } },
    });
    retried.respond({ chunks: DONE_CHUNK });
    try {
      const events = await collect(streamOllamaChat(model({ baseUrl: retried.baseUrl }), context()));
      expect(events.at(-1)).toMatchObject({ type: 'done' });
      expect(retried.requests).toHaveLength(2);
    } finally {
      await retried.stop();
    }
  });

  it('sends the configured headers on catalog requests', async () => {
    const server = await startFakeOllama({ tags: { models: [] } });
    try {
      await new OllamaClient({ baseUrl: server.baseUrl, headers: { 'x-trace': '1' } }).listModels();

      expect(server.requests[0]?.path).toBe('/api/tags');
      expect(server.requests[0]?.headers['x-trace']).toBe('1');
    } finally {
      await server.stop();
    }
  });
});

describe('stream body failures', () => {
  it('does not retry a request whose body failed after the response was established', async () => {
    let requests = 0;
    let pulls = 0;
    const encoder = new TextEncoder();
    const client = new OllamaClient({
      baseUrl: 'http://127.0.0.1:1/api',
      maxRetries: 2,
      fetch: async () => {
        requests += 1;
        const body = new ReadableStream<Uint8Array>({
          pull (controller) {
            pulls += 1;
            if (pulls === 1) {
              controller.enqueue(
                encoder.encode(`${JSON.stringify({ message: { content: 'ok' }, done: false })}\n`)
              );
              return;
            }
            controller.error(new Error('socket died'));
          },
        });
        return new Response(body, { status: 200, headers: { 'content-type': 'application/x-ndjson' } });
      },
    });

    const iterator = client.chat(REQUEST);
    const first = await iterator.next();
    const error = await iterator.next().catch((cause: unknown) => cause);

    expect(first.value).toMatchObject({ message: { content: 'ok' } });
    // The text above already reached the caller, so a second request would replay it.
    expect(error).toBeInstanceOf(OllamaError);
    expect((error as OllamaError).kind).toBe('stream');
    expect((error as Error).message).toContain('socket died');
    expect(requests).toBe(1);
  });
});

describe('server-requested retry delays', () => {
  // The vendored helper fails fast when the server asks for more than the retry budget. What it
  // must not do is substitute its own text for the provider's: the status, the headers and the
  // server's own words are what Pi's retry policy and the user need.
  it("reports the provider's error when the server asks to wait past the retry budget", async () => {
    const server = await startFakeOllama({
      chat: { status: 429, headers: { 'retry-after': '90' }, body: { error: 'rate limited' } },
    });
    try {
      const error = await failure(new OllamaClient({ baseUrl: server.baseUrl }).chat(REQUEST).next());

      expect(error).toBeInstanceOf(OllamaError);
      expect((error as OllamaError).kind).toBe('http');
      expect((error as OllamaError).status).toBe(429);
      expect((error as OllamaError).headers?.get('retry-after')).toBe('90');
      expect((error as Error).message).toBe('429: rate limited (the server asked to retry after 90s)');
      // Still one attempt: the delay is reported, never silently shortened.
      expect(server.requests).toHaveLength(1);
    } finally {
      await server.stop();
    }
  });

  it('keeps the status-only message when the response carries no retry-after', async () => {
    const server = await startFakeOllama({ chat: { status: 400, body: { error: 'bad request' } } });
    try {
      const error = await failure(new OllamaClient({ baseUrl: server.baseUrl }).chat(REQUEST).next());

      expect((error as Error).message).toBe('400: bad request');
      expect(server.requests).toHaveLength(1);
    } finally {
      await server.stop();
    }
  });

  it("keeps the caller's abort when it interrupts the retry backoff", async () => {
    const server = await startFakeOllama({
      chat: { status: 429, headers: { 'retry-after': '0.4' }, body: { error: 'rate limited' } },
    });
    try {
      const controller = new AbortController();
      const pending = failure(new OllamaClient({ baseUrl: server.baseUrl }).chat(REQUEST, controller.signal).next());
      setTimeout(() => controller.abort(), 50);

      const error = (await pending) as Error;
      expect(error.name).toBe('AbortError');
      expect(server.requests).toHaveLength(1);
    } finally {
      await server.stop();
    }
  });
});
