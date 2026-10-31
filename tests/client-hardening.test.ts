import { getEventListeners } from 'node:events';
import { describe, expect, it } from 'vitest';
import { OllamaClient } from '../src/client/index.ts';
import { OllamaError } from '../src/errors.ts';
import { startFakeOllama } from './helpers/fake-ollama.ts';

const REQUEST = { model: 'gemma4:31b', messages: [{ role: 'user' as const, content: 'hi' }] };

// These tests prove a deadline fired by the error it produced and by their own test timeout
// catching a hang, never by a wall-clock comparison: `elapsed < 2_000` measures the machine, not
// the code, and fails on a loaded box while telling nothing about the adapter.

describe('request timeout', () => {
  it('aborts a stream that stalls after the headers, and names the timeout', async () => {
    const server = await startFakeOllama({
      chat: { chunks: [{ message: { role: 'assistant', content: 'partial' }, done: false }], keepOpen: true },
    });
    try {
      const client = new OllamaClient({ baseUrl: server.baseUrl, timeoutMs: 150 });
      const iterator = client.chat(REQUEST);

      const first = await iterator.next();
      expect(first.value).toMatchObject({ message: { content: 'partial' } });

      const error = await iterator.next().catch((cause: unknown) => cause);

      expect(error).toBeInstanceOf(OllamaError);
      expect((error as OllamaError).message).toContain('timed out');
    } finally {
      await server.stop();
    }
  }, 10_000);

  it('aborts a request whose headers never arrive, and names the timeout', async () => {
    const server = await startFakeOllama({ chat: { hangHeaders: true } });
    try {
      const client = new OllamaClient({ baseUrl: server.baseUrl, timeoutMs: 150 });
      const error = await client.chat(REQUEST).next().catch((cause: unknown) => cause);

      expect(error).toBeInstanceOf(OllamaError);
      expect((error as OllamaError).message).toContain('timed out');
    } finally {
      await server.stop();
    }
  }, 10_000);

  it('lets a stream finish when every chunk arrives inside the window', async () => {
    const server = await startFakeOllama({
      chat: {
        chunks: [
          { message: { role: 'assistant', content: 'a' }, done: false },
          { message: { role: 'assistant', content: 'b' }, done: false },
          { message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop' },
        ],
        chunkDelayMs: 50,
      },
    });
    try {
      const contents: string[] = [];
      const client = new OllamaClient({ baseUrl: server.baseUrl, timeoutMs: 150 });
      for await (const chunk of client.chat(REQUEST)) {
        contents.push(chunk.message?.content ?? '');
      }
      expect(contents).toEqual(['a', 'b', '']);
    } finally {
      await server.stop();
    }
  }, 10_000);

  it('does not charge consumer time between chunks against the timeout', async () => {
    const server = await startFakeOllama({
      chat: {
        chunks: [
          { message: { role: 'assistant', content: 'a' }, done: false },
          { message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop' },
        ],
        chunkDelayMs: 250,
      },
    });
    try {
      const client = new OllamaClient({ baseUrl: server.baseUrl, timeoutMs: 150 });
      const iterator = client.chat(REQUEST);
      const first = await iterator.next();
      expect(first.value).toMatchObject({ message: { content: 'a' } });

      await new Promise((resolve) => setTimeout(resolve, 400));

      const second = await iterator.next();
      expect(second.value).toMatchObject({ done: true });
    } finally {
      await server.stop();
    }
  }, 10_000);

  it('does not charge the onResponse hook against the timeout', async () => {
    const server = await startFakeOllama({
      chat: {
        chunks: [
          { message: { role: 'assistant', content: 'a' }, done: false },
          { message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop' },
        ],
      },
    });
    try {
      const client = new OllamaClient({
        baseUrl: server.baseUrl,
        timeoutMs: 150,
        onResponse: async () => {
          await new Promise((resolve) => setTimeout(resolve, 400));
        },
      });
      const contents: string[] = [];
      for await (const chunk of client.chat(REQUEST)) {
        contents.push(chunk.message?.content ?? '');
      }
      expect(contents).toEqual(['a', '']);
    } finally {
      await server.stop();
    }
  }, 10_000);

  it('surfaces the timeout reason when an error body stalls', async () => {
    const server = await startFakeOllama({ chat: { status: 500, hangBody: true } });
    try {
      const client = new OllamaClient({ baseUrl: server.baseUrl, timeoutMs: 150 });
      const error = await client.chat(REQUEST).next().catch((cause: unknown) => cause);

      expect(error).toBeInstanceOf(OllamaError);
      expect((error as OllamaError).kind).toBe('network');
      expect((error as OllamaError).message).toContain('timed out');
    } finally {
      await server.stop();
    }
  }, 10_000);
});

describe('catalog request deadlines', () => {
  it('bounds /api/tags when the response headers never arrive', async () => {
    const server = await startFakeOllama({ stallTags: 'headers' });
    try {
      const client = new OllamaClient({ baseUrl: server.baseUrl, timeoutMs: 150 });
      const error = await client.listModels().catch((cause: unknown) => cause);

      expect(error).toBeInstanceOf(OllamaError);
      expect((error as OllamaError).kind).toBe('network');
      expect((error as Error).message).toContain('/api/tags');
      expect((error as Error).message).toContain('timed out after 150ms');
    } finally {
      await server.stop();
    }
  }, 10_000);

  it('bounds /api/tags when the headers arrive and the body never finishes', async () => {
    const server = await startFakeOllama({ stallTags: 'body' });
    try {
      const client = new OllamaClient({ baseUrl: server.baseUrl, timeoutMs: 150 });
      const error = await client.listModels().catch((cause: unknown) => cause);

      expect(error).toBeInstanceOf(OllamaError);
      expect((error as OllamaError).kind).toBe('network');
      expect((error as Error).message).toContain('timed out after 150ms');
    } finally {
      await server.stop();
    }
  }, 10_000);

  it("keeps the caller's own abort instead of reporting the deadline", async () => {
    const server = await startFakeOllama({ stallTags: 'headers' });
    try {
      const controller = new AbortController();
      const client = new OllamaClient({ baseUrl: server.baseUrl, timeoutMs: 30_000 });
      const pending = client.listModels(controller.signal).catch((cause: unknown) => cause);
      await new Promise((resolve) => setTimeout(resolve, 50));
      controller.abort();

      const error = (await pending) as Error;
      expect(error.name).toBe('AbortError');
      expect(error.message).not.toContain('timed out');
    } finally {
      await server.stop();
    }
  }, 10_000);
});

/** A response whose body fails mid-read: the status arrived, the description did not. */
function unreadableBody (status: number): Response {
  return new Response(
    new ReadableStream({
      start (controller) {
        controller.error(new Error('socket closed'));
      },
    }),
    { status }
  );
}

describe('error body descriptions', () => {
  it('names a catalog error body it could not read', async () => {
    // `json()` reads the body of a non-OK response to describe the failure. When that read
    // itself fails, the message must say so: the status alone leaves "502: " in the transcript
    // and in Pi's error patterns, which is what a user sees at the cloud login prompt.
    const client = new OllamaClient({
      baseUrl: 'http://127.0.0.1:1/api',
      fetch: async () => unreadableBody(502),
    });

    const error = await client.listModels().catch((cause: unknown) => cause);

    expect(error).toBeInstanceOf(OllamaError);
    expect((error as OllamaError).kind).toBe('http');
    expect((error as OllamaError).status).toBe(502);
    expect((error as Error).message).toBe('502: the error body could not be read (socket closed)');
  });

  it('names an empty catalog error body', async () => {
    const client = new OllamaClient({
      baseUrl: 'http://127.0.0.1:1/api',
      fetch: async () => new Response('', { status: 500 }),
    });

    const error = await client.listModels().catch((cause: unknown) => cause);

    expect(error).toBeInstanceOf(OllamaError);
    expect((error as OllamaError).status).toBe(500);
    expect((error as Error).message).toBe('500: the server sent no error body');
  });

  it('names a chat error body it could not read', async () => {
    const client = new OllamaClient({
      baseUrl: 'http://127.0.0.1:1/api',
      maxRetries: 0,
      fetch: async () => unreadableBody(500),
    });

    const error = await client.chat(REQUEST).next().catch((cause: unknown) => cause);

    expect(error).toBeInstanceOf(OllamaError);
    expect((error as OllamaError).kind).toBe('http');
    expect((error as OllamaError).status).toBe(500);
    expect((error as Error).message).toBe('500: the error body could not be read (socket closed)');
  });
});

describe('caller cancellation', () => {
  it('rejects without sending when the caller signal is already aborted', async () => {
    const server = await startFakeOllama();
    try {
      const controller = new AbortController();
      controller.abort();

      const error = await new OllamaClient({ baseUrl: server.baseUrl })
        .chat(REQUEST, controller.signal)
        .next()
        .catch((cause: unknown) => cause);

      expect((error as Error).name).toBe('AbortError');
      expect(server.requests).toHaveLength(0);
    } finally {
      await server.stop();
    }
  });

  it('releases the caller abort listener after a completed chat', async () => {
    const server = await startFakeOllama({
      chat: {
        chunks: [
          { message: { role: 'assistant', content: 'a' }, done: false },
          { message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop' },
        ],
      },
    });
    try {
      const controller = new AbortController();
      for await (const chunk of new OllamaClient({ baseUrl: server.baseUrl }).chat(REQUEST, controller.signal)) {
        expect(chunk).toBeDefined();
      }
      expect(getEventListeners(controller.signal, 'abort')).toEqual([]);
    } finally {
      await server.stop();
    }
  });

  it('releases the caller abort listener when the fetch fails', async () => {
    const controller = new AbortController();
    const client = new OllamaClient({
      baseUrl: 'http://127.0.0.1:1/api',
      fetch: async () => {
        throw new Error('connection failed');
      },
    });
    const error = await client.chat(REQUEST, controller.signal).next().catch((cause: unknown) => cause);

    expect(error).toBeInstanceOf(OllamaError);
    expect((error as OllamaError).kind).toBe('network');
    expect(getEventListeners(controller.signal, 'abort')).toEqual([]);
  });

  it('releases the caller abort listener after an HTTP error', async () => {
    const controller = new AbortController();
    const client = new OllamaClient({
      baseUrl: 'http://127.0.0.1:1/api',
      fetch: async () => new Response('boom', { status: 500 }),
    });
    const error = await client.chat(REQUEST, controller.signal).next().catch((cause: unknown) => cause);

    expect(error).toBeInstanceOf(OllamaError);
    expect((error as OllamaError).kind).toBe('http');
    expect(getEventListeners(controller.signal, 'abort')).toEqual([]);
  });

  it('releases the caller abort listener when the response has no body', async () => {
    const controller = new AbortController();
    const client = new OllamaClient({
      baseUrl: 'http://127.0.0.1:1/api',
      fetch: async () => new Response(null, { status: 200 }),
    });
    const error = await client.chat(REQUEST, controller.signal).next().catch((cause: unknown) => cause);

    expect(error).toBeInstanceOf(OllamaError);
    expect((error as OllamaError).kind).toBe('protocol');
    expect(getEventListeners(controller.signal, 'abort')).toEqual([]);
  });

  it('releases the socket when the consumer stops at the done chunk', async () => {
    const server = await startFakeOllama({
      chat: {
        chunks: [
          { message: { role: 'assistant', content: 'ok' }, done: true, done_reason: 'stop' },
          { message: { role: 'assistant', content: 'late' }, done: false },
        ],
        keepOpen: true,
      },
    });
    try {
      for await (const chunk of new OllamaClient({ baseUrl: server.baseUrl }).chat(REQUEST)) {
        if (chunk.done) break;
      }

      // The line scanner owns the reader, so breaking on the final chunk must cancel the stream:
      // the server sees the connection close instead of holding it until it times out.
      await server.closed(0);
    } finally {
      await server.stop();
    }
  }, 10_000);
});

describe('error bodies', () => {
  it('parses the error field of a JSON body larger than the display cap', async () => {
    const server = await startFakeOllama({
      chat: { status: 500, body: { error: 'boom', pad: 'x'.repeat(8_000) } },
    });
    try {
      // `maxRetries: 0` keeps the one queued reply: a 5xx is retryable by default, and a second
      // attempt would consume the fake server's default success reply instead.
      const error = await new OllamaClient({ baseUrl: server.baseUrl, maxRetries: 0 })
        .chat(REQUEST)
        .next()
        .catch((cause: unknown) => cause);

      expect(error).toBeInstanceOf(OllamaError);
      expect((error as OllamaError).status).toBe(500);
      // The body is 8 KB: past the 4 KB display cap but inside the parse budget, so the user
      // reads the server's message instead of a truncated fragment of its JSON envelope.
      expect((error as Error).message).toBe('500: boom');
    } finally {
      await server.stop();
    }
  });

  it('does not parse a JSON error body past the secondary cap', async () => {
    const server = await startFakeOllama({
      chat: { status: 500, body: { error: 'boom', pad: 'x'.repeat(200_000) } },
    });
    try {
      const error = await new OllamaClient({ baseUrl: server.baseUrl, maxRetries: 0 })
        .chat(REQUEST)
        .next()
        .catch((cause: unknown) => cause);

      expect(error).toBeInstanceOf(OllamaError);
      expect((error as OllamaError).status).toBe(500);
      // 200 KB is not worth parsing: the capped raw body is shown, with its own marker,
      // instead of spending the parse on a body that could not fit the transcript anyway.
      expect((error as Error).message).toContain('{"error":"boom"');
      expect((error as Error).message).toContain('truncated');
      expect((error as Error).message.length).toBeLessThan(4_200);
    } finally {
      await server.stop();
    }
  });

  it('caps a huge error body instead of keeping all of it', async () => {
    const server = await startFakeOllama({ chat: { status: 500, body: { error: 'x'.repeat(2_000_000) } } });
    try {
      // `maxRetries: 0` keeps the one reply the test queued: a 5xx is retryable by default, and a
      // second attempt would consume the fake server's default success reply instead.
      const error = await new OllamaClient({ baseUrl: server.baseUrl, maxRetries: 0 })
        .chat(REQUEST)
        .next()
        .catch((cause: unknown) => cause);

      expect(error).toBeInstanceOf(OllamaError);
      expect((error as OllamaError).status).toBe(500);
      expect((error as Error).message.length).toBeLessThan(4_200);
      expect((error as Error).message).toContain('truncated');
    } finally {
      await server.stop();
    }
  });
});

describe('stream error payloads', () => {
  it('caps a huge error chunk instead of embedding it in the transcript', async () => {
    const server = await startFakeOllama({
      chat: { rawChunks: [JSON.stringify({ error: 'x'.repeat(1_000_000) })] },
    });
    try {
      const error = await new OllamaClient({ baseUrl: server.baseUrl })
        .chat(REQUEST)
        .next()
        .catch((cause: unknown) => cause);

      expect(error).toBeInstanceOf(OllamaError);
      expect((error as OllamaError).kind).toBe('stream');
      // The capped text reaches the transcript, the TUI and Pi's error patterns, so an
      // unbounded payload would be paid for three times.
      expect((error as Error).message.length).toBeLessThan(4_100);
      expect((error as Error).message).toContain('truncated');
    } finally {
      await server.stop();
    }
  });

  it('caps a huge malformed line instead of embedding it in the transcript', async () => {
    const server = await startFakeOllama({ chat: { rawChunks: ['x'.repeat(1_000_000)] } });
    try {
      const error = await new OllamaClient({ baseUrl: server.baseUrl })
        .chat(REQUEST)
        .next()
        .catch((cause: unknown) => cause);

      expect(error).toBeInstanceOf(OllamaError);
      expect((error as OllamaError).kind).toBe('stream');
      expect((error as Error).message).toContain('malformed stream line');
      expect((error as Error).message.length).toBeLessThan(4_200);
      expect((error as Error).message).toContain('truncated');
    } finally {
      await server.stop();
    }
  });
});

describe('stream boundaries', () => {
  it('parses a final line that has no trailing newline', async () => {
    const server = await startFakeOllama({
      chat: {
        rawBody:
          '{"message":{"role":"assistant","content":"a"},"done":false}\n' +
          '{"message":{"role":"assistant","content":"b"},"done":true,"done_reason":"stop"}',
      },
    });
    try {
      const contents: string[] = [];
      for await (const chunk of new OllamaClient({ baseUrl: server.baseUrl }).chat(REQUEST)) {
        contents.push(chunk.message?.content ?? '');
      }
      expect(contents).toEqual(['a', 'b']);
    } finally {
      await server.stop();
    }
  });
});
