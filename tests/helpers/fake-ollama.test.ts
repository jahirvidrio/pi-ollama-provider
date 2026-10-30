import { describe, expect, it } from 'vitest';
import { startFakeOllama } from './fake-ollama.ts';

describe('fake ollama server', () => {
  it('answers /api/version with the configured version', async () => {
    const server = await startFakeOllama();
    try {
      const response = await fetch(`${server.baseUrl}/version`);
      expect(response.status).toBe(200);
      expect(await response.json()).toEqual({ version: '0.12.6' });
    } finally {
      await server.stop();
    }
  });

  it('serves /api/tags and /api/show payloads', async () => {
    const server = await startFakeOllama({
      tags: { models: [{ name: 'gemma4:31b', model: 'gemma4:31b' }] },
      show: { capabilities: ['completion'] },
    });
    try {
      expect(await (await fetch(`${server.baseUrl}/tags`)).json()).toEqual({
        models: [{ name: 'gemma4:31b', model: 'gemma4:31b' }],
      });
      const show = await fetch(`${server.baseUrl}/show`, {
        method: 'POST',
        body: JSON.stringify({ model: 'gemma4:31b' }),
      });
      expect(await show.json()).toEqual({ capabilities: ['completion'] });
    } finally {
      await server.stop();
    }
  });

  it('streams queued NDJSON chat chunks and records the request', async () => {
    const server = await startFakeOllama();
    try {
      server.respond({
        chunks: [
          { message: { role: 'assistant', content: 'Hel' }, done: false },
          { message: { role: 'assistant', content: 'lo' }, done: false },
          { message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop' },
        ],
      });
      const response = await fetch(`${server.baseUrl}/chat`, {
        method: 'POST',
        body: JSON.stringify({ model: 'gemma4:31b', messages: [], stream: true }),
      });

      expect(response.status).toBe(200);
      expect(response.headers.get('content-type')).toBe('application/x-ndjson');
      const text = await response.text();
      expect(text).toBe(
        [
          '{"message":{"role":"assistant","content":"Hel"},"done":false}',
          '{"message":{"role":"assistant","content":"lo"},"done":false}',
          '{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}',
          '',
        ].join('\n')
      );

      expect(server.requests).toHaveLength(1);
      expect(server.requests[0]?.path).toBe('/api/chat');
      expect(server.requests[0]?.body).toEqual({ model: 'gemma4:31b', messages: [], stream: true });
    } finally {
      await server.stop();
    }
  });

  it('writes rawChunks byte chunks verbatim and string lines with a newline', async () => {
    const server = await startFakeOllama();
    try {
      server.respond({
        rawChunks: [
          '{"message":{"content":"ok"},"done":false}',
          new TextEncoder().encode('{"message":{"content":"bye"},"done":false}\n'),
        ],
      });
      const response = await fetch(`${server.baseUrl}/chat`, { method: 'POST', body: '{}' });

      expect(await response.text()).toBe(
        '{"message":{"content":"ok"},"done":false}\n' +
          '{"message":{"content":"bye"},"done":false}\n'
      );
    } finally {
      await server.stop();
    }
  });

  it('returns an HTTP error body without streaming', async () => {
    const server = await startFakeOllama();
    try {
      server.respond({ status: 401, body: { error: 'Unauthorized' } });
      const response = await fetch(`${server.baseUrl}/chat`, { method: 'POST', body: '{}' });

      expect(response.status).toBe(401);
      expect(await response.json()).toEqual({ error: 'Unauthorized' });
    } finally {
      await server.stop();
    }
  });

  it('delivers a mid-stream error chunk after a 200', async () => {
    const server = await startFakeOllama();
    try {
      server.respond({
        chunks: [
          { message: { role: 'assistant', content: 'Yes' }, done: false },
          { error: 'an error was encountered while running the model' },
        ],
      });
      const response = await fetch(`${server.baseUrl}/chat`, { method: 'POST', body: '{}' });

      expect(response.status).toBe(200);
      const lines = (await response.text()).trimEnd().split('\n');
      expect(lines).toHaveLength(2);
      expect(JSON.parse(lines[1] ?? '')).toEqual({
        error: 'an error was encountered while running the model',
      });
    } finally {
      await server.stop();
    }
  });

  it('stops accepting connections after stop()', async () => {
    const server = await startFakeOllama();
    const baseUrl = server.baseUrl;
    await server.stop();

    await expect(fetch(`${baseUrl}/version`)).rejects.toThrow();
  });
});
