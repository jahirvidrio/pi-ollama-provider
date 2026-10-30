import { describe, expect, it } from 'vitest';
import { OllamaClient } from '../src/client.ts';
import { startFakeOllama } from './helpers/fake-ollama.ts';

const REQUEST = { model: 'gemma4:31b', messages: [{ role: 'user' as const, content: 'hi' }] };

/** UTF-8 bytes of one NDJSON line (newline included), split into fixed-size pieces. */
function linePieces (line: string, pieceBytes: number): Uint8Array[] {
  const bytes = new TextEncoder().encode(line);
  const pieces: Uint8Array[] = [];
  for (let offset = 0; offset < bytes.length; offset += pieceBytes) {
    pieces.push(bytes.slice(offset, offset + pieceBytes));
  }
  return pieces;
}

describe('chunked line scanning', () => {
  it('reassembles a single line split across many chunks', async () => {
    // One fragmented oversized line: 2 MiB of content written in 64 KiB pieces.
    const content = 'x'.repeat(2 * 1024 * 1024);
    const line = `${JSON.stringify({ message: { role: 'assistant', content }, done: false })}\n`;
    const server = await startFakeOllama({ chat: { rawChunks: linePieces(line, 64 * 1024) } });
    try {
      const contents: string[] = [];
      for await (const chunk of new OllamaClient({ baseUrl: server.baseUrl }).chat(REQUEST)) {
        contents.push(chunk.message?.content ?? '');
      }
      expect(contents).toEqual([content]);
    } finally {
      await server.stop();
    }
  });

  it('round-trips a UTF-8 character split across chunk boundaries', async () => {
    const line = `${JSON.stringify({ message: { content: 'известни' }, done: false })}\n`;
    const bytes = new TextEncoder().encode(line);
    // The JSON prefix is pure ASCII, so its length is a byte offset: cutting one
    // byte into the Cyrillic text splits a multi-byte character across two chunks.
    const prefix = '{"message":{"content":"';
    const cut = prefix.length + 1;
    const server = await startFakeOllama({
      chat: { rawChunks: [bytes.slice(0, cut), bytes.slice(cut)] },
    });
    try {
      const chunks: unknown[] = [];
      for await (const chunk of new OllamaClient({ baseUrl: server.baseUrl }).chat(REQUEST)) {
        chunks.push(chunk);
      }
      expect(chunks).toEqual([{ message: { content: 'известни' }, done: false }]);
    } finally {
      await server.stop();
    }
  });

  it('skips leading newlines without losing the chunk that follows', async () => {
    const chunk = { message: { role: 'assistant', content: 'ok' }, done: false };
    const server = await startFakeOllama({
      chat: { rawChunks: ['\n', `\n${JSON.stringify(chunk)}\n`] },
    });
    try {
      const chunks: unknown[] = [];
      for await (const yielded of new OllamaClient({ baseUrl: server.baseUrl }).chat(REQUEST)) {
        chunks.push(yielded);
      }
      expect(chunks).toEqual([chunk]);
    } finally {
      await server.stop();
    }
  });
});
