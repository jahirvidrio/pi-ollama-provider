import { describe, expect, it } from 'vitest';
import type { Model } from '@earendil-works/pi-ai';
import { OllamaClient } from '../src/client/index.ts';
import { loadCatalog, toChatModel } from '../src/catalog/index.ts';
import { startFakeOllama } from './helpers/fake-ollama.ts';

const TAGS = {
  models: [
    { name: 'gemma4:31b', model: 'gemma4:31b', capabilities: ['completion', 'vision'] },
    { name: 'gpt-oss:20b', model: 'gpt-oss:20b', capabilities: ['completion', 'tools', 'thinking'] },
    { name: 'nomic-embed-text', model: 'nomic-embed-text', capabilities: ['embedding'] },
    { name: 'legacy:1b', model: 'legacy:1b' },
  ],
};

function shows (): unknown {
  return {
    capabilities: ['completion'],
    model_info: { 'gptoss.context_length': 131072 },
    thinking: { values: ['low', 'medium', 'high'], default: 'medium' },
  };
}

describe('toChatModel', () => {
  it('maps context window, capabilities and named thinking levels', () => {
    const model = toChatModel({
      summary: { name: 'gpt-oss:20b', model: 'gpt-oss:20b', capabilities: ['completion', 'vision', 'thinking'] },
      show: {
        model_info: { 'gptoss.context_length': 131072 },
        thinking: { values: ['low', 'medium', 'high'], default: 'medium' },
      },
      providerId: 'ollama',
      baseUrl: 'http://localhost:11434/api',
    });

    expect(model).toMatchObject({
      id: 'gpt-oss:20b',
      name: 'gpt-oss:20b',
      api: 'ollama-chat',
      provider: 'ollama',
      baseUrl: 'http://localhost:11434/api',
      reasoning: true,
      input: ['text', 'image'],
      contextWindow: 131072,
      maxTokens: 131072,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
    expect(model.thinkingLevelMap).toEqual({
      off: 'off',
      minimal: null,
      low: 'low',
      medium: 'medium',
      high: 'high',
      xhigh: null,
      max: null,
    });
  });

  it('offers off and one on level for a boolean-only thinking model', () => {
    const model = toChatModel({
      summary: { name: 'gpt-oss:20b', model: 'gpt-oss:20b', capabilities: ['completion', 'thinking'] },
      show: { thinking: { values: [false, true], default: true } },
      providerId: 'ollama-cloud',
      baseUrl: 'https://ollama.com/api',
    });

    expect(model.reasoning).toBe(true);
    expect(model.thinkingLevelMap).toEqual({
      off: 'off',
      minimal: null,
      low: null,
      medium: 'true',
      high: null,
      xhigh: null,
      max: null,
    });
  });

  it('falls back to conservative defaults without show metadata', () => {
    const model = toChatModel({
      summary: { name: 'gemma4:31b', model: 'gemma4:31b' },
      show: {},
      providerId: 'ollama',
      baseUrl: 'http://localhost:11434/api',
    });

    expect(model.contextWindow).toBe(4096);
    expect(model.maxTokens).toBe(4096);
    expect(model.reasoning).toBe(false);
    expect(model.thinkingLevelMap).toBeUndefined();
    expect(model.input).toEqual(['text']);
  });
});

describe('loadCatalog', () => {
  it('skips non-chat models when capabilities are reported and keeps unnamed-capability models', async () => {
    const server = await startFakeOllama({
      tags: TAGS,
      show: (body) => (body.model === 'gpt-oss:20b' ? shows() : { capabilities: ['completion'] }),
    });
    try {
      const models = await loadCatalog({
        client: new OllamaClient({ baseUrl: server.baseUrl }),
        providerId: 'ollama',
        allowNetwork: true,
      });

      expect(models.map((model) => model.id)).toEqual(['gemma4:31b', 'gpt-oss:20b', 'legacy:1b']);
    } finally {
      await server.stop();
    }
  });

  it('keeps every model when one /api/show stalls past its deadline', async () => {
    const server = await startFakeOllama({
      tags: TAGS,
      show: (body) => (body.model === 'gpt-oss:20b' ? shows() : { capabilities: ['completion'] }),
      stallShow: (model) => (model === 'gemma4:31b' ? 'headers' : undefined),
    });
    try {
      const models = await loadCatalog({
        client: new OllamaClient({ baseUrl: server.baseUrl, timeoutMs: 150 }),
        providerId: 'ollama',
        allowNetwork: true,
      });

      // One stalled metadata request must cost that model its metadata, not the catalog.
      expect(models.map((model) => model.id)).toEqual(['gemma4:31b', 'gpt-oss:20b', 'legacy:1b']);
      expect(models[0]?.contextWindow).toBe(4096);
      expect(models[0]?.reasoning).toBe(false);
      expect(models[1]?.contextWindow).toBe(131072);
      expect(models[1]?.reasoning).toBe(true);
    } finally {
      await server.stop();
    }
  }, 10_000);

  it('returns stored models without touching the network when offline', async () => {
    const server = await startFakeOllama({ tags: TAGS, show: shows });
    const stored: Model<'ollama-chat'>[] = [
      {
        id: 'cached:7b',
        name: 'cached:7b',
        api: 'ollama-chat',
        provider: 'ollama',
        baseUrl: server.baseUrl,
        reasoning: false,
        input: ['text'],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 8192,
        maxTokens: 8192,
      },
    ];
    try {
      const models = await loadCatalog({
        client: new OllamaClient({ baseUrl: server.baseUrl }),
        providerId: 'ollama',
        allowNetwork: false,
        stored,
      });

      expect(models).toEqual(stored);
      expect(server.requests).toHaveLength(0);
    } finally {
      await server.stop();
    }
  });
});
