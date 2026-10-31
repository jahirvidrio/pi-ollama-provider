/**
 * Provider registration: two independent providers, `ollama` (local server) and
 * `ollama-cloud`, both purely dynamic and served by the native `/api/chat` stream. This module
 * composes a profile with Pi's provider contract; what each deployment answers differently lives
 * in `local.ts` and `cloud.ts`.
 */

import {
  type ApiKeyAuth,
  type AnyModel,
  type AuthResult,
  createProvider,
  type Model,
  type Provider,
  type RefreshModelsContext,
} from '@earendil-works/pi-ai';
import { loadCatalog } from '../catalog/index.ts';
import { streamOllamaChat } from '../chat/index.ts';
import type { OllamaChatConfig } from '../chat/options.ts';
import { createClient } from '../client/index.ts';
import { OLLAMA_CHAT_API } from '../wire.ts';
import { CloudProfile } from './cloud.ts';
import { LocalProfile } from './local.ts';
import { baseUrlOf, type OllamaProviderKind, type ProviderProfile } from './profile.ts';
import {
  CATALOG_TIMEOUT_MS,
  LOGIN_TIMEOUT_MS,
  type OllamaProviderTimeouts,
} from './timeouts.ts';

const PROFILES: Record<OllamaProviderKind, new (loginMs: number) => ProviderProfile> = {
  local: LocalProfile,
  cloud: CloudProfile,
};

/**
 * One provider's `apiKey` auth. `check` is deliberately absent: resolving is cheap and free of side
 * effects, and Pi derives the same status from `resolve()` when no check is declared.
 */
function apiKeyAuth (profile: ProviderProfile): ApiKeyAuth {
  const auth: ApiKeyAuth = {
    name: profile.authName,
    resolve: async ({ ctx, credential }): Promise<AuthResult | undefined> => {
      const resolved = await profile.resolveKey(ctx, credential);
      if (!resolved) return undefined;
      const baseUrl = baseUrlOf(profile, await ctx.env(profile.hostEnv));
      return {
        auth: { apiKey: resolved.key, baseUrl },
        env: { [profile.hostEnv]: baseUrl },
        source: resolved.source,
      };
    },
  };
  if (profile.login) auth.login = profile.login;
  return auth;
}

/** Chat models of this provider that this adapter can serve, restored from disk. */
function isOwnChatModel (model: AnyModel, providerId: string): model is Model<'ollama-chat'> {
  return model.provider === providerId && (model.type ?? 'chat') === 'chat' && model.api === OLLAMA_CHAT_API;
}

/**
 * Builds this provider's catalog. `createProvider` owns the transactional contract: it restores
 * `context.stored` before any network access and persists whatever this returns with `checkedAt`,
 * so this function only has to produce the list.
 */
async function fetchModels (
  profile: ProviderProfile,
  context: RefreshModelsContext,
  catalogMs: number
): Promise<Model<'ollama-chat'>[]> {
  const restored = (context.stored?.models ?? []).filter((model) => isOwnChatModel(model, profile.id));
  const credential = context.credential;
  if (!context.allowNetwork || context.signal.aborted || credential?.type !== 'api_key') {
    return restored;
  }
  // This context has no env accessor, so the endpoint comes from `process.env` directly; the auth
  // path reads the same variable through `AuthContext`.
  const client = createClient(
    baseUrlOf(profile, process.env[profile.hostEnv]),
    { apiKey: credential.key },
    { timeoutMs: catalogMs }
  );
  // `loadCatalog` returns the stored catalog unchanged when `/api/tags` fails, and the deadline
  // turns a server that never answers into exactly that failure instead of a hang.
  return loadCatalog({
    client,
    providerId: profile.id,
    allowNetwork: true,
    stored: restored,
    signal: context.signal,
  });
}

/**
 * Builds one of the two Ollama providers from its profile.
 */
export function createOllamaProvider (
  kind: OllamaProviderKind,
  timeouts: OllamaProviderTimeouts = {}
): Provider<'ollama-chat'> {
  const profile = new PROFILES[kind](timeouts.loginMs ?? LOGIN_TIMEOUT_MS);
  const catalogMs = timeouts.catalogMs ?? CATALOG_TIMEOUT_MS;
  const config: OllamaChatConfig = { kind: profile.kind };

  return createProvider<'ollama-chat'>({
    id: profile.id,
    name: profile.name,
    baseUrl: profile.defaultBaseUrl,
    auth: { apiKey: apiKeyAuth(profile) },
    // Purely dynamic: the catalog comes from the server, never from a static list.
    models: [],
    api: {
      [OLLAMA_CHAT_API]: {
        stream: (model, context, options) =>
          streamOllamaChat(model as Model<'ollama-chat'>, context, options, config),
        streamSimple: (model, context, options) =>
          streamOllamaChat(model as Model<'ollama-chat'>, context, options, config),
      },
    },
    fetchModels: (context) => fetchModels(profile, context, catalogMs),
  });
}
