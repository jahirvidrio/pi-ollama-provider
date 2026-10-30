/**
 * Provider registration: two independent providers, `ollama` (local server) and
 * `ollama-cloud`, both purely dynamic and served by the native `/api/chat` stream.
 */

import {
  type ApiKeyAuth,
  type ApiKeyCredential,
  type AnyModel,
  type AuthContext,
  type AuthResult,
  type Model,
  type Provider,
  type ProviderAuthInteraction,
  type RefreshModelsContext,
  createProvider,
} from '@earendil-works/pi-ai';
import { loadCatalog } from '../catalog/index.ts';
import { createClient } from '../client/index.ts';
import { OLLAMA_CHAT_API, type OllamaChatConfig, streamOllamaChat } from '../chat/index.ts';
import { normalizeOllamaBaseUrl } from '../wire.ts';

export const OLLAMA_PROVIDER_ID = 'ollama';
export const OLLAMA_CLOUD_PROVIDER_ID = 'ollama-cloud';
/** Canonical `/api` form of the `OLLAMA_HOST` default, `http://localhost:11434/`. */
export const DEFAULT_OLLAMA_BASE_URL = 'http://localhost:11434/api';
/** Canonical `/api` form of the `OLLAMA_CLOUD_HOST` default, `https://ollama.com/`. */
export const DEFAULT_OLLAMA_CLOUD_BASE_URL = 'https://ollama.com/api';

/** Ambient variable that names the local server. */
const OLLAMA_HOST_ENV = 'OLLAMA_HOST';
/** Cloud has its own variable, so one environment cannot move both providers (ADR 0007). */
const OLLAMA_CLOUD_HOST_ENV = 'OLLAMA_CLOUD_HOST';
/** Ambient fallback for the cloud key, so the key can stay out of `auth.json`. */
const OLLAMA_API_KEY_ENV = 'OLLAMA_API_KEY';

/** The key a local server is resolved with: Ollama itself requires none. */
const LOCAL_API_KEY = 'local';
/** Where a key came from, as Pi's auth status shows it. */
const STORED_CREDENTIAL = 'stored credential';

export type OllamaProviderKind = 'local' | 'cloud';

/** A request key and the source Pi reports for it. */
interface ResolvedKey {
  key: string;
  source: string;
}

/**
 * One deployment flavor behind one interface: everything the two providers answer differently.
 * The endpoint is not part of the contract beyond naming its variable, because the endpoint is
 * configuration read at request time (ADR 0007); normalizing it is shared work in
 * {@link baseUrlOf}, not something each profile reimplements.
 */
export interface ProviderProfile {
  /** Read by the chat stream: only the local server allocates a context window. */
  readonly kind: OllamaProviderKind;
  /** Provider id Pi registers, e.g. `ollama-cloud`. */
  readonly id: string;
  readonly name: string;
  /** Label of this provider's credential in Pi's auth UI. */
  readonly authName: string;
  /** Ambient variable that names this provider's endpoint. */
  readonly hostEnv: string;
  /** Endpoint used when {@link hostEnv} is unset or blank. */
  readonly defaultBaseUrl: string;
  /** The request key and where it came from; `undefined` when this provider is not configured. */
  resolveKey(ctx: AuthContext, credential: ApiKeyCredential | undefined): Promise<ResolvedKey | undefined>;
  /** Interactive setup, when this provider has a credential to collect. */
  login?(interaction: ProviderAuthInteraction): Promise<ApiKeyCredential>;
}

/**
 * This profile's endpoint: the variable that names it, normalized to the canonical `/api` base,
 * or the built-in default when the variable is unset or blank. Auth and the catalog call this same
 * function, so they can never disagree about where the server is.
 */
function baseUrlOf (profile: ProviderProfile, ambientHost: string | undefined): string {
  const named = ambientHost?.trim();
  return named ? normalizeOllamaBaseUrl(named) : profile.defaultBaseUrl;
}

/**
 * The local server: no credential to collect and no key to send, and the endpoint comes from
 * `OLLAMA_HOST`. A key stored by hand in `auth.json` is still honored, for a server behind a proxy.
 */
export class LocalProfile implements ProviderProfile {
  readonly kind = 'local';
  readonly id = OLLAMA_PROVIDER_ID;
  readonly name = 'Ollama';
  readonly authName = 'Ollama server';
  readonly hostEnv = OLLAMA_HOST_ENV;
  readonly defaultBaseUrl = DEFAULT_OLLAMA_BASE_URL;

  async resolveKey (
    _ctx: AuthContext,
    credential: ApiKeyCredential | undefined
  ): Promise<ResolvedKey> {
    const stored = credential?.key?.trim();
    return stored
      ? { key: stored, source: STORED_CREDENTIAL }
      : { key: LOCAL_API_KEY, source: 'local server' };
  }
}

/**
 * The hosted server: the key is the credential, and the endpoint comes from `OLLAMA_CLOUD_HOST`,
 * so a private deployment or a proxy needs no code change and no second variable.
 */
export class CloudProfile implements ProviderProfile {
  readonly kind = 'cloud';
  readonly id = OLLAMA_CLOUD_PROVIDER_ID;
  readonly name = 'Ollama Cloud';
  readonly authName = 'Ollama Cloud API key';
  readonly hostEnv = OLLAMA_CLOUD_HOST_ENV;
  readonly defaultBaseUrl = DEFAULT_OLLAMA_CLOUD_BASE_URL;

  constructor (private readonly loginMs: number) {}

  async resolveKey (
    ctx: AuthContext,
    credential: ApiKeyCredential | undefined
  ): Promise<ResolvedKey | undefined> {
    const stored = credential?.key?.trim();
    if (stored) return { key: stored, source: STORED_CREDENTIAL };
    const ambient = (await ctx.env(OLLAMA_API_KEY_ENV))?.trim();
    return ambient ? { key: ambient, source: OLLAMA_API_KEY_ENV } : undefined;
  }

  /** Verifying against `/api/tags` proves the key against the configured endpoint. */
  readonly login = async (interaction: ProviderAuthInteraction): Promise<ApiKeyCredential> => {
    const key = (await interaction.prompt({ type: 'secret', message: 'Ollama Cloud API key' })).trim();
    await createClient(
      baseUrlOf(this, process.env[this.hostEnv]),
      { apiKey: key || undefined },
      { timeoutMs: this.loginMs }
    ).listModels(interaction.signal);
    // The key and nothing else: the endpoint stays whatever the environment names (ADR 0007).
    return { type: 'api_key', key: key || undefined };
  };
}

/** The two profiles, by kind: a third deployment is one class above and one entry here. */
const PROFILES: Record<OllamaProviderKind, new (loginMs: number) => ProviderProfile> = {
  local: LocalProfile,
  cloud: CloudProfile,
};

/**
 * One provider's `apiKey` auth. `check` is deliberately absent: resolving is cheap and free of side
 * effects, and Pi derives the same status from `resolve()` when no check is declared.
 */
function apiKeyAuth (profile: ProviderProfile): ApiKeyAuth {
  const login = profile.login;
  return {
    name: profile.authName,
    ...(login !== undefined && { login }),
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
}

/** Chat models of this provider that this adapter can serve, restored from disk. */
function isOwnChatModel (model: AnyModel, providerId: string): model is Model<'ollama-chat'> {
  return model.provider === providerId && (model.type ?? 'chat') === 'chat' && model.api === OLLAMA_CHAT_API;
}

/**
 * Deadline for one catalog request, not for the refresh as a whole: `/api/tags` gets this budget
 * and every `/api/show` gets it again, so a refresh of N models whose requests all stall can spend
 * up to `CATALOG_TIMEOUT_MS` × (1 + ⌈N/4⌉), because the fan-out runs four at a time. A server that
 * accepts the connection and never answers is a hang, not an exception: without a deadline the
 * stored-catalog fallback in `loadCatalog` would never fire and the provider would stay without
 * models. A stalled request costs its model its metadata, a stalled `/api/tags` costs this refresh
 * its network path, and the next refresh retries by itself: the catalog is best-effort and never
 * retried inside one refresh (ADR 0004).
 */
export const CATALOG_TIMEOUT_MS = 30_000;

/** Budget for the cloud login probe: a server that cannot verify inside it is not usable now. */
export const LOGIN_TIMEOUT_MS = 15_000;

/**
 * Deadlines the provider gives its own non-chat requests. The defaults fit a local server or a
 * public cloud endpoint; a host may retune them, and a test shortens them to reach the fallback
 * path without waiting the whole budget.
 */
export interface OllamaProviderTimeouts {
  /**
   * Deadline for one catalog request — `/api/tags`, and again for each `/api/show`; defaults to
   * {@link CATALOG_TIMEOUT_MS}. It is not a budget for the refresh as a whole: see the constant.
   */
  catalogMs?: number;
  /** The cloud login probe (`/api/tags`); defaults to {@link LOGIN_TIMEOUT_MS}. */
  loginMs?: number;
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
  // path reads the same variable through `AuthContext` (ADR 0007).
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
