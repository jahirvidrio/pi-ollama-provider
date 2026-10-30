/**
 * What identifies one provider and where its endpoint comes from: the pieces both profiles and the
 * factory share, so a third deployment reuses them instead of copying a resolution order.
 */

import type { ApiKeyCredential, AuthContext, ProviderAuthInteraction } from '@earendil-works/pi-ai';
import { normalizeOllamaBaseUrl } from '../wire.ts';

export const OLLAMA_PROVIDER_ID = 'ollama';
export const OLLAMA_CLOUD_PROVIDER_ID = 'ollama-cloud';
/** Canonical `/api` form of the `OLLAMA_HOST` default, `http://localhost:11434/`. */
export const DEFAULT_OLLAMA_BASE_URL = 'http://localhost:11434/api';
/** Canonical `/api` form of the `OLLAMA_CLOUD_HOST` default, `https://ollama.com/`. */
export const DEFAULT_OLLAMA_CLOUD_BASE_URL = 'https://ollama.com/api';

/** Ambient variable that names the local server. */
export const OLLAMA_HOST_ENV = 'OLLAMA_HOST';
/** Cloud has its own variable, so one environment cannot move both providers (ADR 0007). */
export const OLLAMA_CLOUD_HOST_ENV = 'OLLAMA_CLOUD_HOST';
/** Ambient fallback for the cloud key, so the key can stay out of `auth.json`. */
export const OLLAMA_API_KEY_ENV = 'OLLAMA_API_KEY';

/** The key a local server is resolved with: Ollama itself requires none. */
export const LOCAL_API_KEY = 'local';
/** Where a key came from, as Pi's auth status shows it. */
export const STORED_CREDENTIAL = 'stored credential';

export type OllamaProviderKind = 'local' | 'cloud';

/** A request key and the source Pi reports for it. */
export interface ResolvedKey {
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
  resolveKey (ctx: AuthContext, credential: ApiKeyCredential | undefined): Promise<ResolvedKey | undefined>;
  /** Interactive setup, when this provider has a credential to collect. */
  login?(interaction: ProviderAuthInteraction): Promise<ApiKeyCredential>;
}

/**
 * This profile's endpoint: the variable that names it, normalized to the canonical `/api` base,
 * or the built-in default when the variable is unset or blank. Auth and the catalog call this same
 * function, so they can never disagree about where the server is.
 */
export function baseUrlOf (profile: ProviderProfile, ambientHost: string | undefined): string {
  const named = ambientHost?.trim();
  return named ? normalizeOllamaBaseUrl(named) : profile.defaultBaseUrl;
}
