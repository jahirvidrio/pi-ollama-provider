/** The local server: no credential to collect and no key to send. */

import type { ApiKeyCredential, AuthContext } from '@earendil-works/pi-ai';
import {
  DEFAULT_OLLAMA_BASE_URL,
  LOCAL_API_KEY,
  OLLAMA_HOST_ENV,
  OLLAMA_PROVIDER_ID,
  type ProviderProfile,
  type ResolvedKey,
  STORED_CREDENTIAL,
} from './profile.ts';

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
