/** The hosted server: the key is the credential, and the endpoint comes from the environment. */

import type { ApiKeyCredential, AuthContext, ProviderAuthInteraction } from '@earendil-works/pi-ai';
import { createClient } from '../client/index.ts';
import {
  baseUrlOf,
  DEFAULT_OLLAMA_CLOUD_BASE_URL,
  OLLAMA_API_KEY_ENV,
  OLLAMA_CLOUD_HOST_ENV,
  OLLAMA_CLOUD_PROVIDER_ID,
  type ProviderProfile,
  type ResolvedKey,
  STORED_CREDENTIAL,
} from './profile.ts';

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
