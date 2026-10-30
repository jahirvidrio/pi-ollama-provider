/**
 * Provider registration: two independent providers, `ollama` (local server) and
 * `ollama-cloud`, both purely dynamic and served by the native `/api/chat` stream.
 */

import {
	type ApiKeyAuth,
	type ApiKeyCredential,
	type AnyModel,
	type AuthCheck,
	type AuthContext,
	type AuthResult,
	type Credential,
	createProvider,
	type Model,
	type Provider,
	type RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { loadCatalog } from "./catalog.ts";
import { createClient, normalizeOllamaBaseUrl } from "./client.ts";
import { OLLAMA_CHAT_API, streamOllamaChat } from "./stream.ts";

export const OLLAMA_PROVIDER_ID = "ollama";
export const OLLAMA_CLOUD_PROVIDER_ID = "ollama-cloud";
export const DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/api";
export const DEFAULT_OLLAMA_CLOUD_BASE_URL = "https://ollama.com/api";

/** Where the chosen server URL is persisted, for both providers. */
const OLLAMA_HOST_ENV = "OLLAMA_HOST";
/** Ambient fallback for the cloud key, so the key can stay out of `auth.json`. */
const OLLAMA_API_KEY_ENV = "OLLAMA_API_KEY";

export type OllamaProviderKind = "local" | "cloud";

interface ProviderIdentity {
	kind: OllamaProviderKind;
	id: string;
	name: string;
	defaultBaseUrl: string;
}

const IDENTITIES: Record<OllamaProviderKind, ProviderIdentity> = {
	local: {
		kind: "local",
		id: OLLAMA_PROVIDER_ID,
		name: "Ollama",
		defaultBaseUrl: DEFAULT_OLLAMA_BASE_URL,
	},
	cloud: {
		kind: "cloud",
		id: OLLAMA_CLOUD_PROVIDER_ID,
		name: "Ollama Cloud",
		defaultBaseUrl: DEFAULT_OLLAMA_CLOUD_BASE_URL,
	},
};

/** `OLLAMA_HOST` stored on the credential by `/login`, normalized to an `/api` base URL. */
function credentialHost(credential: Credential | undefined): string | undefined {
	if (credential?.type !== "api_key") return undefined;
	const host = credential.env?.[OLLAMA_HOST_ENV];
	return typeof host === "string" && host.trim() ? normalizeOllamaBaseUrl(host) : undefined;
}

/**
 * The server URL for one provider: the credential's stored host, then the ambient
 * `OLLAMA_HOST`, then the built-in default. Cloud always uses its fixed endpoint.
 */
async function resolveBaseUrl(
	ctx: AuthContext,
	credential: ApiKeyCredential | undefined,
	identity: ProviderIdentity,
): Promise<string> {
	if (identity.kind === "cloud") return identity.defaultBaseUrl;
	const ambient = (await ctx.env(OLLAMA_HOST_ENV))?.trim();
	const configured = credentialHost(credential) ?? (ambient ? normalizeOllamaBaseUrl(ambient) : undefined);
	return configured ?? identity.defaultBaseUrl;
}

/** Same resolution as {@link resolveBaseUrl} without an `AuthContext`; see `fetchModels`. */
function refreshBaseUrl(context: RefreshModelsContext, identity: ProviderIdentity): string {
	if (identity.kind === "cloud") return identity.defaultBaseUrl;
	const configured = credentialHost(context.credential) ?? (context.baseUrl?.trim() || undefined);
	return (configured ? normalizeOllamaBaseUrl(configured) : undefined) ?? identity.defaultBaseUrl;
}

function apiKeyAuth(identity: ProviderIdentity, loginMs: number): ApiKeyAuth {
	const local = identity.kind === "local";
	return {
		name: local ? "Ollama server" : "Ollama Cloud API key",
		login: async (interaction): Promise<ApiKeyCredential> => {
			if (!local) {
				const key = (await interaction.prompt({ type: "secret", message: "Ollama Cloud API key" })).trim();
				// Verifying against `/api/tags` proves the key before it is persisted.
				await createClient(
					identity.defaultBaseUrl,
					{ apiKey: key || undefined },
					{ timeoutMs: loginMs },
				).listModels(interaction.signal);
				return { type: "api_key", key: key || undefined, env: { [OLLAMA_HOST_ENV]: identity.defaultBaseUrl } };
			}
			const fallback = process.env[OLLAMA_HOST_ENV] ?? identity.defaultBaseUrl;
			const entered = await interaction.prompt({
				type: "text",
				message: "Ollama server URL",
				placeholder: fallback,
			});
			const baseUrl = normalizeOllamaBaseUrl(entered.trim() || fallback);
			await createClient(baseUrl, undefined, { timeoutMs: loginMs }).version(interaction.signal);
			return { type: "api_key", key: undefined, env: { [OLLAMA_HOST_ENV]: baseUrl } };
		},
		check: async ({ ctx, credential }): Promise<AuthCheck | undefined> => {
			if (!local) {
				if (await cloudKey(ctx, credential)) return { type: "api_key", source: "Ollama Cloud API key" };
				return undefined;
			}
			// A local server needs no credential: a URL always resolves, so it is always configured.
			return { type: "api_key", source: credentialHost(credential) ? "stored credential" : "local server" };
		},
		resolve: async ({ ctx, credential }): Promise<AuthResult | undefined> => {
			const baseUrl = await resolveBaseUrl(ctx, credential, identity);
			const env = { [OLLAMA_HOST_ENV]: baseUrl };
			if (!local) {
				const apiKey = await cloudKey(ctx, credential);
				if (!apiKey) return undefined;
				const source = credential?.key ? "stored credential" : OLLAMA_API_KEY_ENV;
				return { auth: { apiKey, baseUrl }, env, source };
			}
			return {
				auth: { apiKey: credential?.key || "local", baseUrl },
				env,
				source: credential?.key ? "stored credential" : "local server",
			};
		},
	};
}

async function cloudKey(
	ctx: AuthContext,
	credential: ApiKeyCredential | undefined,
): Promise<string | undefined> {
	const stored = credential?.key?.trim();
	if (stored) return stored;
	const ambient = (await ctx.env(OLLAMA_API_KEY_ENV))?.trim();
	return ambient ? ambient : undefined;
}

/** Chat models of this provider that this adapter can serve, restored from disk. */
function isOwnChatModel(model: AnyModel, providerId: string): model is Model<"ollama-chat"> {
	return model.provider === providerId && (model.type ?? "chat") === "chat" && model.api === OLLAMA_CHAT_API;
}

/**
 * Deadline for one request of a catalog refresh, not for the refresh as a whole: `/api/tags` gets
 * this budget and every `/api/show` gets it again, so a refresh of N models whose requests all
 * stall can spend up to `CATALOG_TIMEOUT_MS` × (1 + ⌈N/4⌉), because the fan-out runs four at a
 * time. A server that accepts the connection and never answers is not an exception, so without a
 * deadline the stored-catalog fallback in `loadCatalog` would never fire and the provider would
 * stay without models. A stalled request costs its model its metadata; a stalled `/api/tags` costs
 * this refresh its network path, and the provider serves the stored list while the next refresh
 * retries by itself: the catalog is best-effort and never retried inside one refresh (ADR 0004).
 */
export const CATALOG_TIMEOUT_MS = 30_000;

/** Budget for the login probe: a server that cannot verify inside it is not usable right now. */
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
	/** One login probe (`/api/version` or `/api/tags`); defaults to {@link LOGIN_TIMEOUT_MS}. */
	loginMs?: number;
}

/**
 * Builds one of the two Ollama providers. `createProvider` owns the transactional catalog
 * contract: it restores `context.stored` before any network access and persists whatever
 * `fetchModels` returns with `checkedAt`, so this factory only has to produce the list.
 */
export function createOllamaProvider(
	kind: OllamaProviderKind,
	timeouts: OllamaProviderTimeouts = {},
): Provider<"ollama-chat"> {
	const identity = IDENTITIES[kind];
	const config = { kind: identity.kind };
	const catalogMs = timeouts.catalogMs ?? CATALOG_TIMEOUT_MS;
	const loginMs = timeouts.loginMs ?? LOGIN_TIMEOUT_MS;

	return createProvider<"ollama-chat">({
		id: identity.id,
		name: identity.name,
		baseUrl: identity.defaultBaseUrl,
		auth: { apiKey: apiKeyAuth(identity, loginMs) },
		// Purely dynamic: the catalog comes from the server, never from a static list.
		models: [],
		api: {
			[OLLAMA_CHAT_API]: {
				stream: (model, context, options) =>
					streamOllamaChat(model as Model<"ollama-chat">, context, options, config),
				streamSimple: (model, context, options) =>
					streamOllamaChat(model as Model<"ollama-chat">, context, options, config),
			},
		},
		fetchModels: async (context) => {
			const restored = (context.stored?.models ?? []).filter((model) => isOwnChatModel(model, identity.id));
			if (!context.allowNetwork || context.signal.aborted || context.credential?.type !== "api_key") {
				return restored;
			}
			const client = createClient(
				refreshBaseUrl(context, identity),
				{ apiKey: context.credential.key },
				{ timeoutMs: catalogMs },
			);
			// `loadCatalog` returns the stored catalog unchanged when `/api/tags` fails, and the
			// deadline turns a server that never answers into exactly that failure instead of a hang.
			return loadCatalog({
				client,
				providerId: identity.id,
				allowNetwork: true,
				stored: restored,
				signal: context.signal,
			});
		},
	});
}
