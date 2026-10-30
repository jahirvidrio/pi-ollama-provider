import { describe, expect, it, vi } from "vitest";
import {
	type AnyModel,
	type ApiKeyCredential,
	type AuthContext,
	type Credential,
	type Model,
	type ModelsPublication,
	type ProviderAuthInteraction,
	type RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { OllamaError } from "../src/client.ts";
import { DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_CLOUD_BASE_URL, createOllamaProvider } from "../src/provider.ts";
import { startFakeOllama } from "./helpers/fake-ollama.ts";

const TAGS = {
	models: [
		{ name: "gemma4:31b", model: "gemma4:31b", capabilities: ["completion", "vision"] },
		{ name: "gpt-oss:20b", model: "gpt-oss:20b", capabilities: ["completion", "tools", "thinking"] },
		{ name: "nomic-embed-text", model: "nomic-embed-text", capabilities: ["embedding"] },
	],
};

const SHOW = {
	capabilities: ["completion"],
	model_info: { "gptoss.context_length": 131072 },
	thinking: { values: ["low", "medium", "high"], default: "medium" },
};

/** Models the fake server serves, in `/api/tags` order, after the non-chat entry is dropped. */
const DISCOVERED_IDS = ["gemma4:31b", "gpt-oss:20b"];

// The deadline tests here prove a bound by the error the adapter produced and by their own test
// timeout catching a hang, never by a wall-clock comparison: `elapsed < 2_000` measures the machine,
// not the code. The hanging `fetch` below rejects a request that carries no signal, so a probe that
// forgot its deadline fails loudly instead of waiting forever.

function authEnv(values: Record<string, string> = {}): AuthContext {
	return {
		env: async (name) => values[name],
		fileExists: async () => false,
	};
}

function operationSignal(): AbortSignal {
	return new AbortController().signal;
}

/** One login prompt answer; the tests only need the prompt the provider's login issues. */
function loginInteraction(answer: string): ProviderAuthInteraction {
	return {
		signal: operationSignal(),
		prompt: async () => answer,
		notify: () => undefined,
	};
}

/** A `fetch` that answers nothing until its request signal aborts, for offline deadline tests. */
function hangingFetch(_input: string | URL | Request, init?: RequestInit): Promise<Response> {
	return new Promise((_resolve, reject) => {
		const signal = init?.signal;
		if (!signal) {
			reject(new Error("the provider must pass an abort signal to fetch"));
			return;
		}
		if (signal.aborted) {
			reject(signal.reason);
			return;
		}
		signal.addEventListener("abort", () => reject(signal.reason), { once: true });
	});
}

interface RefreshHarness {
	context: RefreshModelsContext;
	publications: ModelsPublication[];
}

function refreshHarness(input: {
	allowNetwork: boolean;
	credential?: Credential;
	stored?: readonly AnyModel[];
}): RefreshHarness {
	const publications: ModelsPublication[] = [];
	return {
		publications,
		context: {
			allowNetwork: input.allowNetwork,
			signal: operationSignal(),
			...(input.credential !== undefined && { credential: input.credential }),
			...(input.stored !== undefined && { stored: { models: input.stored } }),
			publish: async (publication) => {
				publications.push(publication);
				publication.update?.();
				return true;
			},
		},
	};
}

function storedModel(baseUrl: string): Model<"ollama-chat"> {
	return {
		id: "cached:7b",
		name: "cached:7b",
		api: "ollama-chat",
		provider: "ollama",
		baseUrl,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 8192,
		maxTokens: 8192,
	};
}

describe("ollama provider auth", () => {
	it("bounds the local login probe with the login deadline", async () => {
		const server = await startFakeOllama({ stallVersion: "headers" });
		try {
			const provider = createOllamaProvider("local", { loginMs: 150 });
			const login = provider.auth.apiKey?.login;
			if (!login) throw new Error("the local provider is expected to have an interactive login");

			const error = await login(loginInteraction(server.origin)).catch((cause: unknown) => cause);

			expect(error).toBeInstanceOf(OllamaError);
			expect((error as OllamaError).kind).toBe("network");
			expect((error as Error).message).toContain("/api/version");
			expect((error as Error).message).toContain("timed out after 150ms");
		} finally {
			await server.stop();
		}
	}, 10_000);

	it("bounds the cloud login probe with the login deadline", async () => {
		vi.stubGlobal("fetch", hangingFetch);
		try {
			const provider = createOllamaProvider("cloud", { loginMs: 150 });
			const login = provider.auth.apiKey?.login;
			if (!login) throw new Error("the cloud provider is expected to have an interactive login");

			const error = await login(loginInteraction("cloud-secret")).catch((cause: unknown) => cause);

			expect(error).toBeInstanceOf(OllamaError);
			expect((error as OllamaError).kind).toBe("network");
			expect((error as Error).message).toContain("/api/tags");
			expect((error as Error).message).toContain("timed out after 150ms");
		} finally {
			vi.unstubAllGlobals();
		}
	}, 10_000);

	it("resolves the local provider with zero configuration", async () => {
		const provider = createOllamaProvider("local");
		const result = await provider.auth.apiKey?.resolve({ ctx: authEnv(), signal: operationSignal() });

		expect(result?.auth).toEqual({ apiKey: "local", baseUrl: DEFAULT_OLLAMA_BASE_URL });
		expect(result?.env).toEqual({ OLLAMA_HOST: DEFAULT_OLLAMA_BASE_URL });
	});

	it("lets the stored credential's OLLAMA_HOST win over the ambient one", async () => {
		const provider = createOllamaProvider("local");
		const credential: ApiKeyCredential = {
			type: "api_key",
			key: "local",
			env: { OLLAMA_HOST: "http://from-credential:11434" },
		};
		const result = await provider.auth.apiKey?.resolve({
			ctx: authEnv({ OLLAMA_HOST: "http://from-env:11434" }),
			credential,
			signal: operationSignal(),
		});

		expect(result?.auth.baseUrl).toBe("http://from-credential:11434/api");
		expect(result?.env).toEqual({ OLLAMA_HOST: "http://from-credential:11434/api" });
	});

	it("uses the ambient OLLAMA_HOST when the credential carries no host", async () => {
		const provider = createOllamaProvider("local");
		const result = await provider.auth.apiKey?.resolve({
			ctx: authEnv({ OLLAMA_HOST: "http://from-env:11434" }),
			signal: operationSignal(),
		});

		expect(result?.auth).toEqual({ apiKey: "local", baseUrl: "http://from-env:11434/api" });
	});

	it("resolves a stored cloud key with the cloud base URL", async () => {
		const provider = createOllamaProvider("cloud");
		const credential: ApiKeyCredential = { type: "api_key", key: "cloud-secret" };
		const result = await provider.auth.apiKey?.resolve({ ctx: authEnv(), credential, signal: operationSignal() });

		expect(result?.auth).toEqual({ apiKey: "cloud-secret", baseUrl: DEFAULT_OLLAMA_CLOUD_BASE_URL });
		expect(result?.env).toEqual({ OLLAMA_HOST: DEFAULT_OLLAMA_CLOUD_BASE_URL });
	});

	it("uses the ambient OLLAMA_API_KEY when no cloud credential is stored", async () => {
		const provider = createOllamaProvider("cloud");
		const result = await provider.auth.apiKey?.resolve({
			ctx: authEnv({ OLLAMA_API_KEY: "env-secret" }),
			signal: operationSignal(),
		});

		expect(result?.auth.apiKey).toBe("env-secret");
	});

	it("reports the cloud provider as unconfigured without a stored key or OLLAMA_API_KEY", async () => {
		const provider = createOllamaProvider("cloud");
		const result = await provider.auth.apiKey?.resolve({ ctx: authEnv(), signal: operationSignal() });

		expect(result).toBeUndefined();
	});
});

describe("ollama provider catalog", () => {
	it("keeps the stored catalog when /api/tags outlives the catalog deadline", async () => {
		const server = await startFakeOllama({ stallTags: "headers" });
		const stored = storedModel(server.baseUrl);
		try {
			const provider = createOllamaProvider("local", { catalogMs: 150 });
			const harness = refreshHarness({
				allowNetwork: true,
				credential: { type: "api_key", key: "local", env: { OLLAMA_HOST: server.origin } },
				stored: [stored],
			});

			await provider.refreshModels?.(harness.context);

			// A server that accepts the connection and never answers must degrade to the stored
			// catalog, not hold the provider's list hostage forever.
			expect(provider.getModels()).toEqual([stored]);
			// The catalog is best-effort: no retry after the deadline (ADR 0004).
			expect(server.requests.filter((request) => request.path === "/api/tags")).toHaveLength(1);
		} finally {
			await server.stop();
		}
	}, 10_000);

	it("keeps the stored catalog when /api/tags sends headers and stalls the body", async () => {
		const server = await startFakeOllama({ stallTags: "body" });
		const stored = storedModel(server.baseUrl);
		try {
			const provider = createOllamaProvider("local", { catalogMs: 150 });
			const harness = refreshHarness({
				allowNetwork: true,
				credential: { type: "api_key", key: "local", env: { OLLAMA_HOST: server.origin } },
				stored: [stored],
			});

			await provider.refreshModels?.(harness.context);

			expect(provider.getModels()).toEqual([stored]);
		} finally {
			await server.stop();
		}
	}, 10_000);

	it("publishes and persists the discovered catalog when network access is allowed", async () => {
		const server = await startFakeOllama({ tags: TAGS, show: SHOW });
		try {
			const provider = createOllamaProvider("local");
			const harness = refreshHarness({
				allowNetwork: true,
				credential: { type: "api_key", key: "local", env: { OLLAMA_HOST: server.origin } },
			});

			await provider.refreshModels?.(harness.context);

			const models = provider.getModels();
			expect(models.map((model) => model.id)).toEqual(DISCOVERED_IDS);
			expect(models.every((model) => model.provider === "ollama" && model.api === "ollama-chat")).toBe(true);
			expect(models[0]?.baseUrl).toBe(server.baseUrl);
			expect(server.requests.filter((request) => request.path === "/api/tags")).toHaveLength(1);

			expect(harness.publications).toHaveLength(1);
			const persisted = harness.publications[0]?.persist;
			expect(persisted).not.toBeNull();
			expect(persisted?.models.map((model) => model.id)).toEqual(DISCOVERED_IDS);
			expect(persisted?.checkedAt).toBeTypeOf("number");
		} finally {
			await server.stop();
		}
	});

	it("restores the stored catalog without any request when offline", async () => {
		const server = await startFakeOllama({ tags: TAGS, show: SHOW });
		const stored = storedModel(server.baseUrl);
		try {
			const provider = createOllamaProvider("local");
			const harness = refreshHarness({ allowNetwork: false, stored: [stored] });

			await provider.refreshModels?.(harness.context);

			expect(provider.getModels()).toEqual([stored]);
			expect(server.requests).toHaveLength(0);
			expect(harness.publications.some((publication) => publication.persist !== undefined)).toBe(false);
		} finally {
			await server.stop();
		}
	});
});
