/** Wire types and HTTP client for the native Ollama API (`/api/*`). */

import { retryProviderRequest } from "../vendor/pi-ai/utils/provider-retry.js";

export type OllamaErrorKind = "http" | "stream" | "network" | "protocol";

export class OllamaError extends Error {
	readonly kind: OllamaErrorKind;
	/**
	 * HTTP status and response headers, when a response arrived at all. Both are always own
	 * properties, even when undefined: Pi's `isProviderError` probe requires the keys to be
	 * present before it reads `status`, and its retry policy takes `retry-after` and
	 * `x-should-retry` from these headers.
	 */
	readonly status: number | undefined;
	readonly headers: Headers | undefined;

	constructor(kind: OllamaErrorKind, message: string, status?: number, headers?: Headers) {
		super(message);
		this.name = "OllamaError";
		this.kind = kind;
		this.status = status;
		this.headers = headers;
	}
}

export interface OllamaModelSummary {
	name: string;
	model: string;
	modified_at?: string;
	size?: number;
	digest?: string;
	remote_model?: string;
	remote_host?: string;
	details?: {
		format?: string;
		family?: string;
		families?: string[] | null;
		parameter_size?: string;
		quantization_level?: string;
	};
	capabilities?: string[];
}

export interface OllamaShowResponse {
	capabilities?: string[];
	model_info?: Record<string, unknown>;
	thinking?: { values?: (boolean | string)[]; default?: boolean | string };
	parameters?: string;
	details?: OllamaModelSummary["details"];
}

export interface OllamaToolCall {
	function: { name: string; arguments?: Record<string, unknown> };
}

export interface OllamaChatMessage {
	role: "system" | "user" | "assistant" | "tool";
	content: string;
	images?: string[];
	thinking?: string;
	tool_calls?: OllamaToolCall[];
}

export interface OllamaToolDefinition {
	type: "function";
	function: { name: string; description?: string; parameters: Record<string, unknown> };
}

export interface OllamaChatRequest {
	model: string;
	messages: OllamaChatMessage[];
	tools?: OllamaToolDefinition[];
	think?: boolean | string;
	keep_alive?: string | number;
	options?: Record<string, unknown>;
}

export interface OllamaChatChunk {
	model?: string;
	created_at?: string;
	message?: {
		role?: string;
		content?: string;
		thinking?: string;
		tool_calls?: OllamaToolCall[];
		images?: string[];
	};
	done?: boolean;
	done_reason?: string;
	total_duration?: number;
	load_duration?: number;
	prompt_eval_count?: number;
	prompt_eval_cached_count?: number;
	prompt_eval_duration?: number;
	eval_count?: number;
	eval_duration?: number;
	error?: string;
}

/**
 * Accepts an origin, an origin with a path, or an `/api` base URL and returns the
 * canonical `/api` endpoint. Pi stores whatever the user typed during `/login`.
 */
export function normalizeOllamaBaseUrl(input: string): string {
	const trimmed = input.trim().replace(/\/+$/, "");
	return trimmed.endsWith("/api") ? trimmed : `${trimmed}/api`;
}

function isAbort(error: unknown, signal: AbortSignal | undefined): boolean {
	if (signal?.aborted) return true;
	return error instanceof Error && error.name === "AbortError";
}

/**
 * Chooses what to throw when a request or read stopped early: the caller's own abort keeps its
 * `AbortError`, while a stream aborted by the idle timer keeps the `OllamaError` that names it.
 */
function abortReason(
	error: unknown,
	signal: AbortSignal | undefined,
	controllerSignal: AbortSignal,
): unknown {
	if (signal?.aborted) return signal.reason ?? error;
	const reason = controllerSignal.reason;
	return reason instanceof OllamaError ? reason : error;
}

/**
 * Same cap Pi's `error-body` helper applies, so a hostile or buggy server cannot inflate the
 * transcript. Reused for an error chunk and for a malformed stream line: both end up in
 * `errorMessage`, which the transcript persists, the TUI renders and Pi matches its error
 * patterns against.
 */
const MAX_ERROR_BODY_CHARS = 4000;

export function capErrorBody(text: string): string {
	const trimmed = text.trim();
	if (trimmed.length <= MAX_ERROR_BODY_CHARS) return trimmed;
	return `${trimmed.slice(0, MAX_ERROR_BODY_CHARS)}… [truncated ${trimmed.length - MAX_ERROR_BODY_CHARS} chars]`;
}

function providerMessage(text: string): string {
	const capped = capErrorBody(text);
	try {
		const parsed = JSON.parse(capped) as { error?: unknown };
		if (typeof parsed.error === "string") return capErrorBody(parsed.error);
	} catch {
		// Not JSON: the capped raw body is the best description available.
	}
	return capped;
}

/** Status and headers of an Ollama HTTP response, mirroring Pi's `ProviderResponse`. */
export interface OllamaResponseInfo {
	status: number;
	headers: Record<string, string>;
}

function responseHeaders(headers: Headers): Record<string, string> {
	const record: Record<string, string> = {};
	headers.forEach((value, key) => {
		record[key] = value;
	});
	return record;
}

export interface OllamaClientOptions {
	/** Origin or `/api` base URL; normalized before use. */
	baseUrl: string;
	apiKey?: string;
	/** Fetch implementation supplied by Pi for the current request. */
	fetch?: typeof globalThis.fetch;
	/**
	 * Abort a chat request when the server goes this long without progress: from the
	 * request send until the response headers arrive, and then while awaiting each body
	 * chunk. Caller work (the `onResponse` hook and consuming yielded chunks) never counts
	 * against it. Pi passes its own request timeout here.
	 */
	timeoutMs?: number;
	/**
	 * Headers Pi computed for this request (`ProviderRequestOptions.headers`): session, auth,
	 * caller and `transformHeaders` values, already merged. They win over the adapter's own
	 * defaults and a `null` value removes one, which is the rule Pi applies to its defaults.
	 * Names are matched case-insensitively. Every request of the client honors them, the
	 * catalog included.
	 */
	headers?: Record<string, string | null>;
	/**
	 * Pi's client-side retry budget for establishing a chat response. Defaults to
	 * {@link DEFAULT_MAX_RETRIES}, the same budget the OpenAI and Anthropic SDKs use.
	 */
	maxRetries?: number;
	/** Pi's cap on a server-requested retry delay; `0` disables the cap, absent keeps the helper's 60s. */
	maxRetryDelayMs?: number;
	/**
	 * Called once the chat response headers arrive and before its body is consumed, so
	 * callers can observe the request the way Pi's `onResponse` hook expects. Only the
	 * chat request reports a response: the other endpoints return their body at once.
	 */
	onResponse?: (response: OllamaResponseInfo) => void | Promise<void>;
}

/** Same budget the OpenAI and Anthropic SDKs retry with, which Pi's helper mirrors. */
const DEFAULT_MAX_RETRIES = 2;

/**
 * Status that marks an idle timeout. It is a provider-shaped error so Pi's retry policy sees
 * it, but `chat` lists it in `noRetryStatuses`: the wait is already bounded by `timeoutMs`, and
 * retrying would turn a five minute stall into fifteen. A server that genuinely answers 408 is
 * therefore not retried either; Ollama does not send one.
 */
const IDLE_TIMEOUT_STATUS = 408;

export class OllamaClient {
	readonly baseUrl: string;
	private readonly apiKey: string | undefined;
	private readonly fetchImpl: typeof globalThis.fetch;
	private readonly onResponse: OllamaClientOptions["onResponse"];
	private readonly timeoutMs: number | undefined;
	private readonly requestHeaderOverrides: Record<string, string | null> | undefined;
	private readonly maxRetries: number;
	private readonly maxRetryDelayMs: number | undefined;

	constructor(options: OllamaClientOptions) {
		this.baseUrl = normalizeOllamaBaseUrl(options.baseUrl);
		this.apiKey = options.apiKey;
		this.fetchImpl = options.fetch ?? globalThis.fetch;
		this.onResponse = options.onResponse;
		this.timeoutMs = options.timeoutMs;
		this.requestHeaderOverrides = options.headers;
		this.maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
		this.maxRetryDelayMs = options.maxRetryDelayMs;
	}

	/**
	 * The headers of one request: the adapter's defaults, then Pi's request headers, which win
	 * and where `null` removes a default — the same rule `providerHeadersToRecord` applies.
	 * Keys are deduplicated case-insensitively, keeping the last spelling.
	 */
	private requestHeaders(extra?: Record<string, string | null>): Record<string, string> {
		const merged = new Map<string, [string, string]>();
		const apply = (source: Record<string, string | null> | undefined): void => {
			for (const [name, value] of Object.entries(source ?? {})) {
				merged.delete(name.toLowerCase());
				if (value !== null) merged.set(name.toLowerCase(), [name, value]);
			}
		};
		apply(extra);
		if (this.apiKey) apply({ authorization: `Bearer ${this.apiKey}` });
		apply(this.requestHeaderOverrides);
		return Object.fromEntries(merged.values());
	}

	/** The error that closes one idle window; see {@link IDLE_TIMEOUT_STATUS}. */
	private timeoutError(): OllamaError {
		return new OllamaError(
			"network",
			`Ollama stream timed out after ${this.timeoutMs}ms without data`,
			IDLE_TIMEOUT_STATUS,
		);
	}

	private async json<T>(path: string, init: RequestInit, signal?: AbortSignal): Promise<T> {
		let response: Response;
		try {
			response = await this.fetchImpl(`${this.baseUrl}${path}`, {
				...init,
				headers: this.requestHeaders(init.headers as Record<string, string | null> | undefined),
				signal,
			});
		} catch (error) {
			if (isAbort(error, signal)) throw error;
			throw new OllamaError(
				"network",
				`Could not reach Ollama at ${this.baseUrl}: ${error instanceof Error ? error.message : String(error)}`,
			);
		}
		if (!response.ok) {
			const text = await response.text().catch(() => "");
			throw new OllamaError(
				"http",
				`${response.status}: ${providerMessage(text)}`,
				response.status,
				response.headers,
			);
		}
		try {
			return (await response.json()) as T;
		} catch (error) {
			throw new OllamaError(
				"protocol",
				`Ollama returned an invalid JSON body: ${error instanceof Error ? error.message : String(error)}`,
			);
		}
	}

	async version(signal?: AbortSignal): Promise<string> {
		const payload = await this.json<{ version?: unknown }>("/version", { method: "GET" }, signal);
		return typeof payload.version === "string" ? payload.version : "";
	}

	async listModels(signal?: AbortSignal): Promise<OllamaModelSummary[]> {
		const payload = await this.json<{ models?: unknown }>("/tags", { method: "GET" }, signal);
		if (!Array.isArray(payload.models)) {
			throw new OllamaError("protocol", "Ollama /api/tags response has no models array");
		}
		return payload.models as OllamaModelSummary[];
	}

	async showModel(model: string, signal?: AbortSignal): Promise<OllamaShowResponse> {
		return this.json<OllamaShowResponse>(
			"/show",
			{
				method: "POST",
				headers: { "content-type": "application/json" },
				body: JSON.stringify({ model }),
			},
			signal,
		);
	}

	/**
	 * Streams one chat completion as decoded NDJSON chunks. Chunks are yielded as soon
	 * as their line arrives; an `error` chunk or a malformed line throws `OllamaError`
	 * after the chunks that already arrived, because Ollama reports mid-stream failures
	 * inside the stream and never changes the HTTP status.
	 *
	 * Establishing the response follows Pi's retry policy (`maxRetries`, `maxRetryDelayMs`,
	 * `retry-after`, `x-should-retry`). A connection failure is retried: it is the typical
	 * transient case. An idle timeout is not, because `timeoutMs` already bounds it and
	 * retrying would multiply that bounded wait by the retry budget. Once the body is being
	 * consumed nothing is retried, which is the same rule `openai-node` applies to a
	 * streaming call.
	 */
	async *chat(request: OllamaChatRequest, signal?: AbortSignal): AsyncGenerator<OllamaChatChunk> {
		const controller = new AbortController();
		const forwardAbort = (): void => controller.abort(signal?.reason);
		signal?.addEventListener("abort", forwardAbort, { once: true });
		// A signal that is already aborted never fires the listener, so mirror the caller's state
		// before the fetch instead of letting a cancelled request start.
		if (signal?.aborted) forwardAbort();
		let idleTimer: ReturnType<typeof setTimeout> | undefined;
		const clearIdleTimer = (): void => {
			if (idleTimer) clearTimeout(idleTimer);
			idleTimer = undefined;
		};
		const armIdleTimer = (): void => {
			if (this.timeoutMs === undefined || this.timeoutMs <= 0) return;
			clearIdleTimer();
			idleTimer = setTimeout(() => {
				controller.abort(this.timeoutError());
			}, this.timeoutMs);
		};
		/**
		 * One attempt at establishing the response, the only part a retry can replay: once the body
		 * is being consumed a second request would duplicate the text already yielded.
		 */
		const attempt = async (): Promise<Response> => {
			// Start the window before the fetch: a server that accepts the connection but never
			// answers (hung proxy, model still loading before its first byte) would otherwise stall
			// the request forever, because the read loop that arms the timer only starts once the
			// headers arrive.
			armIdleTimer();
			try {
				const response = await this.fetchImpl(`${this.baseUrl}/chat`, {
					method: "POST",
					headers: this.requestHeaders({ "content-type": "application/json" }),
					body: JSON.stringify({ ...request, stream: true }),
					signal: controller.signal,
				});
				// Headers are progress too: restart the window so a stalled error body keeps its own
				// budget. The read loop re-arms it on each awaited chunk.
				armIdleTimer();
				if (response.ok) return response;
				// A stalled error body surfaces the timeout that stopped it instead of an empty
				// status line; other read failures keep the status-only message.
				const text = await response.text().catch((error: unknown) => {
					if (controller.signal.aborted) throw abortReason(error, signal, controller.signal);
					return "";
				});
				throw new OllamaError(
					"http",
					`${response.status}: ${providerMessage(text)}`,
					response.status,
					response.headers,
				);
			} catch (error) {
				// One rule for an interrupted attempt, whichever side stopped it: the caller's own reason
				// wins, then the idle timeout that aborted this attempt, then the raw failure. Reading the
				// two signals separately let the result depend on which abort happened to land first.
				if (signal?.aborted || controller.signal.aborted) {
					throw abortReason(error, signal, controller.signal);
				}
				if (error instanceof OllamaError) throw error;
				// No response at all: the transient failure the retry policy exists to ride out.
				throw new OllamaError(
					"network",
					`Could not reach Ollama at ${this.baseUrl}: ${error instanceof Error ? error.message : String(error)}`,
				);
			} finally {
				// The window covers one attempt: the wait between attempts is backoff, not the
				// server's silence, so a stale timer must not fire during the sleep.
				clearIdleTimer();
			}
		};
		let response: Response;
		try {
			response = await retryProviderRequest(attempt, {
				maxRetries: this.maxRetries,
				maxRetryDelayMs: this.maxRetryDelayMs,
				// The caller's abort interrupts a backoff; the attempt controller belongs to one attempt.
				signal,
				noRetryStatuses: [IDLE_TIMEOUT_STATUS],
			});
			if (this.onResponse) {
				// Caller work never counts against the server: pause the window while the hook runs
				// and let the read loop start it again right before the first chunk.
				clearIdleTimer();
				await this.onResponse({ status: response.status, headers: responseHeaders(response.headers) });
			}
			if (!response.body) {
				throw new OllamaError("protocol", "Ollama /api/chat returned no response body");
			}

			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			// Incremental line scan: `tail` carries the unterminated start of the next line
			// across chunks and never contains a "\n", so each decoded chunk is searched
			// only within itself — the already-buffered prefix is never rescanned.
			// Appending to `tail` builds a V8 cons string that is flattened once per
			// completed line, so total work stays linear in received bytes even when a
			// multi-MiB line arrives fragmented across many chunks.
			let tail = "";
			try {
				while (true) {
					// Hold the window only while the client waits on the server: clearing it as soon
					// as a chunk lands keeps decoding, parsing, and consumer time out of the budget.
					armIdleTimer();
					const { done, value } = await reader.read();
					clearIdleTimer();
					if (done) break;
					const text = decoder.decode(value, { stream: true });
					let cursor = 0;
					let newline = text.indexOf("\n");
					while (newline >= 0) {
						const line = (tail + text.slice(cursor, newline)).trim();
						tail = "";
						if (line.length > 0) yield this.parseChunk(line);
						cursor = newline + 1;
						newline = text.indexOf("\n", cursor);
					}
					tail += text.slice(cursor);
				}
				tail += decoder.decode();
				const last = tail.trim();
				if (last.length > 0) yield this.parseChunk(last);
			} catch (error) {
				if (isAbort(error, signal) || controller.signal.aborted) throw abortReason(error, signal, controller.signal);
				if (error instanceof OllamaError) throw error;
				throw new OllamaError(
					"stream",
					`Ollama chat stream failed: ${error instanceof Error ? error.message : String(error)}`,
				);
			} finally {
				clearIdleTimer();
				// Release the socket instead of only unlocking the reader, so an early break on the
				// final chunk does not leave a connection open until the server closes it.
				await reader.cancel().catch(() => undefined);
				reader.releaseLock();
			}
		} finally {
			// The window can be armed before the fetch and the caller abort listener is attached
			// even earlier, so every exit path (failed fetch, HTTP error, missing body, early
			// generator return) must release both.
			clearIdleTimer();
			signal?.removeEventListener("abort", forwardAbort);
		}
	}

	private parseChunk(line: string): OllamaChatChunk {
		let parsed: unknown;
		try {
			parsed = JSON.parse(line);
		} catch {
			throw new OllamaError("stream", `Ollama sent a malformed stream line: ${capErrorBody(line)}`);
		}
		if (parsed === null || typeof parsed !== "object") {
			throw new OllamaError("stream", `Ollama sent a non-object stream line: ${capErrorBody(line)}`);
		}
		const chunk = parsed as OllamaChatChunk;
		if (typeof chunk.error === "string") {
			throw new OllamaError("stream", capErrorBody(chunk.error));
		}
		return chunk;
	}
}

/** The part of a resolved credential the client needs: the key Pi picked for this request, if any. */
export interface OllamaCredential {
	apiKey?: string;
}

/** Every client option `createClient` does not take positionally. */
export type OllamaClientOverrides = Omit<OllamaClientOptions, "baseUrl" | "apiKey">;

/**
 * The single place an `OllamaClient` is built, so a new option is honored everywhere by
 * construction instead of by memory. Defaults live in the class: `fetch` falls back to the
 * global implementation and `maxRetries` to the SDK budget. The logins, the catalog refresh and
 * the chat stream each used to build the client by hand, which is how `timeoutMs` ended up
 * missing from three of the four sites.
 */
export function createClient(
	baseUrl: string,
	credential?: OllamaCredential,
	overrides?: OllamaClientOverrides,
): OllamaClient {
	return new OllamaClient({ baseUrl, apiKey: credential?.apiKey, ...overrides });
}
