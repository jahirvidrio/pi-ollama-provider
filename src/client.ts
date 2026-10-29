/** Wire types and HTTP client for the native Ollama API (`/api/*`). */

export type OllamaErrorKind = "http" | "stream" | "network" | "protocol";

export class OllamaError extends Error {
	readonly kind: OllamaErrorKind;
	readonly status?: number;

	constructor(kind: OllamaErrorKind, message: string, status?: number) {
		super(message);
		this.name = "OllamaError";
		this.kind = kind;
		if (status !== undefined) this.status = status;
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

/** Same cap Pi's `error-body` helper applies, so a hostile or buggy server cannot inflate the transcript. */
const MAX_ERROR_BODY_CHARS = 4000;

function capErrorBody(text: string): string {
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
	 * Abort a chat stream that goes this long without data, measured from the request and
	 * reset on every chunk. Pi passes its own request timeout here.
	 */
	timeoutMs?: number;
	/**
	 * Called once the chat response headers arrive and before its body is consumed, so
	 * callers can observe the request the way Pi's `onResponse` hook expects. Only the
	 * chat request reports a response: the other endpoints return their body at once.
	 */
	onResponse?: (response: OllamaResponseInfo) => void | Promise<void>;
}

export class OllamaClient {
	readonly baseUrl: string;
	private readonly apiKey: string | undefined;
	private readonly fetchImpl: typeof globalThis.fetch;
	private readonly onResponse: OllamaClientOptions["onResponse"];
	private readonly timeoutMs: number | undefined;

	constructor(options: OllamaClientOptions) {
		this.baseUrl = normalizeOllamaBaseUrl(options.baseUrl);
		this.apiKey = options.apiKey;
		this.fetchImpl = options.fetch ?? globalThis.fetch;
		this.onResponse = options.onResponse;
		this.timeoutMs = options.timeoutMs;
	}

	private headers(extra?: Record<string, string>): Record<string, string> {
		const headers: Record<string, string> = { ...extra };
		if (this.apiKey) headers.authorization = `Bearer ${this.apiKey}`;
		return headers;
	}

	private async json<T>(path: string, init: RequestInit, signal?: AbortSignal): Promise<T> {
		let response: Response;
		try {
			response = await this.fetchImpl(`${this.baseUrl}${path}`, {
				...init,
				headers: this.headers(init.headers as Record<string, string> | undefined),
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
			throw new OllamaError("http", `${response.status}: ${providerMessage(text)}`, response.status);
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
				controller.abort(
					new OllamaError("network", `Ollama stream timed out after ${this.timeoutMs}ms without data`),
				);
			}, this.timeoutMs);
		};
		let response: Response;
		try {
			response = await this.fetchImpl(`${this.baseUrl}/chat`, {
				method: "POST",
				headers: this.headers({ "content-type": "application/json" }),
				body: JSON.stringify({ ...request, stream: true }),
				signal: controller.signal,
			});
		} catch (error) {
			if (isAbort(error, signal) || controller.signal.aborted) throw abortReason(error, signal, controller.signal);
			throw new OllamaError(
				"network",
				`Could not reach Ollama at ${this.baseUrl}: ${error instanceof Error ? error.message : String(error)}`,
			);
		}
		if (!response.ok) {
			const text = await response.text().catch(() => "");
			throw new OllamaError("http", `${response.status}: ${providerMessage(text)}`, response.status);
		}
		if (this.onResponse) {
			await this.onResponse({ status: response.status, headers: responseHeaders(response.headers) });
		}
		if (!response.body) {
			throw new OllamaError("protocol", "Ollama /api/chat returned no response body");
		}

		const reader = response.body.getReader();
		const decoder = new TextDecoder();
		let buffer = "";
		try {
			while (true) {
				armIdleTimer();
				const { done, value } = await reader.read();
				if (done) break;
				buffer += decoder.decode(value, { stream: true });
				let newline = buffer.indexOf("\n");
				while (newline >= 0) {
					const line = buffer.slice(0, newline).trim();
					buffer = buffer.slice(newline + 1);
					if (line.length > 0) yield this.parseChunk(line);
					newline = buffer.indexOf("\n");
				}
			}
			buffer += decoder.decode();
			const tail = buffer.trim();
			if (tail.length > 0) yield this.parseChunk(tail);
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
			signal?.removeEventListener("abort", forwardAbort);
		}
	}

	private parseChunk(line: string): OllamaChatChunk {
		let parsed: unknown;
		try {
			parsed = JSON.parse(line);
		} catch {
			throw new OllamaError("stream", `Ollama sent a malformed stream line: ${line}`);
		}
		if (parsed === null || typeof parsed !== "object") {
			throw new OllamaError("stream", `Ollama sent a non-object stream line: ${line}`);
		}
		const chunk = parsed as OllamaChatChunk;
		if (typeof chunk.error === "string") {
			throw new OllamaError("stream", chunk.error);
		}
		return chunk;
	}
}
