import { createServer, type IncomingHttpHeaders, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";

/** One queued `POST /api/chat` answer. `chunks` streams NDJSON; `body` answers as JSON. */
export interface FakeOllamaReply {
	status?: number;
	body?: unknown;
	chunks?: unknown[];
	/**
	 * Lines written verbatim, for malformed-payload tests. `Uint8Array` chunks are
	 * written as-is so tests control the exact byte boundaries of the stream;
	 * string lines keep the trailing-newline behavior of `chunks`.
	 */
	rawChunks?: (string | Uint8Array)[];
	/** Body written byte-for-byte with no added newline, for EOF-boundary tests. */
	rawBody?: string;
	/** Keep the connection open after the last chunk so callers can test aborts. */
	keepOpen?: boolean;
	/** Accept the request but never answer, to test time-to-first-byte timeouts. */
	hangHeaders?: boolean;
	/** Send the headers then never write a body, to test body-idle timeouts. */
	hangBody?: boolean;
	/** Delay between streamed chunks, to observe incremental parsing. */
	chunkDelayMs?: number;
	/** Extra response headers, e.g. `retry-after` for a retry test. */
	headers?: Record<string, string>;
}

export interface RecordedRequest {
	method: string;
	path: string;
	headers: IncomingHttpHeaders;
	body: unknown;
}

/**
 * `/api/show` payload: either a fixed object, or a function of the parsed request body
 * for tests that need a different payload per model.
 */
export type FakeShowResponder = (body: Record<string, unknown>) => unknown;

/**
 * How a stalled endpoint answers: never send the headers, or send them and never finish the body,
 * so a client either waits to establish the response or waits for its payload.
 */
export type FakeStall = "headers" | "body";

export interface FakeOllamaOptions {
	version?: string;
	tags?: unknown;
	show?: object | FakeShowResponder;
	/** First queued chat answer, equivalent to calling `respond()` right after startup. */
	chat?: FakeOllamaReply;
	/** Stall `GET /api/tags`, the catalog and cloud login probe. */
	stallTags?: FakeStall;
	/** Stall `POST /api/show` for the models named here, leaving the rest of the catalog intact. */
	stallShow?: (model: string) => FakeStall | undefined;
}

export interface FakeOllama {
	/** `http://127.0.0.1:<port>/api`, the value Pi stores as a provider base URL. */
	baseUrl: string;
	/** `http://127.0.0.1:<port>`, without the `/api` suffix. */
	origin: string;
	requests: RecordedRequest[];
	/** Queue the next `POST /api/chat` answer. */
	respond(reply: FakeOllamaReply): void;
	stop(): Promise<void>;
}

const DEFAULT_DONE_CHUNK = { message: { role: "assistant", content: "" }, done: true, done_reason: "stop" };

async function readBody(request: IncomingMessage): Promise<unknown> {
	const parts: Buffer[] = [];
	for await (const part of request) parts.push(part as Buffer);
	const text = Buffer.concat(parts).toString("utf8");
	if (text.length === 0) return undefined;
	try {
		return JSON.parse(text);
	} catch {
		return text;
	}
}

function writeJson(
	response: ServerResponse,
	status: number,
	payload: unknown,
	headers?: Record<string, string>,
): void {
	const text = JSON.stringify(payload);
	response.writeHead(status, {
		"content-type": "application/json",
		"content-length": Buffer.byteLength(text),
		...headers,
	});
	response.end(text);
}

/** Leave a response hanging in the phase named by `mode`, until the client drops the socket. */
function stall(response: ServerResponse, mode: FakeStall): void {
	// A "headers" stall is the absence of a response: doing nothing leaves the client waiting to
	// establish it, which is exactly the phase this mode names.
	if (mode === "headers") return;
	response.writeHead(200, { "content-type": "application/json" });
	response.flushHeaders();
}

async function writeStream(
	response: ServerResponse,
	chunks: (string | Uint8Array)[],
	reply: FakeOllamaReply,
	signal: AbortSignal,
): Promise<void> {
	response.writeHead(reply.status ?? 200, { "content-type": "application/x-ndjson", ...reply.headers });
	const delay = reply.chunkDelayMs ?? 0;
	for (const chunk of chunks) {
		if (signal.aborted) return;
		// Byte chunks are written without an added newline so a test can split one
		// line across many chunks; string lines keep the historical behavior.
		if (typeof chunk === "string") {
			response.write(`${chunk}\n`);
		} else {
			response.write(chunk);
		}
		if (delay > 0) {
			await new Promise((resolve) => setTimeout(resolve, delay));
		}
	}
	if (!reply.keepOpen) response.end();
}

export async function startFakeOllama(options: FakeOllamaOptions = {}): Promise<FakeOllama> {
	let tags: unknown = options.tags ?? { models: [] };
	let show: unknown = options.show ?? {};
	const replies: FakeOllamaReply[] = options.chat ? [options.chat] : [];
	const requests: RecordedRequest[] = [];
	const inFlight = new Set<AbortController>();

	const server = createServer((request, response) => {
		const controller = new AbortController();
		inFlight.add(controller);
		response.on("close", () => {
			controller.abort();
			inFlight.delete(controller);
		});

		void (async () => {
			const path = (request.url ?? "/").split("?")[0] ?? "/";
			const body = await readBody(request);
			requests.push({ method: request.method ?? "GET", path, headers: request.headers, body });

			if (request.method === "GET" && path === "/api/version") {
				writeJson(response, 200, { version: options.version ?? "0.12.6" });
				return;
			}
			if (request.method === "GET" && path === "/api/tags") {
				if (options.stallTags) {
					stall(response, options.stallTags);
					return;
				}
				writeJson(response, 200, tags);
				return;
			}
			if (request.method === "POST" && path === "/api/show") {
				const model = (body as { model?: unknown } | undefined)?.model;
				const stallMode = typeof model === "string" ? options.stallShow?.(model) : undefined;
				if (stallMode) {
					stall(response, stallMode);
					return;
				}
				const payload = typeof show === "function" ? show(body as Record<string, unknown>) : show;
				writeJson(response, 200, payload);
				return;
			}
			if (request.method === "POST" && path === "/api/chat") {
				const reply = replies.shift() ?? { chunks: [DEFAULT_DONE_CHUNK] };
				if (reply.hangHeaders) return;
				if (reply.hangBody) {
					response.writeHead(reply.status ?? 200, { "content-type": "application/x-ndjson", ...reply.headers });
					response.flushHeaders();
					return;
				}
				if (reply.rawBody !== undefined) {
					response.writeHead(reply.status ?? 200, { "content-type": "application/x-ndjson", ...reply.headers });
					response.write(reply.rawBody);
					if (!reply.keepOpen) response.end();
					return;
				}
				if (reply.rawChunks || reply.chunks) {
					const lines = reply.rawChunks ?? (reply.chunks ?? []).map((chunk) => JSON.stringify(chunk));
					await writeStream(response, lines, reply, controller.signal);
					return;
				}
				writeJson(response, reply.status ?? 200, reply.body ?? {}, reply.headers);
				return;
			}
			writeJson(response, 404, { error: "not found" });
		})();
	});

	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const address = server.address() as AddressInfo;
	const origin = `http://127.0.0.1:${address.port}`;

	return {
		baseUrl: `${origin}/api`,
		origin,
		requests,
		respond: (reply) => {
			replies.push(reply);
		},
		stop: async () => {
			for (const controller of inFlight) controller.abort();
			inFlight.clear();
			server.closeAllConnections();
			await new Promise<void>((resolve) => server.close(() => resolve()));
		},
	};
}
