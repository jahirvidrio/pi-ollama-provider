import { createServer, type IncomingHttpHeaders, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";

/** One queued `POST /api/chat` answer. `chunks` streams NDJSON; `body` answers as JSON. */
export interface FakeOllamaReply {
	status?: number;
	body?: unknown;
	chunks?: unknown[];
	/** Lines written verbatim, for malformed-payload tests. */
	rawChunks?: string[];
	/** Body written byte-for-byte with no added newline, for EOF-boundary tests. */
	rawBody?: string;
	/** Keep the connection open after the last chunk so callers can test aborts. */
	keepOpen?: boolean;
	/** Accept the request but never answer, to test time-to-first-byte timeouts. */
	hangHeaders?: boolean;
	/** Delay between streamed chunks, to observe incremental parsing. */
	chunkDelayMs?: number;
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

export interface FakeOllamaOptions {
	version?: string;
	tags?: unknown;
	show?: object | FakeShowResponder;
	/** First queued chat answer, equivalent to calling `respond()` right after startup. */
	chat?: FakeOllamaReply;
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

function writeJson(response: ServerResponse, status: number, payload: unknown): void {
	const text = JSON.stringify(payload);
	response.writeHead(status, {
		"content-type": "application/json",
		"content-length": Buffer.byteLength(text),
	});
	response.end(text);
}

async function writeStream(
	response: ServerResponse,
	chunks: string[],
	reply: FakeOllamaReply,
	signal: AbortSignal,
): Promise<void> {
	response.writeHead(reply.status ?? 200, { "content-type": "application/x-ndjson" });
	const delay = reply.chunkDelayMs ?? 0;
	for (const chunk of chunks) {
		if (signal.aborted) return;
		response.write(`${chunk}\n`);
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
				writeJson(response, 200, tags);
				return;
			}
			if (request.method === "POST" && path === "/api/show") {
				const payload = typeof show === "function" ? show(body) : show;
				writeJson(response, 200, payload);
				return;
			}
			if (request.method === "POST" && path === "/api/chat") {
				const reply = replies.shift() ?? { chunks: [DEFAULT_DONE_CHUNK] };
				if (reply.hangHeaders) return;
				if (reply.rawBody !== undefined) {
					response.writeHead(reply.status ?? 200, { "content-type": "application/x-ndjson" });
					response.write(reply.rawBody);
					if (!reply.keepOpen) response.end();
					return;
				}
				if (reply.rawChunks || reply.chunks) {
					const lines = reply.rawChunks ?? (reply.chunks ?? []).map((chunk) => JSON.stringify(chunk));
					await writeStream(response, lines, reply, controller.signal);
					return;
				}
				writeJson(response, reply.status ?? 200, reply.body ?? {});
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
