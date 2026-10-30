/** Reassembles a byte stream into NDJSON lines, one decoded chunk at a time. */

/**
 * The caller's idle-window policy, expressed as the two moments it cares about: the wait on the
 * server starts, and bytes arrive. The scanner arms it before each read and clears it right after,
 * so decoding, parsing and consumer time never count against the server's budget.
 */
export interface WaitWindow {
	arm(): void;
	clear(): void;
}

/**
 * Yields each trimmed, non-empty line as soon as its newline arrives, then the unterminated tail.
 *
 * `tail` carries the start of the next line across chunks and never contains a "\n", so each
 * decoded chunk is searched only within itself — the already-buffered prefix is never rescanned.
 * Appending to `tail` builds a V8 cons string that is flattened once per completed line, so total
 * work stays linear in received bytes even when a multi-MiB line arrives fragmented across many
 * chunks.
 *
 * The reader belongs to this generator: an early break on the final line still cancels the stream,
 * which releases the socket instead of leaving it open until the server closes it.
 */
export async function* readLines(
	body: ReadableStream<Uint8Array>,
	window?: WaitWindow,
): AsyncGenerator<string> {
	const reader = body.getReader();
	const decoder = new TextDecoder();
	let tail = "";
	try {
		while (true) {
			window?.arm();
			const { done, value } = await reader.read();
			window?.clear();
			if (done) break;
			const text = decoder.decode(value, { stream: true });
			let cursor = 0;
			let newline = text.indexOf("\n");
			while (newline >= 0) {
				const line = (tail + text.slice(cursor, newline)).trim();
				tail = "";
				if (line.length > 0) yield line;
				cursor = newline + 1;
				newline = text.indexOf("\n", cursor);
			}
			tail += text.slice(cursor);
		}
		tail += decoder.decode();
		const last = tail.trim();
		if (last.length > 0) yield last;
	} finally {
		window?.clear();
		await reader.cancel().catch(() => undefined);
		reader.releaseLock();
	}
}
