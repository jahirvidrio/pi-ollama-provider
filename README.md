# pi-ollama-provider

Native [Pi](https://pi.dev) provider for Ollama: local models and Ollama Cloud, speaking
Ollama's own API (`/api/chat`, `/api/tags`, `/api/show`). No OpenAI-compatible shim, no
third-party gateway, no SDK.

Two independent providers are registered:

| Provider | Endpoint | Authentication |
| --- | --- | --- |
| `ollama` | `http://localhost:11434/api` | none (local server) |
| `ollama-cloud` | `https://ollama.com/api` | `OLLAMA_API_KEY` or a stored credential |

## Install

```bash
pi install npm:pi-ollama-provider
```

Or try it for one run:

```bash
pi -e ./extensions/ollama/index.ts
```

## Configure

Cloud needs a key. Either run `/login ollama-cloud` and paste it, or export it:

```bash
export OLLAMA_API_KEY=...
```

Local works with no configuration. Point it elsewhere with `OLLAMA_HOST`:

```bash
export OLLAMA_HOST=http://192.168.1.10:11434
```

A stored credential can carry the host as well, in `auth.json`:

```json
{
  "ollama-cloud": {
    "type": "api_key",
    "key": "...",
    "env": { "OLLAMA_HOST": "https://ollama.com/api" }
  }
}
```

## Use the catalog

Models are discovered from the server, never hard-coded. `/api/tags` lists them and
`/api/show` supplies the real `context_length` and the model's own `thinking.values`, so Pi
reports the true context window and offers only the reasoning levels the model accepts.

> **Limitation in Pi 1.4.0:** non-interactive modes do not refresh dynamic providers.
> `--list-models`, `auth check`, `--print`, `--mode rpc` and `pi update --models` either never
> call the provider's refresh or reject the flag, so the first run must be an interactive `pi`
> session (or `/model`) to populate `models-store.json`. After that, `-p` and RPC see the
> catalog from the store. This is Pi's behavior, not this package's.

## How requests map

| Pi | Ollama |
| --- | --- |
| `contextWindow` | `options.num_ctx` (local only) |
| `maxTokens` | `options.num_predict` |
| thinking level | `think` (`false` off, the model's own level name, or `true` when it is boolean-only) |
| `samplingParams` | merged into `options` last, so they override anything above |
| tool declarations | `tools[].function` JSON Schema |
| tool results | `messages[]` with `role: "tool"` |

`/api/chat` streams NDJSON, and the final chunk carries the usage Pi accounts for:
`prompt_eval_count`, `prompt_eval_cached_count` (reported as cache reads) and `eval_count`.
Ollama publishes no per-token pricing, so cost is zero.

## Why this adapter does not use `ollama-js`

Ollama ships an official JavaScript client (`ollama-js`, MIT, still pre-1.0). It was evaluated
and deliberately **not** adopted: it speaks the same `/api/*` wire this adapter already speaks and
implements none of the contracts a Pi provider has to honor — the ones the retry, header, timeout
and error handling in `src/client.ts` exists for.

On the published `ollama@0.6.4` artifact:

| Pi contract | `ollama-js` 0.6.4 |
| --- | --- |
| Per-request abort (`AbortSignal`) | `chat(request)` accepts no signal; streaming uses its own controller, and `abort()` cancels **every** in-flight stream on the client. |
| `onResponse` before the body | The streaming path discards the `Response`, so status and headers are never observable. |
| Retry policy participation | No retries, and `ResponseError(message, status_code)` carries no `headers`, so Pi's `isProviderError` probe rejects it. |
| Idle window per chunk (`timeoutMs`) | One `AbortController` for the whole request; the NDJSON reader only awaits `read()`. |
| Bounded error payload | `checkOk` reads and parses the whole error body before any cap and logs on that path; a malformed stream line is `console.warn`ed and skipped. |
| Retryable end of stream | Throws a bare `Error("Did not receive done or success response in stream.")`. |
| Caller headers win | `Config.headers` is per client, not per request, and its `fetch` wrapper replaces caller `Content-Type`/`Accept`/`User-Agent` with `User-Agent: ollama-js/<version>`. |

It also adds a `whatwg-fetch` runtime dependency with a side-effect import, and reads
`process.env.OLLAMA_API_KEY` on its own for `ollama.com`, bypassing the credential Pi resolved.
What it does not save is the actual adapter work: chunk-to-event translation, usage accounting and
the transcript transform stay here either way.

The full record — the rejected alternatives and the trigger that would reopen this decision — is
ADR 0003 in the project's decision log.

## Known limits

- Ollama returns tool calls without ids and matches tool results positionally, so Pi's ids are
  synthesized per assistant message (`ollama_call_0`, ...). They are stable within a session but
  not across providers.
- Images inside tool results are dropped with an explicit text marker, because Ollama tool
  messages take text only. User images are sent normally.
- A stream that closes without a `done` chunk is reported as an error, never as a completed
  response.
- Cloud requests do not send `num_ctx`: the hosted service manages its own context.
- **A local context window is an allocation, not just a limit.** `model.contextWindow` travels as
  `options.num_ctx`, so asking a model for a window larger than the memory it can get either stops
  it from loading or spills it to CPU. Two escape hatches: `samplingParams.num_ctx` (merged last,
  so it wins) or a smaller `contextWindow` in the catalog. This is one of the reasons the adapter
  speaks `/api/chat` instead of the OpenAI-compatible `/v1` shim, which drops `num_ctx` and
  replaces the per-chunk idle window with a global one.
- **`timeoutMs` measures the server, not the caller.** The budget covers the wait from the request
  until the response headers and then each wait between two chunks; decoding, consumer work and
  `onResponse` never count against it. Pi passes its own budget here — `retry.provider.timeoutMs`
  when set, otherwise the `httpIdleTimeoutMs` global setting (300 s by default) — so a body that
  stalls between chunks is bounded by that value, not by the adapter.

## Vendored helpers

Two pure helpers from `@earendil-works/pi-ai` live in `vendor/pi-ai/`, byte-identical to the
installed package and copied rather than imported:

| Vendored file | What it provides |
| --- | --- |
| `api/transform-messages.js` | `transformMessages`: drops errored and aborted turns, closes orphaned tool calls, downgrades images the model cannot read |
| `utils/provider-retry.js` | `retryProviderRequest`: the OpenAI/Anthropic retry policy (408/409/429/5xx, `retry-after`, `x-should-retry`) with an interruptible backoff |

Both are MIT (see `vendor/pi-ai/LICENSE`) and both ship in the package, because `files` includes
`vendor/`. Copying is a loader constraint, not a preference: Pi's extension loader maps only
`@earendil-works/pi-ai` and its four documented subpaths (`/compat`, `/models`, `/oauth`,
`/providers/all`), so an unmapped subpath would load a second copy of `pi-ai` into the extension in
a source build and fail outright in the compiled binary, where modules are virtual and nothing
exists on disk to import.

`npm run vendor:check` hashes the six vendored files against the installed `pi-ai` and fails on
drift; when it does, read the upstream change and copy the new bytes over instead of editing the
copies. If upstream ever exports both functions from the package root, delete `vendor/` and import
them.

## Development

```bash
npm install
npm test          # vitest, offline, against a fake Ollama server
npm run typecheck
npm run vendor:check   # the vendored Pi helpers still match the installed pi-ai
```

The tests replay recorded NDJSON through `tests/helpers/fake-ollama.ts`; no server or key is
needed for them.

## License

MIT
