# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-31

First published version. Beta: what this package ships is the Pi provider contract, which can move.

### Added

- Two providers from one extension: `ollama` (a local server, no credential) and `ollama-cloud`
  (`OLLAMA_API_KEY`, or a key stored by `/login ollama-cloud`). Each reads its own host variable —
  `OLLAMA_HOST` and `OLLAMA_CLOUD_HOST` — at request time, so a private deployment, an
  Ollama-compatible gateway or a proxy is configuration and never a code change.
- The endpoint never persists: `/login ollama-cloud` stores the key only, `auth.json` carries no
  host, and `/login ollama` no longer exists because a local server is always configured.
- A dynamic catalog from `/api/tags` plus `/api/show`: the model's real context window, its own
  thinking levels and its vision capability, restored from disk when the server is unreachable.
- Native `/api/chat` streaming: NDJSON chunks translated into Pi assistant events, thinking kept as
  its own content block, and tool calls buffered until the completion chunk so a call split across
  chunks is emitted once with its arguments validated by Pi's parser.
- Participation in Pi's retry policy: `OllamaError` carries `status` and `headers`, the
  establishment of a chat response goes through the shared helper (`retry-after`,
  `x-should-retry`), and an idle timeout is a provider-shaped `408` that is deliberately not retried.
- Request headers computed by Pi merged into every call, including the catalog; a `null` value
  suppresses one of the adapter's defaults.
- Deadlines on all non-chat traffic: the idle window for chat (`timeoutMs`), plus budgets for the
  catalog refresh and the login probe.
- Bounded error payloads: 4 KB for what reaches the transcript, a 64 KB parse budget for a JSON
  envelope, and the same cap on a mid-stream error chunk or a malformed NDJSON line.
- `transformMessages` applied before the conversion, so an aborted turn does not replay a tool call
  without its result and an image reaches only a model that can read it.

### Known limits

- Ollama returns tool calls without ids and matches tool results positionally; ids are synthesized
  per assistant message (`ollama_call_0`, ...).
- Images inside tool results are dropped with an explicit text marker.
- `num_ctx` is sent to the local server only. A large `contextWindow` is an allocation, not a
  limit: the README's "Known limits" lists the escape hatches.
- `timeoutMs` measures server waits only, and a server-requested `retry-after` above the retry
  budget is reported rather than waited out, unless `retry.provider.maxRetryDelayMs` is `0`.

### Verification

- 111 tests against a fake Ollama server, `tsc --noEmit` clean on TypeScript 6.0.3, and the six
  vendored helpers checked against `@earendil-works/pi-ai` 1.4.0 by `npm run vendor:check`.
- Live against Ollama Cloud: catalog discovery, text, thinking levels and a tool call.
- The local provider is covered by the contract tests, not by a live `ollama serve` on this machine.
