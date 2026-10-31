# pi-ollama-provider

Ollama inside [Pi](https://pi.dev): the models on your own machine and the models on Ollama Cloud.
It talks to Ollama directly, using Ollama's own API.

Beta, `0.1.0`.

## Install

```bash
pi install npm:pi-ollama-provider
```

Restart Pi. Nothing else to set up.

## Initialize

**Local models.** Have [Ollama](https://ollama.com/download) running on this machine. Pi finds it at
`http://localhost:11434` and needs no configuration.

**Ollama Cloud.** Create a key at [ollama.com/settings/keys](https://ollama.com/settings/keys), then
run this in Pi and paste it:

```
/login ollama-cloud
```

The key is saved in Pi's credential file (`auth.json`), so this is a one-time step.

**Choose a model.** Start Pi and run `/model`. Pi asks the server which models are available and
lists them, with the context window and reasoning levels each one really has.

> Run that once interactively. Non-interactive runs (`pi -p "..."`) read the list Pi already saved
> instead of asking the server again.

## Settings

Everything here is optional. The defaults work.

### Environment variables

| Variable | Default | What it changes |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Where the local server is, when it is not on this machine. |
| `OLLAMA_CLOUD_HOST` | `https://ollama.com` | Where cloud requests go: a proxy or a private deployment. |
| `OLLAMA_API_KEY` | — | The cloud key, for when it must not live on disk (CI, a shared machine). |

Each variable belongs to its own provider: `OLLAMA_HOST` never moves cloud, and `OLLAMA_CLOUD_HOST`
never moves local. Both accept an address with or without `/api`. The endpoint is read on every
request and never written to disk, so changing the variable takes effect immediately.

### Pi settings

In Pi's `settings.json`, in the agent directory:

| Setting | Default | What it changes |
| --- | --- | --- |
| `retry.provider.timeoutMs` | 300 s | How long a response may go quiet before the request is abandoned. |
| `retry.provider.maxRetryDelayMs` | `60000` | The longest wait honored when a server asks to retry later; `0` removes the limit. |

A server that asks to wait longer than `maxRetryDelayMs` gets its error reported right away, with
the delay it asked for appended — `(the server asked to retry after 90s)` — and Pi retries on its
own instead of holding your turn.

### Context window

Pi uses the context window Ollama reports for each model, and a local request asks the server to
allocate exactly that much. That is why a model with a very large window can be slow or fail to
load: the number is memory, not only a limit. If it happens, choose a model with a smaller window.
Cloud requests do not send it; the hosted service manages its own.

## Appendix: why this does not use `ollama-js`

Ollama ships an official JavaScript client. It was evaluated — the published `ollama@0.6.4`
artifact — and deliberately not adopted: it speaks the same `/api/*` wire and covers none of what a
Pi provider has to honor.

- **No per-request abort.** `chat()` takes no `AbortSignal`, and `abort()` cancels every in-flight
  stream on the client at once.
- **No status or headers.** The streaming path discards the `Response`, so the status code and
  `retry-after` are never visible.
- **No retry participation.** `ResponseError` carries a status code but no headers, so Pi cannot
  tell whether an error is retryable.
- **No per-chunk idle window.** One controller for the whole request, not one wait at a time.
- **Unbounded error bodies**, and a truncated stream thrown as a bare `Error`.
- **Its own environment.** It brings a `whatwg-fetch` side-effect import and reads
  `process.env.OLLAMA_API_KEY` itself, next to the key Pi resolved.

Two things would reopen this decision: `ollama-js` accepting a per-request `AbortSignal`, and its
errors carrying `status` and `headers`. Those are the only parts a client library could remove;
everything else here exists for Pi's contracts.

## License

GPL-3.0-only. The helpers under `vendor/pi-ai/` keep their MIT license (see `vendor/pi-ai/LICENSE`);
both travel in the package.
