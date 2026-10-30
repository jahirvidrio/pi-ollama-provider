# Vendored from `@earendil-works/pi-ai`

These files are **byte-identical copies** taken from the host package, so the adapter can reuse
Pi's own contract helpers without copying its provider layer.

| Vendored file | Upstream path | Purpose |
| --- | --- | --- |
| `api/transform-messages.js` (+ `.d.ts`) | `dist/api/transform-messages.js` | `transformMessages`: drops error/aborted assistant turns, closes orphaned tool calls, downgrades images a model cannot accept, drops foreign thinking |
| `utils/provider-retry.js` (+ `.d.ts`) | `dist/utils/provider-retry.js` | `retryProviderRequest`: mirrors the OpenAI/Anthropic SDK retry policy (408/409/429/5xx, `retry-after`, `x-should-retry`) with an interruptible backoff |

- **Origin**: `@earendil-works/pi-ai` 1.4.0 — <https://github.com/earendil-works/pi>, `packages/ai`
- **License**: MIT, see `LICENSE` in this directory. Copyright (c) 2025 Mario Zechner.
- **Not modified**: the copies keep their original bytes. `types.ts` is ours: it re-exports the
  host's types so the vendored `.d.ts` files resolve their original `"../types.ts"` specifier.

## Why copy instead of importing

The subpaths *are* published (`exports: { "./api/*": …, "./utils/*": … }`), but the Pi extension
loader only maps `@earendil-works/pi-ai` (→ `dist/compat.js`), `/compat`, `/models`, `/oauth` and
`/providers/all`. An unmapped subpath would:

- resolve to a **duplicate copy** of `pi-ai` in the extension's own `node_modules` (Pi warns about
  this in `resource-loader.js`: "Installed copies can bypass the extension loader and create
  duplicate runtime modules"), and
- fail outright in the compiled-binary distribution, where modules are virtual (`virtualModules`,
  `tryNative: false`) and no file exists on disk to import.

Both files have **zero runtime imports**, so a copy is a stable, self-contained unit.

## Re-syncing after a Pi upgrade

```sh
npm run vendor:check   # fails when the host files drifted from these copies
```

When it fails, diff the two versions, decide whether the change matters for this adapter, then
copy the new bytes over and re-run the suite. Do not edit the vendored files in place: the whole
point is that `vendor:check` can compare them.

## Exit ramp

If `transformMessages` and `retryProviderRequest` ever get exported from the package root, delete
this directory and import them from `@earendil-works/pi-ai`.
