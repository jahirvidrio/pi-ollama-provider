# Contributing

## Requirements

Node >= 22.19. An Ollama server is not needed: the suite runs against a fake one.

```sh
npm ci
```

## Checks

```sh
npm test            # tests/, against a fake Ollama server
npm run typecheck   # tsc --noEmit, strict
npm run lint        # neostandard over everything this repository owns
npm run vendor:check
```

`vendor/pi-ai/` holds byte-identical copies of two `@earendil-works/pi-ai` helpers, and
`vendor:check` compares them against the installed package.

## Repository conventions

- `main` is not committed to directly. Changes arrive as a pull request from a topic branch,
  rebased and fast-forward merged, so the history stays linear.
- Commits follow Conventional Commits.
- `.git-blame-ignore-revs` lists the whole-tree reformat commits that `git blame` skips.
- `README.md` states why the code is shaped the way it is — the retry budget, the context window,
  the endpoint resolution, and the absence of `ollama-js`.
