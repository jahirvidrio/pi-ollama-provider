import { defineConfig, globalIgnores } from 'eslint/config';
import { neostandard } from 'neostandard';

// `vendor/**` holds byte-identical upstream code that `vendor:check` guards instead of a linter.
// `.tmp/**` and `.worktrees/**` are local-only — reference checkouts and scratch worktrees, never
// committed — so the linter stays away from the config files they bring with them.
export default defineConfig([
  globalIgnores(['vendor/**', '.tmp/**', '.worktrees/**']),
  neostandard({
    ts: true,
    semi: true,
    files: ['extensions/**/*.ts', 'src/**/*.ts', 'tests/**/*.ts'],
  }),
]);
