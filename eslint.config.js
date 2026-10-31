import { defineConfig, globalIgnores } from 'eslint/config';
import { neostandard } from 'neostandard';

export default defineConfig([
  globalIgnores(['vendor/**']),
  neostandard({
    ts: true,
    semi: true,
    files: ['extensions/**/*.ts', 'src/**/*.ts', 'tests/**/*.ts'],
  }),
]);
