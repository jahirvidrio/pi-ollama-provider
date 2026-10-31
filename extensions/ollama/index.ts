import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { createOllamaProvider } from '../../src/provider/index.ts';

/**
 * Registers the two independent providers: `ollama` for a local server and
 * `ollama-cloud` for Ollama Cloud. Both are dynamic, so Pi discovers their models from
 * the server and persists the catalog between runs.
 */
export default function ollamaExtension (pi: ExtensionAPI): void {
  pi.registerProvider(createOllamaProvider('local'));
  pi.registerProvider(createOllamaProvider('cloud'));
}
