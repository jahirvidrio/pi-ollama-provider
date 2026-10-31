/** Wire types for Ollama's native API (`/api/*`) and the canonical form of its endpoint. */

/** The custom API id this adapter implements, used by the registered providers. */
export const OLLAMA_CHAT_API = 'ollama-chat';

export interface OllamaModelSummary {
  name: string;
  model: string;
  modified_at?: string;
  size?: number;
  digest?: string;
  remote_model?: string;
  remote_host?: string;
  details?: {
    format?: string;
    family?: string;
    families?: string[] | null;
    parameter_size?: string;
    quantization_level?: string;
  };
  capabilities?: string[];
}

export interface OllamaShowResponse {
  capabilities?: string[];
  model_info?: Record<string, unknown>;
  thinking?: { values?: (boolean | string)[]; default?: boolean | string };
  parameters?: string;
  details?: OllamaModelSummary['details'];
}

export interface OllamaToolCall {
  function: { name: string; arguments?: Record<string, unknown> };
}

export interface OllamaChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  images?: string[];
  thinking?: string;
  tool_calls?: OllamaToolCall[];
}

export interface OllamaToolDefinition {
  type: 'function';
  function: { name: string; description?: string; parameters: Record<string, unknown> };
}

export interface OllamaChatRequest {
  model: string;
  messages: OllamaChatMessage[];
  tools?: OllamaToolDefinition[];
  think?: boolean | string;
  keep_alive?: string | number;
  options?: Record<string, unknown>;
}

export interface OllamaChatChunk {
  model?: string;
  created_at?: string;
  message?: {
    role?: string;
    content?: string;
    thinking?: string;
    tool_calls?: OllamaToolCall[];
    images?: string[];
  };
  done?: boolean;
  done_reason?: string;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_cached_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
  error?: string;
}

/**
 * Accepts an origin, an origin with a path, or an `/api` base URL and returns the
 * canonical `/api` endpoint. Pi stores whatever the user typed during `/login`.
 */
export function normalizeOllamaBaseUrl (input: string): string {
  const trimmed = input.trim().replace(/\/+$/, '');
  return trimmed.endsWith('/api') ? trimmed : `${trimmed}/api`;
}
