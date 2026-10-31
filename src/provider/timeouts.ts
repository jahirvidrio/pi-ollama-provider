/** The provider's own non-chat deadlines, and why each of them exists. */

/**
 * Deadline for one catalog request, not for the refresh as a whole: `/api/tags` gets this budget
 * and every `/api/show` gets it again, so a refresh of N models whose requests all stall can spend
 * up to `CATALOG_TIMEOUT_MS` × (1 + ⌈N/4⌉), because the fan-out runs four at a time. A server that
 * accepts the connection and never answers is a hang, not an exception: without a deadline the
 * stored-catalog fallback in `loadCatalog` would never fire and the provider would stay without
 * models. A stalled request costs its model its metadata, a stalled `/api/tags` costs this refresh
 * its network path, and the next refresh retries by itself: the catalog is best-effort and never
 * retried inside one refresh.
 */
export const CATALOG_TIMEOUT_MS = 30_000;

/** Budget for the cloud login probe: a server that cannot verify inside it is not usable now. */
export const LOGIN_TIMEOUT_MS = 15_000;

/**
 * Deadlines the provider gives its own non-chat requests. The defaults fit a local server or a
 * public cloud endpoint; a host may retune them, and a test shortens them to reach the fallback
 * path without waiting the whole budget.
 */
export interface OllamaProviderTimeouts {
  /**
   * Deadline for one catalog request — `/api/tags`, and again for each `/api/show`; defaults to
   * {@link CATALOG_TIMEOUT_MS}. It is not a budget for the refresh as a whole: see the constant.
   */
  catalogMs?: number;
  /** The cloud login probe (`/api/tags`); defaults to {@link LOGIN_TIMEOUT_MS}. */
  loginMs?: number;
}
