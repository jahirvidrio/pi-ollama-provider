/** Pi thinking levels mapped onto the level names one model reports. */

import type { ThinkingLevel, ThinkingLevelMap } from '@earendil-works/pi-ai';
import type { OllamaShowResponse } from '../wire.ts';

/**
 * Pi thinking levels mapped onto the level names the model itself accepts. Named values
 * are matched case-insensitively; a level the model does not offer is `null` so Pi never
 * sends an unsupported named level. A model that only reports booleans offers one on
 * level, exposed as `medium` and mapped to the string `"true"` so the request sends the
 * boolean the model accepts.
 */
export function thinkingLevelMapOf (thinking: OllamaShowResponse['thinking']): ThinkingLevelMap | undefined {
  const values = thinking?.values;
  if (!values || values.length === 0) return undefined;
  const named = values.filter((value): value is string => typeof value === 'string');
  if (named.length === 0) {
    if (!values.includes(true)) return undefined;
    return {
      off: 'off',
      minimal: null,
      low: null,
      medium: 'true',
      high: null,
      xhigh: null,
      max: null,
    };
  }
  const byLevel = new Map(named.map((value) => [value.toLowerCase(), value]));
  const mapped = (level: ThinkingLevel): string | null => byLevel.get(level) ?? null;
  return {
    off: 'off',
    minimal: mapped('minimal'),
    low: mapped('low'),
    medium: mapped('medium'),
    high: mapped('high'),
    xhigh: mapped('xhigh'),
    max: mapped('max'),
  };
}
