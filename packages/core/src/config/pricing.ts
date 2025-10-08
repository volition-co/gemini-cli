/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Pricing information for Gemini models.
 * Prices are in USD per 1 million tokens.
 *
 * Based on Gemini API pricing as of January 2025.
 * See: https://ai.google.dev/gemini-api/docs/pricing
 *
 * Note: These are list prices and may not reflect all pricing tiers,
 * discounts, or subscription-based models. For accurate pricing,
 * users should consult the official pricing page.
 */

export interface ModelPricing {
  /** Price per 1 million input tokens (USD) */
  inputTokenPrice: number;
  /** Price per 1 million output tokens (USD) */
  outputTokenPrice: number;
  /** Price per 1 million cached input tokens (USD), if applicable */
  cachedInputTokenPrice?: number;
}

/**
 * Pricing data for Gemini models.
 * Prices are estimates based on public pricing information.
 */
export const MODEL_PRICING: Record<string, ModelPricing> = {
  // Gemini 2.5 Pro pricing (128k context, pay-as-you-go)
  'gemini-2.5-pro': {
    inputTokenPrice: 1.25,
    outputTokenPrice: 10.0,
    cachedInputTokenPrice: 0.31,
  },

  // Gemini 2.5 Flash pricing
  'gemini-2.5-flash': {
    inputTokenPrice: 0.3,
    outputTokenPrice: 2.5,
    cachedInputTokenPrice: 0.075,
  },

  // Gemini 2.5 Flash Lite pricing
  'gemini-2.5-flash-lite': {
    inputTokenPrice: 0.1,
    outputTokenPrice: 0.4,
    cachedInputTokenPrice: 0.025,
  },
};

/**
 * Gets pricing for a specific model.
 * If the exact model is not found, attempts to find pricing for a similar model.
 *
 * @param modelName - The name of the model
 * @returns The pricing information, or undefined if not found
 */
export function getModelPricing(modelName: string): ModelPricing | undefined {
  // Direct match
  if (MODEL_PRICING[modelName]) {
    return MODEL_PRICING[modelName];
  }

  // Try to find a match by removing version suffixes
  const baseModelName = modelName.replace(/-\d{3}$/, '');
  if (MODEL_PRICING[baseModelName]) {
    return MODEL_PRICING[baseModelName];
  }

  // Try to match by prefix (e.g., "gemini-2.5-flash" for "gemini-2.5-flash-thinking")
  for (const [key, pricing] of Object.entries(MODEL_PRICING)) {
    if (modelName.startsWith(key)) {
      return pricing;
    }
  }

  return undefined;
}

/**
 * Calculates the cost for a given number of tokens.
 *
 * @param tokenCount - Number of tokens
 * @param pricePerMillion - Price per 1 million tokens (USD)
 * @returns Cost in USD
 */
export function calculateTokenCost(
  tokenCount: number,
  pricePerMillion: number,
): number {
  return (tokenCount / 1_000_000) * pricePerMillion;
}

/**
 * Calculates the total cost for model usage.
 *
 * @param modelName - The name of the model
 * @param inputTokens - Number of input tokens (excluding cached)
 * @param outputTokens - Number of output tokens
 * @param cachedTokens - Number of cached input tokens
 * @returns Total cost in USD, or undefined if pricing is not available
 */
export function calculateModelCost(
  modelName: string,
  inputTokens: number,
  outputTokens: number,
  cachedTokens: number = 0,
): number | undefined {
  const pricing = getModelPricing(modelName);
  if (!pricing) {
    return undefined;
  }

  // Calculate cost for non-cached input tokens
  const nonCachedInputTokens = Math.max(0, inputTokens - cachedTokens);
  const inputCost = calculateTokenCost(
    nonCachedInputTokens,
    pricing.inputTokenPrice,
  );

  // Calculate cost for cached tokens (if pricing is available)
  const cachedCost = pricing.cachedInputTokenPrice
    ? calculateTokenCost(cachedTokens, pricing.cachedInputTokenPrice)
    : 0;

  // Calculate cost for output tokens
  const outputCost = calculateTokenCost(outputTokens, pricing.outputTokenPrice);

  return inputCost + cachedCost + outputCost;
}
