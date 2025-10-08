/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from 'vitest';
import {
  getModelPricing,
  calculateTokenCost,
  calculateModelCost,
  MODEL_PRICING,
} from './pricing.js';

describe('pricing', () => {
  describe('getModelPricing', () => {
    it('should return pricing for known models', () => {
      const pricing = getModelPricing('gemini-2.5-pro');
      expect(pricing).toBeDefined();
      expect(pricing?.inputTokenPrice).toBe(1.25);
      expect(pricing?.outputTokenPrice).toBe(5.0);
      expect(pricing?.cachedInputTokenPrice).toBe(0.3125);
    });

    it('should return pricing for versioned models', () => {
      const pricing = getModelPricing('gemini-2.5-pro-001');
      expect(pricing).toBeDefined();
      expect(pricing?.inputTokenPrice).toBe(1.25);
    });

    it('should return pricing for flash models', () => {
      const pricing = getModelPricing('gemini-2.5-flash');
      expect(pricing).toBeDefined();
      expect(pricing?.inputTokenPrice).toBe(0.075);
      expect(pricing?.outputTokenPrice).toBe(0.3);
    });

    it('should return pricing for flash-lite models', () => {
      const pricing = getModelPricing('gemini-2.5-flash-lite');
      expect(pricing).toBeDefined();
      expect(pricing?.inputTokenPrice).toBe(0.015);
      expect(pricing?.outputTokenPrice).toBe(0.06);
    });

    it('should match models by prefix', () => {
      const pricing = getModelPricing('gemini-2.5-flash-thinking-exp');
      expect(pricing).toBeDefined();
      expect(pricing?.inputTokenPrice).toBe(0.075);
    });

    it('should return undefined for unknown models', () => {
      const pricing = getModelPricing('unknown-model');
      expect(pricing).toBeUndefined();
    });
  });

  describe('calculateTokenCost', () => {
    it('should calculate cost correctly', () => {
      const cost = calculateTokenCost(1_000_000, 1.25);
      expect(cost).toBe(1.25);
    });

    it('should calculate cost for partial millions', () => {
      const cost = calculateTokenCost(500_000, 1.25);
      expect(cost).toBe(0.625);
    });

    it('should calculate cost for small token counts', () => {
      const cost = calculateTokenCost(1_000, 1.25);
      expect(cost).toBe(0.00125);
    });

    it('should handle zero tokens', () => {
      const cost = calculateTokenCost(0, 1.25);
      expect(cost).toBe(0);
    });
  });

  describe('calculateModelCost', () => {
    it('should calculate total cost for gemini-2.5-pro', () => {
      const cost = calculateModelCost(
        'gemini-2.5-pro',
        1_000_000,
        1_000_000,
        0,
      );
      expect(cost).toBeDefined();
      expect(cost).toBe(6.25); // 1.25 + 5.0
    });

    it('should calculate cost with cached tokens', () => {
      const cost = calculateModelCost(
        'gemini-2.5-pro',
        1_000_000,
        1_000_000,
        500_000,
      );
      expect(cost).toBeDefined();
      // (1_000_000 - 500_000) * 1.25/1M + 500_000 * 0.3125/1M + 1_000_000 * 5.0/1M
      // = 0.625 + 0.15625 + 5.0 = 5.78125
      expect(cost).toBeCloseTo(5.78125, 5);
    });

    it('should calculate cost for flash model', () => {
      const cost = calculateModelCost(
        'gemini-2.5-flash',
        1_000_000,
        1_000_000,
        0,
      );
      expect(cost).toBeDefined();
      expect(cost).toBe(0.375); // 0.075 + 0.30
    });

    it('should return undefined for unknown model', () => {
      const cost = calculateModelCost('unknown-model', 1_000_000, 1_000_000, 0);
      expect(cost).toBeUndefined();
    });

    it('should handle realistic token counts', () => {
      // Example: 50k input tokens, 10k output tokens, 20k cached
      const cost = calculateModelCost('gemini-2.5-pro', 50_000, 10_000, 20_000);
      expect(cost).toBeDefined();
      // (50_000 - 20_000) * 1.25/1M + 20_000 * 0.3125/1M + 10_000 * 5.0/1M
      // = 0.0375 + 0.00625 + 0.05 = 0.09375
      expect(cost).toBeCloseTo(0.09375, 5);
    });

    it('should handle zero tokens', () => {
      const cost = calculateModelCost('gemini-2.5-pro', 0, 0, 0);
      expect(cost).toBeDefined();
      expect(cost).toBe(0);
    });
  });

  describe('MODEL_PRICING', () => {
    it('should have pricing for all major models', () => {
      expect(MODEL_PRICING['gemini-2.5-pro']).toBeDefined();
      expect(MODEL_PRICING['gemini-2.5-flash']).toBeDefined();
      expect(MODEL_PRICING['gemini-2.5-flash-lite']).toBeDefined();
      expect(MODEL_PRICING['gemini-1.5-pro']).toBeDefined();
      expect(MODEL_PRICING['gemini-1.5-flash']).toBeDefined();
    });

    it('should have consistent pricing structure', () => {
      Object.values(MODEL_PRICING).forEach((pricing) => {
        expect(pricing.inputTokenPrice).toBeGreaterThan(0);
        expect(pricing.outputTokenPrice).toBeGreaterThan(0);
        expect(pricing.outputTokenPrice).toBeGreaterThanOrEqual(
          pricing.inputTokenPrice,
        );
        if (pricing.cachedInputTokenPrice !== undefined) {
          expect(pricing.cachedInputTokenPrice).toBeLessThan(
            pricing.inputTokenPrice,
          );
        }
      });
    });
  });
});
