import { describe, expect, it } from "vitest";
import { isHealStale } from "./heal";

describe("isHealStale", () => {
  const captured = { threshold: 0.5, strokeCount: 2 };

  it("is not stale when threshold and stroke count both match", () => {
    expect(isHealStale(captured, 0.5, 2)).toBe(false);
  });

  it("is stale when the threshold has moved", () => {
    expect(isHealStale(captured, 0.6, 2)).toBe(true);
  });

  it("is stale when the stroke count has changed", () => {
    expect(isHealStale(captured, 0.5, 3)).toBe(true);
    expect(isHealStale(captured, 0.5, 1)).toBe(true);
  });

  it("is never stale without a capture (never healed or just re-healed)", () => {
    expect(isHealStale(undefined, 0.9, 99)).toBe(false);
  });
});
