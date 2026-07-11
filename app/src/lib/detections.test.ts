import { describe, expect, it } from "vitest";
import { MAX_RADIUS } from "./brush";
import { eraseStrokeForBbox, nextCurrentAfterRemoval } from "./detections";

describe("eraseStrokeForBbox", () => {
  it("centers on the bbox midpoint", () => {
    const s = eraseStrokeForBbox([100, 100, 140, 120]);
    expect(s.cx).toBeCloseTo(120);
    expect(s.cy).toBeCloseTo(110);
  });

  it("radius covers half the larger dimension plus a small pad", () => {
    // w=40, h=20 -> larger dimension is w
    const s = eraseStrokeForBbox([100, 100, 140, 120]);
    expect(s.radius).toBe(Math.ceil(40 / 2) + 4);
  });

  it("rounds up a fractional half-extent", () => {
    // w=41 -> half is 20.5, ceil to 21, plus pad
    const s = eraseStrokeForBbox([100, 100, 141, 110]);
    expect(s.radius).toBe(21 + 4);
  });

  it("clamps to the brush radius cap for a huge bbox", () => {
    const s = eraseStrokeForBbox([0, 0, 2000, 2000]);
    expect(s.radius).toBe(MAX_RADIUS);
  });
});

describe("nextCurrentAfterRemoval", () => {
  it("lands on the ring that shifted into the removed slot", () => {
    // 3 rings, index 1 removed, 2 remain -> the old index-2 ring is now at 1
    expect(nextCurrentAfterRemoval(1, 2)).toBe(1);
  });

  it("wraps to 0 when the last ring in the list was removed", () => {
    expect(nextCurrentAfterRemoval(2, 2)).toBe(0);
  });

  it("returns -1 when nothing remains", () => {
    expect(nextCurrentAfterRemoval(0, 0)).toBe(-1);
  });

  it("stays at 0 removing the only ring's neighbor down to one", () => {
    expect(nextCurrentAfterRemoval(0, 1)).toBe(0);
  });
});
