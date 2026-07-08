import { describe, expect, it } from "vitest";
import { fitZoom, pickLevel, visibleTiles, type Level } from "./viewport";

const LEVELS: Level[] = [
  { width: 2000, height: 1200 },
  { width: 1000, height: 600 },
  { width: 500, height: 300 },
];

describe("pickLevel", () => {
  it("uses level 0 at 100% and above", () => {
    expect(pickLevel(LEVELS, 1)).toBe(0);
    expect(pickLevel(LEVELS, 2.5)).toBe(0);
  });
  it("steps down as zoom halves", () => {
    expect(pickLevel(LEVELS, 0.5)).toBe(1);
    expect(pickLevel(LEVELS, 0.25)).toBe(2);
  });
  it("clamps to the coarsest level", () => {
    expect(pickLevel(LEVELS, 0.01)).toBe(2);
  });
});

describe("fitZoom", () => {
  it("fits the long edge", () => {
    expect(fitZoom(LEVELS[0], 1000, 1000)).toBeCloseTo(0.5);
    expect(fitZoom(LEVELS[0], 4000, 300)).toBeCloseTo(0.25);
  });
});

describe("visibleTiles", () => {
  it("covers the viewport at 100% around the center", () => {
    const tiles = visibleTiles(LEVELS, 1, 1000, 600, 800, 600);
    const sharp = tiles.filter((t) => t.level === 0);
    expect(sharp.length).toBeGreaterThan(0);
    // viewport spans image px 600..1400 x 300..900 -> tiles 1..2 x 0..1
    const keys = new Set(sharp.map((t) => `${t.tx},${t.ty}`));
    for (const k of ["1,0", "2,0", "1,1", "2,1"]) {
      expect(keys.has(k)).toBe(true);
    }
  });

  it("orders coarse level before sharp level", () => {
    const tiles = visibleTiles(LEVELS, 1, 1000, 600, 800, 600);
    const levels = tiles.map((t) => t.level);
    const firstSharp = levels.indexOf(0);
    const lastCoarse = levels.lastIndexOf(1);
    expect(lastCoarse).toBeLessThan(firstSharp === -1 ? Infinity : firstSharp);
  });

  it("never emits tiles outside the grid", () => {
    const tiles = visibleTiles(LEVELS, 0.1, 250, 150, 4000, 4000);
    for (const t of tiles) {
      const l = LEVELS[t.level];
      expect(t.tx).toBeGreaterThanOrEqual(0);
      expect(t.ty).toBeGreaterThanOrEqual(0);
      expect(t.tx).toBeLessThan(Math.ceil(l.width / 512));
      expect(t.ty).toBeLessThan(Math.ceil(l.height / 512));
    }
  });

  it("screen rects scale with zoom", () => {
    const [first] = visibleTiles(LEVELS, 1, 1000, 600, 800, 600).filter(
      (t) => t.level === 0,
    );
    expect(first.screenW).toBeCloseTo(512);
    const [half] = visibleTiles(LEVELS, 0.5, 1000, 600, 800, 600).filter(
      (t) => t.level === 1,
    );
    // level-1 tile is 512 level-1 px = 1024 level-0 px, at zoom 0.5 -> 512 screen px
    expect(half.screenW).toBeCloseTo(512);
  });
});
