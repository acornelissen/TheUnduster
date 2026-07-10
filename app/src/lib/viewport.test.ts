import { describe, expect, it } from "vitest";
import { fitZoom, pickLevel, ringsFor, visibleTiles, wheelZoomFactor, type Level } from "./viewport";

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

describe("ringsFor", () => {
  it("maps a centered bbox to the canvas center", () => {
    // canvas 800x600, viewport centered at image (1000, 600), zoom 1:
    // a bbox centered exactly at (1000, 600) maps to screen (400, 300).
    const rings = ringsFor([[980, 580, 1020, 620]], 1, 1000, 600, 800, 600, 12);
    expect(rings).toHaveLength(1);
    expect(rings[0].x).toBeCloseTo(400);
    expect(rings[0].y).toBeCloseTo(300);
    // bbox extent is 40x40 image px; at zoom 1, radius = max(20, 12) = 20
    expect(rings[0].r).toBeCloseTo(20);
  });

  it("enforces the minimum radius for small defects", () => {
    const rings = ringsFor([[998, 598, 1002, 602]], 1, 1000, 600, 800, 600, 12);
    expect(rings[0].r).toBeCloseTo(12);
  });

  it("scales radius with zoom", () => {
    const rings = ringsFor([[980, 580, 1020, 620]], 0.5, 1000, 600, 800, 600, 12);
    // extent 40 image px * zoom 0.5 / 2 = 10, below minR -> clamped to 12
    expect(rings[0].r).toBeCloseTo(12);
    const rings2 = ringsFor([[900, 500, 1100, 700]], 0.5, 1000, 600, 800, 600, 12);
    // extent 200 image px * zoom 0.5 / 2 = 50
    expect(rings2[0].r).toBeCloseTo(50);
  });

  it("filters bboxes fully offscreen", () => {
    const rings = ringsFor(
      [
        [980, 580, 1020, 620], // centered, onscreen
        [10000, 10000, 10010, 10010], // far offscreen
      ],
      1,
      1000,
      600,
      800,
      600,
      12,
    );
    expect(rings).toHaveLength(1);
  });

  it("keeps a ring whose circle still overlaps the canvas edge", () => {
    // bbox center maps just past the right edge, but its radius reaches back in
    const rings = ringsFor([[1390, 580, 1430, 620]], 1, 1000, 600, 800, 600, 12);
    // screen x = (1410 - 1000) * 1 + 400 = 810, r = max(20, 12) = 20;
    // circle spans 790..830, canvas right edge is 800 -> still overlaps
    expect(rings).toHaveLength(1);
  });
});

describe("wheelZoomFactor", () => {
  it("is 1 for zero delta", () => {
    expect(wheelZoomFactor(0, false)).toBe(1);
  });

  it("zooms in on negative delta, out on positive, proportionally", () => {
    expect(wheelZoomFactor(-40, false)).toBeGreaterThan(1);
    expect(wheelZoomFactor(40, false)).toBeLessThan(1);
    // small trackpad delta moves less than a large one
    expect(wheelZoomFactor(-4, false)).toBeLessThan(wheelZoomFactor(-40, false));
  });

  it("clamps a detented mouse-wheel notch", () => {
    expect(wheelZoomFactor(-120, false)).toBe(1.35);
    expect(wheelZoomFactor(120, false)).toBe(1 / 1.35);
  });

  it("responds more strongly to pinch (ctrlKey) at the same delta", () => {
    expect(wheelZoomFactor(-10, true)).toBeGreaterThan(wheelZoomFactor(-10, false));
  });
});
