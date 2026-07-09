import { describe, expect, it } from "vitest";
import { chunkPoints, pushStroke, redoStroke, screenToImage, stepRadius, undoStroke } from "./brush";

const dab = (x: number): { erase: boolean; radius: number; points: [number, number][] } => ({
  erase: false,
  radius: 5,
  points: [[x, 0]],
});

describe("stepRadius", () => {
  it("scales by 1.25 and clamps", () => {
    expect(stepRadius(10, 1)).toBeCloseTo(12.5);
    expect(stepRadius(10, -1)).toBeCloseTo(8);
    expect(stepRadius(2, -1)).toBe(2);
    expect(stepRadius(256, 1)).toBe(256);
  });
});

describe("screenToImage", () => {
  it("maps canvas center to view center", () => {
    expect(screenToImage(400, 300, 0.5, 1000, 800, 800, 600)).toEqual([1000, 800]);
  });
  it("scales offsets by zoom", () => {
    // 100 device px right of center at zoom 0.5 = 200 image px
    expect(screenToImage(500, 300, 0.5, 1000, 800, 800, 600)).toEqual([1200, 800]);
  });
});

describe("stroke stacks", () => {
  it("push clears redo; undo and redo round-trip", () => {
    let s = pushStroke([], [], dab(1));
    s = pushStroke(s.strokes, s.redo, dab(2));
    expect(s.strokes).toHaveLength(2);
    s = undoStroke(s.strokes, s.redo);
    expect(s.strokes).toHaveLength(1);
    expect(s.redo).toHaveLength(1);
    s = redoStroke(s.strokes, s.redo);
    expect(s.strokes).toHaveLength(2);
    expect(s.redo).toHaveLength(0);
    s = undoStroke(s.strokes, s.redo);
    s = pushStroke(s.strokes, s.redo, dab(3));
    expect(s.redo).toHaveLength(0); // new stroke invalidates redo
  });
  it("undo/redo on empty stacks are no-ops", () => {
    expect(undoStroke([], [])).toEqual({ strokes: [], redo: [] });
    expect(redoStroke([], [])).toEqual({ strokes: [], redo: [] });
  });
});

describe("chunkPoints", () => {
  const pts = (n: number): [number, number][] =>
    Array.from({ length: n }, (_, i) => [i, 0]);

  it("empty input yields no chunks", () => {
    expect(chunkPoints([], 4096)).toEqual([]);
  });

  it("exactly max points yields one chunk", () => {
    const chunks = chunkPoints(pts(4096), 4096);
    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toHaveLength(4096);
  });

  it("max + 1 points yields two chunks with an overlap point", () => {
    const points = pts(4097);
    const chunks = chunkPoints(points, 4096);
    expect(chunks).toHaveLength(2);
    expect(chunks[0]).toHaveLength(4096);
    expect(chunks[1]).toHaveLength(2);
    // the second chunk starts by repeating the first chunk's last point
    expect(chunks[1][0]).toEqual(chunks[0][chunks[0].length - 1]);
    expect(chunks[1][1]).toEqual(points[4096]);
  });

  it("chunk sizes never exceed max", () => {
    const chunks = chunkPoints(pts(10000), 4096);
    for (const c of chunks) {
      expect(c.length).toBeLessThanOrEqual(4096);
    }
    // every point is represented (accounting for one-point overlaps)
    const total = chunks.reduce((sum, c) => sum + c.length, 0);
    expect(total).toBe(10000 + (chunks.length - 1));
  });
});
