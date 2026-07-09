import { describe, expect, it } from "vitest";
import { pushStroke, redoStroke, screenToImage, stepRadius, undoStroke } from "./brush";

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
