/** Brush state helpers: pure functions so undo/redo and coordinate math
 * are unit-testable without a canvas. Strokes live in IMAGE pixel
 * coordinates; the backend rasterizes them at heal/export time. */

export interface StrokeData {
  erase: boolean;
  radius: number;
  points: [number, number][];
}

export const MIN_RADIUS = 2;
export const MAX_RADIUS = 256;

// Mirrors MAX_POINTS_PER_STROKE in app/src-tauri/src/masks.rs. Keep in sync:
// validate_strokes rejects the WHOLE stroke list if any stroke exceeds this,
// so the frontend must never send a stroke longer than the backend accepts.
export const MAX_POINTS_PER_STROKE = 4096;
// Mirrors MAX_STROKES in app/src-tauri/src/masks.rs. Keep in sync: same
// whole-list rejection risk as MAX_POINTS_PER_STROKE above.
export const MAX_STROKES = 512;

/** Splits a long point list into consecutive runs of at most `max` points,
 * matching the backend's per-stroke cap. Each chunk after the first repeats
 * the previous chunk's last point first so the rasterized capsule chain has
 * no gap at the seam. */
export function chunkPoints(
  points: [number, number][],
  max: number,
): [number, number][][] {
  if (points.length === 0) return [];
  const chunks: [number, number][][] = [];
  let i = 0;
  while (i < points.length) {
    const start = chunks.length === 0 ? i : i - 1;
    chunks.push(points.slice(start, start + max));
    i = start + max;
  }
  return chunks;
}

export function stepRadius(radius: number, dir: 1 | -1): number {
  const next = dir === 1 ? radius * 1.25 : radius / 1.25;
  return Math.min(Math.max(next, MIN_RADIUS), MAX_RADIUS);
}

export function screenToImage(
  sx: number,
  sy: number,
  zoom: number,
  centerX: number,
  centerY: number,
  canvasW: number,
  canvasH: number,
): [number, number] {
  return [centerX + (sx - canvasW / 2) / zoom, centerY + (sy - canvasH / 2) / zoom];
}

export function pushStroke(
  strokes: StrokeData[],
  redo: StrokeData[],
  s: StrokeData,
): { strokes: StrokeData[]; redo: StrokeData[] } {
  void redo; // a new stroke always invalidates the redo history
  return { strokes: [...strokes, s], redo: [] };
}

export function undoStroke(
  strokes: StrokeData[],
  redo: StrokeData[],
): { strokes: StrokeData[]; redo: StrokeData[] } {
  if (strokes.length === 0) return { strokes, redo };
  return { strokes: strokes.slice(0, -1), redo: [...redo, strokes[strokes.length - 1]] };
}

export function redoStroke(
  strokes: StrokeData[],
  redo: StrokeData[],
): { strokes: StrokeData[]; redo: StrokeData[] } {
  if (redo.length === 0) return { strokes, redo };
  return { strokes: [...strokes, redo[redo.length - 1]], redo: redo.slice(0, -1) };
}
