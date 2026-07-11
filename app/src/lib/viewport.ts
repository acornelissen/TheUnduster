export const TILE = 512;

export interface Level {
  width: number;
  height: number;
}

export interface TileRef {
  level: number;
  tx: number;
  ty: number;
  screenX: number;
  screenY: number;
  screenW: number;
  screenH: number;
}

/** zoom = screen px per level-0 image px. Level i covers 2^i image px per own px. */
export function pickLevel(levels: Level[], zoom: number): number {
  const ideal = Math.floor(-Math.log2(Math.max(zoom, 1e-6)));
  return Math.min(Math.max(ideal, 0), levels.length - 1);
}

export function fitZoom(level0: Level, canvasW: number, canvasH: number): number {
  return Math.min(canvasW / level0.width, canvasH / level0.height);
}

const WHEEL_ZOOM_CLAMP = 1.35;
// A detented mouse wheel notch reports deltaY = +/-120; size k so that
// exactly hits the clamp, i.e. exp(120 * k) == WHEEL_ZOOM_CLAMP.
const WHEEL_ZOOM_K = Math.log(WHEEL_ZOOM_CLAMP) / 120;
const WHEEL_ZOOM_K_CTRL = WHEEL_ZOOM_K * 5;

/**
 * Multiplicative zoom factor for one wheel event. Trackpads emit streams
 * of small-delta events, so a fixed per-event factor (the old 1.15) races
 * away; scaling with delta magnitude keeps trackpads gentle while a
 * detented mouse wheel still moves a full clamped step per notch. Pinch
 * arrives as wheel-with-ctrlKey in WKWebView and needs a stronger
 * response to feel 1:1 with the gesture.
 */
export function wheelZoomFactor(deltaY: number, ctrlKey: boolean): number {
  const k = ctrlKey ? WHEEL_ZOOM_K_CTRL : WHEEL_ZOOM_K;
  const factor = Math.exp(-deltaY * k);
  return Math.min(Math.max(factor, 1 / WHEEL_ZOOM_CLAMP), WHEEL_ZOOM_CLAMP);
}

function tilesForLevel(
  levels: Level[],
  level: number,
  zoom: number,
  centerX: number,
  centerY: number,
  canvasW: number,
  canvasH: number,
  ring: number,
): TileRef[] {
  const l = levels[level];
  const scale = 2 ** level; // level px -> level-0 px
  const screenPerLevelPx = zoom * scale;
  // viewport in level coordinates
  const viewW = canvasW / screenPerLevelPx;
  const viewH = canvasH / screenPerLevelPx;
  const cx = centerX / scale;
  const cy = centerY / scale;
  const x0 = cx - viewW / 2;
  const y0 = cy - viewH / 2;
  const gridX = Math.ceil(l.width / TILE);
  const gridY = Math.ceil(l.height / TILE);
  const tx0 = Math.max(Math.floor(x0 / TILE) - ring, 0);
  const ty0 = Math.max(Math.floor(y0 / TILE) - ring, 0);
  const tx1 = Math.min(Math.ceil((x0 + viewW) / TILE) + ring, gridX);
  const ty1 = Math.min(Math.ceil((y0 + viewH) / TILE) + ring, gridY);
  const out: TileRef[] = [];
  for (let ty = ty0; ty < ty1; ty++) {
    for (let tx = tx0; tx < tx1; tx++) {
      out.push({
        level,
        tx,
        ty,
        screenX: (tx * TILE - x0) * screenPerLevelPx,
        screenY: (ty * TILE - y0) * screenPerLevelPx,
        screenW: TILE * screenPerLevelPx,
        screenH: TILE * screenPerLevelPx,
      });
    }
  }
  return out;
}

export interface Ring {
  x: number;
  y: number;
  r: number;
}

/** Projects one native-resolution defect bbox to a screen-space ring, using
 * the same pan/zoom convention as Viewer's zoomAt/onPointerMove (screen
 * origin at canvas center, image point (centerX, centerY) maps there).
 * Unfiltered -- the caller decides whether an offscreen ring is worth
 * drawing. Shared by ringsFor (below) and Viewer's active-ring highlight,
 * which needs this same projection for a single box that ringsFor's own
 * offscreen filter might otherwise have dropped. */
export function ringForBbox(
  [x0, y0, x1, y1]: [number, number, number, number],
  zoom: number,
  centerX: number,
  centerY: number,
  canvasW: number,
  canvasH: number,
  minR: number,
): Ring {
  const bx = (x0 + x1) / 2;
  const by = (y0 + y1) / 2;
  const x = (bx - centerX) * zoom + canvasW / 2;
  const y = (by - centerY) * zoom + canvasH / 2;
  const extent = Math.max(x1 - x0, y1 - y0);
  const r = Math.max((extent / 2) * zoom, minR);
  return { x, y, r };
}

function ringOnscreen(ring: Ring, canvasW: number, canvasH: number): boolean {
  return ring.x + ring.r >= 0 && ring.x - ring.r <= canvasW && ring.y + ring.r >= 0 && ring.y - ring.r <= canvasH;
}

/** Maps native-resolution defect bboxes to screen-space ring markers.
 * Filters out rings whose bounding circle doesn't intersect the canvas. */
export function ringsFor(
  bboxes: [number, number, number, number][],
  zoom: number,
  centerX: number,
  centerY: number,
  canvasW: number,
  canvasH: number,
  minR: number,
): Ring[] {
  const out: Ring[] = [];
  for (const bbox of bboxes) {
    const ring = ringForBbox(bbox, zoom, centerX, centerY, canvasW, canvasH, minR);
    if (ringOnscreen(ring, canvasW, canvasH)) out.push(ring);
  }
  return out;
}

/** Tiles to draw, coarse underlay first, then the sharp level, with a
 * one-tile prefetch ring on the sharp level. */
export function visibleTiles(
  levels: Level[],
  zoom: number,
  centerX: number,
  centerY: number,
  canvasW: number,
  canvasH: number,
): TileRef[] {
  const sharp = pickLevel(levels, zoom);
  const out: TileRef[] = [];
  if (sharp + 1 < levels.length) {
    out.push(
      ...tilesForLevel(levels, sharp + 1, zoom, centerX, centerY, canvasW, canvasH, 0),
    );
  }
  out.push(...tilesForLevel(levels, sharp, zoom, centerX, centerY, canvasW, canvasH, 1));
  return out;
}
