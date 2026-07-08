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

/** Maps native-resolution defect bboxes to screen-space ring markers, using
 * the same pan/zoom convention as Viewer's zoomAt/onPointerMove (screen
 * origin at canvas center, image point (centerX, centerY) maps there).
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
  for (const [x0, y0, x1, y1] of bboxes) {
    const bx = (x0 + x1) / 2;
    const by = (y0 + y1) / 2;
    const x = (bx - centerX) * zoom + canvasW / 2;
    const y = (by - centerY) * zoom + canvasH / 2;
    const extent = Math.max(x1 - x0, y1 - y0);
    const r = Math.max((extent / 2) * zoom, minR);
    const onscreen =
      x + r >= 0 && x - r <= canvasW && y + r >= 0 && y - r <= canvasH;
    if (onscreen) out.push({ x, y, r });
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
