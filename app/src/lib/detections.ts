/** Pure helpers for the active-detection delete flow (Viewer's z/shift-z
 * cycling + Delete/Backspace). Kept separate from Viewer.svelte so this
 * geometry and index bookkeeping is unit-testable without a canvas. */

import { MAX_RADIUS } from "./brush";

/** A small pad, in image px, added past the bbox's half-extent so the erase
 * circle fully swallows the box edge (and any soft feathering around it)
 * instead of clipping it exactly at the detection boundary. */
const ERASE_PAD = 4;

/** Erase-stroke geometry for deleting one detection: a single center-point
 * stamp (matches the existing Enter-to-stamp shape) whose radius covers the
 * bbox's larger dimension. Clamped to the brush radius cap (brush.ts
 * MAX_RADIUS) so an oversized detection can't request a stroke radius the
 * backend would reject. Consequence of the clamp: a bbox wider than about
 * 2*(MAX_RADIUS - ERASE_PAD) native px gets an erase circle that only
 * partially covers it -- the operator can delete again or hand-erase the
 * remainder. */
export function eraseStrokeForBbox(
  bbox: [number, number, number, number],
): { cx: number; cy: number; radius: number } {
  const [x0, y0, x1, y1] = bbox;
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const extent = Math.max(x1 - x0, y1 - y0);
  const radius = Math.min(Math.ceil(extent / 2) + ERASE_PAD, MAX_RADIUS);
  return { cx, cy, radius };
}

/** Picks the `current` cycling index after the ring at `removedIndex` is
 * deleted from a list that had it removed, given `remainingCount` (the
 * count AFTER removal). Every ring after the removed one shifts down one
 * slot, so the same numeric index now names the next ring -- landing there
 * keeps cycling moving forward instead of jumping back to the start. Wraps
 * to 0 if the removed ring was last, and to -1 once none remain. */
export function nextCurrentAfterRemoval(removedIndex: number, remainingCount: number): number {
  if (remainingCount <= 0) return -1;
  return removedIndex % remainingCount;
}
