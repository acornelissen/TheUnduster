/** Pure heal-state decisions extracted from App.svelte. App keeps the glue
 * (resolving the frame key, reading the stroke store); the comparison logic
 * lives here so its edges have direct test coverage. */

/** The threshold and stroke count captured at the moment a heal was
 * produced -- what the on-screen healed pixels actually match. */
export interface HealCapture {
  threshold: number;
  strokeCount: number;
}

/** True when the displayed frame's heal no longer matches its inputs: the
 * threshold moved or the stroke count changed since the heal that produced
 * what's on screen was captured. No capture (never healed, or the entry was
 * cleared by a re-heal) is never stale -- there is nothing on screen to be
 * stale against. Display-only: SPACE still toggles the existing
 * before/after; a re-heal overwrites the capture and clears this naturally. */
export function isHealStale(
  captured: HealCapture | undefined,
  currentThreshold: number,
  currentStrokeCount: number,
): boolean {
  if (!captured) return false;
  return currentThreshold !== captured.threshold || currentStrokeCount !== captured.strokeCount;
}
