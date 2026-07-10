/** Pure composition helpers for the three-zone status bar. App.svelte wires
 * these to its own state via `$derived`; kept here (rather than inline in
 * the component) so the composition logic -- activity priority order in
 * particular -- has direct test coverage without mounting Svelte. */

export interface ActivityInput {
  modelStatus: "loaded" | "available" | "missing" | "downloading";
  modelProgressText: string;
  exporting: boolean;
  exportDetail: string | null;
  isHealing: boolean;
  healProgress: { done: number; total: number } | null;
  isDetecting: boolean;
  roll: boolean;
  scanDone: boolean;
  scannedCount: number;
  totalCount: number;
}

/** Single-slot activity string, highest priority first: model download >
 * exporting > current-frame healing > current-frame detecting > roll
 * scanning > null (idle). */
export function composeActivity(input: ActivityInput): string | null {
  if (input.modelStatus === "downloading") {
    return `downloading healing model ${input.modelProgressText}`;
  }
  if (input.exporting) {
    return input.exportDetail ? `exporting — ${input.exportDetail}` : "exporting";
  }
  if (input.isHealing) {
    return input.healProgress
      ? `healing ${input.healProgress.done}/${input.healProgress.total}`
      : "healing";
  }
  if (input.isDetecting) {
    return "detecting";
  }
  if (input.roll && !input.scanDone) {
    return `scanning ${input.scannedCount}/${input.totalCount}`;
  }
  return null;
}

export interface LeftZoneInput {
  fileName: string | null;
  position: { index: number; total: number } | null;
  defectCount: number | null;
  threshold: number;
  healed: boolean;
  healStale: boolean;
  brushStatus: string | null;
}

/** Frame-identity string for the left zone: file name, roll position,
 * defect count at the current threshold (or "not yet detected" while none
 * exists), the healed-state indicator, the stale-heal hint (frame state,
 * not activity -- stale-heal only ever occurs on already-healed frames, so
 * it follows directly after), and the live brush status (frame-interaction
 * state, e.g. "brush 24px"). The single-export note lives as a toast (see
 * lib/toasts.ts), not here. */
export function composeLeft(input: LeftZoneInput): string {
  if (!input.fileName) return "";
  const parts = [input.fileName];
  if (input.position) {
    parts.push(`${input.position.index + 1}/${input.position.total}`);
  }
  if (input.defectCount !== null) {
    const noun = input.defectCount === 1 ? "defect" : "defects";
    parts.push(`${input.defectCount} ${noun} at ${input.threshold.toFixed(2)}`);
  } else {
    parts.push("not yet detected");
  }
  if (input.healed) {
    parts.push("healed (space compares)");
  }
  if (input.healStale) {
    parts.push("heal stale (h re-heals)");
  }
  if (input.brushStatus) {
    parts.push(input.brushStatus);
  }
  return parts.join("  ");
}

export interface RightZoneInput {
  roll: boolean;
  approvedCount: number;
  totalCount: number;
  queuedJobCount: number;
}

/** Counts string for the right zone: roll approval progress and queued job
 * count. */
export function composeRight(input: RightZoneInput): string {
  const parts: string[] = [];
  if (input.roll) {
    parts.push(`${input.approvedCount}/${input.totalCount} approved`);
  }
  if (input.queuedJobCount > 0) {
    const noun = input.queuedJobCount === 1 ? "job" : "jobs";
    parts.push(`${input.queuedJobCount} ${noun} queued`);
  }
  return parts.join("  ");
}
