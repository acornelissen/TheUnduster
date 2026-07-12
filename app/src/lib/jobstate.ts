/** Pure derivations over App.svelte's `jobStates` map -- the current-frame
 * job status, queued count, and export-running checks the status bar and
 * activity flags read. App wires these into `$derived`; kept here (like
 * queue.ts's composition) so their edge cases -- prefetch exclusion,
 * running-vs-queued, the per-index lookup -- have direct test coverage
 * without mounting Svelte. */

export type JobKind = "detect" | "heal" | "export" | "prefetch";

/** The shape App holds in `jobStates`: frame index -> its one live job. */
export type JobStates = Record<number, { state: "queued" | "running"; kind: JobKind }>;

/** The kind of the job RUNNING at `index`, or null when nothing is running
 * there (queued-only counts as not running), no roll is open, or the index
 * has no job. Backs `rollDetecting`/`rollHealing`: a frame the operator
 * navigated away from correctly reports null, since the map is keyed by
 * index and `index` is the frame on screen now. */
export function runningKindAt(
  jobStates: JobStates,
  index: number,
  hasRoll: boolean,
): JobKind | null {
  if (!hasRoll) return null;
  const job = jobStates[index];
  return job && job.state === "running" ? job.kind : null;
}

/** Count of jobs across all frames and states, EXCLUDING prefetch: routine
 * neighbor warm-ups fire on every navigation, and counting them would keep
 * the status line churning while the operator is just browsing. */
export function countQueuedJobs(jobStates: JobStates): number {
  return Object.values(jobStates).filter((j) => j.kind !== "prefetch").length;
}

/** True only while an export job is actually RUNNING (not merely queued) --
 * so a live heal in a mixed batch narrates itself instead of the activity
 * slot showing a bare "exporting" for an export that hasn't started. */
export function isExportRunning(jobStates: JobStates): boolean {
  return Object.values(jobStates).some((j) => j.kind === "export" && j.state === "running");
}
