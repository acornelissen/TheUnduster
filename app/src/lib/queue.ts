/** Pure composition of the queue panel's entry list. App.svelte wires this
 * to its own state (`jobStates`, the `queue_snapshot` invoke result, and the
 * open roll's frames); kept here so the running-first, snapshot-ordered,
 * dedupe-by-key logic has direct test coverage without mounting Svelte. */

/** Progress for the single running job, attached only to its entry (see
 * `composeQueueEntries`). `{ done, total }` for a job reporting numeric
 * progress (heal jobs, and export jobs during their heal stage); `{ stage }`
 * for a job that only reports a named phase (export jobs outside the heal
 * stage). Detect jobs have no progress signal yet, so their running entry
 * never carries this field. */
export type QueueProgress = { done: number; total: number } | { stage: string };

export interface QueueEntry {
  key: string;
  label: string;
  state: "running" | "queued";
  progress?: QueueProgress;
}

export interface RunningJob {
  index: number;
  kind: "detect" | "heal" | "export";
}

export interface SnapshotJob {
  index: number;
  kind: "detect" | "heal" | "export";
}

/** Composes the queue panel's rows: running jobs first (from `jobStates`),
 * then pending jobs in the backend's queue order (from `queue_snapshot`).
 * Both inputs are expected to already be filtered by the caller to the open
 * roll's generation and index bounds -- this function only orders and
 * dedupes. A row's key is `${kind}:${index}`; any snapshot row whose key
 * matches a running row is dropped, since a job already running must not
 * also render as queued. */
export function composeQueueEntries(
  running: RunningJob[],
  snapshot: SnapshotJob[],
  frames: { file_name: string }[],
  runningProgress?: QueueProgress | null,
): QueueEntry[] {
  const label = (index: number, kind: string) =>
    `${frames[index]?.file_name ?? `frame ${index}`} — ${kind}`;

  const runningEntries: QueueEntry[] = running.map((j) => ({
    key: `${j.kind}:${j.index}`,
    label: label(j.index, j.kind),
    state: "running",
    ...(runningProgress ? { progress: runningProgress } : {}),
  }));
  const runningKeys = new Set(runningEntries.map((e) => e.key));

  const queuedEntries: QueueEntry[] = snapshot
    .filter((j) => !runningKeys.has(`${j.kind}:${j.index}`))
    .map((j) => ({
      key: `${j.kind}:${j.index}`,
      label: label(j.index, j.kind),
      state: "queued",
    }));

  return [...runningEntries, ...queuedEntries];
}
