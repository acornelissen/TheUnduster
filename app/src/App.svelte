<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { getCurrentWebview } from "@tauri-apps/api/webview";
  import { open, save } from "@tauri-apps/plugin-dialog";
  import Viewer from "./lib/Viewer.svelte";
  import Icon from "./lib/Icon.svelte";
  import Filmstrip from "./lib/Filmstrip.svelte";
  import StatusBar from "./lib/StatusBar.svelte";
  import Toasts from "./lib/Toasts.svelte";
  import LogPanel from "./lib/LogPanel.svelte";
  import QueuePanel from "./lib/QueuePanel.svelte";
  import ShortcutsPanel from "./lib/ShortcutsPanel.svelte";
  import { composeQueueEntries, type QueueEntry, type QueueProgress } from "./lib/queue";
  import { routeDrop, type PathKind } from "./lib/drop";
  import type { Level } from "./lib/viewport";
  import { undoStroke, redoStroke, type StrokeData } from "./lib/brush";
  import {
    composeActivity,
    composeLeft,
    composeRight,
    formatModelProgress,
  } from "./lib/status";
  import { pushToast, dismissToast, pushLog, type Toast } from "./lib/toasts";

  // Monotonic id source for toasts; module-scoped so ids stay unique across
  // the whole component instance regardless of dismiss/collapse churn.
  let nextToastId = 0;

  // Monotonic id source for activity log entries, mirroring nextToastId --
  // lets LogPanel key its {#each} on a stable id instead of the computed
  // reversed-array index, so entries keep their identity across pushes.
  let nextLogId = 0;

  interface ImageInfo {
    id: number;
    width: number;
    height: number;
    levels: Level[];
    healed: boolean;
  }

  interface FrameInfo {
    index: number;
    file_name: string;
    threshold: number;
    approved: boolean;
    exported: boolean;
    defect_count: number | null;
    bboxes: [number, number, number, number][] | null;
    strokes: StrokeData[];
    redo_strokes: StrokeData[];
  }

  interface RollInfo {
    dir: string;
    frames: FrameInfo[];
    generation: number;
  }

  let info: ImageInfo | null = $state(null);
  let toastList: Toast[] = $state([]);
  let activityLog: { id: number; time: string; level: string; message: string }[] = $state([]);
  let logOpen = $state(false);
  let queueOpen = $state(false);
  let shortcutsOpen = $state(false);

  // The last queue_snapshot response, raw (unfiltered). Refreshed when the
  // queue panel opens and on every job event while it stays open; filtered
  // to the open roll's generation and index bounds in `queueEntries` below
  // so a stale fetch can never show another roll's jobs.
  interface QueueJob {
    kind: "detect" | "heal" | "export" | "prefetch";
    index: number;
    generation: number;
  }
  let queueSnapshot: QueueJob[] = $state([]);

  async function refreshQueueSnapshot() {
    try {
      queueSnapshot = await invoke<QueueJob[]>("queue_snapshot");
    } catch (e) {
      pushError(String(e));
    }
  }

  // Every error site funnels through here: pushes an error toast and logs
  // it to the activity log (capped at 100, oldest dropped first). Message
  // copy is passed through verbatim from the call site.
  function pushError(message: string) {
    toastList = pushToast(toastList, "error", message, nextToastId++);
    activityLog = pushLog(
      activityLog,
      { id: nextLogId++, time: new Date().toLocaleTimeString(), level: "error", message },
      100,
    );
  }

  // Info notes (e.g. the single-export summary) funnel through here.
  function pushInfo(message: string) {
    toastList = pushToast(toastList, "info", message, nextToastId++);
    activityLog = pushLog(
      activityLog,
      { id: nextLogId++, time: new Date().toLocaleTimeString(), level: "info", message },
      100,
    );
  }

  function dismissToastById(id: number) {
    toastList = dismissToast(toastList, id);
  }

  let loading: string | null = $state(null);
  let viewer: Viewer | undefined = $state();
  let overlay = $state({ enabled: true, threshold: 0.5 });
  let detected = $state(false);
  // Live defect count at the current slider threshold, fed by the Viewer's
  // `onDetectionsChange` callback once probabilities exist (either via probe
  // or a real detect run). Distinct from the roll's persisted
  // `defect_count`, which stays fixed at whatever threshold last produced a
  // stored scan/detect result -- this one tracks the slider live.
  let liveDefectCount: number | null = $state(null);
  // Single-image mode only: set/cleared directly around the awaited
  // detect/heal invokes below. Roll mode never touches these -- its activity
  // is derived from `jobStates` (see `rollDetecting`/`rollHealing`) so that
  // navigating away mid-job can never leave a stale flag stuck true.
  let detecting = $state(false);
  let healing = $state(false);
  let healProgress: { done: number; total: number } | null = $state(null);
  let detectProgress: { done: number; total: number } | null = $state(null);
  // Roll-mode queue state: index -> { state, kind }, driven entirely by
  // job-queued/job-started/job-done/job-error/queue-idle. Single-image mode
  // never touches this (it invokes detect/heal directly and awaits them).
  // Deliberately NOT the source of a tracked "is the current frame busy"
  // flag -- that is derived below (`currentJob`/`rollDetecting`/
  // `rollHealing`) so it self-corrects when the operator navigates to a
  // different frame mid-job instead of latching on the index that happened
  // to be current when the job started.
  let jobStates: Record<
    number,
    { state: "queued" | "running"; kind: "detect" | "heal" | "export" | "prefetch" }
  > = $state({});
  // Progress for the currently running queue job, attributed by "currently
  // running" since the worker is single-flight (see lib/queue.ts's
  // QueueProgress doc comment). One slot, not keyed by index: only one job
  // is ever running at a time, so there is never more than one progress to
  // show. Reset to null on job-started (a new job's progress starts fresh)
  // and cleared on job-done/job-error/queue-idle.
  let queueProgress: QueueProgress | null = $state(null);
  // The queue-panel key (`${kind}:${index}`) of the RUNNING job whose cancel
  // has been requested but not yet confirmed by a job-cancelled event -- the
  // abort is cooperative, so the row shows "cancelling" in the meantime.
  // Only ever set for a running entry: cancelling a queued entry removes it
  // synchronously. Cleared when any terminal event (cancelled/done/error)
  // arrives for that job, on queue-idle, and on roll swap/close.
  let cancellingKey: string | null = $state(null);
  // The generation the currently-open roll was opened under (from
  // `open_roll`'s response). Primary guard for the job-* listeners below:
  // every job event now carries the generation it was enqueued/run against,
  // so a listener can drop an event belonging to a roll that has since been
  // swapped out, even if a fresh roll's frame happens to reuse the same
  // index (the race the index-only guard could not close). `null` when no
  // roll is open.
  let rollGeneration: number | null = $state(null);

  $effect(() => {
    const un = listen<{ id: number; done: number; total: number }>("heal-progress", (e) => {
      if (info && e.payload.id === info.id) {
        healProgress = { done: e.payload.done, total: e.payload.total };
      }
      // Queue attribution: the worker is single-flight, so this event always
      // belongs to whichever job is currently running -- no id/index guard
      // needed here (unlike the displayed-frame branch above).
      queueProgress = { done: e.payload.done, total: e.payload.total };
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{ id: number; done: number; total: number }>("detect-progress", (e) => {
      if (info && e.payload.id === info.id) {
        detectProgress = { done: e.payload.done, total: e.payload.total };
      }
      // Queue attribution: the worker is single-flight, so this event always
      // belongs to whichever job is currently running -- no id/index guard
      // needed here (unlike the displayed-frame branch above).
      queueProgress = { done: e.payload.done, total: e.payload.total };
    });
    return () => {
      un.then((f) => f());
    };
  });

  // Healing model lifecycle. Starts "loaded" (not "missing") so the header
  // button doesn't flash into existence before the mount-time
  // `inpainter_status` fetch resolves. "fixture" means a dev-only mean-fill
  // stub is loaded in place of real LaMa (debug builds only) -- healing
  // still runs, but produces flat grey fills on brushed areas instead of an
  // actual inpaint, so the toolbar and status bar must make that
  // unmissable (see `healingStubbed` below and the Model toolbar group).
  type ModelStatus = "loaded" | "available" | "missing" | "downloading" | "fixture";
  let modelStatus: ModelStatus = $state("loaded");
  let modelReceived = $state(0);
  let modelTotal: number | null = $state(null);
  // True between clicking the download's Cancel button and the terminal
  // model-cancelled/-error/-done event: the abort is cooperative (the
  // backend notices between chunks), so the button reads "Cancelling" and
  // goes inert in the meantime.
  let modelCancelRequested = $state(false);

  let roll: RollInfo | null = $state(null);
  let currentIndex = $state(0);
  // Derived, not tracked: the current frame's job (if any) always reflects
  // `jobStates[currentIndex]` live, so navigating to a different frame while
  // a job is in flight immediately (and correctly) reports "not busy" here
  // without any listener needing to know the operator moved on.
  const currentJob = $derived(roll ? jobStates[currentIndex] : undefined);
  const rollDetecting = $derived(currentJob?.state === "running" && currentJob.kind === "detect");
  const rollHealing = $derived(currentJob?.state === "running" && currentJob.kind === "heal");
  // Combined single-image + roll activity, for template/guard use so callers
  // don't need to know which mode is active.
  const isDetecting = $derived(detecting || rollDetecting);
  const isHealing = $derived(healing || rollHealing);
  // Count of queued jobs (all states, all frames) for the status line.
  // Prefetch is excluded: routine neighbor warm-ups fire on every roll
  // navigation, and counting them keeps the status line churning while the
  // operator is just browsing. The filmstrip dots and the queue panel still
  // show prefetch jobs.
  const queuedJobCount = $derived(
    Object.values(jobStates).filter((j) => j.kind !== "prefetch").length,
  );
  // True only while an export job is actually running (not merely queued).
  // Feeds the status-activity slot so a live heal narrates itself during a
  // mixed batch instead of the slot showing bare "exporting" for an export
  // that hasn't started yet. Deliberately NOT used to disable the Export
  // button: re-clicking while exports are queued is allowed -- the backend
  // coalesces per-frame, so already-queued frames are skipped and only
  // newly approved work is added.
  const exportRunning = $derived(
    Object.values(jobStates).some((j) => j.kind === "export" && j.state === "running"),
  );
  // The index of the frame actually on screen. `currentIndex` is set
  // synchronously on navigation (stepFrame/selectFrame)
  // before `activate_frame` resolves, so during that window the OLD frame is
  // still displayed while `currentIndex` already points at the NEW one.
  // `displayedIndex` only advances once an activation actually lands (both
  // the reuse and decode paths in activateCurrentFrame), so strokes and
  // persistence -- which must bind to what the operator is looking at --
  // key off this instead of `currentIndex`.
  let displayedIndex = $state(0);
  let scanDone = $state(false);
  let exportingSingle = $state(false);
  let scanFileName: string | null = $state(null);
  let scanFileExt: string | null = $state(null);
  let thresholdSaveTimer: ReturnType<typeof setTimeout> | undefined;

  // Per-frame brush stroke undo/redo stacks, keyed by roll index
  // (`roll:{index}`) or by the single-image's id (`single:{id}`). Roll
  // frames persist to the sidecar via set_frame_strokes; single-image
  // strokes are session-local and never written anywhere.
  let strokeStore: Record<string, { strokes: StrokeData[]; redo: StrokeData[] }> = $state({});
  // Inputs a heal was produced from, keyed the same way as strokeStore
  // (`strokeKey()`). Captured at every point a heal lands for a frame --
  // single-image `requestHeal` success, a roll-mode heal job-done for the
  // current frame, and activation of a frame that arrives already healed --
  // and compared against the frame's current inputs to flag a stale heal
  // (see `healStale` below). Stroke count is a deliberate approximation of
  // "strokes changed": a moved stroke with the same count escapes this
  // check. The cache's provenance hash remains the correctness layer; this
  // is only a display hint. Entries are never removed -- the map is
  // session-scoped and tiny (one entry per frame ever healed), so there is
  // nothing worth the complexity of cleaning up.
  let healInputs: Record<string, { threshold: number; strokeCount: number }> = $state({});
  // Roll-mode only: frame index -> whether the on-disk heal cache matches
  // that frame's CURRENT inputs, i.e. whether reactivating it would replay
  // an existing heal instantly. Distinct from `info.healed` (live registry
  // state, cleared the moment a frame is evicted) -- this is what lets the
  // filmstrip badge and the current-frame status fragment keep showing
  // "healed" for a frame the operator merely navigated away from. Filled by
  // fire-and-forget probes (`probeHealedCached`/`probeHealedCachedAll`);
  // absent entries read as false (never probed, or probed and missed) --
  // see those functions for exactly when each probe fires.
  let healedCached: Record<number, boolean> = $state({});
  // Monotonically increasing activation sequence number. Rapid ,/. presses
  // can fire overlapping `activate_frame` invokes whose resolutions race;
  // each call captures its own `seq` and drops its result if a newer
  // activation has started by the time it resolves, so the UI always ends
  // up showing whichever activation was requested last, not whichever
  // happened to resolve last.
  let activationSeq = 0;
  let activating = false;

  $effect(() => {
    const un = listen<{ id: number; stage: string }>("app-progress", (e) => {
      // "detecting" must NOT gate the loader: the Viewer stays mounted and
      // usable (zoom/pan survive) while a detect runs, surfaced instead via
      // the `detecting` status flag below. Only the open-scan stages gate
      // `loading`, and "ready" always clears it, regardless of the path
      // (success or error) that produced it.
      if (e.payload.stage === "decoding") loading = "Decoding scan";
      else if (e.payload.stage === "building-pyramid") loading = "Building preview";
      else if (e.payload.stage === "ready") loading = null;
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{
      index: number;
      count: number | null;
      bboxes: [number, number, number, number][] | null;
    }>("roll-progress", (e) => {
      if (!roll) return;
      roll.frames[e.payload.index].defect_count = e.payload.count;
      // Rings for a freshly scanned frame appear immediately; without this
      // the viewer only learns bboxes when the roll is reopened.
      if (e.payload.bboxes) {
        roll.frames[e.payload.index].bboxes = e.payload.bboxes;
      }
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{ index: number; message: string }>("roll-frame-error", (e) => {
      if (!roll) return;
      roll.frames[e.payload.index].defect_count = null;
      pushError(`Frame ${roll.frames[e.payload.index].file_name}: ${e.payload.message}`);
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen("roll-done", () => {
      scanDone = true;
      // Queue-idle re-probe: catches any heal caches that came to exist
      // (e.g. from a previous session, or an operator-triggered heal that
      // raced the open-time batch probe) without waiting for the operator
      // to revisit every frame individually. `probeHealedCachedAll` no-ops
      // if the roll has since closed.
      void probeHealedCachedAll();
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{
      index: number;
      kind: "detect" | "heal" | "export" | "prefetch";
      generation: number;
    }>(
      "job-queued",
      (e) => {
        // Generation is the primary guard: a job event belongs to this
        // session's open roll only if it was enqueued/run against the same
        // generation `open_roll` handed back. Without this, a job queued
        // just before a roll swap can land after the swap and be mistaken
        // for a job against the NEW roll's same-index frame.
        if (e.payload.generation !== rollGeneration) return;
        if (queueOpen) void refreshQueueSnapshot();
        jobStates[e.payload.index] = { state: "queued", kind: e.payload.kind };
      },
    );
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{
      index: number;
      kind: "detect" | "heal" | "export" | "prefetch";
      generation: number;
    }>(
      "job-started",
      (e) => {
        if (e.payload.generation !== rollGeneration) return;
        if (queueOpen) void refreshQueueSnapshot();
        // Index-in-jobStates guard stays as belt-and-braces: generation is
        // the primary check above, but this also covers pre-generation
        // edges (e.g. an event racing a listener re-subscribe) where an
        // index was never actually queued this session.
        if (!(e.payload.index in jobStates)) return;
        jobStates[e.payload.index] = { state: "running", kind: e.payload.kind };
        queueProgress = null; // a newly started job's progress starts fresh
        // Heal-inputs capture happens at job START, not completion: the worker
        // reads the frame's persisted threshold/strokes moments before this
        // event fires, so these are the values the heal will actually use --
        // input drift during a minutes-long heal must not be recorded as the
        // heal's provenance. Keyed by the JOB's index (not currentIndex) so
        // Heal-approved batch frames get proper captures too.
        if (e.payload.kind === "heal" && roll) {
          const frame = roll.frames[e.payload.index];
          healInputs[`roll:${e.payload.index}`] = {
            threshold: frame.threshold,
            strokeCount: frame.strokes.length,
          };
        }
      },
    );
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{
      index: number;
      kind: "detect" | "heal" | "export" | "prefetch";
      generation: number;
    }>(
      "job-done",
      (e) => {
        if (e.payload.generation !== rollGeneration) return;
        if (queueOpen) void refreshQueueSnapshot();
        // The job the operator was cancelling finished first: the race is
        // benign, the "cancelling" tag just must not outlive the row.
        if (cancellingKey === `${e.payload.kind}:${e.payload.index}`) cancellingKey = null;
        // Index-in-jobStates guard stays as belt-and-braces: see the
        // job-started listener's comment.
        if (!(e.payload.index in jobStates)) return;
        delete jobStates[e.payload.index];
        queueProgress = null;
        if (e.payload.kind === "heal") {
          // Unconditional on index, unlike the current-frame block below:
          // `healApproved` queues a heal for every approved frame in the
          // roll, and each one's own filmstrip badge must reflect its own
          // completion -- the registry says this frame's cache now matches,
          // full stop, whether or not it's the frame on screen.
          healedCached[e.payload.index] = true;
        }
        // Index-guarded on purpose: only refresh detections / mark healed when
        // the completed job belongs to the frame still on screen. Activity
        // flags themselves are derived (see `rollDetecting`/`rollHealing`), so
        // there is nothing to clear here for a stale/navigated-away index.
        if (e.payload.index === currentIndex) {
          if (e.payload.kind === "detect") {
            detected = true;
            void viewer?.refreshDetections(overlay.threshold);
          } else if (e.payload.kind === "heal" && info) {
            // Export completions land here too now; only a heal makes the
            // registry's healed tiles real, so only a heal may claim healed.
            info = { ...info, healed: true };
          }
        }
      },
    );
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{
      index: number;
      kind: "detect" | "heal" | "export" | "prefetch";
      message: string;
      generation: number;
    }>("job-error", (e) => {
      if (e.payload.generation !== rollGeneration) return;
      if (queueOpen) void refreshQueueSnapshot();
      if (cancellingKey === `${e.payload.kind}:${e.payload.index}`) cancellingKey = null;
      // Index-in-jobStates guard stays as belt-and-braces: see the
      // job-started listener's comment.
      if (!(e.payload.index in jobStates)) return;
      delete jobStates[e.payload.index];
      queueProgress = null;
      // A failed export frame's narration must not linger once the job is
      // gone -- export-progress won't fire to clear it for this frame.
      if (e.payload.kind === "export") exportDetail = null;
      const fileName = roll?.frames[e.payload.index]?.file_name ?? `frame ${e.payload.index}`;
      pushError(`Frame ${fileName}: ${e.payload.message}`);
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    // Mirrors job-error's cleanup, minus the error toast: a cancellation is
    // the operator's own action, so it gets a quiet activity-log line
    // instead. Fires both for queued jobs removed synchronously by
    // cancel_job/cancel_all_jobs and for the running job once its
    // cooperative abort actually lands.
    const un = listen<{
      index: number;
      kind: "detect" | "heal" | "export" | "prefetch";
      generation: number;
    }>("job-cancelled", (e) => {
      if (e.payload.generation !== rollGeneration) return;
      if (queueOpen) void refreshQueueSnapshot();
      if (cancellingKey === `${e.payload.kind}:${e.payload.index}`) cancellingKey = null;
      // Index-in-jobStates guard stays as belt-and-braces: see the
      // job-started listener's comment.
      if (!(e.payload.index in jobStates)) return;
      // Only a running job's cancellation clears the shared progress slot: a
      // queued job being removed says nothing about the job that is running.
      if (jobStates[e.payload.index].state === "running") queueProgress = null;
      delete jobStates[e.payload.index];
      if (e.payload.kind === "export") exportDetail = null;
      const fileName = roll?.frames[e.payload.index]?.file_name ?? `frame ${e.payload.index}`;
      activityLog = pushLog(
        activityLog,
        {
          id: nextLogId++,
          time: new Date().toLocaleTimeString(),
          level: "info",
          message: `Cancelled ${e.payload.kind} for ${fileName}`,
        },
        100,
      );
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    // queue-idle means the worker stopped (drained, roll swapped, or errored
    // out) -- NOT that every job succeeded. Treat it purely as a cleanup
    // signal for straggler jobState entries the done/error events missed
    // (e.g. jobs dropped mid-drain by a generation bump on roll close).
    // Generation is the primary guard: ignore idles from stale workers that
    // may have raced a roll swap (see job-queued listener's comment).
    const un = listen<{ generation: number }>("queue-idle", (e) => {
      if (e.payload.generation !== rollGeneration) return;
      if (queueOpen) void refreshQueueSnapshot();
      queueProgress = null; // the worker stopped, so nothing is running now
      cancellingKey = null; // ...so there is nothing left to be cancelling
      // Grace-period cleanup instead of a blanket wipe: a same-generation
      // idle can race an enqueue whose job a fresh worker is about to run
      // (the worker's emit happens after its empty-check releases the lock).
      // Blanket-wiping would drop that entry and the index guard would then
      // swallow its job-started/done events. Snapshot what the idle saw and
      // delete only entries still identical after a grace window: the raced
      // entry transitions to "running" within milliseconds and survives; a
      // genuine straggler (backend events lost, e.g. to a panic) still
      // clears.
      const seen = Object.entries(jobStates).map(
        ([k, v]) => [Number(k), v.state, v.kind] as [number, string, string],
      );
      const generationAtIdle = rollGeneration;
      setTimeout(() => {
        if (rollGeneration !== generationAtIdle) return; // roll changed; reset already ran
        for (const [k, state, kind] of seen) {
          const cur = jobStates[k];
          if (cur && cur.state === state && cur.kind === kind) {
            delete jobStates[k];
          }
        }
      }, 2000);
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{ index: number; generation: number }>("export-progress", (e) => {
      if (!roll) return;
      // Same guard shape as the job-* listeners: generation is the primary
      // check (a swap mid-export must not badge the new roll's same-index
      // frame), and the bounds check is belt-and-braces against a shorter
      // new roll -- without it this indexing would throw a TypeError.
      if (e.payload.generation !== rollGeneration) return;
      if (e.payload.index < 0 || e.payload.index >= roll.frames.length) return;
      roll.frames[e.payload.index].exported = true;
      exportDetail = null; // this frame is done; the next one narrates itself
    });
    return () => {
      un.then((f) => f());
    };
  });

  // Un-healed approved frames re-run detect + heal during export -- minutes
  // per frame with a real inpainting model. These events keep the export
  // counter visibly alive between per-frame completions.
  let exportDetail: string | null = $state(null);

  $effect(() => {
    const un = listen<{ index: number; stage: string; generation: number }>(
      "export-frame-stage",
      (e) => {
        if (!roll) return;
        // Same guard shape as the export-progress listener: a stale event
        // from a roll swapped mid-export must not narrate against (or index
        // past the end of) the new roll's frames.
        if (e.payload.generation !== rollGeneration) return;
        if (e.payload.index < 0 || e.payload.index >= roll.frames.length) return;
        exportDetail = `${roll.frames[e.payload.index].file_name}: ${e.payload.stage}`;
        queueProgress = { stage: e.payload.stage };
      },
    );
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{ index: number; done: number; total: number; generation: number }>(
      "export-heal-progress",
      (e) => {
        if (!roll) return;
        // Same guard shape as export-frame-stage above.
        if (e.payload.generation !== rollGeneration) return;
        if (e.payload.index < 0 || e.payload.index >= roll.frames.length) return;
        exportDetail = `${roll.frames[e.payload.index].file_name}: healing ${e.payload.done}/${e.payload.total}`;
        queueProgress = { done: e.payload.done, total: e.payload.total };
      },
    );
    return () => {
      un.then((f) => f());
    };
  });

  // Bumped when the queue reports a freshly written thumbnail; the filmstrip
  // uses it to cache-bust its img src (same URL otherwise, so the webview
  // would keep showing the earlier 404).
  let thumbVersions: Record<number, number> = $state({});

  // The loader overlay appears only when loading persists past 150ms:
  // reuse-path frame switches resolve in milliseconds and must not flash it.
  let showLoader = $state(false);
  let loaderTimer: ReturnType<typeof setTimeout> | undefined;

  $effect(() => {
    clearTimeout(loaderTimer);
    if (loading !== null) {
      loaderTimer = setTimeout(() => (showLoader = true), 150);
    } else {
      showLoader = false;
    }
    return () => clearTimeout(loaderTimer);
  });

  // True while a drag carrying files/folders hovers the window; drives the
  // drop-highlight overlay only, never blocks the drop itself.
  let dropActive = $state(false);

  $effect(() => {
    const un = getCurrentWebview().onDragDropEvent(async (event) => {
      const { payload } = event;
      if (payload.type === "enter" || payload.type === "over") {
        dropActive = true;
        return;
      }
      if (payload.type === "leave") {
        dropActive = false;
        return;
      }
      // "drop"
      dropActive = false;
      // Same gate as the picker buttons' disabled state: a drop while an
      // open is already in flight would race info/loading reassignment.
      if (loading !== null) return;
      const paths = payload.paths;
      let kinds: PathKind[];
      try {
        kinds = await Promise.all(paths.map((path) => invoke<PathKind>("path_kind", { path })));
      } catch (e) {
        pushError(String(e));
        return;
      }
      const route = routeDrop(paths, kinds);
      if ("error" in route) {
        pushError(route.error);
        return;
      }
      if (route.action === "scan") {
        await openScanPath(route.path);
      } else {
        await openRollPath(route.path);
      }
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{ index: number }>("roll-thumb", (e) => {
      thumbVersions[e.payload.index] = (thumbVersions[e.payload.index] ?? 0) + 1;
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    (async () => {
      try {
        modelStatus = await invoke<"loaded" | "available" | "missing" | "fixture">(
          "inpainter_status",
        );
      } catch (e) {
        pushError(String(e));
        modelStatus = "missing";
        return;
      }
      // Fetch the load-error detail regardless of which status came back,
      // not just for "fixture". The backend records a load failure (a
      // corrupt/incompatible lama.onnx on disk) unconditionally, but the
      // "fixture" status it would otherwise pair with is debug-only -- the
      // autoload that falls back to the dev stub is #[cfg(debug_assertions)].
      // In a release build the same failed load leaves the file on disk and
      // nothing loaded, i.e. status "available", so gating on "fixture"
      // would leave a packaged operator with the exact eprintln-only dead
      // end this whole change set is meant to close. `inpainter_load_error`
      // returns None cheaply in every no-failure case (including the common
      // "loaded"/"missing" ones), so an unconditional fetch only ever toasts
      // when there is genuinely something to say. This effect runs once at
      // mount (it reads no reactive state synchronously), so a single
      // recorded error toasts exactly once -- no "already shown" guard
      // needed.
      try {
        const loadError = await invoke<string | null>("inpainter_load_error");
        if (loadError) pushError(`healing model failed to load: ${loadError}`);
      } catch (e) {
        pushError(String(e));
      }
    })();
  });

  $effect(() => {
    const un = listen<{ received: number; total: number | null }>("model-progress", (e) => {
      modelReceived = e.payload.received;
      modelTotal = e.payload.total;
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen("model-done", () => {
      modelStatus = "loaded";
      modelReceived = 0;
      modelTotal = null;
      // A cancel that raced completion: the model made it, better outcome.
      modelCancelRequested = false;
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    // The operator's own cancel: same state reset as model-error, but a
    // quiet activity-log line instead of an error toast.
    const un = listen("model-cancelled", () => {
      modelReceived = 0;
      modelTotal = null;
      modelCancelRequested = false;
      activityLog = pushLog(
        activityLog,
        {
          id: nextLogId++,
          time: new Date().toLocaleTimeString(),
          level: "info",
          message: "Healing model download cancelled",
        },
        100,
      );
      (async () => {
        try {
          const s = await invoke<"loaded" | "available" | "missing" | "fixture">(
            "inpainter_status",
          );
          // Same guard as model-error's refetch: a retry click may already
          // be downloading again.
          if (modelStatus !== "downloading") modelStatus = s;
        } catch (e2) {
          pushError(String(e2));
        }
      })();
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{ message: string }>("model-error", (e) => {
      pushError(e.payload.message);
      modelReceived = 0;
      modelTotal = null;
      modelCancelRequested = false;
      (async () => {
        try {
          const s = await invoke<"loaded" | "available" | "missing" | "fixture">(
            "inpainter_status",
          );
          // A retry click may have started a new download while this refetch
          // was in flight; never clobber the live downloading state.
          if (modelStatus !== "downloading") modelStatus = s;
        } catch (e2) {
          pushError(String(e2));
        }
      })();
    });
    return () => {
      un.then((f) => f());
    };
  });

  // Native File menu items emit these instead of invoking a command
  // directly, so the picker flow (permission prompt, dialog) stays owned by
  // the webview exactly as it is from the toolbar buttons -- the menu only
  // triggers the same picker functions a click would.
  $effect(() => {
    const un = listen("menu-open-scan", () => {
      void openScan();
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen("menu-open-roll", () => {
      void openRoll();
    });
    return () => {
      un.then((f) => f());
    };
  });

  async function openScan() {
    const path = await open({
      multiple: false,
      filters: [{ name: "Scans", extensions: ["tif", "tiff", "png", "jpg", "jpeg"] }],
    });
    if (typeof path !== "string") return;
    await openScanPath(path);
  }

  /** Every post-pick step for opening a single scan, shared by the file
   * picker (`openScan`) and a dropped file (`onDragDropEvent`'s "drop"
   * handling): the picker only adds the dialog in front of this. */
  async function openScanPath(path: string) {
    const previousId = info?.id;
    // Null-first, like openRollPath: every successful open must be a real
    // Viewer unmount/remount, because the Viewer's mount is what focuses
    // the canvas so keys work immediately. Swapping `info` old->new
    // directly (scan open while a scan is already open) would keep the
    // same Viewer instance alive and skip that focus. Placed after the
    // picker check so a cancelled dialog changes nothing.
    info = null;
    const hadRoll = roll !== null;
    roll = null;
    rollGeneration = null;
    loading = "Opening scan";
    if (hadRoll) {
      try {
        await invoke("close_roll", {});
      } catch {
        // best effort cleanup; any activated roll frames just linger in the
        // registry rather than blocking the scan the operator asked to open
      }
      // The roll this queue state described is gone: without this, a stale
      // entry surviving into single-image mode (or a future roll reusing the
      // same index) could be mistaken for a live job by the guards in the
      // job-started/job-done/job-error listeners below.
      jobStates = {};
      cancellingKey = null;
      // Same reason for the export narration: it described the closed
      // roll's export, and nothing in single-image mode will clear it.
      exportDetail = null;
      // The filmstrip is gone in single-image mode so this is invisible
      // either way, but a fresh roll opened later reuses index space -- stale
      // entries from the closed roll must not paint a healed badge on an
      // unrelated frame of the next one before its own probes land.
      healedCached = {};
    }
    try {
      info = await invoke<ImageInfo>("open_image", { path });
    } catch (e) {
      pushError(String(e));
      loading = null;
      return;
    }
    // Only after a SUCCESSFUL open: a failed open falls back to the empty
    // stage (info was nulled above, matching openRollPath's failure path),
    // and a later single-frame export must keep defaulting to the last
    // successfully opened file's name and format, not the failed pick's.
    scanFileName = path.split(/[\\/]/).pop() ?? null;
    // A name without a dot has no extension: leave null (filters omitted)
    // rather than treating the whole name as one.
    scanFileExt = scanFileName?.includes(".")
      ? (scanFileName.split(".").pop()?.toLowerCase() ?? null)
      : null;
    detected = false;
    liveDefectCount = null;
    // A freshly opened single image is a brand-new registry entry, so this
    // will normally miss (no cached probs to find) -- kept for symmetry with
    // `activateCurrentFrame` and to stay correct if the registry ever learns
    // to reuse ids for reopened files.
    probeDetected(info.id);
    if (previousId !== undefined) {
      try {
        await invoke("close_image", { id: previousId });
      } catch {
        // best effort cleanup; the replaced image just lingers in the cache
      }
    }
  }

  async function openRoll() {
    const dir = await open({ multiple: false, directory: true });
    if (typeof dir !== "string") return;
    await openRollPath(dir);
  }

  /** Every post-pick step for opening a roll, shared by the folder picker
   * (`openRoll`) and a dropped directory (`onDragDropEvent`'s "drop"
   * handling): the picker only adds the dialog in front of this. */
  async function openRollPath(dir: string) {
    info = null;
    scanDone = false;
    try {
      roll = await invoke<RollInfo>("open_roll", { dir });
    } catch (e) {
      pushError(String(e));
      return;
    }
    // Captured from the SAME response as `roll` itself, so the two are
    // always consistent: the job listeners gate on this to drop events from
    // whatever roll was open before this call.
    rollGeneration = roll.generation;
    // Seed the stroke store from the sidecar-backed strokes each frame
    // already carries, so undo history and painted state survive a reopen.
    const seeded: Record<string, { strokes: StrokeData[]; redo: StrokeData[] }> = {};
    roll.frames.forEach((f, i) => {
      seeded[`roll:${i}`] = { strokes: f.strokes, redo: f.redo_strokes };
    });
    strokeStore = seeded;
    // A new roll starts with an empty queue session: any entries left over
    // from the previous roll (or single-image mode) describe jobs against
    // frames that no longer exist in this roll's index space, and must not
    // be mistaken for this roll's own in-flight work by the job listeners.
    jobStates = {};
    cancellingKey = null;
    // Likewise the export narration: a stale line from the previous roll's
    // export would otherwise linger until this roll's first export event.
    exportDetail = null;
    // Same reason: the previous roll's healed badges must not paint over
    // this one's frames before its own probes (below, plus per-frame probes
    // on activation) land.
    healedCached = {};
    currentIndex = 0;
    if (roll.frames.length > 0) {
      await activateCurrentFrame();
    }
    // Frames healed in a PREVIOUS session must show their badge without the
    // operator having to revisit each one (activation is the only other
    // place healedCached gets filled in) -- one batch probe covers the
    // whole roll right away.
    void probeHealedCachedAll();
    try {
      await invoke("scan_roll");
    } catch (e) {
      pushError(String(e));
    }
  }

  async function exportApproved() {
    if (!roll) return;
    const dir = await open({ directory: true });
    if (typeof dir !== "string") return;
    try {
      await invoke("enqueue_exports", { destDir: dir });
    } catch (e) {
      pushError(String(e));
    }
  }

  async function healApproved() {
    if (!roll) return;
    // Back of queue (front: false) for every approved frame, in frame order,
    // so this never jumps ahead of a job the operator already queued
    // in-viewer via d/h.
    for (const frame of roll.frames) {
      if (!frame.approved) continue;
      try {
        await invoke("enqueue_job", { kind: "heal", index: frame.index, front: false });
      } catch (e) {
        pushError(String(e));
      }
    }
  }

  async function exportSingle() {
    if (!info || exportingSingle) return;
    // The guard brackets the WHOLE operation including the save dialog:
    // a second activation while the picker is open must not start a
    // parallel dialog/export pair.
    exportingSingle = true;
    try {
      const saveOptions: {
        defaultPath?: string;
        filters?: { name: string; extensions: string[] }[];
      } = {
        defaultPath: scanFileName ?? undefined,
      };
      // Lock export format to the source format when known
      if (scanFileExt) {
        saveOptions.filters = [{ name: "Same format", extensions: [scanFileExt] }];
      }
      const dest = await save(saveOptions);
      if (!dest) return;
      const result = await invoke<number>("export_frame", { id: info.id, dest });
      pushInfo(`exported ${result} changed pixel${result === 1 ? "" : "s"}`);
    } catch (e) {
      pushError(String(e));
    } finally {
      exportingSingle = false;
    }
  }

  /** Probes whether probabilities already exist for `id` (a roll frame's
   * scan/detect probs restored into the registry at activation, or a
   * single-image reopen) without running a fresh detect. `components`
   * succeeds iff probabilities are cached; failure just means none exist yet
   * -- benign, the stored-bbox fallback stays in charge. On success this
   * hands the boxes straight to the Viewer (`setDetections`, no second
   * invoke) and flips `detected` (re-arming the Viewer's slider-effect and
   * switching markerSource to live detections) so rings and the live count
   * populate immediately.
   *
   * This is the only CCL walk an activation triggers: `setDetections` records
   * `overlay.threshold` as the Viewer's `lastFetchedThreshold`, so the
   * `detected` flip below -- which reruns the Viewer's slider $effect --
   * finds the threshold unchanged and skips its debounced refetch instead of
   * re-invoking `components` with the identical id+threshold.
   *
   * The registry's own probs restore is a fire-and-forget background task on
   * the Rust side, so a probe taken right at activation can race it and miss.
   * One retry after ~1s covers that; if it still misses, the frame simply has
   * no cache and stays exactly as before this change. Both attempts are
   * stale-guarded against `id` (captured before each await) so a fast
   * frame-to-frame flip can never apply a late probe's result to the wrong
   * frame. */
  function probeDetected(id: number) {
    async function attempt(): Promise<boolean> {
      // Captured ONCE, before the await: these boxes are the answer for THIS
      // threshold. Reading overlay.threshold again after the await would,
      // if the operator moved the slider mid-probe, record T1's boxes as
      // current for T2 -- and the slider effect's lastFetchedThreshold
      // equality check would then suppress the corrective refetch forever.
      // Recording the true fetch threshold keeps that check honest: a
      // mid-await slider move leaves it different from the live threshold,
      // so the effect refetches as it should.
      const threshold = overlay.threshold;
      try {
        const components = await invoke<[number, number, number, number][]>("components", {
          id,
          threshold,
        });
        if (info?.id !== id) return true; // stale: a newer frame is active, but not a miss
        // Fill the Viewer's live detections BEFORE flipping `detected`:
        // markerSource switches sources on the flip, and the frame-switch
        // reset left `detections` empty, so flipping first would paint a
        // rings-less frame if anything redraws mid-refetch. (setDetections
        // also feeds liveDefectCount via onDetectionsChange.) Synchronous --
        // no invoke, so no stale-frame check needed between this and the
        // flip below.
        viewer?.setDetections(components, threshold);
        detected = true;
        liveDefectCount = components.length;
        return true;
      } catch {
        return false;
      }
    }

    void (async () => {
      const ok = await attempt();
      if (ok || info?.id !== id) return;
      setTimeout(() => {
        if (info?.id !== id) return; // stale: frame changed during the wait
        void attempt();
      }, 1000);
    })();
  }

  /** Probes whether roll frame `index`'s on-disk heal cache matches its
   * CURRENT persisted inputs (threshold, strokes, detector/inpainter
   * hashes, source stamp), populating `healedCached[index]`. Fire-and-
   * forget, generation-guarded like every other roll-scoped async result in
   * this file (mirrors the job-event listeners' `generation !==
   * rollGeneration` check): a roll swap while the invoke is in flight must
   * never paint an old roll's answer into the new roll's map. A failed
   * probe (e.g. a transient stat error) is silent and simply leaves
   * whatever `healedCached[index]` already held -- benign, mirroring
   * `probeDetected`'s own silent-miss contract. */
  function probeHealedCached(index: number) {
    if (!roll || rollGeneration === null) return;
    const gen = rollGeneration;
    invoke<boolean>("heal_cached", { index })
      .then((cached) => {
        if (rollGeneration !== gen) return; // stale: roll swapped mid-probe
        healedCached[index] = cached;
      })
      .catch(() => {
        // benign: see doc comment above
      });
  }

  /** Batch counterpart to `probeHealedCached`, run once when a roll opens
   * (see `openRollPath`) and again once the initial scan queue goes idle
   * (see the `roll-done` listener): covers every frame in one round trip so
   * frames healed in a PREVIOUS session show their badge without the
   * operator having to revisit each one first. Same generation-guard
   * discipline as `probeHealedCached`; a failed batch just leaves whatever
   * per-frame probes have already populated.
   *
   * Wholesale-replaces the map rather than merging: a heal job-done event
   * (or a per-frame probe) that lands for some OTHER frame in the narrow
   * window between this snapshot being taken on the backend and this
   * `.then` running would get overwritten back to this snapshot's answer.
   * Accepted, same shape as the backend's own documented inpainter-hash
   * race: the badge self-corrects at the next probe (that frame's own
   * job-done, activation, stroke commit, or threshold save all re-probe or
   * set it directly), and merging to avoid it would need per-index
   * timestamps for a flicker that in practice never lasts past the next
   * operator action. */
  async function probeHealedCachedAll() {
    if (!roll || rollGeneration === null) return;
    const gen = rollGeneration;
    try {
      const results = await invoke<boolean[]>("healed_cached_all");
      if (rollGeneration !== gen) return; // stale: roll swapped mid-probe
      const next: Record<number, boolean> = {};
      results.forEach((cached, i) => {
        if (cached) next[i] = true;
      });
      healedCached = next;
    } catch {
      // benign: see doc comment above
    }
  }

  async function activateCurrentFrame() {
    if (!roll || rollGeneration === null) return;
    // Repeat presses of the same index are already filtered out by the
    // `index === currentIndex` guards in `selectFrame`/`stepFrame` below, so
    // `activating` need not gate re-entry itself -- it just tracks in-flight
    // state. Overlapping activations of *different* indices are allowed to
    // fire; the sequence number below makes the latest one win.
    const seq = ++activationSeq;
    const index = currentIndex;
    loading = "Opening frame";
    overlay.threshold = roll.frames[index].threshold;
    // Snapshot for the restore-case heal-inputs capture below: the slider
    // stays live during the activation await and mutates the same frame
    // object in place, so reading it post-await could record a live edit
    // instead of the persisted values the cached heal actually matched.
    const persistedThreshold = roll.frames[index].threshold;
    const persistedStrokeCount = roll.frames[index].strokes.length;
    // Captured now, schedule-time, same shape as the threshold-save debounce:
    // the backend re-checks this against the live roll under the same lock
    // that registers the decoded image, so a decode that finishes after a
    // roll swap loses instead of registering old-roll pixels into the new
    // roll's same-index frame.
    const generation = rollGeneration;
    activating = true;
    let result: ImageInfo | null;
    try {
      result = await invoke<ImageInfo | null>("activate_frame", { index, generation });
    } catch (e) {
      if (seq !== activationSeq) return; // stale: a newer activation is in flight
      activating = false;
      pushError(String(e));
      loading = null;
      return;
    }
    if (seq !== activationSeq) return; // stale: a newer activation superseded this one
    activating = false;
    if (result === null) {
      // Benign swap race, not a real failure (returned as data, not an
      // error, so nothing here string-matches messages): the backend closed
      // its own freshly-decoded image and declined to register it. Nothing
      // to show the operator -- the frame they're actually looking at
      // (post-swap) is unaffected, and a subsequent activation of the new
      // roll's frame proceeds normally.
      loading = null;
      return;
    }
    info = result;
    // Only advance once the activation actually landed -- see the
    // `displayedIndex` declaration for why this must not track `currentIndex`
    // directly. This covers both the reuse and fresh-decode paths: the
    // backend doesn't distinguish them here, both resolve `result` above.
    displayedIndex = index;
    // Restore case: this frame arrived already healed (cache reuse), and no
    // capture point has recorded its inputs yet. Its current persisted
    // values ARE the provenance inputs that matched the cache -- capture
    // them so the staleness check has something to compare against. A real
    // capture point (heal job-done) always overwrites this if one exists.
    if (result.healed && !(`roll:${index}` in healInputs)) {
      healInputs[`roll:${index}`] = {
        threshold: persistedThreshold,
        strokeCount: persistedStrokeCount,
      };
    }
    // Belt and braces: the backend now guarantees a terminal "ready" emit on
    // both the reuse and fresh-decode paths, which already clears `loading`
    // via the app-progress listener. Clear it here too so a successful
    // activation can never be left stuck behind the loader if that
    // guarantee is ever violated.
    loading = null;
    detected = false;
    liveDefectCount = null;
    // The registry may already hold probabilities for this frame -- from the
    // roll scan, or restored from the probs cache by the backend's
    // fire-and-forget restore. Probe for them so rings/z-cycling/the status
    // count switch to live detections without the operator ever pressing
    // Detect; `probeDetected` no-ops (leaves the stored-bbox fallback) when
    // none exist.
    probeDetected(result.id);
    // Same reasoning as the probs probe above, for the heal cache: an
    // evicted-then-reactivated frame's registry `healed` flag was reset by
    // the eviction, but its on-disk heal cache (if any) still matches --
    // this is what lets the badge/status fragment keep showing "healed"
    // across a re-activation, not just within one continuous view.
    probeHealedCached(index);
    prefetchNeighbors(index);
  }

  // Warms the immediate next/previous frames into the registry so stepping
  // through a roll hits a resident entry instead of paying the full decode.
  // Back of queue (front: false) -- neighbors must never jump ahead of a
  // heal/export/detect the operator actually asked for. Fire-and-forget:
  // rapid stepping is handled by the backend's coalescing (a duplicate
  // enqueue for the same index just returns false) and the job worker's
  // resident fast-path, so no debounce is needed here.
  function prefetchNeighbors(index: number) {
    if (!roll) return; // roll mode only
    for (const neighbor of [index + 1, index - 1]) {
      if (neighbor < 0 || neighbor >= roll.frames.length) continue;
      invoke("enqueue_job", { kind: "prefetch", index: neighbor, front: false }).catch((e) =>
        pushError(String(e)),
      );
    }
  }

  async function selectFrame(index: number) {
    if (!roll || index === currentIndex) return;
    currentIndex = index;
    await activateCurrentFrame();
  }

  function stepFrame(delta: number) {
    if (!roll) return;
    const next = Math.min(Math.max(currentIndex + delta, 0), roll.frames.length - 1);
    if (next === currentIndex) return;
    currentIndex = next;
    void activateCurrentFrame();
  }

  // Approves in place. This used to auto-advance to the next unapproved
  // frame; the operator asked for it back out -- moving on is , and .'s job.
  async function approveCurrent() {
    if (!roll) return;
    const frame = roll.frames[currentIndex];
    frame.approved = true;
    try {
      await invoke("approve_frame", { index: currentIndex, approved: true });
    } catch (e) {
      pushError(String(e));
    }
  }

  // Inverse of approveCurrent: un-marks the current frame.
  async function unapproveCurrent() {
    if (!roll) return;
    const frame = roll.frames[currentIndex];
    frame.approved = false;
    try {
      await invoke("unapprove_frame", { index: currentIndex });
    } catch (e) {
      pushError(String(e));
    }
  }

  function onThresholdInput() {
    if (!roll) return;
    roll.frames[currentIndex].threshold = overlay.threshold;
    clearTimeout(thresholdSaveTimer);
    // Captured now, not read back out of `currentIndex`/`rollGeneration`
    // after the debounce: the operator can navigate to a different frame, or
    // close/open a roll, while this timer is pending, and the invoke below
    // must save/mirror against the frame it was scheduled for, never
    // whichever frame happens to sit at the same index in a roll opened
    // afterward.
    const index = currentIndex;
    const generation = rollGeneration;
    const threshold = overlay.threshold;
    thresholdSaveTimer = setTimeout(() => {
      thresholdSaveTimer = undefined;
      if (generation === null) return;
      invoke<number | null>("set_frame_threshold", { index, threshold, generation })
        .then((count) => {
          // Mirror-back guard, same shape as the job-event generation checks
          // above: a stale generation means the roll was swapped while this
          // sat debounced, so `roll.frames[index]` -- if it even exists --
          // belongs to a different roll and must not be touched. Checked
          // BEFORE the `count === null` branch below (unlike the old shape):
          // `count === null` is ambiguous on its own -- it means either "the
          // write was discarded" (stale generation) or "the write landed but
          // the frame has no resident probabilities to recompute a count
          // from" -- and only the generation check actually distinguishes
          // them. `generation === rollGeneration` here proves the backend
          // did not see a mismatch either (only this client's own roll swaps
          // ever bump either side), so a null count under a matching
          // generation is provably the second case: the threshold WAS
          // persisted, just with nothing to recompute.
          if (!roll || generation !== rollGeneration) return;
          const frame = roll.frames[index];
          if (frame === undefined) return;
          if (count !== null) frame.defect_count = count;
          // The persisted threshold is exactly what `heal_cached` reads for
          // provenance -- same reasoning as the stroke-commit re-probe in
          // `onStrokesChange`: this is the drift `healStale` already tracks
          // for the live heal on screen, applied to the on-disk cache badge.
          probeHealedCached(index);
        })
        .catch((e) => {
          pushError(String(e));
        });
    }, 300);
  }

  async function requestDetect() {
    // `detected` matches the button's disabled state so the d key agrees
    // with the toolbar: probabilities exist, the slider re-thresholds live.
    if (!info || isDetecting || detected) return;
    // Roll mode: the background queue owns the run. Enqueue and return --
    // `detecting`/`detected` follow the job-started/job-done events instead
    // of this call's resolution. The queue is roll-only (jobs run against a
    // roll frame's persisted sidecar), so single-image mode always takes the
    // direct-invoke path below.
    if (roll) {
      try {
        await invoke("enqueue_job", { kind: "detect", index: currentIndex, front: true });
      } catch (e) {
        pushError(String(e));
      }
      return;
    }
    detecting = true;
    detectProgress = null;
    try {
      // The report's `components_at_half` is a fixed-0.5 count used only for
      // the backend's own bookkeeping; the immediately following
      // `refreshDetections` call fetches the live count at the current
      // slider threshold via `onDetectionsChange`, so it is not read here.
      await invoke<{ id: number; components_at_half: number }>("detect", {
        id: info.id,
      });
      detected = true;
      // The Viewer's `{#key info.id}` only remounts on an image swap, never
      // on a detect (loading no longer gates on "detecting"), so this handle
      // stays stable across the call.
      await viewer?.refreshDetections(overlay.threshold);
    } catch (e) {
      pushError(String(e));
      // Belt and braces: the backend now guarantees a terminal "ready" emit
      // on every detect exit path, which already clears `loading`. This
      // catch clears it too in case that guarantee is ever violated, so a
      // failed detect can never leave the app stuck behind the loader.
      loading = null;
    } finally {
      detecting = false;
    }
  }

  async function requestHeal() {
    if (!info || isHealing || isDetecting) return;
    // While an activation is in flight, `overlay.threshold` already belongs
    // to the new frame (activateCurrentFrame sets it before awaiting) but
    // `info`/`displayedIndex` still lag behind the old one; healing now
    // would mix the new frame's threshold with the old frame's strokes and
    // image. Blocking until the switch lands is the coherent choice.
    if (activating) return;
    if (roll) {
      // Roll mode: a heal job resolves its own probabilities at run time --
      // it falls back to a cached or fresh detect internally when none
      // exist yet, so there is no need to enqueue a separate detect job
      // first (the worker's internal fallback landed in Task 2). Enqueueing
      // only the heal job keeps this simple and avoids a redundant detect
      // dispatch; see task-3-report.md for the deviation from the brief's
      // enqueue-detect-then-heal dance.
      //
      // The threshold slider debounce-persists on a ~300ms timer
      // (onThresholdInput); flush it now so the job heals at the threshold
      // actually on screen instead of a stale persisted value.
      if (thresholdSaveTimer !== undefined) {
        clearTimeout(thresholdSaveTimer);
        thresholdSaveTimer = undefined;
        try {
          await invoke("set_frame_threshold", {
            index: currentIndex,
            threshold: overlay.threshold,
          });
        } catch (e) {
          pushError(String(e));
          return;
        }
      }
      try {
        await invoke("enqueue_job", { kind: "heal", index: currentIndex, front: true });
      } catch (e) {
        pushError(String(e));
      }
      return;
    }
    // Single-image mode only from here on (the roll branch above always
    // returns): healing needs live probabilities computed for this session.
    if (!detected) {
      pushError("Run detection before healing");
      return;
    }
    healing = true;
    healProgress = null;
    // Snapshot BEFORE the await: the slider and brush stay live during a
    // long heal, and the capture must record what was actually sent, not
    // whatever the inputs drifted to by the time the invoke resolved.
    const healThreshold = overlay.threshold;
    const healStrokes = currentStrokes();
    try {
      await invoke("heal_frame", {
        id: info.id,
        threshold: healThreshold,
        strokes: healStrokes,
      });
      const key = strokeKey();
      if (key) {
        healInputs[key] = { threshold: healThreshold, strokeCount: healStrokes.length };
      }
      info = { ...info, healed: true };
    } catch (e) {
      pushError(String(e));
    } finally {
      healing = false;
    }
  }

  async function downloadModel() {
    if (modelStatus === "downloading" || modelStatus === "loaded") return;
    modelStatus = "downloading";
    modelReceived = 0;
    modelTotal = null;
    modelCancelRequested = false;
    try {
      await invoke("download_inpaint_model");
    } catch (e) {
      // model-error also fires and re-fetches inpainter_status; this catch
      // just guards against a rejected invoke that never reaches the backend
      // event path at all.
      pushError(String(e));
      try {
        const s = await invoke<"loaded" | "available" | "missing" | "fixture">(
          "inpainter_status",
        );
        if (modelStatus !== "downloading") modelStatus = s;
      } catch (e2) {
        pushError(String(e2));
      }
    }
  }

  function modelProgressText(): string {
    return formatModelProgress(modelReceived, modelTotal);
  }

  async function cancelModelDownload() {
    modelCancelRequested = true;
    try {
      await invoke("cancel_model_download");
    } catch (e) {
      modelCancelRequested = false;
      pushError(String(e));
    }
  }

  function strokeKey(): string | null {
    // Keyed off `displayedIndex`, not `currentIndex`: during the activation
    // window `currentIndex` already points at the frame being switched to
    // while the old frame is still on screen. A stroke committed, undone, or
    // healed in that window must bind to what the operator is actually
    // looking at.
    if (roll) return `roll:${displayedIndex}`;
    if (info) return `single:${info.id}`;
    return null;
  }

  function currentStrokes(): StrokeData[] {
    const key = strokeKey();
    return key ? (strokeStore[key]?.strokes ?? []) : [];
  }

  function currentRedoStrokes(): StrokeData[] {
    const key = strokeKey();
    return key ? (strokeStore[key]?.redo ?? []) : [];
  }

  // True when the displayed frame's heal no longer matches its inputs:
  // the threshold has moved or the stroke count has changed since the heal
  // that produced what's on screen was captured. Display-only -- SPACE still
  // toggles the existing before/after, and a re-heal (any capture point)
  // overwrites the `healInputs` entry, clearing this naturally.
  const healStale = $derived.by(() => {
    const key = strokeKey();
    if (!key) return false;
    const captured = healInputs[key];
    if (!captured) return false;
    const currentStrokeCount = strokeStore[key]?.strokes.length ?? 0;
    return (
      overlay.threshold !== captured.threshold || currentStrokeCount !== captured.strokeCount
    );
  });

  // Three-zone status bar strings. Pure composition lives in lib/status.ts
  // (tested there); these derived calls just wire in the live state. Each
  // snapshots `roll` into a local const first: TS's narrowing of the
  // `$state`-declared `roll` does not survive into the `.filter` closures
  // below without it.
  const statusLeft = $derived.by(() => {
    const r = roll;
    return composeLeft({
      fileName: r ? r.frames[currentIndex].file_name : scanFileName,
      position: r ? { index: currentIndex, total: r.frames.length } : null,
      // Live count at the current slider threshold once probabilities exist
      // (`detected` flips via a real detect or the activation probe);
      // otherwise fall back to the roll's persisted 0.5-threshold count, or
      // null (not yet detected) outside a roll.
      defectCount:
        detected && liveDefectCount !== null
          ? liveDefectCount
          : r
            ? r.frames[currentIndex].defect_count
            : null,
      threshold: overlay.threshold,
      healed: info?.healed ?? false,
      // Roll-only (single-image mode has no heal-cache probe -- `info.healed`
      // already covers it there). Absent entry reads as false, same as
      // `healedCached`'s own "never probed" default.
      healedCached: r ? (healedCached[currentIndex] ?? false) : false,
      healStale,
      // Live call, not a snapshot: brushStatus() reads the Viewer's $state
      // (brushMode/brushRadius), so this derived re-evaluates whenever the
      // brush is toggled or resized -- the same mechanism the old
      // `{#if viewer?.brushStatus()}` markup relied on. `viewer` itself is
      // $state too, so binding it after mount also retriggers this.
      brushStatus: viewer?.brushStatus() ?? null,
    });
  });
  const statusActivity = $derived.by(() => {
    const r = roll;
    return composeActivity({
      modelStatus,
      modelProgressText: modelProgressText(),
      exporting: exportRunning,
      exportDetail,
      isHealing,
      healProgress,
      isDetecting,
      detectProgress,
      roll: r !== null,
      scanDone,
      scannedCount: r ? r.frames.filter((f) => f.defect_count !== null).length : 0,
      totalCount: r ? r.frames.length : 0,
    });
  });
  // Which engine a heal would run with right now, from the same modelStatus
  // the Model toolbar group renders. "missing"/"available"/"downloading" all
  // mean nothing is loaded, i.e. classical fill only. (Dev-build nuance:
  // during a download started FROM the fixture state this briefly says
  // "classical only" while the fixture would still heal -- same transient
  // imprecision the old stub hint had, and healing mid-download is rare.)
  const healingEngine = $derived.by((): "lama" | "placeholder" | "classical" => {
    if (modelStatus === "loaded") return "lama";
    if (modelStatus === "fixture") return "placeholder";
    return "classical";
  });
  // Shared by the status bar and the Heal button titles below -- one
  // definition of "is healing currently the placeholder" so both surfaces
  // can never disagree.
  const healingStubbed = $derived.by(() => healingEngine === "placeholder");
  const statusRight = $derived.by(() => {
    const r = roll;
    return composeRight({
      roll: r !== null,
      approvedCount: r ? r.frames.filter((f) => f.approved).length : 0,
      totalCount: r ? r.frames.length : 0,
      queuedJobCount,
      healingEngine,
    });
  });

  // Queue panel rows: running jobs from jobStates (live event-driven truth
  // for "started"), queued jobs from the last queue_snapshot (the backend
  // queue is the only order source for pending work). Pure composition and
  // dedupe live in lib/queue.ts (tested there); this derived only filters
  // the raw snapshot to the open roll's generation and index bounds first.
  const queueEntries = $derived.by(() => {
    const r = roll;
    if (!r) return [];
    const running = Object.entries(jobStates)
      .filter(([, v]) => v.state === "running")
      .map(([k, v]) => ({ index: Number(k), kind: v.kind }));
    const pending = queueSnapshot.filter(
      (j) => j.generation === rollGeneration && j.index >= 0 && j.index < r.frames.length,
    );
    return composeQueueEntries(running, pending, r.frames, queueProgress, cancellingKey);
  });

  // Queue-panel row action: a queued entry is removed synchronously by
  // cancel_job (its job-cancelled event follows at once); a running entry
  // only gets a cooperative abort request, so it is marked "cancelling"
  // until the worker's own job-cancelled (or a racing done/error) arrives.
  async function cancelQueueEntry(entry: QueueEntry) {
    if (entry.state === "running") cancellingKey = entry.key;
    try {
      await invoke("cancel_job", { kind: entry.kind, index: entry.index });
    } catch (e) {
      if (cancellingKey === entry.key) cancellingKey = null;
      pushError(String(e));
    }
  }

  async function cancelAllQueueJobs() {
    const running = queueEntries.find((e) => e.state === "running");
    if (running) cancellingKey = running.key;
    try {
      await invoke("cancel_all_jobs", {});
    } catch (e) {
      cancellingKey = null;
      pushError(String(e));
    }
  }

  function onStrokesChange(strokes: StrokeData[], redo: StrokeData[]) {
    const key = strokeKey();
    if (!key) return;
    strokeStore[key] = { strokes, redo };
    if (roll) {
      // Mirror into the frame like onThresholdInput does for threshold:
      // the heal-inputs captures read roll.frames[i].strokes as persisted
      // truth, and without this mirror they would read the open-time seed
      // forever (false "heal is stale" that no re-heal can clear).
      roll.frames[displayedIndex].strokes = strokes;
      // `displayedIndex`, matching `strokeKey()` above: persist to the
      // sidecar entry for the frame actually on screen, not whichever frame
      // navigation may already be mid-switch towards.
      const index = displayedIndex;
      invoke("set_frame_strokes", { index, strokes, redoStrokes: redo })
        .then(() => {
          // Re-probe AFTER the persist lands: `heal_cached` reads the
          // frame's SIDECAR-persisted strokes, not the live edit, so probing
          // any earlier would answer against the stroke list this commit is
          // replacing. A stroke edit is exactly the provenance drift
          // `healStale` already tracks for the live heal on screen; this is
          // the same drift, applied to the on-disk cache's badge instead of
          // recomputing a second comparison in JS.
          probeHealedCached(index);
        })
        .catch((e) => {
          pushError(String(e));
        });
    }
  }

  // One panel at a time: opening any of the three closes the other two, so
  // the fixed right-side panels and the centered shortcuts modal can never
  // stack. Opening the queue panel also fetches a fresh snapshot -- the
  // event-driven refreshes above only run while the panel is already open.
  function toggleQueue() {
    queueOpen = !queueOpen;
    if (queueOpen) {
      logOpen = false;
      shortcutsOpen = false;
      void refreshQueueSnapshot();
    }
  }

  function toggleLog() {
    logOpen = !logOpen;
    if (logOpen) {
      queueOpen = false;
      shortcutsOpen = false;
    }
  }

  function toggleShortcuts() {
    shortcutsOpen = !shortcutsOpen;
    if (shortcutsOpen) {
      logOpen = false;
      queueOpen = false;
    }
  }

  function isTypingTarget(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) return false;
    const tag = target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
  }

  function onWindowKey(e: KeyboardEvent) {
    // Undo/redo must work regardless of which element has focus (the canvas
    // owns its own keydown handler, but the operator may be tabbed to a
    // button or the sensitivity slider when they reach for cmd-z), so this
    // check runs before the typing-target guard below and before the `roll`
    // gate that the roll-navigation keys need.
    if (e.metaKey && (e.key === "z" || e.key === "Z")) {
      if (isTypingTarget(e.target)) return;
      // Undo during a frame switch is ambiguous -- `strokeKey()` already
      // points at the new (not-yet-displayed) frame while the operator is
      // still looking at the old one. Dropping the keypress is predictable;
      // the operator can repeat it once the switch lands.
      if (activating) return;
      const key = strokeKey();
      if (!key) return;
      e.preventDefault();
      const before = strokeStore[key] ?? { strokes: [], redo: [] };
      const result = e.shiftKey
        ? redoStroke(before.strokes, before.redo)
        : undoStroke(before.strokes, before.redo);
      // undoStroke/redoStroke return the same array references when there is
      // nothing to undo/redo; skip the no-op persist/store update in that case.
      if (result.strokes === before.strokes && result.redo === before.redo) return;
      onStrokesChange(result.strokes, result.redo);
      return;
    }
    // `?` toggles the shortcuts overlay -- shifted on most layouts already,
    // so this matches the produced character rather than also requiring
    // e.shiftKey (which would miss layouts where `?` isn't shift-2/-slash).
    // isTypingTarget still guards: typing a literal `?` in a text field must
    // not pop the panel open. defaultPrevented mirrors the Escape branch: a
    // keypress an earlier handler already consumed should do one thing.
    if (e.key === "?" && !isTypingTarget(e.target) && !e.defaultPrevented) {
      e.preventDefault();
      toggleShortcuts();
      return;
    }
    // Escape closes whichever panel is open -- but only when one is, and
    // never when the brush already consumed the keypress (the Viewer's
    // canvas-scoped handler runs before this window handler bubbles and
    // calls preventDefault to turn the brush off; one Escape should do one
    // thing). Shortcuts first (it's a modal sitting above everything else),
    // then queue, then log -- the toggles make all three mutually exclusive,
    // but belt-and-braces keeps the order defined.
    if (e.key === "Escape" && (shortcutsOpen || queueOpen || logOpen) && !e.defaultPrevented) {
      e.preventDefault();
      if (shortcutsOpen) {
        shortcutsOpen = false;
      } else if (queueOpen) {
        queueOpen = false;
      } else {
        logOpen = false;
      }
      return;
    }
    if (!roll) return;
    // Roll navigation keys must not fire while the operator is typing in a
    // form control (e.g. the sensitivity slider has focus via keyboard, or
    // any future text input) -- "," "." and "a" are ordinary characters.
    if (isTypingTarget(e.target)) return;
    // Only handle roll-navigation keys; everything else (arrows, d/m/z/Z)
    // stays owned by the canvas via its own onkeydown so focus there keeps
    // working exactly as in single-image mode.
    if (e.key === ",") {
      e.preventDefault();
      stepFrame(-1);
    } else if (e.key === ".") {
      e.preventDefault();
      stepFrame(1);
    } else if (e.key === "a" || e.key === "A") {
      e.preventDefault();
      // shift-a un-approves in place; plain a keeps its existing
      // approve-and-advance behavior (including on an already-approved
      // frame, where it just advances to the next unapproved one).
      if (e.shiftKey) {
        void unapproveCurrent();
      } else {
        void approveCurrent();
      }
    }
  }
</script>

<!-- blur failsafe: some webview drag cancels (Escape, leaving the window
     abruptly) never send a "leave" event, which would wedge the drop
     highlight on until the next drag. -->
<svelte:window onkeydown={onWindowKey} onblur={() => (dropActive = false)} />

<div class="shell">
  <header class="toolbar">
    <!-- File group: always visible -->
    <div class="toolbar-group">
      <button class="btn" title="Open scan" onclick={openScan} disabled={loading !== null}>
        <Icon name="scan" /> Open scan
      </button>
      <button class="btn" title="Open roll (folder)" onclick={openRoll} disabled={loading !== null}>
        <Icon name="roll" /> Open roll
      </button>
    </div>

    <!-- Frame group: visible when info exists -->
    {#if info}
      <div class="toolbar-group">
        <button
          class="btn"
          title={detected ? "Already detected; the slider re-thresholds live" : "Detect (d)"}
          onclick={requestDetect}
          disabled={loading !== null || isDetecting || detected}
        >
          <Icon name="detect" /> {isDetecting ? "Detecting..." : detected ? "Detected" : "Detect"}
        </button>
        <button
          class="btn btn-primary"
          title={healingStubbed
            ? "Heal (h) — placeholder healing model: download the real one for quality results"
            : "Heal (h)"}
          onclick={requestHeal}
          disabled={loading !== null || isDetecting || isHealing || !info}
        >
          <Icon name="heal" /> {isHealing ? "Healing..." : "Heal"}
        </button>
        {#if !roll}
          <button class="btn" title="Export" onclick={exportSingle} disabled={!info.healed || exportingSingle}>
            <Icon name="export" /> {exportingSingle ? "Exporting..." : "Export"}
          </button>
        {/if}
      </div>
    {/if}

    <!-- Roll group: visible when roll exists -->
    {#if roll}
      <div class="toolbar-group">
        {#if roll.frames[currentIndex].approved}
          <button class="btn" title="Unapprove (shift-a)" onclick={unapproveCurrent}>
            <Icon name="unapprove" /> Unapprove
          </button>
        {:else}
          <button class="btn" title="Approve (a)" onclick={approveCurrent}>
            <Icon name="approve" /> Approve
          </button>
        {/if}
        <button
          class="btn btn-primary"
          title={healingStubbed
            ? "Heal approved — placeholder healing model: download the real one for quality results"
            : "Heal approved"}
          onclick={healApproved}
          disabled={roll.frames.every((f) => !f.approved)}
        >
          <Icon name="heal" /> Heal approved
        </button>
        <button
          class="btn"
          title="Export approved"
          onclick={exportApproved}
          disabled={roll.frames.every((f) => !f.approved)}
        >
          <Icon name="export" /> Export approved
        </button>
      </div>
    {/if}

    <!-- Adjust group: visible when info exists -->
    {#if info}
      <div class="toolbar-group">
        <label>
          Sensitivity
          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.01"
            bind:value={overlay.threshold}
            oninput={onThresholdInput}
          />
          <span class="threshold-value">{overlay.threshold.toFixed(2)}</span>
        </label>
      </div>
    {/if}

    <!-- Model group: visible when modelStatus !== "loaded" -->
    {#if modelStatus !== "loaded"}
      <div class="toolbar-group">
        {#if modelStatus === "downloading"}
          <!-- aria-live so screen readers hear progress without focus; the
               text only changes every 250ms (backend throttle), so "polite"
               stays quiet enough. -->
          <div class="model-download" aria-live="polite">
            {#if modelTotal !== null && modelTotal > 0}
              <div
                class="model-progress-track"
                role="progressbar"
                aria-valuenow={Math.floor((modelReceived / modelTotal) * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label="Healing model download"
              >
                <div
                  class="model-progress-fill"
                  style:width={`${(modelReceived / modelTotal) * 100}%`}
                ></div>
              </div>
            {/if}
            <span class="model-progress-text">{modelProgressText()}</span>
            <button
              class="btn"
              onclick={cancelModelDownload}
              disabled={modelCancelRequested}
              aria-label="Cancel healing model download"
            >
              {modelCancelRequested ? "Cancelling" : "Cancel"}
            </button>
          </div>
        {:else}
          <button class="btn" onclick={downloadModel}>
            <Icon name="download" />
            {#if modelStatus === "missing"}
              Download healing model (207 MB)
            {:else if modelStatus === "available"}
              Repair healing model
            {:else if modelStatus === "fixture"}
              Download real healing model (207 MB)
            {/if}
          </button>
        {/if}
      </div>
    {/if}
  </header>
  <section class="stage">
    {#if info}
      <!-- One persistent Viewer: it reacts to `info` changing instead of
           being remounted, keeping the GL context and tile cache warm so
           switching to an already-decoded frame is instant. -->
      <Viewer
        bind:this={viewer}
        {info}
        {overlay}
        {detected}
        healedAvailable={info.healed ?? false}
        onRequestDetect={requestDetect}
        onRequestHeal={requestHeal}
        bboxes={roll ? roll.frames[displayedIndex].bboxes : null}
        strokes={currentStrokes()}
        redoStrokes={currentRedoStrokes()}
        {onStrokesChange}
        onBrushLimit={(message) => pushError(message)}
        onDetectionsChange={(count) => (liveDefectCount = count)}
      />
    {:else if !showLoader}
      <div class="empty-state">
        <svg class="empty-art" viewBox="0 0 96 64" aria-hidden="true">
          <rect x="2" y="2" width="92" height="60" rx="4" fill="none" stroke="var(--border)" stroke-width="2" />
          <rect x="14" y="12" width="68" height="40" rx="2" fill="var(--surround)" />
          <circle cx="30" cy="26" r="3" fill="var(--detect)" />
          <circle cx="58" cy="38" r="2" fill="var(--detect)" />
          <rect x="4" y="6" width="4" height="4" fill="var(--bg-3)" /><rect x="4" y="14" width="4" height="4" fill="var(--bg-3)" /><rect x="4" y="22" width="4" height="4" fill="var(--bg-3)" /><rect x="4" y="30" width="4" height="4" fill="var(--bg-3)" /><rect x="4" y="38" width="4" height="4" fill="var(--bg-3)" /><rect x="4" y="46" width="4" height="4" fill="var(--bg-3)" /><rect x="4" y="54" width="4" height="4" fill="var(--bg-3)" />
          <rect x="88" y="6" width="4" height="4" fill="var(--bg-3)" /><rect x="88" y="14" width="4" height="4" fill="var(--bg-3)" /><rect x="88" y="22" width="4" height="4" fill="var(--bg-3)" /><rect x="88" y="30" width="4" height="4" fill="var(--bg-3)" /><rect x="88" y="38" width="4" height="4" fill="var(--bg-3)" /><rect x="88" y="46" width="4" height="4" fill="var(--bg-3)" /><rect x="88" y="54" width="4" height="4" fill="var(--bg-3)" />
        </svg>
        <p class="empty-title">no scan open</p>
        <div class="empty-actions">
          <button class="btn" onclick={openScan}><Icon name="scan" /> Open scan</button>
          <button class="btn" onclick={openRoll}><Icon name="roll" /> Open roll</button>
        </div>
        <p class="hint">or drop a scan or a roll folder anywhere in this window</p>
      </div>
    {/if}
    {#if showLoader}
      <!-- Delayed overlay (not a stage swap): quick switches never flash it,
           and during a long decode the previous frame stays visible under a
           dimmed preview of what is coming. -->
      <div class="stage-overlay" role="status" aria-busy="true">
        {#if roll}
          <img
            src={`tiles://localhost/thumb/${currentIndex}?v=${thumbVersions[currentIndex] ?? 0}`}
            alt=""
            onerror={(e) => ((e.currentTarget as HTMLImageElement).style.display = "none")}
          />
        {/if}
        <p class="hint">{loading}...</p>
      </div>
    {/if}
    {#if dropActive}
      <div class="drop-overlay" aria-hidden="true"></div>
    {/if}
  </section>
  {#if roll}
    <Filmstrip
      frames={roll.frames}
      {currentIndex}
      {thumbVersions}
      {jobStates}
      {healedCached}
      onSelect={selectFrame}
    />
  {/if}
  {#if info}
    <StatusBar
      left={statusLeft}
      activity={statusActivity}
      right={statusRight}
      {logOpen}
      onToggleLog={toggleLog}
      {queueOpen}
      onToggleQueue={toggleQueue}
    />
  {/if}
</div>

<Toasts toasts={toastList} onDismiss={dismissToastById} />
{#if logOpen}
  <LogPanel entries={activityLog} id="activity-log-panel" />
{/if}
{#if queueOpen}
  <QueuePanel
    entries={queueEntries}
    id="job-queue-panel"
    onCancel={cancelQueueEntry}
    onCancelAll={cancelAllQueueJobs}
  />
{/if}
{#if shortcutsOpen}
  <ShortcutsPanel onClose={() => (shortcutsOpen = false)} />
{/if}

<style>
  .shell {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: var(--bg-0);
    color: var(--text-1);
  }
  header {
    padding: var(--space-2);
    background: var(--bg-1);
  }
  label {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    font-size: var(--text-sm);
  }
  .stage {
    flex: 1;
    min-height: 0;
    position: relative;
  }
  .stage-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    background: rgba(20, 20, 20, 0.75);
    pointer-events: none;
  }
  .stage-overlay img {
    max-height: 70%;
    max-width: 80%;
    filter: blur(2px) brightness(0.8);
    border-radius: var(--radius-1);
  }
  .stage-overlay .hint {
    margin: 0;
  }

  .hint {
    text-align: center;
    color: var(--text-2);
    margin-top: var(--space-6);
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-3);
    margin-top: var(--space-6);
  }
  .empty-art {
    width: 160px;
    height: auto;
    opacity: 0.8;
  }
  .empty-title {
    color: var(--text-2);
    font-size: var(--text-lg);
    margin: 0;
  }
  .empty-actions {
    display: flex;
    gap: var(--space-2);
  }
  .empty-state .hint {
    margin-top: 0;
  }

  .drop-overlay {
    position: absolute;
    inset: 0;
    border: 2px solid var(--accent);
    background: var(--accent-soft);
    pointer-events: none;
  }

  .model-download {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .model-progress-track {
    width: 120px;
    height: 4px;
    border-radius: var(--radius-1);
    background: var(--bg-3);
    overflow: hidden;
  }
  .model-progress-fill {
    height: 100%;
    background: var(--accent);
    border-radius: var(--radius-1);
  }
  .model-progress-text {
    color: var(--text-2);
    font-size: var(--text-sm);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
</style>
