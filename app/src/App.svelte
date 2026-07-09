<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { open } from "@tauri-apps/plugin-dialog";
  import Viewer from "./lib/Viewer.svelte";
  import Filmstrip from "./lib/Filmstrip.svelte";
  import { nextUnapprovedIndex } from "./lib/roll-nav";
  import type { Level } from "./lib/viewport";
  import { undoStroke, redoStroke, type StrokeData } from "./lib/brush";

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
  }

  let info: ImageInfo | null = $state(null);
  let error: string | null = $state(null);
  let loading: string | null = $state(null);
  let viewer: Viewer | undefined = $state();
  let overlay = $state({ enabled: true, threshold: 0.5 });
  let detected = $state(false);
  let componentsAtHalf: number | null = $state(null);
  let detecting = $state(false);
  let healing = $state(false);
  let healProgress: { done: number; total: number } | null = $state(null);

  $effect(() => {
    const un = listen<{ id: number; done: number; total: number }>("heal-progress", (e) => {
      if (info && e.payload.id === info.id) {
        healProgress = { done: e.payload.done, total: e.payload.total };
      }
    });
    return () => {
      un.then((f) => f());
    };
  });

  // Healing model lifecycle. Starts "loaded" (not "missing") so the header
  // button doesn't flash into existence before the mount-time
  // `inpainter_status` fetch resolves.
  let modelStatus: "loaded" | "available" | "missing" | "downloading" = $state("loaded");
  let modelReceived = $state(0);
  let modelTotal: number | null = $state(null);

  let roll: RollInfo | null = $state(null);
  let currentIndex = $state(0);
  // The index of the frame actually on screen. `currentIndex` is set
  // synchronously on navigation (stepFrame/selectFrame/approveAndAdvance)
  // before `activate_frame` resolves, so during that window the OLD frame is
  // still displayed while `currentIndex` already points at the NEW one.
  // `displayedIndex` only advances once an activation actually lands (both
  // the reuse and decode paths in activateCurrentFrame), so strokes and
  // persistence -- which must bind to what the operator is looking at --
  // key off this instead of `currentIndex`.
  let displayedIndex = $state(0);
  let scanDone = $state(false);
  let exporting = $state(false);
  let thresholdSaveTimer: ReturnType<typeof setTimeout> | undefined;

  // Per-frame brush stroke undo/redo stacks, keyed by roll index
  // (`roll:{index}`) or by the single-image's id (`single:{id}`). Roll
  // frames persist to the sidecar via set_frame_strokes; single-image
  // strokes are session-local and never written anywhere.
  let strokeStore: Record<string, { strokes: StrokeData[]; redo: StrokeData[] }> = $state({});
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
      error = `Frame ${roll.frames[e.payload.index].file_name}: ${e.payload.message}`;
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen("roll-done", () => {
      scanDone = true;
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{ index: number }>("export-progress", (e) => {
      if (!roll) return;
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
    const un = listen<{ index: number; stage: string }>("export-frame-stage", (e) => {
      if (!roll) return;
      exportDetail = `${roll.frames[e.payload.index].file_name}: ${e.payload.stage}`;
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{ index: number; done: number; total: number }>(
      "export-heal-progress",
      (e) => {
        if (!roll) return;
        exportDetail = `${roll.frames[e.payload.index].file_name}: healing ${e.payload.done}/${e.payload.total}`;
      },
    );
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{ index: number; message: string }>("export-frame-error", (e) => {
      if (!roll) return;
      error = `Frame ${roll.frames[e.payload.index].file_name}: ${e.payload.message}`;
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen("export-done", () => {
      exporting = false;
      exportDetail = null;
    });
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
        modelStatus = await invoke<"loaded" | "available" | "missing">("inpainter_status");
      } catch (e) {
        error = String(e);
        modelStatus = "missing";
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
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{ message: string }>("model-error", (e) => {
      error = e.payload.message;
      modelReceived = 0;
      modelTotal = null;
      (async () => {
        try {
          const s = await invoke<"loaded" | "available" | "missing">("inpainter_status");
          // A retry click may have started a new download while this refetch
          // was in flight; never clobber the live downloading state.
          if (modelStatus !== "downloading") modelStatus = s;
        } catch (e2) {
          error = String(e2);
        }
      })();
    });
    return () => {
      un.then((f) => f());
    };
  });

  async function openScan() {
    error = null;
    const path = await open({
      multiple: false,
      filters: [{ name: "Scans", extensions: ["tif", "tiff", "png", "jpg", "jpeg"] }],
    });
    if (typeof path !== "string") return;
    const previousId = info?.id;
    const hadRoll = roll !== null;
    roll = null;
    loading = "Opening scan";
    if (hadRoll) {
      try {
        await invoke("close_roll", {});
      } catch {
        // best effort cleanup; any activated roll frames just linger in the
        // registry rather than blocking the scan the operator asked to open
      }
    }
    try {
      info = await invoke<ImageInfo>("open_image", { path });
    } catch (e) {
      error = String(e);
      loading = null;
      return;
    }
    detected = false;
    componentsAtHalf = null;
    if (previousId !== undefined) {
      try {
        await invoke("close_image", { id: previousId });
      } catch {
        // best effort cleanup; the replaced image just lingers in the cache
      }
    }
  }

  async function openRoll() {
    error = null;
    const dir = await open({ multiple: false, directory: true });
    if (typeof dir !== "string") return;
    info = null;
    scanDone = false;
    try {
      roll = await invoke<RollInfo>("open_roll", { dir });
    } catch (e) {
      error = String(e);
      return;
    }
    // Seed the stroke store from the sidecar-backed strokes each frame
    // already carries, so undo history and painted state survive a reopen.
    const seeded: Record<string, { strokes: StrokeData[]; redo: StrokeData[] }> = {};
    roll.frames.forEach((f, i) => {
      seeded[`roll:${i}`] = { strokes: f.strokes, redo: f.redo_strokes };
    });
    strokeStore = seeded;
    currentIndex = 0;
    if (roll.frames.length > 0) {
      await activateCurrentFrame();
    }
    try {
      await invoke("scan_roll");
    } catch (e) {
      error = String(e);
    }
  }

  async function exportApproved() {
    if (!roll || exporting) return;
    error = null;
    const dir = await open({ directory: true });
    if (typeof dir !== "string") return;
    exporting = true;
    try {
      await invoke("export_approved", { destDir: dir });
    } catch (e) {
      error = String(e);
      exporting = false;
    }
  }

  async function activateCurrentFrame() {
    if (!roll) return;
    // Repeat presses of the same index are already filtered out by the
    // `index === currentIndex` guards in `selectFrame`/`stepFrame` below, so
    // `activating` need not gate re-entry itself -- it just tracks in-flight
    // state. Overlapping activations of *different* indices are allowed to
    // fire; the sequence number below makes the latest one win.
    const seq = ++activationSeq;
    const index = currentIndex;
    loading = "Opening frame";
    overlay.threshold = roll.frames[index].threshold;
    activating = true;
    let result: ImageInfo;
    try {
      result = await invoke<ImageInfo>("activate_frame", { index });
    } catch (e) {
      if (seq !== activationSeq) return; // stale: a newer activation is in flight
      activating = false;
      error = String(e);
      loading = null;
      return;
    }
    if (seq !== activationSeq) return; // stale: a newer activation superseded this one
    activating = false;
    info = result;
    // Only advance once the activation actually landed -- see the
    // `displayedIndex` declaration for why this must not track `currentIndex`
    // directly. This covers both the reuse and fresh-decode paths: the
    // backend doesn't distinguish them here, both resolve `result` above.
    displayedIndex = index;
    // Belt and braces: the backend now guarantees a terminal "ready" emit on
    // both the reuse and fresh-decode paths, which already clears `loading`
    // via the app-progress listener. Clear it here too so a successful
    // activation can never be left stuck behind the loader if that
    // guarantee is ever violated.
    loading = null;
    detected = false;
    componentsAtHalf = null;
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

  async function approveAndAdvance() {
    if (!roll) return;
    const frame = roll.frames[currentIndex];
    frame.approved = true;
    try {
      await invoke("approve_frame", { index: currentIndex, approved: true });
    } catch (e) {
      error = String(e);
      return;
    }
    // Wrapping search: an operator may approve out of order, and A should
    // always land on remaining work anywhere in the roll until none is left.
    const next = nextUnapprovedIndex(
      roll.frames.map((f) => f.approved),
      currentIndex,
    );
    if (next !== -1) {
      currentIndex = next;
      await activateCurrentFrame();
    }
  }

  function onThresholdInput() {
    if (!roll) return;
    roll.frames[currentIndex].threshold = overlay.threshold;
    clearTimeout(thresholdSaveTimer);
    thresholdSaveTimer = setTimeout(() => {
      invoke("set_frame_threshold", {
        index: currentIndex,
        threshold: overlay.threshold,
      }).catch((e) => {
        error = String(e);
      });
    }, 300);
  }

  async function requestDetect() {
    if (!info || detecting) return;
    error = null;
    detecting = true;
    try {
      const report = await invoke<{ id: number; components_at_half: number }>("detect", {
        id: info.id,
      });
      detected = true;
      componentsAtHalf = report.components_at_half;
      // The Viewer's `{#key info.id}` only remounts on an image swap, never
      // on a detect (loading no longer gates on "detecting"), so this handle
      // stays stable across the call.
      await viewer?.refreshDetections(overlay.threshold);
    } catch (e) {
      error = String(e);
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
    if (!info || healing || detecting) return;
    // While an activation is in flight, `overlay.threshold` already belongs
    // to the new frame (activateCurrentFrame sets it before awaiting) but
    // `info`/`displayedIndex` still lag behind the old one; healing now
    // would mix the new frame's threshold with the old frame's strokes and
    // image. Blocking until the switch lands is the coherent choice.
    if (activating) return;
    if (!detected && !(roll && roll.frames[currentIndex].defect_count !== null)) {
      error = "Run detection before healing";
      return;
    }
    // Healing needs live probabilities; if only queue results exist, run
    // detect first so the mask reflects the current threshold.
    if (!detected) {
      await requestDetect();
      if (!detected) return; // detect failed; its error is already shown
    }
    error = null;
    healing = true;
    healProgress = null;
    try {
      await invoke("heal_frame", {
        id: info.id,
        threshold: overlay.threshold,
        strokes: currentStrokes(),
      });
      info = { ...info, healed: true };
    } catch (e) {
      error = String(e);
    } finally {
      healing = false;
    }
  }

  async function downloadModel() {
    if (modelStatus === "downloading" || modelStatus === "loaded") return;
    error = null;
    modelStatus = "downloading";
    modelReceived = 0;
    modelTotal = null;
    try {
      await invoke("download_inpaint_model");
    } catch (e) {
      // model-error also fires and re-fetches inpainter_status; this catch
      // just guards against a rejected invoke that never reaches the backend
      // event path at all.
      error = String(e);
      try {
        const s = await invoke<"loaded" | "available" | "missing">("inpainter_status");
        if (modelStatus !== "downloading") modelStatus = s;
      } catch (e2) {
        error = String(e2);
      }
    }
  }

  function modelProgressText(): string {
    if (modelTotal !== null && modelTotal > 0) {
      return `${Math.floor((modelReceived / modelTotal) * 100)}%`;
    }
    return `${(modelReceived / (1024 * 1024)).toFixed(1)} MB`;
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

  function onStrokesChange(strokes: StrokeData[], redo: StrokeData[]) {
    const key = strokeKey();
    if (!key) return;
    strokeStore[key] = { strokes, redo };
    if (roll) {
      // `displayedIndex`, matching `strokeKey()` above: persist to the
      // sidecar entry for the frame actually on screen, not whichever frame
      // navigation may already be mid-switch towards.
      invoke("set_frame_strokes", { index: displayedIndex, strokes, redoStrokes: redo }).catch(
        (e) => {
          error = String(e);
        },
      );
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
      void approveAndAdvance();
    }
  }
</script>

<svelte:window onkeydown={onWindowKey} />

<div class="shell">
  <header>
    <button onclick={openScan} disabled={loading !== null}>Open scan</button>
    <button onclick={openRoll} disabled={loading !== null}>Open roll</button>
    {#if modelStatus !== "loaded"}
      <button onclick={downloadModel} disabled={modelStatus === "downloading"}>
        {#if modelStatus === "missing"}
          Download healing model (207 MB)
        {:else if modelStatus === "available"}
          Repair healing model
        {:else if modelStatus === "downloading"}
          Downloading...
        {/if}
      </button>
    {/if}
    {#if roll}
      <button
        onclick={approveAndAdvance}
        disabled={roll.frames[currentIndex].approved}
      >
        {roll.frames[currentIndex].approved ? "Approved" : "Approve"}
      </button>
      <button
        onclick={exportApproved}
        disabled={exporting || roll.frames.every((f) => !f.approved)}
      >
        {exporting ? "Exporting..." : "Export approved"}
      </button>
    {/if}
    {#if info}
      <button onclick={requestDetect} disabled={loading !== null || detecting}>
        {detecting ? "Detecting..." : "Detect"}
      </button>
      <button onclick={requestHeal} disabled={loading !== null || detecting || healing || !info}>
        {healing ? "Healing..." : "Heal"}
      </button>
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
      </label>
      <p class="status" role="status">
        {#if detected && componentsAtHalf !== null}
          {componentsAtHalf} defect{componentsAtHalf === 1 ? "" : "s"} at 50%
        {:else if roll && roll.frames[currentIndex].defect_count !== null}
          {roll.frames[currentIndex].defect_count} defect{roll.frames[currentIndex]
            .defect_count === 1
            ? ""
            : "s"} at 50% (scanned)
        {:else}
          Not yet detected
        {/if}
        {#if detecting}
          &mdash; Detecting...
        {/if}
        {#if healing}
          &mdash; Healing...{#if healProgress}
            ({healProgress.done}/{healProgress.total} defects){/if}
        {/if}
        {#if info?.healed}
          &mdash; space toggles before/after
        {/if}
        {#if viewer?.brushStatus()}
          &mdash; {viewer.brushStatus()}
        {/if}
        {#if roll}
          &mdash; {roll.frames.filter((f) => f.approved).length}/{roll.frames.length} approved
          {#if !scanDone}
            &mdash; scanning ({roll.frames.filter((f) => f.defect_count !== null).length}/{roll
              .frames.length})
          {/if}
          {#if exporting}
            &mdash; exporting ({roll.frames.filter((f) => f.exported).length}/{roll.frames.filter(
              (f) => f.approved,
            ).length}){#if exportDetail}
              &mdash; {exportDetail}{/if}
          {/if}
        {/if}
      </p>
    {/if}
    {#if modelStatus === "downloading"}
      <p class="status" role="status">downloading healing model {modelProgressText()}</p>
    {:else if modelStatus === "missing" || modelStatus === "available"}
      <p class="status" role="status">healing: classical fill only</p>
    {/if}
    {#if error}<p role="alert">{error}</p>{/if}
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
        bboxes={roll ? roll.frames[currentIndex].bboxes : null}
        strokes={currentStrokes()}
        redoStrokes={currentRedoStrokes()}
        {onStrokesChange}
        onBrushLimit={(message) => (error = message)}
      />
    {:else if !showLoader}
      <p class="hint">Open a scan or a roll to begin.</p>
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
  </section>
  {#if roll}
    <Filmstrip frames={roll.frames} {currentIndex} {thumbVersions} onSelect={selectFrame} />
  {/if}
</div>

<style>
  :global(body) {
    margin: 0;
    background: #262626;
    color: #e8e8e8;
    font-family: system-ui, sans-serif;
  }
  .shell {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  header {
    padding: 0.5rem;
    display: flex;
    gap: 0.75rem;
    align-items: center;
  }
  button {
    font: inherit;
    padding: 0.4rem 0.9rem;
  }
  button:focus-visible {
    outline: 3px solid #6ab0ff;
  }
  label {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.9rem;
  }
  input[type="range"]:focus-visible {
    outline: 3px solid #6ab0ff;
  }
  .status {
    margin: 0;
    color: #bbb;
    font-size: 0.9rem;
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
    gap: 0.5rem;
    background: rgba(38, 38, 38, 0.75);
    pointer-events: none;
  }
  .stage-overlay img {
    max-height: 70%;
    max-width: 80%;
    filter: blur(2px) brightness(0.8);
    border-radius: 4px;
  }
  .stage-overlay .hint {
    margin: 0;
  }

  .hint {
    text-align: center;
    color: #999;
    margin-top: 4rem;
  }
  [role="alert"] {
    color: #ff9c9c;
    margin: 0;
  }
</style>
