<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { open } from "@tauri-apps/plugin-dialog";
  import Viewer from "./lib/Viewer.svelte";
  import Filmstrip from "./lib/Filmstrip.svelte";
  import { nextUnapprovedIndex } from "./lib/roll-nav";
  import type { Level } from "./lib/viewport";

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

  let roll: RollInfo | null = $state(null);
  let currentIndex = $state(0);
  let scanDone = $state(false);
  let exporting = $state(false);
  let thresholdSaveTimer: ReturnType<typeof setTimeout> | undefined;
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
    const un = listen<{ index: number; count: number | null }>("roll-progress", (e) => {
      if (!roll) return;
      roll.frames[e.payload.index].defect_count = e.payload.count;
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
    });
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
    try {
      await invoke("heal_frame", { id: info.id, threshold: overlay.threshold });
      info = { ...info, healed: true };
    } catch (e) {
      error = String(e);
    } finally {
      healing = false;
    }
  }

  function isTypingTarget(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) return false;
    const tag = target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
  }

  function onWindowKey(e: KeyboardEvent) {
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
          &mdash; Healing...
        {/if}
        {#if info?.healed}
          &mdash; space toggles before/after
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
            ).length})
          {/if}
        {/if}
      </p>
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
