<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { open } from "@tauri-apps/plugin-dialog";
  import Viewer from "./lib/Viewer.svelte";
  import type { Level } from "./lib/viewport";

  interface ImageInfo {
    id: number;
    width: number;
    height: number;
    levels: Level[];
  }

  let info: ImageInfo | null = $state(null);
  let error: string | null = $state(null);
  let loading: string | null = $state(null);
  let viewer: Viewer | undefined = $state();
  let overlay = $state({ enabled: true, threshold: 0.5 });
  let detected = $state(false);
  let componentsAtHalf: number | null = $state(null);
  let detecting = $state(false);

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

  async function openScan() {
    error = null;
    const path = await open({
      multiple: false,
      filters: [{ name: "Scans", extensions: ["tif", "tiff", "png", "jpg", "jpeg"] }],
    });
    if (typeof path !== "string") return;
    const previousId = info?.id;
    loading = "Opening scan";
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
</script>

<div class="shell">
  <header>
    <button onclick={openScan} disabled={loading !== null}>Open scan</button>
    {#if info}
      <button onclick={requestDetect} disabled={loading !== null || detecting}>
        {detecting ? "Detecting..." : "Detect"}
      </button>
      <label>
        Sensitivity
        <input
          type="range"
          min="0.05"
          max="0.95"
          step="0.01"
          bind:value={overlay.threshold}
        />
      </label>
      <p class="status" role="status">
        {#if detected && componentsAtHalf !== null}
          {componentsAtHalf} defect{componentsAtHalf === 1 ? "" : "s"} at 50%
        {:else}
          Not yet detected
        {/if}
        {#if detecting}
          &mdash; Detecting...
        {/if}
      </p>
    {/if}
    {#if error}<p role="alert">{error}</p>{/if}
  </header>
  <section class="stage">
    {#if loading}
      <p class="hint" role="status" aria-busy="true">{loading}...</p>
    {:else if info}
      {#key info.id}
        <Viewer bind:this={viewer} {info} {overlay} {detected} onRequestDetect={requestDetect} />
      {/key}
    {:else}
      <p class="hint">Open a scan to begin.</p>
    {/if}
  </section>
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
