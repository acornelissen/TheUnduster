<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
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

  async function openScan() {
    error = null;
    const path = await open({
      multiple: false,
      filters: [{ name: "Scans", extensions: ["tif", "tiff", "png", "jpg", "jpeg"] }],
    });
    if (typeof path !== "string") return;
    const previousId = info?.id;
    try {
      info = await invoke<ImageInfo>("open_image", { path });
    } catch (e) {
      error = String(e);
      return;
    }
    if (previousId !== undefined) {
      try {
        await invoke("close_image", { id: previousId });
      } catch {
        // best effort cleanup; the replaced image just lingers in the cache
      }
    }
  }
</script>

<div class="shell">
  <header>
    <button onclick={openScan}>Open scan</button>
    {#if error}<p role="alert">{error}</p>{/if}
  </header>
  <section class="stage">
    {#if info}
      {#key info.id}
        <Viewer {info} />
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
