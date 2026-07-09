<script lang="ts">
  interface FrameInfo {
    index: number;
    file_name: string;
    threshold: number;
    approved: boolean;
    exported: boolean;
    defect_count: number | null;
    bboxes: [number, number, number, number][] | null;
  }

  let {
    frames,
    currentIndex,
    thumbVersions = {},
    jobStates = {},
    onSelect,
  }: {
    frames: FrameInfo[];
    currentIndex: number;
    thumbVersions?: Record<number, number>;
    jobStates?: Record<number, { state: "queued" | "running"; kind: "detect" | "heal" }>;
    onSelect: (index: number) => void;
  } = $props();

  let listEl: HTMLDivElement | undefined = $state();
  let focusIndex = $state(currentIndex);

  $effect(() => {
    focusIndex = currentIndex;
  });

  $effect(() => {
    // Scroll the current frame into view whenever it changes (keyboard
    // navigation via ,/. at the App level, or a filmstrip click).
    void currentIndex;
    const el = listEl?.querySelector(`[data-index="${currentIndex}"]`);
    el?.scrollIntoView({ block: "nearest", inline: "nearest" });
  });

  function moveFocus(delta: number) {
    const next = Math.min(Math.max(focusIndex + delta, 0), frames.length - 1);
    focusIndex = next;
    const el = listEl?.querySelector<HTMLElement>(`[data-index="${next}"]`);
    el?.focus();
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === "ArrowRight" || e.key === "ArrowDown") {
      e.preventDefault();
      moveFocus(1);
    } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
      e.preventDefault();
      moveFocus(-1);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect(focusIndex);
    }
  }
</script>

<div
  bind:this={listEl}
  class="filmstrip"
  role="listbox"
  aria-label="Roll frames"
>
  {#each frames as frame (frame.index)}
    <div
      id={`frame-${frame.index}`}
      data-index={frame.index}
      role="option"
      aria-selected={frame.index === currentIndex}
      tabindex={frame.index === focusIndex ? 0 : -1}
      class="frame"
      class:current={frame.index === currentIndex}
      onclick={() => onSelect(frame.index)}
      onkeydown={onKey}
    >
      <div class="thumb-wrap">
        <img
          src={`tiles://localhost/thumb/${frame.index}?v=${thumbVersions[frame.index] ?? 0}`}
          alt=""
          class="thumb"
          onerror={(e) => ((e.currentTarget as HTMLImageElement).style.visibility = "hidden")}
          onload={(e) => ((e.currentTarget as HTMLImageElement).style.visibility = "visible")}
        />
        {#if frame.defect_count === null}
          <span class="spinner" aria-hidden="true"></span>
        {:else}
          <span class="badge">{frame.defect_count}</span>
        {/if}
        {#if frame.approved}
          <span class="check" aria-hidden="true">&#10003;</span>
        {/if}
        {#if frame.exported}
          <span class="exported" aria-hidden="true">out</span>
        {/if}
        {#if jobStates[frame.index]?.state === "queued"}
          <span class="job-marker job-queued" title={`${jobStates[frame.index].kind} queued`} aria-hidden="true">&#9675;</span>
        {:else if jobStates[frame.index]?.state === "running"}
          <span class="job-marker job-running" title={`${jobStates[frame.index].kind} running`} aria-hidden="true">&#9679;</span>
        {/if}
      </div>
      <span class="name">{frame.file_name}</span>
    </div>
  {/each}
</div>

<style>
  .filmstrip {
    display: flex;
    gap: 0.5rem;
    overflow-x: auto;
    padding: 0.5rem;
    background: #1b1b1b;
    border-top: 1px solid #333;
  }
  .frame {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.2rem;
    width: 96px;
    flex: 0 0 auto;
    cursor: pointer;
    border-radius: 4px;
    padding: 0.25rem;
  }
  .frame.current {
    background: #2d3f57;
  }
  .frame:focus-visible {
    outline: 3px solid #6ab0ff;
    outline-offset: 2px;
  }
  .thumb-wrap {
    position: relative;
    width: 88px;
    height: 88px;
    background: #333;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .thumb {
    max-width: 100%;
    max-height: 100%;
    display: block;
  }
  .badge {
    position: absolute;
    bottom: 2px;
    right: 2px;
    background: rgba(0, 0, 0, 0.75);
    color: #fff;
    font-size: 0.7rem;
    padding: 0.05rem 0.3rem;
    border-radius: 8px;
  }
  .spinner {
    position: absolute;
    bottom: 4px;
    right: 4px;
    width: 10px;
    height: 10px;
    border: 2px solid #888;
    border-top-color: #ddd;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  .check {
    position: absolute;
    top: 2px;
    left: 2px;
    color: #7CFC00;
    font-size: 0.9rem;
  }
  .exported {
    position: absolute;
    top: 2px;
    right: 2px;
    background: rgba(0, 0, 0, 0.75);
    color: #6ab0ff;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 0.05rem 0.25rem;
    border-radius: 3px;
    text-transform: uppercase;
  }
  .job-marker {
    position: absolute;
    bottom: 2px;
    left: 2px;
    font-size: 0.75rem;
    line-height: 1;
    text-shadow: 0 0 2px rgba(0, 0, 0, 0.9);
  }
  .job-queued {
    color: #f0c674;
  }
  .job-running {
    color: #ff9c3c;
  }
  .name {
    font-size: 0.65rem;
    color: #ccc;
    max-width: 88px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
