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

  // Read once at component init, not per keypress/scroll.
  const prefersReducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const scrollBehavior: ScrollBehavior = prefersReducedMotion ? "auto" : "smooth";

  $effect(() => {
    focusIndex = currentIndex;
  });

  $effect(() => {
    // Scroll the current frame into view whenever it changes (keyboard
    // navigation via ,/. at the App level, or a filmstrip click).
    void currentIndex;
    const el = listEl?.querySelector(`[data-index="${currentIndex}"]`);
    el?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: scrollBehavior });
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
          <span class="badge defect-chip" title="Defect count">{frame.defect_count}</span>
        {/if}
        {#if frame.approved}
          <span class="badge approved-badge" title="Approved" aria-hidden="true">&#10003;</span>
        {/if}
        {#if frame.exported}
          <span class="badge exported-badge" title="Exported" aria-hidden="true">out</span>
        {/if}
        {#if jobStates[frame.index]?.state === "queued"}
          <span class="badge job-marker job-queued" title={`${jobStates[frame.index].kind} queued`} aria-hidden="true">&#9675;</span>
        {:else if jobStates[frame.index]?.state === "running"}
          <span class="badge job-marker job-running" title={`${jobStates[frame.index].kind} running`} aria-hidden="true">&#9679;</span>
        {/if}
      </div>
      <span class="name">{frame.file_name}</span>
    </div>
  {/each}
</div>

<style>
  .filmstrip {
    display: flex;
    gap: var(--space-2);
    overflow-x: auto;
    padding: var(--space-2);
    background: var(--bg-1);
    border-top: 1px solid var(--border);
  }
  .frame {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-1);
    width: 140px;
    flex: 0 0 auto;
    cursor: pointer;
    border-radius: var(--radius-1);
    padding: var(--space-1);
  }
  .frame:hover .thumb-wrap {
    background: var(--bg-3);
  }
  .frame.current .thumb-wrap {
    border-color: var(--accent);
  }
  .frame:focus-visible {
    outline: 3px solid var(--focus);
    outline-offset: 1px;
  }
  .thumb-wrap {
    position: relative;
    width: 100%;
    max-width: 140px;
    height: 96px;
    background: var(--bg-0);
    border: 2px solid transparent;
    border-radius: var(--radius-1);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  .thumb {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .spinner {
    position: absolute;
    bottom: 4px;
    right: 4px;
    width: 10px;
    height: 10px;
    border: 2px solid var(--text-2);
    border-top-color: var(--text-1);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  .approved-badge {
    position: absolute;
    top: 2px;
    left: 2px;
    background: rgba(0, 0, 0, 0.75);
    color: var(--ok);
    line-height: 1;
  }
  .exported-badge {
    position: absolute;
    top: 2px;
    right: 2px;
    background: rgba(0, 0, 0, 0.75);
    color: var(--info);
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
  }
  .defect-chip {
    position: absolute;
    bottom: 2px;
    right: 2px;
    background: rgba(39, 39, 39, 0.85); /* --bg-2 @ 85% */
    color: var(--text-1);
  }
  .job-marker {
    position: absolute;
    bottom: 2px;
    left: 2px;
    line-height: 1;
    background: transparent;
    padding: 0;
    text-shadow: 0 0 2px rgba(0, 0, 0, 0.9);
  }
  .job-queued {
    color: var(--accent);
  }
  .job-running {
    color: var(--accent);
    animation: pulse 1.2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0.4;
    }
  }
  .name {
    font-size: var(--text-xs);
    color: var(--text-2);
    max-width: 140px;
    width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    text-align: center;
  }
</style>
