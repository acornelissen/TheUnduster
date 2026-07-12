<script lang="ts">
  import Icon from "./Icon.svelte";

  type QueueProgress = { done: number; total: number } | { stage: string };

  interface QueueEntry {
    key: string;
    kind: "detect" | "heal" | "export" | "prefetch";
    index: number;
    label: string;
    state: "running" | "queued";
    progress?: QueueProgress;
    cancelling?: true;
  }

  let {
    entries,
    id,
    open = true,
    onCancel,
    onCancelAll,
  }: {
    entries: QueueEntry[];
    id: string;
    open?: boolean;
    onCancel: (entry: QueueEntry) => void;
    onCancelAll: () => void;
  } = $props();

  function hasDoneTotal(p: QueueProgress): p is { done: number; total: number } {
    return "done" in p;
  }
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex -- same rationale as
     LogPanel: a scrollable region with no other focusable descendant needs
     an explicit tabindex for keyboard users to scroll it in WKWebView.
     Rendered from load and toggled with `hidden`, also like LogPanel: the
     status bar's Queue button aria-controls this id, which must exist. -->
<div class="queue-panel" {id} role="region" aria-label="Job queue" tabindex="0" hidden={!open}>
  {#if entries.length === 0}
    <p class="queue-empty">queue is empty</p>
  {:else}
    <div class="queue-actions">
      <button type="button" class="queue-cancel-all" onclick={onCancelAll}>Cancel all</button>
    </div>
    <ul>
      {#each entries as entry (entry.key)}
        <li class="queue-entry">
          <div class="queue-row">
            <span class="queue-label" class:queue-label-queued={entry.state === "queued"}
              >{entry.label}</span
            >
            <span class="badge queue-tag" class:queue-tag-running={entry.state === "running"}
              >{entry.cancelling ? "cancelling" : entry.state}</span
            >
            <button
              type="button"
              class="queue-cancel"
              aria-label={`Cancel ${entry.label}`}
              title={entry.state === "running"
                ? "Stop this job at its next check-in"
                : "Remove from queue"}
              disabled={entry.cancelling}
              onclick={() => onCancel(entry)}
            >
              <Icon name="unapprove" />
            </button>
          </div>
          {#if entry.progress}
            {#if hasDoneTotal(entry.progress)}
              {@const { done, total } = entry.progress}
              <div class="queue-progress">
                <div
                  class="queue-progress-track"
                  role="progressbar"
                  aria-valuenow={done}
                  aria-valuemin={0}
                  aria-valuemax={total}
                  aria-label={entry.label}
                >
                  <div
                    class="queue-progress-fill"
                    style:width={total > 0 ? `${(done / total) * 100}%` : "0%"}
                  ></div>
                </div>
                <span class="queue-progress-text">{done}/{total}</span>
              </div>
            {:else}
              <span class="queue-progress-stage">{entry.progress.stage}</span>
            {/if}
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .queue-panel {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 26px; /* clears the status bar */
    width: 320px;
    z-index: 90;
    background: var(--bg-1);
    border-left: 1px solid var(--border);
    overflow-y: auto;
    padding: var(--space-3);
  }
  .queue-panel[hidden] {
    /* position: fixed would otherwise override the UA's [hidden] rule */
    display: none;
  }
  .queue-panel:focus-visible {
    outline: 3px solid var(--focus);
    outline-offset: 1px;
  }
  .queue-empty {
    color: var(--text-2);
    font-size: var(--text-sm);
    margin: 0;
  }
  .queue-actions {
    display: flex;
    justify-content: flex-end;
    margin-bottom: var(--space-2);
  }
  .queue-cancel-all {
    font-size: var(--text-sm);
    color: var(--text-2);
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-1);
    padding: 2px var(--space-2);
    cursor: pointer;
  }
  .queue-cancel-all:hover {
    color: var(--text-1);
    background: var(--bg-3);
  }
  .queue-cancel-all:focus-visible {
    outline: 3px solid var(--focus);
    outline-offset: 1px;
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .queue-entry {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    font-size: var(--text-sm);
    border-bottom: 1px solid var(--border);
    padding-bottom: var(--space-2);
  }
  .queue-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
  }
  .queue-label {
    color: var(--text-1);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .queue-label-queued {
    color: var(--text-2);
  }
  .queue-cancel {
    flex: 0 0 auto;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    color: var(--text-2);
    background: transparent;
    border: 1px solid transparent;
    border-radius: var(--radius-1);
    cursor: pointer;
    padding: 0;
  }
  .queue-cancel:hover:not(:disabled) {
    color: var(--text-1);
    background: var(--bg-3);
  }
  .queue-cancel:focus-visible {
    outline: 3px solid var(--focus);
    outline-offset: 1px;
  }
  .queue-cancel:disabled {
    /* Matches .btn:disabled's designed state (app.css): recede via the
       muted-label token instead of a blanket opacity wash. */
    color: var(--text-3);
    cursor: default;
  }
  .queue-progress {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .queue-progress-track {
    flex: 1;
    height: 3px;
    border-radius: var(--radius-1);
    background: var(--bg-3);
    overflow: hidden;
  }
  .queue-progress-fill {
    height: 100%;
    background: var(--accent);
    border-radius: var(--radius-1);
  }
  .queue-progress-text {
    flex: 0 0 auto;
    color: var(--text-2);
    font-variant-numeric: tabular-nums;
  }
  .queue-progress-stage {
    color: var(--text-2);
  }
  .queue-tag {
    flex: 0 0 auto;
    color: var(--text-2);
    background: var(--bg-2);
  }
  .queue-tag-running {
    color: var(--on-accent);
    background: var(--accent);
  }
</style>
