<script lang="ts">
  interface QueueEntry {
    key: string;
    label: string;
    state: "running" | "queued";
  }

  let { entries, id }: { entries: QueueEntry[]; id: string } = $props();
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex -- same rationale as
     LogPanel: a scrollable region with no other focusable descendant needs
     an explicit tabindex for keyboard users to scroll it in WKWebView. -->
<div class="queue-panel" {id} role="region" aria-label="Job queue" tabindex="0">
  {#if entries.length === 0}
    <p class="queue-empty">queue is empty</p>
  {:else}
    <ul>
      {#each entries as entry (entry.key)}
        <li class="queue-entry">
          <span class="queue-label" class:queue-label-queued={entry.state === "queued"}
            >{entry.label}</span
          >
          <span class="badge queue-tag" class:queue-tag-running={entry.state === "running"}
            >{entry.state}</span
          >
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
  .queue-panel:focus-visible {
    outline: 3px solid var(--focus);
    outline-offset: 1px;
  }
  .queue-empty {
    color: var(--text-2);
    font-size: var(--text-sm);
    margin: 0;
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
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    font-size: var(--text-sm);
    border-bottom: 1px solid var(--border);
    padding-bottom: var(--space-2);
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
