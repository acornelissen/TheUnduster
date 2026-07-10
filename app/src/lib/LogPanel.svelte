<script lang="ts">
  interface LogEntry {
    time: string;
    level: string;
    message: string;
  }

  let { entries, id }: { entries: LogEntry[]; id: string } = $props();

  // Newest-first: reverse a copy, never mutate the prop array.
  const reversed = $derived([...entries].reverse());
</script>

<div class="log-panel" {id} role="log" aria-label="Activity log">
  {#if reversed.length === 0}
    <p class="log-empty">No activity yet.</p>
  {:else}
    <ul>
      {#each reversed as entry, i (entries.length - 1 - i)}
        <li class="log-entry" class:log-error={entry.level === "error"}>
          <span class="log-time">{entry.time}</span>
          <span class="log-message">{entry.message}</span>
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .log-panel {
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
  .log-empty {
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
  .log-entry {
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-size: var(--text-sm);
    border-bottom: 1px solid var(--border);
    padding-bottom: var(--space-2);
  }
  .log-time {
    color: var(--text-2);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
  }
  .log-message {
    color: var(--text-1);
  }
  .log-error .log-message {
    color: var(--err);
  }
</style>
