<script lang="ts">
  let {
    left,
    activity,
    right,
    logOpen = false,
    onToggleLog = null,
    queueOpen = false,
    onToggleQueue = null,
  }: {
    left: string;
    activity: string | null;
    right: string;
    logOpen?: boolean;
    onToggleLog?: (() => void) | null;
    queueOpen?: boolean;
    onToggleQueue?: (() => void) | null;
  } = $props();
</script>

<div class="status-bar">
  <!-- aria-live is a deliberate three-way split (TheUnduster-dm2): the
       CENTER zone is activity narration (downloading/healing/exporting) and
       is polite-live -- that is the one stream worth announcing unprompted.
       The LEFT zone is frame identity, which changes on every navigation;
       announcing it would talk over the operator's own actions (the
       filmstrip option's accessible name already announces selection). The
       RIGHT zone is standing state (engine, counts); its transitions ride
       along with the events that caused them (toasts, job events), so it
       stays non-live too and is read on demand. -->
  <span class="zone zone-left">{left}</span>
  <span class="zone zone-activity" aria-live="polite">
    {#if activity}
      <span class="dot" aria-hidden="true"></span>{activity}
    {/if}
  </span>
  <span class="zone zone-right">
    <span class="zone-right-text">{right}</span>
    {#if onToggleQueue}
      <button
        class="btn btn-toggle"
        onclick={onToggleQueue}
        aria-expanded={queueOpen}
        aria-controls="job-queue-panel"
      >
        Queue
      </button>
    {/if}
    {#if onToggleLog}
      <button
        class="btn btn-toggle"
        onclick={onToggleLog}
        aria-expanded={logOpen}
        aria-controls="activity-log-panel"
      >
        Log
      </button>
    {/if}
  </span>
</div>

<style>
  .status-bar {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    height: 26px;
    padding: 0 var(--space-3);
    background: var(--bg-1);
    border-top: 1px solid var(--border);
    font-size: var(--text-sm);
    font-variant-numeric: tabular-nums;
  }
  .zone {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .zone-left {
    color: var(--text-2);
    justify-self: start;
  }
  .zone-activity {
    color: var(--text-1);
    justify-self: center;
    display: flex;
    align-items: center;
    gap: var(--space-1);
  }
  .zone-right {
    color: var(--text-2);
    justify-self: end;
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .zone-right-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    flex: 0 0 auto;
  }
  .btn-toggle {
    flex: 0 0 auto;
    font-size: var(--text-xs);
    min-height: 24px;
    padding: 0 var(--space-2);
  }
</style>
