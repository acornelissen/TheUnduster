<script lang="ts">
  let {
    left,
    activity,
    right,
  }: { left: string; activity: string | null; right: string } = $props();
</script>

<div class="status-bar">
  <span class="zone zone-left">{left}</span>
  <span class="zone zone-activity" aria-live="polite">
    {#if activity}
      <span class="dot" aria-hidden="true"></span>{activity}
    {/if}
  </span>
  <span class="zone zone-right">{right}</span>
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
  }
  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    flex: 0 0 auto;
  }
</style>
