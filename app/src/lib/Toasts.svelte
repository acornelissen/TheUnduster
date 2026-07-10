<script lang="ts">
  import type { Toast } from "./toasts";

  let { toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void } = $props();

  // Info toasts auto-dismiss after 4s. One timer per toast id, cleared when
  // the toast leaves the list (dismissed, or the component unmounts) so a
  // stale timer can never fire an onDismiss for an id that's already gone.
  const timers = new Map<number, ReturnType<typeof setTimeout>>();

  $effect(() => {
    const liveIds = new Set(toasts.map((t) => t.id));
    for (const t of toasts) {
      if (t.level === "info" && !timers.has(t.id)) {
        timers.set(
          t.id,
          setTimeout(() => {
            timers.delete(t.id);
            onDismiss(t.id);
          }, 4000),
        );
      }
    }
    for (const [id, timer] of timers) {
      if (!liveIds.has(id)) {
        clearTimeout(timer);
        timers.delete(id);
      }
    }
  });

  $effect(() => {
    return () => {
      for (const timer of timers.values()) clearTimeout(timer);
      timers.clear();
    };
  });
</script>

<div class="toast-stack">
  {#each toasts as toast (toast.id)}
    <div
      class="toast"
      class:toast-error={toast.level === "error"}
      class:toast-info={toast.level === "info"}
      role={toast.level === "error" ? "alert" : "status"}
    >
      <span class="toast-message"
        >{toast.message}{#if toast.count > 1}<span class="toast-count"> x{toast.count}</span
          >{/if}</span
      >
      {#if toast.level === "error"}
        <button class="toast-dismiss" aria-label="Dismiss" onclick={() => onDismiss(toast.id)}>
          &#215;
        </button>
      {/if}
    </div>
  {/each}
</div>

<style>
  .toast-stack {
    position: fixed;
    top: calc(var(--space-6) + var(--space-2));
    right: var(--space-3);
    z-index: 100;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    max-width: 360px;
  }
  .toast {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    background: var(--bg-2);
    border-left: 4px solid var(--info);
    border-radius: var(--radius-1);
    padding: var(--space-2) var(--space-3);
    color: var(--text-1);
    font-size: var(--text-sm);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    /* Entrance is a plain CSS opacity transition (fired via @starting-style
       below), so the global prefers-reduced-motion kill block in app.css
       (transition: none !important) disables it wholesale. */
    opacity: 1;
    transition: opacity 120ms ease-out;
  }
  @starting-style {
    .toast {
      opacity: 0;
    }
  }
  .toast-error {
    border-left-color: var(--err);
  }
  .toast-info {
    border-left-color: var(--info);
  }
  .toast-message {
    flex: 1;
    min-width: 0;
    word-break: break-word;
  }
  .toast-count {
    font-variant-numeric: tabular-nums;
    color: var(--text-2);
  }
  .toast-dismiss {
    flex: 0 0 auto;
    background: transparent;
    border: none;
    color: var(--text-2);
    font-size: var(--text-lg);
    line-height: 1;
    cursor: pointer;
    padding: 0 var(--space-1);
  }
  .toast-dismiss:hover {
    color: var(--text-1);
  }
  .toast-dismiss:focus-visible {
    outline: 3px solid var(--focus);
    outline-offset: 1px;
  }
</style>
