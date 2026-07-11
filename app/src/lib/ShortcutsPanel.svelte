<script lang="ts">
  let { onClose }: { onClose: () => void } = $props();

  let body: HTMLDivElement | undefined = $state();

  // Focus moves into the dialog on open (aria-modal without moving focus is
  // a WCAG dialog failure) and back to whatever had it when the panel
  // closes. The component only exists while the panel is open, so mount is
  // open and effect cleanup is close.
  $effect(() => {
    const previous = document.activeElement;
    body?.focus();
    return () => {
      if (previous instanceof HTMLElement) previous.focus();
    };
  });
</script>

<!-- Borderless button, not a div+role="presentation": a real <button> gets
     click/keyboard activation and focus semantics for free and svelte-check
     raises no a11y warning for it, unlike a non-interactive element with a
     synthetic click handler. -->
<button class="backdrop" aria-label="Close shortcuts" onclick={onClose}></button>

<div class="shortcuts-panel" role="dialog" aria-modal="true" aria-label="keyboard shortcuts">
  <div class="shortcuts-header">
    <h2>Keyboard shortcuts</h2>
    <button class="close-btn" aria-label="Close" onclick={onClose}>&#215;</button>
  </div>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -- same rationale as
       LogPanel/QueuePanel: this is the scrollable region (overflow-y: auto)
       and it has no focusable descendant, so keyboard users in WKWebView
       need an explicit tabindex on it to scroll. It also serves as the
       dialog's initial focus target (see the $effect above). -->
  <div class="shortcuts-body" bind:this={body} tabindex="0">
    <section>
      <h3>Viewer</h3>
      <ul>
        <li><kbd class="kbd">d</kbd> detect</li>
        <li><kbd class="kbd">h</kbd> heal</li>
        <li><kbd class="kbd">space</kbd> before/after (healed)</li>
        <li><kbd class="kbd">m</kbd> overlay</li>
        <li><kbd class="kbd">z</kbd> / <kbd class="kbd">shift-z</kbd> cycle defects</li>
        <li><kbd class="kbd">+</kbd> <kbd class="kbd">-</kbd> zoom</li>
        <li><kbd class="kbd">0</kbd> fit</li>
        <li><kbd class="kbd">1</kbd> 100%</li>
        <li><kbd class="kbd">arrows</kbd> pan</li>
      </ul>
    </section>
    <section>
      <h3>Brush</h3>
      <ul>
        <li><kbd class="kbd">b</kbd> paint</li>
        <li><kbd class="kbd">e</kbd> erase</li>
        <li><kbd class="kbd">[</kbd> <kbd class="kbd">]</kbd> size</li>
        <li><kbd class="kbd">arrows</kbd> nudge (shift: faster)</li>
        <li><kbd class="kbd">enter</kbd> stamp</li>
        <li><kbd class="kbd">esc</kbd> exit</li>
      </ul>
    </section>
    <section>
      <h3>Roll</h3>
      <ul>
        <li><kbd class="kbd">,</kbd> <kbd class="kbd">.</kbd> previous/next frame</li>
        <li><kbd class="kbd">a</kbd> approve</li>
        <li><kbd class="kbd">shift-a</kbd> unapprove</li>
      </ul>
    </section>
    <section>
      <h3>Everywhere</h3>
      <ul>
        <li><kbd class="kbd">cmd-z</kbd> undo</li>
        <li><kbd class="kbd">shift-cmd-z</kbd> redo</li>
        <li><kbd class="kbd">?</kbd> this panel</li>
        <li><kbd class="kbd">esc</kbd> close panels</li>
      </ul>
    </section>
  </div>
</div>

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    z-index: 190;
    background: rgba(20, 20, 20, 0.6);
    border: none;
    padding: 0;
    margin: 0;
    cursor: default;
    -webkit-app-region: no-drag;
  }
  .shortcuts-panel {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 191;
    display: flex;
    flex-direction: column;
    width: 90vw;
    max-width: 560px;
    max-height: 70vh;
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: var(--radius-2);
    overflow: hidden;
  }
  .shortcuts-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
    border-bottom: 1px solid var(--border);
    flex: 0 0 auto;
  }
  .shortcuts-header h2 {
    margin: 0;
    font-size: var(--text-lg);
    color: var(--text-1);
  }
  .close-btn {
    flex: 0 0 auto;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-2);
    color: var(--text-1);
    border: 1px solid var(--border);
    border-radius: var(--radius-1);
    font-size: var(--text-lg);
    line-height: 1;
    cursor: pointer;
  }
  .close-btn:hover {
    background: var(--bg-3);
  }
  .close-btn:focus-visible {
    outline: 3px solid var(--focus);
    outline-offset: 1px;
  }
  .shortcuts-body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--space-4);
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: var(--space-4);
  }
  .shortcuts-body:focus-visible {
    outline: 3px solid var(--focus);
    outline-offset: -3px;
  }
  section h3 {
    margin: 0 0 var(--space-2) 0;
    font-size: var(--text-sm);
    color: var(--text-2);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  section ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    font-size: var(--text-sm);
    color: var(--text-1);
  }
  .kbd {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-1);
    padding: 0 var(--space-1);
    font-size: var(--text-xs);
    font-family: inherit;
  }
</style>
