# Roll Workflow Polish Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close out the accumulated small UX gaps: the filmstrip fills with thumbnails in seconds (not one per detection), single-image mode gets its export button, ring markers stop flashing the wrong frame's boxes, a stale heal announces itself, and the status line counts background jobs.

**Architecture:** Five independent, small changes. The scan queue splits into two passes (thumbnails for every frame first — skipping ones that already exist on disk — then the slow detections, which stop building display pyramids entirely). Single-image export wires the existing `export_frame` command to a save dialog. The bboxes prop joins the `displayedIndex` pattern. Stale-heal marking is a frontend display hint (client-side comparison of the current threshold/strokes against those captured when the heal landed) — provenance in the cache remains the correctness mechanism; this is just honesty in the UI. The job counter reads the existing `jobStates` map.

**Tech Stack:** Existing stack; no new dependencies. `@tauri-apps/plugin-dialog`'s `save()` needs its capability allowed if not already.

## Global Constraints

- Trunk-based: commit directly to `main`, atomic commits, tests green first. No emoji; no Co-Authored-By.
- Two-pass scan keeps every robustness property of the current queue: single-flight flag + drop guard, generation snapshot checked per frame in BOTH passes, per-frame error events + continue, terminal `roll-done`. A frame whose pass-1 decode failed is skipped in pass 2 (no duplicate error events).
- Pass 1 must be cheap on backfill runs: when a frame's thumbnail file already exists, pass 1 skips it entirely (no decode). Pass 2 uses `decode_stage` (no display pyramid — the thumbnail was pass 1's job).
- The stale-heal hint is display-only: it never blocks SPACE (inspecting the old heal stays possible), never mutates anything, and disappears after a re-heal.
- All UI copy plain English, no emoji; button/status states WCAG-consistent with existing patterns.
- Rust fmt + clippy `-D warnings`; svelte-check 0 errors (4-warning baseline); vitest green.

---

### Task 1: Two-pass roll scan

**Files:**
- Modify: `app/src-tauri/src/lib.rs` (`scan_roll`'s worker task)

**Interfaces:**
- Consumes: everything scan_roll already uses, plus `roll::thumb_path` existence checks and `Images::decode_stage`.
- Produces: the spawned task becomes two sequential loops over the SAME `indices` snapshot:
  - **Pass 1 (thumbnails):** for each index — generation check; `frame_path`; if `thumb_path(...)` exists, `continue` (backfill frames already have thumbnails); otherwise `spawn_blocking { Images::prepare(&path) -> write_thumbnail(coarsest level) }` exactly as today, emitting `roll-thumb` on success and `roll-frame-error` + `record_scan_result(gen, idx, None, None)` on failure, remembering failed indices in a `HashSet<usize>`.
  - **Pass 2 (detections):** for each index not in the failed set — generation check; skip frames whose `defect_count` is already Some AND whose probs cache file exists (the backfill case where only the cache write is owed happens in pass 2's detect... careful: `frames_to_scan` already includes counted-but-uncached frames precisely so they re-detect and produce the cache; do NOT skip them); `spawn_blocking { let image = Images::decode_stage(&path); detect; write probs cache; components }` (no pyramid — this is the ff90566 export-queue shape), then `record_scan_result` + `roll-progress` exactly as today.
  - `prepared`/decoded pixels stay one-frame transient in both passes; `roll-done` emitted once after pass 2.
- The user-visible effect: every thumbnail lands within the first seconds of opening a fresh roll; detections then trickle in as before.

- [ ] **Step 1: Implement**

Read the current `scan_roll` task end to end first; this is a restructure of its loop body into two loops sharing the existing helpers, not new machinery. The pass-1 thumb-skip is a plain `.exists()` check (stale-thumb risk on changed files is the pre-existing name-keyed thumbnail tradeoff, unchanged). There is no automated harness for the queue (established); `frames_to_scan` tests already cover the queue-population semantics. Full gates: `cargo test -p unduster-app && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`.

- [ ] **Step 2: Commit**

```bash
git add app/src-tauri
git commit -m "Fill the filmstrip with thumbnails before detections start"
```

---

### Task 2: Single-image export button

**Files:**
- Modify: `app/src/App.svelte`
- Modify (if needed): `app/src-tauri/capabilities/default.json` (allow `dialog:allow-save` — check the existing capability entries; `open` is already allowed)

**Interfaces:**
- Consumes: existing command `export_frame(id: u64, dest: String) -> Result<usize, String>` (returns changed-pixel count), `save` from `@tauri-apps/plugin-dialog`.
- Produces: in single-image mode (no roll), when `info?.healed`, an "Export" button in the header: `save({ defaultPath: <original file name> })` — App does not currently retain the opened path's file name; capture it in `openScan` (`let scanFileName = $state<string | null>(null)`, set from the picked path's basename) — then `invoke("export_frame", { id: info.id, dest })`, disabled while a `exportingSingle` flag is set, surfacing errors via `error` and success via a transient status note (`exported N changed pixels` in the status line, cleared on next action or frame change). Roll mode is untouched (its export flow already exists).

- [ ] **Step 1: Implement**

Follow the header-button and status-line conventions. Verify the dialog save capability by testing an invoke in dev... a windowed check happens at the gate; statically, confirm the capabilities file lists the dialog plugin's save permission (add `"dialog:allow-save"` next to the existing `"dialog:allow-open"` if absent). From app/: `npm run test && npm run check`.

- [ ] **Step 2: Commit**

```bash
git add app/src app/src-tauri
git commit -m "Add the single-image export button"
```

---

### Task 3: Display-truth pair — bboxes index and the job counter

**Files:**
- Modify: `app/src/App.svelte`

**Interfaces:**
- Produces:
  - The Viewer's bboxes prop reads `roll.frames[displayedIndex].bboxes` instead of `currentIndex` (the strokeKey pattern from b5aa252) — ring markers can no longer show the NEXT frame's boxes over the CURRENT frame's pixels during a slow activation.
  - Status line gains `— {n} job{s} queued` whenever `Object.keys(jobStates).length > 0` (derived), giving background work a voice beyond filmstrip markers.

- [ ] **Step 1: Implement**

Two small changes; from app/: `npm run test && npm run check`.

- [ ] **Step 2: Commit**

```bash
git add app/src/App.svelte
git commit -m "Key ring markers to the displayed frame and count queued jobs"
```

---

### Task 4: Stale-heal hint

**Files:**
- Modify: `app/src/App.svelte`
- Modify: `app/src/lib/Viewer.svelte` (only if the hint needs healedAvailable context it lacks; prefer App-only)

**Interfaces:**
- Produces: App tracks `healInputs: Record<string, { threshold: number; strokeCount: number }>` keyed like `strokeKey()` — captured whenever a heal lands for the displayed frame (single-image `requestHeal` success; roll-mode `job-done` kind=heal for the current frame using the frame's persisted threshold; and on activation of a frame that arrives `healed: true`, captured from the frame's current persisted values — the restore case, where those values ARE the provenance inputs that matched). A derived `healStale` compares the current effective threshold and stroke count for the displayed frame against the captured pair; when `info?.healed && healStale`, the status line shows `— heal is stale (h re-heals)`. SPACE keeps working on the old heal (no behavior change). Re-heal replaces the captured inputs; frame switches read the new frame's entry.
- Stroke count is a deliberate approximation (a moved stroke with the same count escapes the hint); the provenance hash in the cache remains the correctness layer. Comment this at the declaration.

- [ ] **Step 1: Implement**

From app/: `npm run test && npm run check`.

- [ ] **Step 2: Commit**

```bash
git add app/src
git commit -m "Mark a heal stale when its inputs have moved on"
```

---

### Task 5: Sweep and manual gate

- [ ] **Step 1: Automated sweep**

Root: `cargo test`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all --check`. app/: `npm run test && npm run check`.

- [ ] **Step 2: Manual gate (human)**

`cd app && mise exec -- npm run tauri dev`:

1. Wipe `.unduster` on a roll and open it: ALL thumbnails appear within the first seconds, detections trickle in behind them.
2. Reopen without wiping: no thumbnail churn (pass 1 skips), backfill detections only if the cache dir was cleared.
3. Open a single image, detect, heal, press Export: save dialog defaults to the original name; the exported file opens correctly.
4. Navigate quickly between a scanned and an unscanned frame: rings never show the wrong frame's boxes during the decode wait.
5. Heal a frame, then move the threshold slider: "heal is stale" appears; SPACE still shows the old heal; `h` re-heals and the hint clears. Same after adding a brush stroke.
6. Queue several jobs: the status line counts them down as the filmstrip markers drain.

- [ ] **Step 3: Close out**

Ledger; `bd close` k9k, 1lq, pwh, e15.

---

## Definition of done

- Fresh rolls show a fully-populated filmstrip in seconds; rescans skip existing thumbnails; detect pass builds no display pyramids.
- Single-image mode can export its heal through a save dialog.
- Ring markers and stroke binding both follow `displayedIndex`; queued work is visible in the status line.
- A stale heal says so without blocking inspection of the old result.
- NOT here: cancellation UI, queue unification, hardening beads (gz6/mrr/sbo).
