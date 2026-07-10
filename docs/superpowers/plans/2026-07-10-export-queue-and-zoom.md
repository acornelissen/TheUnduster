# Export Queue and Zoom Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route roll exports through the existing background job queue (bead TheUnduster-bza) and fix zoom sensitivity plus add on-screen zoom controls (bead TheUnduster-b1q).

**Architecture:** Exports become a third `JobKind` in the per-roll job queue: one job per approved frame, enqueued back-of-queue so they run after any heals the operator queued first. The destination directory lives in a `RollState` slot (the `Copy` job struct can't carry a path). The old `export_approved` spawn loop, its `exporting` AtomicBool, and `ExportFlagGuard` are deleted; the frontend derives "exporting" from `jobStates`. Zoom sensitivity is fixed by scaling the wheel factor with delta magnitude via a pure, tested helper in viewport.ts; a token-styled control cluster overlays the viewer.

**Tech Stack:** Rust (Tauri 2 backend, `app/src-tauri`), Svelte 5 runes + TypeScript (`app/src`), vitest, cargo test.

## Global Constraints

- Trunk-based: commit directly to main, atomic commits, tests green each commit, plain-English why-focused messages, no Co-Authored-By, no emoji anywhere.
- Gates per commit touching Rust: `cargo test -p unduster-app`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all --check`. Per commit touching app/src: `npm run test` and `npm run check` from app/ (0 errors; warning baseline is exactly 4: Viewer.svelte x3, Filmstrip.svelte x1 — must not grow).
- Frontend visuals: design tokens from app/src/app.css only (no new hex/rgba), WCAG 2.2 AA, hit targets >= 24px, `:focus-visible` outline 3px solid var(--focus) offset 1px, reduced-motion respected (global kill block covers transitions/animations), tabular-nums on numeric readouts, lowercase status-fragment copy style.
- TDD for every pure helper (failing test first). Component/command glue without a unit seam is covered by the gates plus the final manual gate.
- Existing concurrency disciplines are load-bearing: generation-tagged jobs, per-job generation re-check in the drain loop, clear-then-recheck exit handshake, pin-before-registry-read with PinGuard, stamp-before-decode. Do not weaken any of them.

---

### Task 1: JobKind::Export and the export destination slot

**Files:**
- Modify: `app/src-tauri/src/jobs.rs` (enum variant + test)
- Modify: `app/src-tauri/src/roll.rs` (dest slot + tests; delete the `exporting` AtomicBool and `clear_exporting`)
- Modify: `app/src-tauri/src/lib.rs` (only what deleting the AtomicBool forces: see step 4)

**Interfaces:**
- Consumes: existing `JobQueue`, `RollState`.
- Produces: `jobs::JobKind::Export` (serializes as `"export"`); `RollState::set_export_dest(PathBuf) -> Result<(), String>` and `RollState::export_dest() -> Result<Option<PathBuf>, String>`. Task 2 depends on both.

- [ ] **Step 1: Failing tests**

In `jobs.rs` tests module:

```rust
#[test]
fn export_kind_is_a_distinct_job() {
    let q = JobQueue::default();
    assert!(q.enqueue(job(JobKind::Heal, 1), false).unwrap());
    assert!(q.enqueue(job(JobKind::Export, 1), false).unwrap()); // distinct kind, kept
    assert!(!q.enqueue(job(JobKind::Export, 1), false).unwrap()); // duplicate coalesced
    assert_eq!(q.len().unwrap(), 2);
}
```

In `roll.rs` tests module (match the file's existing test style for constructing a `RollState`):

```rust
#[test]
fn export_dest_round_trips_and_defaults_to_none() {
    let state = RollState::default();
    assert_eq!(state.export_dest().unwrap(), None);
    state
        .set_export_dest(std::path::PathBuf::from("/tmp/out"))
        .unwrap();
    assert_eq!(
        state.export_dest().unwrap(),
        Some(std::path::PathBuf::from("/tmp/out"))
    );
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p unduster-app export_kind_is_a_distinct_job export_dest_round_trips`
Expected: FAIL to compile (no `Export` variant, no `set_export_dest`).

- [ ] **Step 3: Implement**

`jobs.rs`: add `Export` to `JobKind` (after `Heal`; the serde `snake_case` rename gives `"export"` on the wire for free).

`roll.rs`: add to `RollState`:

```rust
/// Destination directory for queued export jobs. Set by `enqueue_exports`
/// each time the operator picks a directory; export jobs read it at run
/// time, so re-queuing with a new directory redirects still-queued jobs.
pub export_dest: Mutex<Option<PathBuf>>,
```

with:

```rust
pub fn set_export_dest(&self, dest: PathBuf) -> Result<(), String> {
    let mut guard = self.export_dest.lock().map_err(|e| e.to_string())?;
    *guard = Some(dest);
    Ok(())
}

pub fn export_dest(&self) -> Result<Option<PathBuf>, String> {
    let guard = self.export_dest.lock().map_err(|e| e.to_string())?;
    Ok(guard.clone())
}
```

Delete `pub exporting: AtomicBool` and `clear_exporting` from `RollState`.

- [ ] **Step 4: Keep the crate compiling**

Deleting the AtomicBool breaks `export_approved`/`ExportFlagGuard` in lib.rs, which Task 2 rewrites anyway. To keep this commit green without doing Task 2's work here, gut them in the same commit: delete `ExportFlagGuard` and its impl, and reduce `export_approved` to the smallest compiling stub that Task 2 will replace — or, if the deletions cascade too far, do the AtomicBool/`clear_exporting` deletion as the first step of Task 2 instead and keep this task to the additions only. Prefer the smaller diff; say which you chose in your report.

- [ ] **Step 5: Gates and commit**

Run: `cargo test -p unduster-app && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`
Expected: all green.

```bash
git add app/src-tauri/src
git commit -m "Add the export job kind and a roll-level export destination"
```

---

### Task 2: Export jobs run in the queue worker

**Files:**
- Modify: `app/src-tauri/src/lib.rs`

**Interfaces:**
- Consumes: `jobs::JobKind::Export`, `RollState::{set_export_dest, export_dest}` from Task 1; existing `run_job`, `enqueue_job`, `frames_to_export`, `export_frame_meta`, `set_exported`, `export::export_healed`, `cache::{read_heal, write_heal, heal_provenance}`, `stamp_or_skip`, `PinGuard`, events `export-frame-stage`, `export-heal-progress`, `export-progress`.
- Produces: Tauri command `enqueue_exports(dest_dir: String)` (frontend invokes with `{ destDir: dir }`); `run_job` handles `JobKind::Export`. The commands `export_approved` is DELETED from the handler list; events `export-frame-error` and `export-done` are no longer emitted (job-error and queue-idle replace them).

- [ ] **Step 1: Extract the worker spawn**

`enqueue_job` currently inlines try_start + the spawned drain loop. Extract everything from the `if !queue.try_start()` check through the end of the spawned task into:

```rust
/// Claims the worker flag and spawns the drain loop if no worker is
/// running. `spawn_generation` is the roll generation at the caller's
/// enqueue time; see the comment inside for why the terminal queue-idle
/// must carry it rather than a generation read at emit time.
fn spawn_worker_if_idle(app: &tauri::AppHandle, spawn_generation: u64) {
    let queue = app.state::<jobs::JobQueue>();
    if !queue.try_start() {
        return; // a worker is already draining; it will reach the new jobs
    }
    let app_for_task = app.clone();
    tauri::async_runtime::spawn(async move {
        // ... the existing drain body, moved verbatim (JobFlagGuard,
        // 'drain loop, per-job generation check, run_job match,
        // clear-then-recheck handshake, queue-idle with spawn_generation) ...
    });
}
```

`enqueue_job` becomes: validation, enqueue, `job-queued` emit, then `spawn_worker_if_idle(&app, generation)`. The move is verbatim — no logic changes inside the drain body.

- [ ] **Step 2: Add the enqueue_exports command**

```rust
#[tauri::command]
fn enqueue_exports(
    app: tauri::AppHandle,
    roll: State<'_, roll::RollState>,
    queue: State<'_, jobs::JobQueue>,
    dest_dir: String,
) -> Result<(), String> {
    // Validation before anything lands in the queue: errors when no roll
    // is open. frames_to_export already includes previously exported
    // frames -- re-export is deliberate, predictable overwrite.
    let indices = roll.frames_to_export()?;
    roll.set_export_dest(std::path::PathBuf::from(dest_dir))?;
    let generation = roll.generation();
    for index in indices {
        let newly_queued = queue.enqueue(
            jobs::Job {
                kind: jobs::JobKind::Export,
                index,
                generation,
            },
            false, // back of queue: never jumps ahead of queued heals
        )?;
        if newly_queued {
            let _ = app.emit(
                "job-queued",
                JobEvent {
                    index,
                    kind: jobs::JobKind::Export,
                    generation,
                },
            );
        }
    }
    spawn_worker_if_idle(&app, generation);
    Ok(())
}
```

- [ ] **Step 3: Add the Export arm to run_job**

The body is the old `export_approved` per-frame body, adapted: `path`, `file_name`, `frame_threshold` (as `threshold`), `frame_strokes` (as `strokes`), and `image_id` are already in scope from `run_job`'s prelude. Errors return `Err(message)` — the worker's existing match turns that into a `job-error` emit (kind `export`), which the frontend toasts; do NOT emit `export-frame-error`.

```rust
jobs::JobKind::Export => {
    let dest_dir = app
        .state::<roll::RollState>()
        .export_dest()?
        .ok_or("no export destination set")?;
    let dest = dest_dir.join(&file_name);

    // Prefer already-healed registry data (the operator reviewed it).
    // Pin before reading the Arcs so eviction cannot pull the entry out
    // from under the export; the guard unpins on every exit.
    let mut _pin_guard = None;
    let registry_export = match image_id {
        Some(id) => {
            app.state::<jobs::JobQueue>().pin(Some(id))?;
            _pin_guard = Some(PinGuard(app.clone()));
            let images_state = app.state::<Mutex<Images>>();
            let images = images_state.lock().map_err(|e| e.to_string())?;
            images.healed_parts(id)
        }
        None => None,
    };

    if let Some((original, healed, mask)) = registry_export {
        let dest_for_write = dest.clone();
        tauri::async_runtime::spawn_blocking(move || {
            export::export_healed(&original, &healed, &mask, &dest_for_write).map(|_| ())
        })
        .await
        .map_err(|e| e.to_string())
        .and_then(|r| r)?;
    } else {
        // No registry entry: heal cache, then the transient detect/heal
        // pipeline. Identical to run_heal's discipline: detector hash
        // resolved before the closure (read path, benign miss on race);
        // everything provenance-dependent inside one with_inpainter_hashed.
        let detector = app.state::<detect::DetectorState>().inner().clone();
        let inpainter = app.state::<detect::InpainterState>().inner().clone();
        let detector_hash = detector.hash().unwrap_or([0u8; 32]);
        let cache_path = roll::heal_cache_path(&roll_dir, &file_name);
        let app_for_stages = app.clone();
        let path_for_task = path.clone();
        let file_name_for_log = file_name.clone();
        tauri::async_runtime::spawn_blocking(move || {
            let stage = |s: &'static str| {
                let _ = app_for_stages
                    .emit("export-frame-stage", ExportFrameStage { index, stage: s });
            };
            // Stamp BEFORE decoding -- see run_heal's identical comment.
            // A stat failure skips the heal-cache interaction (read AND
            // write); it must never fail the export.
            let stamp = stamp_or_skip(&path_for_task);
            let image = images::Images::decode_stage(&path_for_task)?;
            masks::validate_strokes(&strokes)?;

            inpainter.with_inpainter_hashed(|pair| {
                let inpainter_hash = pair.as_ref().map(|(_, h)| *h).unwrap_or([0u8; 32]);
                let provenance = stamp.as_ref().map(|stamp| {
                    cache::heal_provenance(
                        threshold,
                        HEAL_DILATE_RADIUS,
                        &strokes,
                        &detector_hash,
                        &inpainter_hash,
                        stamp,
                    )
                });

                if let Some(provenance) = &provenance {
                    if let Some((healed, mask)) =
                        cache::read_heal(&cache_path, &image, provenance)
                    {
                        stage("writing");
                        return export::export_healed(&image, &healed, &mask, &dest)
                            .map(|_| ());
                    }
                }

                stage("detecting");
                let probs = detector.detect(&image)?;
                let raw: Vec<bool> = probs.iter().map(|&p| p > threshold).collect();
                let mask = masks::compose_heal_mask(
                    raw,
                    image.width,
                    image.height,
                    HEAL_DILATE_RADIUS,
                    &strokes,
                );
                stage("healing");
                let mut copy = (*image).clone();
                let inp = pair.map(|(inp, _)| inp);
                fd_heal::heal_with_progress(&mut copy, &mask, inp, &mut |done, total| {
                    let _ = app_for_stages.emit(
                        "export-heal-progress",
                        ExportHealProgress { index, done, total },
                    );
                })
                .map_err(|e| e.to_string())?;
                // Cache the fresh heal so the next export or heal of this
                // frame is instant.
                if let Some(provenance) = &provenance {
                    if let Err(_e) =
                        cache::write_heal(&cache_path, &image, &copy, &mask, provenance)
                    {
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "[cache] heal write failed for frame {index} ({file_name_for_log}): {_e}"
                        );
                    }
                }
                stage("writing");
                export::export_healed(&image, &copy, &mask, &dest).map(|_| ())
            })?
        })
        .await
        .map_err(|e| e.to_string())
        .and_then(|r| r)?;
    }

    // set_exported re-checks the generation under its own lock; the
    // worker's top-of-loop check plus this makes a roll-swap-mid-export
    // land nowhere.
    let _ = app
        .state::<roll::RollState>()
        .set_exported(generation, index);
    let _ = app.emit("export-progress", ExportProgress { index });
    Ok(())
}
```

Adapt field/variable names to what `run_job`'s prelude actually binds (`threshold` vs `frame_threshold` etc.) — the prelude is the source of truth. If the existing prelude binds `strokes` by value and the Detect/Heal arms consume it, clone what the Export arm needs.

- [ ] **Step 4: Delete the old path**

Delete: `export_approved` (whole command), `ExportFlagGuard` + Drop impl, `ExportFrameError` struct if now unused (check: nothing else emits `export-frame-error`), and `export_approved` from the `generate_handler![]` list; add `enqueue_exports` there. Keep `ExportFrameStage`, `ExportHealProgress`, `ExportProgress` (still emitted).

- [ ] **Step 5: Gates and commit**

Run: `cargo test -p unduster-app && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`
Expected: all green (clippy will catch any now-dead code missed in step 4).

```bash
git add app/src-tauri/src
git commit -m "Run roll exports through the job queue instead of a private loop"
```

---

### Task 3: Frontend rides the export jobs

**Files:**
- Modify: `app/src/App.svelte`

**Interfaces:**
- Consumes: `enqueue_exports` command and events from Task 2; existing `jobStates`, toast helpers.
- Produces: `jobStates` kind widened to `"detect" | "heal" | "export"`; `rollExporting` derived; no `exporting` $state.

- [ ] **Step 1: Widen the job kind and derive exporting**

In App.svelte:
- `jobStates` type: `Record<number, { state: "queued" | "running"; kind: "detect" | "heal" | "export" }>`.
- Delete `let exporting = $state(false);`.
- Add, next to the other jobStates deriveds:

```ts
// Exporting is queue state now, not a hand-managed flag: any export job
// queued or running means the roll is exporting.
const rollExporting = $derived(
  Object.values(jobStates).some((j) => j.kind === "export"),
);
```

- Every read of `exporting` switches to `rollExporting` (the status-input composition and the Export approved button's disabled prop; grep for the rest).

- [ ] **Step 2: Rewire exportApproved**

```ts
async function exportApproved() {
    if (!roll || rollExporting) return;
    const dir = await open({ directory: true });
    if (typeof dir !== "string") return;
    try {
      await invoke("enqueue_exports", { destDir: dir });
    } catch (e) {
      pushError(String(e));
    }
}
```

- [ ] **Step 3: Listener cleanup**

- Delete the `export-done` listener (its work — clearing the flag and `exportDetail` — is covered by: the flag no longer exists, and `exportDetail` clears per-frame on `export-progress`).
- Delete the `export-frame-error` listener (backend no longer emits it; per-frame failures arrive as `job-error` with kind `"export"`, which the existing job-error listener already toasts). In the job-error listener, when `e.payload.kind === "export"`, also clear `exportDetail` so a failed frame's narration doesn't linger.
- `export-frame-stage`, `export-heal-progress`, `export-progress` listeners stay as they are.

- [ ] **Step 4: Gates**

From app/: `npm run test && npm run check`.
Expected: 63 tests green (status.ts inputs are unchanged in shape — `exporting: boolean` is still a boolean, now fed by the derived); 0 errors, warning count still 4.

- [ ] **Step 5: Commit**

```bash
git add app/src
git commit -m "Derive export state from the job queue in the frontend"
```

---

### Task 4: Zoom sensitivity and on-screen zoom controls

**Files:**
- Modify: `app/src/lib/viewport.ts` (pure helper)
- Test: `app/src/lib/viewport.test.ts` (create if it does not exist; extend if it does)
- Modify: `app/src/lib/Viewer.svelte`

**Interfaces:**
- Consumes: existing `zoomAt`, `fitZoom`, `clampCenter`, `requestFrame` in Viewer.svelte.
- Produces: `wheelZoomFactor(deltaY: number, ctrlKey: boolean): number` in viewport.ts.

- [ ] **Step 1: Failing tests for the wheel factor**

```ts
import { describe, expect, it } from "vitest";
import { wheelZoomFactor } from "./viewport";

describe("wheelZoomFactor", () => {
  it("is 1 for zero delta", () => {
    expect(wheelZoomFactor(0, false)).toBe(1);
  });

  it("zooms in on negative delta, out on positive, proportionally", () => {
    expect(wheelZoomFactor(-40, false)).toBeGreaterThan(1);
    expect(wheelZoomFactor(40, false)).toBeLessThan(1);
    // small trackpad delta moves less than a large one
    expect(wheelZoomFactor(-4, false)).toBeLessThan(wheelZoomFactor(-40, false));
  });

  it("clamps a detented mouse-wheel notch", () => {
    expect(wheelZoomFactor(-120, false)).toBe(1.35);
    expect(wheelZoomFactor(120, false)).toBe(1 / 1.35);
  });

  it("responds more strongly to pinch (ctrlKey) at the same delta", () => {
    expect(wheelZoomFactor(-10, true)).toBeGreaterThan(wheelZoomFactor(-10, false));
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run from app/: `npx vitest run src/lib/viewport.test.ts`
Expected: FAIL — `wheelZoomFactor` is not exported.

- [ ] **Step 3: Implement the helper**

In viewport.ts:

```ts
/**
 * Multiplicative zoom factor for one wheel event. Trackpads emit streams
 * of small-delta events, so a fixed per-event factor (the old 1.15) races
 * away; scaling with delta magnitude keeps trackpads gentle while a
 * detented mouse wheel still moves a full clamped step per notch. Pinch
 * arrives as wheel-with-ctrlKey in WKWebView and needs a stronger
 * response to feel 1:1 with the gesture.
 */
export function wheelZoomFactor(deltaY: number, ctrlKey: boolean): number {
  const k = ctrlKey ? 0.01 : 0.002;
  const factor = Math.exp(-deltaY * k);
  return Math.min(Math.max(factor, 1 / 1.35), 1.35);
}
```

Run the test again: PASS. Then full `npm run test`.

- [ ] **Step 4: Use it in Viewer.svelte**

- Import `wheelZoomFactor` alongside the existing viewport imports.
- Replace `onWheel`'s body:

```ts
function onWheel(e: WheelEvent) {
    e.preventDefault();
    const dpr = window.devicePixelRatio || 1;
    zoomAt(wheelZoomFactor(e.deltaY, e.ctrlKey), e.offsetX * dpr, e.offsetY * dpr);
}
```

- [ ] **Step 5: Make zoom readable by the template and extract fit/actual**

- Change `let zoom = 1;` to `let zoom = $state(1);` (the readout below renders it; all existing reads/writes keep working — verify no new svelte-check warning appears).
- Extract the two snap actions the key handler already contains, and point the key handler at them:

```ts
function zoomFit() {
    zoom = fitZoom(info.levels[0], canvas.width, canvas.height);
    centerX = info.width / 2;
    centerY = info.height / 2;
    clampCenter();
    requestFrame();
}

function zoomActual() {
    zoom = 1;
    clampCenter();
    requestFrame();
}
```

In the key handler, the `"0"` branch becomes `zoomFit();` and the `"1"` branch `zoomActual();` (keep the shared trailing `e.preventDefault(); clampCenter(); requestFrame();` flow exactly as it is — the double clampCenter/requestFrame is harmless; do not restructure the chain).

- [ ] **Step 6: The control cluster**

Inside the element that wraps the canvas (give it `position: relative` in its style rule if it doesn't have it), after the canvas:

```svelte
<div class="zoom-controls">
  <button
    class="btn"
    title="Zoom out (-)"
    aria-label="Zoom out"
    onclick={() => zoomAt(1 / 1.25, canvas.width / 2, canvas.height / 2)}>&minus;</button
  >
  <span class="zoom-readout">{Math.round(zoom * 100)}%</span>
  <button
    class="btn"
    title="Zoom in (+)"
    aria-label="Zoom in"
    onclick={() => zoomAt(1.25, canvas.width / 2, canvas.height / 2)}>+</button
  >
  <button class="btn" title="Fit (0)" aria-label="Fit to window" onclick={zoomFit}>Fit</button>
  <button class="btn" title="100% (1)" aria-label="Actual size" onclick={zoomActual}>1:1</button>
</div>
```

Styles (component-scoped, tokens only):

```css
.zoom-controls {
    position: absolute;
    right: var(--space-3);
    bottom: var(--space-3);
    display: flex;
    align-items: center;
    gap: var(--space-1);
    padding: var(--space-1);
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: var(--radius-1);
}

.zoom-readout {
    min-width: 4ch;
    text-align: center;
    font-size: var(--text-sm);
    color: var(--text-2);
    font-variant-numeric: tabular-nums;
}
```

Notes: the `.btn` class already provides the 26px min-height, hover, and focus-visible ring; the buttons only render when the canvas does (`info` present), so `canvas.width` is always live. The cluster sits above the canvas, so wheel/pointer events over it do not reach the canvas handlers.

- [ ] **Step 7: Gates**

From app/: `npm run test && npm run check`.
Expected: 63 + 4 new = 67 tests green; 0 errors; warning count still exactly 4 (if `zoom` as `$state` trips a new `state_referenced_locally`, restructure the offending read into a closure/derived rather than accepting a fifth warning).

- [ ] **Step 8: Commit**

```bash
git add app/src
git commit -m "Scale wheel zoom with gesture delta and add zoom controls"
```

---

### Task 5: Sweep and manual gate

- [ ] **Step 1: Automated sweep**

Root: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`. app/: `npm run test && npm run check` (0 errors; warning count exactly 4).

- [ ] **Step 2: Manual gate (human)**

`cd app && mise exec -- npm run tauri dev`:

1. Open a roll, approve two frames (one healed in-registry, one untouched), queue a heal on a third, then Export approved: the status bar shows the jobs queued; exports run after the heal; the untouched frame narrates detecting/healing stages during its export; exported badges appear; files land in the chosen directory.
2. Click Export approved again mid-queue: no duplicate jobs (coalesced), still one pass.
3. Swap rolls mid-export: the export stops cleanly, no stray toasts, new roll unaffected.
4. Trackpad zoom: two-finger scroll is controlled, pinch feels 1:1, a mouse wheel notch still moves decisively.
5. Zoom controls: −/+/Fit/1:1 work, readout tracks, keyboard equivalents (+/−/0/1) unchanged, focus rings visible, cluster reads as Darkroom chrome.

- [ ] **Step 3: Close out**

Ledger; `bd close bza b1q`.

---

## Definition of done

- Roll exports are queue jobs: ordered after queued heals, coalesced, generation-guarded, narrated by the existing status/filmstrip/toast chrome; the private export loop, its flag, and its bespoke error/done events are gone.
- Wheel zoom scales with gesture delta (tested pure helper); pinch is stronger; per-event factor clamped.
- Zoom control cluster on the viewer: −/readout/+/Fit/1:1, tokens only, AA-compliant, matching existing shortcuts.
- NOT here: single-image export (stays a dialog flow), UI pass 2 items.
