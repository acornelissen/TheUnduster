# Hardening Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six tracked robustness gaps before the app grows more users than its author: cross-roll event/registry races, cache staleness against mutated source files, queue lost-wakeups, the detector hash pairing race, and two stroke-input hardening items.

**Architecture:** Six independent fixes, each mapped to an open bead. Roll generation becomes part of job events and registry landings (gz6). Cache entries bind to the source file's size+mtime — a format version bump purges old probs files, and heal provenance simply grows two inputs (mrr). The job worker and scan queue close their lost-wakeup windows with a clear-then-recheck handshake (sbo). DetectorState gains an atomic detect-and-hash so cache writes can never tag one model's output with another's hash (uke). Sidecar-loaded strokes are validated at load, dropping bad lists instead of wedging the frame (k2s). And `apply_strokes` caps total rasterization work before doing any, turning the crafted-sidecar CPU bomb into a no-op (2mn).

**Tech Stack:** Existing stack; no new dependencies.

## Global Constraints

- Trunk-based: commit directly to `main`, atomic commits, tests green first. No emoji; no Co-Authored-By.
- Every fix preserves the established degradation postures: validating setters stay the last line of defense; cache misses are silent recomputes; boundary rejects are no-ops or clean errors, never panics.
- Cache format changes are self-migrating: an unreadable/old-version file is deleted on sight and recomputed — no migration code, no user action.
- The lost-wakeup fixes must not introduce busy-spinning or a new race: the handshake pattern is clear-flag -> recheck-work -> re-CAS -> continue-or-exit, and losing the re-CAS to another starter is a valid exit.
- Rust fmt + clippy `-D warnings`; vitest/svelte-check untouched-green except where the frontend genuinely changes (gz6).

---

### Task 1 (2mn): Bound stroke rasterization work

**Files:**
- Modify: `app/src-tauri/src/masks.rs`
- Test: inline

**Interfaces:**
- Produces: `apply_strokes` pre-scans all segments, summing CLAMPED bbox areas (the exact per-segment work bound), and returns without touching the mask when the total exceeds `MAX_RASTER_AREA_FACTOR (32) * width * height` pixel-ops — a crafted sidecar (512 strokes x 4096 points spanning the image at radius 512) becomes an O(segments) no-op instead of ~1e14 pixel writes inside an export task. The pre-scan itself is bounded by the already-enforced stroke/point count caps. Honest editing is unaffected: 32x the image area is far beyond any real retouching session's clamped footprint.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn pathological_stroke_area_is_a_no_op() {
        // 512 strokes x 2 points spanning the whole image at max radius:
        // clamped bboxes each cover the full 64x64 image, total far past
        // 32x image area. Must return untouched without rasterizing.
        let mut mask = vec![false; 64 * 64];
        let strokes: Vec<Stroke> = (0..MAX_STROKES)
            .map(|_| Stroke {
                erase: false,
                radius: MAX_RADIUS,
                points: vec![[0.0, 0.0], [63.0, 63.0]],
            })
            .collect();
        apply_strokes(&mut mask, 64, 64, &strokes);
        assert!(mask.iter().all(|&b| !b), "area bomb must not rasterize");
    }

    #[test]
    fn honest_strokes_stay_under_the_area_cap() {
        // A generous real session: 100 dabs at radius 24 on a small image.
        let mut mask = vec![false; 256 * 256];
        let strokes: Vec<Stroke> = (0..100)
            .map(|i| Stroke {
                erase: false,
                radius: 24.0,
                points: vec![[(i % 16) as f32 * 16.0, (i / 16) as f32 * 16.0]],
            })
            .collect();
        apply_strokes(&mut mask, 256, 256, &strokes);
        assert!(mask.iter().any(|&b| b), "honest strokes must rasterize");
    }
```

Run: `cargo test -p unduster-app masks` — the first test FAILS against current code (the bomb rasterizes).

- [ ] **Step 2: Implement**

```rust
/// Ceiling on total rasterization work, as a multiple of the image area.
/// Honest retouching stays orders of magnitude below this; a crafted
/// sidecar spanning the image with hundreds of max-radius segments would
/// otherwise cost ~1e14 pixel writes inside an export task.
const MAX_RASTER_AREA_FACTOR: u64 = 32;
```

In `apply_strokes`, after the existing degenerate-input guards, compute per segment (and per single-point dab) the clamped bbox dimensions exactly as `stamp_capsule` would, accumulate `(x1 - x0 + 1) as u64 * (y1 - y0 + 1) as u64`, and if the total exceeds `MAX_RASTER_AREA_FACTOR * width as u64 * height as u64`, `return` (debug-log the rejection). Factor the clamped-bbox computation into a small helper shared with `stamp_capsule` so the estimate and the work can never drift.

- [ ] **Step 3: Gates, commit**

`cargo test -p unduster-app && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`

```bash
git add app/src-tauri
git commit -m "Cap total stroke rasterization work before doing any"
```

---

### Task 2 (k2s): Validate sidecar strokes at load

**Files:**
- Modify: `app/src-tauri/src/roll.rs`
- Test: inline

**Interfaces:**
- Produces: `load_sidecar` (or the `merge` step — implementer's choice, wherever frames materialize) runs `crate::masks::validate_strokes` on each frame's `strokes` and `redo_strokes`; an invalid list is replaced with `Vec::new()` and debug-logged. Today an invalid hand-edited stroke wedges the frame: every rasterization boundary correctly rejects the whole list, so new strokes append to a list that never validates, and undo pops the operator's stroke rather than the bad one.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn invalid_sidecar_strokes_are_dropped_at_load() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        // persist a valid stroke, then corrupt it on disk to NaN coords
        state
            .set_strokes(
                0,
                vec![crate::masks::Stroke {
                    erase: false,
                    radius: 5.0,
                    points: vec![[1.0, 1.0]],
                }],
                vec![],
            )
            .unwrap();
        let sidecar = dir.path().join(".unduster/roll.json");
        let text = std::fs::read_to_string(&sidecar).unwrap();
        let text = text.replace("1.0", "1e30"); // out of coordinate range
        std::fs::write(&sidecar, text).unwrap();
        let (info, _) = state.open(dir.path()).unwrap();
        assert!(
            info.frames[0].strokes.is_empty(),
            "invalid strokes must be dropped, not wedge the frame"
        );
    }
```

Run: `cargo test -p unduster-app invalid_sidecar` — FAIL (strokes load as-is).

(If `1e30` survives the coordinate-range check, use `"radius\": 5.0` -> `"radius\": 99999.0` instead — anything `validate_strokes` rejects. Verify the replacement actually lands by asserting the pre-open file contains it.)

- [ ] **Step 2: Implement**

At frame materialization, for each of the two lists: `if crate::masks::validate_strokes(&frame.strokes).is_err() { frame.strokes = Vec::new(); }` with a debug `eprintln!` naming the file. Same for `redo_strokes`.

- [ ] **Step 3: Gates, commit**

```bash
git add app/src-tauri
git commit -m "Drop invalid sidecar strokes at load instead of wedging the frame"
```

---

### Task 3 (uke): Atomic detect-and-hash

**Files:**
- Modify: `app/src-tauri/src/detect.rs`
- Modify: `app/src-tauri/src/lib.rs` (call sites)
- Test: inline in detect.rs

**Interfaces:**
- Produces: `DetectorState::detect_hashed(&self, img: &ImageBuf) -> Result<(Vec<f32>, [u8; 32]), String>` — one lock acquisition covering both the inference and the hash read, mirroring `with_inpainter_hashed`'s rationale. All three production detect sites that also record a hash (run_detect, the scan queue's pass 2, the job worker's detect arm and heal-arm fallback) switch to it and use the RETURNED hash for their probs-cache writes; the standalone `detect()` remains for any caller that doesn't need pairing, and `hash()` remains for the read-only cache-lookup paths (activate restore, heal provenance) where a race means a benign cache miss, documented at each remaining `hash()` call site with one line.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn detect_hashed_pairs_output_with_the_producing_model() {
        let state = DetectorState::default();
        state.load(&fixture()).unwrap();
        let expected = state.hash().unwrap();
        let img = ImageBuf {
            width: 64,
            height: 48,
            channels: 1,
            data: PixelData::U8(vec![128; 64 * 48]),
            icc: None,
            exif: None,
        };
        let (probs, hash) = state.detect_hashed(&img).unwrap();
        assert_eq!(probs.len(), 64 * 48);
        assert_eq!(hash, expected);
        let none = DetectorState::default();
        assert!(none.detect_hashed(&img).is_err());
    }
```

Run: `cargo test -p unduster-app detect_hashed` — FAIL.

- [ ] **Step 2: Implement**

One lock, both reads, matching `detect()`'s error shapes. Then update the three call sites: where they currently resolve `detector.hash()` before `spawn_blocking` and call `detector.detect(&img)` inside, they now call `detect_hashed` inside the closure and use the returned hash for `write_probs` (the pre-spawn hash resolution disappears at those sites). Heal-arm fallback in the job worker likewise. Read each site fully first; the probs-cache write must use the paired hash.

- [ ] **Step 3: Gates, commit**

```bash
git add app/src-tauri
git commit -m "Pair detector output with its model hash under one lock"
```

---

### Task 4 (mrr): Bind cache entries to source content identity

**Files:**
- Modify: `app/src-tauri/src/cache.rs` (probs format v2 with source stamp; heal provenance grows two inputs)
- Modify: `app/src-tauri/src/lib.rs` (all cache read/write sites stat the source)
- Test: inline in cache.rs

**Interfaces:**
- Produces:
  - `pub struct SourceStamp { pub size: u64, pub mtime_nanos: u64 }` and `pub fn source_stamp(path: &Path) -> Result<SourceStamp, String>` (fs::metadata; `modified()` as nanos since epoch, truncated to u64 — collision requires same size AND same nanosecond mtime, good enough against rescans/edits).
  - Probs format version bumps to 2; header gains `size: u64 | mtime_nanos: u64` after the detector hash; `write_probs`/`read_probs` gain a `stamp: &SourceStamp` parameter; a version-1 file (or any unknown version) is deleted on sight (self-migration); a stamp mismatch on a well-formed v2 file is mismatch-keep semantics like dims/hash.
  - `heal_provenance` gains `source: &SourceStamp` (hashed after the model hashes, fixed order); the heal file format is unchanged (provenance absorbs the stamp), but bump `HEAL_MAGIC` to `UNDHEAL2` so pre-stamp heal files purge on first read instead of lingering unmatched forever.
  - Every producer/consumer stats the source: run_detect + scan pass 2 + job worker (they hold the path), activate restore (dir+file_name -> path), run_heal + export + job heal (path already resolved). A stat failure skips the cache interaction (miss/no-write) — never fails the operation.
- Existing cache tests update for the new parameters; provenance-distinguishing test gains the stamp case.

- [ ] **Step 1: Write the failing tests**

Extend the existing suites: `probs_round_trip_within_quantization` and friends thread a stamp; add `probs_reject_stamp_mismatch` (same file, stamp with size+1 -> None, file kept) and `old_version_probs_purge_on_sight` (write a v2 file, patch the version field to 1, read -> None and file DELETED). For heal: extend `heal_provenance_distinguishes_every_input` with a stamp variation; add `old_magic_heal_purges` (patch magic to UNDHEAL1 -> corrupt-delete semantics).

Run: `cargo test -p unduster-app cache` — FAIL (signatures).

- [ ] **Step 2: Implement**

Format and parameter changes per the interface block; call sites per the list, each `match source_stamp(&path) { Ok(s) => ...cache interaction..., Err(_) => ...skip... }`. Keep the checked-arithmetic and bounded-decompress discipline for the two new header fields.

- [ ] **Step 3: Gates, commit**

```bash
git add app/src-tauri
git commit -m "Bind cache entries to the source file's size and mtime"
```

---

### Task 5 (gz6): Roll generation through job events and registry landings

**Files:**
- Modify: `app/src-tauri/src/lib.rs` (events + run_job registry gating + open_roll response)
- Modify: `app/src-tauri/src/roll.rs` (RollInfo gains generation)
- Modify: `app/src/App.svelte` (store the generation; check it in job listeners)
- Test: roll.rs inline for RollInfo; the rest is the established no-harness command surface

**Interfaces:**
- Produces:
  - `JobEvent`/`JobError` gain `generation: u64` (from `job.generation`).
  - `RollInfo` gains `pub generation: u64` (populated at `open` — `RollState::open` returns it; check how `info()` composes and where open_roll builds the response).
  - run_job gates its registry landings: immediately before `set_probs_built`/`set_healed`, `if job.generation != roll.generation() { skip registry landing, keep the cache write }` — closing the same-index-same-dims cross-roll landing (review finding B).
  - Frontend: `rollGeneration` captured at openRoll; all four job listeners drop events whose `generation !== rollGeneration` (replacing the index-in-jobStates heuristic as the primary guard; keep the index guard as belt-and-braces).
- The frontend TS event types gain the field; FrameInfo untouched.

- [ ] **Step 1: Failing test (roll.rs)**

```rust
    #[test]
    fn roll_info_carries_the_generation() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        let state = RollState::default();
        let (info, _) = state.open(dir.path()).unwrap();
        assert_eq!(info.generation, state.generation());
        let (info2, _) = state.open(dir.path()).unwrap();
        assert!(info2.generation > info.generation);
    }
```

Run: `cargo test -p unduster-app roll_info_carries` — FAIL.

- [ ] **Step 2: Implement**

Backend then frontend per the interface block. The registry-landing gate goes in both run_job arms at the point the late `current_id` is resolved (one combined condition). Gates: full Rust + `npm run test && npm run check` from app/.

- [ ] **Step 3: Commit**

```bash
git add app
git commit -m "Thread the roll generation through job events and registry landings"
```

---

### Task 6 (sbo): Close the lost-wakeup windows

**Files:**
- Modify: `app/src-tauri/src/lib.rs` (job worker exit handshake; scan_roll re-arm)
- Modify: `app/src-tauri/src/jobs.rs` (if a combined helper helps; optional)
- Test: jobs.rs inline for any new queue helper; the handshake itself is the no-harness command surface

**Interfaces:**
- Produces:
  - Job worker: when `pop()` returns None, instead of exiting directly: emit nothing yet; `queue.clear_running()`; re-check `queue.is_empty()`; if non-empty, attempt the compare_exchange — on winning, continue draining (a job landed in the window and this worker adopts it); on losing, exit silently (the enqueuer's own start won the flag and spawned a drain). Only after a clear-flag + empty-queue observation does the worker emit `queue-idle` and exit. Structure it so the JobFlagGuard's unconditional clear on unwind still holds (the guard's clear of an already-cleared flag is harmless).
  - scan_roll: after its final frame, the task clears the scanning flag (guard), then re-checks `frames_to_scan()` for the CURRENT generation; if non-empty and the CAS re-wins, it re-arms with a fresh generation snapshot and drains again — closing the "open_roll B while roll A's scan drains: B's scan_roll call returned idempotent-Ok and B never scans" hole. Cap re-arms at a small constant (3) against pathological loops, logging when the cap trips.
  - export_approved's window is documented-accepted at its flag (one comment: the button is disabled while exporting; a racing second invoke reconverges on the first run's events).

- [ ] **Step 1: Implement**

Read all three flag lifecycles first. The job-worker handshake is the delicate one — write it as a labeled loop with the clear/recheck/re-CAS at the bottom, commented step by step. For scan_roll, the re-arm wraps the existing two-pass body in a bounded loop with per-iteration generation/indices re-resolution.

Gates: full Rust suite + clippy + fmt; from app/ `npm run check` (no frontend change expected).

- [ ] **Step 2: Commit**

```bash
git add app/src-tauri
git commit -m "Close the queue lost-wakeup windows with a clear-and-recheck handshake"
```

---

### Task 7: Sweep and manual gate

- [ ] **Step 1: Automated sweep**

Root: `cargo test`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all --check`. app/: `npm run test && npm run check`.

- [ ] **Step 2: Manual gate (human) — short, mostly regression**

`cd app && mise exec -- npm run tauri dev`:

1. Normal loop regression: open the roll, everything restores instantly, heal a frame, export — nothing feels different (hardening must be invisible).
2. Cache identity: `touch` one roll image file in Terminal (updates mtime), reopen the roll — that frame re-detects/re-heals instead of restoring (stamp mismatch), the others restore as before.
3. Roll-swap storm: queue several heals on roll A, immediately open roll B and queue heals there — B's jobs run, B's markers are truthful, A's stragglers never touch B (generation gating).
4. Rapid swap scan: open roll A, instantly open roll B — B's scan runs (the re-arm fix; previously B could silently never scan).
5. Hand-edit a sidecar stroke to garbage, reopen: the frame loads with strokes dropped (debug log), brushing works normally.

- [ ] **Step 3: Close out**

Ledger; `bd close` gz6, mrr, sbo, uke, k2s, 2mn.

---

## Definition of done

- All six beads closed with their scenarios tested or gate-verified; no user-visible behavior change outside the fixed races.
- Cache self-migrates (old formats purge, everything recomputes once).
- No new spinning, no new races: every handshake loses gracefully.
- NOT here: the UI/UX bead set (next plan), queue unification, worker-process isolation (obr).
