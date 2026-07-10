# u8 Probs Retention and Detect Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain detection probabilities as u8 instead of f32 in the image registry (bead TheUnduster-1cy, ~650MB saved per detected 168MP frame) and give detection per-tile progress events (bead TheUnduster-2vf, replacing the ~9s indeterminate stage).

**Architecture:** The registry's `probs: Option<(Vec<f32>, ProbPyramid)>` becomes u8-quantized (`(p * 255).round() as u8`), matching what the disk codec already stores — restores skip the dequantize entirely. Threshold comparisons quantize the threshold once per call and compare in u8 space. Detection in fd-infer gains a per-tile progress callback mirroring fd-heal's `heal_with_progress`, threaded to a `detect-progress` event consumed like `heal-progress` (status bar text + queue panel bar).

**Tech Stack:** Rust (engine/crates/fd-infer, app/src-tauri), Svelte 5 frontend (two small wiring points).

## Global Constraints

- Trunk-based: atomic commits to main, tests green each commit, plain-English why-focused messages, no Co-Authored-By, no emoji.
- Gates: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`; app/: `npm run test` (80 + any new) and `npm run check` (0 errors, exactly 4 baseline warnings: Viewer x3, Filmstrip x1).
- Quantization rule, one source of truth: `q = (p.clamp(0.0, 1.0) * 255.0).round() as u8`, threshold side `qt = (t.clamp(0.0, 1.0) * 255.0).round() as u8`, membership `q > qt` (preserving today's strict `p > threshold`). The disk probs codec already quantizes with this rule — verify in cache.rs and REUSE its function (make it `pub(crate)` if needed) rather than writing a second copy.
- Boundary honesty: membership within half a quantum (~0.002) of a threshold can differ from the f32 era. The heal cache is keyed by provenance (threshold value, not mask bytes), so pre-change cached heals stay valid and replayable; a fresh heal may differ from an old cached one by boundary pixels. This is accepted — record it in code comment form at the comparison site, and do NOT bump the heal codec version.
- Existing concurrency disciplines untouched (generation guards, single-flight worker, pin/eviction).

---

### Task 1: u8 probs in the registry (1cy)

**Files:**
- Modify: `app/src-tauri/src/images.rs` (Entry.probs type, set_probs, components, prob-tile path, retained_bytes)
- Modify: `app/src-tauri/src/cache.rs` (expose the quantize helper; the probs RESTORE path stops dequantizing to f32)
- Modify: `app/src-tauri/src/lib.rs` (detect paths that call set_probs with fresh f32 output quantize at the boundary)
- Modify: `engine/crates/fd-tiles` ONLY if ProbPyramid's builder takes f32 input and needs a u8 entry point — check `build_prob_pyramid`'s signature first; the pyramid levels are already u8-quantized, so the natural change is a u8-input builder variant (or quantize-before-build), NOT a pyramid format change.

**Interfaces:**
- `Entry.probs: Option<(Vec<u8>, ProbPyramid)>`.
- `Images::set_probs(id, probs: Vec<u8>, pyramid: ProbPyramid)` — callers with fresh detector f32 output quantize once at the call boundary using the shared rule.
- `Images::components(id, threshold: f32)` — signature UNCHANGED (frontend passes f32); internally quantizes the threshold once and compares u8. Same for any other threshold consumer found by `grep -n "probs" app/src-tauri/src/images.rs` (heal-mask composition reads probs via a raw bool mask — find it and convert the same way).
- `retained_bytes`: probs term becomes `probs.len()` (×1, not ×4).
- The disk restore path (probs cache → registry): reads u8, builds the pyramid, stores u8 directly — the dequantize-to-f32 step is deleted. The disk WRITE path for fresh detections quantizes once (it already does — verify it can now take the already-quantized buffer without a second pass).

- [ ] **Step 1 (TDD):** failing tests first — components membership parity: for a synthetic probs buffer containing values exactly at, just above, and just below a threshold quantum, `components(id, t)` under u8 equals a hand-computed expected set using the documented rule; retained_bytes reflects ×1; a round-trip through the disk codec and back into the registry is byte-identical (no quantize-dequantize-requantize drift). Run red.
- [ ] **Step 2:** implement, following every consumer found by grep — the compiler finds the rest once the Entry type changes.
- [ ] **Step 3:** gates; commit `"Retain detection probabilities as u8 in the registry"`.

---

### Task 2: Detect per-tile progress (2vf)

**Files:**
- Modify: `engine/crates/fd-infer/src/*` (detect gains a progress-callback variant)
- Modify: `app/src-tauri/src/detect.rs` (DetectorState::detect/detect_hashed thread the callback)
- Modify: `app/src-tauri/src/lib.rs` (run_detect single-image path + run_job Detect arm emit `detect-progress`)
- Modify: `app/src/App.svelte`, `app/src/lib/status.ts` + tests (consume it)

**Interfaces:**
- fd-infer: `pub fn detect_with_progress(..., progress: &mut dyn FnMut(usize, usize))` called once per completed tile with (done, total) — mirror `fd_heal::heal_with_progress`'s exact shape (read it first); the existing `detect(...)` delegates with a no-op callback so no other caller changes.
- Backend event `detect-progress { id: u64, done: usize, total: usize }` for the single-image path (mirroring `heal-progress`'s id-keyed payload, lib.rs:~206/324) and the same event from the roll job arm (the worker is single-flight, so frontend queue attribution needs no index — identical to heal-progress's handling).
- Frontend: a `detectProgress` state alongside `healProgress`, fed by a listener with the same displayed-frame id guard + unconditional queueProgress attribution (copy the heal-progress listener's exact shape, App.svelte:~147-160). Status: `composeActivity`'s detecting branch gains optional detail `detecting (n/m tiles)` — extend `ActivityInput` minimally, failing-first tests for the new copy and for priority unchanged. Queue panel: the running detect row now shows the bar via the existing queueProgress plumbing — zero new wiring beyond the listener feeding it.
- Emission throttling: ~870 tiles over ~9s ≈ 100/s — emit every tile (IPC is cheap and the heal path already emits per defect); if the reviewer disagrees the fix is a modulo, not a redesign.

- [ ] **Step 1 (TDD):** status.test.ts failing-first for `detecting (12/870 tiles)` copy + priority; fd-infer unit test that the callback fires exactly total times with monotonic done (the crate has tile-loop tests to extend — find them).
- [ ] **Step 2:** implement engine → command → events → frontend.
- [ ] **Step 3:** gates both stacks; commit `"Narrate detection tile by tile"`.

---

### Task 3: Sweep and manual gate

- [ ] **Step 1:** full sweep (both stacks, all gates).
- [ ] **Step 2 (human):** detect a large frame — the status bar reads `detecting (n/m tiles)` counting up and the queue panel's running row shows a moving bar; slider behavior and defect counts feel identical to before (boundary quanta aside); a big roll (10+ detected frames) stays responsive where it previously churned the memory budget.
- [ ] **Step 3:** ledger; `bd close 1cy 2vf`.

---

## Definition of done

- Registry probs are u8 end-to-end (fresh detect, disk restore, components, heal-mask compose, prob tiles) with one shared quantization rule and documented boundary semantics; retained_bytes drops ~650MB per detected 168MP frame.
- Detection narrates per tile in the status bar and queue panel through the same plumbing heal uses.
- NOT here: detect speed work (post-real-model, 3uz), heal batching (gmn, separate plan), slider step changes.
