# Heal Window Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Batch nearby defects into shared inpaint windows (bead TheUnduster-gmn) so a dusty frame's clustered specks cost one LaMa forward pass per shared window instead of one per defect. At ~1.3s per 512px window on CPU (CoreML measured 3x slower — not an option), this is the only heal-speed lever.

**Architecture:** The fixed-window path already crops the FULL defect mask per window, so the model already inpaints every masked neighbor in a crop — the per-defect write-back guard is what discards that work (heal.rs:189-202). Batching adds a pure grouping pass (union-find over margin-expanded bbox proximity), tiles each group's union bbox exactly as today, and widens the write-back to all group members' pixels within each window interior. Singleton groups reproduce today's behavior bit for bit. The dynamic-contract path and the classical tier are untouched.

**Tech Stack:** Rust, engine/crates/fd-heal only (no app or frontend changes — the heal-progress payload keeps its done/total-defects meaning).

## Global Constraints

- Trunk-based: atomic commits to main, tests green each commit, plain-English why-focused messages, no Co-Authored-By, no emoji.
- Gates per commit: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`.
- The bit-exactness-outside-mask guarantee is structural (write_back writes only mask-true pixels, heal.rs:27-52) and every existing regression test pinning it stays green unmodified: `unmasked_pixels_are_bit_identical_after_heal`, `edge_hugging_inpaint_defect_does_not_panic`, `full_width_defect_does_not_panic`, `corner_defect_windows_safely`, `image_smaller_than_window_pads`, `defect_wider_than_a_window_is_fully_filled`.
- The Fixed(n) inpainter contract is inviolable: crops are exactly n×n (inpaint.rs:64-69); `window_start`'s shift-never-shrink rule and the interior/margin math (margin=n/8, interior=n−2·margin) are reused verbatim, not reimplemented.
- Progress semantics preserved for the frontend: callback still reports (defects_done, total_defects), monotonic — after a group's windows complete, done advances by the group's member count. The pinned test `heal_reports_per_defect_progress` updates only if its call-sequence expectation changes shape (3 isolated defects stay 3 groups → it should pass unmodified; verify rather than rewrite).
- HealReport counts (`inpainted`, `tiny`) keep their per-defect meaning.

---

### Task 1: Defect grouping (pure)

**Files:**
- Create: `engine/crates/fd-heal/src/group.rs`
- Modify: `engine/crates/fd-heal/src/lib.rs` (module + re-export)

**Interfaces:**
- Consumes: `components::{Defect, Bbox}` (Bbox x1/y1 exclusive; Defect { pixels, bbox }, max_dim()).
- Produces:

```rust
/// A batch of defect indices healed through shared windows. `bbox` is the
/// union of the members' bboxes (exclusive upper bounds, like Bbox).
pub struct Group {
    pub members: Vec<usize>, // indices into the caller's defect slice
    pub bbox: Bbox,
}

/// Groups defects whose margin-expanded bboxes touch or overlap, by
/// union-find with transitive merging. Two defects group when expanding
/// each bbox by `gap` in every direction (saturating at 0) makes them
/// intersect -- i.e. clustered specks whose per-defect windows would
/// largely overlap. `gap` is the window margin (n/8), so grouped members
/// sit close enough that a shared window's context still surrounds them.
/// Singleton results are the degenerate case and carry one member each.
/// Deterministic: members ascend within a group; groups order by their
/// first member.
pub fn group_defects(defects: &[Defect], gap: u32) -> Vec<Group>;
```

- [ ] **Step 1 (TDD, failing first):** tests in group.rs — two specks 3px apart with gap=8 merge into one group with the union bbox; the same specks with gap=1 stay separate; three specks where A touches B and B touches C but A doesn't touch C form ONE transitive group; a lone defect yields a singleton whose bbox equals its own; determinism (shuffled input produces the same grouping by member sets); expansion saturates at image origin (bbox at 0,0 with a large gap doesn't underflow).
- [ ] **Step 2:** implement (a simple O(k²) pairwise pass with union-find is fine — k is dozens, not thousands; note the bound in a comment).
- [ ] **Step 3:** gates; commit `"Group nearby defects for shared inpaint windows"`.

---

### Task 2: Batch the fixed-window heal path

**Files:**
- Modify: `engine/crates/fd-heal/src/heal.rs`
- Modify: `engine/crates/fd-heal/tests/heal_windowed.rs`, `tests/heal.rs` (only as specified)

**Interfaces:**
- Consumes: Task 1's `group_defects`; existing `inpaint_defect_windowed` internals (window_start, margin/interior, clamp_src crop assembly, write-back guard) — refactored, not duplicated.
- Produces: `heal_with_progress`'s Fixed(n) branch heals per GROUP: `inpaint_group_windowed(planes, mask, img_w, img_h, defects, &group, inpainter, n)` — same tiling loop as `inpaint_defect_windowed` but over `group.bbox`, and the write-back loop iterates every member's pixels (each still gated by the window-interior predicate and the lx/ly bounds guard). `inpaint_defect_windowed` becomes the singleton call through the same function (a Group of one), so there is ONE tiling implementation. Signature drift beyond this is the implementer's judgment, disclosed.

Flow changes in `heal_with_progress`:
1. Enumerate components as today; partition into tiny/inpaint tiers as today (tier check per defect, BEFORE grouping — tiny defects never join groups; they classical-fill exactly as now).
2. `let groups = group_defects(&inpaint_tier_defects, n / 8);` — gap = the window margin.
3. Per group: tile `group.bbox` with the existing interior-step loop; per window, crop planes + FULL-mask crop as today; one forward pass; write back every member pixel inside this window's interior. After the group's windows: `add_grain` per member (unchanged per-defect grain), `report.inpainted += members`, `progress(done_so_far, total_defects)` once per group with done advanced by member count.
4. Dynamic-contract path (`window_size() == None`) and the no-inpainter path stay per-defect, byte-identical.

- [ ] **Step 1 (TDD, failing first):** new test in heal_windowed.rs with a COUNTING inpainter fixture (wrap the existing 64px fixture — check how tests construct it — in a call-counting shim, or add an invocation counter the fixture exposes): five 8px specks clustered within one 48px interior region → exactly ONE inpainter call, all five defects' masked pixels healed (`unchanged_masked < threshold` per the existing style), every unmasked pixel bit-identical, `report.inpainted == 5`, progress ends at (5, 5). A second test: two clusters far apart → two calls. Run red (current code makes five/two-plus calls).
- [ ] **Step 2:** implement the refactor + batching. The existing window write-back predicate (`px in [ix,ix1) && py in [iy,iy1)` + `lx/ly < n`) applies per member unchanged — the exact-once property now holds per (member-pixel, window) exactly as it held per (defect-pixel, window).
- [ ] **Step 3:** all existing heal/windowed/components tests green UNMODIFIED (the constraints list); `heal_reports_per_defect_progress` verified (its 3 isolated defects form 3 singleton groups → same call sequence).
- [ ] **Step 4:** gates; commit `"Heal clustered defects through shared inpaint windows"`.

---

### Task 3: Sweep, measurement, and gate

- [ ] **Step 1:** full sweep (workspace gates; app untouched — verify with `git status`, then run app gates anyway as belt and braces: npm run test 80, npm run check 0 errors/4 warnings).
- [ ] **Step 2 (measurement, in-repo test not scratchpad):** a `#[test]` (or `--ignored` bench-style test if runtime demands) building a synthetic 2000×2000 mask with 40 specks in 8 clusters: assert the counting fixture records ≤ 12 calls where per-defect healing would make ≥ 40 — the batching win pinned as a regression test, model-free.
- [ ] **Step 3 (human gate):** heal a real dusty frame with the LaMa model and compare wall time + status narration against memory of the ~1.5s-per-defect era; spot-check healed quality on clustered specks (shared-window fills should look identical or better — more context per window).
- [ ] **Step 4:** ledger; `bd close gmn`.

---

## Definition of done

- Clustered defects share windows: forward-pass count drops proportionally to clustering (pinned by the counting test); isolated defects behave byte-identically to today; every bit-exactness and OOB regression test passes unmodified; progress and HealReport keep per-defect semantics; no app/frontend changes.
- NOT here: dynamic-contract batching, classical-tier changes, parallel window execution across cores (a future lever if batching isn't enough), detect-side anything.
