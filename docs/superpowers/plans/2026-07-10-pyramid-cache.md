# Display Pyramid Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the RGBA display pyramid to the roll's `.unduster/cache/` so revisiting an evicted frame skips the pyramid build (bead TheUnduster-3e5, scope locked by Albert: pyramid cache only; the native decode stays on the activation path).

**Architecture:** A new house-style codec in cache.rs (magic/version/stamp header, zstd per level, corrupt-delete/mismatch-keep discipline) stores `fd_tiles::Pyramid` at `.unduster/cache/<filename>.pyr`. `decode_and_insert` tries the cache before building; a fresh build writes back fire-and-forget off the activation path. A per-roll LRU prune (file mtime, 20GB default, `UNDUSTER_PYRAMID_BUDGET_GB` override) runs after each write. Both activation and prefetch get the win through the shared path.

**Tech Stack:** Rust (app/src-tauri), zstd 0.13 (already a dep), fd-tiles `Pyramid`/`Level`.

## Global Constraints

- Trunk-based: atomic commits to main, tests green each commit, plain-English why-focused messages, no Co-Authored-By, no emoji.
- Gates per commit: `cargo test -p unduster-app && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`. Frontend untouched (no app/src changes expected; if any leak in, `npm run test` 80 + `npm run check` 0 errors/4-warning baseline).
- House codec discipline (mirror probs/heal exactly): 8-byte magic, u32 LE version, stamp-before-decode fail-safe direction, `zstd::bulk::decompress(data, expected_size)` bounded allocation, checked arithmetic on all length fields (corrupt() on overflow), corrupt-delete (bad magic/version/structure/decompression) vs mismatch-keep (well-formed but wrong stamp/dims), atomic write via temp-file + rename.
- The activation path's latency rules: the cache WRITE and the prune must never block or delay activation (fire-and-forget spawn_blocking, the probs-cache-write pattern); the cache READ replaces the pyramid build only when strictly faster paths agree (read happens where the build happened, inside the existing spawn_blocking stage).
- Existing concurrency disciplines untouched: generation guards, tie-loss registration, pin/eviction. The cache is keyed by source file, not roll generation — a roll swap cannot corrupt it.

---

### Task 1: The pyramid codec

**Files:**
- Modify: `app/src-tauri/src/cache.rs`
- Modify: `app/src-tauri/src/roll.rs` (path helper)

**Interfaces:**
- Consumes: `fd_tiles::{Pyramid, Level}` (Level { width: u32, height: u32, rgba: Vec<u8> }), existing `SourceStamp`, the corrupt()/atomic-write helpers in cache.rs (read the file and reuse its private helpers — do not duplicate).
- Produces:

```rust
pub const PYRAMID_MAGIC: &[u8; 8] = b"UNDPYRA1";
pub const PYRAMID_VERSION: u32 = 1;

/// Writes the display pyramid: header
/// magic(8) | version(4) | level_count(4) | size(8) | mtime_nanos(8)
/// then per level: width(4) | height(4) | comp_len(8) | zstd payload.
/// zstd level 1, not the probs codec's 3: RGBA film grain barely
/// compresses at any level, and this write runs once per fresh build --
/// encode speed is worth more than a few percent of disk.
pub fn write_pyramid(path: &Path, pyramid: &Pyramid, stamp: &SourceStamp) -> Result<(), String>;

/// Reads a cached pyramid. None on: missing file, stamp mismatch
/// (source changed -- file kept, house mismatch-keep rule), or corrupt
/// structure (file deleted on sight). Every level's rgba length is
/// validated as width*height*4 with checked arithmetic, and each
/// decompression is bounded to that expected size.
pub fn read_pyramid(path: &Path, stamp: &SourceStamp) -> Option<Pyramid>;
```

- roll.rs: `pub fn pyramid_cache_path(dir: &Path, file_name: &str) -> PathBuf` returning `.unduster/cache/<file_name>.pyr`, next to `probs_cache_path`/`heal_cache_path` and following their doc style.

- [ ] **Step 1: Failing tests** (cache.rs tests module, mirroring the probs tests' structure and helpers)

```rust
#[test]
fn pyramid_round_trips() {
    // Two levels with distinct contents; assert every field survives.
}

#[test]
fn pyramid_stamp_mismatch_returns_none_and_keeps_file() {
    // Write with stamp A, read with stamp B (size+1): None, file exists.
}

#[test]
fn corrupt_pyramid_is_deleted_on_sight() {
    // Patch the magic to b"UNDPYRA0": read returns None, file gone.
}

#[test]
fn truncated_pyramid_is_deleted() {
    // Truncate mid-level: None, file gone.
}

#[test]
fn pyramid_level_length_lie_is_rejected() {
    // Hand-craft a header whose width*height*4 disagrees with the
    // decompressed payload length: None, file deleted (corrupt class).
}
```

Write each test with real assertions against a small synthetic pyramid (e.g. level0 4x4, level1 2x2, recognizable byte patterns). Run: `cargo test -p unduster-app pyramid` — expect compile failures, then red.

- [ ] **Step 2: Implement**

Follow the heal codec's write/read as the template (multi-section payload). Details that are binding:
- Atomic write: same temp+rename helper the other codecs use.
- Read order: open, stamp check FIRST (header stamp vs caller stamp — mismatch → None, keep), then structure (level_count sanity cap: reject > 32 levels as corrupt), then per-level bounded decompress with `width.checked_mul(height).and_then(|p| p.checked_mul(4))` — any overflow or disagreement is the corrupt class (delete).
- level_count == 0 is corrupt (a pyramid always has level 0).

- [ ] **Step 3: Green + gates + commit**

```bash
git add app/src-tauri/src
git commit -m "Add a disk codec for display pyramids"
```

---

### Task 2: Wire the cache into activation and prefetch

**Files:**
- Modify: `app/src-tauri/src/lib.rs` (`decode_and_insert` and its stage helpers)

**Interfaces:**
- Consumes: Task 1's codec, `stamp_or_skip`, `roll::pyramid_cache_path`, the existing `Images::decode_stage`/`pyramid_stage` split (lib.rs ~533-553), the probs-cache fire-and-forget write pattern.
- Produces: no API changes — `decode_and_insert`'s signature and both callers (activate_frame fresh path, prefetch arm) unchanged.

- [ ] **Step 1: Read path**

Inside `decode_and_insert`'s existing blocking stage, in this order:
1. `let stamp = stamp_or_skip(&path);` (existing helper; stat failure skips the cache interaction entirely — read AND write — never fails activation).
2. Native decode as today (`Images::decode_stage`) — unconditional, the entry needs it.
3. Pyramid: if `stamp` is Some and `read_pyramid(&cache_path, &stamp)` returns Some(p), VALIDATE it against the decoded image (`p.levels[0].width == image.width && p.levels[0].height == image.height` — a mismatch here despite a matching stamp means a corrupt-but-well-formed file: delete it and fall through to build); else build via `pyramid_stage` as today.
4. The "building-pyramid" progress emit (activation's `on_decoded` callback) fires as today regardless of path — on a hit the stage is just near-instant. Do not add a new event.

The cache path derivation mirrors the heal cache's: keyed by the frame's own directory (`path.parent()`) + file name. Single images get it too for free (their parent dir carries `.unduster/cache/` exactly as the probs/heal caches already do on the heal paths — verify by reading how run_heal derives it and match).

- [ ] **Step 2: Write path**

Only when the pyramid was freshly BUILT (not on a cache hit): after the registry insert returns, spawn a fire-and-forget `tauri::async_runtime::spawn_blocking` that clones what it needs cheaply — the pyramid lives in the registry now; clone the `Arc`? It is not Arc'd (Pyramid is owned by Entry). Options, pick the one matching the probs-write precedent after reading it: (a) write BEFORE the insert while the pyramid is still owned locally, accepting the write on the blocking stage — NO, that blocks activation; (b) serialize to bytes on the blocking stage cheaply? — encoding 336MB is the expensive part; (c) clone the pyramid for the background write (a ~336MB memcpy, tens of ms — acceptable on the blocking stage, document it) and move the clone into the spawn_blocking. Choose (c) unless the probs pattern shows a better established shape; state the choice in the report.
- Write failures: debug-eprintln, never surfaced (the heal-cache-write pattern).

- [ ] **Step 3: The prune**

After a successful write, in the same background task: scan the cache dir for `*.pyr`, sum sizes; while over budget, delete the oldest-mtime file (never the one just written). Budget: `UNDUSTER_PYRAMID_BUDGET_GB` env, default 20, parsed once with the same shape as `UNDUSTER_PIXEL_BUDGET_GB` (find and mirror it). Touch-on-read: after a successful `read_pyramid` hit, best-effort update the file's mtime (`filetime` crate? NO new deps — use `File::open` + `set_modified` via std (stable since 1.75; verify toolchain) or simply rewrite... simplest std-only: `std::fs::File::options().append(true).open(path)` then `file.set_modified(SystemTime::now())` — verify `set_modified` availability; if unavailable on the pinned toolchain, skip touch-on-read and prune by mtime-of-write only, documenting that LRU degrades to FIFO. State which landed in the report.)
- Prune failures: debug-eprintln, best effort.

- [ ] **Step 4: Tests**

The seams: roll.rs `pyramid_cache_path` unit test; a lib.rs-level test is impractical (needs AppHandle) — but the prune is pure enough to extract: `fn prune_pyramid_cache(dir: &Path, budget_bytes: u64, keep: &Path)` as a free function in cache.rs with unit tests (three files, budget forces one out, oldest goes, `keep` survives even if oldest). TDD failing-first for both.

- [ ] **Step 5: Gates + commit**

```bash
git add app/src-tauri/src
git commit -m "Serve display pyramids from the roll cache on revisit"
```

---

### Task 3: Sweep and manual gate

- [ ] **Step 1: Automated sweep**

Root: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`. app/ (should be untouched): `npm run test && npm run check`.

- [ ] **Step 2: Manual gate (human)**

`cd app && mise exec -- npm run tauri dev`:

1. Open a roll, visit a few frames, check `.unduster/cache/` grows `.pyr` files (one per visited frame, ~100-250MB each).
2. Navigate far enough to force eviction (or restart the app), revisit an early frame: activation should be noticeably faster than a cold visit — the "building-pyramid" stage flashes past; the decode stage remains.
3. Touch a source file (`touch scan.tif`) and revisit: the stale pyramid is ignored (stamp mismatch), a fresh one is built and written.
4. Set `UNDUSTER_PYRAMID_BUDGET_GB=1` and browse: old `.pyr` files get pruned, the newest survives.

- [ ] **Step 3: Close out**

Ledger; `bd close 3e5`.

---

## Definition of done

- Revisiting an evicted or cold frame skips the pyramid build when the source file is unchanged; the codec follows every house rule (stamp-first, bounded decompress, corrupt-delete/mismatch-keep, atomic write).
- Cache writes and pruning never delay activation; a 20GB per-roll budget holds by LRU (or documented FIFO fallback).
- NOT here: lazy native decode / display-first activation (a future plan if revisits still feel slow — the per-level file layout already supports partial reads), single-image-mode special cases beyond what the shared path gives for free, lz4.
