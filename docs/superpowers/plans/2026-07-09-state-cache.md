# Per-Frame State Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop redoing expensive work: persist detection probability maps and heal results per roll frame so relaunch, eviction, threshold play, and export never re-run a model unnecessarily.

**Architecture:** Two cache artifacts per frame under `<roll_dir>/.unduster/cache/`, keyed by file name like thumbnails. Probability maps are u8-quantized and zstd-compressed with the detector model's SHA-256 in the header — threshold-independent, so they never go stale except on a model change. Heals are stored as deltas: the composed mask plus the healed pixel values inside it (bit-exactness means nothing else changed), a few MB per frame, reconstructed losslessly as original + patch; a provenance hash over (threshold, dilate radius, strokes, detector hash, inpainter hash) makes staleness structural — a cache entry either exactly matches the requested heal or is ignored. The scan queue persists probs it already computes (today it throws them away), so a scanned frame's first heal needs no re-detect. The export queue reconstructs cached heals instead of re-running models. Single-image mode stays session-local (no `.unduster` pollution outside rolls).

**Tech Stack:** Existing stack plus `zstd` in the app crate (`sha2` is already a dependency; hashes stay raw `[u8; 32]` throughout). No engine-crate changes.

## Global Constraints

- Trunk-based: commit directly to `main`, atomic commits, tests green first. No emoji; no Co-Authored-By.
- Cache files are a boundary (user-writable directory): every read validates magic, version, dimensions, and payload lengths; anything malformed is ignored and deleted, never trusted, never a panic. Corrupt cache can cost recompute time, never correctness.
- Heal reconstruction must be bit-exact: reconstruct(original, delta) equals the healed image byte for byte (round-trip tested for U8 and U16). Export verification (`export_healed`) continues to enforce the outside-mask guarantee downstream regardless.
- Provenance: a cached heal is used ONLY when its hash matches sha256(threshold_bits || dilate_radius || canonical strokes || detector_hash || inpainter_hash). No partial matches, no "close enough".
- Probs quantization is u8 (`round(p * 255)`), dequantized as `q / 255.0`; the slider granularity is 0.01 so 1/255 resolution is comfortably below it.
- Cache writes are atomic (tmp sibling + rename, the established discipline) and happen off the UI path (inside existing spawn_blocking work or fire-and-forget blocking tasks); a failed cache write logs and continues — caching is an optimization, never a failure source.
- Roll mode only; cache directory is `roll::sidecar-dir/cache` (i.e. `.unduster/cache/`), wiped along with the sidecar.
- Rust fmt + clippy `-D warnings` clean; vitest/svelte-check untouched-green (no frontend changes in this plan).

---

### Task 1: Probs cache codec

**Files:**
- Create: `app/src-tauri/src/cache.rs`
- Modify: `app/src-tauri/src/lib.rs` (add `mod cache;`)
- Modify: `app/src-tauri/Cargo.toml` (add `zstd = "0.13"`)
- Test: inline in `cache.rs`

**Interfaces:**
- Produces:

```rust
pub const PROBS_MAGIC: &[u8; 8] = b"UNDPROB1";

/// Writes width*height probabilities as u8 (round(p*255)), zstd-compressed,
/// with the producing detector's file hash in the header. Atomic.
pub fn write_probs(path: &Path, probs: &[f32], width: u32, height: u32, detector_hash: &[u8; 32]) -> Result<(), String>;

/// Reads a probs cache written by write_probs. Returns None (never Err) when
/// the file is absent, malformed, dimension-mismatched, or produced by a
/// different detector -- malformed files are deleted on sight.
pub fn read_probs(path: &Path, width: u32, height: u32, detector_hash: &[u8; 32]) -> Option<Vec<f32>>;
```

File layout: magic(8) | version u32 LE (=1) | width u32 | height u32 | detector_hash(32) | compressed_len u64 | zstd frame of `width*height` u8 bytes.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn synth_probs(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i % 97) as f32) / 96.0).collect()
    }

    #[test]
    fn probs_round_trip_within_quantization() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.probs");
        let probs = synth_probs(64 * 48);
        let hash = [7u8; 32];
        write_probs(&p, &probs, 64, 48, &hash).unwrap();
        let back = read_probs(&p, 64, 48, &hash).expect("cache readable");
        assert_eq!(back.len(), probs.len());
        for (a, b) in probs.iter().zip(&back) {
            assert!((a - b).abs() <= 0.5 / 255.0 + 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn probs_reject_wrong_dims_hash_and_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.probs");
        let probs = synth_probs(16 * 16);
        write_probs(&p, &probs, 16, 16, &[1u8; 32]).unwrap();
        assert!(read_probs(&p, 16, 17, &[1u8; 32]).is_none()); // dims
        assert!(p.exists(), "dim mismatch is not corruption; file kept");
        assert!(read_probs(&p, 16, 16, &[2u8; 32]).is_none()); // detector changed
        assert!(p.exists(), "hash mismatch is not corruption; file kept");
        let mut bytes = std::fs::read(&p).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&p, &bytes).unwrap();
        assert!(read_probs(&p, 16, 16, &[1u8; 32]).is_none()); // corrupt payload
        assert!(!p.exists(), "corrupt file deleted on sight");
        assert!(read_probs(&p, 16, 16, &[1u8; 32]).is_none()); // absent -> None
    }

    #[test]
    fn probs_write_is_atomic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.probs");
        write_probs(&p, &synth_probs(8 * 8), 8, 8, &[0u8; 32]).unwrap();
        // no tmp siblings left behind
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter(|e| e.as_ref().unwrap().file_name() != "a.probs")
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }
}
```

Run: `cargo test -p unduster-app cache` — FAIL (module missing).

- [ ] **Step 2: Implement**

Straightforward serialization per the layout above. Corruption handling: any header mismatch on magic/version/length arithmetic, or a zstd decode error, or a decompressed length that is not `width*height` → `let _ = std::fs::remove_file(path); return None;`. A dims or detector-hash mismatch with an otherwise well-formed header is NOT corruption — return None and keep the file (a different roll state may still want it... it will simply be overwritten on the next write; keeping it is the least surprising behavior). Write path: serialize into memory, write to `path` with extension `".tmp-unduster"` appended, `fs::rename`. zstd level: `zstd::encode_all(reader, 3)`.

- [ ] **Step 3: Run tests, lint, commit**

Run: `cargo test -p unduster-app && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`

```bash
git add app/src-tauri
git commit -m "Add the probability-map cache codec"
```

---

### Task 2: Heal delta codec

**Files:**
- Modify: `app/src-tauri/src/cache.rs`
- Test: inline

**Interfaces:**
- Produces:

```rust
pub const HEAL_MAGIC: &[u8; 8] = b"UNDHEAL1";

/// Provenance of a heal: any input that changes the output contributes.
/// Strokes are canonicalized as serde_json bytes (deterministic for
/// identical f32 bit patterns, which is exactly the invariant we want).
pub fn heal_provenance(
    threshold: f32,
    dilate_radius: u32,
    strokes: &[crate::masks::Stroke],
    detector_hash: &[u8; 32],
    inpainter_hash: &[u8; 32],
) -> [u8; 32];

/// Persists a heal as a delta: the mask (bitset, zstd) and the healed pixel
/// values inside it (native depth, zstd), plus the provenance. Atomic.
pub fn write_heal(
    path: &Path,
    original: &fd_io::ImageBuf,
    healed: &fd_io::ImageBuf,
    mask: &[bool],
    provenance: &[u8; 32],
) -> Result<(), String>;

/// Reconstructs the healed image (original + patch) IF the cache entry
/// matches the requested provenance and the original's dimensions/depth.
/// Returns the healed copy and the mask. None on any mismatch; malformed
/// files deleted on sight.
pub fn read_heal(
    path: &Path,
    original: &fd_io::ImageBuf,
    provenance: &[u8; 32],
) -> Option<(fd_io::ImageBuf, Vec<bool>)>;
```

File layout: magic(8) | version u32 (=1) | width u32 | height u32 | channels u8 | depth u8 (8 or 16) | provenance(32) | mask_comp_len u64 | zstd(bitset, (w*h+7)/8 bytes, row-major LSB-first) | values_comp_len u64 | zstd(values: for each mask pixel in row-major order, channels values at native depth, LE).

- [ ] **Step 1: Write the failing tests**

```rust
    fn noisy16(w: u32, h: u32) -> fd_io::ImageBuf {
        let n = (w * h * 3) as usize;
        let mut s = 0x2545F4914F6CDD1Du64;
        let data: Vec<u16> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 48) as u16
            })
            .collect();
        fd_io::ImageBuf {
            width: w,
            height: h,
            channels: 3,
            data: fd_io::PixelData::U16(data),
            icc: Some(vec![1, 2, 3]),
            exif: None,
        }
    }

    #[test]
    fn heal_delta_round_trips_bit_exact() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.heal");
        let original = noisy16(40, 30);
        let mut healed = original.clone();
        let mut mask = vec![false; 40 * 30];
        for y in 5..12 {
            for x in 20..35 {
                mask[y * 40 + x] = true;
            }
        }
        if let fd_io::PixelData::U16(v) = &mut healed.data {
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    for c in 0..3 {
                        v[i * 3 + c] = v[i * 3 + c].wrapping_add(1000 + c as u16);
                    }
                }
            }
        }
        let prov = heal_provenance(0.5, 2, &[], &[3u8; 32], &[4u8; 32]);
        write_heal(&p, &original, &healed, &mask, &prov).unwrap();
        let (back, back_mask) = read_heal(&p, &original, &prov).expect("cache hit");
        assert_eq!(back_mask, mask);
        let (fd_io::PixelData::U16(a), fd_io::PixelData::U16(b)) = (&healed.data, &back.data)
        else {
            panic!("depth changed");
        };
        assert_eq!(a, b, "reconstruction must be bit-exact");
        assert_eq!(back.icc, original.icc, "metadata rides the original");
    }

    #[test]
    fn heal_delta_rejects_provenance_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.heal");
        let original = noisy16(16, 16);
        let healed = original.clone();
        let mask = vec![false; 256];
        let prov = heal_provenance(0.5, 2, &[], &[3u8; 32], &[4u8; 32]);
        write_heal(&p, &original, &healed, &mask, &prov).unwrap();
        let other = heal_provenance(0.51, 2, &[], &[3u8; 32], &[4u8; 32]);
        assert!(read_heal(&p, &original, &other).is_none());
        assert!(p.exists(), "provenance miss keeps the file");
    }

    #[test]
    fn heal_provenance_distinguishes_every_input() {
        let base = heal_provenance(0.5, 2, &[], &[0u8; 32], &[0u8; 32]);
        let stroke = crate::masks::Stroke {
            erase: false,
            radius: 5.0,
            points: vec![[1.0, 2.0]],
        };
        assert_ne!(base, heal_provenance(0.6, 2, &[], &[0u8; 32], &[0u8; 32]));
        assert_ne!(base, heal_provenance(0.5, 3, &[], &[0u8; 32], &[0u8; 32]));
        assert_ne!(
            base,
            heal_provenance(0.5, 2, std::slice::from_ref(&stroke), &[0u8; 32], &[0u8; 32])
        );
        assert_ne!(base, heal_provenance(0.5, 2, &[], &[1u8; 32], &[0u8; 32]));
        assert_ne!(base, heal_provenance(0.5, 2, &[], &[0u8; 32], &[1u8; 32]));
    }
```

Run: `cargo test -p unduster-app heal_delta` — FAIL.

- [ ] **Step 2: Implement**

`heal_provenance`: `sha2::Sha256` over `threshold.to_le_bytes()`, `dilate_radius.to_le_bytes()`, `serde_json::to_vec(strokes).unwrap_or_default()`, both hashes, in that order. `write_heal`: validate healed dims/channels/depth equal original's (Err otherwise); collect masked values in row-major order from healed's native data; bitset pack; zstd both; atomic write. `read_heal`: header validation (corruption → delete + None; provenance or dims/depth mismatch with valid header → keep + None); decompress; verify decompressed lengths exactly match the header's implied counts (bitset length, `popcount * channels * depth_bytes`); clone original, scatter values into mask positions, return.

- [ ] **Step 3: Run tests, lint, commit**

Run: `cargo test -p unduster-app && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`

```bash
git add app/src-tauri
git commit -m "Add the heal delta cache codec"
```

---

### Task 3: Model identity hashes

**Files:**
- Modify: `app/src-tauri/src/detect.rs` (DetectorState + InpainterState carry the loaded file's hash)
- Modify: `app/src-tauri/src/models.rs` (reuse its streaming sha256 helper)
- Test: inline in `detect.rs`

**Interfaces:**
- Produces: `DetectorState::hash(&self) -> Option<[u8; 32]>` and `InpainterState::hash(&self) -> Option<[u8; 32]>` — the SHA-256 of the currently loaded model file, `None` when nothing is loaded. Computed once inside `load` (streaming, before or after session build — the file is ≤ ~210MB, a one-time cost of well under a second). The state's inner type grows from `Option<Model>` to `Option<(Model, [u8; 32])>` (or a small struct). `models::verify_sha256` refactors to expose `pub fn file_sha256(path: &Path) -> Result<[u8; 32], String>` that both it and the loads use. An inpainter hash of `None` (classical-only healing) contributes a fixed all-zeros hash to provenance — classical output is deterministic, and the zero sentinel distinguishes it from any real model.

- [ ] **Step 1: Failing tests**

```rust
    #[test]
    fn loaded_states_expose_their_file_hash() {
        let state = DetectorState::default();
        assert!(state.hash().is_none());
        state.load(&fixture()).unwrap();
        let h = state.hash().expect("hash after load");
        // stable across loads of the same file
        state.load(&fixture()).unwrap();
        assert_eq!(state.hash().unwrap(), h);
    }
```

(and the mirror test for `InpainterState` with the tiny-inpaint fixture.)

Run: `cargo test -p unduster-app hash` — FAIL.

- [ ] **Step 2: Implement**

Refactor `verify_sha256(path, expected)` to call the new `file_sha256(path)` and compare. In both `load` methods: `let hash = crate::models::file_sha256(path)?;` then store alongside the model. Update `with_inpainter`/`detect` internals for the tuple/struct — their public signatures do not change. `hash()` locks and copies.

- [ ] **Step 3: Run tests, lint, commit**

Run: `cargo test -p unduster-app && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`

```bash
git add app/src-tauri
git commit -m "Expose the loaded model file hash on detector and inpainter state"
```

---

### Task 4: Cache paths and roll lookup

**Files:**
- Modify: `app/src-tauri/src/roll.rs` (cache dir + id-to-frame lookup)
- Test: inline in `roll.rs`

**Interfaces:**
- Produces: `pub fn cache_dir(dir: &Path) -> PathBuf` (`sidecar_dir(dir).join("cache")`, mirrors `thumbs_dir`); `pub fn probs_cache_path(dir: &Path, file_name: &str) -> PathBuf` (`cache/{file_name}.probs`) and `pub fn heal_cache_path(dir: &Path, file_name: &str) -> PathBuf` (`cache/{file_name}.heal`); `RollState::frame_for_image(&self, id: u64) -> Result<Option<(PathBuf, String)>, String>` returning the roll dir and file name of the frame whose `image_id` is `id` (None when no frame maps to it — single-image mode, or the frame was evicted). This is how id-keyed commands (detect, heal) find their cache files.

- [ ] **Step 1: Failing test**

```rust
    #[test]
    fn frame_for_image_maps_ids_to_files() {
        let dir = tempfile::tempdir().unwrap();
        for n in ["a.png", "b.png"] {
            std::fs::write(dir.path().join(n), b"x").unwrap();
        }
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        state.set_image_id(1, 42).unwrap();
        let (d, name) = state.frame_for_image(42).unwrap().expect("mapped");
        assert_eq!(d, dir.path());
        assert_eq!(name, "b.png");
        assert!(state.frame_for_image(7).unwrap().is_none());
    }
```

Run: `cargo test -p unduster-app frame_for_image` — FAIL.

- [ ] **Step 2: Implement**

`frame_for_image` mirrors the other RollState accessors (lock, no-roll → Ok(None) rather than Err — a closed roll is a normal cache-miss condition here, deviating deliberately from accessors where "no roll" is an error; document that in a doc comment). Path helpers are two-liners beside `thumb_path`.

- [ ] **Step 3: Run tests, lint, commit**

```bash
git add app/src-tauri
git commit -m "Add cache paths and image-id-to-frame lookup"
```

---

### Task 5: Wire the probs cache

**Files:**
- Modify: `app/src-tauri/src/lib.rs` (`run_detect` writes; scan_roll queue writes; `activate_frame` reads)

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `run_detect`: after `set_probs_built` succeeds, if `roll.frame_for_image(id)` maps and `detector.hash()` is Some, write the probs cache (inside a `spawn_blocking`, fire-and-forget task; failures `eprintln!` in debug and are otherwise silent).
  - scan_roll stage 2: after `record_scan_result` succeeds for a frame, write the probs cache from the same `probs` vector (the closure already owns it — write BEFORE dropping it, inside the same blocking stage, sequential with detection; a cache write is milliseconds against a 9s detect).
  - `activate_frame`: on the decode path (fresh entry, no probs), after `set_image_id`, attempt `read_probs` (dims from the entry, current detector hash); on a hit, build the prob pyramid and `set_probs_built`, so the frame arrives detection-ready. On the reuse path do nothing (registry already has whatever it has).
  - The frontend learns a cached detection exists exactly the way it learns queue results exist today: `defect_count`/bboxes from the sidecar drive the UI, and `threshold_mask`/`components` now find probs present. No frontend changes.

- [ ] **Step 1: Implement**

Follow the interface block precisely; read `run_detect`, the scan_roll stage-2 closure, and `activate_frame`'s decode path before editing. The scan queue's write: `roll::probs_cache_path(&roll_dir, &file_name)` is already constructible there (`roll_dir`, `file_name` in scope). `activate_frame`'s read needs the detector hash — `detector: State<detect::DetectorState>` is not currently a parameter of `activate_frame`; add it (command params are injection, no frontend change needed).

There is no automated harness for these command flows (established); the covering tests are the codec tests plus the manual gate. Full gates: `cargo test -p unduster-app && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`; from app/: `npm run check` (invoke signatures unchanged).

- [ ] **Step 2: Commit**

```bash
git add app/src-tauri
git commit -m "Persist and restore probability maps per roll frame"
```

---

### Task 6: Wire the heal cache

**Files:**
- Modify: `app/src-tauri/src/lib.rs` (`run_heal` reads/writes; export queue reads/writes)

**Interfaces:**
- Consumes: codecs, hashes, paths.
- Produces:
  - `run_heal`: compute `provenance = cache::heal_provenance(threshold, HEAL_DILATE_RADIUS, &strokes, &detector_hash_or_zeros, &inpainter_hash_or_zeros)` up front (detector hash: the CURRENT detector's hash — the probs in the registry came from it; use zeros when none loaded). Before the model work, if the frame maps via `frame_for_image` and `read_heal` hits: skip compose/heal entirely, use the reconstructed healed image + mask (still build the pyramid, still `set_healed`), and report a `HealSummary` with `defects`/`tiny`/`inpainted` all zero but a new field `restored: bool` set true (frontend may ignore it; the summary is informational). On a miss: heal as today, then write the cache (fire-and-forget blocking task with the healed Arc, original Arc, mask Arc — cheap clones).
  - Export queue registry-miss path: BEFORE the transient pipeline, try the heal cache: decode the original (`decode_stage`), compute provenance from the frame's threshold + strokes + current hashes, `read_heal`; on a hit, `export_healed(original, reconstructed, mask, dest)` directly (stage event "writing"). On a miss, run the transient pipeline as today and afterwards write the heal cache (so the NEXT export or heal of this frame is instant).
  - `HealSummary` gains `restored: bool` (serde; frontend tolerates unknown fields — no TS change required, though adding it to the interface is fine).

- [ ] **Step 1: Implement**

Read `run_heal` and the export queue fully first. Ordering note for run_heal's cache read: it needs the original image Arc (already fetched) and the mapped path — do the read inside the existing `spawn_blocking` (it is file IO + memcpy at worst). The reconstructed path MUST still go through `set_healed` (dims validation) and pyramid build. The cache write on miss must not extend the lock or the user-visible heal time materially: spawn it after `set_healed` succeeds.

Full gates as in Task 5.

- [ ] **Step 2: Commit**

```bash
git add app/src-tauri
git commit -m "Persist and restore heals as provenance-checked deltas"
```

---

### Task 7: Sweep and manual gate

- [ ] **Step 1: Automated sweep**

Root: `cargo test`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all --check`. app/: `npm run test && npm run check`.

- [ ] **Step 2: Manual gate (human)**

`cd app && mise exec -- npm run tauri dev`, open the lmca roll fresh (wipe `.unduster` first for a clean run):

1. Let the scan finish, then QUIT and relaunch, reopen the roll: activating any scanned frame should be detection-ready immediately (slider live, z cycles, no "detecting" wait) — probs restored from cache.
2. Heal a frame (watch the per-defect count), then relaunch, reopen, press `h` on the same frame at the SAME threshold: near-instant, before/after works — heal restored from cache.
3. Move the threshold slider, heal again: it does the full work (provenance changed), and the new heal replaces the cached one.
4. Approve a frame healed in a PREVIOUS session (registry empty for it), export the roll: that frame exports in seconds ("writing" stage, no "healing" stage).
5. Check `.unduster/cache/` file sizes: probs tens of MB, heals a few MB per frame.
6. Delete `.unduster/`, reopen: everything regenerates from scratch (the wipe story still holds).

- [ ] **Step 3: Close out**

Ledger; `bd close TheUnduster-8di`; update bead 1cy (narrow it to in-memory quantization only); note on e15 that export-path staleness is now structurally solved and only the viewer-display staleness remains.

---

## Definition of done

- Probs and heals survive relaunch and eviction; nothing re-runs a model when a provenance-matching cache entry exists; export of previously-healed frames is model-free.
- All cache reads are boundary-validated; corruption self-heals by deletion and recompute; writes are atomic.
- Heal reconstruction is bit-exact (tested), and export verification still independently enforces the guarantee.
- NOT here: single-image mode caching, in-memory u8 probs (bead 1cy), pyramid persistence (bead 3e5), disk-budget eviction of the cache (documented as future work; expected sizes are modest).
