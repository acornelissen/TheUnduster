# App 3c: Heal and Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Heal a frame's detected defects into a separate healed copy (original untouched), review it with an instant before/after toggle, and export healed files with the untouched-pixel guarantee verified at write time — per frame and for a whole approved roll.

**Architecture:** Healing operates on a deep copy of the decoded image; the original `Arc<ImageBuf>` is never mutated, making the bit-exactness promise structural. The mask is the live threshold over the stored probabilities, dilated 2px (field observation: model masks cover a defect's confident core, and healing an under-covering mask leaves a rim). Healed pixels get their own display pyramid served under a `/healed/` layer on the tiles protocol, so before/after is a tile-path prefix swap — both textures stay resident, the toggle is instant. Export re-verifies every pixel outside the mask is integer-identical to the original, writes to a temp file, and renames atomically; a roll-level export queue reuses the scan queue's discipline (single-flight flag with drop guard, generation checks, one transient frame of memory).

**Tech Stack:** Existing stack. Engine changes: `fd_heal::dilate`, a `layer` discriminant on `fd_tiles::TileKey`. Inpaint model: the committed `engine/fixtures/tiny-inpaint.onnx` for dev/tests (mean-fill); a real LaMa-class model is a separate tracked issue.

## Global Constraints

- Trunk-based: commit directly to `main`, atomic commits, tests green first. No emoji; no Co-Authored-By.
- Originals are NEVER modified: healing clones the pixels; export reads the original only for verification.
- Export is atomic (tmp write, then rename) and MUST verify all outside-mask pixels are byte-identical before the rename; a verification failure aborts the export with the first offending coordinate named.
- Pixels cross to the webview only via the `tiles://` protocol; the healed layer follows the existing rgba response shape (octet-stream + x-tile-width/x-tile-height + CORS expose).
- Heavy work in `spawn_blocking` with staged `app-progress` events and a GUARANTEED terminal "ready" emit on every exit path (the `run_detect` pattern).
- Healed data participates in the existing retained-pixel byte budget (`retained_bytes`, `evict_over_budget`).
- Mask dilation radius is `HEAL_DILATE_RADIUS: u32 = 2`, applied to the thresholded mask before healing.
- Rust: fmt + clippy `-D warnings` clean. TS: svelte-check 0 errors, vitest green.

---

### Task 1: fd-heal mask dilation

**Files:**
- Create: `engine/crates/fd-heal/src/dilate.rs`
- Modify: `engine/crates/fd-heal/src/lib.rs` (module + re-export)
- Test: `engine/crates/fd-heal/tests/dilate.rs`

**Interfaces:**
- Produces: `pub fn dilate(mask: &[bool], width: u32, height: u32, radius: u32) -> Vec<bool>` — Chebyshev (square) dilation: output pixel set when any set input pixel lies within `radius` on both axes. Separable two-pass implementation so a 168MP mask dilates in about a second, not tens.

- [ ] **Step 1: Write the failing test**

`engine/crates/fd-heal/tests/dilate.rs`:

```rust
use fd_heal::dilate;

fn mask_from(rows: &[&str]) -> (Vec<bool>, u32, u32) {
    let h = rows.len() as u32;
    let w = rows[0].len() as u32;
    let m = rows
        .iter()
        .flat_map(|r| r.chars().map(|c| c == '#'))
        .collect();
    (m, w, h)
}

fn render(mask: &[bool], w: u32) -> Vec<String> {
    mask.chunks(w as usize)
        .map(|row| row.iter().map(|&b| if b { '#' } else { '.' }).collect())
        .collect()
}

#[test]
fn radius_zero_is_identity() {
    let (m, w, h) = mask_from(&["..#..", ".....", "#...#"]);
    assert_eq!(dilate(&m, w, h, 0), m);
}

#[test]
fn single_pixel_dilates_to_a_clamped_square() {
    let (m, w, h) = mask_from(&[".....", "..#..", ".....", ".....", "....."]);
    let out = dilate(&m, w, h, 1);
    assert_eq!(
        render(&out, w),
        vec![".###.", ".###.", ".###.", ".....", "....."]
    );
}

#[test]
fn dilation_clamps_at_borders() {
    let (m, w, h) = mask_from(&["#....", ".....", ".....", ".....", "....#"]);
    let out = dilate(&m, w, h, 2);
    let rows = render(&out, w);
    assert_eq!(rows[0], "###..");
    // row 2 is reached by BOTH corners: (0,0) covers cols 0-2, (4,4) covers 2-4
    assert_eq!(rows[2], "#####");
    assert_eq!(rows[4], "..###");
}

#[test]
fn radius_two_covers_a_defect_rim() {
    // The product case: a 2px-under-covering mask grows to swallow the rim.
    let (m, w, h) = mask_from(&[
        ".......", //
        "...#...", //
        "..###..", //
        "...#...", //
        ".......", //
    ]);
    let out = dilate(&m, w, h, 2);
    // every pixel within Chebyshev distance 2 of a set pixel is set
    assert!(out.iter().filter(|&&b| b).count() > m.iter().filter(|&&b| b).count());
    assert!(out[0]); // (0,0) is within 2 of (2,2)
}
```

If a hand-check of an expected string ever disagrees with the implementation, re-derive the string from Chebyshev semantics on paper and fix the EXPECTATION — never bend the implementation to match a wrong expectation.

- [ ] **Step 2: Run test to verify it fails**

Run (repo root): `cargo test -p fd-heal --test dilate`
Expected: compile error — `dilate` not found.

- [ ] **Step 3: Implement**

`engine/crates/fd-heal/src/dilate.rs`:

```rust
//! Mask dilation. Detector masks cover a defect's confident core, not its
//! full visible extent; healing an under-covering mask leaves a visible rim
//! around the fill. Dilating by a couple of pixels before healing swallows
//! the rim (see the 3b-1 field notes).

/// Chebyshev (square-window) dilation: an output pixel is set when any set
/// input pixel lies within `radius` in both axes. Separable two-pass
/// (rows then columns), so cost is O(pixels * radius), not radius squared.
pub fn dilate(mask: &[bool], width: u32, height: u32, radius: u32) -> Vec<bool> {
    if radius == 0 {
        return mask.to_vec();
    }
    let (w, h) = (width as usize, height as usize);
    let r = radius as usize;
    // Pass 1: horizontal max over [x-r, x+r].
    let mut horiz = vec![false; w * h];
    for y in 0..h {
        let row = &mask[y * w..(y + 1) * w];
        let out = &mut horiz[y * w..(y + 1) * w];
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r + 1).min(w);
            out[x] = row[lo..hi].iter().any(|&b| b);
        }
    }
    // Pass 2: vertical max over [y-r, y+r] of the horizontal pass.
    let mut out = vec![false; w * h];
    for y in 0..h {
        let lo = y.saturating_sub(r);
        let hi = (y + r + 1).min(h);
        for x in 0..w {
            out[y * w + x] = (lo..hi).any(|yy| horiz[yy * w + x]);
        }
    }
    out
}
```

Add to `engine/crates/fd-heal/src/lib.rs`: `mod dilate;` and `pub use dilate::dilate;`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p fd-heal`
Expected: all fd-heal tests pass (existing 9 + 4 new).

- [ ] **Step 5: Format, lint, commit**

Run: `cargo fmt && cargo clippy -p fd-heal --all-targets -- -D warnings && cargo test -p fd-heal`

```bash
git add engine/crates/fd-heal
git commit -m "Add separable mask dilation to fd-heal"
```

---

### Task 2: Healed storage in the registry

**Files:**
- Modify: `engine/crates/fd-tiles/src/cache.rs` (TileKey gains `layer: u8`)
- Modify: `engine/crates/fd-tiles/tests/cache.rs` (key helper)
- Modify: `app/src-tauri/src/images.rs` (HealedData, set_healed, healed_tile, has_healed, healed_parts, retained_bytes, ImageInfo.healed)
- Modify: `app/src-tauri/src/lib.rs` (the two `ImageInfo` construction sites gain `healed`)
- Test: inline in `images.rs`

**Interfaces:**
- Consumes: `fd_tiles::{TileKey, TileCache, Pyramid}`, `fd_heal` (mask type is plain `Vec<bool>`).
- Produces: `TileKey { image_id: u64, layer: u8, level: u8, tx: u32, ty: u32 }` (layer 0 = rgba, 1 = healed; `evict_image` drops both layers since it filters on `image_id` only); `pub struct HealedData` (crate-visible fields: `image: Arc<fd_io::ImageBuf>`, `pyramid: Pyramid`, `mask: Arc<Vec<bool>>`); `Images::set_healed(&mut self, id: u64, image: Arc<fd_io::ImageBuf>, pyramid: Pyramid, mask: Arc<Vec<bool>>) -> bool` (false when id unknown OR healed dims differ from the entry's image OR mask length differs from pixel count); `Images::has_healed(&self, id: u64) -> bool`; `Images::healed_parts(&self, id: u64) -> Option<(Arc<fd_io::ImageBuf>, Arc<fd_io::ImageBuf>, Arc<Vec<bool>>)>` returning (original, healed, mask) for export; `Images::healed_tile(&mut self, id, level, tx, ty) -> Option<Arc<Tile>>`; `ImageInfo` gains `pub healed: bool`; `retained_bytes` includes healed image + healed pyramid RGBA.

- [ ] **Step 1: TileKey layer field**

In `engine/crates/fd-tiles/src/cache.rs`:

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TileKey {
    pub image_id: u64,
    /// Which pixel source this tile came from: 0 = original rgba pyramid,
    /// 1 = healed pyramid. Same image id, different pixels.
    pub layer: u8,
    pub level: u8,
    pub tx: u32,
    pub ty: u32,
}
```

Update `engine/crates/fd-tiles/tests/cache.rs`'s helper (the only construction site in the crate):

```rust
fn key(image: u64, tx: u32) -> TileKey {
    TileKey {
        image_id: image,
        layer: 0,
        level: 0,
        tx,
        ty: 0,
    }
}
```

`evict_image` needs no change (filters on `image_id` alone, so healed tiles evict with their image). The other construction site is `app/src-tauri/src/images.rs::tile` — add `layer: 0` there in the same commit so the workspace compiles.

Run: `cargo test -p fd-tiles` — all pass.

- [ ] **Step 2: Failing registry tests**

Add to `app/src-tauri/src/images.rs` tests:

```rust
    #[test]
    fn healed_storage_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        assert!(!info.healed);
        assert!(!images.has_healed(info.id));
        assert!(images.healed_tile(info.id, 0, 0, 0).is_none());

        let original = images.image(info.id).unwrap();
        let mut healed_buf = (*original).clone();
        // change one pixel so healed tiles are distinguishable
        if let fd_io::PixelData::U8(v) = &mut healed_buf.data {
            v[0] = 255;
        }
        let healed = std::sync::Arc::new(healed_buf);
        let pyramid = fd_tiles::Pyramid::build(&healed);
        let mask = std::sync::Arc::new(vec![false; 600 * 400]);
        let base_bytes = images.retained_bytes(info.id).unwrap();
        assert!(images.set_healed(info.id, healed.clone(), pyramid, mask.clone()));

        assert!(images.has_healed(info.id));
        let t = images.healed_tile(info.id, 0, 0, 0).unwrap();
        assert_eq!(t.rgba[0], 255); // healed pixels, not original
        let (orig, heal, m) = images.healed_parts(info.id).unwrap();
        assert_eq!(orig.width, heal.width);
        assert_eq!(m.len(), 600 * 400);

        // healed image + pyramid must count against the eviction budget
        assert!(images.retained_bytes(info.id).unwrap() > base_bytes + 600 * 400);

        images.close(info.id);
        assert!(images.healed_tile(info.id, 0, 0, 0).is_none());
        assert!(!images.has_healed(info.id));
    }

    #[test]
    fn set_healed_rejects_mismatched_dims_and_unknown_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        let wrong = std::sync::Arc::new(fd_io::ImageBuf {
            width: 100,
            height: 100,
            channels: 1,
            data: fd_io::PixelData::U8(vec![0; 100 * 100]),
            icc: None,
            exif: None,
        });
        let pyr = fd_tiles::Pyramid::build(&wrong);
        let mask = std::sync::Arc::new(vec![false; 100 * 100]);
        assert!(!images.set_healed(info.id, wrong.clone(), pyr, mask.clone()));
        let pyr2 = fd_tiles::Pyramid::build(&wrong);
        assert!(!images.set_healed(999, wrong, pyr2, mask));
    }
```

Run: `cargo test -p unduster-app healed` — FAIL (missing methods).

- [ ] **Step 3: Implement**

In `app/src-tauri/src/images.rs`:

```rust
/// A frame's healed pixels: a deep copy of the original with defects filled,
/// its own display pyramid, and the mask that was healed (needed again at
/// export time for the outside-mask verification).
pub struct HealedData {
    pub(crate) image: Arc<fd_io::ImageBuf>,
    pub(crate) pyramid: Pyramid,
    pub(crate) mask: Arc<Vec<bool>>,
}
```

`Entry` gains `healed: Option<HealedData>` (initialize `None` in `insert`). `ImageInfo` gains `pub healed: bool` (set `false` in `insert`). Methods:

```rust
    /// Stores a healed copy. Returns false when the id is unknown, the
    /// healed dims do not match the original, or the mask length is wrong --
    /// mirrors set_probs_built's validate-at-the-boundary posture.
    pub fn set_healed(
        &mut self,
        id: u64,
        image: Arc<fd_io::ImageBuf>,
        pyramid: Pyramid,
        mask: Arc<Vec<bool>>,
    ) -> bool {
        let Some(entry) = self.entries.get_mut(&id) else {
            return false;
        };
        if image.width != entry.image.width
            || image.height != entry.image.height
            || image.channels != entry.image.channels
            || mask.len() != (image.width as usize) * (image.height as usize)
        {
            return false;
        }
        entry.healed = Some(HealedData {
            image,
            pyramid,
            mask,
        });
        true
    }

    pub fn has_healed(&self, id: u64) -> bool {
        self.entries
            .get(&id)
            .is_some_and(|e| e.healed.is_some())
    }

    /// (original, healed, mask) for export verification and encoding.
    pub fn healed_parts(
        &self,
        id: u64,
    ) -> Option<(Arc<fd_io::ImageBuf>, Arc<fd_io::ImageBuf>, Arc<Vec<bool>>)> {
        let entry = self.entries.get(&id)?;
        let healed = entry.healed.as_ref()?;
        Some((entry.image.clone(), healed.image.clone(), healed.mask.clone()))
    }

    pub fn healed_tile(&mut self, id: u64, level: u8, tx: u32, ty: u32) -> Option<Arc<Tile>> {
        let key = TileKey {
            image_id: id,
            layer: 1,
            level,
            tx,
            ty,
        };
        let entry = self.entries.get(&id)?;
        let pyramid = &entry.healed.as_ref()?.pyramid;
        self.cache.get_or_insert(key, || pyramid.tile(level, tx, ty))
    }
```

Extend `retained_bytes`: after the probs block, add

```rust
        if let Some(healed) = &entry.healed {
            let hpx = (healed.image.width as usize) * (healed.image.height as usize);
            total += hpx * healed.image.channels as usize * depth;
            for l in &healed.pyramid.levels {
                total += (l.width as usize) * (l.height as usize) * 4;
            }
            total += healed.mask.len();
        }
```

In `lib.rs`, both `ImageInfo` literal sites (`insert` builds one; `activate_frame`'s reuse path builds one) gain `healed`: `insert` sets `false`; the reuse path sets `images.has_healed(id)` (compute inside the same lock scope that reads `level_dims`).

- [ ] **Step 4: Run tests, lint, commit**

Run: `cargo test -p fd-tiles -p unduster-app && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check`

```bash
git add engine/crates/fd-tiles app/src-tauri
git commit -m "Store healed copies in the registry under a tile-cache layer"
```

---

### Task 3: InpainterState and the heal_frame command

**Files:**
- Modify: `app/src-tauri/src/detect.rs` (InpainterState)
- Modify: `app/src-tauri/src/lib.rs` (load_inpainter, heal_frame, run_heal, setup autoload, handler registration, HEAL_DILATE_RADIUS)
- Test: inline in `detect.rs`

**Interfaces:**
- Consumes: `fd_heal::{heal, dilate, Inpainter}`, `fd_infer::Ep`, `Images::{image, set_healed}`, `Pyramid::build`.
- Produces: `InpainterState` (Clone over `Arc<Mutex<Option<fd_heal::Inpainter>>>`, `load(&self, path) -> Result<(), String>`); command `load_inpainter(path: String)`; command `heal_frame(id: u64, threshold: f32) -> Result<HealSummary, String>` where `HealSummary { id: u64, defects: usize, tiny: usize, inpainted: usize }`; `app-progress` stage `"healing"` with the terminal-`"ready"` guarantee (run_heal wrapper, same shape as run_detect); debug builds autoload `engine/fixtures/tiny-inpaint.onnx` (best-effort, after the detector autoload).

- [ ] **Step 1: Failing InpainterState test**

Add to `app/src-tauri/src/detect.rs`:

```rust
/// Cheaply cloneable handle to the (optional) inpainting model, mirroring
/// DetectorState. None means heal_frame falls back to classical fill only.
#[derive(Clone)]
pub struct InpainterState(Arc<Mutex<Option<fd_heal::Inpainter>>>);

impl Default for InpainterState {
    fn default() -> Self {
        InpainterState(Arc::new(Mutex::new(None)))
    }
}

impl InpainterState {
    pub fn load(&self, path: &Path) -> Result<(), String> {
        let inp = fd_heal::Inpainter::load(path, fd_infer::Ep::Cpu).map_err(|e| e.to_string())?;
        *self.0.lock().map_err(|e| e.to_string())? = Some(inp);
        Ok(())
    }

    /// Runs `f` with mutable access to the loaded inpainter (or None).
    pub fn with_inpainter<R>(
        &self,
        f: impl FnOnce(Option<&mut fd_heal::Inpainter>) -> R,
    ) -> Result<R, String> {
        let mut guard = self.0.lock().map_err(|e| e.to_string())?;
        Ok(f(guard.as_mut()))
    }
}
```

Tests (same file's test module):

```rust
    #[test]
    fn inpainter_state_loads_fixture_and_runs() {
        let state = InpainterState::default();
        state
            .load(
                &std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../engine/fixtures/tiny-inpaint.onnx"),
            )
            .unwrap();
        let has = state.with_inpainter(|i| i.is_some()).unwrap();
        assert!(has);
    }

    #[test]
    fn inpainter_state_defaults_to_none() {
        let state = InpainterState::default();
        assert!(!state.with_inpainter(|i| i.is_some()).unwrap());
        assert!(state.load(Path::new("/nonexistent/x.onnx")).is_err());
    }
```

Run: `cargo test -p unduster-app inpainter` — FAIL (missing type). Then the code above makes it pass.

- [ ] **Step 2: heal_frame command**

In `app/src-tauri/src/lib.rs`:

```rust
/// Dilation applied to the thresholded mask before healing: model masks
/// cover a defect's confident core, and healing an under-covering mask
/// leaves a visible rim (3b-1 field notes / issue TheUnduster-0s4).
const HEAL_DILATE_RADIUS: u32 = 2;

#[derive(serde::Serialize, Clone)]
struct HealSummary {
    id: u64,
    defects: usize,
    tiny: usize,
    inpainted: usize,
}

async fn run_heal(
    images: &State<'_, Mutex<Images>>,
    inpainter: &State<'_, detect::InpainterState>,
    id: u64,
    threshold: f32,
) -> Result<HealSummary, String> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(format!("threshold {threshold} out of range"));
    }
    let (image, mask) = {
        let images = images.lock().map_err(|e| e.to_string())?;
        let image = images.image(id).ok_or_else(|| format!("no image {id}"))?;
        let mask = images
            .threshold_mask(id, threshold)
            .ok_or_else(|| format!("no detection for image {id}"))?;
        (image, mask)
    };
    let inpainter = inpainter.inner().clone();
    let (healed, pyramid, mask, report) =
        tauri::async_runtime::spawn_blocking(move || {
            let mask = fd_heal::dilate(&mask, image.width, image.height, HEAL_DILATE_RADIUS);
            let mut copy = (*image).clone(); // the original Arc stays pristine
            let report = inpainter
                .with_inpainter(|inp| fd_heal::heal(&mut copy, &mask, inp))?
                .map_err(|e| e.to_string())?;
            let healed = std::sync::Arc::new(copy);
            let pyramid = fd_tiles::Pyramid::build(&healed);
            Ok::<_, String>((healed, pyramid, std::sync::Arc::new(mask), report))
        })
        .await
        .map_err(|e| e.to_string())??;
    let mut images = images.lock().map_err(|e| e.to_string())?;
    if !images.set_healed(id, healed, pyramid, mask) {
        return Err(format!("image {id} closed during healing"));
    }
    Ok(HealSummary {
        id,
        defects: report.defects,
        tiny: report.tiny,
        inpainted: report.inpainted,
    })
}

#[tauri::command]
async fn heal_frame(
    app: tauri::AppHandle,
    images: State<'_, Mutex<Images>>,
    inpainter: State<'_, detect::InpainterState>,
    id: u64,
    threshold: f32,
) -> Result<HealSummary, String> {
    let _ = app.emit("app-progress", Progress { id, stage: "healing" });
    let result = run_heal(&images, &inpainter, id, threshold).await;
    let _ = app.emit("app-progress", Progress { id, stage: "ready" });
    result
}

#[tauri::command]
fn load_inpainter(
    state: State<'_, detect::InpainterState>,
    path: String,
) -> Result<(), String> {
    state.load(std::path::Path::new(&path))
}
```

Supporting registry method (images.rs) — the native f32 threshold mask without exposing probs:

```rust
    /// Boolean mask of probs > threshold at native resolution; None until a
    /// detection has stored probabilities for this image.
    pub fn threshold_mask(&self, id: u64, threshold: f32) -> Option<Vec<bool>> {
        let entry = self.entries.get(&id)?;
        let (probs, _) = entry.probs.as_ref()?;
        Some(probs.iter().map(|&p| p > threshold).collect())
    }
```

Wire-up: `.manage(detect::InpainterState::default())`; add `load_inpainter` and `heal_frame` to `generate_handler!`; in `setup()`'s debug block, after the detector autoload: `let _ = app.state::<detect::InpainterState>().load(&fixtures.join("tiny-inpaint.onnx"));`

Registry test for `threshold_mask` (images.rs tests): reuse the probs fixture pattern from `prob_tiles_and_components_roundtrip` — set probs with one hot square, assert `threshold_mask(id, 0.5)` has exactly the square's pixel count true and `threshold_mask(id, 0.9)` is all false, `threshold_mask` on an un-detected image is None.

- [ ] **Step 3: Run tests, lint, commit**

Run: `cargo test -p unduster-app && cargo clippy -p unduster-app --all-targets -- -D warnings && cargo fmt --check`

```bash
git add app/src-tauri
git commit -m "Heal a frame into a stored copy with a dilated mask"
```

---

### Task 4: Healed layer on the tiles protocol

**Files:**
- Modify: `app/src-tauri/src/protocol.rs`
- Test: inline (extended)

**Interfaces:**
- Consumes: `Images::healed_tile`.
- Produces: `Layer::Healed`; paths `/healed/{id}/{level}/{tx}/{ty}` -> 200 with the same octet-stream + dimension-header shape as Rgba, 404 when the image has no healed data; existing layers byte-identical.

- [ ] **Step 1: Failing tests**

Extend `protocol.rs` tests:

```rust
    #[test]
    fn parses_healed_layer() {
        assert_eq!(
            parse_tile_path("/healed/3/1/7/2"),
            Some((Layer::Healed, 3, 1, 7, 2))
        );
        assert_eq!(parse_tile_path("/healed/3/1/7"), None);
        assert_eq!(parse_tile_path("/healed/3/1/7/2/9"), None);
    }

    #[test]
    fn healed_tile_404_before_heal() {
        let images = Mutex::new(Images::default());
        let roll = Mutex::new(None);
        assert_eq!(tile_response(&images, &roll, "/healed/1/0/0/0").status(), 404);
    }
```

Run: `cargo test -p unduster-app --lib protocol` — FAIL (no `Healed` variant).

- [ ] **Step 2: Implement**

`Layer` gains `Healed`. In `parse_tile_path`, extend the prefix match:

```rust
    let (layer, rest) = if let Some(rest) = trimmed.strip_prefix("probs/") {
        (Layer::Probs, rest)
    } else if let Some(rest) = trimmed.strip_prefix("healed/") {
        (Layer::Healed, rest)
    } else if let Some(rest) = trimmed.strip_prefix("thumb/") {
        // ... existing thumb handling unchanged
```

(Adapt to the actual current structure — thumb parsing has its own two-segment shape; keep it byte-identical.) In `tile_response`, `Layer::Healed` mirrors the Rgba arm against `images.healed_tile(id, level, tx, ty)` with a dev-build 404 log line `"[tiles] 404 healed {path} (no heal yet is normal)"`.

- [ ] **Step 3: Run tests, lint, commit**

Run: `cargo test -p unduster-app && cargo clippy -p unduster-app --all-targets -- -D warnings && cargo fmt --check`

```bash
git add app/src-tauri/src/protocol.rs
git commit -m "Serve healed tiles under a healed layer"
```

---

### Task 5: Before/after toggle and heal wiring in the UI

**Files:**
- Modify: `app/src/lib/Viewer.svelte` (space toggle, h key, healedAvailable prop, split base/display paths)
- Modify: `app/src/lib/renderer.ts` (tiles carry an explicit probPath)
- Modify: `app/src/lib/renderer.test.ts` (probPathFor unchanged; no test changes expected — verify)
- Modify: `app/src/App.svelte` (requestHeal, healing status, info.healed patch)

**Interfaces:**
- Consumes: `heal_frame` command, `/healed/` tile URLs, `ImageInfo.healed`.
- Produces: Viewer props gain `healedAvailable: boolean` and `onRequestHeal: () => void`; SPACE toggles `showHealed` when `healedAvailable` (resets on frame change and when availability drops); `h` requests healing. `tilePaths()` returns `{ path, probPath, ... }` where `path` is the DISPLAY source (`/healed`-prefixed when showing healed) and `probPath` is always derived from the BASE path — the red overlay must keep marking the original defect locations on both sides of the toggle. `renderer.draw` uses `t.probPath` instead of computing `probPathFor(t.path)`. App: `requestHeal` invokes `heal_frame(info.id, overlay.threshold)`, tracks `healing` state in the status line, and patches `info = { ...info, healed: true }` on success; status shows "space toggles before/after" when healed.

- [ ] **Step 1: Renderer path split**

In `renderer.ts`, `draw`'s tile parameter type gains `probPath: string`, and the prob fetch becomes:

```ts
      const probTex = overlay.enabled
        ? this.ensure(t.probPath, t.tileW, t.tileH, { single: true })
        : undefined;
```

(`probPathFor` stays exported and tested; `tilePaths` becomes its caller.)

- [ ] **Step 2: Viewer changes**

Props:

```ts
  let {
    info,
    overlay,
    detected,
    healedAvailable,
    onRequestDetect,
    onRequestHeal,
    bboxes = null,
  }: {
    info: ImageInfo;
    overlay: Overlay;
    detected: boolean;
    healedAvailable: boolean;
    onRequestDetect: () => void;
    onRequestHeal: () => void;
    bboxes?: [number, number, number, number][] | null;
  } = $props();
```

State + reset (extend the existing `info.id` effect and add an availability guard):

```ts
  let showHealed = $state(false);

  // inside the existing lastInfoId effect body, alongside detections reset:
  showHealed = false;

  $effect(() => {
    // If healed data vanishes (frame evicted and re-decoded), drop the toggle.
    if (!healedAvailable) showHealed = false;
  });
```

`tilePaths()`:

```ts
  function tilePaths() {
    return visibleTiles(info.levels, zoom, centerX, centerY, canvas.width, canvas.height).map(
      (t) => {
        const l = info.levels[t.level];
        const tileW = Math.min(l.width - t.tx * TILE, TILE);
        const tileH = Math.min(l.height - t.ty * TILE, TILE);
        const base = `/${info.id}/${t.level}/${t.tx}/${t.ty}`;
        return {
          path: showHealed ? `/healed${base}` : base,
          probPath: probPathFor(base),
          screenX: t.screenX,
          screenY: t.screenY,
          screenW: t.screenW,
          screenH: t.screenH,
          tileW,
          tileH,
        };
      },
    );
  }
```

(import `probPathFor` from `./renderer`.) Keys in `onKey`, before the pan block:

```ts
    if (e.key === "h") {
      e.preventDefault();
      onRequestHeal();
      return;
    } else if (e.key === " ") {
      if (healedAvailable) {
        e.preventDefault();
        showHealed = !showHealed;
        requestFrame();
      }
      return;
    }
```

Update the canvas aria-label to mention "h heals, space toggles before and after".

- [ ] **Step 3: App wiring**

```ts
  let healing = $state(false);

  async function requestHeal() {
    if (!info || healing || detecting) return;
    if (!detected && !(roll && roll.frames[currentIndex].defect_count !== null)) {
      error = "Run detection before healing";
      return;
    }
    // Healing needs live probabilities; if only queue results exist, run
    // detect first so the mask reflects the current threshold.
    if (!detected) {
      await requestDetect();
      if (!detected) return; // detect failed; its error is already shown
    }
    error = null;
    healing = true;
    try {
      await invoke("heal_frame", { id: info.id, threshold: overlay.threshold });
      info = { ...info, healed: true };
    } catch (e) {
      error = String(e);
    } finally {
      healing = false;
    }
  }
```

Pass `healedAvailable={info.healed ?? false}` and `onRequestHeal={requestHeal}` to the Viewer; add a "Heal" button next to Detect (`disabled={loading !== null || detecting || healing || !info}`, label `{healing ? "Healing..." : "Heal"}`); status line gains `{#if healing}&mdash; Healing...{/if}` and, when `info?.healed`, the hint `&mdash; space toggles before/after`.

The `ImageInfo` interface in App.svelte and Viewer.svelte gains `healed: boolean`.

- [ ] **Step 4: Verify, commit**

From app/: `npm run test && npm run check` (0 errors). From root: `cargo test -p unduster-app`, clippy, fmt (unchanged Rust).

```bash
git add app/src
git commit -m "Wire healing and an instant before/after toggle into the viewer"
```

---

### Task 6: Export core with outside-mask verification

**Files:**
- Create: `app/src-tauri/src/export.rs`
- Modify: `app/src-tauri/src/lib.rs` (mod, export_frame command, registration)
- Test: inline in `export.rs`

**Interfaces:**
- Consumes: `fd_io::{ImageBuf, PixelData, encode}`, `Images::healed_parts`.
- Produces: `ExportReport { pub changed_pixels: usize }`; `pub fn export_healed(original: &ImageBuf, healed: &ImageBuf, mask: &[bool], dest: &Path) -> Result<ExportReport, String>` — verifies EVERY channel of every outside-mask pixel is integer-identical (first mismatch aborts with its coordinate), counts changed inside-mask pixels, encodes to a hidden temp sibling (`.unduster-tmp-<file_name>`, real extension preserved so `fd_io::encode` routes correctly), atomically renames onto `dest`; ICC/EXIF ride along because the healed ImageBuf carries the original's metadata (the deep copy preserved it). Command `export_frame(id: u64, dest: String) -> Result<ExportReport, String>` (async, spawn_blocking).

- [ ] **Step 1: Failing tests**

`app/src-tauri/src/export.rs` (module with inline tests; RED = missing module):

```rust
//! Healed-file export: the spec's untouched-pixel guarantee is enforced
//! here, at write time, not merely trusted from the heal step.

use std::path::Path;

use fd_io::{ImageBuf, PixelData};

pub struct ExportReport {
    pub changed_pixels: usize,
}

/// Verifies the healed copy differs from the original ONLY inside the mask,
/// then writes it atomically to `dest` (temp sibling + rename). Any
/// outside-mask difference aborts before anything is written.
pub fn export_healed(
    original: &ImageBuf,
    healed: &ImageBuf,
    mask: &[bool],
    dest: &Path,
) -> Result<ExportReport, String> {
    if original.width != healed.width
        || original.height != healed.height
        || original.channels != healed.channels
    {
        return Err("healed dimensions do not match the original".to_string());
    }
    let px = (original.width as usize) * (original.height as usize);
    if mask.len() != px {
        return Err("mask length does not match the image".to_string());
    }
    let ch = original.channels as usize;
    let mut changed = 0usize;
    let mut check = |i: usize, differs: bool| -> Result<(), String> {
        if differs {
            if mask[i] {
                changed += 1;
            } else {
                let (x, y) = (i % original.width as usize, i / original.width as usize);
                return Err(format!(
                    "untouched-pixel guarantee violated at ({x}, {y}); export aborted"
                ));
            }
        }
        Ok(())
    };
    match (&original.data, &healed.data) {
        (PixelData::U8(a), PixelData::U8(b)) => {
            for i in 0..px {
                check(i, a[i * ch..(i + 1) * ch] != b[i * ch..(i + 1) * ch])?;
            }
        }
        (PixelData::U16(a), PixelData::U16(b)) => {
            for i in 0..px {
                check(i, a[i * ch..(i + 1) * ch] != b[i * ch..(i + 1) * ch])?;
            }
        }
        _ => return Err("healed bit depth does not match the original".to_string()),
    }

    let file_name = dest
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("bad destination: {}", dest.display()))?;
    let tmp = dest.with_file_name(format!(".unduster-tmp-{file_name}"));
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    fd_io::encode(&tmp, healed).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, dest).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", dest.display())
    })?;
    Ok(ExportReport {
        changed_pixels: changed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32, fill: u8) -> ImageBuf {
        ImageBuf {
            width: w,
            height: h,
            channels: 1,
            data: PixelData::U8(vec![fill; (w * h) as usize]),
            icc: Some(vec![1, 2, 3]),
            exif: None,
        }
    }

    #[test]
    fn exports_a_valid_heal_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let original = img(16, 16, 100);
        let mut healed = original.clone();
        let mut mask = vec![false; 256];
        mask[17] = true;
        if let PixelData::U8(v) = &mut healed.data {
            v[17] = 200;
        }
        let dest = dir.path().join("out.png");
        let report = export_healed(&original, &healed, &mask, &dest).unwrap();
        assert_eq!(report.changed_pixels, 1);
        assert!(dest.exists());
        assert!(!dir.path().join(".unduster-tmp-out.png").exists());
        let back = fd_io::decode(&dest).unwrap();
        assert_eq!(back.icc.as_deref(), Some([1u8, 2, 3].as_slice())); // metadata rode along
    }

    #[test]
    fn refuses_an_outside_mask_difference() {
        let dir = tempfile::tempdir().unwrap();
        let original = img(16, 16, 100);
        let mut healed = original.clone();
        if let PixelData::U8(v) = &mut healed.data {
            v[5] = 42; // tampered OUTSIDE the (empty) mask
        }
        let mask = vec![false; 256];
        let dest = dir.path().join("out.png");
        let err = export_healed(&original, &healed, &mask, &dest).unwrap_err();
        assert!(err.contains("untouched-pixel"));
        assert!(err.contains("(5, 0)"));
        assert!(!dest.exists()); // nothing written
    }

    #[test]
    fn end_to_end_with_a_real_heal() {
        // heal with the classical tier, then export: the engine guarantee
        // and the export verification must agree.
        let dir = tempfile::tempdir().unwrap();
        let mut noisy = img(64, 64, 0);
        if let PixelData::U8(v) = &mut noisy.data {
            for (i, p) in v.iter_mut().enumerate() {
                *p = ((i * 37) % 251) as u8;
            }
        }
        let original = noisy.clone();
        let mut mask = vec![false; 64 * 64];
        for y in 30..33 {
            for x in 30..33 {
                mask[y * 64 + x] = true;
            }
        }
        fd_heal::heal(&mut noisy, &mask, None).unwrap();
        let dest = dir.path().join("healed.tif");
        let report = export_healed(&original, &noisy, &mask, &dest).unwrap();
        assert!(report.changed_pixels > 0);
        assert!(dest.exists());
    }
}
```

The `check` closure captures `changed` mutably and returns Result — if the borrow checker objects to the closure shape, inline the logic in each match arm (repeat the small block, do not restructure the loop).

- [ ] **Step 2: Command**

`lib.rs`:

```rust
#[tauri::command]
async fn export_frame(
    images: State<'_, Mutex<Images>>,
    id: u64,
    dest: String,
) -> Result<usize, String> {
    let (original, healed, mask) = {
        let images = images.lock().map_err(|e| e.to_string())?;
        images
            .healed_parts(id)
            .ok_or_else(|| format!("image {id} has no healed data to export"))?
    };
    let report = tauri::async_runtime::spawn_blocking(move || {
        export::export_healed(&original, &healed, &mask, std::path::Path::new(&dest))
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(report.changed_pixels)
}
```

Add `mod export;` and register the command.

- [ ] **Step 3: Run tests, lint, commit**

Run: `cargo test -p unduster-app && cargo clippy -p unduster-app --all-targets -- -D warnings && cargo fmt --check`

```bash
git add app/src-tauri
git commit -m "Export healed files atomically with the untouched-pixel check"
```

---### Task 7: Roll export queue and UI

**Files:**
- Modify: `app/src-tauri/src/roll.rs` (Frame.exported, exporting flag, set_exported, frames_to_export)
- Modify: `app/src-tauri/src/lib.rs` (export_approved command + queue task + ExportFlagGuard)
- Modify: `app/src/App.svelte` (Export approved button + dialog + listeners), `app/src/lib/Filmstrip.svelte` (exported marker)
- Test: inline in `roll.rs` and `lib.rs`

**Interfaces:**
- Consumes: everything above plus the scan-queue patterns (`generation`, single-flight AtomicBool, drop guard, per-frame error continuation).
- Produces: `Frame` gains `#[serde(default)] pub exported: bool` (persisted); `RollState` gains `pub exporting: AtomicBool` + `clear_exporting`; `RollState::frames_to_export() -> Result<Vec<usize>, String>` (approved frames, in order — exported ones INCLUDED: pressing export again re-exports approved work, predictably overwriting); `RollState::set_exported(&self, generation: u64, index: usize) -> Result<(), String>` (generation-checked like record_scan_result, saves sidecar); command `export_approved(dest_dir: String) -> Result<(), String>` — single-flight, spawns a task that per approved frame: generation check; if the frame's `image_id` resolves to a registry entry WITH healed data, export that; otherwise decode + detect + heal (frame's stored threshold, dilated) + export transiently, holding at most one frame's pixels; emits `export-progress {index}`, `export-frame-error {index, message}`, `export-done`; destination file = `<dest_dir>/<file_name>` (same name and extension as the original). UI: an "Export approved" button (directory picker; disabled while exporting or when nothing is approved), an exporting status, and a second filmstrip marker for exported frames.

- [ ] **Step 1: Roll support (tests first)**

Tests in `roll.rs`:

```rust
    #[test]
    fn frames_to_export_lists_approved_in_order_including_exported() {
        let dir = tempfile::tempdir().unwrap();
        for n in ["a.png", "b.png", "c.png"] {
            std::fs::write(dir.path().join(n), b"x").unwrap();
        }
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        state.set_approved(0, true).unwrap();
        state.set_approved(2, true).unwrap();
        state.set_exported(state.generation(), 0).unwrap();
        assert_eq!(state.frames_to_export().unwrap(), vec![0, 2]);
    }

    #[test]
    fn set_exported_rejects_a_stale_generation_and_persists() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        let stale = state.generation();
        state.set_exported(state.generation(), 0).unwrap();
        state.open(dir.path()).unwrap(); // bumps generation, reloads sidecar
        assert!(state.set_exported(stale, 0).unwrap_err().contains("roll changed"));
        // persisted across the reopen:
        let info = state.open(dir.path()).unwrap().0;
        assert!(info.frames[0].exported);
    }
```

Implementation mirrors `record_scan_result`/`frames_to_scan` exactly (generation re-check under the roll lock, `roll.save()` after mutation). `Frame` gains the serde-defaulted field; `FrameInfo` mirrors it so the frontend sees it.

- [ ] **Step 2: export_approved queue**

`lib.rs` — mirror `scan_roll`'s shape precisely: compare_exchange on `roll.exporting`, clear on sync-body setup failure, `ExportFlagGuard` (a second drop-guard struct calling `clear_exporting`), generation snapshot, per-frame loop with the generation check first, per-frame error events + continue, `export-done` at the end. The per-frame body:

```rust
            // Prefer already-healed registry data (the operator reviewed it).
            let registry_export = {
                let roll_state = app_for_task.state::<roll::RollState>();
                let images_state = app_for_task.state::<Mutex<Images>>();
                match roll_state.image_id(index) {
                    Ok(Some(id)) => {
                        let images = match images_state.lock() {
                            Ok(g) => g,
                            Err(_) => continue,
                        };
                        images.healed_parts(id)
                    }
                    _ => None,
                }
            };
            let dest = dest_dir.join(&file_name);
            let outcome = if let Some((original, healed, mask)) = registry_export {
                tauri::async_runtime::spawn_blocking(move || {
                    export::export_healed(&original, &healed, &mask, &dest).map(|_| ())
                })
                .await
                .map_err(|e| e.to_string())
                .and_then(|r| r)
            } else {
                // Transient pipeline: decode, detect, heal, export -- one
                // frame's pixels at a time, dropped at the closure's end.
                let detector = detector.clone();
                let inpainter = inpainter.clone();
                let threshold = frame_threshold; // read from the frame earlier
                tauri::async_runtime::spawn_blocking(move || {
                    let prepared = images::Images::prepare(&path)?;
                    let probs = detector.detect(&prepared.image)?;
                    let mask = {
                        let raw: Vec<bool> = probs.iter().map(|&p| p > threshold).collect();
                        fd_heal::dilate(&raw, prepared.image.width, prepared.image.height, HEAL_DILATE_RADIUS)
                    };
                    let mut copy = (*prepared.image).clone();
                    inpainter
                        .with_inpainter(|inp| fd_heal::heal(&mut copy, &mask, inp))?
                        .map_err(|e| e.to_string())?;
                    export::export_healed(&prepared.image, &copy, &mask, &dest).map(|_| ())
                })
                .await
                .map_err(|e| e.to_string())
                .and_then(|r| r)
            };
            match outcome {
                Ok(()) => {
                    let _ = app_for_task
                        .state::<roll::RollState>()
                        .set_exported(generation, index);
                    let _ = app_for_task.emit("export-progress", ExportProgress { index });
                }
                Err(message) => {
                    let _ = app_for_task
                        .emit("export-frame-error", ExportFrameError { index, message });
                }
            }
```

(`frame_threshold` and `file_name`/`path` read from RollState per iteration like scan_roll does; `ExportProgress { index: usize }` / `ExportFrameError { index: usize, message: String }` serde structs.)

- [ ] **Step 3: UI**

App.svelte: `exporting = $state(false)`; listeners for the three export events (mark `roll.frames[i].exported = true` on progress; error → `error` string; done → `exporting = false`); button:

```svelte
    {#if roll}
      <button
        onclick={exportApproved}
        disabled={exporting || roll.frames.every((f) => !f.approved)}
      >
        {exporting ? "Exporting..." : "Export approved"}
      </button>
    {/if}
```

`exportApproved` picks a directory via the dialog (`open({ directory: true })`), sets `exporting = true`, invokes `export_approved`. Filmstrip: frames with `exported` show a second marker (a small "out" tag styled like the check, distinct color); `FrameInfo` interfaces in both components gain `exported: boolean`.

- [ ] **Step 4: Verify, commit**

Run: `cargo test -p unduster-app && cargo clippy -p unduster-app --all-targets -- -D warnings && cargo fmt --check`; from app/: `npm run test && npm run check`.

```bash
git add app
git commit -m "Export approved roll frames through a healing queue"
```

---

### Task 8: Sweep and manual gate

- [ ] **Step 1: Automated sweep**

Root: `cargo test` all crates green, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`. app/: `npm run test && npm run check`.

- [ ] **Step 2: Manual gate (human)**

`cd app && mise exec -- npm run tauri dev`, open the real roll:

1. On a frame with detections, press `d` (live detect), then `h` — status shows "Healing...", then the summary; press SPACE — the image flips between original and healed instantly; defects visibly filled (fixture inpainter: mean-fill on large defects, classical fill on specks — crude fills are expected until a real LaMa model lands).
2. The red overlay marks the same spots on both sides of the toggle.
3. Approve a couple of frames, press "Export approved", pick a destination folder: progress marks appear per frame, files land with original names, and they open in Preview with intact metadata.
4. Quit, relaunch, reopen the roll: exported markers persist.
5. Tamper check (optional but satisfying): the export refuses to write if anything outside the mask changed — trust the test suite here.

- [ ] **Step 3: Close out**

Ledger entry; `bd close TheUnduster-09e` and `bd close TheUnduster-0s4` (dilation shipped).

---

## Definition of done for plan 3c

- Healing never mutates an original; the copy carries ICC/EXIF; the mask is dilated 2px.
- Before/after is instant (both pyramids resident, tile-path prefix swap).
- Export verifies the untouched-pixel guarantee at write time, writes atomically, and works per frame and per approved roll with scan-queue-grade robustness.
- NOT here: a real LaMa ONNX model (tracked separately), brush mask editing (plan 3b-3), pyramid persistence (tracked separately).
