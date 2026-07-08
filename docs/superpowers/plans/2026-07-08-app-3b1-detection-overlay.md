# App 3b-1: Detection Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect defects on an opened scan and review them live: a shader-thresholded probability overlay driven by a sensitivity slider, Z-key navigation through detections, and an async open with progress so big scans never freeze the UI.

**Architecture:** The shell keeps the decoded `ImageBuf` (needed for inference, later healing) and, after `detect`, a quantized u8 probability pyramid (2x2 max-pool so specks stay visible zoomed out). Probability tiles ride the existing `tiles://` protocol under a `/probs/` prefix as single-channel bytes. The webview renders them as a second texture with the threshold applied in the fragment shader — the sensitivity slider is a uniform, one frame, no engine round-trip (the spec's 16ms budget). Defect components for Z-navigation come from `fd_heal::components` on the thresholded mask, engine-side.

**Tech Stack:** Existing stack (Tauri 2 shell in the root cargo workspace, Svelte 5 + WebGL2). New crate deps for the shell: fd-infer, fd-heal. Test detector: the committed `engine/fixtures/tiny-detector.onnx`.

## Global Constraints

- Trunk-based: commit directly to `main`, atomic commits, tests green first. No emoji; no Co-Authored-By.
- Pixels and probability bytes cross to the webview ONLY via the `tiles://` protocol; commands stay metadata-JSON.
- Sensitivity slider changes must not re-run inference or refetch tiles — shader uniform only.
- Heavy engine work (decode, pyramid, inference) runs off the main thread via `tauri::async_runtime::spawn_blocking`, with progress events `app-progress` (payload `{ id, stage }`).
- Keyboard additions: `d` run detection, `m` toggle overlay, `z`/`Z` next/previous detection; all existing keys keep working; canvas stays the single focus target, aria-label updated.
- The detector session lives behind a Mutex (ort sessions are `&mut`); inference must not hold the `Images` lock (clone the `Arc<ImageBuf>` out first).
- Memory note (accepted, documented): keeping `ImageBuf` per open image adds ~600MB for a 100MP 16-bit RGB scan; single-image scope for 3b-1, roll-scale eviction lands in 3b-2. `close_image` frees it.
- Rust: fmt + clippy `-D warnings` clean. TS: svelte-check no errors, vitest green.

---

### Task 1: Async open_image with progress and a loader (closes bead TheUnduster-44m)

**Files:**
- Modify: `app/src-tauri/src/lib.rs`
- Modify: `app/src-tauri/src/images.rs` (keep the ImageBuf on Entry)
- Modify: `app/src/App.svelte`
- Test: `app/src-tauri/src/images.rs` inline tests (extended)

**Interfaces:**
- Consumes: existing `Images` registry.
- Produces: `Entry` now holds `image: Arc<fd_io::ImageBuf>`; `Images::open` unchanged signature; `Images::image(&self, id: u64) -> Option<Arc<ImageBuf>>` (Task 3 uses it). Command `open_image` becomes `async fn`, does decode + pyramid inside `spawn_blocking`, and emits `app-progress` events with payloads `{"id": 0, "stage": "decoding"}` then `{"id": <id>, "stage": "building-pyramid"}` then `{"id": <id>, "stage": "ready"}`. App.svelte listens and renders a loading state (spinner text + `aria-busy`).

- [ ] **Step 1: Extend the registry (test first)**

Add to `app/src-tauri/src/images.rs` tests:

```rust
    #[test]
    fn image_pixels_are_retained_for_inference() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        let img = images.image(info.id).expect("pixels retained");
        assert_eq!((img.width, img.height), (600, 400));
        images.close(info.id);
        assert!(images.image(info.id).is_none());
    }
```

Run: `cargo test -p unduster-app image_pixels` — FAIL (no `image` method).

Implement: `Entry { image: Arc<fd_io::ImageBuf>, pyramid: Pyramid }`; in `open`, `let image = Arc::new(img);` build pyramid from `&image`, store both; add:

```rust
    pub fn image(&self, id: u64) -> Option<Arc<fd_io::ImageBuf>> {
        self.entries.get(&id).map(|e| e.image.clone())
    }
```

Run: `cargo test -p unduster-app` — green.

- [ ] **Step 2: Make the command async with progress**

Replace `open_image` in `app/src-tauri/src/lib.rs` — heavy work moves into a registry `prepare` function that runs on the blocking pool, so the command holds the `Images` lock only for a cheap insert:

```rust
use tauri::Emitter;

#[derive(serde::Serialize, Clone)]
struct Progress {
    id: u64,
    stage: &'static str,
}

#[tauri::command]
async fn open_image(
    app: tauri::AppHandle,
    state: tauri::State<'_, Mutex<Images>>,
    path: String,
) -> Result<ImageInfo, String> {
    let _ = app.emit("app-progress", Progress { id: 0, stage: "decoding" });
    let prepared = tauri::async_runtime::spawn_blocking(move || {
        Images::prepare(std::path::Path::new(&path))
    })
    .await
    .map_err(|e| e.to_string())??;
    let _ = app.emit("app-progress", Progress { id: 0, stage: "building-pyramid" });
    let info = {
        let mut images = state.lock().map_err(|e| e.to_string())?;
        images.insert(prepared)
    };
    let _ = app.emit("app-progress", Progress { id: info.id, stage: "ready" });
    Ok(info)
}
```

with registry support (decode AND pyramid both inside `prepare`, which runs in the blocking pool; `insert` is a cheap map insert under the lock):

```rust
pub struct Prepared {
    image: Arc<fd_io::ImageBuf>,
    pyramid: Pyramid,
}

impl Images {
    /// Heavy half of open: decode + pyramid, no registry access. Blocking.
    pub fn prepare(path: &Path) -> Result<Prepared, String> {
        let img = fd_io::decode(path).map_err(|e| e.to_string())?;
        let image = Arc::new(img);
        let pyramid = Pyramid::build(&image);
        Ok(Prepared { image, pyramid })
    }

    /// Cheap half: registry insert under the caller's lock.
    pub fn insert(&mut self, prepared: Prepared) -> ImageInfo {
        let id = self.next_id;
        self.next_id += 1;
        let levels = prepared
            .pyramid
            .levels
            .iter()
            .map(|l| LevelInfo { width: l.width, height: l.height })
            .collect();
        let info = ImageInfo {
            id,
            width: prepared.image.width,
            height: prepared.image.height,
            levels,
        };
        self.entries.insert(
            id,
            Entry { image: prepared.image, pyramid: prepared.pyramid },
        );
        info
    }
}
```

Keep the existing sync `Images::open` delegating to `prepare` + `insert` so all current tests pass unchanged. Note `Pyramid::build(&image)` takes `&ImageBuf` — `&Arc<ImageBuf>` derefs.

- [ ] **Step 3: Loader UI**

In `app/src/App.svelte`: add a `loading: string | null` state; before `invoke`, set `loading = "Opening scan"`; subscribe once in an effect to progress events:

```ts
import { listen } from "@tauri-apps/api/event";

let loading: string | null = $state(null);

$effect(() => {
  const un = listen<{ id: number; stage: string }>("app-progress", (e) => {
    loading =
      e.payload.stage === "ready"
        ? null
        : e.payload.stage === "decoding"
          ? "Decoding scan"
          : "Building preview";
  });
  return () => {
    un.then((f) => f());
  };
});
```

Render inside `.stage` when `loading` is set (and clear `loading = null` in openScan's catch):

```svelte
{#if loading}
  <p class="hint" role="status" aria-busy="true">{loading}...</p>
{/if}
```

Hide the hint/viewer swap appropriately (`{#if info && !loading}`). Keyboard/focus behavior unchanged.

- [ ] **Step 4: Verify, commit**

`cargo test -p unduster-app && cargo clippy -p unduster-app --all-targets -- -D warnings && cargo fmt --check`; from app/: `npm run check && npm run test`. Manual (deferred to final task): reopen the 96MP scan and see the loader.

```bash
git add app
git commit -m "Open scans asynchronously with progress and a loader"
```

Close bead: `bd close TheUnduster-44m`.

---

### Task 2: Detector state and load_detector command

**Files:**
- Create: `app/src-tauri/src/detect.rs`
- Modify: `app/src-tauri/src/lib.rs`, `app/src-tauri/Cargo.toml`
- Test: inline in `detect.rs`

**Interfaces:**
- Consumes: `fd_infer::{Detector, Ep}`.
- Produces: `DetectorState(Mutex<Option<Detector>>)` managed by Tauri; command `load_detector(path: String) -> Result<(), String>`; `DetectorState::detect(&self, img: &ImageBuf) -> Result<Vec<f32>, String>` (locks internally, errors "no detector loaded" when unset). Dev convenience: `run()` tries `../engine/fixtures/tiny-detector.onnx` relative to the src-tauri dir at startup in debug builds only, ignoring failure.

- [ ] **Step 1: Failing test**

`app/src-tauri/src/detect.rs` (module with inline tests; RED = missing module):

```rust
use std::path::Path;
use std::sync::{Arc, Mutex};

use fd_infer::{Detector, Ep};
use fd_io::ImageBuf;

/// Cheaply cloneable handle: Task 3's detect command clones it into a
/// spawn_blocking closure (which needs 'static).
#[derive(Clone)]
pub struct DetectorState(Arc<Mutex<Option<Detector>>>);

impl Default for DetectorState {
    fn default() -> Self {
        DetectorState(Arc::new(Mutex::new(None)))
    }
}

impl DetectorState {
    pub fn load(&self, path: &Path) -> Result<(), String> {
        let det = Detector::load(path, Ep::Cpu).map_err(|e| e.to_string())?;
        *self.0.lock().map_err(|e| e.to_string())? = Some(det);
        Ok(())
    }

    pub fn detect(&self, img: &ImageBuf) -> Result<Vec<f32>, String> {
        let mut guard = self.0.lock().map_err(|e| e.to_string())?;
        match guard.as_mut() {
            Some(det) => det.probabilities(img).map_err(|e| e.to_string()),
            None => Err("no detector loaded".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fd_io::PixelData;

    fn fixture() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../engine/fixtures/tiny-detector.onnx")
    }

    #[test]
    fn detect_without_model_names_the_problem() {
        let state = DetectorState::default();
        let img = ImageBuf {
            width: 8,
            height: 8,
            channels: 1,
            data: PixelData::U8(vec![0; 64]),
            icc: None,
            exif: None,
        };
        assert!(state.detect(&img).unwrap_err().contains("no detector"));
    }

    #[test]
    fn loads_fixture_and_detects() {
        let state = DetectorState::default();
        state.load(&fixture()).unwrap();
        let img = ImageBuf {
            width: 64,
            height: 48,
            channels: 1,
            data: PixelData::U8(vec![128; 64 * 48]),
            icc: None,
            exif: None,
        };
        let probs = state.detect(&img).unwrap();
        assert_eq!(probs.len(), 64 * 48);
        assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)));
    }

    #[test]
    fn load_missing_model_errors_with_path() {
        let state = DetectorState::default();
        let err = state.load(Path::new("/nonexistent/model.onnx")).unwrap_err();
        assert!(err.contains("model.onnx"));
    }
}
```

Wire-up in `lib.rs`: `mod detect;`, manage `detect::DetectorState::default()`, command:

```rust
#[tauri::command]
fn load_detector(
    state: tauri::State<'_, detect::DetectorState>,
    path: String,
) -> Result<(), String> {
    state.load(std::path::Path::new(&path))
}
```

Debug-build default load inside `run()` before `.run(...)`:

```rust
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                use tauri::Manager;
                let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../engine/fixtures/tiny-detector.onnx");
                let _ = app.state::<detect::DetectorState>().load(&fixture);
            }
            Ok(())
        })
```

Cargo.toml gains `fd-infer = { path = "../../engine/crates/fd-infer" }` and (for Task 3) `fd-heal = { path = "../../engine/crates/fd-heal" }`.

- [ ] **Step 2: Verify, commit**

`cargo test -p unduster-app` (3 new tests green; note ort downloads its runtime on first fd-infer build of this crate — already cached from the engine build), clippy, fmt.

```bash
git add app
git commit -m "Manage the ONNX detector in shell state with a load command"
```

---

### Task 3: detect command with probability pyramid

**Files:**
- Modify: `app/src-tauri/src/images.rs` (prob pyramid storage + builder)
- Modify: `app/src-tauri/src/lib.rs` (detect command)
- Test: inline in `images.rs`

**Interfaces:**
- Consumes: `DetectorState::detect`, `Images::image`.
- Produces: `ProbPyramid { levels: Vec<ProbLevel> }`, `ProbLevel { width: u32, height: u32, data: Vec<u8> }` (level dims MUST equal the display pyramid's level dims); `build_prob_pyramid(probs: &[f32], level_dims: &[(u32, u32)]) -> ProbPyramid` (quantize u8, then 2x2 MAX-pool per level so a 3px speck survives to coarse levels); `Images::set_probs(&mut self, id, ProbPyramid)`, `Images::prob_tile(&mut self, id, level, tx, ty) -> Option<(u32, u32, Vec<u8>)>` (single-channel tile bytes, edge-sized like RGBA tiles); `Images::components(&self, id, threshold: f32) -> Option<Vec<[u32; 4]>>` (bboxes from `fd_heal::components` on the NATIVE-res thresholded map — store the native f32 probs too for this and for 3b-2's brush compose). Command `detect(id) -> Result<DetectReport, String>` where `DetectReport { id: u64, components_at_half: usize }`, async, progress stages "detecting" / "ready", inference WITHOUT holding the Images lock.

- [ ] **Step 1: Failing tests**

Add to `images.rs` tests:

```rust
    #[test]
    fn prob_pyramid_max_pools_and_matches_level_dims() {
        // 4x4 probs with one hot pixel; two levels (4x4, 2x2)
        let mut probs = vec![0.0f32; 16];
        probs[5] = 0.9; // (1,1)
        let p = build_prob_pyramid(&probs, &[(4, 4), (2, 2)]);
        assert_eq!(p.levels.len(), 2);
        assert_eq!(p.levels[0].data[5], (0.9f32 * 255.0 + 0.5) as u8);
        // max-pool keeps the speck at (0,0) of the 2x2 level
        assert_eq!(p.levels[1].data[0], p.levels[0].data[5]);
        assert_eq!(p.levels[1].data[3], 0);
    }

    #[test]
    fn prob_tiles_and_components_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        let mut probs = vec![0.0f32; 600 * 400];
        for y in 100..104 {
            for x in 200..205 {
                probs[y * 600 + x] = 0.8;
            }
        }
        images.set_probs(info.id, probs.clone());
        let (w, h, bytes) = images.prob_tile(info.id, 0, 0, 0).unwrap();
        assert_eq!((w, h), (512, 400));
        assert_eq!(bytes.len(), (512 * 400) as usize);
        assert!(bytes[100 * 512 + 200] > 200);
        let comps = images.components(info.id, 0.5).unwrap();
        assert_eq!(comps.len(), 1);
        let b = comps[0];
        assert_eq!((b[0], b[1], b[2], b[3]), (200, 100, 205, 104));
        assert!(images.prob_tile(999, 0, 0, 0).is_none());
        assert!(images.components(info.id, 0.9).unwrap().is_empty());
    }
```

(`set_probs` takes the native f32 vec and builds+stores the pyramid internally, retaining the f32 map: signature `pub fn set_probs(&mut self, id: u64, probs: Vec<f32>) -> bool`.)

Run: FAIL (missing functions).

- [ ] **Step 2: Implement**

In `images.rs`:

```rust
pub struct ProbLevel {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

pub struct ProbPyramid {
    pub levels: Vec<ProbLevel>,
}

/// Quantize native-res probabilities and max-pool down the given level dims
/// (which must match the display pyramid). Max, not mean: a 3px dust speck
/// must stay visible when zoomed out.
pub fn build_prob_pyramid(probs: &[f32], level_dims: &[(u32, u32)]) -> ProbPyramid {
    let (w0, h0) = level_dims[0];
    let mut levels = Vec::with_capacity(level_dims.len());
    let base: Vec<u8> = probs
        .iter()
        .map(|p| (p.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect();
    debug_assert_eq!(base.len(), (w0 * h0) as usize);
    levels.push(ProbLevel { width: w0, height: h0, data: base });
    for &(w, h) in &level_dims[1..] {
        let prev = levels.last().unwrap();
        let mut data = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let mut m = 0u8;
                for dy in 0..2u32 {
                    for dx in 0..2u32 {
                        let (sx, sy) = (x * 2 + dx, y * 2 + dy);
                        if sx < prev.width && sy < prev.height {
                            m = m.max(prev.data[(sy * prev.width + sx) as usize]);
                        }
                    }
                }
                data[(y * w + x) as usize] = m;
            }
        }
        levels.push(ProbLevel { width: w, height: h, data });
    }
    ProbPyramid { levels }
}
```

Entry grows `probs: Option<(Vec<f32>, ProbPyramid)>` (None until detect). `set_probs` computes `level_dims` from the entry's display pyramid, builds, stores, returns true (false when id unknown). `prob_tile` mirrors `Pyramid::tile`'s grid/edge logic on `ProbLevel` (single channel — factor the tile-extraction loop if it stays readable, don't force it). `components`:

```rust
    pub fn components(&self, id: u64, threshold: f32) -> Option<Vec<[u32; 4]>> {
        let entry = self.entries.get(&id)?;
        let (probs, _) = entry.probs.as_ref()?;
        let mask: Vec<bool> = probs.iter().map(|&p| p > threshold).collect();
        let (w, h) = (entry.image.width, entry.image.height);
        Some(
            fd_heal::components(&mask, w, h)
                .into_iter()
                .map(|d| [d.bbox.x0, d.bbox.y0, d.bbox.x1, d.bbox.y1])
                .collect(),
        )
    }
```

`close()` drops probs with the entry (already does). Commands in `lib.rs`:

```rust
#[derive(serde::Serialize, Clone)]
struct DetectReport {
    id: u64,
    components_at_half: usize,
}

#[tauri::command]
async fn detect(
    app: tauri::AppHandle,
    images: tauri::State<'_, Mutex<Images>>,
    detector: tauri::State<'_, detect::DetectorState>,
    id: u64,
) -> Result<DetectReport, String> {
    let img = {
        let images = images.lock().map_err(|e| e.to_string())?;
        images.image(id).ok_or_else(|| format!("no image {id}"))?
    }; // lock released; inference runs on the Arc clone
    let _ = app.emit("app-progress", Progress { id, stage: "detecting" });
    let detector = detector.inner().clone(); // DetectorState is Clone over an Arc
    let probs =
        tauri::async_runtime::spawn_blocking(move || detector.detect(&img))
            .await
            .map_err(|e| e.to_string())??;
    let report = {
        let mut images = images.lock().map_err(|e| e.to_string())?;
        if !images.set_probs(id, probs) {
            return Err(format!("image {id} closed during detection"));
        }
        DetectReport {
            id,
            components_at_half: images.components(id, 0.5).unwrap_or_default().len(),
        }
    };
    let _ = app.emit("app-progress", Progress { id, stage: "ready" });
    Ok(report)
}

#[tauri::command]
fn components(
    images: tauri::State<'_, Mutex<Images>>,
    id: u64,
    threshold: f32,
) -> Result<Vec<[u32; 4]>, String> {
    let images = images.lock().map_err(|e| e.to_string())?;
    images.components(id, threshold).ok_or_else(|| "no detection for image".to_string())
}
```

`DetectorState` is already `Clone` over an `Arc` (Task 2); `tauri::State::inner()` gives `&DetectorState`, so `detector.inner().clone()` moves a cheap handle into the closure. Register both commands.

- [ ] **Step 3: Verify, commit**

`cargo test -p unduster-app` (new tests + all prior green), clippy, fmt.

```bash
git add app
git commit -m "Run detection into a max-pooled probability pyramid"
```

---

### Task 4: probs layer on the tiles protocol

**Files:**
- Modify: `app/src-tauri/src/protocol.rs`
- Test: inline (extended)

**Interfaces:**
- Consumes: `Images::prob_tile`.
- Produces: URL `tiles://localhost/probs/{id}/{level}/{tx}/{ty}` -> 200, `Content-Type: application/octet-stream`, `x-tile-width`/`x-tile-height`, body single-channel u8; 404 when image/tile missing OR no detection ran; existing image-tile URLs unchanged. `parse_tile_path` gains the layer: returns `Option<(Layer, u64, u8, u32, u32)>` with `enum Layer { Rgba, Probs }` (no leading segment = Rgba).

- [ ] **Step 1: Failing tests**

Extend `protocol.rs` tests:

```rust
    #[test]
    fn parses_probs_layer() {
        assert_eq!(parse_tile_path("/probs/3/1/7/2"), Some((Layer::Probs, 3, 1, 7, 2)));
        assert_eq!(parse_tile_path("/3/1/7/2"), Some((Layer::Rgba, 3, 1, 7, 2)));
        assert_eq!(parse_tile_path("/probs/3/1/7"), None);
        assert_eq!(parse_tile_path("/unknown/3/1/7/2"), None);
    }

    #[test]
    fn probs_tile_404_before_detection() {
        let images = Mutex::new(Images::default());
        assert_eq!(tile_response(&images, "/probs/1/0/0/0").status(), 404);
    }
```

Adjust the two existing parse tests to the new tuple shape. Run: FAIL.

- [ ] **Step 2: Implement**

```rust
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Layer {
    Rgba,
    Probs,
}

pub fn parse_tile_path(path: &str) -> Option<(Layer, u64, u8, u32, u32)> {
    let trimmed = path.trim_start_matches('/');
    let (layer, rest) = match trimmed.strip_prefix("probs/") {
        Some(rest) => (Layer::Probs, rest),
        None => (Layer::Rgba, trimmed),
    };
    let mut parts = rest.split('/');
    let id = parts.next()?.parse().ok()?;
    let level = parts.next()?.parse().ok()?;
    let tx = parts.next()?.parse().ok()?;
    let ty = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((layer, id, level, tx, ty))
}
```

(Note `"unknown/3/1/7/2"` correctly fails: `"unknown".parse::<u64>()` errors.) In `tile_response`, match the layer: `Layer::Rgba` as today; `Layer::Probs` -> `images.prob_tile(id, level, tx, ty)` mapping `Some((w, h, bytes))` to 200 and `None` to 404.

- [ ] **Step 3: Verify, commit**

`cargo test -p unduster-app`, clippy, fmt.

```bash
git add app
git commit -m "Serve probability tiles under a probs layer"
```

---

### Task 5: Overlay rendering and review controls

**Files:**
- Modify: `app/src/lib/renderer.ts` (second texture, threshold uniform)
- Modify: `app/src/lib/Viewer.svelte` (slider, keys d/m/z/Z, detection jumps)
- Modify: `app/src/App.svelte` (detect wiring, status line)
- Test: `app/src/lib/renderer.test.ts` (key scheme), `app/src/lib/viewport.test.ts` untouched

**Interfaces:**
- Consumes: `/probs/` tile URLs (Task 4), `detect` + `components` commands (Task 3).
- Produces: `TileRenderer.draw(tiles, canvasW, canvasH, overlay: { enabled: boolean; threshold: number })` — when enabled, fetches `/probs${path}` per tile as a LUMINANCE texture and the fragment shader tints pixels whose prob exceeds the threshold; slider input drives only a redraw (uniform), never refetch (textures carry raw probs). Viewer props gain `onRequestDetect: () => void`; keyboard: `d` requests detection, `m` toggles overlay, `z`/`Z` cycle components (zoom to 100%, center on bbox center). App tracks `detected` state, shows the defect count, exposes a labelled `<input type="range">` sensitivity slider (0.05..0.95 step 0.01, default 0.5) bound to the overlay threshold.

- [ ] **Step 1: Failing renderer test**

```ts
import { probPathFor } from "./renderer";

describe("probPathFor", () => {
  it("prefixes the layer and preserves the tile path", () => {
    expect(probPathFor("/3/0/1/2")).toBe("/probs/3/0/1/2");
  });
});
```

Run: FAIL. (The texture-store key scheme derives from the fetched path, so this one pure function pins the contract.)

- [ ] **Step 2: Implement renderer changes**

Shader replaces FRAG:

```glsl
#version 300 es
precision mediump float;
in vec2 vUv;
out vec4 color;
uniform sampler2D tile;
uniform sampler2D probs;
uniform float threshold;
uniform float overlayOn;
void main() {
  vec4 base = texture(tile, vUv);
  float p = texture(probs, vUv).r;
  float hit = overlayOn * step(threshold, p) * step(0.004, p); // never tint p==0
  color = mix(base, vec4(1.0, 0.25, 0.25, 1.0), hit * 0.55);
}
```

`export function probPathFor(path: string): string { return "/probs" + path; }`

In `draw(tiles, w, h, overlay)`: set `threshold`/`overlayOn` uniforms once per frame; per tile, bind unit 0 = rgba texture; unit 1 = prob texture when overlay enabled and `ensure(probPathFor(t.path))` returns one (uploaded with `gl.R8`/`gl.RED` + UNPACK_ALIGNMENT 1 — the byte length `w*h` distinguishes prob tiles from rgba in `ensure`'s upload branch: pass the format explicitly instead, via `ensure(path, { single: boolean })`), else bind a static 1x1 zero texture (create once in the constructor) so the shader always has both samplers. A prob fetch 404 (no detection yet) leaves the zero texture — overlay is simply invisible.

- [ ] **Step 3: Viewer + App wiring**

Viewer additions (props `{ info, overlay, onRequestDetect }` where `overlay = { enabled, threshold }` is `$state` owned by App and passed down; Svelte 5: accept as plain props and let App re-render — the draw loop reads them each frame via the props reference):

```ts
else if (e.key === "d") onRequestDetect();
else if (e.key === "m") { overlay.enabled = !overlay.enabled; requestFrame(); }
else if (e.key === "z" || e.key === "Z") cycleDetection(e.key === "z" ? 1 : -1);
```

```ts
let detections: [number, number, number, number][] = $state([]);
let current = -1;

export async function refreshDetections(threshold: number) {
  detections = await invoke("components", { id: info.id, threshold });
  current = -1;
}

function cycleDetection(dir: 1 | -1) {
  if (!detections.length) return;
  current = (current + dir + detections.length) % detections.length;
  const [x0, y0, x1, y1] = detections[current];
  zoom = 1;
  centerX = (x0 + x1) / 2;
  centerY = (y0 + y1) / 2;
  clampCenter();
  requestFrame();
}
```

`frame()` passes `overlay` to `renderer.draw(...)`; slider changes call `requestFrame()` (uniform-only redraw) and, debounced 250ms, `refreshDetections(threshold)` so `z` targets stay in sync (the DISPLAY updates in one frame; the component list may lag by the debounce — acceptable and stated).

App: `overlay = $state({ enabled: true, threshold: 0.5 })`; hold the viewer instance with `let viewer; ... <Viewer bind:this={viewer} .../>` so App can call its exported `refreshDetections`; a `detect` button + `d` passthrough calling `invoke("detect", { id: info.id })`, storing `components_at_half` for the status line ("N defects at 50%"), then `viewer.refreshDetections(overlay.threshold)`; the slider:

```svelte
<label>
  Sensitivity
  <input
    type="range"
    min="0.05"
    max="0.95"
    step="0.01"
    bind:value={overlay.threshold}
  />
</label>
```

Update the canvas aria-label to mention d/m/z keys.

- [ ] **Step 4: Verify, commit**

From app/: `npm run test && npm run check`; from root: `cargo test -p unduster-app`, clippy, fmt.

```bash
git add app
git commit -m "Render the detection overlay with a live sensitivity slider"
```

---

### Task 6: End-to-end verification (manual gate + wrap-up)

**Files:**
- Modify: `.superpowers/sdd/progress.md` (ledger)

- [ ] **Step 1: Automated sweep**

Root: `cargo test --workspace` variants green (`cargo test -p unduster-app` + engine crates), clippy, fmt. app/: vitest + svelte-check.

- [ ] **Step 2: Manual gate (human)**

`cd app && mise exec -- npm run tauri dev`, open the 96MP scan: loader shows while opening; press `d` — "detecting" progress then a defect count; `m` toggles the red overlay; drag the sensitivity slider — tint updates live with NO stutter (this is the one-frame budget made visible; the tiny fixture detector fires on noise, which is fine — it proves the path); `z`/`Z` jump between detections at 100%. Note: with the fixture detector the "defects" are noise blobs — the real model swap is a config change later.

- [ ] **Step 3: Close out**

Ledger entry; `bd close TheUnduster-44m` if not already; commit ledger is unnecessary (gitignored scratch).

---

## Definition of done for plan 3b-1

- Async open with visible progress; detection runs without freezing the UI.
- Sensitivity slider re-thresholds in-frame (shader uniform), never re-runs inference.
- Probability bytes flow only over the tiles protocol; commands stay JSON metadata.
- z/Z navigation works from engine-side components (fd-heal reuse).
- NOT here (3b-2): rolls, filmstrip, brush editing, sidecar persistence, approve flow, memory eviction for multi-frame sessions.
