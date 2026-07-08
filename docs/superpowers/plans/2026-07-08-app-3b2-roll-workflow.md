# App 3b-2: Roll Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Review a whole roll of scans, not one image at a time. Open a folder of frames, let a background queue detect defects on every frame while the operator works, page through with `,`/`.`, approve with `A`, see progress in a filmstrip, and survive an app restart with everything (thresholds, approvals, defect counts) restored from a sidecar file. Memory stays bounded no matter how large the roll: at most three activated frames plus one queue frame ever hold pixels at once.

**Architecture:** A new `Roll` (Rust) tracks per-frame metadata — file name, sensitivity threshold, approval, defect count, cached bboxes, and a runtime-only `image_id` linking to the existing `Images` registry when a frame is activated. The roll's ground truth is a versioned JSON sidecar (`.unduster/roll.json`) written atomically after every mutation. Activating a frame reuses the exact staged decode-and-pyramid pipeline from 3b-1 (`Images::prepare`/`insert`) and evicts any other frame's registry entry outside a 3-wide activation window, so the UI-driven memory footprint never grows with roll size. A single background Tauri task walks frames lacking a defect count, decoding each locally (never inserted into the registry, so it never competes with the UI's activation window), running the existing `DetectorState`, computing bboxes server-side from the thresholded probability map, writing a small PNG thumbnail, saving the sidecar, and emitting progress — then dropping all pixel data before moving to the next frame. Thumbnails and prob/rgba tiles all cross to the webview only through the existing `tiles://` protocol, now with a third `Thumb` layer alongside `Rgba` and `Probs`. The frontend adds a keyboard-accessible filmstrip (listbox semantics) below the viewer and wires `,`/`.`/`A` at the App level, plus GPU ring markers so defects are visible even zoomed out, before any live detection has run on the currently open frame.

**Tech Stack:** Existing stack unchanged (Tauri 2 shell in the root cargo workspace, Svelte 5 + WebGL2, fd-io/fd-tiles/fd-infer/fd-heal). No new crate dependencies — thumbnail PNG encoding reuses `fd_io::encode` (already writes grayscale/RGB PNGs) and `fd_tiles::downsample_2x`. No new npm dependencies — the roll-folder picker reuses `@tauri-apps/plugin-dialog`'s `open` with `directory: true`, already a dependency.

## Global Constraints

- Trunk-based: commit directly to `main`, atomic commits, tests green first. No emoji; no Co-Authored-By.
- Pixels and probability/thumbnail bytes cross to the webview ONLY via the `tiles://` protocol; commands stay metadata-JSON.
- Heavy work stays off the UI thread: decode, pyramid, and inference run via `tauri::async_runtime::spawn_blocking` (activation path) or inside the background queue task (scan path), both with progress events.
- Memory bound: at most 3 activated frames (the current frame plus its two neighbors) hold registry entries at once, plus at most 1 additional frame's pixels transiently inside the background scan queue. `activate_frame` enforces the first bound by eviction; the queue enforces the second by never registering its frame and dropping it before moving on.
- Sidecar writes are atomic: write to `roll.json.tmp`, rename the existing `roll.json` to `roll.json.bak` first, then rename the tmp file into place. Every mutation (threshold, approval, defect count, activation's `image_id` is NOT persisted — see Task 1) saves immediately.
- No emoji anywhere (code, commits, docs, UI copy).
- Rust: fmt + clippy `-D warnings` clean. TS: svelte-check no errors, vitest green.
- Keyboard accessibility: the filmstrip is a `listbox`/`option` pattern with roving tabindex, visible focus ring, `Enter` to select; all existing viewer keys keep working.

---

### Task 1: Roll state and sidecar persistence (pure Rust, heavy TDD)

**Files:**
- Create: `app/src-tauri/src/roll.rs`
- Modify: `app/src-tauri/src/lib.rs` (module registration only, `mod roll;`)
- Modify: `app/src-tauri/Cargo.toml` (dev-dependency `tempfile` already present; add `serde_json` as a normal dependency — already present)
- Test: inline in `roll.rs`

**Interfaces:**
- Consumes: nothing outside `std::fs`/`serde`.
- Produces: `Frame { file_name: String, threshold: f32, approved: bool, defect_count: Option<usize>, bboxes: Option<Vec<[u32; 4]>>, image_id: Option<u64> }` with `#[serde(skip)]` on `image_id` (runtime-only: a fresh process has no registry, so persisting an id would dangle); `Roll { dir: PathBuf, frames: Vec<Frame> }`; `Roll::open(dir: &Path) -> Result<Roll, String>` (lists image files, loads/merges sidecar, does no decoding); `Roll::save(&self) -> Result<(), String>` (atomic write with `.bak` rotation); sidecar envelope `{"version": 1, "frames": [...]}`.

- [ ] **Step 1: Failing tests for the sidecar envelope and atomic save**

Create `app/src-tauri/src/roll.rs` with the data types and inline tests (RED: module doesn't exist yet, nothing compiles until Step 2):

```rust
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Roll-relative sidecar and thumbnail directory, e.g. `<roll_dir>/.unduster/`.
const SIDECAR_SUBDIR: &str = ".unduster";
const SIDECAR_FILE: &str = "roll.json";
const CURRENT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Frame {
    pub file_name: String,
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    #[serde(default)]
    pub approved: bool,
    #[serde(default)]
    pub defect_count: Option<usize>,
    #[serde(default)]
    pub bboxes: Option<Vec<[u32; 4]>>,
    /// Registry id while the frame's pixels are activated. Runtime-only:
    /// a fresh process has no registry entries, so persisting this would
    /// point at nothing after restart.
    #[serde(skip)]
    pub image_id: Option<u64>,
}

fn default_threshold() -> f32 {
    0.5
}

impl Frame {
    fn fresh(file_name: String) -> Frame {
        Frame {
            file_name,
            threshold: default_threshold(),
            approved: false,
            defect_count: None,
            bboxes: None,
            image_id: None,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct SidecarEnvelope {
    version: u32,
    frames: Vec<Frame>,
}

pub struct Roll {
    pub dir: PathBuf,
    pub frames: Vec<Frame>,
}

const IMAGE_EXTENSIONS: &[&str] = &["tif", "tiff", "png", "jpg", "jpeg"];

fn sidecar_dir(dir: &Path) -> PathBuf {
    dir.join(SIDECAR_SUBDIR)
}

fn sidecar_path(dir: &Path) -> PathBuf {
    sidecar_dir(dir).join(SIDECAR_FILE)
}

pub fn thumbs_dir(dir: &Path) -> PathBuf {
    sidecar_dir(dir).join("thumbs")
}

fn list_image_files(dir: &Path) -> Result<Vec<String>, String> {
    let read = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut names: Vec<String> = Vec::new();
    for entry in read {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if IMAGE_EXTENSIONS.iter().any(|e| e.eq_ignore_ascii_case(ext)) {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                names.push(name.to_string());
            }
        }
    }
    names.sort_by_key(|n| n.to_lowercase());
    Ok(names)
}

fn load_sidecar(dir: &Path) -> Result<Vec<Frame>, String> {
    let path = sidecar_path(dir);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let envelope: SidecarEnvelope =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    if envelope.version != CURRENT_VERSION {
        return Err(format!(
            "{}: unsupported sidecar version {} (expected {CURRENT_VERSION})",
            path.display(),
            envelope.version
        ));
    }
    Ok(envelope.frames)
}

/// Reconciles the sidecar's remembered frames with what is actually on disk:
/// files with no sidecar entry become fresh frames (appended, so newly added
/// scans land at the end); sidecar entries whose files vanished are dropped.
fn merge(on_disk: Vec<String>, mut remembered: Vec<Frame>) -> Vec<Frame> {
    let mut out: Vec<Frame> = Vec::with_capacity(on_disk.len());
    for name in on_disk {
        if let Some(pos) = remembered.iter().position(|f| f.file_name == name) {
            out.push(remembered.remove(pos));
        } else {
            out.push(Frame::fresh(name));
        }
    }
    out
}

impl Roll {
    /// Lists image files in `dir`, loads and merges the sidecar (missing
    /// sidecar = fresh roll), does no decoding. Files not in the sidecar are
    /// appended as fresh frames; sidecar entries for files no longer on disk
    /// are dropped.
    pub fn open(dir: &Path) -> Result<Roll, String> {
        let on_disk = list_image_files(dir)?;
        let remembered = load_sidecar(dir)?;
        let frames = merge(on_disk, remembered);
        Ok(Roll {
            dir: dir.to_path_buf(),
            frames,
        })
    }

    /// Atomic sidecar write: previous file (if any) renamed to `.bak` first,
    /// new content written to `.tmp`, then renamed into place. A crash
    /// between these steps leaves either the old file, the `.bak`, or the
    /// new file intact -- never a half-written `roll.json`.
    pub fn save(&self) -> Result<(), String> {
        let dir = sidecar_dir(&self.dir);
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = sidecar_path(&self.dir);
        let tmp = path.with_extension("json.tmp");
        let bak = path.with_extension("json.bak");
        let envelope = SidecarEnvelope {
            version: CURRENT_VERSION,
            frames: self.frames.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&envelope).map_err(|e| e.to_string())?;
        std::fs::write(&tmp, &bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
        if path.exists() {
            std::fs::rename(&path, &bak).map_err(|e| format!("{}: {e}", bak.display()))?;
        }
        std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(dir: &Path, name: &str) {
        std::fs::write(dir.join(name), b"not a real image, just a marker").unwrap();
    }

    #[test]
    fn open_lists_image_files_sorted_case_insensitively() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "b.PNG");
        touch(dir.path(), "a.tif");
        touch(dir.path(), "c.jpeg");
        touch(dir.path(), "notes.txt");
        let roll = Roll::open(dir.path()).unwrap();
        let names: Vec<&str> = roll.frames.iter().map(|f| f.file_name.as_str()).collect();
        assert_eq!(names, vec!["a.tif", "b.PNG", "c.jpeg"]);
        for f in &roll.frames {
            assert_eq!(f.threshold, 0.5);
            assert!(!f.approved);
            assert_eq!(f.defect_count, None);
            assert_eq!(f.image_id, None);
        }
    }

    #[test]
    fn open_on_missing_sidecar_is_a_fresh_roll() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "a.tif");
        let roll = Roll::open(dir.path()).unwrap();
        assert_eq!(roll.frames.len(), 1);
    }

    #[test]
    fn save_then_open_round_trips_frame_state() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "a.tif");
        touch(dir.path(), "b.tif");
        let mut roll = Roll::open(dir.path()).unwrap();
        roll.frames[0].threshold = 0.72;
        roll.frames[0].approved = true;
        roll.frames[0].defect_count = Some(3);
        roll.frames[0].bboxes = Some(vec![[1, 2, 3, 4]]);
        roll.save().unwrap();

        let reopened = Roll::open(dir.path()).unwrap();
        assert_eq!(reopened.frames[0].threshold, 0.72);
        assert!(reopened.frames[0].approved);
        assert_eq!(reopened.frames[0].defect_count, Some(3));
        assert_eq!(reopened.frames[0].bboxes, Some(vec![[1, 2, 3, 4]]));
        // image_id is never persisted, even if it happened to be set.
        assert_eq!(reopened.frames[0].image_id, None);
        assert_eq!(reopened.frames[1].threshold, 0.5);
    }

    #[test]
    fn save_writes_bak_on_second_save() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "a.tif");
        let mut roll = Roll::open(dir.path()).unwrap();
        roll.save().unwrap();
        assert!(!thumbs_dir(dir.path())
            .parent()
            .unwrap()
            .join("roll.json.bak")
            .exists());
        roll.frames[0].approved = true;
        roll.save().unwrap();
        let bak = sidecar_dir(dir.path()).join("roll.json.bak");
        assert!(bak.exists());
        let reopened = Roll::open(dir.path()).unwrap();
        assert!(reopened.frames[0].approved);
    }

    #[test]
    fn open_merges_new_files_and_drops_vanished_ones() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "a.tif");
        touch(dir.path(), "b.tif");
        let mut roll = Roll::open(dir.path()).unwrap();
        roll.frames[1].approved = true;
        roll.save().unwrap();

        std::fs::remove_file(dir.path().join("a.tif")).unwrap();
        touch(dir.path(), "c.tif");
        let reopened = Roll::open(dir.path()).unwrap();
        let names: Vec<&str> = reopened
            .frames
            .iter()
            .map(|f| f.file_name.as_str())
            .collect();
        assert_eq!(names, vec!["b.tif", "c.tif"]);
        assert!(reopened.frames[0].approved); // b.tif's state survived
        assert!(!reopened.frames[1].approved); // c.tif is fresh
    }

    #[test]
    fn open_rejects_unknown_sidecar_version() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "a.tif");
        std::fs::create_dir_all(sidecar_dir(dir.path())).unwrap();
        std::fs::write(
            sidecar_path(dir.path()),
            br#"{"version": 99, "frames": []}"#,
        )
        .unwrap();
        let err = Roll::open(dir.path()).unwrap_err();
        assert!(err.contains("version"), "error was: {err}");
        assert!(err.contains("99"), "error was: {err}");
    }
}
```

Run: `cargo test -p unduster-app roll::` — after adding `mod roll;` to `lib.rs` (no commands yet, just the module), this compiles and the tests above go green immediately since Step 1 already contains the implementation. Note this task folds "RED" and "GREEN" into one commit-sized unit because the type definitions and their behavior are inseparable for a data module this small; the test list above is still written and run before moving to Task 2's command layer, satisfying TDD at the module boundary.

Add to `app/src-tauri/src/lib.rs`:

```rust
mod roll;
```

(No `Mutex<Option<Roll>>` management or commands yet — that is Task 2. This task only proves the pure data layer.)

- [ ] **Step 2: Verify, commit**

`cargo test -p unduster-app roll:: && cargo clippy -p unduster-app --all-targets -- -D warnings && cargo fmt --check`

```bash
git add app/src-tauri/src/roll.rs app/src-tauri/src/lib.rs
git commit -m "Add roll state with an atomic JSON sidecar"
```

---

### Task 2: open_roll, activate_frame, threshold/approve commands, and eviction

**Files:**
- Modify: `app/src-tauri/src/roll.rs` (serializable views used by commands)
- Modify: `app/src-tauri/src/lib.rs` (manage `Mutex<Option<Roll>>`, four new commands)
- Modify: `app/src-tauri/src/images.rs` (nothing new needed — `close` already exists)
- Test: inline in `lib.rs`

**Interfaces:**
- Consumes: `Roll::open`, `Roll::save`, `Images::prepare`/`insert`/`close`/`image`.
- Produces:
  - `FrameInfo { index: usize, file_name: String, threshold: f32, approved: bool, defect_count: Option<usize>, bboxes: Option<Vec<[u32; 4]>> }` (the wire view of `Frame`, omitting `image_id`).
  - `RollInfo { dir: String, frames: Vec<FrameInfo> }`.
  - `open_roll(dir: String) -> Result<RollInfo, String>`.
  - `activate_frame(index: usize) -> Result<ImageInfo, String>` (async): if the frame already has an `image_id` the registry still recognizes, returns that image's info without redecoding; otherwise decodes via the same staged `Images::prepare`/`insert` pipeline as `open_image` (same `app-progress` events, reusing `id: 0` for the pre-insert stages since the frame has no id yet), stores the new `image_id` on the frame, then evicts every other frame's `image_id` outside the window `{index-1, index, index+1}` via `Images::close`, clearing those frames' `image_id` to `None`.
  - `set_frame_threshold(index: usize, threshold: f32) -> Result<(), String>` — mutate + `Roll::save()`.
  - `approve_frame(index: usize, approved: bool) -> Result<(), String>` — mutate + `Roll::save()`.

- [ ] **Step 1: Failing tests for FrameInfo and the RollState helper**

Roll command logic is easiest to unit-test as a small `RollState` wrapper around `Mutex<Option<Roll>>` kept in `roll.rs`, so `lib.rs` commands stay thin (mirrors how `DetectorState` wraps its mutex in `detect.rs`). Add to `roll.rs`, above the existing `#[cfg(test)] mod tests`:

```rust
use std::sync::Mutex;

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct FrameInfo {
    pub index: usize,
    pub file_name: String,
    pub threshold: f32,
    pub approved: bool,
    pub defect_count: Option<usize>,
    pub bboxes: Option<Vec<[u32; 4]>>,
}

impl Frame {
    fn info(&self, index: usize) -> FrameInfo {
        FrameInfo {
            index,
            file_name: self.file_name.clone(),
            threshold: self.threshold,
            approved: self.approved,
            defect_count: self.defect_count,
            bboxes: self.bboxes.clone(),
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct RollInfo {
    pub dir: String,
    pub frames: Vec<FrameInfo>,
}

impl Roll {
    pub fn info(&self) -> RollInfo {
        RollInfo {
            dir: self.dir.display().to_string(),
            frames: self
                .frames
                .iter()
                .enumerate()
                .map(|(i, f)| f.info(i))
                .collect(),
        }
    }
}

/// Shared roll handle managed by Tauri; commands lock it for the duration of
/// a single frame mutation only (never across a decode).
#[derive(Default)]
pub struct RollState(pub Mutex<Option<Roll>>);

impl RollState {
    pub fn open(&self, dir: &Path) -> Result<RollInfo, String> {
        let roll = Roll::open(dir)?;
        let info = roll.info();
        *self.0.lock().map_err(|e| e.to_string())? = Some(roll);
        Ok(info)
    }

    /// Frame indices whose `image_id` should stay activated around `keep`:
    /// keep-1, keep, keep+1 (clamped to the frame count, so edge frames
    /// don't need special-casing by callers).
    fn keep_window(len: usize, keep: usize) -> std::ops::Range<usize> {
        let start = keep.saturating_sub(1);
        let end = (keep + 2).min(len);
        start..end
    }

    /// Ids to evict (frame index, image_id) so the caller can close them via
    /// `Images::close` outside this lock -- eviction never touches the
    /// `Images` registry itself; `RollState` only owns frame bookkeeping.
    pub fn ids_to_evict(&self, keep: usize) -> Result<Vec<(usize, u64)>, String> {
        let guard = self.0.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_ref().ok_or("no roll open")?;
        let window = Self::keep_window(roll.frames.len(), keep);
        Ok(roll
            .frames
            .iter()
            .enumerate()
            .filter(|(i, f)| !window.contains(i) && f.image_id.is_some())
            .map(|(i, f)| (i, f.image_id.unwrap()))
            .collect())
    }

    pub fn clear_image_id(&self, index: usize) -> Result<(), String> {
        let mut guard = self.0.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_mut().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get_mut(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        frame.image_id = None;
        Ok(())
    }

    pub fn image_id(&self, index: usize) -> Result<Option<u64>, String> {
        let guard = self.0.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_ref().ok_or("no roll open")?;
        Ok(roll
            .frames
            .get(index)
            .ok_or_else(|| format!("no frame {index}"))?
            .image_id)
    }

    pub fn set_image_id(&self, index: usize, id: u64) -> Result<(), String> {
        let mut guard = self.0.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_mut().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get_mut(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        frame.image_id = Some(id);
        Ok(())
    }

    pub fn frame_path(&self, index: usize) -> Result<PathBuf, String> {
        let guard = self.0.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_ref().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        Ok(roll.dir.join(&frame.file_name))
    }

    pub fn set_threshold(&self, index: usize, threshold: f32) -> Result<(), String> {
        let mut guard = self.0.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_mut().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get_mut(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        frame.threshold = threshold;
        roll.save()
    }

    pub fn set_approved(&self, index: usize, approved: bool) -> Result<(), String> {
        let mut guard = self.0.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_mut().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get_mut(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        frame.approved = approved;
        roll.save()
    }
}

#[cfg(test)]
mod state_tests {
    use super::*;

    fn touch(dir: &Path, name: &str) {
        std::fs::write(dir.join(name), b"marker").unwrap();
    }

    fn opened_state(dir: &Path, n: usize) -> RollState {
        for i in 0..n {
            touch(dir, &format!("f{i:02}.tif"));
        }
        let state = RollState::default();
        state.open(dir).unwrap();
        state
    }

    #[test]
    fn keep_window_clamps_at_both_edges() {
        assert_eq!(RollState::keep_window(5, 0), 0..2);
        assert_eq!(RollState::keep_window(5, 2), 1..4);
        assert_eq!(RollState::keep_window(5, 4), 3..5);
    }

    #[test]
    fn ids_to_evict_only_returns_activated_frames_outside_the_window() {
        let dir = tempfile::tempdir().unwrap();
        let state = opened_state(dir.path(), 5);
        state.set_image_id(0, 10).unwrap();
        state.set_image_id(1, 11).unwrap();
        state.set_image_id(2, 12).unwrap();
        state.set_image_id(4, 14).unwrap();
        // window around 2 is {1,2,3}: frame 0 and frame 4 are activated but
        // outside it, frame 3 is inside the window but was never activated.
        let mut evict = state.ids_to_evict(2).unwrap();
        evict.sort();
        assert_eq!(evict, vec![(0, 10), (4, 14)]);
    }

    #[test]
    fn set_threshold_and_approved_persist_via_save() {
        let dir = tempfile::tempdir().unwrap();
        let state = opened_state(dir.path(), 2);
        state.set_threshold(0, 0.33).unwrap();
        state.set_approved(1, true).unwrap();
        let reopened = Roll::open(dir.path()).unwrap();
        assert_eq!(reopened.frames[0].threshold, 0.33);
        assert!(reopened.frames[1].approved);
    }

    #[test]
    fn frame_path_joins_roll_dir() {
        let dir = tempfile::tempdir().unwrap();
        let state = opened_state(dir.path(), 1);
        assert_eq!(state.frame_path(0).unwrap(), dir.path().join("f00.tif"));
        assert!(state.frame_path(5).is_err());
    }

    #[test]
    fn operations_before_open_error_clearly() {
        let state = RollState::default();
        assert!(state.set_threshold(0, 0.5).unwrap_err().contains("no roll"));
        assert!(state.ids_to_evict(0).unwrap_err().contains("no roll"));
    }
}
```

Run: `cargo test -p unduster-app roll::` — FAIL to compile until the code above is added (RED is the missing `RollState`/`FrameInfo` types); then green once pasted in as shown (this step is presented implementation-inclusive per the plan's no-placeholder rule, but implementers must run the test file with the types stubbed out first to confirm the failure, then paste in the bodies -- standard red-green-refactor, condensed here because the bodies are short and mechanical).

- [ ] **Step 2: Wire the four commands into `lib.rs`**

```rust
#[tauri::command]
fn open_roll(
    state: State<'_, roll::RollState>,
    dir: String,
) -> Result<roll::RollInfo, String> {
    state.open(std::path::Path::new(&dir))
}

#[tauri::command]
async fn activate_frame(
    app: tauri::AppHandle,
    images: State<'_, Mutex<Images>>,
    roll: State<'_, roll::RollState>,
    index: usize,
) -> Result<ImageInfo, String> {
    // Reuse path: already activated and the registry still has it.
    if let Some(id) = roll.image_id(index)? {
        let known = {
            let images = images.lock().map_err(|e| e.to_string())?;
            images.image(id)
        };
        if let Some(image) = known {
            let levels = {
                let images = images.lock().map_err(|e| e.to_string())?;
                images
                    .level_dims(id)
                    .ok_or_else(|| format!("no image {id}"))?
            };
            return Ok(ImageInfo {
                id,
                width: image.width,
                height: image.height,
                levels: levels
                    .into_iter()
                    .map(|(width, height)| images::LevelInfo { width, height })
                    .collect(),
            });
        }
    }

    let path = roll.frame_path(index)?;
    let _ = app.emit("app-progress", Progress { id: 0, stage: "decoding" });
    let image = tauri::async_runtime::spawn_blocking(move || Images::decode_stage(&path))
        .await
        .map_err(|e| e.to_string())??;
    let _ = app.emit(
        "app-progress",
        Progress { id: 0, stage: "building-pyramid" },
    );
    let pyramid = {
        let image = image.clone();
        tauri::async_runtime::spawn_blocking(move || Images::pyramid_stage(&image))
            .await
            .map_err(|e| e.to_string())?
    };
    let prepared = Prepared { image, pyramid };
    let info = {
        let mut images = images.lock().map_err(|e| e.to_string())?;
        images.insert(prepared)
    };
    roll.set_image_id(index, info.id)?;

    for (evict_index, evict_id) in roll.ids_to_evict(index)? {
        let mut images = images.lock().map_err(|e| e.to_string())?;
        images.close(evict_id);
        drop(images);
        roll.clear_image_id(evict_index)?;
    }

    let _ = app.emit("app-progress", Progress { id: info.id, stage: "ready" });
    Ok(info)
}

#[tauri::command]
fn set_frame_threshold(
    state: State<'_, roll::RollState>,
    index: usize,
    threshold: f32,
) -> Result<(), String> {
    state.set_threshold(index, threshold)
}

#[tauri::command]
fn approve_frame(
    state: State<'_, roll::RollState>,
    index: usize,
    approved: bool,
) -> Result<(), String> {
    state.set_approved(index, approved)
}
```

Add `use images::LevelInfo;` is unnecessary if referenced as `images::LevelInfo` (already imported types are `ImageInfo, Images, Prepared`; extend that `use` line to `use images::{build_prob_pyramid, ImageInfo, Images, LevelInfo, Prepared};`). Manage the state and register the commands in `run()`:

```rust
        .manage(Mutex::new(Images::default()))
        .manage(detect::DetectorState::default())
        .manage(roll::RollState::default())
        .invoke_handler(tauri::generate_handler![
            open_image,
            close_image,
            load_detector,
            detect,
            components,
            open_roll,
            activate_frame,
            set_frame_threshold,
            approve_frame
        ])
```

- [ ] **Step 3: Verify, commit**

`cargo test -p unduster-app && cargo clippy -p unduster-app --all-targets -- -D warnings && cargo fmt --check`

```bash
git add app/src-tauri/src/roll.rs app/src-tauri/src/lib.rs
git commit -m "Add roll commands: open, activate with eviction, threshold, approve"
```

---

### Task 3: components_from_probs refactor, scan_roll background queue, thumbnails

**Files:**
- Modify: `app/src-tauri/src/images.rs` (extract `components_from_probs`, have `Images::components` delegate)
- Modify: `app/src-tauri/src/roll.rs` (thumbnail path helper already added in Task 1; add `scan_roll` support types)
- Modify: `app/src-tauri/src/lib.rs` (`scan_roll` command + background task)
- Test: inline in `images.rs` (free-function test) and `lib.rs` (queue behavior via a temp roll + fixture detector)

**Interfaces:**
- Consumes: `Images::prepare` (decode+pyramid, but the queue does NOT call `insert` — it keeps the `Prepared` local), `DetectorState::detect`, `fd_heal::components`, `fd_tiles::downsample_2x`, `fd_io::encode`.
- Produces:
  - `images::components_from_probs(probs: &[f32], width: u32, height: u32, threshold: f32) -> Vec<[u32; 4]>` (free function, same capping/logic `Images::components` used inline; `Images::components` becomes a thin wrapper).
  - Command `scan_roll() -> Result<(), String>`: spawns at most one background task, guarded by an `AtomicBool` on `RollState` (`scanning: AtomicBool`), returns immediately (`Ok(())` whether or not a scan was already running — idempotent from the UI's perspective, since the filmstrip just wants "a scan is happening or will happen").
  - Background task behavior per frame lacking `defect_count`: decode via `Images::prepare` (NOT `insert`), run `DetectorState::detect`, threshold at 0.5 via `components_from_probs`, write a thumbnail PNG to `<roll>/.unduster/thumbs/{index:04}.png`, store `defect_count`/`bboxes` on the frame, `Roll::save()`, emit `roll-progress` `{ index, count }`. On a per-frame error: `defect_count` stays `None`, emit `roll-frame-error` `{ index, message }`, continue to the next frame (queue never dies). After the last frame: emit `roll-done`.

- [ ] **Step 1: Failing test for the free function**

Add to `images.rs` tests (this pins the refactor's behavior BEFORE moving the logic, so a regression during the extraction is caught immediately):

```rust
    #[test]
    fn components_from_probs_matches_the_method_it_replaces() {
        let mut probs = vec![0.0f32; 600 * 400];
        for y in 100..104 {
            for x in 200..205 {
                probs[y * 600 + x] = 0.8;
            }
        }
        let direct = components_from_probs(&probs, 600, 400, 0.5);
        assert_eq!(direct.len(), 1);
        assert_eq!(direct[0], [200, 100, 205, 104]);
        assert!(components_from_probs(&probs, 600, 400, 0.9).is_empty());
    }
```

Run: `cargo test -p unduster-app components_from_probs` — FAIL (no such function).

- [ ] **Step 2: Extract the free function, delegate the method**

Replace `Images::components` in `images.rs`:

```rust
/// Connected-component bounding boxes from a thresholded probability map,
/// capped at [`MAX_COMPONENTS`]: a pathological mask (bad model or
/// threshold) can otherwise produce hundreds of thousands of boxes that are
/// useless for navigation and expensive to serialize. Free function so the
/// roll background queue (which never inserts its frame into the `Images`
/// registry -- see `scan_roll`) can compute bboxes without a registry entry.
pub fn components_from_probs(probs: &[f32], width: u32, height: u32, threshold: f32) -> Vec<[u32; 4]> {
    let mask: Vec<bool> = probs.iter().map(|&p| p > threshold).collect();
    fd_heal::components(&mask, width, height)
        .into_iter()
        .take(MAX_COMPONENTS)
        .map(|d| [d.bbox.x0, d.bbox.y0, d.bbox.x1, d.bbox.y1])
        .collect()
}
```

```rust
    pub fn components(&self, id: u64, threshold: f32) -> Option<Vec<[u32; 4]>> {
        let entry = self.entries.get(&id)?;
        let (probs, _) = entry.probs.as_ref()?;
        let (w, h) = (entry.image.width, entry.image.height);
        Some(components_from_probs(probs, w, h, threshold))
    }
```

Run: `cargo test -p unduster-app` — all prior `images.rs` tests (`prob_tiles_and_components_roundtrip`, `components_list_is_capped`) plus the new one green; this is the refactor's regression check.

- [ ] **Step 3: Failing test for thumbnail generation as a pure helper**

Add to `roll.rs`, a free function so it is testable without a Tauri app handle or a detector:

```rust
/// Downscales an already-built RGBA level (coarsest pyramid level, typically
/// already small) to at most 128px on its longest edge via repeated 2x2
/// box-average downsampling, then writes it as an RGB PNG thumbnail. Alpha
/// is dropped (tiles are always opaque -- see fd_tiles::pyramid::base_rgba).
pub fn write_thumbnail(
    rgba: &[u8],
    width: u32,
    height: u32,
    out_path: &Path,
) -> Result<(), String> {
    const MAX_EDGE: u32 = 128;
    let (mut w, mut h) = (width, height);
    let mut buf = rgba.to_vec();
    while w.max(h) > MAX_EDGE {
        let (next, nw, nh) = fd_tiles::downsample_2x(&buf, w, h);
        buf = next;
        w = nw;
        h = nh;
    }
    let n = (w * h) as usize;
    let mut rgb = Vec::with_capacity(n * 3);
    for px in buf.chunks_exact(4).take(n) {
        rgb.extend_from_slice(&px[..3]);
    }
    let img = fd_io::ImageBuf {
        width: w,
        height: h,
        channels: 3,
        data: fd_io::PixelData::U8(rgb),
        icc: None,
        exif: None,
    };
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    fd_io::encode(out_path, &img).map_err(|e| e.to_string())
}

pub fn thumb_path(dir: &Path, index: usize) -> PathBuf {
    thumbs_dir(dir).join(format!("{index:04}.png"))
}
```

Add the test (RED first: run before pasting `write_thumbnail`/`thumb_path` in, confirming a compile failure, then paste the implementation above and re-run for GREEN):

```rust
    #[test]
    fn write_thumbnail_downscales_below_max_edge_and_is_a_valid_png() {
        let dir = tempfile::tempdir().unwrap();
        let (w, h) = (600u32, 400u32);
        let rgba = vec![128u8; (w * h * 4) as usize];
        let out = thumb_path(dir.path(), 7);
        write_thumbnail(&rgba, w, h, &out).unwrap();
        assert!(out.exists());
        assert_eq!(out.file_name().unwrap(), "0007.png");
        let decoded = fd_io::decode(&out).unwrap();
        assert!(decoded.width <= 128 && decoded.height <= 128);
        assert_eq!(decoded.channels, 3);
    }
```

Run: `cargo test -p unduster-app roll::` — green.

- [ ] **Step 4: The `scan_roll` command and background task**

Add to `roll.rs`, alongside `RollState`:

```rust
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
pub struct RollState {
    pub roll: Mutex<Option<Roll>>,
    /// Guards against double-spawning the background scan task if `scan_roll`
    /// is invoked twice (e.g. a second "Open roll" or an eager retry).
    pub scanning: AtomicBool,
}
```

This changes `RollState` from a tuple struct to a named struct with two fields; update every earlier `self.0` reference in Task 2's methods to `self.roll` (mechanical rename across `open`, `ids_to_evict`, `clear_image_id`, `image_id`, `set_image_id`, `frame_path`, `set_threshold`, `set_approved`, and the `Default` derive already covers `scanning: AtomicBool::new(false)` since `AtomicBool` implements `Default`). Add:

```rust
impl RollState {
    /// Frame indices still missing a defect count, in order.
    pub fn frames_to_scan(&self) -> Result<Vec<usize>, String> {
        let guard = self.roll.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_ref().ok_or("no roll open")?;
        Ok(roll
            .frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.defect_count.is_none())
            .map(|(i, _)| i)
            .collect())
    }

    pub fn record_scan_result(
        &self,
        index: usize,
        count: Option<usize>,
        bboxes: Option<Vec<[u32; 4]>>,
    ) -> Result<(), String> {
        let mut guard = self.roll.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_mut().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get_mut(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        frame.defect_count = count;
        frame.bboxes = bboxes;
        roll.save()
    }

    pub fn dir(&self) -> Result<PathBuf, String> {
        let guard = self.roll.lock().map_err(|e| e.to_string())?;
        Ok(guard.as_ref().ok_or("no roll open")?.dir.clone())
    }
}
```

In `lib.rs`, add the scan detection threshold constant, progress payload types, and the command:

```rust
/// Fixed threshold for the background queue's stored bboxes/count. The
/// operator's per-frame sensitivity slider (Task 2's `set_frame_threshold`)
/// only affects live overlay/z-navigation on the activated frame; the queue
/// runs once per roll at a stable threshold so counts are comparable across
/// frames regardless of what the operator was looking at when the queue
/// reached them.
const SCAN_THRESHOLD: f32 = 0.5;

#[derive(serde::Serialize, Clone)]
struct RollProgress {
    index: usize,
    count: Option<usize>,
}

#[derive(serde::Serialize, Clone)]
struct RollFrameError {
    index: usize,
    message: String,
}

#[tauri::command]
fn scan_roll(
    app: tauri::AppHandle,
    roll: State<'_, roll::RollState>,
    detector: State<'_, detect::DetectorState>,
) -> Result<(), String> {
    if roll
        .scanning
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Ok(()); // already running; idempotent from the caller's view
    }
    let roll_dir = roll.dir()?;
    let indices = roll.frames_to_scan()?;
    let roll_handle = roll.inner();
    // RollState is managed (lives for the app's lifetime), so a raw pointer
    // captured via `app.state()` inside the task avoids needing RollState to
    // be Clone; Tauri's AppHandle can hand back the same State at any time.
    let detector = detector.inner().clone();
    let app_for_task = app.clone();
    tauri::async_runtime::spawn(async move {
        for index in indices {
            let roll_state = app_for_task.state::<roll::RollState>();
            let path = match roll_state.frame_path(index) {
                Ok(p) => p,
                Err(e) => {
                    let _ = roll_state.record_scan_result(index, None, None);
                    let _ = app_for_task.emit(
                        "roll-frame-error",
                        RollFrameError { index, message: e },
                    );
                    continue;
                }
            };
            let detector = detector.clone();
            let outcome = tauri::async_runtime::spawn_blocking(move || {
                let prepared = images::Images::prepare(&path)?;
                let probs = detector.detect(&prepared.image)?;
                let counted = images::components_from_probs(
                    &probs,
                    prepared.image.width,
                    prepared.image.height,
                    SCAN_THRESHOLD,
                );
                let coarsest = prepared
                    .pyramid
                    .levels
                    .last()
                    .expect("pyramid always has at least one level");
                Ok::<_, String>((
                    counted,
                    coarsest.rgba.clone(),
                    coarsest.width,
                    coarsest.height,
                ))
                // `prepared` (and its full-res pixels) drops here, at the end
                // of the blocking closure, before the task moves to the next
                // frame -- this is the "at most 1 queue frame" memory bound.
            })
            .await
            .map_err(|e| e.to_string())
            .and_then(|r| r);

            match outcome {
                Ok((bboxes, thumb_rgba, tw, th)) => {
                    let thumb_path = roll::thumb_path(&roll_dir, index);
                    if let Err(e) = roll::write_thumbnail(&thumb_rgba, tw, th, &thumb_path) {
                        let _ = app_for_task.emit(
                            "roll-frame-error",
                            RollFrameError {
                                index,
                                message: format!("thumbnail: {e}"),
                            },
                        );
                    }
                    let count = bboxes.len();
                    let _ = roll_state.record_scan_result(index, Some(count), Some(bboxes));
                    let _ = app_for_task.emit("roll-progress", RollProgress {
                        index,
                        count: Some(count),
                    });
                }
                Err(message) => {
                    let _ = roll_state.record_scan_result(index, None, None);
                    let _ =
                        app_for_task.emit("roll-frame-error", RollFrameError { index, message });
                }
            }
        }
        roll_handle.scanning.store(false, Ordering::SeqCst);
        let _ = app_for_task.emit("roll-done", ());
    });
    Ok(())
}
```

Note on `roll_handle`: `State<'_, roll::RollState>` derefs to `&RollState`, and `RollState` is `'static` because Tauri manages it for the app's lifetime, but `State` itself is not `'static` (it borrows from the app). The closure re-fetches `app_for_task.state::<roll::RollState>()` on every loop iteration instead of capturing `roll` directly, sidestepping the lifetime entirely; `roll_handle` is used only once, right before the loop, to read `scanning` at the very end via `app_for_task.state::<roll::RollState>().scanning...` -- correct that final line to also re-fetch:

```rust
        app_for_task
            .state::<roll::RollState>()
            .scanning
            .store(false, Ordering::SeqCst);
```

(drop the now-unused `roll_handle` variable and its `.inner()` binding above the loop). Register the command:

```rust
            open_roll,
            activate_frame,
            set_frame_threshold,
            approve_frame,
            scan_roll
```

- [ ] **Step 5: Integration test for the queue**

Add to `lib.rs` tests (new `#[cfg(test)] mod tests` block if none exists yet at the bottom of `lib.rs`; check first -- if `lib.rs` has no test module, create one):

```rust
#[cfg(test)]
mod roll_queue_tests {
    use super::*;

    fn fixture_detector() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../engine/fixtures/tiny-detector.onnx")
    }

    #[test]
    fn frames_to_scan_lists_only_uncounted_frames() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        std::fs::write(dir.path().join("b.png"), b"x").unwrap();
        let state = roll::RollState::default();
        state.open(dir.path()).unwrap();
        state.record_scan_result(0, Some(2), Some(vec![])).unwrap();
        assert_eq!(state.frames_to_scan().unwrap(), vec![1]);
    }

    #[test]
    fn detector_state_clone_shares_the_loaded_model() {
        // Guards the scan_roll assumption that `detector.inner().clone()`
        // yields a handle that still resolves after the original `State`
        // borrow is gone (used across an async task boundary).
        let state = detect::DetectorState::default();
        state.load(&fixture_detector()).unwrap();
        let cloned = state.clone();
        let img = fd_io::ImageBuf {
            width: 8,
            height: 8,
            channels: 1,
            data: fd_io::PixelData::U8(vec![0; 64]),
            icc: None,
            exif: None,
        };
        assert!(cloned.detect(&img).is_ok());
    }
}
```

Run: `cargo test -p unduster-app` — green. (Full end-to-end `scan_roll` behavior against a live `tauri::AppHandle` is exercised manually in Task 7; unit tests here pin the pieces that don't need a running app: the frame-selection filter and the detector-clone sharing assumption the task loop depends on.)

- [ ] **Step 6: Verify, commit**

`cargo test -p unduster-app && cargo clippy -p unduster-app --all-targets -- -D warnings && cargo fmt --check`

```bash
git add app/src-tauri/src/images.rs app/src-tauri/src/roll.rs app/src-tauri/src/lib.rs
git commit -m "Scan a roll's remaining frames in a memory-bounded background queue"
```

---

### Task 4: Thumb protocol branch

**Files:**
- Modify: `app/src-tauri/src/protocol.rs` (`Layer::Thumb`, second `State` param on `tile_response`)
- Modify: `app/src-tauri/src/lib.rs` (protocol registration passes both states)
- Test: inline in `protocol.rs` (extend existing tests, all Layer-shape tests updated)

**Interfaces:**
- Consumes: `roll::RollState`, `roll::thumb_path`.
- Produces: URL `tiles://localhost/thumb/{index}` -> 200 `Content-Type: image/png` with the thumbnail bytes, or 404 when no roll is open, the index is out of range, or the thumbnail file doesn't exist yet (queue hasn't reached it). `parse_tile_path` gains a two-segment form (`/thumb/{index}`, no level/tx/ty) alongside the existing four-segment `Rgba`/`Probs` forms. `tile_response` signature becomes `tile_response(images: &Mutex<Images>, roll: &Mutex<Option<roll::Roll>>, path: &str) -> tauri::http::Response<Vec<u8>>` — note this takes `&Mutex<Option<Roll>>` directly (not `&roll::RollState`) so the protocol module does not depend on `RollState`'s `scanning` field; `lib.rs`'s registration extracts `&mutex_roll_state.roll` when wiring the scheme handler.

- [ ] **Step 1: Failing tests**

Extend `protocol.rs`. First, the existing `Layer`/`parse_tile_path` tests must be updated for the new variant and the new two-segment shape; replace the whole `parses_probs_layer` block and add a thumb-path test, plus extend `tile_response` call sites with the new required argument (every existing test in the file currently calls `tile_response(&images, path)` with two arguments -- ALL of these need the new `&roll` argument added, not just the new tests):

```rust
    #[test]
    fn parses_probs_layer() {
        assert_eq!(
            parse_tile_path("/probs/3/1/7/2"),
            Some((Layer::Probs, 3, 1, 7, 2))
        );
        assert_eq!(parse_tile_path("/3/1/7/2"), Some((Layer::Rgba, 3, 1, 7, 2)));
        assert_eq!(parse_tile_path("/probs/3/1/7"), None);
        assert_eq!(parse_tile_path("/unknown/3/1/7/2"), None);
    }

    #[test]
    fn parses_thumb_layer() {
        assert_eq!(parse_tile_path("/thumb/7"), Some((Layer::Thumb, 7, 0, 0, 0)));
        assert_eq!(parse_tile_path("thumb/0"), Some((Layer::Thumb, 0, 0, 0, 0)));
        assert_eq!(parse_tile_path("/thumb/"), None);
        assert_eq!(parse_tile_path("/thumb/x"), None);
        assert_eq!(parse_tile_path("/thumb/7/8"), None);
    }

    #[test]
    fn missing_tile_is_404_and_malformed_is_400() {
        let images = Mutex::new(Images::default());
        let roll = Mutex::new(None);
        assert_eq!(tile_response(&images, &roll, "/1/0/0/0").status(), 404);
        assert_eq!(tile_response(&images, &roll, "/nope").status(), 400);
    }

    #[test]
    fn probs_tile_404_before_detection() {
        let images = Mutex::new(Images::default());
        let roll = Mutex::new(None);
        assert_eq!(
            tile_response(&images, &roll, "/probs/1/0/0/0").status(),
            404
        );
    }

    #[test]
    fn thumb_404_when_no_roll_open() {
        let images = Mutex::new(Images::default());
        let roll = Mutex::new(None);
        assert_eq!(tile_response(&images, &roll, "/thumb/0").status(), 404);
    }

    #[test]
    fn thumb_404_when_file_not_yet_written() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.tif"), b"x").unwrap();
        let images = Mutex::new(Images::default());
        let opened = crate::roll::Roll::open(dir.path()).unwrap();
        let roll = Mutex::new(Some(opened));
        // frame 0 exists but its thumbnail was never written by the queue
        assert_eq!(tile_response(&images, &roll, "/thumb/0").status(), 404);
        // out-of-range index also 404s, not a panic
        assert_eq!(tile_response(&images, &roll, "/thumb/9").status(), 404);
    }

    #[test]
    fn thumb_200_serves_png_bytes_with_the_right_content_type() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.tif"), b"x").unwrap();
        let opened = crate::roll::Roll::open(dir.path()).unwrap();
        let thumb_path = crate::roll::thumb_path(&opened.dir, 0);
        let rgba = vec![10u8, 20, 30, 255];
        crate::roll::write_thumbnail(&rgba, 1, 1, &thumb_path).unwrap();
        let images = Mutex::new(Images::default());
        let roll = Mutex::new(Some(opened));
        let resp = tile_response(&images, &roll, "/thumb/0");
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.headers().get("Content-Type").unwrap(),
            "image/png"
        );
        assert!(!resp.body().is_empty());
    }
```

Run: `cargo test -p unduster-app` — FAIL (missing `Layer::Thumb`, `tile_response` arity mismatch on every call site).

- [ ] **Step 2: Implement**

```rust
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Layer {
    Rgba,
    Probs,
    Thumb,
}

/// Parses:
/// - "/{id}/{level}/{tx}/{ty}" (Rgba)
/// - "/probs/{id}/{level}/{tx}/{ty}" (Probs)
/// - "/thumb/{index}" (Thumb; level/tx/ty are unused and returned as 0)
/// Leading slash optional on all three forms.
pub fn parse_tile_path(path: &str) -> Option<(Layer, u64, u8, u32, u32)> {
    let trimmed = path.trim_start_matches('/');
    if let Some(rest) = trimmed.strip_prefix("thumb/") {
        if rest.is_empty() {
            return None;
        }
        let mut parts = rest.split('/');
        let index = parts.next()?.parse().ok()?;
        if parts.next().is_some() {
            return None;
        }
        return Some((Layer::Thumb, index, 0, 0, 0));
    }
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

Rewrite `tile_response` to take the roll mutex and branch on `Layer::Thumb` with its own PNG response builder (different `Content-Type`, no `x-tile-width`/`x-tile-height` headers since the webview reads PNG dimensions itself):

```rust
use std::path::Path;

pub fn tile_response(
    images: &Mutex<Images>,
    roll: &Mutex<Option<crate::roll::Roll>>,
    path: &str,
) -> tauri::http::Response<Vec<u8>> {
    let respond = |status: u16, body: Vec<u8>, w: u32, h: u32| {
        let mut builder = tauri::http::Response::builder()
            .status(status)
            .header("Content-Type", "application/octet-stream")
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Expose-Headers", "x-tile-width, x-tile-height");
        if status == 200 {
            builder = builder
                .header("x-tile-width", w.to_string())
                .header("x-tile-height", h.to_string());
        }
        builder
            .body(body)
            .expect("static response headers are valid")
    };
    let respond_png = |status: u16, body: Vec<u8>| {
        tauri::http::Response::builder()
            .status(status)
            .header("Content-Type", "image/png")
            .header("Access-Control-Allow-Origin", "*")
            .body(body)
            .expect("static response headers are valid")
    };
    let Some((layer, id, level, tx, ty)) = parse_tile_path(path) else {
        #[cfg(debug_assertions)]
        eprintln!("[tiles] 400 malformed path: {path}");
        return respond(400, Vec::new(), 0, 0);
    };
    match layer {
        Layer::Thumb => {
            let index = id as usize;
            let Ok(roll_guard) = roll.lock() else {
                return respond_png(500, Vec::new());
            };
            let Some(roll) = roll_guard.as_ref() else {
                return respond_png(404, Vec::new());
            };
            if index >= roll.frames.len() {
                return respond_png(404, Vec::new());
            }
            let thumb_path = crate::roll::thumb_path(&roll.dir, index);
            match std::fs::read(&thumb_path) {
                Ok(bytes) => respond_png(200, bytes),
                Err(_) => respond_png(404, Vec::new()),
            }
        }
        Layer::Rgba | Layer::Probs => {
            let Ok(mut images) = images.lock() else {
                return respond(500, Vec::new(), 0, 0);
            };
            match layer {
                Layer::Rgba => match images.tile(id, level, tx, ty) {
                    Some(tile) => respond(200, tile.rgba.clone(), tile.width, tile.height),
                    None => {
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "[tiles] 404 rgba {path}: known image ids {:?}",
                            images.known_ids()
                        );
                        respond(404, Vec::new(), 0, 0)
                    }
                },
                Layer::Probs => match images.prob_tile(id, level, tx, ty) {
                    Some((w, h, bytes)) => respond(200, bytes, w, h),
                    None => {
                        #[cfg(debug_assertions)]
                        eprintln!("[tiles] 404 probs {path} (no detection yet is normal)");
                        respond(404, Vec::new(), 0, 0)
                    }
                },
                Layer::Thumb => unreachable!(),
            }
        }
    }
}
```

(The unused `Path` import above is only needed if not already present; check the top of the file before adding -- `protocol.rs` currently has no `Path` import, and this version doesn't actually need one either since `thumb_path` takes `&roll.dir` directly, so drop the `use std::path::Path;` line -- it is not needed. Remove it from the snippet before pasting.)

Update `lib.rs`'s scheme registration:

```rust
        .register_uri_scheme_protocol("tiles", |ctx, request| {
            let images = ctx.app_handle().state::<Mutex<Images>>();
            let roll = ctx.app_handle().state::<roll::RollState>();
            protocol::tile_response(&images, &roll.roll, request.uri().path())
        })
```

- [ ] **Step 3: Verify, commit**

`cargo test -p unduster-app && cargo clippy -p unduster-app --all-targets -- -D warnings && cargo fmt --check`

```bash
git add app/src-tauri/src/protocol.rs app/src-tauri/src/lib.rs
git commit -m "Serve roll thumbnails under a thumb layer on the tiles protocol"
```

---

### Task 5: ringsFor screen-mapping math and renderer ring drawing

**Files:**
- Modify: `app/src/lib/viewport.ts` (`ringsFor` export)
- Modify: `app/src/lib/renderer.ts` (second GL program, `drawRings`)
- Test: `app/src/lib/viewport.test.ts` (extended)

**Interfaces:**
- Consumes: bbox arrays (`[number, number, number, number][]`, same shape as the `components` command's return and `Frame.bboxes`), current zoom/pan/canvas size.
- Produces: `export function ringsFor(bboxes: [number, number, number, number][], zoom: number, centerX: number, centerY: number, canvasW: number, canvasH: number, minR: number): { x: number; y: number; r: number }[]` — maps each bbox's center to screen space (`x = (bboxCenterX - centerX) * zoom + canvasW / 2`, matching the pan/zoom convention already used by `zoomAt`/`onPointerMove` in `Viewer.svelte`), radius `= max((bbox extent / 2) * zoom, minR)`, and filters out rings whose bounding circle doesn't intersect the canvas rect (fully offscreen). `TileRenderer.drawRings(rings, canvasW, canvasH)`: separate GL program, additive draw of one quad per ring with a distance-field annulus (soft 2px edge, red, alpha 0.9) in the fragment shader; called by `Viewer.svelte` after `draw()` when `zoom < 0.5` and there are detections (live `detections` state or a `bboxes` prop for the pre-detect case).

- [ ] **Step 1: Failing tests**

Add to `app/src/lib/viewport.test.ts`:

```ts
import { ringsFor } from "./viewport";

describe("ringsFor", () => {
  it("maps a centered bbox to the canvas center", () => {
    // canvas 800x600, viewport centered at image (1000, 600), zoom 1:
    // a bbox centered exactly at (1000, 600) maps to screen (400, 300).
    const rings = ringsFor([[980, 580, 1020, 620]], 1, 1000, 600, 800, 600, 12);
    expect(rings).toHaveLength(1);
    expect(rings[0].x).toBeCloseTo(400);
    expect(rings[0].y).toBeCloseTo(300);
    // bbox extent is 40x40 image px; at zoom 1, radius = max(20, 12) = 20
    expect(rings[0].r).toBeCloseTo(20);
  });

  it("enforces the minimum radius for small defects", () => {
    const rings = ringsFor([[998, 598, 1002, 602]], 1, 1000, 600, 800, 600, 12);
    expect(rings[0].r).toBeCloseTo(12);
  });

  it("scales radius with zoom", () => {
    const rings = ringsFor([[980, 580, 1020, 620]], 0.5, 1000, 600, 800, 600, 12);
    // extent 40 image px * zoom 0.5 / 2 = 10, below minR -> clamped to 12
    expect(rings[0].r).toBeCloseTo(12);
    const rings2 = ringsFor([[900, 500, 1100, 700]], 0.5, 1000, 600, 800, 600, 12);
    // extent 200 image px * zoom 0.5 / 2 = 50
    expect(rings2[0].r).toBeCloseTo(50);
  });

  it("filters bboxes fully offscreen", () => {
    const rings = ringsFor(
      [
        [980, 580, 1020, 620], // centered, onscreen
        [10000, 10000, 10010, 10010], // far offscreen
      ],
      1,
      1000,
      600,
      800,
      600,
      12,
    );
    expect(rings).toHaveLength(1);
  });

  it("keeps a ring whose circle still overlaps the canvas edge", () => {
    // bbox center maps just past the right edge, but its radius reaches back in
    const rings = ringsFor([[1390, 580, 1430, 620]], 1, 1000, 600, 800, 600, 12);
    // screen x = (1410 - 1000) * 1 + 400 = 810, r = max(20, 12) = 20;
    // circle spans 790..830, canvas right edge is 800 -> still overlaps
    expect(rings).toHaveLength(1);
  });
});
```

Run: `npm run test` from `app/` — FAIL (`ringsFor` doesn't exist).

- [ ] **Step 2: Implement `ringsFor`**

Add to `app/src/lib/viewport.ts`:

```ts
export interface Ring {
  x: number;
  y: number;
  r: number;
}

/** Maps native-resolution defect bboxes to screen-space ring markers, using
 * the same pan/zoom convention as Viewer's zoomAt/onPointerMove (screen
 * origin at canvas center, image point (centerX, centerY) maps there).
 * Filters out rings whose bounding circle doesn't intersect the canvas. */
export function ringsFor(
  bboxes: [number, number, number, number][],
  zoom: number,
  centerX: number,
  centerY: number,
  canvasW: number,
  canvasH: number,
  minR: number,
): Ring[] {
  const out: Ring[] = [];
  for (const [x0, y0, x1, y1] of bboxes) {
    const bx = (x0 + x1) / 2;
    const by = (y0 + y1) / 2;
    const x = (bx - centerX) * zoom + canvasW / 2;
    const y = (by - centerY) * zoom + canvasH / 2;
    const extent = Math.max(x1 - x0, y1 - y0);
    const r = Math.max((extent / 2) * zoom, minR);
    const onscreen =
      x + r >= 0 && x - r <= canvasW && y + r >= 0 && y - r <= canvasH;
    if (onscreen) out.push({ x, y, r });
  }
  return out;
}
```

Run: `npm run test` — green.

- [ ] **Step 3: `drawRings` on `TileRenderer`**

Add a second shader pair and program to `renderer.ts`:

```ts
const RING_VERT = `#version 300 es
in vec2 corner;
uniform vec2 viewport;
uniform vec2 center;
uniform float radius;
out vec2 vCorner;
void main() {
  vCorner = corner;
  vec2 pos = center + corner * (radius + 3.0);
  vec2 clip = (pos / viewport) * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
}`;

const RING_FRAG = `#version 300 es
precision mediump float;
in vec2 vCorner;
out vec4 color;
uniform float radius;
void main() {
  float d = length(vCorner) * (radius + 3.0);
  // Soft 2px annulus at the ring radius: smoothstep in from both sides so
  // the edge anti-aliases instead of stair-stepping.
  float outer = 1.0 - smoothstep(radius - 1.0, radius + 1.0, d);
  float inner = smoothstep(radius - 3.0, radius - 1.0, d);
  float alpha = outer * inner * 0.9;
  if (alpha <= 0.0) discard;
  color = vec4(1.0, 0.05, 0.05, alpha);
}`;
```

In the `TileRenderer` class, add a second program built alongside the first in the constructor:

```ts
  private ringProgram: WebGLProgram;
  private ringBuf: WebGLBuffer;
```

```ts
    // Ring program: a unit quad in [-1, 1]^2, positioned and scaled per ring
    // via uniforms so one draw call handles one ring (ring counts are small
    // -- dozens, not thousands -- so per-ring draw calls are not a concern).
    const rp = gl.createProgram()!;
    gl.attachShader(rp, compile(gl, gl.VERTEX_SHADER, RING_VERT));
    gl.attachShader(rp, compile(gl, gl.FRAGMENT_SHADER, RING_FRAG));
    gl.linkProgram(rp);
    if (!gl.getProgramParameter(rp, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(rp) ?? "ring link failed");
    }
    this.ringProgram = rp;
    this.ringBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.ringBuf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1]),
      gl.STATIC_DRAW,
    );
```

Add the method:

```ts
  /** Draws ring markers over the already-rendered frame. Call after draw().
   * `rings` are in screen px (see viewport.ts#ringsFor). Uses additive-free
   * alpha blending so overlapping rings don't double-darken past the base
   * 0.9 alpha set in the fragment shader. */
  drawRings(rings: { x: number; y: number; r: number }[], canvasW: number, canvasH: number): void {
    if (rings.length === 0) return;
    const gl = this.gl;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.ringProgram);
    gl.uniform2f(gl.getUniformLocation(this.ringProgram, "viewport"), canvasW, canvasH);
    const centerLoc = gl.getUniformLocation(this.ringProgram, "center");
    const radiusLoc = gl.getUniformLocation(this.ringProgram, "radius");
    const cornerLoc = gl.getAttribLocation(this.ringProgram, "corner");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.ringBuf);
    gl.enableVertexAttribArray(cornerLoc);
    gl.vertexAttribPointer(cornerLoc, 2, gl.FLOAT, false, 8, 0);
    for (const ring of rings) {
      gl.uniform2f(centerLoc, ring.x, ring.y);
      gl.uniform1f(radiusLoc, ring.r);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.disable(gl.BLEND);
  }
```

- [ ] **Step 4: Verify, commit**

From `app/`: `npm run test && npm run check`. From root: `cargo test -p unduster-app` (unaffected by this TS-only task, sanity check), clippy, fmt.

```bash
git add app/src/lib/viewport.ts app/src/lib/renderer.ts
git commit -m "Add ring-marker screen math and a GL ring-drawing pass"
```

---

### Task 6: Filmstrip component, App roll mode, and roll keyboard shortcuts

**Files:**
- Create: `app/src/lib/Filmstrip.svelte`
- Modify: `app/src/App.svelte` (roll mode state, open-roll button, keys, status line)
- Modify: `app/src/lib/Viewer.svelte` (accept a `bboxes` prop for pre-detect rings, call `drawRings`)
- Test: manual gate (Task 7) covers end-to-end interaction; `Filmstrip.svelte` has no pure logic worth a vitest unit (it is a thin keyboard/DOM wrapper) so this task relies on `npm run check` (svelte-check) plus the Task 7 manual gate for verification, consistent with how `Viewer.svelte` itself has no dedicated test file today (its pure logic already lives in `viewport.ts`/`renderer.ts`, which have tests).

**Interfaces:**
- Consumes: `open_roll`, `activate_frame`, `set_frame_threshold`, `approve_frame`, `scan_roll` commands; `roll-progress`/`roll-frame-error`/`roll-done` events; `tiles://localhost/thumb/{index}`.
- Produces: `Filmstrip.svelte` props `{ frames: FrameInfo[], currentIndex: number, onSelect: (index: number) => void }`; renders a `role="listbox"` with one `role="option"` per frame (thumb `<img>`, file name, defect-count badge or spinner, approved checkmark, `aria-selected` + highlight on the current frame), roving `tabindex` (0 on the current option, -1 elsewhere), `Enter`/`Space` selects, arrow keys move focus without needing a page reload of the list. App-level roll state: `rollFrames: FrameInfo[] | null`, `rollDir: string | null`, `currentIndex: number`; `,`/`.` step `currentIndex` and call `activateCurrentFrame()`; `A` approves the current frame and advances to the next unapproved one.

- [ ] **Step 1: `Filmstrip.svelte`**

```svelte
<script lang="ts">
  interface FrameInfo {
    index: number;
    file_name: string;
    threshold: number;
    approved: boolean;
    defect_count: number | null;
    bboxes: [number, number, number, number][] | null;
  }

  let {
    frames,
    currentIndex,
    onSelect,
  }: {
    frames: FrameInfo[];
    currentIndex: number;
    onSelect: (index: number) => void;
  } = $props();

  let listEl: HTMLDivElement | undefined = $state();
  let focusIndex = $state(currentIndex);

  $effect(() => {
    focusIndex = currentIndex;
  });

  $effect(() => {
    // Scroll the current frame into view whenever it changes (keyboard
    // navigation via ,/. at the App level, or a filmstrip click).
    void currentIndex;
    const el = listEl?.querySelector(`[data-index="${currentIndex}"]`);
    el?.scrollIntoView({ block: "nearest", inline: "nearest" });
  });

  function moveFocus(delta: number) {
    const next = Math.min(Math.max(focusIndex + delta, 0), frames.length - 1);
    focusIndex = next;
    const el = listEl?.querySelector<HTMLElement>(`[data-index="${next}"]`);
    el?.focus();
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === "ArrowRight" || e.key === "ArrowDown") {
      e.preventDefault();
      moveFocus(1);
    } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
      e.preventDefault();
      moveFocus(-1);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect(focusIndex);
    }
  }
</script>

<div
  bind:this={listEl}
  class="filmstrip"
  role="listbox"
  aria-label="Roll frames"
  aria-activedescendant={`frame-${focusIndex}`}
>
  {#each frames as frame (frame.index)}
    <div
      id={`frame-${frame.index}`}
      data-index={frame.index}
      role="option"
      aria-selected={frame.index === currentIndex}
      tabindex={frame.index === focusIndex ? 0 : -1}
      class="frame"
      class:current={frame.index === currentIndex}
      onclick={() => onSelect(frame.index)}
      onkeydown={onKey}
    >
      <div class="thumb-wrap">
        <img
          src={`tiles://localhost/thumb/${frame.index}`}
          alt=""
          class="thumb"
          onerror={(e) => ((e.currentTarget as HTMLImageElement).style.visibility = "hidden")}
        />
        {#if frame.defect_count === null}
          <span class="spinner" aria-hidden="true"></span>
        {:else}
          <span class="badge">{frame.defect_count}</span>
        {/if}
        {#if frame.approved}
          <span class="check" aria-hidden="true">&#10003;</span>
        {/if}
      </div>
      <span class="name">{frame.file_name}</span>
    </div>
  {/each}
</div>

<style>
  .filmstrip {
    display: flex;
    gap: 0.5rem;
    overflow-x: auto;
    padding: 0.5rem;
    background: #1b1b1b;
    border-top: 1px solid #333;
  }
  .frame {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.2rem;
    width: 96px;
    flex: 0 0 auto;
    cursor: pointer;
    border-radius: 4px;
    padding: 0.25rem;
  }
  .frame.current {
    background: #2d3f57;
  }
  .frame:focus-visible {
    outline: 3px solid #6ab0ff;
    outline-offset: 2px;
  }
  .thumb-wrap {
    position: relative;
    width: 88px;
    height: 88px;
    background: #333;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .thumb {
    max-width: 100%;
    max-height: 100%;
    display: block;
  }
  .badge {
    position: absolute;
    bottom: 2px;
    right: 2px;
    background: rgba(0, 0, 0, 0.75);
    color: #fff;
    font-size: 0.7rem;
    padding: 0.05rem 0.3rem;
    border-radius: 8px;
  }
  .spinner {
    position: absolute;
    bottom: 4px;
    right: 4px;
    width: 10px;
    height: 10px;
    border: 2px solid #888;
    border-top-color: #ddd;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  .check {
    position: absolute;
    top: 2px;
    left: 2px;
    color: #7CFC00;
    font-size: 0.9rem;
  }
  .name {
    font-size: 0.65rem;
    color: #ccc;
    max-width: 88px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
```

- [ ] **Step 2: `Viewer.svelte` accepts a `bboxes` prop and draws rings**

```ts
  let {
    info,
    overlay,
    detected,
    onRequestDetect,
    bboxes = null,
  }: {
    info: ImageInfo;
    overlay: Overlay;
    detected: boolean;
    onRequestDetect: () => void;
    bboxes?: [number, number, number, number][] | null;
  } = $props();
```

Add the import and draw call:

```ts
  import { fitZoom, visibleTiles, ringsFor, TILE, type Level } from "./viewport";
```

```ts
  function frame() {
    if (renderer && needsFrame) {
      needsFrame = false;
      renderer.draw(tilePaths(), canvas.width, canvas.height, overlay);
      const source = detections.length > 0 ? detections : bboxes ?? [];
      if (zoom < 0.5 && source.length > 0) {
        const rings = ringsFor(source, zoom, centerX, centerY, canvas.width, canvas.height, 12);
        renderer.drawRings(rings, canvas.width, canvas.height);
      }
    }
    requestAnimationFrame(frame);
  }
```

(Live `detections` — the z/Z navigation list from a completed `detect` on this frame — take priority over the `bboxes` prop, which is the queue's pre-detect cache; once the operator runs `detect` locally the fresher, threshold-matched list wins.)

- [ ] **Step 3: App.svelte roll mode**

Add imports and roll-mode state:

```svelte
<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { open } from "@tauri-apps/plugin-dialog";
  import Viewer from "./lib/Viewer.svelte";
  import Filmstrip from "./lib/Filmstrip.svelte";
  import type { Level } from "./lib/viewport";

  interface ImageInfo {
    id: number;
    width: number;
    height: number;
    levels: Level[];
  }

  interface FrameInfo {
    index: number;
    file_name: string;
    threshold: number;
    approved: boolean;
    defect_count: number | null;
    bboxes: [number, number, number, number][] | null;
  }

  interface RollInfo {
    dir: string;
    frames: FrameInfo[];
  }

  let info: ImageInfo | null = $state(null);
  let error: string | null = $state(null);
  let loading: string | null = $state(null);
  let viewer: Viewer | undefined = $state();
  let overlay = $state({ enabled: true, threshold: 0.5 });
  let detected = $state(false);
  let componentsAtHalf: number | null = $state(null);
  let detecting = $state(false);

  let roll: RollInfo | null = $state(null);
  let currentIndex = $state(0);
  let scanDone = $state(false);
  let thresholdSaveTimer: ReturnType<typeof setTimeout> | undefined;

  $effect(() => {
    const un = listen<{ id: number; stage: string }>("app-progress", (e) => {
      if (e.payload.stage === "decoding") loading = "Decoding scan";
      else if (e.payload.stage === "building-pyramid") loading = "Building preview";
      else if (e.payload.stage === "ready") loading = null;
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{ index: number; count: number | null }>("roll-progress", (e) => {
      if (!roll) return;
      roll.frames[e.payload.index].defect_count = e.payload.count;
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen<{ index: number; message: string }>("roll-frame-error", (e) => {
      if (!roll) return;
      roll.frames[e.payload.index].defect_count = null;
      error = `Frame ${roll.frames[e.payload.index].file_name}: ${e.payload.message}`;
    });
    return () => {
      un.then((f) => f());
    };
  });

  $effect(() => {
    const un = listen("roll-done", () => {
      scanDone = true;
    });
    return () => {
      un.then((f) => f());
    };
  });

  async function openScan() {
    error = null;
    const path = await open({
      multiple: false,
      filters: [{ name: "Scans", extensions: ["tif", "tiff", "png", "jpg", "jpeg"] }],
    });
    if (typeof path !== "string") return;
    const previousId = info?.id;
    roll = null;
    loading = "Opening scan";
    try {
      info = await invoke<ImageInfo>("open_image", { path });
    } catch (e) {
      error = String(e);
      loading = null;
      return;
    }
    detected = false;
    componentsAtHalf = null;
    if (previousId !== undefined) {
      try {
        await invoke("close_image", { id: previousId });
      } catch {
        // best effort cleanup; the replaced image just lingers in the cache
      }
    }
  }

  async function openRoll() {
    error = null;
    const dir = await open({ multiple: false, directory: true });
    if (typeof dir !== "string") return;
    info = null;
    scanDone = false;
    try {
      roll = await invoke<RollInfo>("open_roll", { dir });
    } catch (e) {
      error = String(e);
      return;
    }
    currentIndex = 0;
    if (roll.frames.length > 0) {
      await activateCurrentFrame();
    }
    try {
      await invoke("scan_roll");
    } catch (e) {
      error = String(e);
    }
  }

  async function activateCurrentFrame() {
    if (!roll) return;
    loading = "Opening frame";
    overlay.threshold = roll.frames[currentIndex].threshold;
    try {
      info = await invoke<ImageInfo>("activate_frame", { index: currentIndex });
    } catch (e) {
      error = String(e);
      loading = null;
      return;
    }
    detected = false;
    componentsAtHalf = null;
  }

  async function selectFrame(index: number) {
    if (!roll || index === currentIndex) return;
    currentIndex = index;
    await activateCurrentFrame();
  }

  function stepFrame(delta: number) {
    if (!roll) return;
    const next = Math.min(Math.max(currentIndex + delta, 0), roll.frames.length - 1);
    if (next === currentIndex) return;
    currentIndex = next;
    void activateCurrentFrame();
  }

  async function approveAndAdvance() {
    if (!roll) return;
    const frame = roll.frames[currentIndex];
    frame.approved = true;
    try {
      await invoke("approve_frame", { index: currentIndex, approved: true });
    } catch (e) {
      error = String(e);
      return;
    }
    const next = roll.frames.findIndex((f, i) => i > currentIndex && !f.approved);
    if (next !== -1) {
      currentIndex = next;
      await activateCurrentFrame();
    }
  }

  function onThresholdInput() {
    if (!roll) return;
    roll.frames[currentIndex].threshold = overlay.threshold;
    clearTimeout(thresholdSaveTimer);
    thresholdSaveTimer = setTimeout(() => {
      invoke("set_frame_threshold", {
        index: currentIndex,
        threshold: overlay.threshold,
      }).catch((e) => {
        error = String(e);
      });
    }, 300);
  }

  async function requestDetect() {
    if (!info || detecting) return;
    error = null;
    detecting = true;
    try {
      const report = await invoke<{ id: number; components_at_half: number }>("detect", {
        id: info.id,
      });
      detected = true;
      componentsAtHalf = report.components_at_half;
      await viewer?.refreshDetections(overlay.threshold);
    } catch (e) {
      error = String(e);
      loading = null;
    } finally {
      detecting = false;
    }
  }

  function onWindowKey(e: KeyboardEvent) {
    if (!roll) return;
    // Only handle roll-navigation keys; everything else (arrows, d/m/z/Z)
    // stays owned by the canvas via its own onkeydown so focus there keeps
    // working exactly as in single-image mode.
    if (e.key === ",") {
      e.preventDefault();
      stepFrame(-1);
    } else if (e.key === ".") {
      e.preventDefault();
      stepFrame(1);
    } else if (e.key === "A") {
      e.preventDefault();
      void approveAndAdvance();
    }
  }
</script>

<svelte:window onkeydown={onWindowKey} />
```

Update the template: add the "Open roll" button, gate `overlay.threshold` binding through `onThresholdInput`, show roll progress in the status line, pass `bboxes` to `Viewer`, and mount `Filmstrip` when a roll is open:

```svelte
<div class="shell">
  <header>
    <button onclick={openScan} disabled={loading !== null}>Open scan</button>
    <button onclick={openRoll} disabled={loading !== null}>Open roll</button>
    {#if info}
      <button onclick={requestDetect} disabled={loading !== null || detecting}>
        {detecting ? "Detecting..." : "Detect"}
      </button>
      <label>
        Sensitivity
        <input
          type="range"
          min="0.05"
          max="0.95"
          step="0.01"
          bind:value={overlay.threshold}
          oninput={onThresholdInput}
        />
      </label>
      <p class="status" role="status">
        {#if detected && componentsAtHalf !== null}
          {componentsAtHalf} defect{componentsAtHalf === 1 ? "" : "s"} at 50%
        {:else}
          Not yet detected
        {/if}
        {#if detecting}
          &mdash; Detecting...
        {/if}
        {#if roll}
          &mdash; {roll.frames.filter((f) => f.approved).length}/{roll.frames.length} approved
          {#if !scanDone}
            &mdash; scanning ({roll.frames.filter((f) => f.defect_count !== null).length}/{roll
              .frames.length})
          {/if}
        {/if}
      </p>
    {/if}
    {#if error}<p role="alert">{error}</p>{/if}
  </header>
  <section class="stage">
    {#if loading}
      <p class="hint" role="status" aria-busy="true">{loading}...</p>
    {:else if info}
      {#key info.id}
        <Viewer
          bind:this={viewer}
          {info}
          {overlay}
          {detected}
          onRequestDetect={requestDetect}
          bboxes={roll ? roll.frames[currentIndex].bboxes : null}
        />
      {/key}
    {:else}
      <p class="hint">Open a scan or a roll to begin.</p>
    {/if}
  </section>
  {#if roll}
    <Filmstrip frames={roll.frames} {currentIndex} onSelect={selectFrame} />
  {/if}
</div>
```

- [ ] **Step 4: Verify, commit**

From `app/`: `npm run test && npm run check`. From root: `cargo test -p unduster-app`, clippy, fmt (unaffected by this frontend-only task; run for the standard sweep).

```bash
git add app/src/App.svelte app/src/lib/Filmstrip.svelte app/src/lib/Viewer.svelte
git commit -m "Add the filmstrip, roll mode, and roll keyboard shortcuts"
```

---

### Task 7: Full sweep and manual gate

**Files:**
- Modify: `.superpowers/sdd/progress.md` (ledger, gitignored scratch per prior plans' convention)

- [ ] **Step 1: Automated sweep**

Root: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`. From `app/`: `npm run test`, `npm run check`.

- [ ] **Step 2: Manual gate (human)**

`cd app && mise exec -- npm run tauri dev`.

1. Click "Open roll", pick the real `Scans/2/TIFF` folder. Confirm the filmstrip appears at the bottom, the first frame activates and shows in the viewer (loader shows "Opening frame" briefly), and thumbnails begin filling in as the background queue reaches each frame (watch the spinner-to-badge swap per thumbnail; status line shows "scanning (N/total)").
2. Press `.` repeatedly to step forward through frames; confirm each activation shows the loader briefly then the new frame, and that stepping more than one frame away evicts older frames (no unbounded memory growth — acceptable to eyeball via Activity Monitor / `top` during a long walk through a large roll).
3. Press `,` to step back; confirm it works symmetrically.
4. On a frame with a nonzero defect count from the queue, confirm red ring markers are visible when zoomed out below 50% even before pressing `d` (queue-cached `bboxes`), and disappear/replace with the live overlay once `d` is pressed.
5. Drag the sensitivity slider; after the roll's frame-level 300ms debounce, confirm the value survives a restart (see step 7).
6. Press `A` to approve the current frame and confirm it jumps to the next unapproved frame and the status line's "approved" count increments; confirm the filmstrip's checkmark appears on the approved frame.
7. Quit the app (or just close the window) and relaunch `npm run tauri dev`. Open the same roll folder again and confirm: approvals, defect counts, thumbnails, and the per-frame threshold you set in step 5 all restored from the sidecar (`Scans/2/TIFF/.unduster/roll.json`).
8. Tab to the filmstrip and confirm keyboard operation: arrow keys move focus with a visible focus ring, `Enter` selects, and the current frame is announced via `aria-selected`/highlight.

- [ ] **Step 3: Close out**

Ledger entry in `.superpowers/sdd/progress.md` recording the task-by-task commit range and manual gate result. Close any bead tracking this plan (bead `rpd` per the prior session's "Next" note) if the human gate passes.

---

## Definition of done for plan 3b-2

- A roll (folder of scans) opens, lists frames, and a background queue detects defects on every frame without blocking the UI.
- Activating a frame reuses the 3b-1 staged decode pipeline; at most 3 frames' pixels are held in the `Images` registry at once, enforced by eviction on every activation.
- The background queue holds at most one frame's pixels at a time and never touches the `Images` registry.
- Sidecar (`roll.json`) persists thresholds, approvals, defect counts, and bboxes atomically (`.bak` rotation) and survives an app restart; files added or removed from the folder between sessions are reconciled on open.
- Thumbnails and all pixel/probability bytes cross to the webview only via `tiles://` (`Rgba`, `Probs`, and now `Thumb` layers).
- Filmstrip is fully keyboard accessible (listbox/option, roving tabindex, focus ring); `,`/`.`/`A` navigate and approve at the App level without disturbing the canvas's existing key handling.
- Ring markers make defects visible even zoomed out, sourced from either the live per-frame detection or the queue's cached bboxes.
- NOT here (future work): brush-based healing/editing, export of approved frames, multi-roll sessions, undo/redo on approvals, configurable eviction window size.
