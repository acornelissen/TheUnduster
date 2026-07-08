//! Roll state and sidecar persistence, plus the `RollState` wrapper that
//! Tauri commands in `lib.rs` drive.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

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

#[derive(Debug)]
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

/// Reads and parses the sidecar envelope at `path`, if present. `Ok(None)`
/// means the file does not exist; any other read/parse/version failure is an
/// error.
fn read_envelope(path: &Path) -> Result<Option<SidecarEnvelope>, String> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
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
    Ok(Some(envelope))
}

/// Loads the sidecar, recovering from `roll.json.bak` if `roll.json` itself
/// is missing. `save()` writes in three steps -- rename current file to
/// `.bak`, write `.tmp`, rename `.tmp` into place -- so a crash in that
/// window can leave only the `.bak` behind. Without this fallback that
/// crash silently drops all remembered frame state on next open.
fn load_sidecar(dir: &Path) -> Result<Vec<Frame>, String> {
    let path = sidecar_path(dir);
    if let Some(envelope) = read_envelope(&path)? {
        return Ok(envelope.frames);
    }
    let bak = path.with_extension("json.bak");
    if let Some(envelope) = read_envelope(&bak)? {
        #[cfg(debug_assertions)]
        eprintln!(
            "{}: missing, recovered frame state from {}",
            path.display(),
            bak.display()
        );
        return Ok(envelope.frames);
    }
    Ok(Vec::new())
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
pub struct RollState {
    pub roll: Mutex<Option<Roll>>,
    /// Guards against double-spawning the background scan task if `scan_roll`
    /// is invoked twice (e.g. a second "Open roll" or an eager retry).
    pub scanning: AtomicBool,
}

impl RollState {
    pub fn open(&self, dir: &Path) -> Result<RollInfo, String> {
        let roll = Roll::open(dir)?;
        let info = roll.info();
        *self.roll.lock().map_err(|e| e.to_string())? = Some(roll);
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
        let guard = self.roll.lock().map_err(|e| e.to_string())?;
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
        let mut guard = self.roll.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_mut().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get_mut(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        frame.image_id = None;
        Ok(())
    }

    pub fn image_id(&self, index: usize) -> Result<Option<u64>, String> {
        let guard = self.roll.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_ref().ok_or("no roll open")?;
        Ok(roll
            .frames
            .get(index)
            .ok_or_else(|| format!("no frame {index}"))?
            .image_id)
    }

    pub fn set_image_id(&self, index: usize, id: u64) -> Result<(), String> {
        let mut guard = self.roll.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_mut().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get_mut(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        frame.image_id = Some(id);
        Ok(())
    }

    pub fn frame_path(&self, index: usize) -> Result<PathBuf, String> {
        let guard = self.roll.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_ref().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        Ok(roll.dir.join(&frame.file_name))
    }

    pub fn set_threshold(&self, index: usize, threshold: f32) -> Result<(), String> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(format!(
                "threshold must be finite and within [0, 1], got {threshold}"
            ));
        }
        let mut guard = self.roll.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_mut().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get_mut(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        frame.threshold = threshold;
        roll.save()
    }

    pub fn set_approved(&self, index: usize, approved: bool) -> Result<(), String> {
        let mut guard = self.roll.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_mut().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get_mut(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        frame.approved = approved;
        roll.save()
    }

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
    fn set_threshold_rejects_non_finite_or_out_of_range_values() {
        let dir = tempfile::tempdir().unwrap();
        let state = opened_state(dir.path(), 1);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.01] {
            let err = state.set_threshold(0, bad).unwrap_err();
            assert!(err.contains("threshold"), "error was: {err}");
        }
        // Valid boundary values are accepted.
        state.set_threshold(0, 0.0).unwrap();
        state.set_threshold(0, 1.0).unwrap();
    }

    #[test]
    fn dir_returns_the_open_rolls_directory() {
        let dir = tempfile::tempdir().unwrap();
        let state = opened_state(dir.path(), 1);
        assert_eq!(state.dir().unwrap(), dir.path());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(dir: &Path, name: &str) {
        std::fs::write(dir.join(name), b"not a real image, just a marker").unwrap();
    }

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
    fn open_after_crash_between_save_renames_recovers_from_bak() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "a.tif");
        let mut roll = Roll::open(dir.path()).unwrap();
        roll.frames[0].approved = true;
        roll.frames[0].threshold = 0.9;
        roll.save().unwrap(); // first save: writes roll.json, no .bak yet
        roll.frames[0].defect_count = Some(7);
        roll.save().unwrap(); // second save: roll.json -> roll.json.bak, then .tmp -> roll.json

        // Simulate a crash in the window after roll.json was renamed to
        // .bak but before .tmp was renamed into place: delete roll.json,
        // leaving only roll.json.bak.
        std::fs::remove_file(sidecar_path(dir.path())).unwrap();
        assert!(sidecar_dir(dir.path()).join("roll.json.bak").exists());

        let reopened = Roll::open(dir.path()).unwrap();
        assert!(reopened.frames[0].approved);
        assert_eq!(reopened.frames[0].threshold, 0.9);
    }

    #[test]
    fn open_merges_new_file_between_two_surviving_frames() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "a.tif");
        touch(dir.path(), "c.tif");
        let mut roll = Roll::open(dir.path()).unwrap();
        roll.frames[0].approved = true; // a.tif
        roll.frames[1].threshold = 0.8; // c.tif
        roll.save().unwrap();

        touch(dir.path(), "b.tif");
        let reopened = Roll::open(dir.path()).unwrap();
        let names: Vec<&str> = reopened
            .frames
            .iter()
            .map(|f| f.file_name.as_str())
            .collect();
        assert_eq!(names, vec!["a.tif", "b.tif", "c.tif"]);
        assert!(reopened.frames[0].approved); // a.tif's state survived
        assert!(!reopened.frames[1].approved); // b.tif is fresh
        assert_eq!(reopened.frames[1].threshold, 0.5);
        assert_eq!(reopened.frames[2].threshold, 0.8); // c.tif's state survived
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
