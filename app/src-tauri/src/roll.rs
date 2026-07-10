//! Roll state and sidecar persistence, plus the `RollState` wrapper that
//! Tauri commands in `lib.rs` drive.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
    pub exported: bool,
    #[serde(default)]
    pub defect_count: Option<usize>,
    #[serde(default)]
    pub bboxes: Option<Vec<[u32; 4]>>,
    #[serde(default)]
    pub strokes: Vec<crate::masks::Stroke>,
    #[serde(default)]
    pub redo_strokes: Vec<crate::masks::Stroke>,
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
            exported: false,
            defect_count: None,
            bboxes: None,
            strokes: Vec::new(),
            redo_strokes: Vec::new(),
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

/// Thumbnail path for a frame, keyed by its full file name (including
/// extension, e.g. `raw0001.tiff.png`) rather than its index -- indices shift
/// when files are added/removed between sessions (`merge` in this module),
/// which would otherwise silently pair a frame with a stale thumbnail for a
/// different image. The extension is kept in the key (not stripped to the
/// stem) to avoid collisions between same-stem files of different formats.
pub fn thumb_path(dir: &Path, file_name: &str) -> PathBuf {
    thumbs_dir(dir).join(format!("{file_name}.png"))
}

pub fn cache_dir(dir: &Path) -> PathBuf {
    sidecar_dir(dir).join("cache")
}

pub fn probs_cache_path(dir: &Path, file_name: &str) -> PathBuf {
    cache_dir(dir).join(format!("{file_name}.probs"))
}

pub fn heal_cache_path(dir: &Path, file_name: &str) -> PathBuf {
    cache_dir(dir).join(format!("{file_name}.heal"))
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
        let mut frames = merge(on_disk, remembered);

        // Validate strokes at frame materialization: invalid lists are replaced
        // with empty vectors. Strokes and redo_strokes are validated independently.
        for frame in &mut frames {
            if crate::masks::validate_strokes(&frame.strokes).is_err() {
                #[cfg(debug_assertions)]
                eprintln!("invalid strokes in frame: {} (dropping)", frame.file_name);
                frame.strokes = Vec::new();
            }
            if crate::masks::validate_strokes(&frame.redo_strokes).is_err() {
                #[cfg(debug_assertions)]
                eprintln!(
                    "invalid redo_strokes in frame: {} (dropping)",
                    frame.file_name
                );
                frame.redo_strokes = Vec::new();
            }
        }

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
    pub exported: bool,
    pub defect_count: Option<usize>,
    pub bboxes: Option<Vec<[u32; 4]>>,
    pub strokes: Vec<crate::masks::Stroke>,
    pub redo_strokes: Vec<crate::masks::Stroke>,
}

impl Frame {
    fn info(&self, index: usize) -> FrameInfo {
        FrameInfo {
            index,
            file_name: self.file_name.clone(),
            threshold: self.threshold,
            approved: self.approved,
            exported: self.exported,
            defect_count: self.defect_count,
            bboxes: self.bboxes.clone(),
            strokes: self.strokes.clone(),
            redo_strokes: self.redo_strokes.clone(),
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct RollInfo {
    pub dir: String,
    pub frames: Vec<FrameInfo>,
    /// The `RollState` generation this roll was opened under. `Roll::info`
    /// cannot populate this itself -- generation is `RollState` bookkeeping,
    /// not `Roll` state -- so `RollState::open` fills it in after the
    /// generation bump. The frontend threads this back on every job event so
    /// it can drop events from a roll that has since been swapped out.
    pub generation: u64,
}

impl Roll {
    /// `generation` is 0 here; `Roll` has no notion of `RollState`
    /// generation. `RollState::open` overwrites it with the post-bump value
    /// before returning -- see `RollInfo::generation`'s doc comment.
    pub fn info(&self) -> RollInfo {
        RollInfo {
            dir: self.dir.display().to_string(),
            frames: self
                .frames
                .iter()
                .enumerate()
                .map(|(i, f)| f.info(i))
                .collect(),
            generation: 0,
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
    /// Incremented on every `open` and `close` (i.e. every roll swap/
    /// teardown). `scan_roll` captures this value when it spawns its
    /// background queue task and re-checks it before processing each frame
    /// via `generation()` -- see that function's doc comment for where the
    /// enforcement actually happens.
    pub generation: AtomicU64,
    /// Frame view recency (most recent last), driving byte-budget eviction:
    /// least-recently-viewed frames release their pixels first.
    lru: Mutex<Vec<usize>>,
    /// Destination directory for queued export jobs. Set by `enqueue_exports`
    /// each time the operator picks a directory; export jobs read it at run
    /// time, so re-queuing with a new directory redirects still-queued jobs.
    pub export_dest: Mutex<Option<PathBuf>>,
}

/// Outcome of `RollState::set_image_id`: either the write landed (carrying
/// whatever id it superseded, if any), or it was discarded because
/// `generation` no longer matched the live roll. Distinct from a plain
/// `Option<Option<u64>>` so call sites read as "written vs lost" rather than
/// a doubly-nested `None`.
#[derive(Debug, PartialEq, Eq)]
pub enum SetImageId {
    /// The id was recorded. Carries the id it replaced, if any -- the
    /// caller must close that superseded image or it leaks.
    Written(Option<u64>),
    /// `generation` no longer matched the live roll; nothing was written.
    /// The caller must close its own freshly-produced image (there is
    /// nothing else to close -- the frame this generation pointed at may
    /// not even exist anymore).
    GenerationLost,
}

impl RollState {
    /// Opens `dir` as the new roll, replacing whatever roll was previously
    /// held. Returns the fresh roll's info alongside any `image_id`s that
    /// were still live on the *old* roll's frames -- the wholesale
    /// replacement means those ids would otherwise never be closed in
    /// `Images`, leaking activated frames across roll switches. Callers must
    /// close the returned ids.
    pub fn open(&self, dir: &Path) -> Result<(RollInfo, Vec<u64>), String> {
        let roll = Roll::open(dir)?;
        let mut info = roll.info();
        let mut guard = self.roll.lock().map_err(|e| e.to_string())?;
        let stale_ids = guard
            .take()
            .map(|old| {
                old.frames
                    .iter()
                    .filter_map(|f| f.image_id)
                    .collect::<Vec<u64>>()
            })
            .unwrap_or_default();
        *guard = Some(roll);
        drop(guard);
        if let Ok(mut lru) = self.lru.lock() {
            lru.clear();
        }
        // `info.generation` must reflect the generation THIS open produced,
        // not whatever was live before it -- read post-bump so a caller
        // (open_roll, then the frontend) can gate on "does this job event
        // belong to the roll I just opened."
        info.generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        Ok((info, stale_ids))
    }

    /// Closes the current roll (if any), returning any `image_id`s still
    /// live on its frames so the caller can close them in `Images`. Used for
    /// the "open a single scan while a roll was open" path, where there is
    /// no replacement roll to fold the teardown into.
    pub fn close(&self) -> Result<Vec<u64>, String> {
        let mut guard = self.roll.lock().map_err(|e| e.to_string())?;
        let stale_ids = guard
            .take()
            .map(|old| {
                old.frames
                    .iter()
                    .filter_map(|f| f.image_id)
                    .collect::<Vec<u64>>()
            })
            .unwrap_or_default();
        drop(guard);
        if let Ok(mut lru) = self.lru.lock() {
            lru.clear();
        }
        self.generation.fetch_add(1, Ordering::SeqCst);
        Ok(stale_ids)
    }

    /// Frame indices whose `image_id` should stay activated around `keep`:
    /// keep-1, keep, keep+1 (clamped to the frame count, so edge frames
    /// don't need special-casing by callers).
    fn keep_window(len: usize, keep: usize) -> std::ops::Range<usize> {
        let start = keep.saturating_sub(1);
        let end = (keep + 2).min(len);
        start..end
    }

    /// Activated frames (index, image_id) ordered least-recently-viewed
    /// first, excluding the keep window around `keep`. The caller pairs this
    /// with per-image sizes from `Images` and a byte budget to decide what
    /// actually gets closed -- eviction never touches the registry from
    /// here; `RollState` only owns frame bookkeeping.
    pub fn eviction_candidates(&self, keep: usize) -> Result<Vec<(usize, u64)>, String> {
        let guard = self.roll.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_ref().ok_or("no roll open")?;
        let window = Self::keep_window(roll.frames.len(), keep);
        let lru = self.lru.lock().map_err(|e| e.to_string())?;
        let recency = |i: &usize| lru.iter().position(|x| x == i).unwrap_or(0);
        let mut out: Vec<(usize, u64)> = roll
            .frames
            .iter()
            .enumerate()
            .filter(|(i, f)| !window.contains(i) && f.image_id.is_some())
            .map(|(i, f)| (i, f.image_id.unwrap()))
            .collect();
        out.sort_by_key(|(i, _)| recency(i));
        Ok(out)
    }

    /// Records that `index` was just viewed (most recent last).
    pub fn touch(&self, index: usize) -> Result<(), String> {
        let mut lru = self.lru.lock().map_err(|e| e.to_string())?;
        lru.retain(|&i| i != index);
        lru.push(index);
        Ok(())
    }

    /// The most recently touched frame index -- i.e. the frame actually on
    /// screen, as opposed to a job's own `index`, which for a prefetch job is
    /// a neighbor rather than the displayed frame. `evict_over_budget` must
    /// be called with this value (not the prefetch job's index) so the
    /// keep-window it derives protects the real current frame, never a
    /// neighbor being warmed in the background. None before any frame has
    /// ever been activated this roll.
    pub fn current_index(&self) -> Result<Option<usize>, String> {
        let lru = self.lru.lock().map_err(|e| e.to_string())?;
        Ok(lru.last().copied())
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

    /// Records the frame's live registry id, but only if `generation` still
    /// matches the live roll -- re-checked under the same lock that guards
    /// the write, `set_exported`-style: a decode already in flight when the
    /// operator opens a new roll must not register its (old-roll) pixels
    /// into the new roll's same-index frame just because that frame's
    /// `image_id` starts out `None`.
    ///
    /// On a landed write, returns the id it replaces (if any) so the caller
    /// can close it: concurrent activations of the same frame each decode
    /// independently, and the superseded image would otherwise be orphaned
    /// in the registry -- roughly a gigabyte of leaked pixels per rapid
    /// re-click on a large scan.
    pub fn set_image_id(
        &self,
        generation: u64,
        index: usize,
        id: u64,
    ) -> Result<SetImageId, String> {
        let mut guard = self.roll.lock().map_err(|e| e.to_string())?;
        if self.generation.load(Ordering::SeqCst) != generation {
            return Ok(SetImageId::GenerationLost);
        }
        let roll = guard.as_mut().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get_mut(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        let previous = frame.image_id.filter(|&old| old != id);
        frame.image_id = Some(id);
        Ok(SetImageId::Written(previous))
    }

    /// Records the frame's live registry id only if the frame has none AND
    /// `generation` still matches the live roll. Returns true when the id
    /// was recorded, false otherwise -- the caller lost the race (to
    /// another racer OR to a roll swap) and must close its own image either
    /// way, so both loss modes collapse to the same `false`. Check and set
    /// are one operation under the roll lock, so two racers can never both
    /// believe they won, and a stale generation can never claim an empty
    /// slot that belongs to a different roll.
    ///
    /// Prefetch-only counterpart to `set_image_id`: a background warm-up
    /// racing an activation of the same frame must LOSE the tie, never
    /// supersede -- superseding would close the image the activation just
    /// put on screen. Activation keeps `set_image_id`'s supersede semantics
    /// (latest activation wins, superseded image closed) unchanged.
    pub fn set_image_id_if_absent(
        &self,
        generation: u64,
        index: usize,
        id: u64,
    ) -> Result<bool, String> {
        let mut guard = self.roll.lock().map_err(|e| e.to_string())?;
        if self.generation.load(Ordering::SeqCst) != generation {
            return Ok(false);
        }
        let roll = guard.as_mut().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get_mut(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        if frame.image_id.is_some() {
            return Ok(false);
        }
        frame.image_id = Some(id);
        Ok(true)
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

    /// Path, file name, stored threshold, and strokes for a frame, read under
    /// a single lock acquisition: the transient export pipeline needs all four
    /// together (path to decode, file name to name the destination file,
    /// threshold to build the mask, strokes to apply), and reading them
    /// separately would risk observing an inconsistent frame across two lock
    /// windows.
    pub fn export_frame_meta(
        &self,
        index: usize,
    ) -> Result<(PathBuf, String, f32, Vec<crate::masks::Stroke>), String> {
        let guard = self.roll.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_ref().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        Ok((
            roll.dir.join(&frame.file_name),
            frame.file_name.clone(),
            frame.threshold,
            frame.strokes.clone(),
        ))
    }

    /// Sets a frame's threshold together with the recomputed component count
    /// and boxes, in one sidecar write. The sole setter for a frame's
    /// threshold: the operator's sensitivity slider only has fresh
    /// counts/bboxes to offer when the frame's image is registry-resident
    /// with probabilities
    /// (`components` in lib.rs resolves that and passes `None` otherwise) --
    /// writing threshold there too keeps this a single seam instead of a
    /// second call racing the first.
    ///
    /// `generation` is re-checked under the same lock that guards the write,
    /// like `record_scan_result`/`set_exported`: the calling command spans
    /// several separate lock acquisitions (image-id lookup, components
    /// computation, this write) with nothing held across them, and
    /// `open_roll` is free to swap the roll between any two of them -- a
    /// threshold and count computed against roll A must never land in roll
    /// B's sidecar just because the swap won the race.
    ///
    /// A mismatch returns `Ok(false)` (write discarded) rather than
    /// `set_exported`'s `Err`: this setter's caller is the interactive
    /// slider save, where a lost race against a roll swap is a benign no-op
    /// the operator should never see an error toast for -- not a queue task
    /// whose discards are logged. Returning the discard as data keeps the
    /// caller from string-matching an error message to tell "benign race"
    /// from real failures.
    pub fn set_threshold_and_components(
        &self,
        generation: u64,
        index: usize,
        threshold: f32,
        count: Option<usize>,
        bboxes: Option<Vec<[u32; 4]>>,
    ) -> Result<bool, String> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(format!(
                "threshold must be finite and within [0, 1], got {threshold}"
            ));
        }
        let mut guard = self.roll.lock().map_err(|e| e.to_string())?;
        if self.generation.load(Ordering::SeqCst) != generation {
            return Ok(false);
        }
        let roll = guard.as_mut().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get_mut(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        frame.threshold = threshold;
        frame.defect_count = count;
        frame.bboxes = bboxes;
        roll.save()?;
        Ok(true)
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

    pub fn set_strokes(
        &self,
        index: usize,
        strokes: Vec<crate::masks::Stroke>,
        redo_strokes: Vec<crate::masks::Stroke>,
    ) -> Result<(), String> {
        let mut guard = self.roll.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_mut().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get_mut(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        frame.strokes = strokes;
        frame.redo_strokes = redo_strokes;
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
            .filter(|(_, f)| {
                // Unscanned frames, plus scanned frames whose probability
                // cache file is missing: rolls scanned before the cache
                // existed (or after a cache wipe) backfill in the background
                // instead of silently never caching.
                f.defect_count.is_none() || !probs_cache_path(&roll.dir, &f.file_name).exists()
            })
            .map(|(i, _)| i)
            .collect())
    }

    /// Approved frame indices (in order, exported frames INCLUDED --
    /// pressing "Export approved" again re-exports all approved work,
    /// predictably overwriting whatever landed before) AND the roll
    /// generation, read under one lock acquisition. `enqueue_exports` used
    /// to call a `frames_to_export()`-shaped indices-only accessor then
    /// `generation()` as two separate locks; a roll swap landing between
    /// them would tag the OLD roll's indices with the NEW generation, and
    /// the worker's per-job generation check -- which trusts the tag, not
    /// the indices' actual origin -- would then wave those jobs through
    /// against the wrong roll. Snapshotting both together closes that
    /// window: the generation returned here is provably the one that
    /// produced these exact indices.
    pub fn approved_snapshot(&self) -> Result<(Vec<usize>, u64), String> {
        let guard = self.roll.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_ref().ok_or("no roll open")?;
        let indices = roll
            .frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.approved)
            .map(|(i, _)| i)
            .collect();
        Ok((indices, self.generation.load(Ordering::SeqCst)))
    }

    /// A single frame's `image_id` presence AND the roll generation, read
    /// under one lock acquisition. `enqueue_job`'s validation shape used to
    /// be `roll.image_id(index)?` (bounds/roll-open check, discarding the
    /// id) followed by a separate `roll.generation()` call; the same
    /// two-lock race as `approved_snapshot` applies -- a swap between the
    /// two calls could tag a validated-against-the-OLD-roll index with the
    /// NEW generation. Validation semantics are unchanged: errors when no
    /// roll is open or `index` is out of range.
    pub fn frame_snapshot(&self, index: usize) -> Result<(Option<u64>, u64), String> {
        let guard = self.roll.lock().map_err(|e| e.to_string())?;
        let roll = guard.as_ref().ok_or("no roll open")?;
        let image_id = roll
            .frames
            .get(index)
            .ok_or_else(|| format!("no frame {index}"))?
            .image_id;
        Ok((image_id, self.generation.load(Ordering::SeqCst)))
    }

    /// Records a queue result, but only if the roll the scan started against
    /// is still the live one: a frame decoded against roll A must never land
    /// in roll B's sidecar when a swap happens mid-decode. The generation is
    /// re-checked under the same lock that guards the write, closing the
    /// window between the queue loop's top-of-iteration check and this write.
    pub fn record_scan_result(
        &self,
        generation: u64,
        index: usize,
        count: Option<usize>,
        bboxes: Option<Vec<[u32; 4]>>,
    ) -> Result<(), String> {
        let mut guard = self.roll.lock().map_err(|e| e.to_string())?;
        if self.generation.load(Ordering::SeqCst) != generation {
            return Err("roll changed during scan; result discarded".to_string());
        }
        let roll = guard.as_mut().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get_mut(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        frame.defect_count = count;
        frame.bboxes = bboxes;
        roll.save()
    }

    /// Records that a frame was exported, but only if the roll the export
    /// started against is still the live one: mirrors `record_scan_result`'s
    /// generation re-check under the same lock that guards the write, so a
    /// frame exported against roll A never marks roll B's sidecar done.
    pub fn set_exported(&self, generation: u64, index: usize) -> Result<(), String> {
        let mut guard = self.roll.lock().map_err(|e| e.to_string())?;
        if self.generation.load(Ordering::SeqCst) != generation {
            return Err("roll changed during export; result discarded".to_string());
        }
        let roll = guard.as_mut().ok_or("no roll open")?;
        let frame = roll
            .frames
            .get_mut(index)
            .ok_or_else(|| format!("no frame {index}"))?;
        frame.exported = true;
        roll.save()
    }

    pub fn dir(&self) -> Result<PathBuf, String> {
        let guard = self.roll.lock().map_err(|e| e.to_string())?;
        Ok(guard.as_ref().ok_or("no roll open")?.dir.clone())
    }

    pub fn set_export_dest(&self, dest: PathBuf) -> Result<(), String> {
        let mut guard = self.export_dest.lock().map_err(|e| e.to_string())?;
        *guard = Some(dest);
        Ok(())
    }

    pub fn export_dest(&self) -> Result<Option<PathBuf>, String> {
        let guard = self.export_dest.lock().map_err(|e| e.to_string())?;
        Ok(guard.clone())
    }

    /// Clears the in-progress scan flag. Pulled out as its own method so the
    /// `scan_roll` task's drop guard has a single, testable reset authority
    /// -- one line, called unconditionally (including on panic unwind).
    pub fn clear_scanning(&self) {
        self.scanning.store(false, Ordering::SeqCst);
    }

    /// Current roll generation. `scan_roll` snapshots this when its queue
    /// task spawns and compares against it before processing each frame
    /// (see `lib.rs::scan_roll`); a mismatch means the roll was replaced or
    /// closed mid-scan, and the task breaks out rather than writing stale
    /// thumbnails/results into the wrong roll's directory or sidecar.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    /// Maps a registry image id back to its roll directory and file name.
    /// Returns Ok(None) when no frame has this id -- either no roll is open
    /// (a normal cache-miss condition, not an error) or the frame was evicted.
    /// This deviates from sibling accessors: "no roll" is Ok(None), not Err,
    /// because id-keyed commands (detect, heal) expect closed rolls to be
    /// a benign cache-miss, not a failure.
    pub fn frame_for_image(&self, id: u64) -> Result<Option<(PathBuf, String)>, String> {
        let guard = self.roll.lock().map_err(|e| e.to_string())?;
        let roll = match guard.as_ref() {
            Some(r) => r,
            None => return Ok(None),
        };
        for frame in &roll.frames {
            if frame.image_id == Some(id) {
                return Ok(Some((roll.dir.clone(), frame.file_name.clone())));
            }
        }
        Ok(None)
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
    fn open_returns_the_previous_rolls_live_ids_and_reflects_the_new_roll() {
        let dir_a = tempfile::tempdir().unwrap();
        let dir_b = tempfile::tempdir().unwrap();
        touch(dir_b.path(), "g00.tif");
        let state = opened_state(dir_a.path(), 3);
        let gen = state.generation();
        state.set_image_id(gen, 0, 100).unwrap();
        state.set_image_id(gen, 1, 101).unwrap();
        // frame 2 never activated -- must not appear in the stale ids.

        let (info, mut stale_ids) = state.open(dir_b.path()).unwrap();
        stale_ids.sort();
        assert_eq!(stale_ids, vec![100, 101]);

        // The state now reflects roll B, not leftover A bookkeeping.
        assert_eq!(info.dir, dir_b.path().display().to_string());
        assert_eq!(state.dir().unwrap(), dir_b.path());
        assert_eq!(state.image_id(0).unwrap(), None);
    }

    #[test]
    fn open_on_a_fresh_state_returns_no_stale_ids() {
        let dir = tempfile::tempdir().unwrap();
        let state = RollState::default();
        let (_, stale_ids) = state.open(dir.path()).unwrap();
        assert!(stale_ids.is_empty());
    }

    #[test]
    fn close_returns_live_ids_and_clears_the_roll() {
        let dir = tempfile::tempdir().unwrap();
        let state = opened_state(dir.path(), 2);
        state.set_image_id(state.generation(), 0, 5).unwrap();

        let stale_ids = state.close().unwrap();
        assert_eq!(stale_ids, vec![5]);
        assert!(state.dir().is_err()); // no roll open anymore
    }

    #[test]
    fn close_on_a_fresh_state_returns_no_ids() {
        let state = RollState::default();
        assert_eq!(state.close().unwrap(), Vec::<u64>::new());
    }

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

    #[test]
    fn generation_increments_on_open_and_close() {
        // Pins the enforcement mechanism scan_roll relies on: every roll
        // swap or teardown must be observable to an in-flight scan task via
        // a generation bump, so it can detect "the roll I was scanning is
        // gone" and stop before writing into the wrong roll's files.
        let dir_a = tempfile::tempdir().unwrap();
        let dir_b = tempfile::tempdir().unwrap();
        let state = RollState::default();
        let g0 = state.generation();
        state.open(dir_a.path()).unwrap();
        let g1 = state.generation();
        assert!(g1 > g0);
        state.open(dir_b.path()).unwrap();
        let g2 = state.generation();
        assert!(g2 > g1);
        state.close().unwrap();
        let g3 = state.generation();
        assert!(g3 > g2);
    }

    #[test]
    fn keep_window_clamps_at_both_edges() {
        assert_eq!(RollState::keep_window(5, 0), 0..2);
        assert_eq!(RollState::keep_window(5, 2), 1..4);
        assert_eq!(RollState::keep_window(5, 4), 3..5);
    }

    #[test]
    fn eviction_candidates_order_least_recently_viewed_first() {
        let dir = tempfile::tempdir().unwrap();
        for n in ["a.png", "b.png", "c.png", "d.png", "e.png"] {
            std::fs::write(dir.path().join(n), b"x").unwrap();
        }
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        let gen = state.generation();
        for (i, id) in [(0, 10), (1, 11), (3, 13), (4, 14)] {
            state.set_image_id(gen, i, id).unwrap();
        }
        // view order: 0, 4, 3, 1 -> among candidates outside window(1)={0,1,2},
        // frame 4 was viewed before 3, but 0 is oldest... 0 is inside window.
        for i in [0, 4, 3, 1] {
            state.touch(i).unwrap();
        }
        let candidates = state.eviction_candidates(1).unwrap();
        // outside window {0,1,2}: frames 4 and 3; 4 viewed before 3
        assert_eq!(candidates, vec![(4, 14), (3, 13)]);
    }

    #[test]
    fn lru_clears_on_roll_swap() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        state.touch(0).unwrap();
        state.open(dir.path()).unwrap(); // swap clears recency
        state.set_image_id(state.generation(), 0, 5).unwrap();
        // frame 0 activated but inside every window; no candidates, no panic
        assert!(state.eviction_candidates(0).unwrap().is_empty());
    }

    #[test]
    fn ids_to_evict_only_returns_activated_frames_outside_the_window() {
        let dir = tempfile::tempdir().unwrap();
        let state = opened_state(dir.path(), 5);
        let gen = state.generation();
        state.set_image_id(gen, 0, 10).unwrap();
        state.set_image_id(gen, 1, 11).unwrap();
        state.set_image_id(gen, 2, 12).unwrap();
        state.set_image_id(gen, 4, 14).unwrap();
        // window around 2 is {1,2,3}: frame 0 and frame 4 are activated but
        // outside it, frame 3 is inside the window but was never activated.
        let mut evict = state.eviction_candidates(2).unwrap();
        evict.sort();
        assert_eq!(evict, vec![(0, 10), (4, 14)]);
    }

    #[test]
    fn set_threshold_and_approved_persist_via_save() {
        let dir = tempfile::tempdir().unwrap();
        let state = opened_state(dir.path(), 2);
        state
            .set_threshold_and_components(state.generation(), 0, 0.33, None, None)
            .unwrap();
        state.set_approved(1, true).unwrap();
        let reopened = Roll::open(dir.path()).unwrap();
        assert_eq!(reopened.frames[0].threshold, 0.33);
        assert!(reopened.frames[1].approved);
    }

    #[test]
    fn set_threshold_and_components_rejects_non_finite_or_out_of_range_values() {
        let dir = tempfile::tempdir().unwrap();
        let state = opened_state(dir.path(), 1);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.01] {
            let err = state
                .set_threshold_and_components(state.generation(), 0, bad, None, None)
                .unwrap_err();
            assert!(err.contains("threshold"), "error was: {err}");
        }
        // Valid boundary values are accepted.
        state
            .set_threshold_and_components(state.generation(), 0, 0.0, None, None)
            .unwrap();
        state
            .set_threshold_and_components(state.generation(), 0, 1.0, None, None)
            .unwrap();
    }

    #[test]
    fn set_threshold_and_components_persists_all_three_together() {
        let dir = tempfile::tempdir().unwrap();
        let state = opened_state(dir.path(), 1);
        let written = state
            .set_threshold_and_components(
                state.generation(),
                0,
                0.72,
                Some(2),
                Some(vec![[1, 2, 3, 4], [5, 6, 7, 8]]),
            )
            .unwrap();
        assert!(written);
        let reopened = Roll::open(dir.path()).unwrap();
        assert_eq!(reopened.frames[0].threshold, 0.72);
        assert_eq!(reopened.frames[0].defect_count, Some(2));
        assert_eq!(
            reopened.frames[0].bboxes,
            Some(vec![[1, 2, 3, 4], [5, 6, 7, 8]])
        );
    }

    #[test]
    fn set_threshold_and_components_accepts_none_when_not_resident() {
        let dir = tempfile::tempdir().unwrap();
        let state = opened_state(dir.path(), 1);
        // Seed a prior count/bboxes, then move the threshold with no fresh
        // components (the not-resident/no-probs case) -- the frame's stale
        // count and boxes are cleared, not left dangling under the new
        // threshold they no longer match.
        state
            .set_threshold_and_components(
                state.generation(),
                0,
                0.4,
                Some(3),
                Some(vec![[0, 0, 1, 1]]),
            )
            .unwrap();
        state
            .set_threshold_and_components(state.generation(), 0, 0.6, None, None)
            .unwrap();
        let reopened = Roll::open(dir.path()).unwrap();
        assert_eq!(reopened.frames[0].threshold, 0.6);
        assert_eq!(reopened.frames[0].defect_count, None);
        assert_eq!(reopened.frames[0].bboxes, None);
    }

    #[test]
    fn set_threshold_and_components_discards_a_stale_generation_write() {
        let dir = tempfile::tempdir().unwrap();
        let state = opened_state(dir.path(), 1);
        state
            .set_threshold_and_components(
                state.generation(),
                0,
                0.4,
                Some(3),
                Some(vec![[0, 0, 1, 1]]),
            )
            .unwrap();
        let stale = state.generation();
        state.open(dir.path()).unwrap(); // bumps generation, reloads sidecar
        let written = state
            .set_threshold_and_components(stale, 0, 0.9, Some(9), Some(vec![[9, 9, 9, 9]]))
            .unwrap();
        assert!(!written);
        // The stale write left threshold, count, and bboxes all untouched.
        let reopened = Roll::open(dir.path()).unwrap();
        assert_eq!(reopened.frames[0].threshold, 0.4);
        assert_eq!(reopened.frames[0].defect_count, Some(3));
        assert_eq!(reopened.frames[0].bboxes, Some(vec![[0, 0, 1, 1]]));
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
    fn approved_snapshot_returns_the_same_generation_the_state_reports_and_only_approved_indices() {
        // Pins the bead-caw contract: indices and generation must come from
        // ONE lock acquisition, so a roll swap between reading them (the
        // old two-call shape) can never tag old-roll indices with the new
        // generation.
        let dir = tempfile::tempdir().unwrap();
        for n in ["a.png", "b.png", "c.png", "d.png"] {
            std::fs::write(dir.path().join(n), b"x").unwrap();
        }
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        state.set_approved(1, true).unwrap();
        state.set_approved(3, true).unwrap();
        state.set_exported(state.generation(), 1).unwrap(); // exported frames stay included
        let (indices, generation) = state.approved_snapshot().unwrap();
        assert_eq!(indices, vec![1, 3]);
        assert_eq!(generation, state.generation());
    }

    #[test]
    fn approved_snapshot_errors_when_no_roll_open() {
        let state = RollState::default();
        assert!(state.approved_snapshot().unwrap_err().contains("no roll"));
    }

    #[test]
    fn frame_snapshot_returns_image_id_presence_and_generation_together() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        let gen = state.generation();
        let (image_id, generation) = state.frame_snapshot(0).unwrap();
        assert_eq!(image_id, None);
        assert_eq!(generation, gen);
        state.set_image_id(gen, 0, 7).unwrap();
        let (image_id, generation) = state.frame_snapshot(0).unwrap();
        assert_eq!(image_id, Some(7));
        assert_eq!(generation, gen);
    }

    #[test]
    fn frame_snapshot_errors_on_bad_index_and_no_roll() {
        let state = RollState::default();
        assert!(state.frame_snapshot(0).unwrap_err().contains("no roll"));
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        state.open(dir.path()).unwrap();
        assert!(state.frame_snapshot(5).unwrap_err().contains("no frame"));
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
        assert!(state
            .set_exported(stale, 0)
            .unwrap_err()
            .contains("roll changed"));
        // persisted across the reopen:
        let info = state.open(dir.path()).unwrap().0;
        assert!(info.frames[0].exported);
    }

    #[test]
    fn operations_before_open_error_clearly() {
        let state = RollState::default();
        assert!(state
            .set_threshold_and_components(state.generation(), 0, 0.5, None, None)
            .unwrap_err()
            .contains("no roll"));
        assert!(state
            .eviction_candidates(0)
            .unwrap_err()
            .contains("no roll"));
    }

    #[test]
    fn set_image_id_returns_the_id_it_replaces() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        let gen = state.generation();
        assert_eq!(
            state.set_image_id(gen, 0, 7).unwrap(),
            SetImageId::Written(None)
        );
        assert_eq!(
            state.set_image_id(gen, 0, 9).unwrap(),
            SetImageId::Written(Some(7))
        ); // superseded
        assert_eq!(
            state.set_image_id(gen, 0, 9).unwrap(),
            SetImageId::Written(None)
        ); // same id: nothing to close
        assert_eq!(state.image_id(0).unwrap(), Some(9));
    }

    #[test]
    fn set_image_id_loses_a_stale_generation_and_leaves_image_id_untouched() {
        // Pins the bead-619 gap: a decode already running when the operator
        // swaps rolls must not register its pixels into the new roll's
        // same-index frame just because that frame's image_id started None.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        let stale = state.generation();
        state.open(dir.path()).unwrap(); // swap bumps generation, clears image_id
        assert_eq!(
            state.set_image_id(stale, 0, 999).unwrap(),
            SetImageId::GenerationLost
        );
        // The stale write left the fresh roll's frame untouched.
        assert_eq!(state.image_id(0).unwrap(), None);
    }

    #[test]
    fn set_image_id_if_absent_loses_to_an_existing_id() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        let gen = state.generation();
        // Empty slot: the if-absent set wins and records the id.
        assert!(state.set_image_id_if_absent(gen, 0, 7).unwrap());
        assert_eq!(state.image_id(0).unwrap(), Some(7));
        // Occupied slot: the if-absent set loses -- the existing id stays,
        // never superseded. This is the prefetch tie-loss contract: an
        // activation that landed first keeps its (displayed) image.
        assert!(!state.set_image_id_if_absent(gen, 0, 9).unwrap());
        assert_eq!(state.image_id(0).unwrap(), Some(7));
        // Bad index still errors rather than reading as a lost tie.
        assert!(state
            .set_image_id_if_absent(gen, 5, 1)
            .unwrap_err()
            .contains("no frame"));
    }

    #[test]
    fn set_image_id_if_absent_loses_a_stale_generation_and_leaves_image_id_untouched() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        let stale = state.generation();
        state.open(dir.path()).unwrap(); // swap bumps generation, clears image_id
                                         // Even though the fresh roll's frame 0 has no image_id (an empty
                                         // slot the if-absent check would otherwise happily claim), the
                                         // stale generation must lose -- false, same as a tied race.
        assert!(!state.set_image_id_if_absent(stale, 0, 999).unwrap());
        assert_eq!(state.image_id(0).unwrap(), None);
    }

    #[test]
    fn scan_flag_clears_when_setup_fails_before_spawn() {
        // scan_roll's sync body must clear the flag it just set when no roll
        // is open; this pins the RollState pieces that path relies on.
        let state = RollState::default();
        assert!(state.dir().is_err());
        assert!(state.frames_to_scan().is_err());
    }

    #[test]
    fn clear_scanning_resets_the_flag() {
        let state = RollState::default();
        state.scanning.store(true, Ordering::SeqCst);
        state.clear_scanning();
        assert!(!state.scanning.load(Ordering::SeqCst));
    }

    #[test]
    fn strokes_persist_in_the_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        let strokes = vec![crate::masks::Stroke {
            erase: false,
            radius: 12.0,
            points: vec![[100.0, 200.0], [110.0, 210.0]],
        }];
        let redo = vec![crate::masks::Stroke {
            erase: true,
            radius: 8.0,
            points: vec![[5.0, 5.0]],
        }];
        state.set_strokes(0, strokes.clone(), redo.clone()).unwrap();
        // reopen from disk: both lists survive
        let (info, _) = state.open(dir.path()).unwrap();
        assert_eq!(info.frames[0].strokes, strokes);
        assert_eq!(info.frames[0].redo_strokes, redo);
    }

    #[test]
    fn set_strokes_rejects_bad_index() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        assert!(state.set_strokes(5, vec![], vec![]).is_err());
    }

    #[test]
    fn export_frame_meta_carries_strokes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        let strokes = vec![crate::masks::Stroke {
            erase: false,
            radius: 4.0,
            points: vec![[1.0, 1.0]],
        }];
        state.set_strokes(0, strokes.clone(), vec![]).unwrap();
        let (_, _, _, meta_strokes) = state.export_frame_meta(0).unwrap();
        assert_eq!(meta_strokes, strokes);
    }

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

    #[test]
    fn frame_for_image_maps_ids_to_files() {
        let dir = tempfile::tempdir().unwrap();
        for n in ["a.png", "b.png"] {
            std::fs::write(dir.path().join(n), b"x").unwrap();
        }
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        state.set_image_id(state.generation(), 1, 42).unwrap();
        let (d, name) = state.frame_for_image(42).unwrap().expect("mapped");
        assert_eq!(d, dir.path());
        assert_eq!(name, "b.png");
        assert!(state.frame_for_image(7).unwrap().is_none());
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
        let out = thumb_path(dir.path(), "raw0001.tiff");
        write_thumbnail(&rgba, w, h, &out).unwrap();
        assert!(out.exists());
        assert_eq!(out.file_name().unwrap(), "raw0001.tiff.png");
        let decoded = fd_io::decode(&out).unwrap();
        assert!(decoded.width <= 128 && decoded.height <= 128);
        assert_eq!(decoded.channels, 3);
    }

    #[test]
    fn thumb_path_keeps_extension_to_avoid_stem_collisions() {
        // Two files with the same stem but different formats must not
        // collide on the same thumbnail path.
        let dir = tempfile::tempdir().unwrap();
        let tiff = thumb_path(dir.path(), "raw0001.tiff");
        let png = thumb_path(dir.path(), "raw0001.png");
        assert_ne!(tiff, png);
        assert_eq!(tiff.file_name().unwrap(), "raw0001.tiff.png");
        assert_eq!(png.file_name().unwrap(), "raw0001.png.png");
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

    #[test]
    fn invalid_sidecar_strokes_are_dropped_at_load() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        let state = RollState::default();
        state.open(dir.path()).unwrap();
        // persist a valid stroke, then corrupt it on disk to out-of-range coords
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
        // Verify the corruption actually landed
        let corrupted = std::fs::read_to_string(&sidecar).unwrap();
        assert!(
            corrupted.contains("1e30"),
            "corruption must be present in file"
        );
        let (info, _) = state.open(dir.path()).unwrap();
        assert!(
            info.frames[0].strokes.is_empty(),
            "invalid strokes must be dropped, not wedge the frame"
        );
    }
}
