//! Roll state and sidecar persistence: pure data layer, no Tauri commands
//! yet. `Roll::open`/`Roll::save` are exercised only by the tests below
//! until a later task wires this module into command handlers, so the
//! module is allowed to be dead code outside of `cfg(test)` builds.
#![cfg_attr(not(test), allow(dead_code))]

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
