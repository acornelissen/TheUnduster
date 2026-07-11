use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use fd_infer::{Detector, Ep};
use fd_io::ImageBuf;

/// Loaded model together with its file SHA-256.
#[cfg_attr(not(test), allow(dead_code))]
struct LoadedDetector {
    detector: Detector,
    hash: [u8; 32],
}

/// Cheaply cloneable handle: Task 3's detect command clones it into a
/// spawn_blocking closure (which needs 'static).
#[derive(Clone)]
pub struct DetectorState(Arc<Mutex<Option<LoadedDetector>>>);

impl Default for DetectorState {
    fn default() -> Self {
        DetectorState(Arc::new(Mutex::new(None)))
    }
}

impl DetectorState {
    pub fn load(&self, path: &Path) -> Result<(), String> {
        // CoreML first: on Apple Silicon it detects a 168MP frame in ~9s at
        // ~2.9 GB peak versus ~36s at 11+ GB on the CPU EP (which was
        // enough, stacked with a decoded roll frame, to get the app
        // memory-killed). Thresholded output measured identical on a real
        // scan; formal benchmark tracked separately (TheUnduster-3uz).
        let det = Detector::load(path, Ep::CoreML)
            .or_else(|_| Detector::load(path, Ep::Cpu))
            .map_err(|e| e.to_string())?;
        // Hashes the file as loaded; assumes nothing else in the models dir
        // writes concurrently (TOCTOU note -- the app is the only writer in
        // practice, via `models::download_inpaint_model`'s atomic rename).
        let hash = crate::models::file_sha256(path)?;
        *self.0.lock().map_err(|e| e.to_string())? = Some(LoadedDetector {
            detector: det,
            hash,
        });
        Ok(())
    }

    pub fn detect(&self, img: &ImageBuf) -> Result<Vec<f32>, String> {
        self.detect_with_progress(img, &mut |_, _| {})
    }

    /// `detect` with a per-tile progress callback `(done, total)`, threaded
    /// straight through to `fd_infer::Detector::probabilities_with_progress`.
    /// Mirrors the crate-level split (`detect` is the no-op-callback
    /// wrapper): no production caller needs this today (single-image mode
    /// and the roll job's Detect arm both go through `detect_hashed_with_progress`
    /// instead, to keep the cache-provenance hash paired with the same lock
    /// acquisition), kept for API symmetry and direct testing.
    pub fn detect_with_progress(
        &self,
        img: &ImageBuf,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<Vec<f32>, String> {
        let mut guard = self.0.lock().map_err(|e| e.to_string())?;
        match guard.as_mut() {
            Some(loaded) => loaded
                .detector
                .probabilities_with_progress(img, progress)
                .map_err(|e| e.to_string()),
            None => Err("no detector loaded".to_string()),
        }
    }

    /// Runs inference and reads the model's file hash under one lock
    /// acquisition, mirroring `InpainterState::with_inpainter_hashed`'s
    /// rationale: `detect()` and `hash()` taken as two separate locks would
    /// let a model load complete between them, so the model that actually
    /// produced `probs` could be model B while the hash returned (from a
    /// call before or after) says model A. Pairing them under a single guard
    /// makes "the hash recorded for cache provenance" and "the model that
    /// detected" the same observation, closing that race. Errors exactly
    /// like `detect()` when no detector is loaded.
    pub fn detect_hashed(&self, img: &ImageBuf) -> Result<(Vec<f32>, [u8; 32]), String> {
        self.detect_hashed_with_progress(img, &mut |_, _| {})
    }

    /// `detect_hashed` with a per-tile progress callback `(done, total)`.
    /// Used by the single-image detect command and the roll job's Detect
    /// arm to narrate a long detect tile by tile; the hash pairing guarantee
    /// documented on `detect_hashed` holds identically here (same lock, same
    /// closure).
    pub fn detect_hashed_with_progress(
        &self,
        img: &ImageBuf,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<(Vec<f32>, [u8; 32]), String> {
        let mut guard = self.0.lock().map_err(|e| e.to_string())?;
        match guard.as_mut() {
            Some(loaded) => {
                let probs = loaded
                    .detector
                    .probabilities_with_progress(img, progress)
                    .map_err(|e| e.to_string())?;
                Ok((probs, loaded.hash))
            }
            None => Err("no detector loaded".to_string()),
        }
    }

    /// The producing detector's file hash, for cache provenance. `None` on
    /// no detector loaded, or (via `.ok()`) on a poisoned mutex -- unlike
    /// sibling methods, poisoning is not propagated as `Err` here: a
    /// poisoned state already fails `detect()` loudly on the hot path, so a
    /// cache write/read on this path just skips instead of erroring.
    pub fn hash(&self) -> Option<[u8; 32]> {
        let guard = self.0.lock().ok()?;
        guard.as_ref().map(|loaded| loaded.hash)
    }
}

/// Loaded inpainting model together with its file SHA-256.
#[cfg_attr(not(test), allow(dead_code))]
struct LoadedInpainter {
    inpainter: fd_heal::Inpainter,
    hash: [u8; 32],
}

/// Cheaply cloneable handle to the (optional) inpainting model, mirroring
/// DetectorState. None means heal_frame falls back to classical fill only.
#[derive(Clone)]
pub struct InpainterState {
    inner: Arc<Mutex<Option<LoadedInpainter>>>,
    /// Set by `load_fixture` and cleared by `load`. This is the fixture-ness
    /// detection seam: the debug-build autoload in `lib.rs` is the one place
    /// that KNOWS it is falling back to the mean-fill dev stub instead of
    /// real LaMa, so it records that fact explicitly at load time by calling
    /// `load_fixture` rather than `load`. `models::inpainter_status` reads
    /// this flag directly -- comparing file hashes after the fact would be
    /// guessing at something the loader already knew for certain.
    fixture: Arc<AtomicBool>,
    /// A real `lama.onnx` that exists on disk but failed to load (corrupt or
    /// incompatible file), as opposed to no file at all. Recorded by the
    /// setup path via `record_load_error` so `inpainter_load_error` can
    /// surface it to the frontend through the same polled-status channel
    /// `inpainter_status` already uses, instead of the previous eprintln-only
    /// dead end. Cleared on the next successful `load`.
    load_error: Arc<Mutex<Option<String>>>,
}

impl Default for InpainterState {
    fn default() -> Self {
        InpainterState {
            inner: Arc::new(Mutex::new(None)),
            fixture: Arc::new(AtomicBool::new(false)),
            load_error: Arc::new(Mutex::new(None)),
        }
    }
}

impl InpainterState {
    pub fn load(&self, path: &Path) -> Result<(), String> {
        self.load_internal(path)?;
        self.fixture.store(false, Ordering::SeqCst);
        *self.load_error.lock().map_err(|e| e.to_string())? = None;
        Ok(())
    }

    /// Loads `path` exactly like `load`, but marks the result as the dev
    /// fixture inpainter rather than real LaMa (see the `fixture` field doc
    /// comment for why this is a separate entry point rather than a bool
    /// parameter on `load`: the distinct name makes every call site's intent
    /// readable without checking an argument).
    pub fn load_fixture(&self, path: &Path) -> Result<(), String> {
        self.load_internal(path)?;
        self.fixture.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn load_internal(&self, path: &Path) -> Result<(), String> {
        // CPU on purpose, with measurements (2026-07-10 benchmark): LaMa's
        // FFC/Fourier blocks are not CoreML-convertible -- the CoreML EP
        // shatters the graph into 621 partitions (46% node coverage), runs
        // ~3x SLOWER than CPU (4.0s vs 1.3s per 512px window), and diverges
        // up to ~44/255 inside the healed region. Do not "upgrade" this to
        // CoreML-first like the detector without re-measuring.
        let inp = fd_heal::Inpainter::load(path, fd_infer::Ep::Cpu).map_err(|e| e.to_string())?;
        let hash = crate::models::file_sha256(path)?;
        *self.inner.lock().map_err(|e| e.to_string())? = Some(LoadedInpainter {
            inpainter: inp,
            hash,
        });
        Ok(())
    }

    /// True when the currently loaded model (if any) was loaded via
    /// `load_fixture` rather than `load`. Meaningless when nothing is
    /// loaded; callers that care check `with_inpainter(|i| i.is_some())`
    /// first, same as `models::inpainter_status` does.
    pub fn is_fixture(&self) -> bool {
        self.fixture.load(Ordering::SeqCst)
    }

    /// Records a real-LaMa load failure for `inpainter_load_error` to read
    /// back. Swallows a poisoned lock the same way `DetectorState::hash`
    /// does elsewhere in this file: recording a diagnostic detail is not
    /// worth propagating a hard error over.
    pub fn record_load_error(&self, message: String) {
        if let Ok(mut guard) = self.load_error.lock() {
            *guard = Some(message);
        }
    }

    /// The most recent real-LaMa load failure, if any and if not since
    /// superseded by a successful `load`.
    pub fn load_error(&self) -> Option<String> {
        self.load_error.lock().ok()?.clone()
    }

    /// Runs `f` with mutable access to the loaded inpainter (or None).
    pub fn with_inpainter<R>(
        &self,
        f: impl FnOnce(Option<&mut fd_heal::Inpainter>) -> R,
    ) -> Result<R, String> {
        let mut guard = self.inner.lock().map_err(|e| e.to_string())?;
        Ok(f(guard.as_mut().map(|loaded| &mut loaded.inpainter)))
    }

    /// Runs `f` with mutable access to the loaded inpainter AND its file hash,
    /// observed under one lock acquisition. This is the atomicity guarantee
    /// heal-cache provenance depends on: `with_inpainter` and `hash()` taken
    /// as two separate locks would let a model download complete between
    /// them, so the model that actually heals could be model B while the
    /// provenance recorded (from a hash taken before or after) says model A
    /// -- or the zeros sentinel for "no model". Pairing them under a single
    /// guard makes "the hash used for provenance" and "the model that heals"
    /// the same observation, closing that race.
    pub fn with_inpainter_hashed<R>(
        &self,
        f: impl FnOnce(Option<(&mut fd_heal::Inpainter, [u8; 32])>) -> R,
    ) -> Result<R, String> {
        let mut guard = self.inner.lock().map_err(|e| e.to_string())?;
        Ok(f(guard
            .as_mut()
            .map(|loaded| (&mut loaded.inpainter, loaded.hash))))
    }

    // No production caller: heal-cache provenance always reads the inpainter's
    // hash through with_inpainter_hashed (one lock, paired with the model
    // that actually heals -- see that method's doc comment). Kept for tests
    // and any future non-racy caller.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn hash(&self) -> Option<[u8; 32]> {
        let guard = self.inner.lock().ok()?;
        guard.as_ref().map(|loaded| loaded.hash)
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
    fn detect_with_progress_fires_the_callback() {
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
        let mut calls: Vec<(usize, usize)> = Vec::new();
        let probs = state
            .detect_with_progress(&img, &mut |done, total| calls.push((done, total)))
            .unwrap();
        assert_eq!(probs.len(), 64 * 48);
        assert!(!calls.is_empty());
        assert_eq!(calls.last().unwrap().0, calls.last().unwrap().1);
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
    fn detect_hashed_with_progress_fires_the_callback_and_pairs_the_hash() {
        let state = DetectorState::default();
        state.load(&fixture()).unwrap();
        let expected_hash = state.hash().unwrap();
        let img = ImageBuf {
            width: 64,
            height: 48,
            channels: 1,
            data: PixelData::U8(vec![128; 64 * 48]),
            icc: None,
            exif: None,
        };
        let mut calls: Vec<(usize, usize)> = Vec::new();
        let (probs, hash) = state
            .detect_hashed_with_progress(&img, &mut |done, total| calls.push((done, total)))
            .unwrap();
        assert_eq!(probs.len(), 64 * 48);
        assert_eq!(hash, expected_hash);
        assert!(
            !calls.is_empty(),
            "at least one tile should fire the callback"
        );
        let total = calls[0].1;
        for (i, &(done, t)) in calls.iter().enumerate() {
            assert_eq!(done, i + 1);
            assert_eq!(t, total);
        }
    }

    #[test]
    fn detect_hashed_with_progress_errors_without_a_model_like_detect_hashed() {
        let state = DetectorState::default();
        let img = ImageBuf {
            width: 8,
            height: 8,
            channels: 1,
            data: PixelData::U8(vec![0; 64]),
            icc: None,
            exif: None,
        };
        assert!(state
            .detect_hashed_with_progress(&img, &mut |_, _| {})
            .unwrap_err()
            .contains("no detector"));
    }

    #[test]
    fn load_missing_model_errors_with_path() {
        let state = DetectorState::default();
        let err = state
            .load(Path::new("/nonexistent/model.onnx"))
            .unwrap_err();
        assert!(err.contains("model.onnx"));
    }

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

    fn inpaint_fixture() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../engine/fixtures/tiny-inpaint.onnx")
    }

    #[test]
    fn detect_hashed_pairs_output_with_the_producing_model() {
        let state = DetectorState::default();
        state.load(&fixture()).unwrap();
        let expected = state.hash().unwrap();
        let img = ImageBuf {
            width: 64,
            height: 48,
            channels: 1,
            data: PixelData::U8(vec![128; 64 * 48]),
            icc: None,
            exif: None,
        };
        let (probs, hash) = state.detect_hashed(&img).unwrap();
        assert_eq!(probs.len(), 64 * 48);
        assert_eq!(hash, expected);
        let none = DetectorState::default();
        assert!(none.detect_hashed(&img).is_err());
    }

    #[test]
    fn loaded_detector_exposes_file_hash() {
        let state = DetectorState::default();
        assert!(state.hash().is_none());
        state.load(&fixture()).unwrap();
        let h = state.hash().expect("hash after load");
        // stable across loads of the same file
        state.load(&fixture()).unwrap();
        assert_eq!(state.hash().unwrap(), h);
    }

    #[test]
    fn loaded_inpainter_exposes_file_hash() {
        let state = InpainterState::default();
        assert!(state.hash().is_none());
        state.load(&inpaint_fixture()).unwrap();
        let h = state.hash().expect("hash after load");
        // stable across loads of the same file
        state.load(&inpaint_fixture()).unwrap();
        assert_eq!(state.hash().unwrap(), h);
    }

    #[test]
    fn with_inpainter_hashed_pairs_the_model_with_its_own_hash() {
        let state = InpainterState::default();
        // No model loaded: the closure sees None.
        assert!(state.with_inpainter_hashed(|pair| pair.is_none()).unwrap());

        state.load(&inpaint_fixture()).unwrap();
        // Capture the expected hash BEFORE entering the closure: calling
        // state.hash() from inside would try to lock the same non-reentrant
        // mutex that with_inpainter_hashed already holds, and deadlock.
        let expected_hash = state.hash().expect("hash after load");
        let saw_matching_pair = state
            .with_inpainter_hashed(|pair| {
                let (_inp, hash) = pair.expect("model loaded");
                hash == expected_hash
            })
            .unwrap();
        assert!(saw_matching_pair);
    }

    #[test]
    fn fresh_state_is_not_a_fixture() {
        let state = InpainterState::default();
        assert!(!state.is_fixture());
    }

    #[test]
    fn load_fixture_marks_the_state_as_a_fixture() {
        // The detection seam: `load_fixture` is a distinctly-named entry
        // point the debug autoload calls precisely because it KNOWS it is
        // falling back to the dev stub -- `inpainter_status` (models.rs)
        // reads this flag rather than inferring fixture-ness later by
        // comparing hashes.
        let state = InpainterState::default();
        state.load_fixture(&inpaint_fixture()).unwrap();
        assert!(state.with_inpainter(|i| i.is_some()).unwrap());
        assert!(state.is_fixture());
    }

    #[test]
    fn plain_load_never_marks_the_state_as_a_fixture() {
        // Fixture-ness is a property of the CALL SITE, not the file: loading
        // the same fixture file through the regular `load` (as
        // `load_inpainter`/`download_inpaint_model` do) must not be
        // mistaken for the dev autoload.
        let state = InpainterState::default();
        state.load(&inpaint_fixture()).unwrap();
        assert!(!state.is_fixture());
    }

    #[test]
    fn a_real_load_after_a_fixture_load_clears_the_fixture_flag() {
        let state = InpainterState::default();
        state.load_fixture(&inpaint_fixture()).unwrap();
        assert!(state.is_fixture());
        state.load(&inpaint_fixture()).unwrap();
        assert!(!state.is_fixture());
    }

    #[test]
    fn fresh_state_has_no_load_error() {
        let state = InpainterState::default();
        assert!(state.load_error().is_none());
    }

    #[test]
    fn record_load_error_is_readable_back() {
        let state = InpainterState::default();
        state.record_load_error("corrupt onnx header".to_string());
        assert_eq!(state.load_error().as_deref(), Some("corrupt onnx header"));
    }

    #[test]
    fn a_later_successful_load_clears_a_recorded_load_error() {
        let state = InpainterState::default();
        state.record_load_error("corrupt onnx header".to_string());
        state.load(&inpaint_fixture()).unwrap();
        assert!(state.load_error().is_none());
    }
}
