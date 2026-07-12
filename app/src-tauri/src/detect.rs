use std::ops::ControlFlow;
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
pub struct DetectorState {
    model: Arc<Mutex<Option<LoadedDetector>>>,
    /// The loaded model's file hash, mirrored out of `model` into its own
    /// briefly-held lock so `hash()` never waits on `model`. Detection holds
    /// `model` for the whole tile sweep (~9s per frame, a full roll during a
    /// scan), and `enqueue_exports` reads this hash from the main thread --
    /// reading it through `model` would freeze the app for that window, the
    /// same defect fixed for the inpainter (TheUnduster-2jv). Written only at
    /// load, so this lock is never held long enough to contend.
    hash: Arc<Mutex<Option<[u8; 32]>>>,
}

impl Default for DetectorState {
    fn default() -> Self {
        DetectorState {
            model: Arc::new(Mutex::new(None)),
            hash: Arc::new(Mutex::new(None)),
        }
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
        // Mirror the hash before taking `model`, matching the inpainter: the
        // standalone `hash()` read stays cheap and current.
        *self.hash.lock().map_err(|e| e.to_string())? = Some(hash);
        *self.model.lock().map_err(|e| e.to_string())? = Some(LoadedDetector {
            detector: det,
            hash,
        });
        Ok(())
    }

    // No production caller since the export path switched to
    // detect_with_progress for its per-tile cancellation check-in; kept for
    // API symmetry and direct testing, same as detect_with_progress was
    // before it gained that caller.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn detect(&self, img: &ImageBuf) -> Result<Vec<f32>, String> {
        self.detect_with_progress(img, &mut |_, _| ControlFlow::Continue(()))
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
        progress: &mut dyn FnMut(usize, usize) -> ControlFlow<()>,
    ) -> Result<Vec<f32>, String> {
        let mut guard = self.model.lock().map_err(|e| e.to_string())?;
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
        self.detect_hashed_with_progress(img, &mut |_, _| ControlFlow::Continue(()))
    }

    /// `detect_hashed` with a per-tile progress callback `(done, total)`.
    /// Used by the single-image detect command and the roll job's Detect
    /// arm to narrate a long detect tile by tile; the hash pairing guarantee
    /// documented on `detect_hashed` holds identically here (same lock, same
    /// closure).
    pub fn detect_hashed_with_progress(
        &self,
        img: &ImageBuf,
        progress: &mut dyn FnMut(usize, usize) -> ControlFlow<()>,
    ) -> Result<(Vec<f32>, [u8; 32]), String> {
        let mut guard = self.model.lock().map_err(|e| e.to_string())?;
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
    ///
    /// Reads the `hash` mirror, NOT `model`: detection holds `model` for
    /// seconds-to-minutes and this runs on the main-thread `enqueue_exports`
    /// command, so locking `model` here would freeze the app (TheUnduster-2jv).
    pub fn hash(&self) -> Option<[u8; 32]> {
        *self.hash.lock().ok()?
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
    /// The loaded model's file hash, mirrored out of `inner` into its own
    /// briefly-held lock so `hash()` can read it WITHOUT waiting on `inner`.
    /// A heal holds `inner` for its entire multi-minute run (see
    /// `with_inpainter_hashed`), and `enqueue_exports` reads this hash from
    /// the main thread -- reading it through `inner` froze the whole app
    /// until the heal finished (TheUnduster-2jv). Written only at load, so
    /// this lock is never held long enough to contend.
    hash: Arc<Mutex<Option<[u8; 32]>>>,
}

impl Default for InpainterState {
    fn default() -> Self {
        InpainterState {
            inner: Arc::new(Mutex::new(None)),
            fixture: Arc::new(AtomicBool::new(false)),
            load_error: Arc::new(Mutex::new(None)),
            hash: Arc::new(Mutex::new(None)),
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
        // Mirror the hash into its own lock BEFORE taking `inner`, so a
        // standalone `hash()` read is never even briefly stale against the
        // model that's about to become current, and never has to wait on
        // `inner`. `with_inpainter_hashed` still reads its paired hash from
        // `inner` for the provenance atomicity guarantee it documents.
        *self.hash.lock().map_err(|e| e.to_string())? = Some(hash);
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

    /// Heal-cache provenance must NOT use this: it reads the hash through
    /// with_inpainter_hashed (one lock, paired with the model that actually
    /// heals -- see that method's doc comment). This standalone read is for
    /// callers where a racing model swap is harmless, like the export skip
    /// decision (`export_provenance_hex`), where a mismatch only causes a
    /// spare re-export.
    ///
    /// Reads the `hash` mirror, NOT `inner`: a running heal holds `inner` for
    /// minutes, and this is called from the main-thread `enqueue_exports`
    /// command, so locking `inner` here froze the app (TheUnduster-2jv).
    pub fn hash(&self) -> Option<[u8; 32]> {
        *self.hash.lock().ok()?
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
            .detect_with_progress(&img, &mut |done, total| {
                calls.push((done, total));
                ControlFlow::Continue(())
            })
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
            .detect_hashed_with_progress(&img, &mut |done, total| {
                calls.push((done, total));
                ControlFlow::Continue(())
            })
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
            .detect_hashed_with_progress(&img, &mut |_, _| ControlFlow::Continue(()))
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
    fn detector_hash_does_not_block_while_a_detect_holds_the_model() {
        // Detection holds `model` for its whole tile sweep; `hash()` must read
        // the mirror instead, so the main-thread enqueue_exports command never
        // waits on it. Mirror of the inpainter test. See TheUnduster-2jv.
        use std::sync::mpsc;
        use std::time::Duration;

        let state = DetectorState::default();
        state.load(&fixture()).unwrap();

        // Hold `model` the way a running detect does, releasing on command.
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let detect_state = state.clone();
        let detect = std::thread::spawn(move || {
            let _guard = detect_state.model.lock().unwrap();
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });

        entered_rx.recv().unwrap();

        let hash_state = state.clone();
        let (hash_tx, hash_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = hash_tx.send(hash_state.hash());
        });
        let answered = hash_rx.recv_timeout(Duration::from_secs(2));

        release_tx.send(()).unwrap();
        detect.join().unwrap();

        assert!(
            matches!(answered, Ok(Some(_))),
            "hash() blocked on the model lock held by a running detect"
        );
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
    fn hash_does_not_block_while_a_heal_holds_the_inpainter() {
        // A running heal holds `inner` for its whole duration (minutes) via
        // with_inpainter_hashed. `hash()` must NOT contend on that lock:
        // enqueue_exports calls it on the main thread, so a blocked hash()
        // freezes the entire app until the heal finishes. See TheUnduster-2jv.
        use std::sync::mpsc;
        use std::time::Duration;

        let state = InpainterState::default();
        state.load(&inpaint_fixture()).unwrap();

        // Stand-in for a long-running heal: hold `inner` inside the closure
        // until the test explicitly releases it.
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let heal_state = state.clone();
        let heal = std::thread::spawn(move || {
            heal_state
                .with_inpainter_hashed(|_pair| {
                    entered_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                })
                .unwrap();
        });

        // Wait until the heal is inside the closure with the lock held.
        entered_rx.recv().unwrap();

        // Read the hash off-thread so a blocking lock shows up as a timeout
        // instead of wedging the test runner.
        let hash_state = state.clone();
        let (hash_tx, hash_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = hash_tx.send(hash_state.hash());
        });
        let answered = hash_rx.recv_timeout(Duration::from_secs(2));

        // Let the heal finish regardless of the assertion outcome.
        release_tx.send(()).unwrap();
        heal.join().unwrap();

        assert!(
            matches!(answered, Ok(Some(_))),
            "hash() blocked on the inpainter lock held by a running heal"
        );
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
