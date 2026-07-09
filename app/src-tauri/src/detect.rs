use std::path::Path;
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
        let mut guard = self.0.lock().map_err(|e| e.to_string())?;
        match guard.as_mut() {
            Some(loaded) => loaded
                .detector
                .probabilities(img)
                .map_err(|e| e.to_string()),
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
pub struct InpainterState(Arc<Mutex<Option<LoadedInpainter>>>);

impl Default for InpainterState {
    fn default() -> Self {
        InpainterState(Arc::new(Mutex::new(None)))
    }
}

impl InpainterState {
    pub fn load(&self, path: &Path) -> Result<(), String> {
        let inp = fd_heal::Inpainter::load(path, fd_infer::Ep::Cpu).map_err(|e| e.to_string())?;
        let hash = crate::models::file_sha256(path)?;
        *self.0.lock().map_err(|e| e.to_string())? = Some(LoadedInpainter {
            inpainter: inp,
            hash,
        });
        Ok(())
    }

    /// Runs `f` with mutable access to the loaded inpainter (or None).
    pub fn with_inpainter<R>(
        &self,
        f: impl FnOnce(Option<&mut fd_heal::Inpainter>) -> R,
    ) -> Result<R, String> {
        let mut guard = self.0.lock().map_err(|e| e.to_string())?;
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
        let mut guard = self.0.lock().map_err(|e| e.to_string())?;
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
        let guard = self.0.lock().ok()?;
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
}
