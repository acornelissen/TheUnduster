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

    #[cfg_attr(not(test), allow(dead_code))]
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
}
